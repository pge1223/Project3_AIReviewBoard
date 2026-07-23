"""
Document Chunker
================
DocumentExtractionResult(PDF/DOCX/PPTX) 또는 CleanedWebContent(HTML)를 입력받아
RAG 임베딩에 쓸 청크 목록(ChunkingResult)을 생성한다. LangGraph 없이 단독 실행 가능하며
임베딩/Chroma 저장은 이 모듈의 범위 밖이다.

처리 흐름:
  1) 어댑터로 UnifiedBlock 리스트 생성 (원본 무변형)
  2) 논리 단위(logical unit)로 분할 — 페이지/슬라이드/섹션 경계, heading, table을 기준으로 나눔
  3) 단위별로:
     - table 단위: 일반 본문과 절대 합치지 않고 자체 규칙으로 청크 생성
     - body/toc 단위: RecursiveCharacterTextSplitter(add_start_index=True)로 분할하고
       start_index를 이용해 원본 블록에 역매핑
  4) 각 청크에 결정적 chunk_id 부여 후 ChunkingResult로 반환
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai.rag.parsers.schemas import DocumentExtractionResult
from ai.rag.preprocessing.schemas import CleanedWebContent

from ai.rag.chunking.adapters import (
    UnifiedBlock,
    adapt_document_extraction_result,
    adapt_cleaned_web_content,
)
from ai.rag.chunking.config import (
    TOC_HEADING_KEYWORDS,
    TOC_STRUCTURAL_LINE_MATCH_RATIO,
    PSEUDO_HEADING_MARKERS,
    PSEUDO_HEADING_MAX_TITLE_LENGTH,
    LIST_ITEM_MARKER_PATTERN,
    LIST_ITEM_MIN_MARKER_COUNT,
    TAIL_CHUNK_MIN_CHARS,
    extract_whole_line_heading_title,
)
from ai.rag.chunking.schemas import (
    Chunk,
    ChunkingConfig,
    ChunkingResult,
    ChunkLocationType,
    ChunkSourceContext,
    ContentKind,
)

_TOC_DOTTED_LINE_RE = re.compile(r"(\.{2,}|·{2,})\s*\d{1,4}\s*$")
_UNIT_JOIN_SEPARATOR = "\n\n"

_PSEUDO_HEADING_PREFIX_RE = re.compile(rf"^\s*[{re.escape(PSEUDO_HEADING_MARKERS)}]\s*")
_PSEUDO_HEADING_TITLE_DELIMITER_RE = re.compile(r"[:：\-\n]")
_LIST_ITEM_BOUNDARY_RE = re.compile(rf"(?:\A|\n)[ \t]*(?:{LIST_ITEM_MARKER_PATTERN})")
_EVALUATION_BULLET_RE = re.compile(r"^\s*[-*•·]\s+")
_EVALUATION_SCORE_RE = re.compile(r"(?:\(\s*\d{1,3}\s*\)|\d{1,3}\s*점)\s*$")
_EVALUATION_SECTION_NAMES = (
    "혁신성",
    "확장성",
    "적용성",
    "실현 가능성",
    "실현가능성",
    "사회적 가치성",
    "계획 적정성",
    "운영 혁신성",
    "거버넌스 우수성",
    "도시 문제 해결성",
    "도시 경쟁력",
    "지속 가능성",
    "지속가능성",
)
_EVALUATION_DOCUMENT_MARKERS = ("평가 기준", "평가기준", "평가 항목", "평가항목", "배점")


@dataclass
class _LogicalUnit:
    """
    heading 블록(있으면) + 다음 heading/table/위치변경 전까지의 본문 블록.

    heading_block: 실제 heading(BlockType.TITLE/WebBlockType.HEADING) 블록. all_blocks()에
        포함되어 본문 텍스트 구성에도 쓰인다.
    heading_text_override: 의사(pseudo) heading 감지 또는 table의 section_title 상속처럼,
        "표시할 section_title은 있지만 그 텍스트를 본문에서 별도 블록으로 떼어내진 않는" 경우에 사용.
    """
    kind: str  # "body" | "table" | "toc"
    location_type: ChunkLocationType
    location_number: Optional[int]
    heading_block: Optional[UnifiedBlock] = None
    heading_text_override: Optional[str] = None
    body_blocks: list[UnifiedBlock] = field(default_factory=list)

    @property
    def heading_text(self) -> Optional[str]:
        if self.heading_block is not None:
            return self.heading_block.content
        return self.heading_text_override

    def all_blocks(self) -> list[UnifiedBlock]:
        return ([self.heading_block] if self.heading_block else []) + self.body_blocks


def chunk_document(
    source: DocumentExtractionResult | CleanedWebContent,
    context: ChunkSourceContext,
    config: Optional[ChunkingConfig] = None,
) -> ChunkingResult:
    """
    문서를 청킹하여 ChunkingResult를 반환한다. 입력(source)과 그 하위 blocks는 절대 mutate하지 않는다.
    """
    config = config or ChunkingConfig()
    warnings: list[str] = []

    if isinstance(source, DocumentExtractionResult):
        resolved_file_type: Optional[str] = source.file_type.value
        if context.file_type and context.file_type != resolved_file_type:
            warnings.append(
                f"ChunkSourceContext.file_type('{context.file_type}')과 실제 파싱 결과 "
                f"file_type('{resolved_file_type}')이 달라, 실제 파싱 결과를 우선 사용합니다."
            )
        unified_blocks = adapt_document_extraction_result(source)
    elif isinstance(source, CleanedWebContent):
        resolved_file_type = context.file_type  # HTML엔 file_type 개념이 없어 context 값을 그대로 사용 (대개 None)
        unified_blocks = adapt_cleaned_web_content(source)
    else:
        raise TypeError(f"지원하지 않는 입력 타입입니다: {type(source)!r}")

    if not unified_blocks:
        return ChunkingResult(
            document_id=context.document_id,
            chunks=[],
            chunk_count=0,
            warnings=warnings,
            chunking_version=config.chunking_version,
            config=config,
        )

    units = _segment_into_logical_units(unified_blocks)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(config.separators),
        add_start_index=True,
    )
    config_fingerprint = config.fingerprint()

    chunks: list[Chunk] = []
    global_index = 0

    for unit in units:
        if unit.kind == "table":
            pieces, unit_warnings = _process_table_unit(unit, config)
            content_kind = ContentKind.TABLE
            indexable = True
        else:
            pieces, unit_warnings = _process_text_unit(unit, config, splitter)
            content_kind = ContentKind.TOC if unit.kind == "toc" else ContentKind.BODY
            indexable = unit.kind != "toc"

            if unit.kind == "body":
                body_only_text = "\n".join(b.content for b in unit.body_blocks)
                if _looks_structurally_like_toc(body_only_text):
                    unit_warnings.append(
                        f"섹션(heading='{unit.heading_text}', 위치={unit.location_type.value}/{unit.location_number})이 "
                        "목차처럼 보이는 구조(점선/페이지 번호)를 가지고 있지만 "
                        "'목차'/'차례'/'Contents' heading이 확인되지 않아 BODY로 유지했습니다."
                    )

        warnings.extend(unit_warnings)

        for local_chunk_index, (piece_content, source_block_orders, source_block_ids, local_positions, extra_metadata) in enumerate(pieces):
            if not piece_content.strip():
                continue

            chunk_id = _generate_chunk_id(
                document_id=context.document_id,
                chunking_version=config.chunking_version,
                config_fingerprint=config_fingerprint,
                location_type=unit.location_type,
                location_number=unit.location_number,
                source_block_local_positions=local_positions,
                local_chunk_index=local_chunk_index,
                content=piece_content,
            )

            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=context.document_id,
                content=piece_content,
                chunk_index=global_index,
                content_kind=content_kind,
                source_type=context.source_type,
                source_url=context.source_url,
                source_page_url=context.source_page_url,
                source_filename=context.source_filename,
                file_type=resolved_file_type,
                location_type=unit.location_type,
                location_number=unit.location_number,
                section_title=unit.heading_text,
                source_block_ids=source_block_ids,
                source_block_orders=source_block_orders,
                char_count=len(piece_content),
                indexable=indexable,
                chunking_version=config.chunking_version,
                metadata=extra_metadata,
            ))
            global_index += 1

    return ChunkingResult(
        document_id=context.document_id,
        chunks=chunks,
        chunk_count=len(chunks),
        warnings=warnings,
        chunking_version=config.chunking_version,
        config=config,
    )


# ---------------------------------------------------------------------------
# 논리 단위 분할
# ---------------------------------------------------------------------------

def _segment_into_logical_units(blocks: list[UnifiedBlock]) -> list[_LogicalUnit]:
    """
    active_heading_text: 가장 최근에 확인된 section 제목(진짜 heading 또는 의사-heading)의 텍스트.
    table 단위를 만나도 초기화하지 않아, 표가 직전 section_title을 상속하도록 한다.
    페이지/슬라이드가 바뀌면(location 변경) 기존 정책대로 초기화한다.
    """
    units: list[_LogicalUnit] = []
    current: Optional[_LogicalUnit] = None
    active_heading_text: Optional[str] = None
    last_location: Optional[tuple] = None

    for block in blocks:
        location_key = (block.location_type, block.location_number)
        if last_location is not None and location_key != last_location:
            active_heading_text = None  # 페이지/슬라이드 경계 처리 정책은 기존대로 유지
        last_location = location_key

        if block.kind == "table":
            # 표는 항상 독립 단위. heading_block 자체는 없지만, 직전 section_title은 상속한다.
            if current is not None:
                units.append(current)
                current = None
            units.append(_LogicalUnit(
                kind="table",
                location_type=block.location_type,
                location_number=block.location_number,
                heading_text_override=active_heading_text,
                body_blocks=[block],
            ))
            continue

        if block.kind == "heading":
            if current is not None:
                units.append(current)
            is_toc = _is_toc_heading(block.content)
            active_heading_text = block.content
            current = _LogicalUnit(
                kind="toc" if is_toc else "body",
                location_type=block.location_type,
                location_number=block.location_number,
                heading_block=block,
                body_blocks=[],
            )
            continue

        # 의사(pseudo) heading 감지: "□ 접수방법: ..." 형태의 단락도 section_title 후보로 인정.
        # 단, 본문 내용을 별도로 떼어내지 않고 그대로 body_blocks에 포함시킨다.
        # "1) 개요"/"① 법제처 법령 학습"처럼 블록 전체가 통째로 짧은 제목인 경우(PDF에 흔함)는
        # _extract_pseudo_heading_title()이 못 잡으므로 extract_whole_line_heading_title()로 보강한다.
        pseudo_title = None
        if block.kind in ("paragraph", "list"):
            pseudo_title = _extract_pseudo_heading_title(block.content)
            if pseudo_title is None:
                pseudo_title = extract_whole_line_heading_title(block.content)
        if pseudo_title is not None:
            if current is not None:
                units.append(current)
            active_heading_text = pseudo_title
            current = _LogicalUnit(
                kind="body",
                location_type=block.location_type,
                location_number=block.location_number,
                heading_text_override=pseudo_title,
                body_blocks=[block],
            )
            continue

        # 일반 본문/리스트 블록: 새 단위가 필요한 경우(직전이 table이었거나 위치가 바뀐 경우)엔
        # active_heading_text를 상속한 새 단위를 시작하고, 아니면 기존 단위에 이어붙인다.
        location_changed = (
            current is None
            or block.location_type != current.location_type
            or block.location_number != current.location_number
        )
        if location_changed:
            if current is not None:
                units.append(current)
            current = _LogicalUnit(
                kind="body",
                location_type=block.location_type,
                location_number=block.location_number,
                heading_text_override=active_heading_text,
                body_blocks=[],
            )
        current.body_blocks.append(block)

    if current is not None:
        units.append(current)

    return units


def _is_toc_heading(heading_text: str) -> bool:
    normalized = heading_text.strip().lower()
    return any(keyword in normalized for keyword in TOC_HEADING_KEYWORDS)


def _extract_pseudo_heading_title(content: str) -> Optional[str]:
    """
    "□ 접수방법: 이메일로 제출..." 같은 단락에서 짧은 제목만 추출한다.
    제목 뒤에 본문이 이어지더라도 content 자체는 건드리지 않고, 여기서 추출한 문자열만
    section_title 용도로 별도 반환한다 (호출자가 content를 자르지 않음).
    """
    match = _PSEUDO_HEADING_PREFIX_RE.match(content)
    if match is None:
        return None

    remainder = content[match.end():]
    delimiter_match = _PSEUDO_HEADING_TITLE_DELIMITER_RE.search(remainder)
    candidate = remainder[: delimiter_match.start()] if delimiter_match else remainder
    candidate = candidate.strip()

    if not candidate or len(candidate) > PSEUDO_HEADING_MAX_TITLE_LENGTH:
        return None
    return candidate


def _find_list_item_boundaries(text: str) -> list[int]:
    """텍스트 내에서 목록 마커(-, □, ※, ①~⑩)가 시작되는 위치 목록을 반환"""
    return [match.start() for match in _LIST_ITEM_BOUNDARY_RE.finditer(text)]


def _evaluation_section_title(line: str) -> Optional[str]:
    """평가표의 짧은 항목 제목(예: ``확장성 (20)``)을 반환한다."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60 or _EVALUATION_BULLET_RE.match(stripped):
        return None
    if not any(name in stripped for name in _EVALUATION_SECTION_NAMES):
        return None
    if _EVALUATION_SCORE_RE.search(stripped) or stripped in _EVALUATION_SECTION_NAMES:
        return stripped
    return None


def _looks_like_evaluation_criteria_text(text: str) -> bool:
    """일반 목록을 과도하게 잘게 나누지 않도록 평가표 신호가 충분할 때만 활성화한다."""
    marker_count = sum(1 for marker in _EVALUATION_DOCUMENT_MARKERS if marker in text)
    bullet_count = sum(
        1 for line in text.splitlines() if _EVALUATION_BULLET_RE.match(line)
    )
    section_count = sum(
        1 for line in text.splitlines() if _evaluation_section_title(line) is not None
    )
    return marker_count >= 1 and bullet_count >= 2 and section_count >= 1


def _split_evaluation_criteria_text(
    text: str,
    config: ChunkingConfig,
) -> list[tuple[str, int, int, dict]]:
    """평가표를 ``평가 항목 + 세부 질문 1개`` 단위로 분할한다.

    질문 부분은 원문 범위를 가리키고, 항목 제목은 검색 문맥을 보존하기 위해 앞에 반복한다.
    이 방식으로 같은 페이지의 ``문제 정의``와 ``확장성`` 문항이 한 청크에 섞이지 않는다.
    """
    if not _looks_like_evaluation_criteria_text(text):
        return []

    lines: list[tuple[int, int, str]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        body = raw_line.rstrip("\r\n")
        lines.append((offset, offset + len(body), body))
        offset += len(raw_line)
    if offset < len(text):
        lines.append((offset, len(text), text[offset:]))

    pieces: list[tuple[str, int, int, dict]] = []
    active_title: Optional[str] = None
    first_bullet_start: Optional[int] = None
    index = 0
    while index < len(lines):
        line_start, _, line = lines[index]
        title = _evaluation_section_title(line)
        if title is not None:
            active_title = title
            index += 1
            continue
        if not _EVALUATION_BULLET_RE.match(line):
            index += 1
            continue

        if first_bullet_start is None:
            first_bullet_start = line_start
        item_start = line_start
        item_end = lines[index][1]
        cursor = index + 1
        while cursor < len(lines):
            _, next_end, next_line = lines[cursor]
            if _EVALUATION_BULLET_RE.match(next_line) or _evaluation_section_title(next_line):
                break
            item_end = next_end
            cursor += 1

        raw_item = text[item_start:item_end].strip()
        piece = f"{active_title}\n{raw_item}" if active_title else raw_item
        if piece:
            metadata = {
                "evaluation_criterion": True,
                "criterion_title": active_title,
                "criterion_question": raw_item,
            }
            if len(piece) <= config.chunk_size:
                pieces.append((piece, item_start, item_end, metadata))
            else:
                available = max(1, config.chunk_size - len(active_title or "") - 1)
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=available,
                    chunk_overlap=0,
                    separators=list(config.separators),
                    add_start_index=True,
                )
                for doc in sub_splitter.create_documents([raw_item]):
                    sub = doc.page_content
                    local_start = doc.metadata.get("start_index") or 0
                    sub_start = item_start + local_start
                    rendered = f"{active_title}\n{sub}" if active_title else sub
                    pieces.append(
                        (
                            rendered,
                            sub_start,
                            sub_start + len(sub),
                            {**metadata, "criterion_question": sub, "oversized_criterion_split": True},
                        )
                    )
        index = cursor

    # 평가표 오탐으로 원문 대부분을 잃는 것보다 기존 청킹으로 폴백하는 편이 안전하다.
    if len(pieces) < 2:
        return []
    preamble = text[: first_bullet_start or 0].strip()
    if preamble:
        preamble_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=0,
            separators=list(config.separators),
            add_start_index=True,
        )
        preamble_pieces = []
        for doc in preamble_splitter.create_documents([preamble]):
            start = doc.metadata.get("start_index") or 0
            preamble_pieces.append(
                (
                    doc.page_content,
                    start,
                    start + len(doc.page_content),
                    {"evaluation_criteria_preamble": True},
                )
            )
        pieces[0:0] = preamble_pieces
    return pieces


def _looks_structurally_like_toc(body_text: str) -> bool:
    """구조적 목차 의심 판정. 확정에는 쓰지 않고 warning 트리거로만 사용한다 (과탐 방지 우선)."""
    lines = [line for line in body_text.splitlines() if line.strip()]
    if not lines:
        return False
    matches = sum(1 for line in lines if _TOC_DOTTED_LINE_RE.search(line))
    return (matches / len(lines)) >= TOC_STRUCTURAL_LINE_MATCH_RATIO


# ---------------------------------------------------------------------------
# 본문/목차 단위 처리 (RecursiveCharacterTextSplitter + start_index 기반 원본 블록 역매핑)
# ---------------------------------------------------------------------------

def _build_unit_text_and_offsets(
    blocks: list[UnifiedBlock], join_separator: str = _UNIT_JOIN_SEPARATOR
) -> tuple[str, list[tuple[int, int, int, UnifiedBlock]]]:
    """
    단위 내 블록들을 이어붙인 텍스트와, 각 블록이 그 텍스트에서 차지하는 [start, end) 구간
    (+ 단위 내부 로컬 위치)을 반환한다.

    로컬 위치(local_index)는 이 단위 안에서의 0-based 순번으로, 문서 전체 order와 달리
    다른(앞선) 단위에 블록이 추가/삭제되어도 값이 흔들리지 않는다 — chunk_id 안정성에 사용된다.
    """
    parts: list[str] = []
    offsets: list[tuple[int, int, int, UnifiedBlock]] = []
    cursor = 0

    for local_index, block in enumerate(blocks):
        start = cursor
        end = start + len(block.content)
        offsets.append((start, end, local_index, block))
        parts.append(block.content)
        cursor = end

        if local_index < len(blocks) - 1:
            parts.append(join_separator)
            cursor += len(join_separator)

    return "".join(parts), offsets


def _process_text_unit(
    unit: _LogicalUnit, config: ChunkingConfig, splitter: RecursiveCharacterTextSplitter
) -> tuple[list[tuple[str, list[int], list[str], list[int], dict]], list[str]]:
    """
    Returns: (pieces, warnings), 각 piece는 (content, source_block_orders, source_block_ids,
    local_positions, extra_metadata). local_positions는 chunk_id 해시 전용이며 Chunk 필드에는 노출되지 않는다.
    """
    all_blocks = unit.all_blocks()
    unit_text, offsets = _build_unit_text_and_offsets(all_blocks)

    if not unit_text.strip():
        return [], []

    warnings: list[str] = []

    evaluation_ranges = _split_evaluation_criteria_text(unit_text, config)
    marker_positions = _find_list_item_boundaries(unit_text)
    use_list_aware_split = len(unit_text) > config.chunk_size and len(marker_positions) >= LIST_ITEM_MIN_MARKER_COUNT

    if evaluation_ranges:
        piece_ranges_with_metadata = evaluation_ranges
    elif use_list_aware_split:
        piece_ranges_with_metadata = [
            (*piece, {}) for piece in _split_list_like_text(unit_text, marker_positions, config, splitter)
        ]
    else:
        piece_ranges_with_metadata = []
        for doc in splitter.create_documents([unit_text]):
            content = doc.page_content
            start = doc.metadata.get("start_index")
            if start is None or start < 0:
                # add_start_index가 위치를 못 찾은 예외적인 경우에 대한 방어적 폴백
                start = unit_text.find(content)
                warnings.append(
                    "start_index 기반 원본 블록 매핑에 실패해 대체 검색(find)을 사용했습니다 "
                    f"(위치={unit.location_type.value}/{unit.location_number})."
                )
                if start < 0:
                    start = 0
            piece_ranges_with_metadata.append((content, start, start + len(content), {}))

    if not evaluation_ranges:
        merged_ranges = _merge_small_tail_piece(
            [(content, start, end) for content, start, end, _ in piece_ranges_with_metadata],
            unit_text,
            config.chunk_size,
        )
        metadata_by_range = {
            (start, end): metadata
            for _, start, end, metadata in piece_ranges_with_metadata
        }
        piece_ranges_with_metadata = [
            (content, start, end, metadata_by_range.get((start, end), {}))
            for content, start, end in merged_ranges
        ]

    results: list[tuple[str, list[int], list[str], list[int], dict]] = []
    for content, start, end, extra_metadata in piece_ranges_with_metadata:
        if not content.strip():
            continue

        covered = [
            (local_index, block)
            for (block_start, block_end, local_index, block) in offsets
            if block_start < end and block_end > start
        ]

        if not covered:
            # 그래도 매핑이 안 되면 단위의 모든 블록을 보수적으로 귀속시킨다 (누락 방지 우선)
            covered = list(enumerate(all_blocks))
            warnings.append(
                "원본 블록 매핑이 비어 있어 단위 내 모든 블록을 청크에 귀속시켰습니다 "
                f"(위치={unit.location_type.value}/{unit.location_number})."
            )

        source_block_orders = [block.order for _, block in covered]
        source_block_ids = [block.source_block_id for _, block in covered if block.source_block_id is not None]
        local_positions = [local_index for local_index, _ in covered]
        results.append((content, source_block_orders, source_block_ids, local_positions, extra_metadata))

    return results, warnings


def _split_list_like_text(
    unit_text: str,
    marker_positions: list[int],
    config: ChunkingConfig,
    splitter: RecursiveCharacterTextSplitter,
) -> list[tuple[str, int, int]]:
    """
    "-"/"□"/"※"/"①"~"⑩" 마커가 반복되는 텍스트를 항목 단위로 나누고, 각 항목을 중간에서
    자르지 않으면서 chunk_size 이하로 그리디하게 묶는다. 항목 하나가 그 자체로 chunk_size를
    넘으면 그 항목만 RecursiveCharacterTextSplitter로 추가 분할한다.

    Returns: [(content, start, end), ...] (unit_text 기준 절대 오프셋)
    """
    boundaries = list(marker_positions)
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries
    boundaries.append(len(unit_text))

    raw_items = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]

    results: list[tuple[str, int, int]] = []
    group_start: Optional[int] = None
    group_end: Optional[int] = None

    def flush() -> None:
        nonlocal group_start, group_end
        if group_start is not None:
            content = unit_text[group_start:group_end].strip()
            if content:
                results.append((content, group_start, group_end))
        group_start, group_end = None, None

    for item_start, item_end in raw_items:
        if (item_end - item_start) > config.chunk_size:
            flush()
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size, chunk_overlap=0,
                separators=list(config.separators), add_start_index=True,
            )
            item_text = unit_text[item_start:item_end]
            for doc in sub_splitter.create_documents([item_text]):
                local_start = doc.metadata.get("start_index") or 0
                abs_start = item_start + local_start
                results.append((doc.page_content, abs_start, abs_start + len(doc.page_content)))
            continue

        if group_start is None:
            group_start, group_end = item_start, item_end
        elif (item_end - group_start) <= config.chunk_size:
            group_end = item_end
        else:
            flush()
            group_start, group_end = item_start, item_end

    flush()
    return results


def _merge_small_tail_piece(
    piece_ranges: list[tuple[str, int, int]], unit_text: str, chunk_size: int
) -> list[tuple[str, int, int]]:
    """마지막 조각이 TAIL_CHUNK_MIN_CHARS 미만이면 직전 조각과 병합을 시도한다 (초과 시 병합하지 않음)"""
    if len(piece_ranges) < 2:
        return piece_ranges

    last_content, _, last_end = piece_ranges[-1]
    if len(last_content) >= TAIL_CHUNK_MIN_CHARS:
        return piece_ranges

    _, prev_start, _ = piece_ranges[-2]
    merged_content = unit_text[prev_start:last_end].strip()
    if not merged_content or len(merged_content) > chunk_size:
        return piece_ranges

    return piece_ranges[:-2] + [(merged_content, prev_start, last_end)]


# ---------------------------------------------------------------------------
# 표 단위 처리 (일반 본문과 절대 병합하지 않음)
# ---------------------------------------------------------------------------

def _process_table_unit(
    unit: _LogicalUnit, config: ChunkingConfig
) -> tuple[list[tuple[str, list[int], list[str], list[int], dict]], list[str]]:
    """표 단위는 항상 정확히 1개의 원본 블록으로 구성되므로 local_positions는 항상 [0]이다."""
    table_block = unit.body_blocks[0]
    content = table_block.content
    source_block_orders = [table_block.order]
    source_block_ids = [table_block.source_block_id] if table_block.source_block_id else []
    local_positions = [0]
    warnings: list[str] = []

    if len(content) <= config.chunk_size:
        return [(content, source_block_orders, source_block_ids, local_positions, {})], warnings

    lines = content.split("\n")
    header = lines[0] if lines else ""
    rows = lines[1:]

    if header and len(header) >= config.chunk_size:
        warnings.append(
            f"표 헤더 자체가 chunk_size({config.chunk_size})보다 길거나 같아 헤더 반복 규칙을 정확히 "
            f"적용할 수 없습니다 (위치={unit.location_type.value}/{unit.location_number})."
        )

    results: list[tuple[str, list[int], list[str], list[int], dict]] = []
    current_group: list[str] = []
    row_join_overhead = 1  # "\n"

    def flush() -> None:
        if current_group:
            piece = "\n".join([header] + current_group) if header else "\n".join(current_group)
            results.append((piece, source_block_orders, source_block_ids, local_positions, {}))

    current_len = len(header)

    for row in rows:
        single_row_len = (len(header) + row_join_overhead + len(row)) if header else len(row)

        if single_row_len > config.chunk_size:
            flush()
            current_group = []
            current_len = len(header)

            effective_size = max(1, config.chunk_size - len(header) - row_join_overhead) if header else config.chunk_size
            row_splitter = RecursiveCharacterTextSplitter(
                chunk_size=effective_size, chunk_overlap=0, separators=list(config.separators)
            )
            for sub in row_splitter.split_text(row):
                piece = (header + "\n" + sub) if header else sub
                results.append((piece, source_block_orders, source_block_ids, local_positions, {"oversized_row_split": True}))
            continue

        candidate_len = len(row) + row_join_overhead
        if current_group and current_len + candidate_len > config.chunk_size:
            flush()
            current_group = []
            current_len = len(header)

        current_group.append(row)
        current_len += candidate_len

    flush()
    return results, warnings


# ---------------------------------------------------------------------------
# 결정적 chunk_id
# ---------------------------------------------------------------------------

def _generate_chunk_id(
    *,
    document_id: str,
    chunking_version: str,
    config_fingerprint: str,
    location_type: ChunkLocationType,
    location_number: Optional[int],
    source_block_local_positions: list[int],
    local_chunk_index: int,
    content: str,
) -> str:
    """
    문서 전체 기준 전역 chunk_index는 해시에 포함하지 않는다 — 문서 앞부분에 청크가 추가되어도
    이후 관련 없는 청크들의 chunk_id가 흔들리지 않도록 하기 위함.

    source_block_orders(문서 전체 기준 절대 order)도 해시에 넣지 않는다 — order는 parser가
    문서 전체를 훑으며 매기는 전역 순번이라, 앞선(다른) 논리 단위에 블록이 추가/삭제되기만 해도
    현재 단위의 order 값이 함께 밀려버린다. 대신 이 단위 내부에서만 유효한 local_position을
    사용해, 다른 단위의 변경에 영향받지 않도록 한다.
    """
    raw = "\x1f".join([
        document_id,
        chunking_version,
        config_fingerprint,
        location_type.value,
        str(location_number),
        ",".join(str(pos) for pos in source_block_local_positions),
        str(local_chunk_index),
        content,
    ])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"chk_{digest[:16]}"
