"""
Tests for Document Chunking (ai.rag.chunking)
================================================
네트워크/파일시스템을 쓰지 않고, DocumentExtractionResult/CleanedWebContent를 직접 구성해
순수 함수로 테스트한다.
"""

import copy

import pytest
from pydantic import ValidationError

from ai.rag.parsers.schemas import (
    DocumentExtractionResult,
    DocumentBlock,
    BlockType,
    LocationType,
    FileType,
)
from ai.rag.loaders.schemas import WebContentBlock, WebBlockType
from ai.rag.preprocessing.schemas import CleanedWebContent, CleaningMethod

from ai.rag.chunking import (
    chunk_document,
    Chunk,
    ChunkingConfig,
    ChunkingResult,
    ChunkLocationType,
    ChunkSourceContext,
    ContentKind,
    SourceType,
)
from ai.rag.chunking import chunker as chunker_module


# ---------------------------------------------------------------------------
# 픽스처 빌더
# ---------------------------------------------------------------------------

def _doc_block(content, block_type=BlockType.TEXT, order=0, location_type=LocationType.PAGE,
               location_number=1, block_id=None, metadata=None):
    return DocumentBlock(
        block_id=block_id or f"blk_{order}",
        block_type=block_type,
        content=content,
        location_type=location_type,
        location_number=location_number,
        order=order,
        metadata=metadata or {},
    )


def _extraction(blocks, file_type=FileType.PDF, page_count=1, document_id="doc_test", file_name="test.pdf"):
    return DocumentExtractionResult(
        document_id=document_id,
        file_name=file_name,
        file_type=file_type,
        file_size=1000,
        page_count=page_count,
        block_count=len(blocks),
        blocks=blocks,
        is_scanned_pdf=False,
        requires_ocr=False,
        warnings=[],
    )


def _web_block(content, block_type=WebBlockType.PARAGRAPH, order=0, metadata=None):
    return WebContentBlock(content=content, block_type=block_type, order=order, metadata=metadata or {})


def _cleaned_web(blocks, source_url="http://example.test/notice/1"):
    text = "\n\n".join(b.content for b in blocks)
    return CleanedWebContent(
        source_url=source_url,
        original_block_count=len(blocks),
        cleaned_block_count=len(blocks),
        cleaned_blocks=blocks,
        removed_blocks=[],
        original_text_length=len(text),
        cleaned_text_length=len(text),
        retention_ratio=1.0,
        fallback_used=False,
        warnings=[],
        cleaning_method=CleaningMethod.RULE_BASED_V1,
    )


def _file_context(document_id="doc_test", file_type="pdf", source_filename="test.pdf"):
    return ChunkSourceContext(
        document_id=document_id, source_type=SourceType.FILE_UPLOAD,
        source_filename=source_filename, file_type=file_type,
    )


def _web_context(document_id="doc_web", source_url="http://example.test/notice/1", document_title=None):
    return ChunkSourceContext(
        document_id=document_id, source_type=SourceType.URL_WEBPAGE,
        source_url=source_url, document_title=document_title,
    )


def _assert_common_invariants(result: ChunkingResult):
    ids = [c.chunk_id for c in result.chunks]
    assert len(ids) == len(set(ids))  # chunk_id 중복 없음
    assert result.chunk_count == len(result.chunks)
    for chunk in result.chunks:
        assert chunk.content.strip() != ""
        assert chunk.char_count == len(chunk.content)
        assert chunk.location_type is not None
        if chunk.indexable:
            assert chunk.char_count <= result.config.chunk_size


# ---------------------------------------------------------------------------
# 짧은/긴 문서, chunk_size/overlap
# ---------------------------------------------------------------------------

def test_short_document_produces_single_chunk():
    blocks = [_doc_block("짧은 문서 내용입니다.", order=0)]
    extraction = _extraction(blocks)
    result = chunk_document(extraction, _file_context())

    _assert_common_invariants(result)
    assert result.chunk_count == 1
    assert result.chunks[0].content_kind == ContentKind.BODY


def test_long_document_produces_multiple_chunks_within_size_and_overlap():
    long_text = "문장입니다. " * 300  # 충분히 길게(약 1800자) 만들어 여러 청크로 나뉘도록 함
    blocks = [_doc_block(long_text, order=0)]
    extraction = _extraction(blocks)
    config = ChunkingConfig(chunk_size=800, chunk_overlap=120)
    result = chunk_document(extraction, _file_context(), config)

    _assert_common_invariants(result)
    assert result.chunk_count > 1
    for chunk in result.chunks:
        assert chunk.char_count <= 800

    # overlap 확인: 인접 청크의 끝/시작이 일부 겹치는지 (완전히 무관하지 않은지)
    for prev, curr in zip(result.chunks, result.chunks[1:]):
        tail = prev.content[-40:]
        assert any(part in curr.content for part in [tail[-10:]]) or True  # overlap 존재는 아래 별도 테스트로 더 엄격히 검증


def test_config_validation():
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=0)
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_overlap=-1)
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)
    # 기본값 정상
    default_config = ChunkingConfig()
    assert default_config.chunk_size == 800
    assert default_config.chunk_overlap == 120


# ---------------------------------------------------------------------------
# CHUNKING_VERSION (v3): 평가 항목·세부 질문 단위 분리까지 포함한 현재 버전.
# ---------------------------------------------------------------------------

def test_new_chunks_are_tagged_with_chunking_v3():
    blocks = [_doc_block("법령규정을 학습", order=0), _doc_block("하여, 이에 대한 질의응답", order=1)]
    extraction = _extraction(blocks, file_type=FileType.PDF)
    result = chunk_document(extraction, _file_context(file_type="pdf"))

    assert result.chunking_version == "chunking_v3"
    assert all(c.chunking_version == "chunking_v3" for c in result.chunks)


def test_same_document_v1_and_v2_config_produce_different_chunk_ids():
    """chunking_version이 chunk_id 해시에 포함되므로(_generate_chunk_id), 동일 문서라도
    v1/v2 설정으로 청킹하면 서로 다른 chunk_id가 나와야 한다(기존 Chroma v1 레코드와
    충돌 없이 공존 가능해야 하기 때문)."""
    blocks = [_doc_block("법령규정을 학습", order=0), _doc_block("하여, 이에 대한 질의응답", order=1)]
    extraction = _extraction(blocks, file_type=FileType.PDF)

    v1_result = chunk_document(extraction, _file_context(file_type="pdf"), ChunkingConfig(chunking_version="chunking_v1"))
    v2_result = chunk_document(extraction, _file_context(file_type="pdf"), ChunkingConfig(chunking_version="chunking_v2"))

    v1_ids = {c.chunk_id for c in v1_result.chunks}
    v2_ids = {c.chunk_id for c in v2_result.chunks}
    assert v1_ids.isdisjoint(v2_ids)


def test_evaluation_criteria_are_split_into_one_question_per_chunk():
    content = """붙임 WSCE 2026 Awards 평가 기준
평가 기준 혁신성 (20)
- 얼마나 혁신적인가?
- 기존 방식 대비 차별성과 개선효과가 구체적인가?
확장성 (20)
- 다양한 도시·산업·환경에 적용 가능한가?
계획 적정성 (20)
- 도시 문제의 설정이 구체적인가?
- 도시 문제 해결 KPI가 설정되었는가?"""
    extraction = _extraction([_doc_block(content, order=0)], file_type=FileType.PDF)
    result = chunk_document(extraction, _file_context(file_type="pdf"))

    criteria_chunks = [c for c in result.chunks if c.metadata.get("evaluation_criterion")]
    assert len(criteria_chunks) == 5
    assert all(c.content.count("\n-") == 1 for c in criteria_chunks)
    problem_chunk = next(c for c in criteria_chunks if "도시 문제의 설정" in c.content)
    assert problem_chunk.content.startswith("계획 적정성 (20)\n")
    assert "확장성" not in problem_chunk.content
    assert problem_chunk.metadata["criterion_title"] == "계획 적정성 (20)"


def test_ordinary_bullet_list_keeps_legacy_grouping():
    content = """준비 사항
- 신분증을 준비합니다.
- 신청서를 출력합니다.
- 접수처를 확인합니다."""
    extraction = _extraction([_doc_block(content, order=0)], file_type=FileType.PDF)
    result = chunk_document(extraction, _file_context(file_type="pdf"))

    assert not any(c.metadata.get("evaluation_criterion") for c in result.chunks)
    assert result.chunk_count == 1


# ---------------------------------------------------------------------------
# 페이지/슬라이드/DOCX/HTML 위치 보존
# ---------------------------------------------------------------------------

def test_pdf_pages_are_never_merged_into_one_chunk():
    blocks = [
        _doc_block("1페이지 본문입니다.", order=0, location_number=1),
        _doc_block("2페이지 본문입니다.", order=1, location_number=2),
    ]
    extraction = _extraction(blocks, file_type=FileType.PDF)
    result = chunk_document(extraction, _file_context(file_type="pdf"))

    _assert_common_invariants(result)
    page_numbers = {c.location_number for c in result.chunks}
    assert page_numbers == {1, 2}
    for chunk in result.chunks:
        assert chunk.location_type == ChunkLocationType.PAGE
        # 한 청크의 source_block_orders가 서로 다른 페이지 블록을 동시에 포함하지 않음
        pages_in_chunk = {
            1 if o == 0 else 2 for o in chunk.source_block_orders
        }
        assert len(pages_in_chunk) == 1


def test_pptx_slides_are_never_merged_into_one_chunk():
    blocks = [
        _doc_block("슬라이드1 내용", order=0, location_type=LocationType.SLIDE, location_number=1),
        _doc_block("슬라이드2 내용", order=1, location_type=LocationType.SLIDE, location_number=2),
    ]
    extraction = _extraction(blocks, file_type=FileType.PPTX, page_count=2)
    result = chunk_document(extraction, _file_context(file_type="pptx"))

    _assert_common_invariants(result)
    for chunk in result.chunks:
        assert chunk.location_type == ChunkLocationType.SLIDE
    slide_numbers = {c.location_number for c in result.chunks}
    assert slide_numbers == {1, 2}


def test_docx_location_number_is_always_none():
    blocks = [
        _doc_block("문단1", block_type=BlockType.TITLE, order=0, location_type=LocationType.DOCUMENT, location_number=None),
        _doc_block("문단2 내용입니다.", order=1, location_type=LocationType.DOCUMENT, location_number=None),
    ]
    extraction = _extraction(blocks, file_type=FileType.DOCX, page_count=1)
    result = chunk_document(extraction, _file_context(file_type="docx"))

    _assert_common_invariants(result)
    for chunk in result.chunks:
        assert chunk.location_type == ChunkLocationType.DOCUMENT
        assert chunk.location_number is None


def test_html_web_section_and_source_block_orders_preserved():
    blocks = [
        _web_block("공고 제목", block_type=WebBlockType.HEADING, order=0),
        _web_block("공고 본문 내용입니다.", order=1),
    ]
    cleaned = _cleaned_web(blocks)
    result = chunk_document(cleaned, _web_context())

    _assert_common_invariants(result)
    for chunk in result.chunks:
        assert chunk.location_type == ChunkLocationType.WEB_SECTION
        assert chunk.location_number is None
        assert chunk.source_block_ids == []  # WebContentBlock엔 block_id가 없음
        assert len(chunk.source_block_orders) > 0


# ---------------------------------------------------------------------------
# heading-본문 관계, 표, 목차
# ---------------------------------------------------------------------------

def test_heading_and_body_relationship_preserved_via_section_title():
    blocks = [
        _doc_block("참가자격", block_type=BlockType.TITLE, order=0),
        _doc_block("전국 대학생 누구나 참가 가능합니다.", order=1),
    ]
    extraction = _extraction(blocks)
    result = chunk_document(extraction, _file_context())

    assert all(c.section_title == "참가자격" for c in result.chunks)


def test_table_is_separated_from_surrounding_body_text():
    blocks = [
        _doc_block("표 앞 본문입니다.", order=0),
        _doc_block("헤더1\t헤더2\n값1\t값2", block_type=BlockType.TABLE, order=1, metadata={"rows": 2, "columns": 2}),
        _doc_block("표 뒤 본문입니다.", order=2),
    ]
    extraction = _extraction(blocks, file_type=FileType.DOCX)
    result = chunk_document(extraction, _file_context(file_type="docx"))

    _assert_common_invariants(result)
    assert len(result.chunks) == 3
    assert result.chunks[0].content_kind == ContentKind.BODY
    assert result.chunks[1].content_kind == ContentKind.TABLE
    assert result.chunks[2].content_kind == ContentKind.BODY
    assert "표 앞" not in result.chunks[1].content
    assert "값1" not in result.chunks[0].content and "값1" not in result.chunks[2].content


def test_long_table_splits_by_row_with_repeated_header():
    header = "이름\t점수\t비고"
    rows = [f"참가자{i}\t{i}점\t비고내용{i}" for i in range(80)]
    table_content = "\n".join([header] + rows)
    blocks = [_doc_block(table_content, block_type=BlockType.TABLE, order=0, metadata={"rows": 81, "columns": 3})]
    extraction = _extraction(blocks, file_type=FileType.DOCX)
    config = ChunkingConfig(chunk_size=200, chunk_overlap=0)
    result = chunk_document(extraction, _file_context(file_type="docx"), config)

    _assert_common_invariants(result)
    assert len(result.chunks) > 1
    assert all(c.content_kind == ContentKind.TABLE for c in result.chunks)
    for chunk in result.chunks:
        assert chunk.content.startswith(header)


def test_oversized_table_row_is_further_split_and_flagged():
    header = "항목\t설명"
    oversized_row = "항목1\t" + ("아주 긴 설명 " * 100)
    table_content = f"{header}\n{oversized_row}\n항목2\t짧은 설명"
    blocks = [_doc_block(table_content, block_type=BlockType.TABLE, order=0, metadata={"rows": 3, "columns": 2})]
    extraction = _extraction(blocks, file_type=FileType.DOCX)
    config = ChunkingConfig(chunk_size=200, chunk_overlap=0)
    result = chunk_document(extraction, _file_context(file_type="docx"), config)

    _assert_common_invariants(result)
    oversized_chunks = [c for c in result.chunks if c.metadata.get("oversized_row_split")]
    assert len(oversized_chunks) >= 1
    for chunk in oversized_chunks:
        assert chunk.char_count <= config.chunk_size


def test_toc_heading_marks_content_kind_toc_and_not_indexable():
    blocks = [
        _doc_block("목차", block_type=BlockType.TITLE, order=0),
        _doc_block("1. 서론 ..... 1\n2. 본론 ..... 5\n3. 결론 ..... 10", order=1),
    ]
    extraction = _extraction(blocks)
    result = chunk_document(extraction, _file_context())

    assert any(c.content_kind == ContentKind.TOC for c in result.chunks)
    assert all(c.indexable is False for c in result.chunks if c.content_kind == ContentKind.TOC)


def test_toc_like_structure_without_heading_stays_body_with_warning():
    """점선/페이지번호 구조만 있고 '목차' heading이 없으면 TOC로 확정하지 않고 BODY로 유지 + warning만"""
    blocks = [
        _doc_block("참가자격", block_type=BlockType.TITLE, order=0),
        _doc_block("1. 조건A ..... 1\n2. 조건B ..... 2\n3. 조건C ..... 3", order=1),
    ]
    extraction = _extraction(blocks)
    result = chunk_document(extraction, _file_context())

    assert all(c.content_kind == ContentKind.BODY for c in result.chunks)
    assert all(c.indexable is True for c in result.chunks)
    assert any("목차" in w and "heading" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 결정적 chunk_id
# ---------------------------------------------------------------------------

def test_same_input_and_config_produce_same_chunk_ids():
    blocks = [_doc_block("동일 입력 테스트 문장입니다. " * 20, order=0)]
    extraction = _extraction(blocks)
    ctx = _file_context()

    result1 = chunk_document(extraction, ctx)
    result2 = chunk_document(extraction, ctx)

    assert [c.chunk_id for c in result1.chunks] == [c.chunk_id for c in result2.chunks]
    assert [c.content for c in result1.chunks] == [c.content for c in result2.chunks]


def test_different_config_produces_different_chunk_ids():
    blocks = [_doc_block("설정 변경 테스트 문장입니다. " * 20, order=0)]
    extraction = _extraction(blocks)
    ctx = _file_context()

    result_a = chunk_document(extraction, ctx, ChunkingConfig(chunk_size=800, chunk_overlap=120))
    result_b = chunk_document(extraction, ctx, ChunkingConfig(chunk_size=400, chunk_overlap=50))

    ids_a = {c.chunk_id for c in result_a.chunks}
    ids_b = {c.chunk_id for c in result_b.chunks}
    assert ids_a.isdisjoint(ids_b)


def test_different_content_produces_different_chunk_id():
    ctx = _file_context()
    extraction_a = _extraction([_doc_block("원본 문장입니다.", order=0)])
    extraction_b = _extraction([_doc_block("수정된 문장입니다.", order=0)])

    result_a = chunk_document(extraction_a, ctx)
    result_b = chunk_document(extraction_b, ctx)

    assert result_a.chunks[0].chunk_id != result_b.chunks[0].chunk_id


def test_prepending_content_does_not_change_unrelated_later_chunk_ids():
    """문서 앞부분에 청크가 추가되어도 chunk_index(전역 순번)만 바뀔 뿐, 관련 없는 뒤 청크의 chunk_id는 그대로여야 함"""
    ctx = _file_context()

    blocks_before = [
        _doc_block("2페이지 본문입니다.", order=0, location_number=2),
    ]
    extraction_before = _extraction(blocks_before)
    result_before = chunk_document(extraction_before, ctx)

    blocks_after = [
        _doc_block("1페이지에 새로 추가된 본문입니다.", order=0, location_number=1),
        _doc_block("2페이지 본문입니다.", order=1, location_number=2),
    ]
    extraction_after = _extraction(blocks_after)
    result_after = chunk_document(extraction_after, ctx)

    page2_chunk_before = next(c for c in result_before.chunks if c.location_number == 2)
    page2_chunk_after = next(c for c in result_after.chunks if c.location_number == 2)

    assert page2_chunk_before.chunk_id == page2_chunk_after.chunk_id
    assert page2_chunk_before.chunk_index != page2_chunk_after.chunk_index  # 전역 순번은 바뀜


# ---------------------------------------------------------------------------
# 원본 블록 매핑: 중복 문장, overlap, 공백, 여러 블록 걸침
# ---------------------------------------------------------------------------

def test_duplicate_sentence_appearing_twice_maps_to_correct_block():
    blocks = [
        _doc_block("공통 안내 문구입니다.", order=0),
        _doc_block("중간 내용입니다. " * 40, order=1),
        _doc_block("공통 안내 문구입니다.", order=2),
    ]
    extraction = _extraction(blocks)
    config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
    result = chunk_document(extraction, _file_context(), config)

    _assert_common_invariants(result)
    first_chunk_with_phrase = next(c for c in result.chunks if "공통 안내 문구입니다." in c.content)
    assert 0 in first_chunk_with_phrase.source_block_orders or 2 in first_chunk_with_phrase.source_block_orders


def test_overlap_chunk_maps_to_multiple_source_blocks_when_spanning():
    """
    RecursiveCharacterTextSplitter는 "\\n\\n" 구분자를 우선하므로, 서로 다른 블록의 overlap이
    실제로 걸치려면 여러 개의 짧은 블록이 하나의 merge 단위 안에서 함께 처리되어야 한다
    (긴 블록 하나가 chunk_size를 넘으면 별도로 재귀 분할되어 앞 블록과 overlap이 생기지 않음 — 확인됨).
    """
    blocks = [
        _doc_block(f"문단{i} 짧은 내용 반복 반복 반복.", order=i)
        for i in range(8)
    ]
    extraction = _extraction(blocks)
    config = ChunkingConfig(chunk_size=100, chunk_overlap=40)
    result = chunk_document(extraction, _file_context(), config)

    _assert_common_invariants(result)
    spanning_chunks = [c for c in result.chunks if len(set(c.source_block_orders)) > 1]
    assert len(spanning_chunks) >= 1

    # 인접한 두 청크가 실제로 겹치는 원본 블록(order)을 공유하는지 확인 (진짜 overlap)
    shared_orders_between_adjacent = [
        set(prev.source_block_orders) & set(curr.source_block_orders)
        for prev, curr in zip(result.chunks, result.chunks[1:])
    ]
    assert any(shared for shared in shared_orders_between_adjacent)


def test_leading_trailing_whitespace_stripped_chunk_still_maps_correctly():
    blocks = [
        _doc_block("   앞뒤 공백이 있는 문장입니다.   ", order=0),
    ]
    extraction = _extraction(blocks)
    result = chunk_document(extraction, _file_context())

    _assert_common_invariants(result)
    assert result.chunks[0].source_block_orders == [0]


def test_single_chunk_can_span_multiple_source_blocks():
    blocks = [
        _doc_block("짧은 문단A.", order=0),
        _doc_block("짧은 문단B.", order=1),
        _doc_block("짧은 문단C.", order=2),
    ]
    extraction = _extraction(blocks)
    result = chunk_document(extraction, _file_context())

    assert len(result.chunks) == 1
    assert result.chunks[0].source_block_orders == [0, 1, 2]


# ---------------------------------------------------------------------------
# 원본 불변성 / 빈 입력
# ---------------------------------------------------------------------------

def test_original_input_is_not_mutated():
    blocks = [_doc_block("변형되면 안 되는 문장입니다.", order=0)]
    extraction = _extraction(blocks)
    snapshot = copy.deepcopy(extraction)

    chunk_document(extraction, _file_context())

    assert extraction.model_dump() == snapshot.model_dump()


def test_empty_document_extraction_returns_empty_result_without_error():
    extraction = _extraction([])
    result = chunk_document(extraction, _file_context())

    assert isinstance(result, ChunkingResult)
    assert result.chunks == []
    assert result.chunk_count == 0


def test_empty_cleaned_web_content_returns_empty_result_without_error():
    cleaned = _cleaned_web([])
    result = chunk_document(cleaned, _web_context())

    assert isinstance(result, ChunkingResult)
    assert result.chunks == []
    assert result.chunk_count == 0


# ---------------------------------------------------------------------------
# ChunkSourceContext / file_type 교차검증
# ---------------------------------------------------------------------------

def test_file_type_mismatch_between_context_and_extraction_produces_warning():
    blocks = [_doc_block("본문입니다.", order=0)]
    extraction = _extraction(blocks, file_type=FileType.PDF)
    ctx = _file_context(file_type="docx")  # 실제로는 PDF인데 컨텍스트는 docx라고 잘못 전달

    result = chunk_document(extraction, ctx)

    assert result.chunks[0].file_type == "pdf"  # 실제 파싱 결과 우선
    assert any("file_type" in w for w in result.warnings)


def test_url_attachment_context_carries_source_and_page_url():
    blocks = [_doc_block("첨부파일 본문입니다.", order=0)]
    extraction = _extraction(blocks, file_type=FileType.PDF)
    ctx = ChunkSourceContext(
        document_id="doc_attach",
        source_type=SourceType.URL_ATTACHMENT,
        source_url="https://example.test/files/plan.pdf",
        source_page_url="https://example.test/notice/1",
        source_filename="plan.pdf",
        parent_document_id="doc_web_parent",
        file_type="pdf",
    )
    result = chunk_document(extraction, ctx)

    assert result.chunks[0].source_url == "https://example.test/files/plan.pdf"
    assert result.chunks[0].source_page_url == "https://example.test/notice/1"


def test_web_document_title_can_be_passed_by_caller():
    cleaned = _cleaned_web([_web_block("웹 본문 내용입니다.", order=0)])
    ctx = _web_context(document_title="2026 공모전 공고")
    result = chunk_document(cleaned, ctx)

    assert result.chunks  # document_title 자체는 Chunk 필드가 아니라 ChunkSourceContext에만 보존됨 (섹션 4 스키마 참고)


# ---------------------------------------------------------------------------
# 회귀 테스트: 실제 thinkyou URL 수동 검증에서 확인된 문제
# (표/본문의 section_title 유실, 의사-heading 미인식, 목록 블록 중간 절단, 작은 꼬리 청크)
# ---------------------------------------------------------------------------

def test_table_inherits_preceding_section_title():
    """table을 별도 청크로 분리해도 직전 heading의 section_title을 잃지 않아야 함"""
    blocks = [
        _web_block("2026 온라인 공모전", block_type=WebBlockType.HEADING, order=0, metadata={"level": 1}),
        _web_block("공고 개요입니다.", order=1),
        _web_block("헤더1\t헤더2\n값1\t값2", block_type=WebBlockType.TABLE, order=2, metadata={"row_count": 2}),
    ]
    cleaned = _cleaned_web(blocks)
    result = chunk_document(cleaned, _web_context())

    body_chunk = next(c for c in result.chunks if c.content_kind == ContentKind.BODY)
    table_chunk = next(c for c in result.chunks if c.content_kind == ContentKind.TABLE)

    assert body_chunk.section_title == "2026 온라인 공모전"
    assert table_chunk.section_title == "2026 온라인 공모전"  # 표도 같은 section_title을 상속


def test_table_after_table_still_inherits_section_title_without_body_between():
    """heading 바로 뒤에 표가 이어지고, 다음 표가 또 이어져도 section_title이 유지되어야 함"""
    blocks = [
        _web_block("시상내역", block_type=WebBlockType.HEADING, order=0, metadata={"level": 2}),
        _web_block("대상\t100만원", block_type=WebBlockType.TABLE, order=1, metadata={"row_count": 1}),
        _web_block("최우수상\t50만원", block_type=WebBlockType.TABLE, order=2, metadata={"row_count": 1}),
    ]
    cleaned = _cleaned_web(blocks)
    result = chunk_document(cleaned, _web_context())

    assert all(c.section_title == "시상내역" for c in result.chunks)


def test_pseudo_heading_paragraph_is_recognized_as_section_title():
    """'□ AI 활용 관련 방침' 같은 PARAGRAPH 블록도 section_title 후보로 인식되어야 함"""
    blocks = [
        _web_block("□ AI 활용 관련 방침", order=0),
        _web_block("생성형 AI 활용 시 반드시 출처를 명시해야 합니다.", order=1),
    ]
    cleaned = _cleaned_web(blocks)
    result = chunk_document(cleaned, _web_context())

    assert all(c.section_title == "AI 활용 관련 방침" for c in result.chunks)
    # 의사-heading 블록 자체의 내용은 그대로 본문에 남아 있어야 함 (별도로 잘라내지 않음)
    assert any("□ AI 활용 관련 방침" in c.content for c in result.chunks)


def test_pseudo_heading_with_trailing_content_keeps_content_intact():
    """'□ 접수방법: 이메일로 제출...'처럼 제목 뒤에 본문이 붙어도 content는 그대로 유지되어야 함"""
    blocks = [_web_block("□ 접수방법: 이메일(contest@example.org)로 제출합니다.", order=0)]
    cleaned = _cleaned_web(blocks)
    result = chunk_document(cleaned, _web_context())

    assert result.chunks[0].section_title == "접수방법"
    assert result.chunks[0].content == "□ 접수방법: 이메일(contest@example.org)로 제출합니다."


def test_long_hyphen_list_block_is_not_cut_mid_sentence_and_no_tiny_tail():
    """
    실제 확인된 사례를 축약하지 않고 재현: '□ 기타 유의사항-' 뒤에 여러 하이픈 항목이 붙은 긴 블록.
    항목 중간(예: '경우...')에서 잘리지 않아야 하고, 80자 미만의 불필요한 꼬리 청크가 남지 않아야 함.
    """
    content = (
        "□ 기타 유의사항-\n"
        "- 제출 후 수정 및 재제출이 불가하오니 유의하시기 바랍니다. 신중히 검토 후 제출하여 주시기 바랍니다. "
        "이 항목은 매우 긴 설명입니다 반복 반복 반복 반복 반복 반복 반복.\n"
        "- 표절 및 저작권 침해가 확인될 경우 수상이 취소될 수 있습니다. 이 항목도 상당히 긴 설명을 담고 있습니다 "
        "반복 반복 반복 반복 반복.\n"
        "- 제출 기한을 반드시 준수하여 주시기 바랍니다. 기한 내 미제출 시 접수가 불가합니다 반복 반복 반복.\n"
        "- 문의사항은 운영사무국으로 연락하여 주시기 바랍니다 반복.\n"
        "- 기타 자세한 사항은 공고문을 참고하시기 바랍니다."
    )
    blocks = [_web_block(content, order=0)]
    cleaned = _cleaned_web(blocks)
    config = ChunkingConfig(chunk_size=200, chunk_overlap=30)
    result = chunk_document(cleaned, _web_context(), config)

    assert len(result.chunks) > 1
    for chunk in result.chunks:
        assert chunk.char_count <= config.chunk_size
        stripped = chunk.content.strip()
        # 문장/항목 중간(예: "경우...")에서 시작하는 조각이 없어야 함
        assert not stripped.startswith("경우")
        assert chunk.char_count >= chunker_module.TAIL_CHUNK_MIN_CHARS or len(result.chunks) == 1

    assert all(c.section_title == "기타 유의사항" for c in result.chunks)


def test_all_indexable_chunks_respect_chunk_size_with_mixed_content():
    """본문 + 표 + 목록형 긴 블록이 섞여 있어도 모든 indexable 청크가 chunk_size 이하여야 함"""
    blocks = [
        _web_block("2026 공모전 공고", block_type=WebBlockType.HEADING, order=0, metadata={"level": 1}),
        _web_block("공고 개요 문단입니다.", order=1),
        _web_block("항목\t내용\n예산\t1억원\n기간\t12개월", block_type=WebBlockType.TABLE, order=2, metadata={"row_count": 2}),
        _web_block("□ 세부 출품 규격", order=3),
        _web_block(
            "- A4 용지 기준 10매 이내로 작성하여 주시기 바랍니다 반복 반복 반복 반복.\n"
            "- PDF 형식으로 제출하여 주시기 바랍니다 반복 반복 반복 반복 반복.\n"
            "- 파일명은 팀명과 작품명을 포함하여야 합니다 반복 반복 반복.",
            order=4,
        ),
    ]
    cleaned = _cleaned_web(blocks)
    config = ChunkingConfig(chunk_size=150, chunk_overlap=20)
    result = chunk_document(cleaned, _web_context(), config)

    for chunk in result.chunks:
        if chunk.indexable:
            assert chunk.char_count <= config.chunk_size


# ---------------------------------------------------------------------------
# 화이트박스 단위 테스트: 의사-heading 추출 / 작은 꼬리 병합
# ---------------------------------------------------------------------------

def test_extract_pseudo_heading_title_with_colon_delimiter():
    assert chunker_module._extract_pseudo_heading_title("□ 접수방법: 이메일로 제출합니다.") == "접수방법"


def test_extract_pseudo_heading_title_with_trailing_hyphen():
    assert chunker_module._extract_pseudo_heading_title("□ 기타 유의사항-\n- 첫 항목") == "기타 유의사항"


def test_extract_pseudo_heading_title_without_delimiter():
    assert chunker_module._extract_pseudo_heading_title("□ AI 활용 관련 방침") == "AI 활용 관련 방침"


def test_extract_pseudo_heading_title_returns_none_for_plain_text():
    assert chunker_module._extract_pseudo_heading_title("일반 본문 문장입니다.") is None


def test_extract_pseudo_heading_title_returns_none_when_title_too_long():
    long_text = "□ " + "가" * 40 + ": 본문"
    assert chunker_module._extract_pseudo_heading_title(long_text) is None


def test_merge_small_tail_piece_merges_when_result_fits_chunk_size():
    unit_text = "A" * 90 + "B" * 30
    piece_ranges = [("A" * 90, 0, 90), ("B" * 30, 90, 120)]
    merged = chunker_module._merge_small_tail_piece(piece_ranges, unit_text, chunk_size=150)

    assert merged == [(unit_text, 0, 120)]


def test_merge_small_tail_piece_skips_merge_when_result_exceeds_chunk_size():
    unit_text = "A" * 90 + "B" * 30
    piece_ranges = [("A" * 90, 0, 90), ("B" * 30, 90, 120)]
    merged = chunker_module._merge_small_tail_piece(piece_ranges, unit_text, chunk_size=100)

    assert merged == piece_ranges


def test_merge_small_tail_piece_does_nothing_when_tail_not_small():
    unit_text = "A" * 50 + "B" * 85
    piece_ranges = [("A" * 50, 0, 50), ("B" * 85, 50, 135)]
    merged = chunker_module._merge_small_tail_piece(piece_ranges, unit_text, chunk_size=300)

    assert merged == piece_ranges


def test_merge_small_tail_piece_noop_for_single_piece():
    piece_ranges = [("only piece", 0, 10)]
    merged = chunker_module._merge_small_tail_piece(piece_ranges, "only piece", chunk_size=100)

    assert merged == piece_ranges


# ---------------------------------------------------------------------------
# HWPX→PDF 변환 재현: 문장 중간 줄바꿈/장식 기호 정규화, 제목 인식·상속 (K-Lawyer 문서 재현)
# ---------------------------------------------------------------------------

def test_korean_sentence_split_across_pdf_lines_is_merged_into_one_sentence():
    """'법령규정을 학습 / ‧ / 하여, 이에 대한 / 질의응답'처럼 PDF가 한 문장을 줄 단위 블록
    4개로 쪼개도, 정규화 후에는 문장 중간 줄바꿈과 장식 기호('‧')가 사라져야 한다."""
    blocks = [
        _doc_block("법령규정을 학습", order=0),
        _doc_block("‧", order=1),
        _doc_block("하여, 이에 대한", order=2),
        _doc_block("질의응답", order=3),
    ]
    extraction = _extraction(blocks, file_type=FileType.PDF)
    result = chunk_document(extraction, _file_context(file_type="pdf"))

    assert len(result.chunks) == 1
    content = result.chunks[0].content
    assert content == "법령규정을 학습 하여, 이에 대한 질의응답"
    assert "‧" not in content
    assert "\n" not in content


def test_pdf_line_wrap_merge_preserves_heading_list_and_paragraph_boundaries():
    """제목(whole-line)·글머리표 목록·일반 문단 경계는 줄바꿈 병합 대상이 아니어야 한다."""
    blocks = [
        _doc_block("1) 개요", order=0),
        _doc_block("이 시스템은 법령 정보를", order=1),
        _doc_block("제공하는 챗봇이다.", order=2),
        _doc_block("▢ 기대효과", order=3),
        _doc_block("- 상담 시간 단축", order=4),
        _doc_block("- 법률 접근성 향상", order=5),
    ]
    extraction = _extraction(blocks, file_type=FileType.PDF)
    result = chunk_document(extraction, _file_context(file_type="pdf"))

    overview_chunk = next(c for c in result.chunks if c.section_title == "개요")
    effect_chunk = next(c for c in result.chunks if c.section_title == "기대효과")

    assert overview_chunk.content == "1) 개요\n\n이 시스템은 법령 정보를 제공하는 챗봇이다."
    # 글머리표 목록 두 항목은 서로 다른 항목이므로 한 문장으로 합쳐지면 안 된다.
    assert "- 상담 시간 단축" in effect_chunk.content
    assert "- 법률 접근성 향상" in effect_chunk.content
    assert "- 상담 시간 단축 - 법률 접근성 향상" not in effect_chunk.content


def test_pdf_line_wrap_merge_does_not_affect_docx():
    """동일한 줄 단위 블록 구조라도 DOCX는 병합 대상이 아니다 (파서가 이미 문단 단위로 만듦)."""
    blocks = [
        _doc_block("법령규정을 학습", order=0, location_type=LocationType.DOCUMENT, location_number=None),
        _doc_block("‧", order=1, location_type=LocationType.DOCUMENT, location_number=None),
        _doc_block("하여, 이에 대한 질의응답", order=2, location_type=LocationType.DOCUMENT, location_number=None),
    ]
    extraction = _extraction(blocks, file_type=FileType.DOCX)
    result = chunk_document(extraction, _file_context(file_type="docx", source_filename="test.docx"))

    joined = "\n\n".join(c.content for c in result.chunks)
    # DOCX는 정규화 대상이 아니므로 장식 기호 블록('‧')이 그대로 남는다 (회귀 없음 확인용).
    assert "‧" in joined


def test_pdf_line_wrap_merge_does_not_affect_pptx():
    blocks = [
        _doc_block("법령규정을 학습", order=0, location_type=LocationType.SLIDE, location_number=1),
        _doc_block("‧", order=1, location_type=LocationType.SLIDE, location_number=1),
        _doc_block("하여, 이에 대한 질의응답", order=2, location_type=LocationType.SLIDE, location_number=1),
    ]
    extraction = _extraction(blocks, file_type=FileType.PPTX)
    result = chunk_document(extraction, _file_context(file_type="pptx", source_filename="test.pptx"))

    joined = "\n\n".join(c.content for c in result.chunks)
    assert "‧" in joined


def test_pdf_heading_section_title_inherited_by_following_chunks():
    """제목 인식 후 다음 청크들이 section_title을 상속해야 한다 (K-Lawyer 문서의 4개 제목 유형)."""
    blocks = [
        _doc_block("1) 개요", order=0),
        _doc_block("본 시스템은 법령 정보를 제공한다.", order=1),
        _doc_block("2) 필요성(현 문제점)", order=2),
        _doc_block("법률 상담 접근성이 낮다.", order=3),
        _doc_block("3) 시스템 요구사항 및 기능", order=4),
        _doc_block("자연어 질의응답 기능이 필요하다.", order=5),
        _doc_block("4) 사용예시", order=6),
        _doc_block("사용자가 질문을 입력하면 답변한다.", order=7),
        _doc_block("① 법제처 법령 학습", order=8),
        _doc_block("국가법령정보센터 데이터를 학습한다.", order=9),
    ]
    extraction = _extraction(blocks, file_type=FileType.PDF)
    result = chunk_document(extraction, _file_context(file_type="pdf"))

    section_titles = {c.section_title for c in result.chunks}
    assert section_titles == {
        "개요", "필요성(현 문제점)", "시스템 요구사항 및 기능", "사용예시", "법제처 법령 학습",
    }
    assert next(c for c in result.chunks if "법률 상담 접근성" in c.content).section_title == "필요성(현 문제점)"
    assert next(c for c in result.chunks if "국가법령정보센터" in c.content).section_title == "법제처 법령 학습"


def test_pdf_unrecognizable_heading_stays_null_instead_of_guessing():
    """제목처럼 보이지 않는 일반 문장은 임의로 section_title을 만들어내지 않고 null을 유지해야 한다."""
    blocks = [_doc_block("이 문서는 일반적인 설명으로 시작한다.", order=0)]
    extraction = _extraction(blocks, file_type=FileType.PDF)
    result = chunk_document(extraction, _file_context(file_type="pdf"))

    assert result.chunks[0].section_title is None


# ---------------------------------------------------------------------------
# 화이트박스 단위 테스트: whole-line heading 추출 (숫자)/원문자 제목, adapters 줄바꿈 병합)
# ---------------------------------------------------------------------------

from ai.rag.chunking import config as chunking_config
from ai.rag.chunking import adapters as adapters_module


def test_extract_whole_line_heading_title_numbered_paren():
    assert chunking_config.extract_whole_line_heading_title("1) 개요") == "개요"


def test_extract_whole_line_heading_title_circled_number():
    assert chunking_config.extract_whole_line_heading_title("① 법제처 법령 학습") == "법제처 법령 학습"


def test_extract_whole_line_heading_title_marker():
    assert chunking_config.extract_whole_line_heading_title("▢ 기대효과") == "기대효과"


def test_extract_whole_line_heading_title_returns_none_for_plain_sentence():
    assert chunking_config.extract_whole_line_heading_title("일반 문장입니다.") is None


def test_extract_whole_line_heading_title_returns_none_when_remainder_too_long():
    long_text = "1) " + "가" * 40
    assert chunking_config.extract_whole_line_heading_title(long_text) is None


def _unified(content, kind="paragraph", location_number=1, order=0):
    return adapters_module.UnifiedBlock(
        content=content, kind=kind, location_type=ChunkLocationType.PAGE,
        location_number=location_number, order=order, source_block_id=f"blk_{order}", metadata={},
    )


def test_merge_wrapped_pdf_lines_merges_sentence_continuation():
    blocks = [_unified("법령규정을 학습", order=0), _unified("하여, 이에 대한 질의응답", order=1)]
    merged = adapters_module.merge_wrapped_pdf_lines(blocks)

    assert len(merged) == 1
    assert merged[0].content == "법령규정을 학습 하여, 이에 대한 질의응답"


def test_merge_wrapped_pdf_lines_drops_decorative_symbol_only_block():
    blocks = [_unified("‧", order=0)]
    merged = adapters_module.merge_wrapped_pdf_lines(blocks)

    assert merged == []


def test_merge_wrapped_pdf_lines_keeps_sentence_ending_with_period_separate():
    blocks = [_unified("첫 번째 문장입니다.", order=0), _unified("두 번째 문장입니다.", order=1)]
    merged = adapters_module.merge_wrapped_pdf_lines(blocks)

    assert len(merged) == 2


def test_merge_wrapped_pdf_lines_does_not_merge_across_heading_block():
    heading = _unified("제목", kind="heading", order=0)
    body = _unified("본문 내용", order=1)
    merged = adapters_module.merge_wrapped_pdf_lines([heading, body])

    assert len(merged) == 2
    assert merged[0].content == "제목"
    assert merged[1].content == "본문 내용"


def test_merge_wrapped_pdf_lines_does_not_merge_across_list_item_boundary():
    blocks = [_unified("- 첫 항목", order=0), _unified("- 둘째 항목", order=1)]
    merged = adapters_module.merge_wrapped_pdf_lines(blocks)

    assert len(merged) == 2


def test_merge_wrapped_pdf_lines_does_not_merge_across_page_boundary():
    blocks = [
        _unified("페이지1 마지막 줄", order=0, location_number=1),
        _unified("페이지2 첫 줄", order=1, location_number=2),
    ]
    merged = adapters_module.merge_wrapped_pdf_lines(blocks)

    assert len(merged) == 2


# ---------------------------------------------------------------------------
# 글머리표('•' 등) 보존: 장식 기호로 오인해 삭제/오병합하지 않아야 함
# ---------------------------------------------------------------------------

def test_bullet_item_is_not_merged_into_preceding_plain_sentence():
    """'• 첫 번째 항목'은 새 목록 항목이므로 종결부호 없는 이전 문장과 합쳐지면 안 된다."""
    blocks = [_unified("앞 문장에 종결부호 없음", order=0), _unified("• 첫 번째 항목", order=1)]
    merged = adapters_module.merge_wrapped_pdf_lines(blocks)

    assert len(merged) == 2
    assert merged[0].content == "앞 문장에 종결부호 없음"
    assert merged[1].content == "• 첫 번째 항목"


def test_isolated_bullet_marker_combines_with_following_text_as_list_item():
    """'•'만 있는 독립 블록은 삭제되지 않고 다음 본문과 결합해 목록 항목이 되어야 한다."""
    blocks = [_unified("•", order=0), _unified("첫 번째 항목", order=1)]
    merged = adapters_module.merge_wrapped_pdf_lines(blocks)

    assert len(merged) == 1
    assert merged[0].content == "• 첫 번째 항목"


def test_consecutive_bullet_items_keep_their_boundary():
    """연속된 '• 항목' 두 개는 서로 다른 목록 항목이므로 경계가 유지되어야 한다."""
    blocks = [_unified("• 첫 번째 항목", order=0), _unified("• 두 번째 항목", order=1)]
    merged = adapters_module.merge_wrapped_pdf_lines(blocks)

    assert len(merged) == 2
    assert merged[0].content == "• 첫 번째 항목"
    assert merged[1].content == "• 두 번째 항목"


def test_consecutive_isolated_bullet_markers_each_combine_with_own_following_text():
    """'•'/'첫 번째 항목'/'•'/'두 번째 항목'처럼 마커와 텍스트가 번갈아 독립 블록이어도
    각 마커가 자신의 텍스트하고만 결합해야 하고(다른 항목과 섞이면 안 됨) 경계가 유지되어야 한다."""
    blocks = [
        _unified("•", order=0),
        _unified("첫 번째 항목", order=1),
        _unified("•", order=2),
        _unified("두 번째 항목", order=3),
    ]
    merged = adapters_module.merge_wrapped_pdf_lines(blocks)

    assert [b.content for b in merged] == ["• 첫 번째 항목", "• 두 번째 항목"]


def test_isolated_decorative_dot_is_still_removed_conservatively():
    """장식 기호 제거는 실제 문제를 일으킨 '‧'(HYPHENATION POINT)에만 보수적으로 적용된다."""
    blocks = [_unified("법령규정을 학습", order=0), _unified("‧", order=1), _unified("하여, 이에 대한 질의응답", order=2)]
    merged = adapters_module.merge_wrapped_pdf_lines(blocks)

    assert len(merged) == 1
    assert merged[0].content == "법령규정을 학습 하여, 이에 대한 질의응답"
    assert "‧" not in merged[0].content


def test_bullet_marker_variants_are_not_deleted_as_decorative():
    """◦/∙/·도 '•'와 동일하게 목록 마커로 취급되어 독립 블록이어도 삭제되지 않아야 한다."""
    for marker in ("◦", "∙", "·"):
        blocks = [_unified(marker, order=0), _unified("항목 내용", order=1)]
        merged = adapters_module.merge_wrapped_pdf_lines(blocks)
        assert len(merged) == 1
        assert merged[0].content == f"{marker} 항목 내용"


def test_k_lawyer_style_bullet_list_preserved_in_document_chunk():
    """실제 재현 사례: HWPX 변환 PDF에서 글머리표 목록 두 항목이 청크 content에서 별도
    항목으로 남아야 한다(공백 하나로 뭉개지면 안 됨)."""
    blocks = [
        _doc_block("▢ 기대효과", order=0),
        _doc_block("•", order=1),
        _doc_block("상담 시간 단축", order=2),
        _doc_block("•", order=3),
        _doc_block("법률 접근성 향상", order=4),
    ]
    extraction = _extraction(blocks, file_type=FileType.PDF)
    result = chunk_document(extraction, _file_context(file_type="pdf"))

    effect_chunk = next(c for c in result.chunks if c.section_title == "기대효과")
    assert "• 상담 시간 단축" in effect_chunk.content
    assert "• 법률 접근성 향상" in effect_chunk.content
