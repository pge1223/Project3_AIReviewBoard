# 작성자: 용준/Claude(2026-07-30, 요청: RAG-007 색인 청크 품질 정제)
# 목적: 100자 미만 청크를 무조건 삭제하지 않는다(요청 5번) — 같은 document_id의
#       인접(같은/한 페이지 차이) page + 같은 section인 이웃 청크와 먼저 병합을
#       시도하고, 병합할 수 없으면(용량 초과·다른 section 등) 법령 조문·통계
#       수치·정의문처럼 짧아도 독립적으로 의미 있는 패턴만 보존하며, 그 외(메뉴·버튼·
#       페이지 번호 등 의미 없는 짧은 텍스트)는 제거한다.
#
#       ai/rag/chunking/chunker.py::_merge_small_tail_piece가 "같은 논리 단위 안의
#       마지막 조각"만 병합하는 좁은 범위인 것과 달리, 이 모듈은 청킹이 끝난 뒤 문서
#       전체 청크 시퀀스를 대상으로 인접 청크와 병합한다 — 별도 정책이 필요한 이유는
#       RAG-007 색인 스크립트가 여러 소스를 한 번에 처리하며, 그 결과물(chunk_document의
#       Chunk가 아니라 ExternalEvidenceDocument로 변환되기 직전의 plain dict)에 대해
#       한 번 더 정리가 필요하기 때문이다.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_MIN_CHARS_DEFAULT = 100
_MAX_CHARS_DEFAULT = 800

# 요청 5번 "독립적으로 의미가 있는 항목"— 짧아도 보존한다.
_PRESERVE_SHORT_CHUNK_PATTERNS: list[re.Pattern] = [
    re.compile(r"제\s*\d+\s*조"),  # 법령 조문 번호
    re.compile(r"\d+(\.\d+)?\s*(%|퍼센트|명|건|원|만원|억원|개)"),  # 통계 지표명과 수치
    re.compile(r"(이|가)?란\s|을\s*말한다|의\s*정의|정의(한다|는)"),  # 정의문
    re.compile(r"제\s*\d+\s*항|제\s*\d+\s*호"),  # 조문 하위 항/호
]


@dataclass
class ShortChunkResult:
    kept: list[dict[str, Any]] = field(default_factory=list)
    merged_count: int = 0
    preserved_short: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)


def _matches_preserve_pattern(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PRESERVE_SHORT_CHUNK_PATTERNS)


def _same_or_adjacent_page(a_page: Any, b_page: Any) -> bool:
    if a_page is None and b_page is None:
        return True
    if a_page is None or b_page is None:
        return False
    try:
        return abs(int(a_page) - int(b_page)) <= 1
    except (TypeError, ValueError):
        return False


def _same_section(a_section: Any, b_section: Any) -> bool:
    a = (a_section or "").strip()
    b = (b_section or "").strip()
    if not a and not b:
        return True
    return a == b


def _extend_page_range(target: dict[str, Any], other: dict[str, Any]) -> None:
    """병합 시 target의 page_start/page_end를 target·other 양쪽의 기존 범위(또는
    단일 page)를 모두 포함하도록 넓힌다 — 인접 페이지 청크가 합쳐지면 병합된 청크가
    실제로 걸치는 페이지 범위를 그대로 보존하기 위함(요청: "병합 후 출처 범위 metadata
    유지")."""
    starts = [
        value
        for value in (target.get("page_start"), target.get("page"), other.get("page_start"), other.get("page"))
        if value is not None
    ]
    ends = [
        value
        for value in (target.get("page_end"), target.get("page"), other.get("page_end"), other.get("page"))
        if value is not None
    ]
    if starts:
        target["page_start"] = min(starts)
    if ends:
        target["page_end"] = max(ends)


def _mergeable(a: dict[str, Any], b: dict[str, Any], max_chars: int) -> bool:
    if a.get("document_id") != b.get("document_id"):
        return False
    if not _same_or_adjacent_page(a.get("page"), b.get("page")):
        return False
    if not _same_section(a.get("section"), b.get("section")):
        return False
    combined_len = len((a.get("content") or "").strip()) + len((b.get("content") or "").strip()) + 1
    return combined_len <= max_chars


def merge_or_filter_short_chunks(
    chunks: list[dict[str, Any]],
    *,
    min_chars: int = _MIN_CHARS_DEFAULT,
    max_chars: int = _MAX_CHARS_DEFAULT,
) -> ShortChunkResult:
    """chunks는 document_id별로 원래 순서(문서 내 등장 순서)가 유지된 상태로 입력되어야
    한다 — 서로 다른 document_id가 섞여 있어도 document_id별로 순서를 보존한 채 묶어서
    처리한다(입력 리스트 자체의 document_id 간 순서는 무관).

    각 짧은 청크(100자 미만)에 대해: (1) 직전에 이미 처리된 이웃 청크와 병합 시도,
    (2) 실패하면 바로 다음 청크와 병합 시도(둘 다 같은 document_id + 같은/인접 page +
    같은 section이고 병합 결과가 max_chars 이내일 때만), (3) 그래도 안 되면 보존
    패턴(법령 조문/통계 수치/정의문) 확인 후 짧아도 유지, (4) 어디에도 해당 없으면
    제거 목록으로 뺀다.

    100자 미만 청크를 무조건 0건으로 만드는 것이 목표가 아니다 — 반환된
    ShortChunkResult.preserved_short로 "짧지만 의도적으로 보존한" 항목을 그대로 보고할
    수 있다."""
    by_document: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for chunk in chunks:
        document_id = chunk.get("document_id")
        if document_id not in by_document:
            by_document[document_id] = []
            order.append(document_id)
        by_document[document_id].append(chunk)

    result = ShortChunkResult()

    for document_id in order:
        doc_chunks = by_document[document_id]
        working: list[dict[str, Any]] = []
        index = 0
        total = len(doc_chunks)
        while index < total:
            chunk = doc_chunks[index]
            content = (chunk.get("content") or "").strip()
            if len(content) >= min_chars:
                working.append(dict(chunk))
                index += 1
                continue

            if working and _mergeable(working[-1], chunk, max_chars):
                previous = working[-1]
                _extend_page_range(previous, chunk)
                previous["content"] = (previous.get("content") or "").rstrip() + "\n" + content
                metadata = previous.setdefault("metadata", {})
                metadata.setdefault("merged_from_chunk_ids", []).append(chunk.get("chunk_id"))
                result.merged_count += 1
                index += 1
                continue

            if index + 1 < total and _mergeable(chunk, doc_chunks[index + 1], max_chars):
                next_chunk = dict(doc_chunks[index + 1])
                _extend_page_range(next_chunk, chunk)
                next_chunk["content"] = content + "\n" + (next_chunk.get("content") or "").lstrip()
                metadata = next_chunk.setdefault("metadata", {})
                metadata.setdefault("merged_from_chunk_ids", []).append(chunk.get("chunk_id"))
                working.append(next_chunk)
                result.merged_count += 1
                index += 2
                continue

            if _matches_preserve_pattern(content):
                working.append(dict(chunk))
                result.preserved_short.append(chunk)
                index += 1
                continue

            result.dropped.append(chunk)
            index += 1

        result.kept.extend(working)

    return result


__all__ = ["ShortChunkResult", "merge_or_filter_short_chunks"]
