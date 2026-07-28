"""Canonical document types used by indexing and ideation retrieval."""

from __future__ import annotations

import re
from typing import Optional

ANNOUNCEMENT = "announcement"
EVALUATION_CRITERIA = "evaluation_criteria"
APPLICATION_FORM_POC = "application_form_poc"
APPLICATION_FORM_BEST_PRACTICE = "application_form_best_practice"
OTHER = "other"

_CANONICAL_TYPES = {
    ANNOUNCEMENT,
    EVALUATION_CRITERIA,
    APPLICATION_FORM_POC,
    APPLICATION_FORM_BEST_PRACTICE,
    OTHER,
}

_LEGACY_TYPE_MAPPING = {
    "공고문": ANNOUNCEMENT,
    "평가기준": EVALUATION_CRITERIA,
    "기타": OTHER,
}

_ANNOUNCEMENT_QUERY_RE = re.compile(
    r"개최\s*목적|주최|주관|참가\s*자격|신청\s*자격|컨소시엄|접수처|제출\s*서류|"
    r"모집\s*기간|결과\s*발표|발표\s*심사|추진\s*일정|공고|"
    r"제출.+조건|부문.+조건|압축\s*파일|파일명"
)
_EVALUATION_QUERY_RE = re.compile(r"평가\s*항목|평가\s*기준|심사\s*기준|배점|가점|선정\s*기준")
_POC_QUERY_RE = re.compile(
    r"실증|PoC|P\.?O\.?C|AI\s*도입(?:의)?\s*필요성",
    re.IGNORECASE,
)
_BEST_PRACTICE_QUERY_RE = re.compile(r"우수\s*사례|우수사례")
_FORM_QUERY_RE = re.compile(
    r"작성|서식|수행\s*보고서|요약서|적어야|기술해야|제시해야|어떻게\s*제시|어떤\s*내용"
)


def normalize_document_type(
    value: Optional[str],
    *,
    document_name: Optional[str] = None,
    text: Optional[str] = None,
) -> str:
    """Return one canonical type, including a deterministic legacy fallback.

    Existing v5 chunks predate Chroma ``document_type`` metadata.  They are
    classified from their title/content at retrieval time so a full re-embed is
    not required merely to gain type-aware ranking.
    """
    normalized_value = str(value or "").strip()
    if normalized_value in _CANONICAL_TYPES:
        return normalized_value
    if normalized_value in _LEGACY_TYPE_MAPPING:
        return _LEGACY_TYPE_MAPPING[normalized_value]

    haystack = f"{document_name or ''}\n{text or ''}"
    lowered = haystack.lower()

    if _POC_QUERY_RE.search(haystack) and (
        "작성 요령" in haystack
        or "수행보고서" in haystack
        or "신청 서식" in haystack
        or "신청서" in haystack
    ):
        return APPLICATION_FORM_POC
    if _BEST_PRACTICE_QUERY_RE.search(haystack) and (
        "작성 요령" in haystack or "신청 서식" in haystack or "신청서" in haystack
    ):
        return APPLICATION_FORM_BEST_PRACTICE

    # The old classifier only emitted "신청서양식".  Split it using the
    # filename/content, while keeping an unknown form out of announcement
    # ranking.
    if normalized_value == "신청서양식":
        if _POC_QUERY_RE.search(haystack):
            return APPLICATION_FORM_POC
        if _BEST_PRACTICE_QUERY_RE.search(haystack):
            return APPLICATION_FORM_BEST_PRACTICE
        return OTHER

    if "평가기준" in haystack or "평가 기준" in haystack or (
        "평가항목" in haystack and "배점" in haystack
    ):
        return EVALUATION_CRITERIA
    if (
        "공고문" in haystack
        or ("공고" in haystack and ("접수" in haystack or "신청" in haystack))
        or ("주최" in haystack and "주관" in haystack)
    ):
        return ANNOUNCEMENT

    # Preserve a small filename fallback for legacy mojibake titles: stable
    # ASCII tokens such as PoC and attachment numbers survive the corruption.
    if "poc" in lowered:
        return APPLICATION_FORM_POC
    return OTHER


def preferred_document_types(query: str) -> tuple[str, ...]:
    """Return document types relevant to a query, in preference order."""
    preferences: list[str] = []
    announcement_query = bool(_ANNOUNCEMENT_QUERY_RE.search(query))
    form_query = bool(_FORM_QUERY_RE.search(query))
    if _EVALUATION_QUERY_RE.search(query):
        preferences.extend((EVALUATION_CRITERIA, ANNOUNCEMENT))
    # "우수사례"가 들어가도 참가 조건·접수·파일명처럼 대회 규칙을 묻는 질문은
    # 신청서 양식보다 공고문이 정답이다. 반대로 작성/서식 질문은 해당 양식을 먼저 둔다.
    if announcement_query and not form_query:
        preferences.append(ANNOUNCEMENT)
    if _POC_QUERY_RE.search(query):
        preferences.append(APPLICATION_FORM_POC)
    if _BEST_PRACTICE_QUERY_RE.search(query):
        preferences.append(APPLICATION_FORM_BEST_PRACTICE)
    if announcement_query:
        preferences.append(ANNOUNCEMENT)
    return tuple(dict.fromkeys(preferences))


__all__ = [
    "ANNOUNCEMENT",
    "EVALUATION_CRITERIA",
    "APPLICATION_FORM_POC",
    "APPLICATION_FORM_BEST_PRACTICE",
    "OTHER",
    "normalize_document_type",
    "preferred_document_types",
]
