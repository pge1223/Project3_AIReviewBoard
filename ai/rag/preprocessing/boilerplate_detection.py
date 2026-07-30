# 작성자: 용준/Claude(2026-07-30, 요청: RAG-007 색인 청크 품질 정제)
# 목적: html_cleaner.py의 페이지 내부 규칙만으로는 잡히지 않는 "여러 문서/페이지에
#       걸쳐 반복되는 문구"(사이트 공통 메뉴, PDF 머리글·바닥글)를 탐지한다.
#       html_cleaner.py::clean_page_content()는 한 페이지만 보고 판단하므로 원천적으로
#       "이 문구가 다른 문서에도 반복되는지"는 알 수 없다 — 이 모듈은 배치(여러
#       문서/페이지) 전체를 보고 판단하는 별도 단계다.
#
#       알고리즘은 scripts/audit_rag007_collection.py가 이미 실제 운영 데이터에서
#       검증한 "첫/마지막 1~2줄, 길이 8~120자, ≥N개의 서로 다른 group에 반복 등장하면
#       boilerplate"를 그대로 재사용 가능한 순수 함수로 뽑아낸 것이다 — 감사 스크립트와
#       색인 스크립트가 서로 다른 로직을 쓰면 "감사에서는 잡히는데 색인에서는 안
#       걸러지는" 불일치가 생기므로, 감사 스크립트도 이 모듈을 import해서 쓴다.
from __future__ import annotations

import re
from collections import defaultdict

_MIN_BOUNDARY_LINE_LENGTH = 8
_MAX_BOUNDARY_LINE_LENGTH = 120


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def extract_boundary_lines(text: str, *, min_len: int = _MIN_BOUNDARY_LINE_LENGTH, max_len: int = _MAX_BOUNDARY_LINE_LENGTH) -> set[str]:
    """텍스트의 첫/마지막 1~2줄 중 길이 8~120자인 것만 뽑는다 — 머리글/바닥글/메뉴
    항목은 보통 이 범위의 짧은 줄로 반복되고, 너무 짧은 줄(예: 페이지 번호 "3")이나
    너무 긴 줄(본문 문단)은 반복 탐지 대상에서 제외한다."""
    lines = [_normalized_text(line) for line in (text or "").splitlines()]
    lines = [line for line in lines if min_len <= len(line) <= max_len]
    return set(lines[:2] + lines[-2:])


def find_repeated_boundary_texts(
    items: list[tuple[str, str]],
    *,
    min_group_count: int = 3,
    min_len: int = _MIN_BOUNDARY_LINE_LENGTH,
    max_len: int = _MAX_BOUNDARY_LINE_LENGTH,
) -> set[str]:
    """items = [(group_key, text), ...]. 각 텍스트의 경계 줄을 뽑아, 서로 다른
    group_key ≥ min_group_count개에 반복 등장하는 줄만 반환한다.

    두 가지 용도로 재사용한다(group_key만 다르게 넘기면 됨):
    - 문서 간 반복(사이트 공통 메뉴 등): group_key=document_id, 입력=배치 전체의
      완성된 청크 텍스트.
    - 문서 내 반복(PDF 머리글/바닥글): group_key=page_number, 입력=청킹 전 페이지별
      원문 텍스트(같은 document_id 안에서만 호출).

    min_group_count=3은 우연히 겹치는 문구(예: 흔한 인사말)를 반복 문구로 오판하지
    않기 위한 최소 임계값이다 — 실제 운영 데이터(reports/rag007_e2e_20260730/
    index_quality.json)에서 이 값으로 37건의 의심 청크를 찾아냈다."""
    boundary_groups: dict[str, set[str]] = defaultdict(set)
    for group_key, text in items:
        for line in extract_boundary_lines(text, min_len=min_len, max_len=max_len):
            boundary_groups[line].add(group_key)
    return {line for line, groups in boundary_groups.items() if len(groups) >= min_group_count}


def is_boilerplate_text(text: str, repeated_lines: set[str]) -> bool:
    """text의 경계 줄이 repeated_lines와 하나라도 겹치면 boilerplate로 판단한다."""
    if not repeated_lines:
        return False
    return bool(extract_boundary_lines(text) & repeated_lines)


__all__ = [
    "extract_boundary_lines",
    "find_repeated_boundary_texts",
    "is_boilerplate_text",
]
