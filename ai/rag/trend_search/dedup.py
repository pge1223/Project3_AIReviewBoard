"""
Trend Search Result Post-processing (실시간 이슈 검색)
============================================================
네이버 검색 API 원시 응답에는 동일 기사 재배포·출처 없는 결과가 섞여 들어올 수
있다. 여기서는 순수 함수로 중복 제거·필수 필드 검증만 한다 — 유사도 알고리즘을
새로 들여오지 않고, 완전 일치(정규화 후) 기준의 단순한 필터만 적용한다.
"""

import re


def filter_has_source(items: list[dict]) -> list[dict]:
    """title/source_url이 모두 채워진 결과만 남긴다 — 출처 없는 자료는 인용
    대상이 될 수 없다(RAG-007 source_validator와 동일한 원칙)."""
    return [
        item
        for item in items
        if (item.get("title") or "").strip() and (item.get("source_url") or "").strip()
    ]


def dedupe_by_url(items: list[dict]) -> list[dict]:
    """동일 source_url이 여러 번 나오면 처음 등장한 것만 남긴다."""
    seen: set[str] = set()
    result: list[dict] = []
    for item in items:
        url = (item.get("source_url") or "").strip()
        if url in seen:
            continue
        seen.add(url)
        result.append(item)
    return result


def dedupe_by_title(items: list[dict]) -> list[dict]:
    """제목이 사실상 동일한(같은 기사의 재배포) 결과를 제거한다. 공백·특수문자를
    지운 뒤 완전히 같은 문자열인 경우만 중복으로 본다 — 과도하게 억제하지 않는다."""
    seen_titles: set[str] = set()
    result: list[dict] = []
    for item in items:
        normalized = re.sub(r"[\s\W]+", "", item.get("title") or "").lower()
        if not normalized or normalized in seen_titles:
            continue
        seen_titles.add(normalized)
        result.append(item)
    return result


__all__ = ["filter_has_source", "dedupe_by_url", "dedupe_by_title"]
