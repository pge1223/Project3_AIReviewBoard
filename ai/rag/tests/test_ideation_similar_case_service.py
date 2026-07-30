"""
Unit Tests for ai.rag.orchestration.ideation_similar_case_service (RAG-006 wiring)
=======================================================================================
용준/Claude(2026-07-29, 요청: target/criteria/외부근거/전문가판단 분리). similar_success_cases
컬렉션은 이 작업 시점에 실제 시딩된 문서가 없다 — 이 테스트는 real chromadb를 세우지 않고
fake SimilarCaseSearchService(.search()만 구현)로 "빈 컬렉션에서도 예외 없이 진행"과
"예상치 못한 예외도 fail-closed로 처리"만 검증한다(ideation_external_evidence_service의
동일 성격 테스트와 같은 원칙).
"""

from ai.rag.orchestration.ideation_similar_case_service import (
    make_ideation_similar_case_lookup,
    search_ideation_similar_cases,
)
from ai.rag.similar_cases.schemas import ComparisonMode, SimilarCaseSearchResponse


class _FakeSimilarCaseService:
    def __init__(self, response=None, *, raise_error=None):
        self._response = response
        self._raise_error = raise_error
        self.calls: list = []

    def search(self, request):
        self.calls.append(request)
        if self._raise_error is not None:
            raise self._raise_error
        return self._response


def _empty_response(query_text: str = "질의") -> SimilarCaseSearchResponse:
    return SimilarCaseSearchResponse(
        results=[],
        total_results=0,
        has_rejected_cases=False,
        comparison_mode=ComparisonMode.SELECTED_CASE_GAP,
        query_text=query_text,
        warnings=["현재 조건과 유사한 공개 사례를 찾지 못했습니다."],
    )


def test_empty_collection_returns_empty_result_without_error():
    service = _FakeSimilarCaseService(response=_empty_response())
    result = search_ideation_similar_cases("planning_expert", "스마트시티 공모전 아이디어", service)

    assert result["similar_cases"] == []
    assert result["warnings"] == ["현재 조건과 유사한 공개 사례를 찾지 못했습니다."]


def test_unexpected_exception_is_fail_closed_not_propagated():
    service = _FakeSimilarCaseService(raise_error=RuntimeError("boom"))
    result = search_ideation_similar_cases("dev_expert", "질의", service)

    assert result["similar_cases"] == []
    assert len(result["warnings"]) == 1


def test_cache_avoids_duplicate_search_calls():
    service = _FakeSimilarCaseService(response=_empty_response())
    cache: dict = {}
    search_ideation_similar_cases("planning_expert", "질의", service, cache=cache)
    search_ideation_similar_cases("planning_expert", "질의", service, cache=cache)

    assert len(service.calls) == 1


def test_make_lookup_returns_callable_matching_persona_query_contract():
    service = _FakeSimilarCaseService(response=_empty_response())
    lookup = make_ideation_similar_case_lookup(service)

    result = lookup("planning_expert", "질의")
    assert result == {"similar_cases": [], "warnings": [_empty_response().warnings[0]]}
