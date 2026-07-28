"""
Unit Tests for ai.rag.orchestration.ideation_trend_search_service (주제 브레인스토밍 —
네이버 트렌드 검색 연동)
====================================================================================
실제 네트워크/Chroma 없이 가짜 TrendSearchService만 사용한다. candidate_planning 노드
통합은 ai/meeting/tests/test_ideation_discovery_graph.py 쪽에서 별도로 검증한다(ai/meeting은
ai.rag를 import하지 않아야 하므로 이 파일에서는 orchestration 계층만 테스트한다).
"""

from ai.rag.orchestration.ideation_trend_search_service import (
    make_ideation_trend_search_lookup,
    search_ideation_trend_evidence,
)
from ai.rag.trend_search.exceptions import TrendSearchError
from ai.rag.trend_search.schemas import TrendEvidenceItem, TrendSearchResponse


class _FakeTrendSearchService:
    def __init__(self, *, response=None, error=None):
        self._response = response
        self._error = error
        self.calls: list[str] = []

    def search(self, request):
        self.calls.append(request.query_context)
        if self._error is not None:
            raise self._error
        return self._response


def _response(*, used=True, items=None, warnings=None):
    return TrendSearchResponse(
        items=items or [],
        query_text="query",
        used_naver_search=used,
        warnings=warnings or [],
    )


def test_search_returns_trend_evidence_dicts():
    item = TrendEvidenceItem(
        title="이슈", snippet="설명", source_url="https://a", publisher="p", published_at="2026-07-20"
    )
    service = _FakeTrendSearchService(response=_response(items=[item]))

    result = search_ideation_trend_evidence("공모전 최근 이슈", service)

    assert result["used_naver_search"] is True
    assert len(result["trend_evidence"]) == 1
    entry = result["trend_evidence"][0]
    assert entry["title"] == "이슈"
    assert entry["source_url"] == "https://a"
    assert entry["reference_only"] is True


def test_no_results_does_not_raise():
    service = _FakeTrendSearchService(response=_response(used=False, warnings=["비활성화"]))
    result = search_ideation_trend_evidence("질의", service)
    assert result["trend_evidence"] == []
    assert result["warnings"] == ["비활성화"]


def test_search_failure_is_fail_open():
    service = _FakeTrendSearchService(error=TrendSearchError("네트워크 오류"))
    result = search_ideation_trend_evidence("질의", service)
    assert result["trend_evidence"] == []
    assert result["used_naver_search"] is False
    assert result["warnings"]


def test_unexpected_exception_is_fail_open():
    service = _FakeTrendSearchService(error=RuntimeError("예상치 못한 오류"))
    result = search_ideation_trend_evidence("질의", service)
    assert result["trend_evidence"] == []


def test_lookup_stops_after_max_queries_per_session():
    """요청: "트렌드 검색은 세션당(사실상 요청당) 딱 한 번만 실행돼야 한다" — candidate_planning
    이 같은 lookup 콜러블을 여러 번 불러도(예: 재추천 재시도) 캡을 넘으면 실제 검색 없이
    빈 결과만 반환해야 한다."""
    service = _FakeTrendSearchService(response=_response(items=[]))
    lookup = make_ideation_trend_search_lookup(service, max_queries_per_session=1)

    first = lookup("질의1")
    second = lookup("질의2")

    assert service.calls == ["질의1"]  # 두 번째 호출은 실제 검색을 하지 않았다.
    assert first["used_naver_search"] is True
    assert second == {"trend_evidence": [], "used_naver_search": False, "warnings": []}


def test_lookup_signature_accepts_extra_kwargs_without_error():
    """ai/meeting/graph/ideation_conv_discovery.py::_call_trend_search_lookup은
    trend_search_lookup(query_text)만 호출하지만, 다른 호출부가 생기더라도 안전하도록
    lookup이 **kwargs를 받아 흡수한다."""
    service = _FakeTrendSearchService(response=_response(items=[]))
    lookup = make_ideation_trend_search_lookup(service)
    result = lookup("질의", unexpected_kwarg="ignored")
    assert isinstance(result, dict)
    assert "trend_evidence" in result
