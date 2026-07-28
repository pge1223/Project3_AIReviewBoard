"""
Unit Tests for ai.rag.trend_search (주제 브레인스토밍 — 네이버 트렌드 검색)
====================================================================================
requests_mock으로 네이버 검색 오픈API 호출을 가로채며, 실제 외부 네트워크에는 접속하지
않는다(ai/rag/tests/test_url_loader.py와 동일한 컨벤션). ai.rag.external_research(RAG-007)
와는 별도 패키지이므로 그 테스트/픽스처를 재사용하지 않는다.
"""

import pytest
import requests
import requests_mock

from ai.rag.trend_search.config import TrendSearchConfig
from ai.rag.trend_search.dedup import dedupe_by_title, dedupe_by_url, filter_has_source
from ai.rag.trend_search.exceptions import TrendProviderTimeoutError, TrendProviderUnavailableError
from ai.rag.trend_search.naver_provider import NAVER_NEWS_SEARCH_URL, NaverSearchProvider
from ai.rag.trend_search.schemas import TrendSearchRequest
from ai.rag.trend_search.service import TrendSearchService


def _naver_item(title="제목", link="https://news.example.com/1", description="설명", pub_date="Mon, 27 Jul 2026 09:00:00 +0900"):
    return {
        "title": title,
        "originallink": link,
        "link": link,
        "description": description,
        "pubDate": pub_date,
    }


class TestNaverSearchProvider:
    def test_disabled_without_credentials_raises_unavailable(self):
        provider = NaverSearchProvider(enabled=True)  # client_id/secret 없음
        with requests_mock.Mocker() as m:
            with pytest.raises(TrendProviderUnavailableError):
                provider.search("query")
            assert not m.called

    def test_enabled_false_raises_unavailable_even_with_credentials(self):
        provider = NaverSearchProvider(client_id="id", client_secret="secret", enabled=False)
        with pytest.raises(TrendProviderUnavailableError):
            provider.search("query")

    def test_returns_parsed_items_without_real_network(self):
        provider = NaverSearchProvider(client_id="id", client_secret="secret", enabled=True)
        with requests_mock.Mocker() as m:
            m.get(
                NAVER_NEWS_SEARCH_URL,
                json={"items": [_naver_item(title="<b>고령층</b> 키오스크 이슈", description="설명 &quot;내용&quot;")]},
            )
            results = provider.search("키오스크")

        assert len(results) == 1
        item = results[0]
        assert item["title"] == "고령층 키오스크 이슈"  # HTML 태그 제거
        assert item["snippet"] == '설명 "내용"'  # HTML 엔티티 복원
        assert item["source_url"] == "https://news.example.com/1"
        assert item["publisher"] == "news.example.com"
        assert item["published_at"] == "2026-07-27"

    def test_timeout_raises_timeout_error(self):
        provider = NaverSearchProvider(
            client_id="id", client_secret="secret", enabled=True, timeout_seconds=0.01
        )
        with requests_mock.Mocker() as m:
            m.get(NAVER_NEWS_SEARCH_URL, exc=requests.exceptions.Timeout)
            with pytest.raises(TrendProviderTimeoutError):
                provider.search("query")

    def test_http_error_status_raises_unavailable(self):
        provider = NaverSearchProvider(client_id="id", client_secret="secret", enabled=True)
        with requests_mock.Mocker() as m:
            m.get(NAVER_NEWS_SEARCH_URL, status_code=500)
            with pytest.raises(TrendProviderUnavailableError):
                provider.search("query")

    def test_items_missing_link_still_parsed_with_empty_source_url(self):
        provider = NaverSearchProvider(client_id="id", client_secret="secret", enabled=True)
        with requests_mock.Mocker() as m:
            m.get(NAVER_NEWS_SEARCH_URL, json={"items": [{"title": "t", "description": "d"}]})
            results = provider.search("query")
        assert results[0]["source_url"] == ""

    def test_no_real_network_call_made(self):
        """이 provider가 requests 이외의 방식으로 네트워크에 접근하지 않는지 소스로 재확인
        (ai/rag/tests/test_external_research_provider.py::test_no_real_network_call_made와
        동일한 원칙)."""
        import inspect

        from ai.rag.trend_search import naver_provider

        source = inspect.getsource(naver_provider)
        for forbidden in ("httpx.", "urllib.request", "http.client"):
            assert forbidden not in source


class TestDedup:
    def test_filter_has_source_drops_missing_title_or_url(self):
        items = [
            {"title": "t1", "source_url": "https://a"},
            {"title": "", "source_url": "https://b"},
            {"title": "t3", "source_url": ""},
        ]
        assert filter_has_source(items) == [{"title": "t1", "source_url": "https://a"}]

    def test_dedupe_by_url_keeps_first_occurrence(self):
        items = [
            {"title": "a", "source_url": "https://x"},
            {"title": "a-재배포", "source_url": "https://x"},
            {"title": "b", "source_url": "https://y"},
        ]
        deduped = dedupe_by_url(items)
        assert [i["source_url"] for i in deduped] == ["https://x", "https://y"]

    def test_dedupe_by_title_ignores_whitespace_and_case(self):
        items = [
            {"title": "고령층 키오스크 이슈"},
            {"title": "고령층  키오스크  이슈"},  # 공백만 다름
            {"title": "다른 이슈"},
        ]
        deduped = dedupe_by_title(items)
        assert len(deduped) == 2


class _FakeProvider:
    def __init__(self, *, enabled=True, results=None, error=None):
        self._enabled = enabled
        self._results = results if results is not None else []
        self._error = error
        self.name = "fake_naver"
        self.calls = 0

    @property
    def enabled(self):
        return self._enabled

    def search(self, query_text, *, display=20):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._results


class TestTrendSearchService:
    def test_skips_when_config_disabled(self):
        provider = _FakeProvider()
        service = TrendSearchService(provider, config=TrendSearchConfig(enabled=False))
        response = service.search(TrendSearchRequest(query_context="이슈"))
        assert response.items == []
        assert response.used_naver_search is False
        assert response.warnings
        assert provider.calls == 0

    def test_skips_when_provider_not_ready(self):
        provider = _FakeProvider(enabled=False)
        service = TrendSearchService(provider, config=TrendSearchConfig(enabled=True))
        response = service.search(TrendSearchRequest(query_context="이슈"))
        assert response.items == []
        assert provider.calls == 0

    def test_fail_open_on_provider_error(self):
        provider = _FakeProvider(error=TrendProviderUnavailableError("네트워크 오류"))
        service = TrendSearchService(provider, config=TrendSearchConfig(enabled=True))
        response = service.search(TrendSearchRequest(query_context="이슈"))
        assert response.items == []
        assert response.used_naver_search is False
        assert response.warnings

    def test_fail_open_on_unexpected_exception(self):
        provider = _FakeProvider(error=RuntimeError("예상치 못한 오류"))
        service = TrendSearchService(provider, config=TrendSearchConfig(enabled=True))
        response = service.search(TrendSearchRequest(query_context="이슈"))
        assert response.items == []

    def test_success_dedupes_and_respects_top_k(self):
        raw = [
            {"title": "이슈1", "source_url": "https://a", "publisher": "p1", "snippet": "", "published_at": "2026-07-20"},
            {"title": "이슈1", "source_url": "https://a", "publisher": "p1", "snippet": "", "published_at": "2026-07-20"},
            {"title": "이슈2", "source_url": "https://b", "publisher": "p2", "snippet": "", "published_at": "2026-07-21"},
            {"title": "이슈3", "source_url": "https://c", "publisher": "p3", "snippet": "", "published_at": "2026-07-22"},
            {"title": "", "source_url": "https://d", "publisher": "p4", "snippet": "", "published_at": None},
        ]
        provider = _FakeProvider(results=raw)
        service = TrendSearchService(provider, config=TrendSearchConfig(enabled=True, top_k=2))
        response = service.search(TrendSearchRequest(query_context="이슈", top_k=2))

        assert response.used_naver_search is True
        assert len(response.items) == 2  # top_k=2로 제한
        assert {i.title for i in response.items} == {"이슈1", "이슈2"}
        assert all(item.retrieved_at for item in response.items)  # 검색 시각이 채워진다.
        assert not response.warnings
