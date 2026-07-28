import requests
import requests_mock

from ai.rag.external_research.exceptions import ExternalProviderUnavailableError
from ai.rag.external_research.providers.naver_news import NAVER_NEWS_API_URL, NaverNewsFetcher
from ai.rag.external_research.schemas import ExternalResearchRequest


def _request() -> ExternalResearchRequest:
    return ExternalResearchRequest(
        domain="competition",
        evaluation_criteria=["시장성"],
        reviewer_role="planning",
    )


def test_naver_api_hub_request_and_response_normalization():
    fetch = NaverNewsFetcher(
        client_id="client-id",
        client_secret="client-secret",
        display=3,
    )
    payload = {
        "items": [
            {
                "title": "<b>AI</b> 공모전 소식",
                "originallink": "https://www.example.com/news/1",
                "link": "https://n.news.naver.com/article/1",
                "description": "공공서비스 <b>AI</b> 공모전이 열립니다.",
                "pubDate": "Mon, 27 Jul 2026 09:30:00 +0900",
            }
        ]
    }

    with requests_mock.Mocker() as mock:
        mock.get(NAVER_NEWS_API_URL, json=payload)
        results = fetch(_request(), "AI 공모전")

        sent = mock.last_request
        assert sent.headers["X-NCP-APIGW-API-KEY-ID"] == "client-id"
        assert sent.headers["X-NCP-APIGW-API-KEY"] == "client-secret"
        assert sent.qs["query"] == ["ai 공모전"]
        assert sent.qs["display"] == ["3"]
        assert sent.qs["sort"] == ["date"]

    assert len(results) == 1
    result = results[0]
    assert result["title"] == "AI 공모전 소식"
    assert result["content"] == "공공서비스 AI 공모전이 열립니다."
    assert result["publisher"] == "example.com"
    assert result["source_url"] == "https://www.example.com/news/1"
    assert result["published_at"] == "2026-07-27"
    assert result["evidence_type"] == "news"
    assert result["semantic_score"] is None


def test_naver_api_hub_error_does_not_expose_credentials():
    fetch = NaverNewsFetcher(client_id="client-id", client_secret="very-secret")
    with requests_mock.Mocker() as mock:
        mock.get(NAVER_NEWS_API_URL, status_code=401, json={"message": "unauthorized"})
        try:
            fetch(_request(), "AI 공모전")
        except ExternalProviderUnavailableError as exc:
            message = str(exc)
        else:
            raise AssertionError("401 응답은 provider 오류로 변환되어야 합니다.")

    assert "client-id" not in message
    assert "very-secret" not in message


def test_naver_api_hub_timeout_is_normalized():
    fetch = NaverNewsFetcher(client_id="client-id", client_secret="client-secret")
    with requests_mock.Mocker() as mock:
        mock.get(NAVER_NEWS_API_URL, exc=requests.Timeout())
        try:
            fetch(_request(), "AI 공모전")
        except Exception as exc:
            assert type(exc).__name__ == "ExternalProviderTimeoutError"
        else:
            raise AssertionError("timeout은 provider timeout 오류로 변환되어야 합니다.")
