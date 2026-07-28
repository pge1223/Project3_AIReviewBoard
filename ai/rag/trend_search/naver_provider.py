"""
Naver Search API Provider (실시간 이슈 검색)
==================================================
네이버 뉴스 검색으로 실시간 이슈를 가져오는 provider. Client ID/Secret은 생성자로
주입받는다 — 이 파일 자체는 실제 키 값을 알지 못한다(하드코딩 금지 원칙,
ai/rag/external_research/providers/public_api_provider.py와 동일한 철학).

pge/Claude(2026-07-28, 실측: 401/errorCode 024 "NID AUTH Result Invalid") — 팀(재인)
확인 결과 개발자센터(developers.naver.com) 검색 오픈API가 NCP(NAVER API HUB,
apigw.ntruss.com)로 완전히 이관됐다. URL/인증 헤더만 NCP API Hub 방식으로 바뀌고
(X-Naver-Client-Id/Secret -> X-NCP-APIGW-API-KEY-ID/KEY), 응답 스키마(items/title/
description/originallink/link/pubDate)는 기존과 동일해 _raw_item_to_dict 이하 파싱
로직은 그대로 둔다.
"""

import logging
import re
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import urlparse

import requests

from ai.rag.trend_search.exceptions import TrendProviderTimeoutError, TrendProviderUnavailableError

logger = logging.getLogger(__name__)

NAVER_NEWS_SEARCH_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"

# pge/Claude(2026-07-28, 실측: 431 Request Header Fields Too Large) — 호출부
# (ideation_conv_discovery.py::_trend_search_query)에서 이미 짧게 만들어 보내지만, 검색어가
# 길어질수록 뉴스 검색 결과 품질도 떨어지므로 provider 레벨에서도 방어적으로 자른다(호출부
# 버그가 다시 생겨도 여기서 막힌다).
_MAX_QUERY_CHARS = 200


def _strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").replace("&quot;", '"').replace("&amp;", "&")


def _publisher_from_link(url: str) -> str:
    try:
        return urlparse(url).netloc or "네이버 뉴스"
    except ValueError:
        return "네이버 뉴스"


def _parse_pubdate(pub_date: Optional[str]) -> Optional[str]:
    if not pub_date:
        return None
    try:
        return parsedate_to_datetime(pub_date).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _raw_item_to_dict(item: dict) -> dict:
    link = item.get("originallink") or item.get("link", "")
    return {
        "title": _strip_html_tags(item.get("title", "")),
        "snippet": _strip_html_tags(item.get("description", "")),
        "source_url": link,
        "publisher": _publisher_from_link(link),
        "published_at": _parse_pubdate(item.get("pubDate")),
    }


class NaverSearchProvider:
    """네이버 뉴스 검색(NCP API Hub) provider. 인증키가 없거나 enabled=False면 실제로
    아무것도 호출하지 않고 명확한 예외를 던진다 — 결과가 없는 것처럼 조용히
    비어있는 리스트를 반환하지 않는다(예외는 TrendSearchService가 잡아 fail-open
    처리한다)."""

    def __init__(
        self,
        *,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        enabled: bool = False,
        timeout_seconds: float = 5.0,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._enabled = enabled
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "naver_search"

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._client_id) and bool(self._client_secret)

    def search(self, query_text: str, *, display: int = 20) -> list[dict]:
        if not self.enabled:
            raise TrendProviderUnavailableError(
                "NaverSearchProvider가 비활성화되어 있거나 Client ID/Secret이 없습니다."
            )

        headers = {
            "X-NCP-APIGW-API-KEY-ID": self._client_id,
            "X-NCP-APIGW-API-KEY": self._client_secret,
        }
        params = {"query": query_text[:_MAX_QUERY_CHARS], "display": display, "sort": "date"}

        try:
            response = requests.get(
                NAVER_NEWS_SEARCH_URL, headers=headers, params=params, timeout=self._timeout_seconds
            )
        except requests.Timeout as exc:
            raise TrendProviderTimeoutError(
                f"네이버 검색 API 호출이 timeout_seconds={self._timeout_seconds}초를 초과했습니다."
            ) from exc
        except requests.RequestException as exc:
            raise TrendProviderUnavailableError(
                f"네이버 검색 API 호출 중 오류가 발생했습니다: {type(exc).__name__}"
            ) from exc

        if response.status_code != 200:
            raise TrendProviderUnavailableError(
                f"네이버 검색 API가 오류 상태를 반환했습니다: {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise TrendProviderUnavailableError("네이버 검색 API 응답을 파싱하지 못했습니다.") from exc

        raw_items = payload.get("items", [])
        return [_raw_item_to_dict(item) for item in raw_items if isinstance(item, dict)]


__all__ = ["NaverSearchProvider", "NAVER_NEWS_SEARCH_URL"]
