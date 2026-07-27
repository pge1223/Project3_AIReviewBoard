"""NAVER API HUB 뉴스 검색 transport.

NAVER 응답을 ``PublicApiProvider``가 받는 중립 dict 형식으로 변환한다. 인증 정보는
호출자가 주입하며 로그나 예외 메시지에 포함하지 않는다.
"""

from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import requests

from ai.rag.external_research.exceptions import (
    ExternalProviderTimeoutError,
    ExternalProviderUnavailableError,
)
from ai.rag.external_research.schemas import ExternalResearchRequest

NAVER_NEWS_API_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _plain_text(value: Any) -> str:
    return _HTML_TAG_RE.sub("", html.unescape(str(value or ""))).strip()


def _iso_date(value: Any) -> str | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed.date().isoformat()


def _publisher_from_url(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname.removeprefix("www.")


class NaverNewsFetcher:
    """NCP NAVER API HUB 뉴스 검색을 수행하는 ``PublicApiFetchFn`` 구현."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        timeout_seconds: float = 5.0,
        display: int = 10,
        sort: str = "date",
    ):
        if not client_id.strip() or not client_secret.strip():
            raise ValueError("NAVER API HUB Client ID/Secret이 필요합니다.")
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout_seconds = timeout_seconds
        self._display = max(1, min(display, 100))
        self._sort = sort if sort in {"sim", "date"} else "date"

    def __call__(
        self,
        request: ExternalResearchRequest,
        query_text: str,
    ) -> list[dict]:
        headers = {
            "X-NCP-APIGW-API-KEY-ID": self._client_id,
            "X-NCP-APIGW-API-KEY": self._client_secret,
        }
        params = {
            "query": query_text,
            "display": self._display,
            "start": 1,
            "sort": self._sort,
            "format": "json",
        }
        try:
            response = requests.get(
                NAVER_NEWS_API_URL,
                headers=headers,
                params=params,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise ExternalProviderTimeoutError(
                f"NAVER 뉴스 검색이 {self._timeout_seconds}초를 초과했습니다."
            ) from exc
        except (requests.RequestException, ValueError) as exc:
            raise ExternalProviderUnavailableError(
                f"NAVER 뉴스 검색을 사용할 수 없습니다: {type(exc).__name__}"
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("items", []), list):
            raise ExternalProviderUnavailableError("NAVER 뉴스 검색 응답 형식이 올바르지 않습니다.")

        retrieved_at = datetime.now(timezone.utc).date().isoformat()
        results: list[dict] = []
        for item in payload.get("items", []):
            if not isinstance(item, dict):
                continue
            source_url = str(item.get("originallink") or item.get("link") or "").strip()
            publisher = _publisher_from_url(source_url)
            title = _plain_text(item.get("title"))
            content = _plain_text(item.get("description")) or title
            if not source_url or not publisher or not title or not content:
                continue

            digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:20]
            published_at = _iso_date(item.get("pubDate"))
            results.append(
                {
                    "source_id": f"NAVER-NEWS-{digest}",
                    "document_id": f"NAVER-NEWS-{digest}",
                    "chunk_id": f"NAVER-NEWS-{digest}-01",
                    "title": title,
                    "evidence_type": "news",
                    "publisher": publisher,
                    "source_url": source_url,
                    "domain": request.domain,
                    "evaluation_criteria": [],
                    "supported_roles": [],
                    "content": content,
                    "published_at": published_at,
                    "retrieved_at": retrieved_at,
                    "semantic_score": None,
                    "provider": "naver_api_hub",
                    "naver_link": item.get("link"),
                }
            )
        return results


__all__ = ["NAVER_NEWS_API_URL", "NaverNewsFetcher"]
