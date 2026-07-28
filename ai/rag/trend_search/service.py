"""
Trend Search Service (실시간 이슈 검색)
============================================
네이버 검색 API(NaverSearchProvider)로 최신 이슈를 검색하고, 중복·출처 없는 결과를
정리해 상위 K건을 반환한다. ai/rag/external_research/search_service.py
(ExternalResearchService, RAG-007)와는 완전히 분리된 별도 서비스다 — 실시간 뉴스는
의미 유사도/신선도 점수를 계산하지 않는다(검색 자체가 이미 최신순).

LangGraph나 ai.meeting.graph에 의존하지 않으며 단독으로 생성/호출할 수 있다.
provider 호출이 실패해도(타임아웃/인증 오류 등) 예외를 밖으로 던지지 않고 빈
결과 + warning으로 반환한다 — 아이디어 발굴 회의 자체를 막지 않기 위해서다
(ExternalResearchService._call_public_api_provider와 동일한 fail-open 원칙).
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from ai.rag.trend_search.config import TrendSearchConfig
from ai.rag.trend_search.dedup import dedupe_by_title, dedupe_by_url, filter_has_source
from ai.rag.trend_search.exceptions import TrendSearchError
from ai.rag.trend_search.naver_provider import NaverSearchProvider
from ai.rag.trend_search.schemas import TrendEvidenceItem, TrendSearchRequest, TrendSearchResponse

logger = logging.getLogger(__name__)


class TrendSearchService:
    """실시간 이슈 검색 서비스. provider를 생성자로 주입받으며, 새 HTTP 클라이언트나
    인증키를 내부에서 만들지 않는다."""

    def __init__(self, provider: Optional[NaverSearchProvider] = None, *, config: Optional[TrendSearchConfig] = None):
        self._provider = provider
        self._config = config or TrendSearchConfig()

    def search(self, request: TrendSearchRequest) -> TrendSearchResponse:
        start = time.monotonic()
        query_text = request.query_context.strip()
        top_k = min(request.top_k, self._config.top_k)

        logger.info(
            "[TREND_SEARCH_START] input_type=%s query_len=%d top_k=%d",
            request.input_type,
            len(query_text),
            top_k,
        )

        provider_ready = self._provider is not None and self._provider.enabled
        if not self._config.enabled or not provider_ready:
            logger.info(
                "[TREND_SEARCH_SKIPPED] config_enabled=%s provider_ready=%s",
                self._config.enabled,
                provider_ready,
            )
            return TrendSearchResponse(
                items=[],
                query_text=query_text,
                used_naver_search=False,
                warnings=["실시간 이슈 검색이 비활성화되어 있어 트렌드 근거 없이 진행합니다."],
            )

        try:
            raw_items = self._provider.search(query_text, display=max(top_k * 4, 20))
        except TrendSearchError as exc:
            # pge/Claude(2026-07-27, 실측: 실제 키를 넣어도 실패 원인을 알 수 없었음) —
            # 기존에는 예외 클래스 이름만 남겨서 "인증키 없음"/"HTTP 401"/"네트워크 오류"를
            # 구분할 수 없었다. naver_provider.py가 raise 시점에 상태 코드·오류 종류를 이미
            # str(exc)에 담아두므로 그대로 로그에 남긴다.
            logger.warning(
                "[TREND_SEARCH_FAILED] provider_name=%s error_code=%s detail=%s",
                self._provider.name,
                type(exc).__name__,
                str(exc),
            )
            return TrendSearchResponse(
                items=[],
                query_text=query_text,
                used_naver_search=False,
                warnings=["실시간 이슈 검색에 실패해 트렌드 근거 없이 진행합니다."],
            )
        except Exception as exc:  # noqa: BLE001 - fail-open이 목적, 회의 흐름을 막지 않는다.
            logger.warning(
                "[TREND_SEARCH_UNEXPECTED_ERROR] provider_name=%s error_code=%s detail=%s",
                self._provider.name,
                type(exc).__name__,
                str(exc),
            )
            return TrendSearchResponse(
                items=[],
                query_text=query_text,
                used_naver_search=False,
                warnings=["실시간 이슈 검색 중 오류가 발생해 트렌드 근거 없이 진행합니다."],
            )

        filtered = filter_has_source(raw_items)
        deduped = dedupe_by_title(dedupe_by_url(filtered))
        top_items = deduped[:top_k]

        retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        items = [TrendEvidenceItem(**item, retrieved_at=retrieved_at) for item in top_items]

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "[TREND_SEARCH_COMPLETE] raw_count=%d result_count=%d duration_ms=%d",
            len(raw_items),
            len(items),
            duration_ms,
        )
        return TrendSearchResponse(items=items, query_text=query_text, used_naver_search=True, warnings=[])


__all__ = ["TrendSearchService"]
