"""
Trend Search (실시간 이슈 검색)
=====================================
아이디어 발굴(candidate_planning) 직전에 네이버 검색 API로 "요즘 이슈"를 가져오는
독립 패키지. ai/rag/external_research(RAG-007, 사전 색인된 통계·정책 자료)와는
완전히 분리돼 있다 — 실시간 뉴스는 신뢰도·신선도 스코어링 대상이 아니라 참고
자료일 뿐이므로, RAG-007의 청크/랭킹 스키마를 재사용하지 않는다.
"""

from ai.rag.trend_search.config import TrendSearchConfig
from ai.rag.trend_search.naver_provider import NaverSearchProvider
from ai.rag.trend_search.schemas import TrendEvidenceItem, TrendSearchRequest, TrendSearchResponse
from ai.rag.trend_search.service import TrendSearchService

__all__ = [
    "TrendSearchConfig",
    "NaverSearchProvider",
    "TrendEvidenceItem",
    "TrendSearchRequest",
    "TrendSearchResponse",
    "TrendSearchService",
]
