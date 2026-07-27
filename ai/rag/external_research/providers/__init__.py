"""
External Research Providers (RAG-007)
============================================
검색 소스를 provider로 분리한다. DatasetProvider(1순위, 사전 수집 Chroma 데이터셋)와
PublicApiProvider(선택적 실시간 API), NAVER API HUB 뉴스 transport를 포함한다.
"""

from ai.rag.external_research.providers.base import ExternalEvidenceCandidate, ExternalResearchProvider
from ai.rag.external_research.providers.dataset_provider import DatasetProvider
from ai.rag.external_research.providers.naver_news import NAVER_NEWS_API_URL, NaverNewsFetcher
from ai.rag.external_research.providers.public_api_provider import PublicApiFetchFn, PublicApiProvider

__all__ = [
    "ExternalEvidenceCandidate",
    "ExternalResearchProvider",
    "DatasetProvider",
    "NAVER_NEWS_API_URL",
    "NaverNewsFetcher",
    "PublicApiProvider",
    "PublicApiFetchFn",
]
