"""
External Market/Policy Research (RAG-007)
================================================
AI 위원의 역할·평가 기준에 맞춰 외부 공개 통계·시장·정책 자료를 검색한다.
RAG-003(현재 프로젝트 업로드 문서 검색), RAG-006(공개 성공 사례 검색)과는 완전히
분리된 독립 모듈이며, LangGraph나 ai.meeting.graph에 의존하지 않는다.

외부자료는 항상 참고 자료(reference_only=True)이며 현재 문서의 직접 평가 근거가
아니다. 자세한 내용은 README.md 참고.
"""

from ai.rag.external_research.config import (
    ExternalResearchConfig,
    FreshnessConfig,
    PublicApiProviderConfig,
)
from ai.rag.external_research.exceptions import (
    ExternalCollectionUnavailableError,
    ExternalEvidenceIndexingError,
    ExternalEvidenceSearchError,
    ExternalProviderTimeoutError,
    ExternalProviderUnavailableError,
    ExternalResearchError,
    ExternalResearchValidationError,
    ExternalSourceValidationError,
)
from ai.rag.external_research.freshness import compute_freshness
from ai.rag.external_research.indexing_service import (
    EmbedderLike,
    ExternalEvidenceIndexingService,
    ExternalEvidenceIndexingSummary,
)
from ai.rag.external_research.providers import (
    DatasetProvider,
    ExternalEvidenceCandidate,
    ExternalResearchProvider,
    NAVER_NEWS_API_URL,
    NaverNewsFetcher,
    PublicApiFetchFn,
    PublicApiProvider,
)
from ai.rag.external_research.query_builder import build_external_research_query, get_role_query_terms
from ai.rag.external_research.ranking import (
    compute_criteria_score,
    compute_final_score,
    compute_role_score,
)
from ai.rag.external_research.repository import ExternalEvidenceHit, ExternalEvidenceRepository, build_evidence_record_id
from ai.rag.external_research.schemas import (
    ExternalEvidenceDocument,
    ExternalEvidenceResult,
    ExternalEvidenceType,
    ExternalResearchRequest,
    ExternalResearchResponse,
    FreshnessStatus,
)
from ai.rag.external_research.search_service import ExternalResearchService
from ai.rag.external_research.source_validator import validate_source_metadata

__all__ = [
    "ExternalResearchConfig",
    "FreshnessConfig",
    "PublicApiProviderConfig",
    "ExternalResearchError",
    "ExternalResearchValidationError",
    "ExternalCollectionUnavailableError",
    "ExternalEvidenceIndexingError",
    "ExternalEvidenceSearchError",
    "ExternalProviderUnavailableError",
    "ExternalProviderTimeoutError",
    "ExternalSourceValidationError",
    "ExternalEvidenceType",
    "FreshnessStatus",
    "ExternalEvidenceDocument",
    "ExternalResearchRequest",
    "ExternalEvidenceResult",
    "ExternalResearchResponse",
    "ExternalEvidenceRepository",
    "ExternalEvidenceHit",
    "build_evidence_record_id",
    "EmbedderLike",
    "ExternalEvidenceIndexingService",
    "ExternalEvidenceIndexingSummary",
    "DatasetProvider",
    "NAVER_NEWS_API_URL",
    "NaverNewsFetcher",
    "PublicApiProvider",
    "PublicApiFetchFn",
    "ExternalEvidenceCandidate",
    "ExternalResearchProvider",
    "build_external_research_query",
    "get_role_query_terms",
    "compute_role_score",
    "compute_criteria_score",
    "compute_final_score",
    "compute_freshness",
    "validate_source_metadata",
    "ExternalResearchService",
]
