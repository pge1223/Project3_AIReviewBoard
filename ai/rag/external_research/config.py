"""
External Research Configuration (RAG-007)
================================================
컬렉션 이름, 검색 임계값, 랭킹 가중치, 최신성 기준, 실시간 API 활성화 여부를
서비스 코드에 하드코딩하지 않고 여기서만 관리한다. ai/rag의 기존 config.py들
(ai.rag.converters.config, ai.rag.similar_cases.config)과 동일하게 pydantic-settings
없이 os.environ을 직접 읽는 이 프로젝트의 ai/rag 스타일을 따른다.

지원 환경변수:
    RAG_EXTERNAL_COLLECTION            (기본 "external_market_policy_evidence")
    RAG_EXTERNAL_TOP_K                 (기본 5)
    RAG_EXTERNAL_MIN_SCORE             (기본 0.45)
    RAG_EXTERNAL_ENABLE_DATASET        (기본 true)
    RAG_EXTERNAL_ENABLE_PUBLIC_API     (기본 false — 키가 있어도 명시적으로 켜야 활성화)
    RAG_EXTERNAL_DOMAIN_FALLBACK       (기본 true)
    RAG_EXTERNAL_NEWS_FRESHNESS_WEIGHT (기본 0.30)
"""

import os
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from ai.rag.external_research.schemas import ExternalEvidenceType

DEFAULT_COLLECTION_NAME: str = "external_market_policy_evidence"
DEFAULT_TOP_K: int = 5
DEFAULT_MAX_TOP_K: int = 20
DEFAULT_MIN_SCORE: float = 0.45

DEFAULT_SEMANTIC_WEIGHT: float = 0.55
DEFAULT_ROLE_WEIGHT: float = 0.20
DEFAULT_CRITERIA_WEIGHT: float = 0.15
DEFAULT_FRESHNESS_WEIGHT: float = 0.10
DEFAULT_NEWS_FRESHNESS_WEIGHT: float = 0.30

DEFAULT_MAX_EVIDENCE_PER_SOURCE: int = 3
DEFAULT_CANDIDATE_K_MULTIPLIER: int = 4

# 사전 수집 데이터 우선 원칙(섹션 5) — 실시간 검색은 기본 비활성화.
DEFAULT_ENABLE_DATASET_SEARCH: bool = True
DEFAULT_ENABLE_PUBLIC_API_SEARCH: bool = False


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_str(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value


class ExternalResearchConfig(BaseModel):
    """외부자료 색인/검색/랭킹 실행 설정."""

    collection_name: str = Field(
        default_factory=lambda: _env_str("RAG_EXTERNAL_COLLECTION") or DEFAULT_COLLECTION_NAME
    )
    default_top_k: int = Field(default_factory=lambda: _env_int("RAG_EXTERNAL_TOP_K", DEFAULT_TOP_K), ge=1)
    max_top_k: int = Field(default=DEFAULT_MAX_TOP_K, ge=1)
    min_similarity_score: float = Field(
        default_factory=lambda: _env_float("RAG_EXTERNAL_MIN_SCORE", DEFAULT_MIN_SCORE)
    )

    enable_dataset_search: bool = Field(
        default_factory=lambda: _env_bool("RAG_EXTERNAL_ENABLE_DATASET", DEFAULT_ENABLE_DATASET_SEARCH)
    )
    enable_public_api_search: bool = Field(
        default_factory=lambda: _env_bool("RAG_EXTERNAL_ENABLE_PUBLIC_API", DEFAULT_ENABLE_PUBLIC_API_SEARCH)
    )

    domain_filter_fallback_to_all: bool = Field(
        default_factory=lambda: _env_bool("RAG_EXTERNAL_DOMAIN_FALLBACK", True)
    )

    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT
    role_weight: float = DEFAULT_ROLE_WEIGHT
    criteria_weight: float = DEFAULT_CRITERIA_WEIGHT
    freshness_weight: float = DEFAULT_FRESHNESS_WEIGHT
    # 뉴스는 시의성이 핵심이므로 별도 최신성 비중을 사용한다. 나머지 세 가중치는
    # search_service에서 기존 비율을 유지한 채 합계가 1이 되도록 재조정한다.
    news_freshness_weight: float = Field(
        default_factory=lambda: _env_float(
            "RAG_EXTERNAL_NEWS_FRESHNESS_WEIGHT", DEFAULT_NEWS_FRESHNESS_WEIGHT
        ),
        ge=0.0,
        le=1.0,
    )

    max_evidence_per_source: int = Field(default=DEFAULT_MAX_EVIDENCE_PER_SOURCE, ge=1)
    candidate_k_multiplier: int = Field(default=DEFAULT_CANDIDATE_K_MULTIPLIER, ge=1)

    # 외부자료는 항상 참고 자료다 — RAG-005 근거 충족도/숫자 점수 허용 정책에 영향을 주지
    # 않는다는 것을 명시하는 상수 플래그(문서화 목적, 변경 불가).
    reference_only: bool = Field(default=True, frozen=True)

    @model_validator(mode="after")
    def _top_k_within_max(self) -> "ExternalResearchConfig":
        if self.default_top_k > self.max_top_k:
            raise ValueError(f"default_top_k({self.default_top_k})는 max_top_k({self.max_top_k})를 초과할 수 없습니다")
        return self


# 자료 유형별 최신성 기준(일). 발행 후 이 기간 이내면 CURRENT, 1.5배 이내면 AGING, 그
# 이상이면 STALE로 판정한다(freshness.py 참고). LAW는 "최신 개정일 기준"이라는 요구사항을
# 정확히 만족하려면 법령 전용 개정 이력 데이터가 필요하지만, 이 프로젝트는 그런 데이터를
# 갖고 있지 않으므로 reference_date(자료가 표시하는 기준일 = 통상 최신 개정일)를 그대로
# 사용하고 GUIDELINE과 동일한 임계값을 적용한다 — README 한계 항목에 명시.
DEFAULT_FRESHNESS_THRESHOLD_DAYS: dict[ExternalEvidenceType, int] = {
    ExternalEvidenceType.NEWS: 30,
    ExternalEvidenceType.STATISTICS: 3 * 365,
    ExternalEvidenceType.MARKET: 2 * 365,
    ExternalEvidenceType.POLICY: 2 * 365,
    ExternalEvidenceType.LAW: 2 * 365,
    ExternalEvidenceType.GUIDELINE: 3 * 365,
    ExternalEvidenceType.PUBLIC_DATA: 3 * 365,
    ExternalEvidenceType.RESEARCH_REPORT: 2 * 365,
}


class FreshnessConfig(BaseModel):
    """자료 유형별 최신성 판정 기준(일 단위). 코드에 하드코딩하지 않고 이 설정으로 관리한다."""

    threshold_days: dict[ExternalEvidenceType, int] = Field(
        default_factory=lambda: dict(DEFAULT_FRESHNESS_THRESHOLD_DAYS)
    )
    default_threshold_days: int = 2 * 365

    def threshold_for(self, evidence_type: ExternalEvidenceType) -> int:
        return self.threshold_days.get(evidence_type, self.default_threshold_days)


class PublicApiProviderConfig(BaseModel):
    """실시간 공공데이터 API provider 실행 설정. 실제 엔드포인트/키는 여기서 관리하지
    않는다 — 그 값은 호출자가 주입하는 transport 구현체 책임이다(섹션 25)."""

    timeout_seconds: float = Field(default=5.0, gt=0)
    max_results: int = Field(default=10, ge=1)


__all__ = [
    "ExternalResearchConfig",
    "FreshnessConfig",
    "PublicApiProviderConfig",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_TOP_K",
    "DEFAULT_MAX_TOP_K",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_SEMANTIC_WEIGHT",
    "DEFAULT_ROLE_WEIGHT",
    "DEFAULT_CRITERIA_WEIGHT",
    "DEFAULT_FRESHNESS_WEIGHT",
    "DEFAULT_NEWS_FRESHNESS_WEIGHT",
    "DEFAULT_MAX_EVIDENCE_PER_SOURCE",
    "DEFAULT_CANDIDATE_K_MULTIPLIER",
    "DEFAULT_ENABLE_DATASET_SEARCH",
    "DEFAULT_ENABLE_PUBLIC_API_SEARCH",
    "DEFAULT_FRESHNESS_THRESHOLD_DAYS",
]
