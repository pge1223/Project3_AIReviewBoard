"""
Pydantic Schemas for External Market/Policy Research (RAG-007)
=====================================================================
"""

import math
import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ai.rag.external_research.exceptions import ExternalResearchValidationError

# 요청 스키마 단계의 절대 상한선(sanity check). 실제 서비스가 쓰는 조정 가능한 상한은
# ExternalResearchConfig.max_top_k(기본 20)이며, 서비스가 두 값 중 더 작은 쪽으로
# clamp한다 — 순환 import를 피하기 위해 이 파일은 config.py를 import하지 않는다.
MAX_TOP_K_CEILING: int = 100

MAX_QUERY_CONTEXT_LENGTH: int = 2000

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ExternalEvidenceType(str, Enum):
    """외부자료 유형. 실제 데이터가 없는 유형의 결과를 임의로 만들어내지 않는다."""

    NEWS = "news"
    STATISTICS = "statistics"
    MARKET = "market"
    POLICY = "policy"
    PUBLIC_DATA = "public_data"
    GUIDELINE = "guideline"
    LAW = "law"
    RESEARCH_REPORT = "research_report"


class FreshnessStatus(str, Enum):
    """자료 기준일 기반 최신성 상태."""

    CURRENT = "current"
    AGING = "aging"
    STALE = "stale"
    UNKNOWN = "unknown"


def _require_non_blank(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ExternalResearchValidationError(f"{field_name}는 빈 문자열일 수 없습니다")
    return value


def _validate_date_format(value: Optional[str], field_name: str) -> Optional[str]:
    """ISO 형식(YYYY-MM-DD)이 아니면 거부한다. None은 허용(날짜를 모르면 비워두는 것이지
    임의로 만들어내지 않는다는 원칙)."""
    if value is None:
        return None
    if not _DATE_RE.match(value):
        raise ExternalResearchValidationError(
            f"{field_name}는 YYYY-MM-DD 형식이어야 합니다: {value!r}"
        )
    return value


class ExternalEvidenceDocument(BaseModel):
    """색인 대상 외부자료 청크 1건. 출처가 없는 자료는 색인하지 않는다(출처 필드 필수)."""

    source_id: str
    document_id: str
    chunk_id: str

    title: str
    evidence_type: ExternalEvidenceType

    publisher: str
    source_url: str

    domain: str
    evaluation_criteria: list[str] = Field(default_factory=list)
    supported_roles: list[str] = Field(default_factory=list)

    content: str

    reference_date: Optional[str] = None
    published_at: Optional[str] = None
    retrieved_at: Optional[str] = None

    region: Optional[str] = None
    period: Optional[str] = None

    metric_name: Optional[str] = None
    metric_value: Optional[float | str] = None
    metric_unit: Optional[str] = None

    page: Optional[int] = None
    section: Optional[str] = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_id", "document_id", "chunk_id", "title", "publisher", "source_url", "domain", "content"
    )
    @classmethod
    def _not_blank(cls, v: str, info) -> str:
        return _require_non_blank(v, info.field_name)

    @field_validator("reference_date")
    @classmethod
    def _reference_date_format(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_format(v, "reference_date")

    @field_validator("published_at")
    @classmethod
    def _published_at_format(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_format(v, "published_at")

    @field_validator("retrieved_at")
    @classmethod
    def _retrieved_at_format(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_format(v, "retrieved_at")


class ExternalResearchRequest(BaseModel):
    """ExternalResearchService.search()의 입력."""

    domain: str
    evaluation_criteria: list[str]
    reviewer_role: str

    query_context: Optional[str] = None
    region: Optional[str] = None
    reference_date: Optional[str] = None

    evidence_types: Optional[list[ExternalEvidenceType]] = None

    top_k: int = 5
    min_score: Optional[float] = None

    trace_id: Optional[str] = None

    @field_validator("domain", "reviewer_role")
    @classmethod
    def _not_blank(cls, v: str, info) -> str:
        return _require_non_blank(v, info.field_name)

    @field_validator("evaluation_criteria")
    @classmethod
    def _criteria_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ExternalResearchValidationError("evaluation_criteria는 비어 있을 수 없습니다")
        for item in v:
            if not item or not item.strip():
                raise ExternalResearchValidationError("evaluation_criteria 항목은 빈 문자열일 수 없습니다")
        return v

    @field_validator("reference_date")
    @classmethod
    def _reference_date_format(cls, v: Optional[str]) -> Optional[str]:
        return _validate_date_format(v, "reference_date")

    @field_validator("query_context")
    @classmethod
    def _query_context_length(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > MAX_QUERY_CONTEXT_LENGTH:
            raise ExternalResearchValidationError(
                f"query_context는 {MAX_QUERY_CONTEXT_LENGTH}자를 초과할 수 없습니다"
            )
        return v

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ExternalResearchValidationError("top_k는 1 이상이어야 합니다")
        if v > MAX_TOP_K_CEILING:
            raise ExternalResearchValidationError(f"top_k는 {MAX_TOP_K_CEILING}를 초과할 수 없습니다")
        return v

    @field_validator("min_score")
    @classmethod
    def _min_score_finite(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if math.isnan(v) or math.isinf(v):
            raise ExternalResearchValidationError("min_score는 NaN 또는 무한대일 수 없습니다")
        return v


class ExternalEvidenceResult(BaseModel):
    """유사도/역할/평가기준/최신성 점수가 모두 반영된 최종 외부자료 검색 결과 1건."""

    source_id: str
    document_id: str
    chunk_id: str

    title: str
    evidence_type: ExternalEvidenceType

    publisher: str
    source_url: str

    domain: str
    supported_roles: list[str] = Field(default_factory=list)
    matched_criteria: list[str] = Field(default_factory=list)

    quote: str

    reference_date: Optional[str] = None
    published_at: Optional[str] = None
    retrieved_at: Optional[str] = None

    date_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    region: Optional[str] = None
    period: Optional[str] = None

    metric_name: Optional[str] = None
    metric_value: Optional[float | str] = None
    metric_unit: Optional[str] = None

    page: Optional[int] = None
    section: Optional[str] = None

    semantic_score: float
    role_score: float
    criteria_score: float
    freshness_score: float
    final_score: float

    retrieval_source: str

    # 외부자료는 항상 참고 자료이며 현재 문서의 직접 평가 근거가 아니다(RAG-005 근거
    # 충족도/숫자 점수 허용 정책과 무관함).
    reference_only: bool = True


class ExternalResearchResponse(BaseModel):
    """ExternalResearchService.search()의 반환값."""

    results: list[ExternalEvidenceResult] = Field(default_factory=list)

    total_results: int = 0
    query_text: str
    reviewer_role: str

    used_dataset_search: bool = False
    used_public_api_search: bool = False

    trace_id: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)

    reference_only: bool = True

    @model_validator(mode="after")
    def _total_results_matches(self) -> "ExternalResearchResponse":
        if self.total_results != len(self.results):
            self.total_results = len(self.results)
        return self


__all__ = [
    "ExternalEvidenceType",
    "FreshnessStatus",
    "ExternalEvidenceDocument",
    "ExternalResearchRequest",
    "ExternalEvidenceResult",
    "ExternalResearchResponse",
    "MAX_TOP_K_CEILING",
    "MAX_QUERY_CONTEXT_LENGTH",
]
