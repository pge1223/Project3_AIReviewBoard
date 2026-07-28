"""
Pydantic Schemas for Trend Search (실시간 이슈 검색)
==========================================================
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ai.rag.trend_search.exceptions import TrendSearchValidationError

MAX_QUERY_CONTEXT_LENGTH: int = 2000

InputType = Literal["issue", "interest", "technology", "rough_idea"]


class TrendSearchRequest(BaseModel):
    """TrendSearchService.search()의 입력."""

    query_context: str
    input_type: InputType = "issue"
    top_k: int = 5

    @field_validator("query_context")
    @classmethod
    def _query_context_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise TrendSearchValidationError("query_context는 빈 문자열일 수 없습니다")
        if len(v) > MAX_QUERY_CONTEXT_LENGTH:
            raise TrendSearchValidationError(
                f"query_context는 {MAX_QUERY_CONTEXT_LENGTH}자를 초과할 수 없습니다"
            )
        return v

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise TrendSearchValidationError("top_k는 1 이상이어야 합니다")
        return v


class TrendEvidenceItem(BaseModel):
    """검색·정리(dedup/필터링)까지 끝난 트렌드 근거 1건. 항상 참고 자료다 — 확정
    평가 근거나 공모전 심사 기준으로 쓰이지 않는다."""

    title: str
    snippet: str = ""
    source_url: str
    publisher: str
    published_at: Optional[str] = None
    retrieved_at: Optional[str] = None

    reference_only: bool = Field(default=True, frozen=True)


class TrendSearchResponse(BaseModel):
    """TrendSearchService.search()의 반환값."""

    items: list[TrendEvidenceItem] = Field(default_factory=list)
    query_text: str
    used_naver_search: bool = False
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "InputType",
    "MAX_QUERY_CONTEXT_LENGTH",
    "TrendSearchRequest",
    "TrendEvidenceItem",
    "TrendSearchResponse",
]
