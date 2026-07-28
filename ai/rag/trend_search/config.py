"""
Trend Search Configuration (실시간 이슈 검색)
======================================================
네이버 검색 API 활성화 여부·타임아웃·세션당 검색 횟수 상한을 서비스 코드에
하드코딩하지 않고 여기서만 관리한다. ai/rag/external_research/config.py와 동일하게
pydantic-settings 없이 os.environ을 직접 읽는 이 프로젝트의 ai/rag 스타일을 따른다.

지원 환경변수:
    RAG_TREND_ENABLE_NAVER_SEARCH       (기본 false — 인증키가 없으면 어차피 비활성)
    RAG_TREND_TOP_K                     (기본 5)
    RAG_TREND_TIMEOUT_SECONDS           (기본 5.0)
    RAG_TREND_MAX_QUERIES_PER_SESSION   (기본 1 — 세션당 검색 폭주 방지)
"""

import os

from pydantic import BaseModel, Field

DEFAULT_TOP_K: int = 5
DEFAULT_TIMEOUT_SECONDS: float = 5.0
DEFAULT_MAX_QUERIES_PER_SESSION: int = 1


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


class TrendSearchConfig(BaseModel):
    """실시간 이슈 검색 실행 설정."""

    enabled: bool = Field(default_factory=lambda: _env_bool("RAG_TREND_ENABLE_NAVER_SEARCH", False))
    top_k: int = Field(default_factory=lambda: _env_int("RAG_TREND_TOP_K", DEFAULT_TOP_K), ge=1)
    timeout_seconds: float = Field(
        default_factory=lambda: _env_float("RAG_TREND_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS), gt=0
    )
    max_queries_per_session: int = Field(
        default_factory=lambda: _env_int(
            "RAG_TREND_MAX_QUERIES_PER_SESSION", DEFAULT_MAX_QUERIES_PER_SESSION
        ),
        ge=1,
    )


__all__ = [
    "TrendSearchConfig",
    "DEFAULT_TOP_K",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_QUERIES_PER_SESSION",
]
