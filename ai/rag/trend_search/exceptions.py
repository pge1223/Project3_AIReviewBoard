"""
Custom Exceptions for Trend Search (실시간 이슈 검색)
==========================================================
"""

from typing import Optional


class TrendSearchError(Exception):
    """트렌드 검색 모듈 예외의 공통 베이스."""

    user_message: str = "실시간 이슈 검색을 처리하지 못했습니다."

    def __init__(self, message: str, *, user_message: Optional[str] = None):
        super().__init__(message)
        if user_message is not None:
            self.user_message = user_message


class TrendSearchValidationError(TrendSearchError):
    """검색 요청 데이터가 필수 조건을 만족하지 않는 경우."""


class TrendProviderUnavailableError(TrendSearchError):
    """provider가 비활성화됐거나 필요한 인증키가 주입되지 않은 경우, 혹은 호출 자체가
    실패한 경우."""


class TrendProviderTimeoutError(TrendSearchError):
    """provider 호출이 설정된 시간 안에 끝나지 않은 경우."""


__all__ = [
    "TrendSearchError",
    "TrendSearchValidationError",
    "TrendProviderUnavailableError",
    "TrendProviderTimeoutError",
]
