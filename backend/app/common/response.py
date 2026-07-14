from typing import Any, Optional
from pydantic import BaseModel


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


def success_response(data: Any = None, message: str = "성공"):
    return ApiResponse(success=True, message=message, data=data)


def error_response(message: str = "실패"):
    return ApiResponse(success=False, message=message, data=None)