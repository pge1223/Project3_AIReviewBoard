"""개발 환경용 아이디어 회의 추적 로그.

사용자 발언과 위원 발언은 민감정보를 포함할 수 있으므로 기본적으로 꺼져 있다. 로그에는
검증된 화면용 문장과 결정적인 라우팅 사유만 제한 길이로 남기며 프롬프트/RAG 원문/LLM 원본
응답은 받지 않는다.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger("ai.meeting.ideation_trace")

_SESSION_ID: ContextVar[str | None] = ContextVar("ideation_trace_session_id", default=None)
_REQUEST_ID: ContextVar[str | None] = ContextVar("ideation_trace_request_id", default=None)

# backend/.env is loaded by pydantic-settings and is not guaranteed to be copied
# into os.environ.  The API layer therefore supplies the resolved settings once
# at import time.  Keeping the fallback makes this module usable in graph-only
# tests and command-line tools that configure it through environment variables.
_TRACE_ENABLED_OVERRIDE: bool | None = None
_CONTENT_LIMIT_OVERRIDE: int | None = None
_STREAM_DELTAS_OVERRIDE: bool | None = None

_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_RRN_RE = re.compile(r"\b\d{6}\s*[- ]?\s*[1-4]\d{6}\b")
_PHONE_RE = re.compile(r"(?<!\d)(01[016789]|0\d{1,2})[- ]?\d{3,4}[- ]?\d{4}(?!\d)")
_SECRET_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]{12,})\b", re.IGNORECASE)


def _enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def trace_enabled() -> bool:
    if _TRACE_ENABLED_OVERRIDE is not None:
        return _TRACE_ENABLED_OVERRIDE
    return _enabled("ENABLE_IDEATION_TRACE_LOGS")


def stream_delta_trace_enabled() -> bool:
    enabled = (
        _STREAM_DELTAS_OVERRIDE
        if _STREAM_DELTAS_OVERRIDE is not None
        else _enabled("IDEATION_TRACE_STREAM_DELTAS")
    )
    return trace_enabled() and enabled


def _content_limit() -> int:
    if _CONTENT_LIMIT_OVERRIDE is not None:
        return _CONTENT_LIMIT_OVERRIDE
    try:
        return max(0, min(int(os.getenv("IDEATION_TRACE_CONTENT_MAX_CHARS", "500")), 2000))
    except ValueError:
        return 500


def configure_ideation_trace(
    *,
    enabled: bool | None = None,
    content_max_chars: int | None = None,
    stream_deltas: bool | None = None,
) -> None:
    """Apply resolved application settings; passing ``None`` restores env fallback."""
    global _TRACE_ENABLED_OVERRIDE, _CONTENT_LIMIT_OVERRIDE, _STREAM_DELTAS_OVERRIDE
    _TRACE_ENABLED_OVERRIDE = enabled
    _CONTENT_LIMIT_OVERRIDE = (
        None if content_max_chars is None else max(0, min(int(content_max_chars), 2000))
    )
    _STREAM_DELTAS_OVERRIDE = stream_deltas


def sanitize_preview(value: Any, *, limit: int | None = None) -> str:
    """제어문자와 대표 개인정보/비밀값을 마스킹한 한 줄 preview."""
    text = "" if value is None else str(value)
    text = _SECRET_RE.sub("[SECRET]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _RRN_RE.sub("[RRN]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = " ".join(text.split())
    max_chars = _content_limit() if limit is None else max(0, min(limit, 2000))
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def bind_trace_context(session_id: str | None, request_id: str | None = None) -> tuple[Token, Token]:
    return _SESSION_ID.set(session_id), _REQUEST_ID.set(request_id)


def reset_trace_context(tokens: tuple[Token, Token]) -> None:
    session_token, request_token = tokens
    _SESSION_ID.reset(session_token)
    _REQUEST_ID.reset(request_token)


def current_session_id() -> str | None:
    """bind_trace_context가 이번 요청에 바인딩해 둔 session_id(없으면 None).

    가은/Claude(2026-07-27) — ideation_llm_log가 _safe_call_json/
    _safe_call_structured_json 호출부마다 session_id를 새로 인자로 받지 않고도
    로그 파일에 세션을 식별해 남길 수 있도록 노출한다."""
    return _SESSION_ID.get()


def _safe_log_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_preview(value)
    if isinstance(value, (list, tuple)):
        return [_safe_log_value(item) for item in value]
    if isinstance(value, dict):
        return {sanitize_preview(key, limit=80): _safe_log_value(item) for key, item in value.items()}
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return sanitize_preview(value)


def trace_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """고정 이벤트명과 key=value 필드만 기록한다. 자유 형식 추론은 받지 않는다."""
    if not trace_enabled():
        return
    payload: dict[str, Any] = {}
    session_id = fields.pop("session_id", None) or _SESSION_ID.get()
    request_id = fields.pop("request_id", None) or _REQUEST_ID.get()
    if session_id:
        payload["session"] = session_id
    if request_id:
        payload["request"] = request_id
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = _safe_log_value(value)
    rendered = " ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for key, value in payload.items()
    )
    logger.log(level, "[%s]%s%s", event, " " if rendered else "", rendered)


def is_late_request_event(event_request_id: str | None, active_request_id: str | None) -> bool:
    """Return true and warn when an event belongs to an older request."""
    if not event_request_id or not active_request_id or event_request_id == active_request_id:
        return False
    trace_event(
        "IDEATION_LATE_REQUEST_EVENT",
        level=logging.WARNING,
        event_request_id=event_request_id,
        active_request_id=active_request_id,
    )
    return True


__all__ = [
    "bind_trace_context",
    "configure_ideation_trace",
    "current_session_id",
    "is_late_request_event",
    "reset_trace_context",
    "sanitize_preview",
    "stream_delta_trace_enabled",
    "trace_enabled",
    "trace_event",
]
