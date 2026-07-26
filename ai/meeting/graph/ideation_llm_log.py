"""로컬 디버깅용 LLM 원본 응답 로그.

ideation_trace.py의 trace_event는 개인정보 보호를 위해 내용을 요약/마스킹해 파이썬
logging 핸들러(대개 콘솔)로만 내보낸다 — 검증에 실패한 LLM 원본 응답이 정확히 무엇을
반환했는지는 그 로그만으로 재현할 수 없다. 이 모듈은 그와 별개로, 검증 성공/실패
여부와 무관하게 LLM이 실제로 반환한 원문 전체를 로컬 logs/ 디렉터리의 txt 파일에
그대로 append한다 — 기본은 꺼져 있고(ENABLE_IDEATION_LLM_RESPONSE_LOG=1로 켠다),
버그 재현을 위해 원본 응답을 직접 봐야 할 때만 켜서 쓴다. logs/는 레포 .gitignore
대상이라 커밋되지 않는다.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from .ideation_trace import current_session_id

_LOCK = threading.Lock()
_LOG_DIR_OVERRIDE: str | None = None
_ENABLED_OVERRIDE: bool | None = None

_DEFAULT_LOG_DIR = "logs/ideation_llm_calls"


def _enabled() -> bool:
    if _ENABLED_OVERRIDE is not None:
        return _ENABLED_OVERRIDE
    value = os.getenv("ENABLE_IDEATION_LLM_RESPONSE_LOG")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _log_dir() -> Path:
    return Path(_LOG_DIR_OVERRIDE or os.getenv("IDEATION_LLM_RESPONSE_LOG_DIR") or _DEFAULT_LOG_DIR)


def configure_ideation_llm_log(*, enabled: bool | None = None, log_dir: str | None = None) -> None:
    """설정값을 주입한다(테스트/앱 부팅 시 사용). None을 넘기면 환경변수 기본값으로 되돌린다."""
    global _ENABLED_OVERRIDE, _LOG_DIR_OVERRIDE
    _ENABLED_OVERRIDE = enabled
    _LOG_DIR_OVERRIDE = log_dir


def log_llm_response(
    *,
    node_name: str,
    attempt: int,
    ok: bool,
    reason: str | None,
    raw_response: str,
) -> None:
    """LLM 원본 응답 1건을 날짜별 txt 파일에 append한다.

    검증 실패(reason이 있는 경우)도 포함해 전부 남긴다 — 성공한 호출만 남기면 정작
    디버깅에 필요한 "왜 실패했는가"를 볼 수 없기 때문이다. 파일 쓰기 자체가 실패해도
    (디스크 권한 등) 회의 진행에는 영향을 주지 않는다."""
    if not _enabled():
        return
    now = datetime.now(timezone.utc)
    session_id = current_session_id() or "-"
    status = "OK" if ok else f"FAIL:{reason or 'unknown'}"
    header = (
        f"[{now:%Y-%m-%d %H:%M:%S.%f}] session={session_id} node={node_name} "
        f"attempt={attempt} status={status}"
    )
    try:
        log_dir = _log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / f"{now:%Y-%m-%d}.txt"
        with _LOCK:
            with file_path.open("a", encoding="utf-8") as fh:
                fh.write(header + "\n")
                fh.write((raw_response or "").rstrip("\n") + "\n")
                fh.write("-" * 80 + "\n")
    except OSError:
        pass


__all__ = ["configure_ideation_llm_log", "log_llm_response"]
