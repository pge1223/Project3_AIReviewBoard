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
from datetime import datetime
from pathlib import Path

from .ideation_trace import current_session_id

_LOCK = threading.Lock()
_LOG_DIR_OVERRIDE: str | None = None
_ENABLED_OVERRIDE: bool | None = None
# 가은/Claude(2026-07-27, 요청: "로그 이름 뒤에 시간도 붙여달라") — 날짜만으로는 하루 동안의
# 여러 서버 실행(재시작)이 한 파일에 계속 이어져 어느 실행에서 난 로그인지 구분하기 어렵다.
# 이 값은 프로세스에서 실제로 처음 로그를 쓰는 시점에 한 번만 계산해 캐시한다(매 호출마다
# 새 시각을 쓰면 호출마다 새 파일이 생겨 log 조각이 흩어진다) — ai/rag의 url_pdf_analyzer
# 로그가 이미 쓰는 "실행 1회당 파일 1개, 이름에 날짜+시간" 관례와 맞춘다.
_SESSION_LOG_FILENAME: str | None = None

# 가은/Claude(2026-07-27, 실측: "로그가 안 쌓이는 것 같다") — 상대경로 "logs/ideation_llm_calls"를
# 쓰면 백엔드 서버 프로세스의 실행 위치(cwd)에 따라 실제로는 backend/logs/ideation_llm_calls
# 처럼 엉뚱한 곳에 쌓인다(uvicorn을 backend/ 디렉터리에서 띄우는 경우). 이 파일
# (ai/meeting/graph/ideation_llm_log.py) 위치를 기준으로 레포 루트를 계산해 항상 같은
# 절대경로(<repo_root>/logs/ideation_llm_calls)에 쓰도록 고정한다 — 어디서 서버를
# 띄우든 사용자가 찾는 위치가 달라지지 않는다.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LOG_DIR = str(_REPO_ROOT / "logs" / "ideation_llm_calls")


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
    global _ENABLED_OVERRIDE, _LOG_DIR_OVERRIDE, _SESSION_LOG_FILENAME
    _ENABLED_OVERRIDE = enabled
    _LOG_DIR_OVERRIDE = log_dir
    # 서버가 재시작될 때마다(=이 함수가 다시 호출될 때마다) 새 파일명을 다시 계산하게
    # 리셋한다 — 그래야 재시작 전/후 로그가 같은 파일에 섞이지 않는다.
    _SESSION_LOG_FILENAME = None


def _log_file_name() -> str:
    global _SESSION_LOG_FILENAME
    if _SESSION_LOG_FILENAME is None:
        _SESSION_LOG_FILENAME = f"{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    return _SESSION_LOG_FILENAME


def log_llm_response(
    *,
    node_name: str,
    attempt: int,
    ok: bool,
    reason: str | None,
    raw_response: str,
) -> None:
    """LLM 원본 응답 1건을 서버 실행(프로세스)별 txt 파일에 append한다(파일명은 이 프로세스가
    처음 로그를 쓴 시각의 날짜+시간).

    검증 실패(reason이 있는 경우)도 포함해 전부 남긴다 — 성공한 호출만 남기면 정작
    디버깅에 필요한 "왜 실패했는가"를 볼 수 없기 때문이다. 파일 쓰기 자체가 실패해도
    (디스크 권한 등) 회의 진행에는 영향을 주지 않는다."""
    if not _enabled():
        return
    # 가은/Claude(2026-07-27) — 로컬 디버깅 로그라 UTC 대신 로컬 시각을 쓴다. UTC를 쓰면
    # 한국 시간 기준 자정 근처에서 파일 날짜가 하루 어긋나 "오늘 로그가 안 보인다"는
    # 혼동을 만든다.
    now = datetime.now()
    session_id = current_session_id() or "-"
    status = "OK" if ok else f"FAIL:{reason or 'unknown'}"
    header = (
        f"[{now:%Y-%m-%d %H:%M:%S.%f}] session={session_id} node={node_name} "
        f"attempt={attempt} status={status}"
    )
    try:
        log_dir = _log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / _log_file_name()
        with _LOCK:
            with file_path.open("a", encoding="utf-8") as fh:
                fh.write(header + "\n")
                fh.write((raw_response or "").rstrip("\n") + "\n")
                fh.write("-" * 80 + "\n")
    except OSError:
        pass


__all__ = ["configure_ideation_llm_log", "log_llm_response"]
