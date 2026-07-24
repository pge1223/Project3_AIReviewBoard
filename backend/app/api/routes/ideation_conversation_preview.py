# 작성자: 용준/Claude(2026-07-20)
# 목적: 개발용 "아이디어 발전 회의(ideation)" 대화형(conversational) 프리뷰 API.
#       기존 배치형 프리뷰(ideation_preview.py, POST /ideation-preview 한 번 호출로 전체
#       회의를 끝까지 돌림)와 달리, 이 라우터는 질문 하나마다 사용자 응답을 기다리는
#       구조를 검증하기 위한 것이다 — 역시 정식 API가 아니다.
#
# 정식 기능과의 차이(요청 범위 그대로):
#   - MongoDB에 저장하지 않는다. 세션은 프로세스 메모리에만 있는 in-memory 저장소
#     (_SessionStore)에 보관되며, 서버가 재시작되면 모든 세션이 사라진다.
#   - TTL(_SESSION_TTL_SECONDS)이 지난 세션은 자동 만료되고, 세션 수가 상한
#     (_MAX_SESSIONS)을 넘으면 가장 오래된 세션부터 제거한다(둘 다 메모리 누수 방지).
#   - 기존 배치형 /ideation-preview, 심사형 analyze_project()/run_meeting()은 전혀
#     건드리지 않는다.
#
# settings.ENABLE_IDEATION_PREVIEW가 False(기본값)면 main.py가 이 라우터 자체를 앱에
# 등록하지 않는다 — ideation_preview.py와 같은 플래그를 공유한다(둘 다 같은 "개발용
# ideation 프리뷰" 묶음이라 별도 플래그를 새로 만들지 않았다).
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import OpenAI

from app.core.llm import trace_openai_client
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ideation-conversation", tags=["ideation-conversation-preview (dev only)"])

# ideation_preview.py와 동일한 방식으로 ai/meeting을 sys.path에 올린다.
_MEETING_DIR = Path(__file__).resolve().parents[4] / "ai" / "meeting"
if str(_MEETING_DIR) not in sys.path:
    sys.path.insert(0, str(_MEETING_DIR))

from graph import (  # noqa: E402
    IdeationCancelled,
    IdeationConvState,
    active_stage_for,
    bind_trace_context,
    configure_ideation_trace,
    continue_ideation_expert_turn,
    finalize_ideation_conversation,
    is_late_request_event,
    reset_trace_context,
    reply_ideation_conversation,
    reply_to_interjection,
    sanitize_preview,
    start_ideation_conversation,
    trace_event,
)

configure_ideation_trace(
    enabled=settings.ENABLE_IDEATION_TRACE_LOGS,
    content_max_chars=settings.IDEATION_TRACE_CONTENT_MAX_CHARS,
    stream_deltas=settings.IDEATION_TRACE_STREAM_DELTAS,
)

from app.api.routes.meetings import _role_retrieval_service  # noqa: E402
# 용준/Claude(2026-07-21, 요청: 실시간 스트리밍) — 스트리밍 llm_call 생성 로직은 별도
# 모듈(ideation_conversation_streaming.py)로 분리했다(FastAPI/OpenAI 클라이언트 배선은
# 이 파일이, "프롬프트를 보고 어떻게 스트리밍할지 결정"하는 순수 로직은 그 모듈이 맡는다
# — 순수 로직 쪽이 실제 OpenAI 없이도 유닛 테스트가 가능해야 하기 때문이다).
from app.api.routes.ideation_conversation_streaming import make_streaming_llm_call  # noqa: E402

# ---------------------------------------------------------------------------
# 안전장치 상한값 (요청 "개발용 API" 절)
# ---------------------------------------------------------------------------
_SESSION_TTL_SECONDS = 30 * 60  # 30분 이상 응답 없는 세션은 만료된 것으로 간주한다.
_MAX_SESSIONS = 200  # 이 이상이면 가장 오래전에 활동한 세션부터 제거한다(메모리 상한).
_MAX_ROUNDS_CAP = 3  # 요청 max_rounds가 이보다 크면 잘라낸다.
_MAX_TEXT_LENGTH = 2000  # 공모전 설명/아이디어/답변 1건당 최대 길이(문자 수).
# 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — 발언 순서가 더 이상 라운드당
# 고정 횟수가 아니라 쟁점·반론에 따라 동적으로 늘어난다(최대 MAX_EXPERT_TURNS_PER_ROUND=8회
# + discussion_facilitator 1회 + sufficiency/판정류 호출 여유). 각 구조화 호출은 재시도
# 1회까지 더 셀 수 있어(_safe_call_structured_json) 여유를 넉넉히 둔다.
_MAX_LLM_CALLS_PER_REQUEST = 24  # 한 HTTP 요청에서 허용하는 최대 LLM 호출 수(재시도 포함 여유).
# "잠시만" 취소 확인 대기 상한(초) — 취소 신호를 보낸 뒤 워커 스레드가 실제로 세션 락을
# 반납할 때까지 짧게 폴링한다(요청: "취소 완료 전에 새 reply를 보내 세션 lock 409가
# 발생하지 않게").
_CANCEL_CONFIRM_TIMEOUT_SECONDS = 5.0


class _SessionRecord:
    __slots__ = (
        "state",
        "created_at",
        "last_active_at",
        "busy_lock",
        "active_request_id",
        "cancel_event",
        "use_rag",
        "project_id",
    )

    def __init__(self, state: IdeationConvState, *, use_rag: bool = False, project_id: Optional[str] = None):
        self.state = state
        # 용준/Claude(2026-07-22, RAG 근거 유실 수정 2탄): /start 시점의 use_rag/project_id를
        # 세션에 보관한다 — evidence_lookup은 콜러블이라 그래프 state(dict, 체크포인터가
        # 직렬화할 수 있어야 함)에 넣을 수 없었고, 그래서 /reply·/reply/stream이 재개할 때
        # evidence_lookup을 다시 만들 방법이 없어 항상 None으로 진행됐다(RAG 검색 자체가
        # 호출되지 않음 — 최초 /start 턴에서만 검색되고 그 이후 모든 턴은 검색 없이 진행됨).
        # ReplyRequest에 project_id/use_rag를 새로 받지 않고(기존 API 계약 유지), /start 때
        # 결정된 값을 세션에 저장해두고 재사용한다.
        self.use_rag = use_rag
        self.project_id = project_id
        now = time.time()
        self.created_at = now
        self.last_active_at = now
        # 용준/Claude(2026-07-21, 요청: 세션 안정성) — 같은 session_id에 동시에 여러
        # reply/finalize 요청이 들어오면(사용자가 전송 버튼을 연타하거나 중복 탭을 열어도)
        # 그래프가 같은 previous_state를 동시에 두 번 진행시켜 메시지가 뒤섞이거나
        # candidate_regeneration_count 같은 카운터가 잘못 갱신될 수 있다. 세션마다 락 하나로
        # "지금 이 세션을 처리 중인 요청이 있는지"만 표시한다(전역 락이 아니라 세션별이라
        # 서로 다른 세션의 요청은 완전히 병렬로 처리된다).
        self.busy_lock = threading.Lock()
        # 용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소) — 지금 이 세션에서 진행 중인
        # 스트리밍 요청의 request_id와, 그 요청을 취소하라는 신호를 전달하는 이벤트. 스트리밍
        # 요청이 아니면(/reply, /finalize) 둘 다 None으로 유지된다.
        self.active_request_id: str | None = None
        self.cancel_event: threading.Event | None = None


class _SessionStore:
    """개발용 인메모리 세션 저장소. 스레드 세이프하게(threading.Lock) TTL 만료와 최대
    세션 수 제한을 처리한다 — 정식 서비스에서는 MongoDB 등 영속 저장소로 교체돼야 한다
    (최종 보고서 "정식 전환 시 교체할 임시 코드" 참고)."""

    def __init__(self, ttl_seconds: int, max_sessions: int):
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._sessions: dict[str, _SessionRecord] = {}
        self._lock = threading.Lock()

    def _sweep_expired_locked(self) -> None:
        now = time.time()
        expired = [
            sid for sid, rec in self._sessions.items() if now - rec.last_active_at > self._ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

    def _evict_oldest_locked(self) -> None:
        while len(self._sessions) >= self._max_sessions:
            oldest_id = min(self._sessions, key=lambda sid: self._sessions[sid].last_active_at)
            del self._sessions[oldest_id]

    def create(self, state: IdeationConvState, *, use_rag: bool = False, project_id: Optional[str] = None) -> None:
        with self._lock:
            self._sweep_expired_locked()
            self._evict_oldest_locked()
            self._sessions[state["session_id"]] = _SessionRecord(state, use_rag=use_rag, project_id=project_id)

    def get(self, session_id: str) -> IdeationConvState:
        with self._lock:
            self._sweep_expired_locked()
            record = self._sessions.get(session_id)
            if record is None:
                raise KeyError(session_id)
            return record.state

    def update(self, session_id: str, state: IdeationConvState) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                # 갱신 도중 TTL 만료로 사라졌을 수 있다 — 새로 만들어 계속 진행한다.
                record = _SessionRecord(state)
                self._sessions[session_id] = record
            record.state = state
            record.last_active_at = time.time()

    def get_record(self, session_id: str) -> _SessionRecord | None:
        """용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소) — busy_lock을 잡지 않고 레코드
        참조만 반환한다(취소 API는 "지금 이 세션을 처리 중"이어도 신호를 보낼 수 있어야
        하므로 try_acquire와 달리 락을 요구하지 않는다)."""
        with self._lock:
            return self._sessions.get(session_id)

    def try_acquire(self, session_id: str) -> _SessionRecord | None:
        """이 세션을 지금 처리해도 되면 레코드를 반환하고 락을 잡는다. 세션이 없으면
        None, 이미 다른 요청이 처리 중이면(락이 이미 잡혀 있으면) 별도 예외
        (_SessionBusyError)를 던진다 — "세션 없음"과 "세션 사용 중"을 호출부가 다른 HTTP
        상태 코드(404 vs 409)로 구분해야 하기 때문이다. 성공하면 반드시
        release(session_id)를 호출해야 한다(finally 블록에서)."""
        with self._lock:
            self._sweep_expired_locked()
            record = self._sessions.get(session_id)
        if record is None:
            return None
        if not record.busy_lock.acquire(blocking=False):
            raise _SessionBusyError(session_id)
        return record

    def release(self, session_id: str) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
        if record is not None:
            try:
                record.busy_lock.release()
            except RuntimeError:
                pass  # 이미 풀려 있으면(방어적) 조용히 무시한다.


class _SessionBusyError(Exception):
    """요청 1건이 이미 이 세션을 처리 중일 때(동시 reply/finalize 방지) 던진다."""

    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.session_id = session_id


_store = _SessionStore(_SESSION_TTL_SECONDS, _MAX_SESSIONS)


def _require_preview_enabled() -> None:
    if not settings.ENABLE_IDEATION_PREVIEW:
        raise HTTPException(status_code=404, detail="Not Found")


def _require_streaming_enabled() -> None:
    if not settings.ENABLE_IDEATION_STREAMING:
        raise HTTPException(status_code=404, detail="Not Found")


def _acquire_session_or_404(session_id: str) -> IdeationConvState:
    """이 세션을 지금 처리해도 되면 previous_state를 반환하고 세션 락을 잡는다(호출부가
    반드시 finally에서 _store.release(session_id)를 불러야 한다). 세션이 없으면 404,
    이미 다른 요청이 처리 중이면 _SessionBusyError를 그대로 전파한다(호출부가 409로
    변환한다)."""
    record = _store.try_acquire(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없거나 만료되었습니다.")
    return record.state


def _acquire_session_record_or_404(session_id: str) -> "_SessionRecord":
    """_acquire_session_or_404와 동일하지만 record 자체(취소 이벤트 등록용)를 반환한다."""
    record = _store.try_acquire(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없거나 만료되었습니다.")
    return record


def _clamp_text(value: str, field_name: str) -> str:
    value = (value or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail=f"{field_name}은(는) 비어 있을 수 없습니다.")
    if len(value) > _MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name}은(는) 최대 {_MAX_TEXT_LENGTH}자까지 입력할 수 있습니다.",
        )
    return value


def _clamp_optional_text(value: str, field_name: str) -> str:
    """_clamp_text와 달리 빈 문자열을 허용한다 — user_idea가 이 경우다(요청 6번: "user_idea가
    빈 문자열이어도 start API가 400을 반환하지 않음"). 길이 상한만 검사한다. 빈 값은
    initial_conv_state()가 discovery 모드로 해석한다(ideation_conv_state.py 참고)."""
    value = (value or "").strip()
    if len(value) > _MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name}은(는) 최대 {_MAX_TEXT_LENGTH}자까지 입력할 수 있습니다.",
        )
    return value


def _build_llm_call(session_id: str, model: str):
    """실제 OpenAI 호출. 요청 1건당 LLM 호출 횟수 상한(_MAX_LLM_CALLS_PER_REQUEST)을 두어
    라우팅 버그로 인한 루프·과호출을 방지한다(ideation_preview.py의 동일 안전장치와 같은
    정책). API 키/내부 프롬프트는 절대 응답에 담지 않는다 — 이 함수는 호출 결과 텍스트만
    반환하고, 실패 시에도 프롬프트 원문이 아니라 서버 로그에만 상세를 남긴다."""
    client = trace_openai_client(OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=1))
    call_count = {"n": 0}

    def llm_call(prompt: str) -> str:
        call_count["n"] += 1
        if call_count["n"] > _MAX_LLM_CALLS_PER_REQUEST:
            raise RuntimeError(
                f"[{session_id}] 프리뷰 LLM 호출 상한({_MAX_LLM_CALLS_PER_REQUEST}회) 초과 — "
                "루프 또는 재시도 폭주 의심, 중단합니다."
            )
        started = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        elapsed = time.time() - started
        logger.info(
            "[%s] 대화형 프리뷰 LLM 호출 #%d model=%s elapsed=%.1fs usage=%s",
            session_id,
            call_count["n"],
            model,
            elapsed,
            resp.usage.model_dump() if resp.usage else None,
        )
        return resp.choices[0].message.content

    return llm_call


def _build_streaming_backends(session_id: str, model: str):
    """실제 OpenAI 클라이언트를 감싸 make_streaming_llm_call()이 기대하는 두 콜백을
    만든다 — stream_chat_completion(prompt)는 원문 텍스트 조각을 순서대로 만들어내는
    제너레이터(OpenAI stream=True), call_chat_completion(prompt)은 기존 _build_llm_call과
    동일한 블로킹 호출(스트리밍 대상이 아닌 분류·후보 생성 등에 쓴다). 호출 횟수 상한은
    make_streaming_llm_call이 중앙에서 관리하므로 여기서는 세지 않는다."""
    client = trace_openai_client(OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=1))

    def stream_chat_completion(prompt: str):
        started = time.time()
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            stream=True,
        )
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                text = getattr(chunk.choices[0].delta, "content", None)
                if text:
                    yield text
        finally:
            logger.info(
                "[%s] 대화형 프리뷰 스트리밍 LLM 호출 model=%s elapsed=%.1fs", session_id, model, time.time() - started
            )

    def call_chat_completion(prompt: str) -> str:
        started = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        logger.info(
            "[%s] 대화형 프리뷰 LLM 호출(비-스트리밍 구간) model=%s elapsed=%.1fs",
            session_id,
            model,
            time.time() - started,
        )
        return resp.choices[0].message.content

    return stream_chat_completion, call_chat_completion


_RAG_SPEAKER_META = {
    "planning_expert": ("기획 위원", "planning"),
    "dev_expert": ("개발 위원", "technology"),
}


def _lookup_accepts_runtime_scope(fn) -> bool:
    """lookup 콜러블이 runtime_scope 키워드 인자를 받는지 검사한다(ai/meeting/graph/
    ideation_nodes.py::_lookup_accepts_runtime_scope와 동일한 목적 — 여기서는 테스트가 흔히
    주입하는 (persona_id, query) 2-인자 lambda까지 그대로 감싸야 하는 _trace_evidence_lookup에
    적용한다). ai.rag가 실제로 만드는 lookup은 runtime_scope를 받지만, 테스트용 fake는 받지
    않을 수 있으므로 지원 여부를 먼저 확인해 하위 호환을 유지한다."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if "runtime_scope" in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _trace_evidence_lookup(lookup, *, project_id: str, top_k: int):
    """모든 아이디어 회의 RAG 호출을 발언 생성 직전/직후 로그로 감싼다.

    검색 구현이나 반환값은 바꾸지 않는다. 프롬프트와 청크 원문 전체는 기록하지 않고,
    호출 여부를 확인하는 데 필요한 검색어 미리보기·문서명·chunk_id·점수만 남긴다.
    """
    lookup_accepts_runtime_scope = _lookup_accepts_runtime_scope(lookup)

    def traced_lookup(persona_id: str, query: str, *, runtime_scope: Optional[dict] = None):
        speaker_name, role_id = _RAG_SPEAKER_META.get(persona_id, (persona_id, None))
        started = time.perf_counter()
        trace_event(
            "IDEATION_RAG_SEARCH_START",
            speaker=persona_id,
            speaker_name=speaker_name,
            role=role_id,
            project_id=project_id,
            top_k=top_k,
            query=sanitize_preview(query, limit=200),
            timing="전문가 발언 생성 전",
        )
        try:
            # 용준/Claude(2026-07-23, 요청: stale closure 수정) — runtime_scope는
            # ai/meeting/graph 노드가 evidence_lookup을 호출하는 바로 그 순간의 최신 graph
            # state에서 읽은 session_id/selected_candidate_document_id다. lookup(아래
            # _evidence_lookup_for가 만든 실제 ai.rag 콜러블)이 이 값을 받으면 lookup을
            # 만들 때 캡처해둔 closure 스냅샷(요청 시작 시점 값)보다 우선한다.
            if lookup_accepts_runtime_scope:
                evidence = lookup(persona_id, query, runtime_scope=runtime_scope)
            else:
                evidence = lookup(persona_id, query)
        except Exception:
            trace_event(
                "IDEATION_RAG_SEARCH_FAILED",
                level=logging.ERROR,
                speaker=persona_id,
                speaker_name=speaker_name,
                role=role_id,
                project_id=project_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            raise

        trace_event(
            "IDEATION_RAG_SEARCH_COMPLETE",
            speaker=persona_id,
            speaker_name=speaker_name,
            role=role_id,
            project_id=project_id,
            result_count=len(evidence),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            sources=[
                {
                    "document": item.get("document_name"),
                    "chunk_id": item.get("chunk_id"),
                    "page": item.get("page"),
                    "score": item.get("score"),
                }
                for item in evidence
                if isinstance(item, dict)
            ],
        )
        # 용준/Claude(2026-07-22, 요청: 로그 — IDEATION_TARGET_EVIDENCE_SEARCHED) — criteria/
        # target(그중 선택된 후보 target/사용자 답변 target)이 실제로 몇 건씩 검색됐는지 남긴다.
        # ai.rag의 반환 dict가 이미 document_role/ideation_source_type을 담고 있으므로(요청:
        # 역할별 검색 데이터 구성 + 세션 범위 검색) 여기서는 그 값을 세는 것만 한다 — 판정
        # 로직 자체는 ai.rag에 있다.
        criteria_count = sum(1 for item in evidence if isinstance(item, dict) and item.get("document_role") == "criteria")
        target_items = [item for item in evidence if isinstance(item, dict) and item.get("document_role") == "target"]
        candidate_target_count = sum(
            1 for item in target_items if item.get("ideation_source_type") == "ideation_candidate"
        )
        user_answer_target_count = sum(
            1 for item in target_items if item.get("ideation_source_type") == "user_session_answer"
        )
        expected_roles = {"criteria", "target"}
        found_roles = {item.get("document_role") for item in evidence if isinstance(item, dict)}
        missing_document_roles = sorted(expected_roles - found_roles)
        trace_event(
            "IDEATION_TARGET_EVIDENCE_SEARCHED",
            speaker=persona_id,
            criteria_count=criteria_count,
            target_count=len(target_items),
            candidate_target_count=candidate_target_count,
            user_answer_target_count=user_answer_target_count,
            missing_document_roles=missing_document_roles,
        )
        return evidence

    setattr(traced_lookup, "trace_project_id", project_id)
    setattr(traced_lookup, "trace_top_k", top_k)
    return traced_lookup


def _evidence_lookup_for(
    use_rag: bool,
    project_id: Optional[str],
    *,
    session_id: Optional[str] = None,
    selected_candidate_document_id: Optional[str] = None,
):
    if not use_rag:
        trace_event(
            "IDEATION_RAG_CONFIGURATION",
            enabled=False,
            project_id=project_id,
            reason="use_rag_false",
        )
        return None
    from ai.rag.orchestration.ideation_evidence_service import make_ideation_evidence_lookup

    lookup = make_ideation_evidence_lookup(
        project_id=project_id,
        role_retrieval_service=_role_retrieval_service,
        top_k=5,
        session_id=session_id,
        selected_candidate_document_id=selected_candidate_document_id,
    )
    lookup = _trace_evidence_lookup(lookup, project_id=project_id, top_k=5)
    trace_event(
        "IDEATION_RAG_CONFIGURATION",
        enabled=True,
        project_id=project_id,
        top_k=5,
        roles={"planning_expert": "planning", "dev_expert": "technology"},
        session_id=session_id,
        selected_candidate_document_id=selected_candidate_document_id,
    )
    return lookup


def _index_target_evidence_for(use_rag: bool, project_id: Optional[str]):
    """용준/Claude(2026-07-22, 요청: 선택된 아이디어/사용자 답변을 target evidence로 색인) —
    evidence_lookup/ground_claims와 동일한 정책: use_rag=False거나 project_id가 없으면(요청
    17-3번 "project_id 없는 세션에서 전역 target 문서를 만들면 안 됨") 색인 콜러블 자체를
    주입하지 않는다 — ai/meeting/graph 쪽은 index_target_evidence=None이면 색인을 건너뛰고
    안전하게 진행한다(evidence_lookup=None과 같은 패턴)."""
    if not use_rag or not project_id:
        return None
    from ai.rag.orchestration.ideation_target_indexing_service import (
        IdeationTargetIndexingError,
        index_selected_candidate_as_target,
        index_user_answer_as_target,
    )
    from app.api.routes.documents import _get_indexing_service

    def index_target_evidence(kind: str, payload: dict) -> dict:
        indexing_service = _get_indexing_service()
        started = time.perf_counter()
        session_id = payload.get("session_id")
        if kind == "candidate":
            candidate_id = payload.get("candidate_id")
            trace_fields = {"session_id": session_id, "candidate_id": candidate_id, "source_type": "ideation_candidate"}
            try:
                result = index_selected_candidate_as_target(
                    indexing_service=indexing_service,
                    project_id=project_id,
                    session_id=session_id,
                    candidate_id=candidate_id,
                    candidate=payload.get("candidate") or {},
                )
            except IdeationTargetIndexingError as exc:
                trace_event(
                    "IDEATION_TARGET_EVIDENCE_UPSERT",
                    level=logging.WARNING,
                    created_or_updated="failed",
                    error=sanitize_preview(str(exc), limit=100),
                    **trace_fields,
                )
                raise
        elif kind == "user_answer":
            user_message_id = payload.get("user_message_id")
            trace_fields = {
                "session_id": session_id,
                "user_message_id": user_message_id,
                "source_type": "user_session_answer",
            }
            try:
                result = index_user_answer_as_target(
                    indexing_service=indexing_service,
                    project_id=project_id,
                    session_id=session_id,
                    user_message_id=user_message_id,
                    answer_text=payload.get("answer_text") or "",
                    pending_question=payload.get("pending_question"),
                    pending_question_topic=payload.get("pending_question_topic"),
                )
            except IdeationTargetIndexingError as exc:
                trace_event(
                    "IDEATION_TARGET_EVIDENCE_UPSERT",
                    level=logging.WARNING,
                    created_or_updated="failed",
                    error=sanitize_preview(str(exc), limit=100),
                    **trace_fields,
                )
                raise
        else:
            raise ValueError(f"알 수 없는 target evidence 종류입니다: {kind!r}")

        trace_event(
            "IDEATION_TARGET_EVIDENCE_UPSERT",
            project_id=project_id,
            document_id=result.document_id,
            chunk_count=result.chunk_count,
            content_hash=result.content_hash,
            created_or_updated="upserted",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            **trace_fields,
        )
        return {"document_id": result.document_id, "chunk_count": result.chunk_count, "status": "ok"}

    return index_target_evidence


# 용준/Claude(2026-07-22, 요청: RAG 근거 실제 활용 강화) — ai/meeting/graph는 ai.rag를 직접
# import하지 않는다(ai/rag/tests/test_meeting_evidence_service.py::TestScopeBoundary가
# 강제하는 경계). evidence_lookup과 같은 이유로, claim-evidence 연결 검증(RAG-004 관련성
# 판정 재사용)도 이 API 레이어가 실제 구현을 주입한다.
_ROLE_RELEVANCE_KEYWORDS = {
    "planning_expert": ["실현 가능성", "경제성", "평가기준", "심사", "차별성", "사업성", "공모전"],
    "dev_expert": ["데이터", "API", "구현", "보안", "성능", "아키텍처", "연동", "기술"],
}


def _ground_claims_for(use_rag: bool):
    if not use_rag:
        return None
    from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl

    def grounder(persona_id: str, claims, retrieved_evidence: list[dict]) -> dict:
        return _ground_claims_impl(
            claims, retrieved_evidence, role_keywords=_ROLE_RELEVANCE_KEYWORDS.get(persona_id)
        )

    return grounder


def _evidence_planner_for(use_rag: bool):
    """용준/Claude(2026-07-23, Phase 1 "Shadow Deterministic Evidence Planner", Phase 2 "Active
    Evidence Injection") — evidence_lookup/ground_claims와 같은 lazy-import 정책(ai/meeting은
    ai.rag를 모른다). SHADOW/DISCUSSION 둘 다 꺼져 있으면(기본값) planner 콜러블 자체를
    주입하지 않는다 — make_conv_discussion_node가 evidence_planner=None으로 실행되어 기존
    동작과 완전히 동일하다.

    두 플래그가 동시에 켜져도 planner는 턴당 한 번만 실행된다(ideation_conv_nodes.py::
    _run_shadow_evidence_planner가 유일한 호출 지점) — 여기서는 그 결과를 "실제 발언에 쓸지"
    여부만 반환된 콜러블의 active 속성(evidence_lookup.trace_project_id와 같은 패턴, 시그니처
    변경 없이 그래프 조립 함수들을 그대로 통과시킨다)으로 표시한다."""
    if not use_rag or not (
        settings.ENABLE_IDEATION_EVIDENCE_PLANNER_SHADOW or settings.ENABLE_IDEATION_EVIDENCE_PLANNER_DISCUSSION
    ):
        return None
    from ai.rag.orchestration.ideation_evidence_planner import build_evidence_plan

    def planner(*, persona_id, effective_issue, retrieved_evidence, runtime_scope, shadow_history):
        return build_evidence_plan(
            persona_id=persona_id,
            effective_issue=effective_issue,
            retrieved_evidence=retrieved_evidence,
            runtime_scope=runtime_scope,
            shadow_history=shadow_history,
        )

    planner.active = bool(settings.ENABLE_IDEATION_EVIDENCE_PLANNER_DISCUSSION)
    return planner


def _serialize_state(state: IdeationConvState) -> dict:
    """API 응답 형태로 변환한다. 실패해도 원본 예외·프롬프트·API 키는 절대 포함하지 않는다.

    용준/Claude(2026-07-21): discovery(아이디어 발굴) 모드 필드(ideation_mode/idea_candidates/
    selected_idea/selection_reason)를 추가했다 — 기존 필드는 하나도 삭제·이름 변경하지
    않았으므로(순수 추가) 기존 프론트/클라이언트는 새 필드를 무시하면 그대로 동작한다
    (요청: 기존 API 요청/응답과 최대한 하위 호환).

    용준/Claude(2026-07-21): active_stage 추가 — ideation_mode는 세션이 "최초 진입할 때"
    discovery였는지 refinement였는지만 기록하고 이후 절대 바뀌지 않는다(요청 사항 그대로
    유지). 그런데 discovery 세션도 후보를 선택한 뒤에는 refinement와 같은 흐름을 타므로,
    프론트가 배지를 정확히 표시하려면 "현재 진행 단계"가 별도로 필요하다 — active_stage가
    그 값이다(ideation_conv_state.py::active_stage_for 참고). 이 필드도 순수 추가다.

    용준/Claude(2026-07-21, 질문 주제 구조화): resolved_topics/pending_question_topic도
    순수 추가 필드다 — 기존 클라이언트는 무시하면 그대로 동작한다. state에 없으면(구버전
    세션) 각각 빈 배열/None으로 기본값을 준다. messages 안의 개별 메시지도 이제 선택적으로
    structured 필드(judgment/reason/suggestion/confirmed/unconfirmed)를 가질 수 있다 —
    content 문자열은 그대로 유지되므로 messages 자체의 필드 목록은 바뀌지 않는다(추가 키가
    각 메시지 dict 안에 하나 늘었을 뿐).

    용준/Claude(2026-07-21, /board 실 연동): original_idea_candidates/selection_intent/
    user_selection_message/source_candidates/merge_analysis도 순수 추가 필드다 —
    ideation_conv_state.py/ideation_conv_discovery.py가 이미 state에 계산해 두는 값을
    노출만 한다(새 로직 없음). /board의 결과 화면·후보 결합 컨텍스트 표시가 이 값들을
    쓴다 — conversation_context의 최근 메시지에 우연히 남아있는 것에 기대지 않고 구조화된
    필드로 직접 읽게 하기 위함이다."""
    return {
        "session_id": state["session_id"],
        "phase": state["phase"],
        "round": state["round"],
        "max_rounds": state["max_rounds"],
        # 가은/Claude(2026-07-21): /board "주제 아이디어 회의" 화면 헤더에 지금 어떤 공모전
        # 주제로 좁혀가는 중인지 보여주려고 노출한다. notice_and_criteria는 start 시점에
        # 넣어둔 값이라 세션을 이어받아(resume) 다시 그려도 그대로 유지된다 — 순수 추가 필드.
        "competition_name": (state.get("notice_and_criteria") or {}).get("competition_name", ""),
        "messages": state["messages"],
        "consensus": state["consensus"],
        "unresolved_issues": state["unresolved_issues"],
        "idea_proposal": state.get("idea_proposal"),
        "idea_canvas": state.get("idea_canvas"),
        "ideation_mode": state.get("ideation_mode", "refinement"),
        "active_stage": active_stage_for(state["phase"]),
        "idea_candidates": state.get("idea_candidates", []),
        "original_idea_candidates": state.get("original_idea_candidates", []),
        "selected_idea": state.get("selected_idea"),
        # 용준/Claude(2026-07-22, 요청: 선택된 아이디어를 target 문서로 생성) — 순수 추가
        # 필드. 색인이 주입되지 않았거나(use_rag=False) 실패했으면 None이다.
        "selected_idea_document_id": state.get("selected_idea_document_id"),
        "selection_reason": state.get("selection_reason"),
        "resolved_topics": state.get("resolved_topics", []),
        "pending_question_topic": state.get("pending_question_topic"),
        "selection_intent": state.get("selection_intent"),
        "user_selection_message": state.get("user_selection_message"),
        "source_candidates": state.get("source_candidates", []),
        "merge_analysis": state.get("merge_analysis"),
        # 용준/Claude(2026-07-21, 요청: 위원 간 실제 회의로 개편) — 순수 추가 필드. 기존
        # 클라이언트는 무시하면 그대로 동작한다. 구버전 세션(discussion_rounds 키가 없는
        # state)에는 빈 배열을 기본값으로 준다.
        "discussion_rounds": state.get("discussion_rounds", []),
        # 가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입): 순수 추가 필드 — 세션
        # 시작 시 넘긴 항목을 그대로 노출한다(디버깅/향후 화면 표시용, 세션 중 바뀌지 않음).
        "application_form_items": state.get("application_form_items", []),
        "error": (
            {"code": "IDEATION_CONV_NODE_FAILED", "message": f"{state.get('failed_node')} 노드에서 실패했습니다."}
            if state["phase"] == "failed"
            else None
        ),
    }


class StartRequest(BaseModel):
    competition_name: str
    competition_document: str = ""
    # 용준/Claude(2026-07-21): 요청 6번 "user_idea가 빈 문자열이어도 start API가 400을
    # 반환하지 않음" — 기본값을 빈 문자열로 두어 필드 자체를 생략해도 되게 한다(기존
    # 클라이언트가 항상 user_idea를 채워 보내던 것과 완전히 호환된다). 빈 값/공백만 있으면
    # discovery(아이디어 발굴) 모드로 시작한다 — 공모전명은 계속 필수다.
    user_idea: str = ""
    max_rounds: int = 3
    use_rag: bool = False
    project_id: Optional[str] = None
    model: str = Field(default="")  # 비워두면 settings.reviewer_model()(LLM_PROFILE 기준) 사용 — 개발용 오버라이드 허용.
    # 가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입): 순수 추가 필드. 프론트가
    # documents.py::get_application_form_analysis 결과를 그대로 넘긴다([{"field_name",
    # "description","char_limit"}], 양식에 있는 만큼 전부 — 개수 상한 없음, 2026-07-22
    # 제거됨) — discussion 프롬프트에 참고 자료로만 주입된다(질문 주제·순서는 안 바꿈,
    # ideation_conv_discussion.txt 참고). 비워도(기본값 빈 리스트) 기존 클라이언트와
    # 완전히 동일하게 동작한다.
    application_form_items: list[dict] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    message: str
    model: str = Field(default="")
    # 용준/Claude(2026-07-22, 요청: "잠시만" 버튼 — 질문 대상 선택): 선택 필드다(기존
    # /reply, /reply/stream 호출과 하위 호환) — 값이 있으면 지정 위원이 먼저 답하는
    # reply_to_interjection 경로로 라우팅한다(아래 reply_conversation_stream 참고).
    target_speaker_id: Optional[str] = None
    # 사용자가 누구의 발언에 반응하는지와 누가 먼저 답할지는 의미가 다르므로 별도 보존한다.
    opinion_target_speaker_id: Optional[str] = None
    interrupted_speaker_id: Optional[str] = None
    interrupted_request_id: Optional[str] = None
    active_issue_id: Optional[str] = None
    # 재인/Claude(2026-07-23, 아바타 페이싱 연동 — 실측: "진행자 2번·기획 1번·개발 1번이
    # 2초 간격으로 그냥 다 나왔다"): true면 이 reply가 새 라운드의 첫 위원 발언을 만들
    # 때도(예: 아이디어 후보 선택 직후) 그 발언 1건에서 멈춘다 — reply_ideation_conversation의
    # stop_after_expert_turn 그대로. 기본값 False로 기존 클라이언트(아바타 없는 테스트 등)는
    # 전혀 영향받지 않는다 — 아바타가 있는 화면만 매번 true로 보낸다.
    single_turn: bool = False


class ContinueTurnRequest(BaseModel):
    # 재인/Claude(2026-07-23, 아바타 페이싱 연동): 새 사용자 발언이 없는 호출이라
    # ReplyRequest와 달리 message 필드가 없다 — 아바타가 방금 발언 재생 도중(끝나기 3초
    # 전) "다음 위원 미리 준비" 신호로 부르는 용도.
    model: str = Field(default="")


class FinalizeRequest(BaseModel):
    model: str = Field(default="")


class CancelRequest(BaseModel):
    request_id: Optional[str] = None


# 가은/Claude(2026-07-21): 실측 제보 — 이 화면(board "작성 전" 흐름)이 늘 DEV_LLM_REVIEWER_
# MODEL로 고정돼 있어서, dev 프로필에 gpt-5-nano처럼 느린 추론 모델을 넣으면(파싱/JSON
# 형식 테스트용으로 dev 프로필을 그렇게 쓰는 게 팀 의도) 실제 사용자가 쓰는 이 대화형
# 회의까지 덩달아 느려졌다. 다른 LLM 호출(documents.py/meetings.py)과 똑같이
# LLM_PROFILE(dev|quality|premium)을 그대로 따르게 통일한다.
def _effective_model(requested: str) -> str:
    return requested.strip() or settings.reviewer_model()


@router.post("/start")
async def start_conversation(request: StartRequest):
    _require_preview_enabled()
    if request.use_rag and not request.project_id:
        raise HTTPException(status_code=400, detail="use_rag=true이면 project_id가 필요합니다.")

    competition_name = _clamp_text(request.competition_name, "competition_name")
    user_idea_text = _clamp_optional_text(request.user_idea, "user_idea")
    competition_document = (request.competition_document or "")[:_MAX_TEXT_LENGTH]
    effective_max_rounds = max(1, min(request.max_rounds, _MAX_ROUNDS_CAP))

    session_id = f"IDEA-CONV-{uuid.uuid4().hex[:8]}"
    notice_and_criteria = {"competition_name": competition_name, "notice_document": competition_document}
    user_idea = {"description": user_idea_text}

    llm_call = _build_llm_call(session_id, _effective_model(request.model))
    # 용준/Claude(2026-07-22, 요청: 세션 범위 검색) — /start 시점에는 아직 후보를 선택하지
    # 않았으므로(discovery 모드 시작) selected_candidate_document_id는 항상 None이다.
    evidence_lookup = _evidence_lookup_for(request.use_rag, request.project_id, session_id=session_id)
    ground_claims = _ground_claims_for(request.use_rag)
    index_target_evidence = _index_target_evidence_for(request.use_rag, request.project_id)
    evidence_planner = _evidence_planner_for(request.use_rag)

    logger.info("[ideation-conversation] 시작 session_id=%s max_rounds=%d", session_id, effective_max_rounds)
    try:
        state = await run_in_threadpool(
            start_ideation_conversation,
            session_id=session_id,
            notice_and_criteria=notice_and_criteria,
            user_idea=user_idea,
            llm_call=llm_call,
            max_rounds=effective_max_rounds,
            evidence_lookup=evidence_lookup,
            ground_claims=ground_claims,
            index_target_evidence=index_target_evidence,
            evidence_planner=evidence_planner,
            application_form_items=request.application_form_items,
        )
    except Exception:
        logger.exception("[ideation-conversation] 시작 실패 session_id=%s", session_id)
        raise HTTPException(status_code=502, detail="대화형 회의 시작 중 오류가 발생했습니다. 서버 로그를 확인하세요.")

    # 용준/Claude(2026-07-22, RAG 근거 유실 수정 2탄): 이후 /reply·/reply/stream이 evidence_lookup을
    # 다시 만들 수 있도록 이번 세션의 use_rag/project_id를 함께 저장한다.
    _store.create(state, use_rag=request.use_rag, project_id=request.project_id)
    return _serialize_state(state)


@router.post("/{session_id}/reply")
async def reply_conversation(session_id: str, request: ReplyRequest):
    _require_preview_enabled()
    try:
        record = _acquire_session_record_or_404(session_id)
    except _SessionBusyError:
        raise HTTPException(status_code=409, detail="이 세션은 이미 다른 요청을 처리하고 있습니다.")
    previous_state = record.state

    trace_tokens = bind_trace_context(session_id)
    try:
        message = _clamp_text(request.message, "message")
        trace_event("IDEATION_REQUEST_STARTED", mode="sync", user_message=sanitize_preview(message))
        if previous_state.get("phase") in (
            "expert_discussion",
            "awaiting_user_decision",
            "discussion_complete",
        ):
            trace_event(
                "IDEATION_USER_INTERJECTION",
                issue=previous_state.get("active_issue_id"),
                content_length=len(message),
                user_message=sanitize_preview(message),
            )
        llm_call = _build_llm_call(session_id, _effective_model(request.model))
        # 용준/Claude(2026-07-22, RAG 근거 유실 수정 2탄): /start에서 저장해둔 use_rag/project_id로
        # evidence_lookup을 다시 만든다 — 예전에는 여기서 evidence_lookup을 아예 넘기지 않아
        # (기본값 None) 첫 턴 이후 모든 턴이 RAG 검색 없이 진행됐다.
        # 용준/Claude(2026-07-22, 요청: 세션 범위 검색 + 후보 변경 시 이전 candidate target
        # 제외; 2026-07-23 정정) — 아래 selected_candidate_document_id는 이 요청이 "시작될
        # 때"(candidate_selection 노드 실행 전)의 값일 뿐이다 — 후보 선택과 첫 전문가 검색이
        # 같은 /reply 안에서 이어지면(candidate_selection이 to_refinement로 곧장
        # planning_expert_discussion까지 진행) 이 값은 여전히 이전 상태(대부분 None)다.
        # 실제 "지금 선택된 후보" 값은 그래프 노드가 evidence_lookup을 호출하는 순간의
        # runtime_scope(ai/meeting/graph/ideation_conv_nodes.py::_runtime_scope_for)가
        # 매번 다시 계산해 우선 적용한다 — 아래 인자는 그 runtime_scope가 없는 호출(예:
        # /start 이전 단계)을 위한 하위 호환 기본값일 뿐이다.
        evidence_lookup = _evidence_lookup_for(
            record.use_rag,
            record.project_id,
            session_id=session_id,
            selected_candidate_document_id=previous_state.get("selected_idea_document_id"),
        )
        ground_claims = _ground_claims_for(record.use_rag)
        index_target_evidence = _index_target_evidence_for(record.use_rag, record.project_id)
        evidence_planner = _evidence_planner_for(record.use_rag)

        try:
            state = await run_in_threadpool(
                reply_ideation_conversation,
                previous_state=previous_state,
                user_message=message,
                llm_call=llm_call,
                evidence_lookup=evidence_lookup,
                ground_claims=ground_claims,
                index_target_evidence=index_target_evidence,
                evidence_planner=evidence_planner,
                stop_after_expert_turn=request.single_turn,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:
            logger.exception("[ideation-conversation] 답변 처리 실패 session_id=%s", session_id)
            raise HTTPException(status_code=502, detail="답변 처리 중 오류가 발생했습니다. 서버 로그를 확인하세요.")

        _store.update(session_id, state)
        return _serialize_state(state)
    finally:
        _store.release(session_id)
        trace_event("IDEATION_SESSION_UNLOCKED", mode="sync")
        reset_trace_context(trace_tokens)


# NDJSON 스트림 소비자(비동기 이벤트 제너레이터)가 "이 스트림이 끝났다"를 판별하는 데
# 쓰는 내부 전용 sentinel — 실제 이벤트 dict와 절대 혼동되지 않도록 object() 그대로 쓴다.
_STREAM_DONE_SENTINEL = object()
# 클라이언트 연결 종료 여부를 이 주기(초)로 재확인한다 — 그동안 새 이벤트가 없어도
# event_queue.get(timeout=...)이 이 시간 뒤에 깨어나 disconnect 체크로 돌아온다(요청:
# "클라이언트 연결 종료 시 가능한 범위에서 LLM 스트림 정리").
_DISCONNECT_POLL_SECONDS = 1.0


@router.post("/{session_id}/reply/stream")
async def reply_conversation_stream(session_id: str, request: ReplyRequest, http_request: Request):
    """POST /reply와 같은 일(사용자 답변 반영)을 하지만, 응답을 한 번에 돌려주는 대신
    NDJSON(application/x-ndjson) 한 줄마다 이벤트 하나씩 흘려보낸다 — 사용자에게 보이는
    메시지를 만드는 LLM 호출은 실제 OpenAI 토큰이 도착하는 즉시 message_delta로 나간다
    (가짜 타이핑 효과 아님, ideation_conversation_streaming.py::make_streaming_llm_call 참고).

    그래프 실행(reply_ideation_conversation)은 동기 함수라 별도 스레드에서 돌리고, 그
    스레드가 sink()로 큐에 넣는 이벤트를 이 async 제너레이터가 꺼내 그대로 클라이언트에게
    전달하는 "생산자(스레드)-소비자(이벤트 루프)" 패턴을 쓴다.

    용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소 + 지정 위원 우선 응답): 세션마다
    이번 요청의 request_id/cancel_event를 등록하고(POST /cancel이 이 값을 보고 신호를
    보낸다), on_snapshot 콜백으로 그래프 노드가 완료될 때마다 세션 state를 증분 저장한다 —
    그래야 도중에 취소돼도 이미 완료된 발언은 canonical state에 남고, 취소된(미완성) 발언만
    빠진다. request.target_speaker_id가 있으면(사용자가 "잠시만"으로 특정 위원을 지정해
    질문한 경우) reply_ideation_conversation 대신 reply_to_interjection으로 라우팅한다."""
    _require_preview_enabled()
    _require_streaming_enabled()
    message = _clamp_text(request.message, "message")
    target_speaker_id = request.target_speaker_id
    if target_speaker_id is not None and target_speaker_id not in ("planning_expert", "dev_expert", "both"):
        raise HTTPException(status_code=400, detail="target_speaker_id는 planning_expert/dev_expert/both 중 하나여야 합니다.")
    opinion_target_speaker_id = request.opinion_target_speaker_id
    if opinion_target_speaker_id is not None and opinion_target_speaker_id not in (
        "planning_expert",
        "dev_expert",
        "both",
    ):
        raise HTTPException(
            status_code=400,
            detail="opinion_target_speaker_id는 planning_expert/dev_expert/both 중 하나여야 합니다.",
        )
    interrupted_speaker_id = request.interrupted_speaker_id
    if interrupted_speaker_id is not None and interrupted_speaker_id not in (
        "planning_expert",
        "dev_expert",
    ):
        raise HTTPException(
            status_code=400,
            detail="interrupted_speaker_id는 planning_expert/dev_expert 중 하나여야 합니다.",
        )

    try:
        record = _acquire_session_record_or_404(session_id)
    except _SessionBusyError:
        raise HTTPException(status_code=409, detail="이 세션은 이미 다른 요청을 처리하고 있습니다.")
    previous_state = record.state

    model = _effective_model(request.model)
    event_queue: "queue.Queue[object]" = queue.Queue()

    request_id = f"REQ-{uuid.uuid4().hex[:10]}"
    cancel_event = threading.Event()
    record.active_request_id = request_id
    record.cancel_event = cancel_event

    def sink(event: dict) -> None:
        if is_late_request_event(event.get("request_id"), request_id):
            return
        event.setdefault("request_id", request_id)
        event_queue.put(event)

    # 용준/Claude(2026-07-22, RAG 근거 유실 수정 2탄): /start에서 저장해둔 record.use_rag/
    # project_id로 evidence_lookup을 다시 만든다 — 예전에는 아래 reply_to_interjection/
    # reply_ideation_conversation 호출에 evidence_lookup을 아예 넘기지 않아 첫 턴 이후
    # 모든 스트리밍 턴이 RAG 검색 없이 진행됐다.
    evidence_lookup = _evidence_lookup_for(
        record.use_rag,
        record.project_id,
        session_id=session_id,
        selected_candidate_document_id=previous_state.get("selected_idea_document_id"),
    )
    ground_claims = _ground_claims_for(record.use_rag)
    index_target_evidence = _index_target_evidence_for(record.use_rag, record.project_id)
    evidence_planner = _evidence_planner_for(record.use_rag)

    def worker() -> None:
        trace_tokens = bind_trace_context(session_id, request_id)
        try:
            stream_chat_completion, call_chat_completion = _build_streaming_backends(session_id, model)
            llm_call = make_streaming_llm_call(
                session_id,
                sink,
                stream_chat_completion=stream_chat_completion,
                call_chat_completion=call_chat_completion,
                max_calls=_MAX_LLM_CALLS_PER_REQUEST,
                cancel_event=cancel_event,
                request_id=request_id,
            )
            sink({"type": "request_started", "request_id": request_id})
            trace_event(
                "IDEATION_REQUEST_STARTED",
                mode="stream",
                target_speaker=target_speaker_id,
                opinion_target_speaker=opinion_target_speaker_id,
                interrupted_speaker=interrupted_speaker_id,
                user_message=sanitize_preview(message),
            )
            is_user_interjection = target_speaker_id is not None or previous_state.get("phase") in (
                "expert_discussion",
                "awaiting_user_decision",
                "discussion_complete",
            )
            if is_user_interjection:
                trace_event(
                    "IDEATION_USER_INTERJECTION",
                    target_speaker=target_speaker_id,
                    opinion_target_speaker=opinion_target_speaker_id,
                    interrupted_speaker=interrupted_speaker_id,
                    issue=previous_state.get("active_issue_id"),
                    interrupted_request_id=None,
                    content_length=len(message),
                    user_message=sanitize_preview(message),
                )
            if target_speaker_id is not None:
                trace_event(
                    "IDEATION_RESUME_STARTED",
                    resume_target_speaker=target_speaker_id,
                    phase=previous_state.get("phase"),
                    next_route=previous_state.get("next_route"),
                )
                state = reply_to_interjection(
                    previous_state=previous_state,
                    user_message=message,
                    target_speaker_id=target_speaker_id,
                    opinion_target_speaker_id=opinion_target_speaker_id or target_speaker_id,
                    interrupted_speaker_id=interrupted_speaker_id,
                    llm_call=llm_call,
                    evidence_lookup=evidence_lookup,
                    ground_claims=ground_claims,
                    index_target_evidence=index_target_evidence,
                    evidence_planner=evidence_planner,
                )
            else:
                state = reply_ideation_conversation(
                    previous_state=previous_state,
                    user_message=message,
                    llm_call=llm_call,
                    evidence_lookup=evidence_lookup,
                    ground_claims=ground_claims,
                    index_target_evidence=index_target_evidence,
                    evidence_planner=evidence_planner,
                    stop_after_expert_turn=request.single_turn,
                )
            _store.update(session_id, state)
            sink({"type": "state", "state": _serialize_state(state)})
            if state.get("phase") == "failed":
                failed_node = state.get("failed_node")
                sink(
                    {
                        "type": "error",
                        "code": "IDEATION_CONV_NODE_FAILED",
                        "message": f"{failed_node or '알 수 없는'} 노드에서 회의 처리가 실패했습니다.",
                        "failed_node": failed_node,
                    }
                )
        except IdeationCancelled as exc:
            # 요청 13번 — 취소는 일반 오류가 아니다(phase="failed"로 만들지 않는다). 취소
            # 시점까지 완료된 발언이 있으면(exc.partial_state) 그것만 canonical state로
            # 저장한다 — 미완성 발언은 애초에 partial_state에 포함되지 않으므로, 화면에
            # 남기는 것은 순전히 프런트의 로컬 기록 책임이다(요청 14번).
            logger.info("[%s] 스트리밍 요청이 사용자에 의해 취소됨 request_id=%s", session_id, request_id)
            if exc.partial_state is not None:
                _store.update(session_id, exc.partial_state)
                trace_event(
                    "IDEATION_PARTIAL_STATE_SAVED",
                    phase=exc.partial_state.get("phase"),
                    message_count=len(exc.partial_state.get("messages", [])),
                )
            trace_event(
                "IDEATION_GRAPH_CANCELLED",
                phase=(exc.partial_state or previous_state).get("phase"),
                next_route=(exc.partial_state or previous_state).get("next_route"),
                completed_messages=len((exc.partial_state or previous_state).get("messages", [])),
            )
            sink({"type": "cancelled", "request_id": request_id})
        except ValueError as exc:
            sink({"type": "error", "code": "invalid_request", "message": str(exc)})
        except Exception:
            logger.exception("[ideation-conversation] 스트리밍 답변 처리 실패 session_id=%s", session_id)
            sink(
                {
                    "type": "error",
                    "code": "llm_failure",
                    "message": "답변 처리 중 오류가 발생했습니다. 서버 로그를 확인하세요.",
                }
            )
        finally:
            record.active_request_id = None
            record.cancel_event = None
            _store.release(session_id)
            trace_event("IDEATION_SESSION_UNLOCKED", mode="stream")
            reset_trace_context(trace_tokens)
            event_queue.put(_STREAM_DONE_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    def _get_next_event() -> object:
        try:
            return event_queue.get(timeout=_DISCONNECT_POLL_SECONDS)
        except queue.Empty:
            return None  # 타임아웃 — 아직 다음 이벤트가 없다(정상, disconnect 재확인용).

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            if await http_request.is_disconnected():
                break
            item = await loop.run_in_executor(None, _get_next_event)
            if item is None:
                continue
            if item is _STREAM_DONE_SENTINEL:
                break
            yield (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(event_generator(), media_type="application/x-ndjson; charset=utf-8")


@router.post("/{session_id}/continue-turn/stream")
async def continue_expert_turn_stream(session_id: str, request: ContinueTurnRequest, http_request: Request):
    """재인/Claude(2026-07-23, 아바타 페이싱 연동): 새 사용자 발언 없이, 지금 진행 중인
    라운드에서 다음 위원(기획/개발) 발언 딱 1건만 더 만들어 스트리밍한다 — POST
    /reply/stream과 같은 NDJSON/세션 락/취소 메커니즘을 그대로 재사용하되, 사용자 입력을
    반영하는 대신 continue_ideation_expert_turn(그래프의 기존 _route_next_expert_turn 결과를
    그대로 따르는 함수 — 회의 로직은 바뀌지 않는다)을 부른다.

    아바타가 방금 위원 발언을 재생하는 도중(재생 끝나기 3초 전) "다음 위원 미리 준비"
    신호로 호출하는 용도다. previous_state["phase"]가 "expert_discussion"이 아니면(라운드가
    이미 끝났거나 사용자 입력 대기 중) 400을 반환한다 — 프론트가 avatarPacingTimer를 잘못된
    시점에 걸었다는 뜻이므로 방어적으로 막는다."""
    _require_preview_enabled()
    _require_streaming_enabled()

    try:
        record = _acquire_session_record_or_404(session_id)
    except _SessionBusyError:
        raise HTTPException(status_code=409, detail="이 세션은 이미 다른 요청을 처리하고 있습니다.")
    previous_state = record.state

    if previous_state.get("phase") != "expert_discussion":
        _store.release(session_id)
        raise HTTPException(
            status_code=400,
            detail=f"continue-turn은 phase가 'expert_discussion'일 때만 호출할 수 있습니다(현재: {previous_state.get('phase')!r}).",
        )

    model = _effective_model(request.model)
    event_queue: "queue.Queue[object]" = queue.Queue()

    request_id = f"REQ-{uuid.uuid4().hex[:10]}"
    cancel_event = threading.Event()
    record.active_request_id = request_id
    record.cancel_event = cancel_event

    def sink(event: dict) -> None:
        if is_late_request_event(event.get("request_id"), request_id):
            return
        event.setdefault("request_id", request_id)
        event_queue.put(event)

    evidence_lookup = _evidence_lookup_for(
        record.use_rag,
        record.project_id,
        session_id=session_id,
        selected_candidate_document_id=previous_state.get("selected_idea_document_id"),
    )
    ground_claims = _ground_claims_for(record.use_rag)
    index_target_evidence = _index_target_evidence_for(record.use_rag, record.project_id)
    evidence_planner = _evidence_planner_for(record.use_rag)

    def worker() -> None:
        trace_tokens = bind_trace_context(session_id, request_id)
        try:
            stream_chat_completion, call_chat_completion = _build_streaming_backends(session_id, model)
            llm_call = make_streaming_llm_call(
                session_id,
                sink,
                stream_chat_completion=stream_chat_completion,
                call_chat_completion=call_chat_completion,
                max_calls=_MAX_LLM_CALLS_PER_REQUEST,
                cancel_event=cancel_event,
                request_id=request_id,
            )
            sink({"type": "request_started", "request_id": request_id})
            trace_event("IDEATION_REQUEST_STARTED", mode="continue_turn")
            state = continue_ideation_expert_turn(
                previous_state=previous_state,
                llm_call=llm_call,
                evidence_lookup=evidence_lookup,
                ground_claims=ground_claims,
                index_target_evidence=index_target_evidence,
                evidence_planner=evidence_planner,
            )
            _store.update(session_id, state)
            sink({"type": "state", "state": _serialize_state(state)})
            if state.get("phase") == "failed":
                failed_node = state.get("failed_node")
                sink(
                    {
                        "type": "error",
                        "code": "IDEATION_CONV_NODE_FAILED",
                        "message": f"{failed_node or '알 수 없는'} 노드에서 회의 처리가 실패했습니다.",
                        "failed_node": failed_node,
                    }
                )
        except IdeationCancelled as exc:
            logger.info("[%s] continue-turn 요청이 사용자에 의해 취소됨 request_id=%s", session_id, request_id)
            if exc.partial_state is not None:
                _store.update(session_id, exc.partial_state)
            sink({"type": "cancelled", "request_id": request_id})
        except ValueError as exc:
            sink({"type": "error", "code": "invalid_request", "message": str(exc)})
        except Exception:
            logger.exception("[ideation-conversation] continue-turn 처리 실패 session_id=%s", session_id)
            sink(
                {
                    "type": "error",
                    "code": "llm_failure",
                    "message": "답변 처리 중 오류가 발생했습니다. 서버 로그를 확인하세요.",
                }
            )
        finally:
            record.active_request_id = None
            record.cancel_event = None
            _store.release(session_id)
            trace_event("IDEATION_SESSION_UNLOCKED", mode="continue_turn")
            reset_trace_context(trace_tokens)
            event_queue.put(_STREAM_DONE_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    def _get_next_event() -> object:
        try:
            return event_queue.get(timeout=_DISCONNECT_POLL_SECONDS)
        except queue.Empty:
            return None

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            if await http_request.is_disconnected():
                break
            item = await loop.run_in_executor(None, _get_next_event)
            if item is None:
                continue
            if item is _STREAM_DONE_SENTINEL:
                break
            yield (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(event_generator(), media_type="application/x-ndjson; charset=utf-8")


@router.post("/start/stream")
async def start_conversation_stream(request: StartRequest, http_request: Request):
    """가은/Claude(2026-07-22, 요청: 회의 시작 대기 체감 개선 1단계): POST /start와 같은
    일(세션 생성 + 첫 그래프 실행)을 하지만, reply/stream과 같은 NDJSON 통로로 진행
    이벤트를 흘려보낸다. 후보 생성/실현 가능성 검토 호출은 화면 메시지를 만들지 않는
    호출이라(ideation_conversation_streaming.py::_PHASE_ONLY_LABELS) message_delta 없이
    phase 이벤트("아이디어 후보를 만들고 있습니다" -> "후보의 실현 가능성을 검토하고
    있습니다")만 나간다 — 기다리는 동안 "지금 뭐 하는 중인지"를 보여주는 것이 목적이다.
    refinement 모드(user_idea 있음)로 시작하면 라운드테이블 발언이 reply/stream과 동일하게
    message_delta(타이핑 효과)로 스트리밍된다 — 같은 make_streaming_llm_call을 그대로
    쓰기 때문에 추가 구현 없이 따라온다.

    reply/stream과의 차이: 새 세션이므로 세션 락(acquire/release)이 없고, state는 그래프가
    끝까지 성공했을 때만 _store.create로 저장한다. 스트리밍 중 연결이 끊겨도 워커는 계속
    실행돼 세션 자체는 정상 생성되지만, 클라이언트가 최종 state(session_id 포함)를 받지
    못했으면 그 세션은 사용되지 않고 TTL(30분)로 정리된다 — 프론트가 session_id를
    sessionStorage에 저장하는 시점이 최종 state 수신 시점이기 때문이다.

    라우트 경로 주의: FastAPI는 등록 순서대로 매칭하지만 "/start/stream"은 고정 경로라
    "/{session_id}/reply/stream" 같은 패턴과 충돌하지 않는다("/start"를 session_id로
    오인할 수 있는 경로는 GET /{session_id}뿐이고 그것은 GET, 이것은 POST다)."""
    _require_preview_enabled()
    _require_streaming_enabled()
    if request.use_rag and not request.project_id:
        raise HTTPException(status_code=400, detail="use_rag=true이면 project_id가 필요합니다.")

    competition_name = _clamp_text(request.competition_name, "competition_name")
    user_idea_text = _clamp_optional_text(request.user_idea, "user_idea")
    competition_document = (request.competition_document or "")[:_MAX_TEXT_LENGTH]
    effective_max_rounds = max(1, min(request.max_rounds, _MAX_ROUNDS_CAP))

    session_id = f"IDEA-CONV-{uuid.uuid4().hex[:8]}"
    notice_and_criteria = {"competition_name": competition_name, "notice_document": competition_document}
    user_idea = {"description": user_idea_text}
    model = _effective_model(request.model)
    # 용준/Claude(2026-07-22, 요청: 세션 범위 검색 / RAG 근거 유실 수정 2탄)의 비스트리밍
    # /start와 동일한 RAG 배선 — 가은/Claude(2026-07-22): 스트리밍 시작 경로만 이 배선이
    # 빠져 있으면 회의 시작을 스트리밍으로 하는 세션만 근거 품질이 조용히 떨어지는
    # 회귀가 되므로 그대로 맞춘다.
    evidence_lookup = _evidence_lookup_for(request.use_rag, request.project_id, session_id=session_id)
    ground_claims = _ground_claims_for(request.use_rag)
    index_target_evidence = _index_target_evidence_for(request.use_rag, request.project_id)
    evidence_planner = _evidence_planner_for(request.use_rag)

    logger.info("[ideation-conversation] 스트리밍 시작 session_id=%s max_rounds=%d", session_id, effective_max_rounds)

    event_queue: "queue.Queue[object]" = queue.Queue()

    def sink(event: dict) -> None:
        event_queue.put(event)

    def worker() -> None:
        try:
            stream_chat_completion, call_chat_completion = _build_streaming_backends(session_id, model)
            llm_call = make_streaming_llm_call(
                session_id,
                sink,
                stream_chat_completion=stream_chat_completion,
                call_chat_completion=call_chat_completion,
                max_calls=_MAX_LLM_CALLS_PER_REQUEST,
            )
            state = start_ideation_conversation(
                session_id=session_id,
                notice_and_criteria=notice_and_criteria,
                user_idea=user_idea,
                llm_call=llm_call,
                max_rounds=effective_max_rounds,
                evidence_lookup=evidence_lookup,
                ground_claims=ground_claims,
                index_target_evidence=index_target_evidence,
                evidence_planner=evidence_planner,
                application_form_items=request.application_form_items,
            )
            # 용준/Claude(2026-07-22, RAG 근거 유실 수정 2탄) 패턴과 동일 — 이후
            # /reply·/reply/stream이 evidence_lookup을 다시 만들 수 있도록 이번 세션의
            # use_rag/project_id를 함께 저장한다(스트리밍 시작 경로도 예외 없이).
            _store.create(state, use_rag=request.use_rag, project_id=request.project_id)
            sink({"type": "state", "state": _serialize_state(state)})
        except ValueError as exc:
            sink({"type": "error", "code": "invalid_request", "message": str(exc)})
        except Exception:
            logger.exception("[ideation-conversation] 스트리밍 시작 실패 session_id=%s", session_id)
            sink(
                {
                    "type": "error",
                    "code": "llm_failure",
                    "message": "대화형 회의 시작 중 오류가 발생했습니다. 서버 로그를 확인하세요.",
                }
            )
        finally:
            event_queue.put(_STREAM_DONE_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    def _get_next_event() -> object:
        try:
            return event_queue.get(timeout=_DISCONNECT_POLL_SECONDS)
        except queue.Empty:
            return None

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            if await http_request.is_disconnected():
                break
            item = await loop.run_in_executor(None, _get_next_event)
            if item is None:
                continue
            if item is _STREAM_DONE_SENTINEL:
                break
            yield (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(event_generator(), media_type="application/x-ndjson; charset=utf-8")


def _wait_for_release(record: "_SessionRecord", timeout: float) -> bool:
    """busy_lock을 짧게 획득해봄으로써(성공하면 즉시 반납) "지금 이 세션을 처리 중인 요청이
    남아있지 않은지"를 확인한다. 실제 처리 로직은 전혀 수행하지 않는다 — 오직 락 상태
    확인용이다."""
    acquired = record.busy_lock.acquire(timeout=timeout)
    if acquired:
        record.busy_lock.release()
    return acquired


@router.post("/{session_id}/cancel")
async def cancel_conversation(session_id: str, request: CancelRequest):
    """용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소) — 지금 이 세션에서 진행 중인
    스트리밍 요청을 취소한다. request.request_id가 주어지면 그 요청과 일치할 때만 신호를
    보내고(다른 request_id면 이미 끝난 요청이므로 조용히 무시), 없으면 "지금 활성 요청
    아무거나"를 취소한다(멱등 — 활성 요청이 이미 없어도 에러 없이 성공 처리한다).

    session_locked=false를 반환할 때까지(또는 타임아웃까지) 워커 스레드가 실제로 세션
    락을 반납하길 기다린다 — 프런트가 이 응답을 받은 뒤에만 다음 reply를 보내야
    세션 lock 409를 피할 수 있다(요청: "취소 완료 전에 새 reply를 보내 409가 발생하지
    않게")."""
    _require_preview_enabled()
    cancel_started = time.perf_counter()
    record = _store.get_record(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없거나 만료되었습니다.")

    cancel_event = record.cancel_event
    active_request_id = record.active_request_id
    trace_tokens = bind_trace_context(session_id, active_request_id)
    trace_event(
        "IDEATION_CANCEL_REQUESTED",
        supplied_request_id=request.request_id,
        active_request_id=active_request_id,
        phase=record.state.get("phase"),
        next_route=record.state.get("next_route"),
    )
    try:
        if cancel_event is not None and (request.request_id is None or request.request_id == active_request_id):
            cancel_event.set()
            trace_event("IDEATION_CANCEL_SIGNALLED")

        released = await run_in_threadpool(_wait_for_release, record, _CANCEL_CONFIRM_TIMEOUT_SECONDS)
        trace_event(
            "IDEATION_CANCEL_COMPLETED",
            session_locked=not released,
            lock_released=released,
            cancel_latency_ms=round((time.perf_counter() - cancel_started) * 1000, 1),
        )
        return {"cancelled": True, "session_locked": not released}
    finally:
        reset_trace_context(trace_tokens)


@router.post("/{session_id}/finalize")
async def finalize_conversation(session_id: str, request: FinalizeRequest):
    _require_preview_enabled()
    try:
        previous_state = _acquire_session_or_404(session_id)
    except _SessionBusyError:
        raise HTTPException(status_code=409, detail="이 세션은 이미 다른 요청을 처리하고 있습니다.")

    try:
        llm_call = _build_llm_call(session_id, _effective_model(request.model))

        try:
            state = await run_in_threadpool(
                finalize_ideation_conversation,
                previous_state=previous_state,
                llm_call=llm_call,
            )
        except ValueError as exc:
            # phase != awaiting_user_decision — 요청 9~10항: 사용자가 확정 버튼을 눌러도
            # 아직 회의가 그 단계에 도달하지 않았으면 최종 종합을 거부한다.
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:
            logger.exception("[ideation-conversation] 최종 확정 실패 session_id=%s", session_id)
            raise HTTPException(status_code=502, detail="최종 확정 중 오류가 발생했습니다. 서버 로그를 확인하세요.")

        _store.update(session_id, state)
        return _serialize_state(state)
    finally:
        _store.release(session_id)


@router.get("/_debug/sessions")
async def _debug_list_sessions():
    """재인/Claude(2026-07-23) — 임시 디버그 전용 라우트. 실측 중복 메시지/멈춤 버그를
    브라우저 devtools 없이 서버 쪽에서 직접 확인하려고 추가했다. 조사 끝나면 지운다."""
    _require_preview_enabled()
    with _store._lock:
        items = sorted(_store._sessions.items(), key=lambda kv: kv[1].last_active_at, reverse=True)
        out = []
        for sid, rec in items[:10]:
            msgs = rec.state.get("messages", [])
            out.append(
                {
                    "session_id": sid,
                    "phase": rec.state.get("phase"),
                    "last_active_at": rec.last_active_at,
                    "message_count": len(msgs),
                    "forced_next_speaker": rec.state.get("forced_next_speaker"),
                    "active_request_id": rec.active_request_id,
                    "messages": [
                        {"message_id": m["message_id"], "speaker_id": m["speaker_id"], "content": (m.get("content") or "")[:60]}
                        for m in msgs
                    ],
                }
            )
        return out


@router.get("/{session_id}")
async def get_conversation(session_id: str):
    _require_preview_enabled()
    try:
        state = _store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없거나 만료되었습니다.")
    return _serialize_state(state)
