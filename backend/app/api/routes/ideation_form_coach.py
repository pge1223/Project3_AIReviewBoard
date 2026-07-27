from __future__ import annotations

import copy
import logging
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ai.meeting.form_coach import (
    LLMAnswerAssessmentProvider,
    LLMExpertDelegationProvider,
    LLMExpertGuidanceProvider,
    LLMExpertReviewProvider,
    LLMSynthesisProvider,
    LookupEvidenceProvider,
    NullEvidenceProvider,
    finalize_session,
    reply_to_session,
    start_session,
    update_application_draft,
)
from app.config import settings
from app.core.llm import trace_openai_client


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ideation-form-coach", tags=["ideation-form-coach-v2"])

_SESSION_TTL_SECONDS = 30 * 60
_MAX_SESSIONS = 200
_MAX_TEXT_LENGTH = 4000


class _Record:
    def __init__(self, state: dict):
        self.state = state
        self.last_active_at = time.time()
        self.lock = threading.Lock()


class _Store:
    def __init__(self):
        self._records: dict[str, _Record] = {}
        self._lock = threading.Lock()

    def _sweep(self) -> None:
        now = time.time()
        expired = [sid for sid, record in self._records.items() if now - record.last_active_at > _SESSION_TTL_SECONDS]
        for sid in expired:
            del self._records[sid]
        while len(self._records) >= _MAX_SESSIONS:
            oldest = min(self._records, key=lambda sid: self._records[sid].last_active_at)
            del self._records[oldest]

    def create(self, state: dict) -> None:
        with self._lock:
            self._sweep()
            self._records[state["session_id"]] = _Record(state)

    def acquire(self, session_id: str) -> _Record:
        with self._lock:
            self._sweep()
            record = self._records.get(session_id)
        if record is None:
            raise KeyError(session_id)
        if not record.lock.acquire(blocking=False):
            raise RuntimeError("busy")
        return record

    def release(self, record: _Record) -> None:
        record.last_active_at = time.time()
        record.lock.release()

    def get(self, session_id: str) -> dict:
        with self._lock:
            self._sweep()
            record = self._records.get(session_id)
            if record is None:
                raise KeyError(session_id)
            record.last_active_at = time.time()
            return copy.deepcopy(record.state)


_store = _Store()


class StartRequest(BaseModel):
    competition_name: str
    competition_document: str = ""
    selected_idea: dict
    idea_canvas: dict = Field(default_factory=dict)
    application_form_items: list[dict]
    legacy_context: dict = Field(default_factory=dict)
    project_id: str | None = None
    use_rag: bool = False
    model: str = ""


class ReplyRequest(BaseModel):
    message: str
    model: str = ""


class FinalizeRequest(BaseModel):
    model: str = ""


class DraftUpdateRequest(BaseModel):
    value: str = ""


def _require_enabled() -> None:
    if not settings.ENABLE_IDEATION_PREVIEW or not settings.ENABLE_FORM_COACH_V2:
        raise HTTPException(status_code=404, detail="Not Found")


def _text(value: str, field_name: str, *, required: bool = True) -> str:
    normalized = (value or "").strip()
    if required and not normalized:
        raise HTTPException(status_code=400, detail=f"{field_name}은(는) 비어 있을 수 없습니다.")
    if len(normalized) > _MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"{field_name}은(는) 최대 {_MAX_TEXT_LENGTH}자까지 입력할 수 있습니다.")
    return normalized


def _build_llm_call(session_id: str, requested_model: str, *, max_calls: int = 1):
    model = requested_model.strip() or settings.reviewer_model()
    client = trace_openai_client(OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=1))
    call_count = 0

    def call(prompt: str) -> str:
        nonlocal call_count
        if call_count >= max_calls:
            raise RuntimeError("form_coach_v2는 요청당 LLM을 한 번만 호출할 수 있습니다.")
        call_count += 1
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    return call


def _serialize(state: dict) -> dict:
    return copy.deepcopy(state)


def _evidence_provider_for(
    *,
    use_rag: bool,
    project_id: str | None,
    session_id: str,
    selected_candidate_document_id: str | None,
):
    if not use_rag or not project_id:
        return NullEvidenceProvider()
    from ai.rag.orchestration.ideation_evidence_service import make_ideation_evidence_lookup
    from app.api.routes.meetings import _role_retrieval_service

    lookup = make_ideation_evidence_lookup(
        project_id=project_id,
        role_retrieval_service=_role_retrieval_service,
        top_k=5,
        session_id=session_id,
        selected_candidate_document_id=selected_candidate_document_id,
    )
    return LookupEvidenceProvider(
        lookup,
        session_id=session_id,
        selected_candidate_document_id=selected_candidate_document_id,
    )


def _acquire(session_id: str) -> _Record:
    try:
        return _store.acquire(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없거나 만료되었습니다.")
    except RuntimeError:
        raise HTTPException(status_code=409, detail="이 세션은 이미 다른 요청을 처리하고 있습니다.")


@router.post("/start")
async def start_form_coach(request: StartRequest):
    _require_enabled()
    competition_name = _text(request.competition_name, "competition_name")
    if not request.selected_idea:
        raise HTTPException(status_code=400, detail="selected_idea가 필요합니다.")
    if not request.application_form_items:
        raise HTTPException(status_code=400, detail="application_form_items가 필요합니다.")
    session_id = f"FORM-COACH-{uuid.uuid4().hex[:8]}"
    try:
        state = await run_in_threadpool(
            start_session,
            session_id=session_id,
            competition_name=competition_name,
            competition_document=_text(request.competition_document, "competition_document", required=False),
            selected_idea=request.selected_idea,
            idea_canvas=request.idea_canvas,
            application_form_items=request.application_form_items,
            legacy_context=request.legacy_context,
            project_id=request.project_id,
            use_rag=request.use_rag,
            llm_call=_build_llm_call(session_id, request.model),
            answer_assessment_provider=LLMAnswerAssessmentProvider(
                _build_llm_call(f"{session_id}:assessment", request.model)
            ),
            evidence_provider=_evidence_provider_for(
                use_rag=request.use_rag,
                project_id=request.project_id,
                session_id=session_id,
                selected_candidate_document_id=request.legacy_context.get("selected_idea_document_id"),
            ),
            expert_review_provider=LLMExpertReviewProvider(
                _build_llm_call(f"{session_id}:expert", request.model)
            ),
            expert_guidance_provider=LLMExpertGuidanceProvider(
                _build_llm_call(f"{session_id}:guidance", request.model)
            ),
            expert_delegation_provider=LLMExpertDelegationProvider(
                _build_llm_call(
                    f"{session_id}:delegation",
                    request.model,
                    max_calls=4,
                )
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("[form-coach-v2] 시작 실패 session_id=%s", session_id)
        raise HTTPException(status_code=502, detail="신청 양식 코치 시작 중 오류가 발생했습니다.")
    _store.create(state)
    return _serialize(state)


@router.post("/{session_id}/reply")
async def reply_form_coach(session_id: str, request: ReplyRequest):
    _require_enabled()
    record = _acquire(session_id)
    try:
        answer = _text(request.message, "message")
        record.state = await run_in_threadpool(
            reply_to_session,
            record.state,
            answer=answer,
            llm_call=_build_llm_call(session_id, request.model),
            answer_assessment_provider=LLMAnswerAssessmentProvider(
                _build_llm_call(f"{session_id}:assessment", request.model)
            ),
            evidence_provider=_evidence_provider_for(
                use_rag=bool(record.state.get("use_rag")),
                project_id=record.state.get("project_id"),
                session_id=session_id,
                selected_candidate_document_id=record.state.get("selected_idea_document_id"),
            ),
            expert_review_provider=LLMExpertReviewProvider(
                _build_llm_call(f"{session_id}:expert", request.model)
            ),
            expert_guidance_provider=LLMExpertGuidanceProvider(
                _build_llm_call(f"{session_id}:guidance", request.model)
            ),
            expert_delegation_provider=LLMExpertDelegationProvider(
                _build_llm_call(
                    f"{session_id}:delegation",
                    request.model,
                    max_calls=4,
                )
            ),
        )
        return _serialize(record.state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception:
        logger.exception("[form-coach-v2] 답변 처리 실패 session_id=%s", session_id)
        raise HTTPException(status_code=502, detail="신청 양식 답변 처리 중 오류가 발생했습니다.")
    finally:
        _store.release(record)


@router.post("/{session_id}/finalize")
async def finalize_form_coach(session_id: str, request: FinalizeRequest):
    _require_enabled()
    record = _acquire(session_id)
    try:
        record.state = await run_in_threadpool(
            finalize_session,
            record.state,
            synthesis_provider=LLMSynthesisProvider(
                _build_llm_call(f"{session_id}:synthesis", request.model)
            ),
        )
        return _serialize(record.state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("[form-coach-v2] synthesis failed session_id=%s", session_id)
        raise HTTPException(status_code=502, detail="최종 제안서 정리 중 오류가 발생했습니다.")
    finally:
        _store.release(record)


@router.patch("/{session_id}/draft/{field_id}")
async def update_form_coach_draft(
    session_id: str,
    field_id: str,
    request: DraftUpdateRequest,
):
    _require_enabled()
    record = _acquire(session_id)
    try:
        record.state = update_application_draft(
            record.state,
            field_id=field_id,
            value=_text(request.value, "value", required=False),
        )
        return _serialize(record.state)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        _store.release(record)


@router.get("/{session_id}")
async def get_form_coach(session_id: str):
    _require_enabled()
    try:
        return _store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없거나 만료되었습니다.")
