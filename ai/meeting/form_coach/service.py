from __future__ import annotations

import json
import uuid
from typing import Callable

from .answer_assessment import AnswerAssessmentProvider, NullAnswerAssessmentProvider
from .draft_mapper import incremental_draft_patches, provisional_phase_patch
from .evidence_provider import EvidenceProvider, NullEvidenceProvider
from .expert_delegation import (
    ExpertDelegationProvider,
    NullExpertDelegationProvider,
    counterpart_review_required,
    owner_for_phase,
)
from .expert_review import ExpertReviewProvider, NullExpertReviewProvider
from .expert_guidance import ExpertGuidanceProvider, NullExpertGuidanceProvider
from .phase_resolver import connected_fields, first_phase, next_phase, phase_complete
from .prompt_builder import build_prompt
from .response_validator import validate_response
from .state import make_message, new_session, now_iso
from .synthesis import NullSynthesisProvider, SynthesisProvider


LLMCall = Callable[[str], str]

_DEFAULT_EXPERT_TYPES = {
    "problem_definition": ["planning"],
    "target_and_context": ["planning"],
    "purpose_and_value": ["planning"],
    "solution_scenario": ["planning"],
    "ai_role_and_mvp": ["development"],
    "effect_and_validation": ["planning"],
    "differentiation_and_risk": ["planning", "development"],
}
_DELEGATION_TERMS = (
    "잘 모르",
    "모르겠",
    "전문가가 추천",
    "전문가 추천",
    "전문가에게 맡",
    "추천해 주세요",
    "추천해주세요",
)
_DELEGATION_ACCEPT_TERMS = {"권고안 적용", "이대로 적용", "적용할게요", "네", "예", "좋아요"}
_DELEGATION_REJECT_TERMS = {"다시 직접 선택", "직접 선택", "다시 선택", "아니요"}


def _latest_facilitator_question(state: dict) -> str:
    for message in reversed(state.get("messages") or []):
        if message.get("speaker_id") != "ideation_facilitator":
            continue
        structured = message.get("structured")
        if not isinstance(structured, dict):
            continue
        ui_message = structured.get("ui_message")
        if isinstance(ui_message, dict):
            return str(ui_message.get("question") or "").strip()
    return ""


def _apply_state_update(state: dict, update: dict) -> None:
    confirmed_state = state.setdefault("confirmed_state", {})
    for key, value in update.items():
        if str(value).strip():
            confirmed_state[key] = str(value).strip()


def _apply_candidate_state_update(state: dict, update: dict) -> None:
    candidate_state = state.setdefault("candidate_state", {})
    for key, value in update.items():
        if str(value).strip():
            candidate_state[key] = str(value).strip()


def _is_delegation_request(answer: str, assessment: dict) -> bool:
    return bool(
        assessment.get("delegation_requested")
        or assessment.get("classification") == "expert_delegation"
        or any(term in answer for term in _DELEGATION_TERMS)
    )


def _delegation_state_update(phase: str, question: str, value: str) -> dict:
    if phase == "problem_definition":
        return {"problem": value}
    if phase == "target_and_context":
        if any(term in question for term in ("도입", "운영 주체", "기관")):
            return {"operator": value}
        if any(term in question for term in ("현장", "장소", "상황")):
            return {"context": value}
        return {"target_user": value, "target": value}
    if phase == "purpose_and_value":
        return {"value": value, "purpose": value}
    if phase == "solution_scenario":
        return {"solution": value}
    if phase == "ai_role_and_mvp":
        if "데이터" in question:
            return {"required_data": value}
        if any(term in question for term in ("MVP", "첫 버전", "구현 범위")):
            return {"mvp_scope": value}
        return {"ai_role": value}
    if phase == "effect_and_validation":
        if any(term in question for term in ("검증", "측정")):
            return {"validation_method": value}
        return {"expected_effect": value}
    if phase == "differentiation_and_risk":
        if any(term in question for term in ("리스크", "위험", "제약")):
            return {"risk": value}
        return {"differentiation": value}
    return {}


def _delegation_message(
    *,
    state: dict,
    phase: str,
    fields: list[dict],
    result: dict,
    evidence: list[dict],
) -> dict:
    proposal = result.get("proposal") or {}
    review = result.get("review") or {}
    facilitator = result.get("facilitator") or {}
    owner_type = str(result.get("owner_expert_type") or "planning")
    counterpart_type = str(result.get("counterpart_expert_type") or "development")
    mentors = []
    if str(proposal.get("spoken_text") or "").strip():
        mentors.append(
            {
                "expert_type": owner_type,
                "name": "기획 전문가" if owner_type == "planning" else "개발 전문가",
                "text": str(proposal["spoken_text"]).strip(),
            }
        )
    if str(review.get("spoken_text") or "").strip():
        mentors.append(
            {
                "expert_type": counterpart_type,
                "name": "기획 전문가" if counterpart_type == "planning" else "개발 전문가",
                "text": str(review["spoken_text"]).strip(),
            }
        )
    recommendation = str(
        facilitator.get("final_recommendation")
        or result.get("suggested_value")
        or ""
    ).strip()
    facilitator_text = str(facilitator.get("spoken_text") or "").strip()
    question = "이 권고안을 현재 항목의 임시 초안으로 적용할까요?"
    choices = [
        {
            "id": "apply_delegation",
            "label": "권고안 적용",
            "description": "임시 권고안을 확정하고 다음 단계로 이동",
            "value": "권고안 적용",
        },
        {
            "id": "reject_delegation",
            "label": "다시 직접 선택",
            "description": "권고안을 적용하지 않고 현재 질문으로 돌아가기",
            "value": "다시 직접 선택",
        },
    ]
    structured = {
        "internal_state": {
            "phase": phase,
            "current_form_fields": [row.get("field_id") for row in fields],
            "decision_reason": "사용자가 현재 결정을 전문가에게 위임함",
            "delegation_status": "awaiting_confirmation",
        },
        "ui_message": {
            "facilitator_text": facilitator_text or recommendation,
            "post_answer_mentor_messages": mentors[:2],
            "question": question,
            "choices": choices,
        },
        "choices": choices,
        "current_field_ids": [row.get("field_id") for row in fields],
        "draft_patch": [],
        "answer_assessment": {
            "classification": "expert_delegation",
            "delegation_requested": True,
        },
        "next_action": "confirm_delegation",
    }
    message = make_message(
        speaker_id="ideation_facilitator",
        speaker_name="진행자",
        role="진행자",
        message_type="summary",
        content=facilitator_text or recommendation,
        structured=structured,
    )
    message["evidence"] = evidence
    return message


def _run_delegation(
    state: dict,
    *,
    phase: str,
    fields: list[dict],
    pending_question: str,
    evidence_provider: EvidenceProvider,
    delegation_provider: ExpertDelegationProvider,
) -> dict:
    owner_type, _ = owner_for_phase(phase)
    counterpart_type = "development" if owner_type == "planning" else "planning"
    query = " ".join(
        (
            str(state.get("competition_name") or ""),
            phase,
            pending_question,
            json.dumps(state.get("confirmed_state") or {}, ensure_ascii=False),
        )
    )
    evidence_by_expert = {
        owner_type: evidence_provider.retrieve(
            query=query,
            context=state,
            expert_types=[owner_type],
        )
    }
    if counterpart_review_required(phase):
        evidence_by_expert[counterpart_type] = evidence_provider.retrieve(
            query=query,
            context=state,
            expert_types=[counterpart_type],
        )
    result = delegation_provider.delegate(
        phase=phase,
        pending_question=pending_question,
        state=state,
        evidence_by_expert=evidence_by_expert,
    )
    suggested_value = str(result.get("suggested_value") or "").strip()
    if not suggested_value:
        raise ValueError("전문가가 적용 가능한 권고안을 생성하지 못했습니다.")
    state_update = _delegation_state_update(phase, pending_question, suggested_value)
    patches = provisional_phase_patch(
        state=state,
        phase=phase,
        value=suggested_value,
        preferred_field_ids=[str(row.get("field_id")) for row in fields],
    )
    _apply_application_patches(state, patches)
    state.setdefault("candidate_state", {})["delegation_candidate"] = suggested_value
    state["pending_delegation"] = {
        "phase": phase,
        "question": pending_question,
        "suggested_value": suggested_value,
        "state_update": state_update,
        "draft_field_ids": [patch.get("field_id") for patch in patches],
        "result": result,
    }
    message = _delegation_message(
        state=state,
        phase=phase,
        fields=fields,
        result=result,
        evidence=[
            *evidence_by_expert.get(owner_type, []),
            *evidence_by_expert.get(counterpart_type, []),
        ],
    )
    message["structured"]["draft_patch"] = patches
    message["structured"]["internal_state"]["draft_patch"] = patches
    state["messages"].append(message)
    state["phase"] = "awaiting_delegation_confirmation"
    state["updated_at"] = now_iso()
    return state


def _sync_idea_canvas(state: dict) -> None:
    selected = state.get("selected_idea") or {}
    canvas = state.setdefault("idea_canvas", {})
    for key, value in selected.items():
        if key not in canvas or canvas.get(key) in (None, "", []):
            canvas[key] = value

    confirmed = state.get("confirmed_state") or {}
    confirmed_mapping = {
        "problem": confirmed.get("problem"),
        "target_user": confirmed.get("target_user") or confirmed.get("target"),
        "core_value": confirmed.get("value") or confirmed.get("purpose"),
        "solution": confirmed.get("solution"),
        "differentiation": confirmed.get("differentiation"),
    }
    for key, value in confirmed_mapping.items():
        if str(value or "").strip():
            canvas[key] = str(value).strip()

    risk = confirmed.get("risk")
    if isinstance(risk, list):
        canvas["risks"] = [str(item).strip() for item in risk if str(item).strip()]
    elif str(risk or "").strip():
        canvas["risks"] = [str(risk).strip()]


def _apply_application_patches(state: dict, patches: list[dict]) -> None:
    by_id = {row["field_id"]: row for row in state.get("application_form_draft", [])}
    for patch in patches:
        row = by_id.get(patch.get("field_id"))
        if row is None:
            continue
        row["value"] = str(patch.get("value") or "")
        row["provisional_value"] = str(patch.get("provisional_value") or "")
        row["source"] = patch.get("source") or "conversation"
        row["status"] = patch.get("status") or "draft"
        row["confidence"] = patch.get("confidence") or "medium"
        row["editable"] = patch.get("editable") is not False


def _pre_answer_mentor_messages(
    *,
    guidance: dict,
    mentor_types: list[str],
) -> list[dict]:
    messages = []
    for expert_type in mentor_types:
        text = str(guidance.get(expert_type) or "").strip()
        if not text:
            continue
        messages.append(
            {
                "expert_type": expert_type,
                "name": "기획 전문가" if expert_type == "planning" else "개발 전문가",
                "text": text[:300],
            }
        )
    return messages[:2]


def _coach_turn(
    state: dict,
    *,
    llm_call: LLMCall,
    answer_assessment_provider: AnswerAssessmentProvider,
    evidence_provider: EvidenceProvider,
    expert_review_provider: ExpertReviewProvider,
    expert_guidance_provider: ExpertGuidanceProvider,
    expert_delegation_provider: ExpertDelegationProvider,
    latest_user_answer: str,
) -> dict:
    current_phase = str(state.get("current_phase") or first_phase())
    fields = connected_fields(state["application_form_draft"], current_phase)
    if not fields:
        raise ValueError("대화에 사용할 신청 양식 항목을 찾을 수 없습니다.")

    previous_question = _latest_facilitator_question(state)
    answer_assessment = (
        answer_assessment_provider.assess(
            phase=current_phase,
            current_field=fields[0],
            state=state,
            previous_question=previous_question,
            latest_user_answer=latest_user_answer,
        )
        if latest_user_answer.strip()
        else {}
    )
    if latest_user_answer.strip() and _is_delegation_request(
        latest_user_answer,
        answer_assessment,
    ):
        return _run_delegation(
            state,
            phase=current_phase,
            fields=fields,
            pending_question=previous_question,
            evidence_provider=evidence_provider,
            delegation_provider=expert_delegation_provider,
        )
    mentor_types = [
        str(value)
        for value in answer_assessment.get("mentor_types") or []
        if str(value) in {"planning", "development"}
    ]
    needs_expert_review = bool(
        latest_user_answer.strip()
        and answer_assessment.get("mentor_needed")
    )
    if needs_expert_review and not mentor_types:
        mentor_types = list(_DEFAULT_EXPERT_TYPES.get(current_phase, ["planning"]))

    query = " ".join(
        filter(
            None,
            (
                state.get("competition_name"),
                current_phase,
                json.dumps(state.get("selected_idea") or {}, ensure_ascii=False),
                json.dumps(state.get("idea_canvas") or {}, ensure_ascii=False),
                " ".join(str(row.get("field_name") or "") for row in fields),
                latest_user_answer,
            ),
        )
    )
    evidence = (
        evidence_provider.retrieve(
            query=query,
            context=state,
            expert_types=mentor_types,
        )
        if needs_expert_review
        else []
    )
    expert_review = {"planning": "", "development": ""}
    if needs_expert_review:
        expert_review = expert_review_provider.review(
            phase=current_phase,
            current_field=fields[0],
            state=state,
            latest_user_answer=latest_user_answer,
            mentor_types=mentor_types,
            evidence=evidence,
            mentor_reason=str(answer_assessment.get("mentor_reason") or ""),
        )

    # The facilitator runs last so its reflection and next question can use the
    # authoritative assessment plus any expert/RAG result.
    prompt = build_prompt(
        state=state,
        current_field=fields[0],
        current_phase=current_phase,
        previous_phase=None,
        previous_field=None,
        latest_user_answer=latest_user_answer,
        evidence=evidence,
        expert_review=expert_review,
        answer_assessment=answer_assessment,
    )
    raw_response = llm_call(prompt)
    message_data, state_update, form_patches, resulting_phase, next_action = validate_response(
        raw_response,
        current_phase=current_phase,
        confirmed_state=state["confirmed_state"],
        connected_form_fields=fields,
        latest_user_answer=latest_user_answer,
        expert_review=expert_review,
        answer_assessment_override=answer_assessment or None,
        application_form_draft=state["application_form_draft"],
        previous_question=previous_question,
    )
    _apply_state_update(state, state_update)
    _apply_candidate_state_update(
        state,
        message_data["structured"].get("candidate_state_update") or {},
    )
    _sync_idea_canvas(state)
    guidance_request = (
        message_data["structured"]["internal_state"].get("pre_answer_guidance")
        or {}
    )
    pre_answer_evidence = []
    pre_answer_messages = []
    if guidance_request.get("needed"):
        guidance_types = [
            str(value)
            for value in guidance_request.get("mentor_types") or []
            if str(value) in {"planning", "development"}
        ]
        ui_message = message_data["structured"]["ui_message"]
        guidance_query = " ".join(
            filter(
                None,
                (
                    state.get("competition_name"),
                    resulting_phase,
                    ui_message.get("question"),
                    " ".join(
                        str(choice.get("label") or "")
                        for choice in ui_message.get("choices") or []
                    ),
                ),
            )
        )
        pre_answer_evidence = evidence_provider.retrieve(
            query=guidance_query,
            context=state,
            expert_types=guidance_types,
        )
        guidance = expert_guidance_provider.guide(
            phase=resulting_phase,
            state=state,
            facilitator_text=str(ui_message.get("facilitator_text") or ""),
            question=str(ui_message.get("question") or ""),
            choices=ui_message.get("choices") or [],
            mentor_types=guidance_types,
            guidance_reason=str(guidance_request.get("reason") or ""),
            evidence=pre_answer_evidence,
        )
        pre_answer_messages = _pre_answer_mentor_messages(
            guidance=guidance,
            mentor_types=guidance_types,
        )
    message_data["structured"]["ui_message"][
        "pre_answer_mentor_messages"
    ] = pre_answer_messages
    message_data["structured"]["pre_answer_mentor_messages"] = pre_answer_messages
    _apply_application_patches(state, form_patches)
    fallback_patches = incremental_draft_patches(
        state=state,
        phase=current_phase,
        state_update=state_update,
        candidate_state_update=message_data["structured"].get("candidate_state_update") or {},
        preferred_field_ids=message_data["structured"].get("current_field_ids") or [],
        skip_field_ids={str(patch.get("field_id")) for patch in form_patches},
    )
    _apply_application_patches(state, fallback_patches)
    all_form_patches = [*form_patches, *fallback_patches]
    message_data["structured"]["draft_patch"] = all_form_patches
    message_data["structured"]["internal_state"]["draft_patch"] = all_form_patches
    state["messages"].append(
        make_message(
            speaker_id="ideation_facilitator",
            speaker_name="진행자",
            role="진행자",
            message_type="summary",
            content=message_data["content"],
            structured=message_data["structured"],
        )
    )
    state["messages"][-1]["evidence"] = [*evidence, *pre_answer_evidence]
    state["current_phase"] = resulting_phase
    current_ids = message_data["structured"]["current_field_ids"]
    state["current_field_id"] = current_ids[0] if current_ids else None
    state["round"] += 1
    state["updated_at"] = now_iso()

    if current_phase == "application_drafting" and latest_user_answer and next_action == "generate_application_draft":
        state["phase"] = "discussion_complete"
        state["current_phase"] = "complete"
        state["current_field_id"] = None
    else:
        state["phase"] = "awaiting_form_answer"
    return state


def update_application_draft(state: dict, *, field_id: str, value: str) -> dict:
    row = next(
        (
            item
            for item in state.get("application_form_draft") or []
            if str(item.get("field_id")) == str(field_id)
        ),
        None,
    )
    if row is None:
        raise ValueError("수정할 신청서 항목을 찾을 수 없습니다.")
    normalized = str(value or "").strip()
    char_limit = row.get("char_limit")
    if isinstance(char_limit, int) and char_limit > 0:
        normalized = normalized[:char_limit]
    row["value"] = normalized
    row["provisional_value"] = ""
    row["source"] = "user" if normalized else None
    row["status"] = "draft" if normalized else "in_progress"
    row["confidence"] = "high" if normalized else None
    row["editable"] = True
    state["updated_at"] = now_iso()
    return state


def start_session(
    *,
    competition_name: str,
    competition_document: str,
    selected_idea: dict,
    idea_canvas: dict | None,
    application_form_items: list[dict],
    llm_call: LLMCall,
    answer_assessment_provider: AnswerAssessmentProvider | None = None,
    evidence_provider: EvidenceProvider | None = None,
    expert_review_provider: ExpertReviewProvider | None = None,
    expert_guidance_provider: ExpertGuidanceProvider | None = None,
    expert_delegation_provider: ExpertDelegationProvider | None = None,
    legacy_context: dict | None = None,
    session_id: str | None = None,
    project_id: str | None = None,
    use_rag: bool = False,
) -> dict:
    state = new_session(
        session_id=session_id or f"FORM-COACH-{uuid.uuid4().hex[:8]}",
        competition_name=competition_name,
        competition_document=competition_document,
        selected_idea=selected_idea,
        idea_canvas=idea_canvas,
        application_form_items=application_form_items,
        legacy_context=legacy_context,
        project_id=project_id,
        use_rag=use_rag,
    )
    if not state["application_form_draft"]:
        raise ValueError("작성할 신청 양식 항목이 없습니다.")
    return _coach_turn(
        state,
        llm_call=llm_call,
        answer_assessment_provider=answer_assessment_provider or NullAnswerAssessmentProvider(),
        evidence_provider=evidence_provider or NullEvidenceProvider(),
        expert_review_provider=expert_review_provider or NullExpertReviewProvider(),
        expert_guidance_provider=expert_guidance_provider or NullExpertGuidanceProvider(),
        expert_delegation_provider=expert_delegation_provider or NullExpertDelegationProvider(),
        latest_user_answer="",
    )


def reply_to_session(
    state: dict,
    *,
    answer: str,
    llm_call: LLMCall,
    answer_assessment_provider: AnswerAssessmentProvider | None = None,
    evidence_provider: EvidenceProvider | None = None,
    expert_review_provider: ExpertReviewProvider | None = None,
    expert_guidance_provider: ExpertGuidanceProvider | None = None,
    expert_delegation_provider: ExpertDelegationProvider | None = None,
) -> dict:
    if state.get("phase") not in {"awaiting_form_answer", "awaiting_delegation_confirmation"}:
        raise ValueError("현재 사용자 답변을 받을 수 없는 세션입니다.")
    answer = answer.strip()
    if not answer:
        raise ValueError("답변은 비어 있을 수 없습니다.")
    state["messages"].append(
        make_message(
            speaker_id="user",
            speaker_name="사용자",
            role="사용자",
            message_type="answer",
            content=answer,
        )
    )
    if state.get("phase") == "awaiting_delegation_confirmation":
        pending = state.get("pending_delegation") or {}
        if answer in _DELEGATION_ACCEPT_TERMS:
            delegation_phase = str(
                pending.get("phase") or state.get("current_phase") or first_phase()
            )
            state_update = pending.get("state_update") or {}
            _apply_state_update(state, state_update)
            state.setdefault("candidate_state", {})["delegation_candidate"] = ""
            state["pending_delegation"] = None
            _sync_idea_canvas(state)
            patches = incremental_draft_patches(
                state=state,
                phase=delegation_phase,
                state_update=state_update,
                candidate_state_update={},
                preferred_field_ids=[
                    str(field_id) for field_id in pending.get("draft_field_ids") or []
                ],
                skip_field_ids=set(),
            )
            _apply_application_patches(state, patches)
            resulting_phase = (
                next_phase(delegation_phase)
                if phase_complete(delegation_phase, state.get("confirmed_state") or {})
                else delegation_phase
            )
            state["current_phase"] = resulting_phase or "application_drafting"
            state["phase"] = "awaiting_form_answer"
            return _coach_turn(
                state,
                llm_call=llm_call,
                answer_assessment_provider=answer_assessment_provider or NullAnswerAssessmentProvider(),
                evidence_provider=evidence_provider or NullEvidenceProvider(),
                expert_review_provider=expert_review_provider or NullExpertReviewProvider(),
                expert_guidance_provider=expert_guidance_provider or NullExpertGuidanceProvider(),
                expert_delegation_provider=expert_delegation_provider or NullExpertDelegationProvider(),
                latest_user_answer="",
            )
        if answer in _DELEGATION_REJECT_TERMS:
            delegated_field_ids = {
                str(field_id) for field_id in pending.get("draft_field_ids") or []
            }
            for row in state.get("application_form_draft") or []:
                if (
                    str(row.get("field_id")) in delegated_field_ids
                    and row.get("source") == "expert_delegation"
                ):
                    row["provisional_value"] = ""
                    row["source"] = None
                    row["confidence"] = None
            state.setdefault("candidate_state", {})["delegation_candidate"] = ""
            state["pending_delegation"] = None
            state["phase"] = "awaiting_form_answer"
            return _coach_turn(
                state,
                llm_call=llm_call,
                answer_assessment_provider=answer_assessment_provider or NullAnswerAssessmentProvider(),
                evidence_provider=evidence_provider or NullEvidenceProvider(),
                expert_review_provider=expert_review_provider or NullExpertReviewProvider(),
                expert_guidance_provider=expert_guidance_provider or NullExpertGuidanceProvider(),
                expert_delegation_provider=expert_delegation_provider or NullExpertDelegationProvider(),
                latest_user_answer="",
            )
        delegated_field_ids = {
            str(field_id) for field_id in pending.get("draft_field_ids") or []
        }
        for row in state.get("application_form_draft") or []:
            if (
                str(row.get("field_id")) in delegated_field_ids
                and row.get("source") == "expert_delegation"
            ):
                row["provisional_value"] = ""
                row["source"] = None
                row["confidence"] = None
        state.setdefault("candidate_state", {})["delegation_candidate"] = ""
        state["pending_delegation"] = None
        state["phase"] = "awaiting_form_answer"
    return _coach_turn(
        state,
        llm_call=llm_call,
        answer_assessment_provider=answer_assessment_provider or NullAnswerAssessmentProvider(),
        evidence_provider=evidence_provider or NullEvidenceProvider(),
        expert_review_provider=expert_review_provider or NullExpertReviewProvider(),
        expert_guidance_provider=expert_guidance_provider or NullExpertGuidanceProvider(),
        expert_delegation_provider=expert_delegation_provider or NullExpertDelegationProvider(),
        latest_user_answer=answer,
    )


def _base_proposal(state: dict) -> dict:
    if state.get("phase") != "discussion_complete":
        raise ValueError("현재 완료할 수 없는 세션입니다.")
    selected = state.get("selected_idea") or {}
    confirmed = state.get("confirmed_state") or {}
    values = {
        str(row.get("field_name")): row.get("value")
        for row in state.get("application_form_draft", [])
        if row.get("value")
    }
    return {
        "idea_name": confirmed.get("selected_idea") or selected.get("title") or "확정된 아이디어",
        "problem_definition": confirmed.get("problem") or "",
        "target_user": confirmed.get("target_user") or confirmed.get("target") or "",
        "operator": confirmed.get("operator") or "",
        "application_context": confirmed.get("context") or "",
        "core_user_value": confirmed.get("value") or confirmed.get("purpose") or "",
        "key_features": selected.get("main_features") or [],
        "required_data": confirmed.get("required_data") or "",
        "tech_direction": confirmed.get("ai_role") or "",
        "mvp_scope": confirmed.get("mvp_scope") or "",
        "differentiation": confirmed.get("differentiation") or "",
        "risks_and_mitigations": confirmed.get("risk") or [],
        "success_metrics": confirmed.get("expected_effect") or "",
        "expert_final_opinions": {},
        "unverified_assumptions": [],
        "final_recommendation": "신청 양식 초안을 기준으로 후속 검토 필요",
        "application_form": values,
    }


def finalize_session(
    state: dict,
    *,
    synthesis_provider: SynthesisProvider | None = None,
) -> dict:
    proposal = _base_proposal(state)
    synthesized = (synthesis_provider or NullSynthesisProvider()).synthesize(state)
    if isinstance(synthesized, dict):
        allowed_additions = {
            "one_line_pitch",
            "expert_final_opinions",
            "unverified_assumptions",
            "final_recommendation",
            "final_recommendation_reason",
            "next_actions",
            "discovery_history",
        }
        for key, value in synthesized.items():
            if key in allowed_additions and value not in (None, "", []):
                proposal[key] = value
        # Confirmed V2 values and empty/unresolved slots remain authoritative.
        # Synthesis cannot silently promote an unconfirmed assumption.
    state["idea_proposal"] = proposal
    state["phase"] = "finalized"
    state["updated_at"] = now_iso()
    return state
