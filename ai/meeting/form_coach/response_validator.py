from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from .phase_resolver import PHASE_LABELS, next_phase, phase_complete


_CONFIRMED_STATE_KEYS = {
    "problem",
    "target_user",
    "operator",
    "context",
    "value",
    "target",
    "purpose",
    "solution",
    "ai_role",
    "mvp_scope",
    "required_data",
    "expected_effect",
    "validation_method",
    "differentiation",
    "risk",
}
_PHASE_STATE_KEYS = {
    "problem_definition": {"problem"},
    "target_and_context": {"target", "target_user", "operator", "context"},
    "purpose_and_value": {"purpose", "value"},
    "solution_scenario": {"solution"},
    "ai_role_and_mvp": {"ai_role", "mvp_scope", "required_data"},
    "effect_and_validation": {"expected_effect", "validation_method"},
    "differentiation_and_risk": {"differentiation", "risk"},
    "application_drafting": set(),
}
_CANDIDATE_STATE_KEYS = {
    "solution_candidate",
    "target_candidate",
    "operator_candidate",
    "context_candidate",
    "value_candidate",
}
_OFF_TRACK_CANDIDATE_FIELDS = {
    "solution": "solution_candidate",
    "solution_candidate": "solution_candidate",
    "target": "target_candidate",
    "target_user": "target_candidate",
    "target_candidate": "target_candidate",
    "operator": "operator_candidate",
    "operator_candidate": "operator_candidate",
    "context": "context_candidate",
    "context_candidate": "context_candidate",
    "value": "value_candidate",
    "purpose": "value_candidate",
    "value_candidate": "value_candidate",
}
_REPEATED_QUESTION_FALLBACKS = {
    "problem_definition": "\ubc29\uae08 \ub2f5\ubcc0\uc5d0\uc11c \uac00\uc7a5 \uba3c\uc800 \ud574\uacb0\ud560 \ubb38\uc81c \uc0c1\ud669 \ud558\ub098\ub97c \uace8\ub77c\uc8fc\uc138\uc694.",
    "target_and_context": "\ubc29\uae08 \ub2f5\ubcc0\uc5d0\uc11c \ud575\uc2ec \uc218\ud61c\uc790 \ub610\ub294 \ub3c4\uc785 \uc8fc\uccb4 \ud55c \uacf3\uc744 \uace8\ub77c\uc8fc\uc138\uc694.",
    "purpose_and_value": "\ubc29\uae08 \ub2f5\ubcc0\uc5d0\uc11c \uc774 \uc11c\ube44\uc2a4\uac00 \uac00\uc7a5 \uc6b0\uc120\ud574\uc57c \ud560 \ubcc0\ud654 \ud558\ub098\ub97c \uace8\ub77c\uc8fc\uc138\uc694.",
    "solution_scenario": "\ubc29\uae08 \ub2f5\ubcc0\uc5d0\uc11c \uc0ac\uc6a9\uc790\uac00 \uac00\uc7a5 \uba3c\uc800 \uccb4\uac10\ud560 \ub3c4\uc6c0 \ud558\ub098\ub97c \uace8\ub77c\uc8fc\uc138\uc694.",
    "ai_role_and_mvp": "\ubc29\uae08 \ub2f5\ubcc0\uc5d0\uc11c AI\uac00 \uccab \ubc84\uc804\uc5d0\uc11c \ub9e1\uc744 \ud575\uc2ec \uc5ed\ud560 \ud558\ub098\ub97c \uace8\ub77c\uc8fc\uc138\uc694.",
    "effect_and_validation": "\ubc29\uae08 \ub2f5\ubcc0\uc5d0\uc11c \uac00\uc7a5 \uba3c\uc800 \uce21\uc815\ud560 \ubcc0\ud654 \ud558\ub098\ub97c \uace8\ub77c\uc8fc\uc138\uc694.",
    "differentiation_and_risk": "\ubc29\uae08 \ub2f5\ubcc0\uc5d0\uc11c \uba3c\uc800 \uac80\ud1a0\ud560 \ucc28\ubcc4\uc810 \ub610\ub294 \ub9ac\uc2a4\ud06c \ud558\ub098\ub97c \uace8\ub77c\uc8fc\uc138\uc694.",
    "application_drafting": "\ubc29\uae08 \ub2f5\ubcc0\uc744 \ubc18\uc601\ud560 \uc2e0\uccad\uc11c \ud56d\ubaa9 \ud558\ub098\ub97c \uace8\ub77c\uc8fc\uc138\uc694.",
}
_FALLBACK_CHOICE_LENSES = {
    "problem_definition": (
        ("영향이 가장 큰 상황", "사용자에게 미치는 영향이 큰 문제부터 선택"),
        ("가장 자주 발생하는 상황", "반복적으로 나타나는 문제부터 선택"),
        ("먼저 해결 가능한 상황", "이번 제안에서 범위를 잡기 쉬운 문제부터 선택"),
    ),
    "target_and_context": (
        ("직접 불편을 겪는 사용자", "서비스의 핵심 수혜자를 우선 결정"),
        ("도입·운영 주체", "서비스를 실제로 도입하고 운영할 주체를 결정"),
        ("우선 적용 현장", "첫 적용 범위를 구체적인 현장으로 결정"),
    ),
    "purpose_and_value": (
        ("사용자 경험 변화", "핵심 사용자가 직접 체감할 변화를 우선"),
        ("업무·운영 변화", "처리 과정이나 운영 효율의 변화를 우선"),
        ("사회적 가치 변화", "문제 해결이 만드는 공공적 가치를 우선"),
    ),
    "solution_scenario": (
        ("사용자 행동 지원", "사용자가 문제 상황에서 받는 도움을 중심으로 결정"),
        ("담당자 판단 지원", "운영 담당자의 판단과 대응을 중심으로 결정"),
        ("반복 과정 간소화", "반복되는 절차를 줄이는 해결 방식을 중심으로 결정"),
    ),
    "ai_role_and_mvp": (
        ("분석·분류 지원", "입력 정보를 분석하고 구분하는 역할부터 시작"),
        ("추천·판단 지원", "사용자나 담당자의 다음 결정을 돕는 역할부터 시작"),
        ("예측·감지 지원", "변화나 위험 신호를 미리 찾는 역할부터 시작"),
    ),
    "effect_and_validation": (
        ("시간·비용 변화", "처리 시간이나 운영 비용의 변화를 측정"),
        ("이용 편의 변화", "접근성과 사용자 경험의 변화를 측정"),
        ("정확도·품질 변화", "결과의 정확성과 서비스 품질 변화를 측정"),
    ),
    "differentiation_and_risk": (
        ("사용자 경험 차별점", "기존 방식과 다른 사용자 경험을 우선 검토"),
        ("운영 방식 차별점", "도입과 운영 과정의 차이를 우선 검토"),
        ("구현 리스크 우선", "가장 먼저 줄여야 할 기술·운영 위험을 검토"),
    ),
    "application_drafting": (
        ("현재 초안 보완", "대화에서 확정된 내용으로 현재 항목을 보완"),
        ("근거 연결 보완", "공고 목적과 평가 관점의 연결을 보완"),
        ("표현 구체화", "추상적인 표현을 제출 가능한 문장으로 보완"),
    ),
}
_DEFAULT_MENTOR_TYPES = {
    "problem_definition": ("planning",),
    "target_and_context": ("planning",),
    "purpose_and_value": ("planning",),
    "solution_scenario": ("planning",),
    "ai_role_and_mvp": ("development",),
    "effect_and_validation": ("planning",),
    "differentiation_and_risk": ("planning", "development"),
}
_EXPERT_TYPES = {"planning", "development"}
_PRE_GUIDANCE_REASONS = {
    "important_direction",
    "evaluation_alignment",
    "tradeoff",
    "implementation_constraint",
}
_SPECIFICITY_VALUES = {"none", "broad", "specific"}
_UNIVERSAL_TARGET_TERMS = ("전 연령", "모든 사람", "누구나")
_MENTOR_DECISION_TERMS = (
    "빠진",
    "후보",
    "해결 방식",
    "수혜자",
    "운영 주체",
    "적용 현장",
    "차이",
    "범위",
    "제약",
    "우선",
    "발생 상황",
    "측정",
)
_INTERNAL_TEXT_MARKERS = (
    "현재 작성 중인",
    "현재 작성에 연결되는",
    "지금까지 확정된 내용",
    "현재까지 확정된 내용",
    "decision_reason",
    "current_form_fields",
    "confirmed_summary",
)


def parse_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _clean_choices(value) -> list[dict]:
    choices: list[dict] = []
    for index, item in enumerate(value if isinstance(value, list) else []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        choice_id = str(item.get("id") or f"choice_{index + 1}")
        if choice_id in {"custom", "direct_input"}:
            continue
        choices.append(
            {
                "id": choice_id,
                "label": label[:100],
                "description": str(item.get("description") or "").strip()[:180],
                "value": str(item.get("value") or label).strip()[:300],
            }
        )
        if len(choices) == 3:
            break
    if choices:
        choices.append(
            {
                "id": "custom",
                "label": "직접 입력 또는 결합",
                "description": "원하는 상황을 직접 설명",
                "value": "",
            }
        )
    return choices


def _fallback_choices(phase: str) -> list[dict]:
    choices = [
        {
            "id": f"fallback_{index + 1}",
            "label": label,
            "description": description,
            "value": label,
        }
        for index, (label, description) in enumerate(
            _FALLBACK_CHOICE_LENSES.get(phase, _FALLBACK_CHOICE_LENSES["application_drafting"])
        )
    ]
    choices.append(
        {
            "id": "custom",
            "label": "직접 입력 또는 결합",
            "description": "원하는 방향을 직접 설명",
            "value": "",
        }
    )
    return choices


def _ensure_choices(choices: list[dict], phase: str) -> list[dict]:
    content_choices = [choice for choice in choices if choice.get("id") != "custom"]
    existing_labels = {str(choice.get("label") or "") for choice in content_choices}
    for fallback in _fallback_choices(phase):
        if fallback["id"] == "custom" or fallback["label"] in existing_labels:
            continue
        if len(content_choices) >= 3:
            break
        content_choices.append(fallback)
        existing_labels.add(fallback["label"])
    return [
        *content_choices[:3],
        {
            "id": "custom",
            "label": "직접 입력 또는 결합",
            "description": "원하는 방향을 직접 설명",
            "value": "",
        },
    ]


def _state_update(internal_state: dict) -> dict:
    raw = internal_state.get("confirmed_state_update")
    if not isinstance(raw, dict):
        return {}
    return {
        key: str(value).strip()
        for key, value in raw.items()
        if key in _CONFIRMED_STATE_KEYS and str(value or "").strip()
    }


def _phase_state_update(internal_state: dict, current_phase: str) -> dict:
    allowed_keys = _PHASE_STATE_KEYS.get(current_phase, set())
    update = {
        key: value
        for key, value in _state_update(internal_state).items()
        if key in allowed_keys
    }
    if "target_user" in update:
        update["target"] = update["target_user"]
    elif "target" in update:
        update["target_user"] = update["target"]
    if "value" in update:
        update["purpose"] = update["value"]
    elif "purpose" in update:
        update["value"] = update["purpose"]
    return update


def _candidate_state_update(
    internal_state: dict,
    *,
    assessment: dict,
    latest_user_answer: str,
) -> dict:
    raw = internal_state.get("candidate_state_update")
    update = {
        str(key): str(value).strip()
        for key, value in raw.items()
        if key in _CANDIDATE_STATE_KEYS and str(value or "").strip()
    } if isinstance(raw, dict) else {}
    if assessment.get("fits_current_phase"):
        return {}
    category = str(assessment.get("off_track_category") or "").strip()
    candidate_field = _OFF_TRACK_CANDIDATE_FIELDS.get(category)
    if candidate_field and candidate_field not in update and latest_user_answer.strip():
        update[candidate_field] = latest_user_answer.strip()
    return update


def _normalized_question(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower(), flags=re.UNICODE)


def _is_repeated_question(question: str, previous_question: str) -> bool:
    current = _normalized_question(question)
    previous = _normalized_question(previous_question)
    if not current or not previous:
        return False
    return current == previous or SequenceMatcher(None, current, previous).ratio() >= 0.86


def _application_patches(internal_state: dict, draft: list[dict], *, allowed: bool) -> list[dict]:
    if not allowed:
        return []
    by_id = {str(row.get("field_id")): row for row in draft}
    patches = []
    raw_patches = internal_state.get("draft_patch")
    for patch in raw_patches if isinstance(raw_patches, list) else []:
        if not isinstance(patch, dict):
            continue
        field_id = str(patch.get("field_id") or "")
        row = by_id.get(field_id)
        value = str(patch.get("value") or "").strip()
        if row is None or not value:
            continue
        char_limit = row.get("char_limit")
        if isinstance(char_limit, int) and char_limit > 0:
            value = value[:char_limit]
        status = "confirmed" if str(patch.get("status")) == "confirmed" else "draft"
        patches.append(
            {
                "field_id": field_id,
                "form_field": str(row.get("field_name") or "").strip(),
                "value": value,
                "provisional_value": "",
                "source": "conversation",
                "status": status,
                "confidence": str(patch.get("confidence") or "medium"),
                "editable": True,
            }
        )
    return patches


def _sentences(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    parts = [part.strip() for part in re.split(r"(?<=[.!?。])\s+", cleaned) if part.strip()]
    return " ".join(parts[:limit])[:500]


def _answer_assessment(
    internal_state: dict,
    *,
    current_phase: str,
    latest_user_answer: str,
    proposed_update: dict,
) -> dict:
    raw = internal_state.get("answer_assessment")
    source = raw if isinstance(raw, dict) else {}
    has_answer = bool(latest_user_answer.strip())
    if not has_answer:
        return {
            "fits_current_phase": True,
            "classification": "on_track",
            "off_track_category": "",
            "specificity": "none",
            "needs_clarification": False,
            "should_advance_phase": False,
            "mentor_needed": False,
            "mentor_reason": "none",
            "mentor_types": [],
            "delegation_requested": False,
        }

    classification = str(source.get("classification") or "").strip()
    delegation_requested = bool(
        source.get("delegation_requested") or classification == "expert_delegation"
    )
    if delegation_requested:
        return {
            "fits_current_phase": True,
            "classification": "expert_delegation",
            "off_track_category": "",
            "specificity": "none",
            "needs_clarification": False,
            "should_advance_phase": False,
            "mentor_needed": False,
            "mentor_reason": "none",
            "mentor_types": [],
            "delegation_requested": True,
        }

    fits = source.get("fits_current_phase")
    fits = fits if isinstance(fits, bool) else True
    specificity = str(source.get("specificity") or "").strip()
    if specificity not in _SPECIFICITY_VALUES:
        specificity = "specific" if proposed_update else "broad"
    should_advance = source.get("should_advance_phase")
    should_advance = should_advance if isinstance(should_advance, bool) else bool(proposed_update)
    needs_clarification = source.get("needs_clarification")
    needs_clarification = (
        needs_clarification if isinstance(needs_clarification, bool) else not should_advance
    )
    if not fits or specificity == "broad":
        needs_clarification = True
    if needs_clarification:
        should_advance = False

    is_universal_target = (
        current_phase == "target_and_context"
        and any(term in latest_user_answer for term in _UNIVERSAL_TARGET_TERMS)
        and bool(proposed_update.get("target_user") or proposed_update.get("target"))
    )
    if is_universal_target:
        fits = True
        specificity = "specific"
        needs_clarification = False
        should_advance = True

    mentor_needed = source.get("mentor_needed")
    mentor_needed = mentor_needed if isinstance(mentor_needed, bool) else needs_clarification
    if needs_clarification:
        mentor_needed = True
    if is_universal_target:
        mentor_needed = False
    raw_types = source.get("mentor_types")
    mentor_types = [
        str(value)
        for value in raw_types if str(value) in _EXPERT_TYPES
    ] if isinstance(raw_types, list) else []
    if mentor_needed and not mentor_types:
        mentor_types = list(_DEFAULT_MENTOR_TYPES.get(current_phase, ("planning",)))
    if not mentor_needed:
        mentor_types = []
    mentor_reason = str(source.get("mentor_reason") or "").strip()
    if not mentor_needed:
        mentor_reason = "none"
    elif not mentor_reason or mentor_reason == "none":
        mentor_reason = (
            "broad_or_off_track"
            if needs_clarification or not fits
            else "important_direction"
        )

    return {
        "fits_current_phase": fits,
        "classification": "on_track" if fits else "off_track",
        "off_track_category": (
            ""
            if fits
            else str(source.get("off_track_category") or "unknown").strip()
        ),
        "specificity": specificity,
        "needs_clarification": needs_clarification,
        "should_advance_phase": should_advance,
        "mentor_needed": mentor_needed,
        "mentor_reason": mentor_reason,
        "mentor_types": mentor_types[:2],
        "delegation_requested": False,
    }


def _pre_answer_guidance(
    internal_state: dict,
    *,
    resulting_phase: str,
    latest_user_answer: str,
    next_action: str,
) -> dict:
    raw = internal_state.get("pre_answer_guidance")
    source = raw if isinstance(raw, dict) else {}
    needed = bool(source.get("needed"))
    reason = str(source.get("reason") or "none").strip()
    mentor_types = [
        str(value)
        for value in source.get("mentor_types") or []
        if str(value) in _EXPERT_TYPES
    ]
    if (
        next_action != "ask_user"
        or reason not in _PRE_GUIDANCE_REASONS
        or (resulting_phase == "problem_definition" and not latest_user_answer.strip())
    ):
        return {"needed": False, "reason": "none", "mentor_types": []}
    if needed and not mentor_types:
        mentor_types = list(_DEFAULT_MENTOR_TYPES.get(resulting_phase, ("planning",)))
    return {
        "needed": needed,
        "reason": reason if needed else "none",
        "mentor_types": mentor_types[:2] if needed else [],
    }


def _mentor_is_actionable(text: str, latest_user_answer: str) -> bool:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return False
    answer_terms = [
        term
        for term in re.findall(r"[0-9A-Za-z가-힣]+", latest_user_answer)
        if len(term) >= 2
    ]
    quotes_or_interprets = any(term in normalized_text for term in answer_terms)
    gives_decision_basis = any(term in normalized_text for term in _MENTOR_DECISION_TERMS)
    return quotes_or_interprets or gives_decision_basis


def _fallback_mentor_text(
    *,
    expert_type: str,
    current_phase: str,
    latest_user_answer: str,
    assessment: dict,
) -> str:
    answer = re.sub(r"\s+", " ", latest_user_answer).strip()[:60]
    category = assessment.get("off_track_category")
    if category == "solution_candidate":
        return (
            f"'{answer}'은 해결 방식 후보입니다. 먼저 이 방식이 해결할 구체적인 문제 상황을 정해야 합니다."
        )
    missing_by_phase = {
        "problem_definition": "문제가 실제로 발생하는 상황이나 이용 과정",
        "target_and_context": "핵심 수혜자, 도입·운영 주체, 적용 현장 중 이번에 정할 역할",
        "purpose_and_value": "이 변화로 가장 먼저 개선할 사용자 경험이나 결과",
        "solution_scenario": "사용자가 도움을 받는 시점과 핵심 행동",
        "ai_role_and_mvp": "첫 버전에서 AI가 맡을 한 가지 역할과 구현 범위",
        "effect_and_validation": "효과를 확인할 측정 기준",
        "differentiation_and_risk": "비교할 차별점 또는 먼저 줄일 구현 리스크",
    }
    missing = missing_by_phase.get(current_phase, "다음 결정을 위한 한 가지 기준")
    perspective = "구현 관점에서 " if expert_type == "development" else ""
    return f"'{answer}'에는 {perspective}{missing}이 빠져 있습니다. 이 요소를 하나 정하면 다음 선택을 좁힐 수 있습니다."


def _normalize_mentor_messages(
    value,
    *,
    assessment: dict,
    expert_review: dict,
    has_user_answer: bool,
    current_phase: str,
    latest_user_answer: str,
) -> list[dict]:
    if not has_user_answer:
        return []
    if (
        not assessment.get("mentor_needed")
        and assessment.get("specificity") == "specific"
        and assessment.get("should_advance_phase")
    ):
        return []
    candidates = value if isinstance(value, list) else [value]
    normalized: dict[str, dict] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        expert_type = str(item.get("expert_type") or "").strip()
        if expert_type not in _EXPERT_TYPES:
            continue
        text = _sentences(item.get("text"), 3)
        if not text or not _mentor_is_actionable(text, latest_user_answer):
            continue
        normalized[expert_type] = {
            "expert_type": expert_type,
            "name": "기획 전문가" if expert_type == "planning" else "개발 전문가",
            "text": text,
        }

    def message_for(expert_type: str) -> dict | None:
        if expert_type in normalized:
            return normalized[expert_type]
        text = _sentences(expert_review.get(expert_type), 3)
        if not _mentor_is_actionable(text, latest_user_answer):
            text = _fallback_mentor_text(
                expert_type=expert_type,
                current_phase=current_phase,
                latest_user_answer=latest_user_answer,
                assessment=assessment,
            )
        return {
            "expert_type": expert_type,
            "name": "기획 전문가" if expert_type == "planning" else "개발 전문가",
            "text": text,
        }

    required = assessment.get("mentor_types") if assessment.get("mentor_needed") else []
    if required:
        result = []
        for expert_type in required:
            message = message_for(expert_type)
            if message is None:
                raise RuntimeError(f"답변 보완에 필요한 {expert_type} 전문가 발화가 없습니다.")
            result.append(message)
    else:
        result = [normalized[key] for key in ("planning", "development") if key in normalized][:2]

    if len(result) > 1:
        for message in result:
            message["text"] = _sentences(message["text"], 1)
    return result


def _confirmed_summary(value, merged_state: dict) -> dict:
    source = value if isinstance(value, dict) else {}
    target_user = str(
        merged_state.get("target_user")
        or merged_state.get("target")
        or source.get("target_user")
        or source.get("target")
        or ""
    ).strip()
    value_text = str(
        merged_state.get("value")
        or merged_state.get("purpose")
        or source.get("value")
        or source.get("purpose")
        or ""
    ).strip()
    return {
        "problem": str(merged_state.get("problem") or source.get("problem") or "").strip(),
        "target_user": target_user,
        "operator": str(merged_state.get("operator") or source.get("operator") or "").strip(),
        "context": str(merged_state.get("context") or source.get("context") or "").strip(),
        "value": value_text,
        "target": target_user,
        "purpose": value_text,
        "scenario": str(merged_state.get("solution") or source.get("scenario") or "").strip(),
    }


def validate_response(
    raw: str,
    *,
    current_phase: str,
    confirmed_state: dict,
    connected_form_fields: list[dict],
    latest_user_answer: str,
    expert_review: dict,
    application_form_draft: list[dict],
    answer_assessment_override: dict | None = None,
    previous_question: str = "",
) -> tuple[dict, dict, list[dict], str, str]:
    payload = parse_json_object(raw)
    internal_state = payload.get("internal_state") if isinstance(payload.get("internal_state"), dict) else {}
    ui_message = payload.get("ui_message") if isinstance(payload.get("ui_message"), dict) else {}

    # Accept the previous flat contract during rollout, but never expose its internal narration.
    if not internal_state:
        internal_state = {
            "current_form_fields": payload.get("current_field_ids"),
            "confirmed_state_update": payload.get("confirmed_state_update"),
            "draft_patch": payload.get("draft_patch"),
            "next_action": payload.get("next_action"),
        }
    if not ui_message:
        ui_message = {
            "facilitator_text": payload.get("spoken_text"),
            "question": payload.get("user_question"),
            "choices": payload.get("choices"),
            "post_answer_mentor_messages": [],
        }

    proposed_state_update = _phase_state_update(internal_state, current_phase)
    assessment_state = dict(internal_state)
    if isinstance(answer_assessment_override, dict):
        assessment_state["answer_assessment"] = answer_assessment_override
    assessment = _answer_assessment(
        assessment_state,
        current_phase=current_phase,
        latest_user_answer=latest_user_answer,
        proposed_update=proposed_state_update,
    )
    candidate_state_update = _candidate_state_update(
        internal_state,
        assessment=assessment,
        latest_user_answer=latest_user_answer,
    )
    state_update = (
        proposed_state_update
        if assessment["fits_current_phase"]
        and not assessment["needs_clarification"]
        and assessment["should_advance_phase"]
        else {}
    )
    merged_state = {**confirmed_state, **state_update}
    if current_phase == "application_drafting":
        resulting_phase = current_phase
    elif phase_complete(current_phase, merged_state):
        resulting_phase = next_phase(current_phase) or current_phase
    else:
        resulting_phase = current_phase

    known_ids = {str(row.get("field_id")) for row in application_form_draft}
    raw_field_ids = internal_state.get("current_form_fields")
    current_field_ids = [
        str(field_id)
        for field_id in raw_field_ids if str(field_id) in known_ids
    ] if isinstance(raw_field_ids, list) else []
    if not current_field_ids:
        current_field_ids = [str(row.get("field_id")) for row in connected_form_fields[:1]]

    choices = _clean_choices(ui_message.get("choices"))
    choices = _ensure_choices(choices, resulting_phase)
    question = str(ui_message.get("question") or "").strip()
    phase_label = PHASE_LABELS.get(resulting_phase, resulting_phase)
    if not question:
        question = f"이번 단계에서 '{phase_label}'의 우선 방향을 하나 골라주세요."

    if latest_user_answer.strip() and _is_repeated_question(question, previous_question):
        question = _REPEATED_QUESTION_FALLBACKS.get(resulting_phase, question)

    facilitator_text = _sentences(ui_message.get("facilitator_text"), 2)
    if not facilitator_text or any(marker in facilitator_text for marker in _INTERNAL_TEXT_MARKERS):
        facilitator_text = f"이제 {phase_label}을 한 단계 더 구체화해볼게요."

    raw_post_answer_messages = ui_message.get("post_answer_mentor_messages")
    # Old model responses used mentor_messages exclusively for post-answer review.
    if raw_post_answer_messages is None:
        raw_post_answer_messages = ui_message.get("mentor_messages")
    if not isinstance(raw_post_answer_messages, list) and ui_message.get("mentor_message") is not None:
        raw_post_answer_messages = ui_message.get("mentor_message")
    post_answer_mentor_messages = _normalize_mentor_messages(
        raw_post_answer_messages,
        assessment=assessment,
        expert_review=expert_review,
        has_user_answer=bool(latest_user_answer.strip()),
        current_phase=current_phase,
        latest_user_answer=latest_user_answer,
    )

    form_patches = _application_patches(
        internal_state,
        application_form_draft,
        allowed=(
            assessment["fits_current_phase"]
            and not assessment["needs_clarification"]
            and (bool(state_update) or current_phase == "application_drafting")
        ),
    )
    next_action = str(internal_state.get("next_action") or "ask_user")
    if next_action not in {"ask_user", "confirm_summary", "generate_application_draft"}:
        next_action = "ask_user"
    pre_answer_guidance = _pre_answer_guidance(
        internal_state,
        resulting_phase=resulting_phase,
        latest_user_answer=latest_user_answer,
        next_action=next_action,
    )

    normalized_internal = {
        "phase": resulting_phase,
        "current_form_fields": current_field_ids,
        "confirmed_summary": _confirmed_summary(internal_state.get("confirmed_summary"), merged_state),
        "decision_reason": str(internal_state.get("decision_reason") or "").strip()[:500],
        "pre_answer_guidance": pre_answer_guidance,
        "answer_assessment": assessment,
        "confirmed_state_update": state_update,
        "candidate_state_update": candidate_state_update,
        "draft_patch": form_patches,
        "next_action": next_action,
    }
    normalized_ui = {
        "facilitator_text": facilitator_text,
        "pre_answer_mentor_messages": [],
        "post_answer_mentor_messages": post_answer_mentor_messages,
        # Compatibility alias for old clients and stored sessions.
        "mentor_messages": post_answer_mentor_messages,
        "question": question,
        "choices": choices,
    }
    structured = {
        "internal_state": normalized_internal,
        "ui_message": normalized_ui,
        "phase": resulting_phase,
        "current_field_ids": current_field_ids,
        "current_field_id": current_field_ids[0] if current_field_ids else None,
        "user_question": question,
        "choices": choices,
        "pre_answer_mentor_messages": [],
        "post_answer_mentor_messages": post_answer_mentor_messages,
        "mentor_messages": post_answer_mentor_messages,
        "draft_patch": form_patches,
        "confirmed_state_update": state_update,
        "candidate_state_update": candidate_state_update,
        "next_action": next_action,
        "needs_user_decision": True,
    }
    return (
        {"content": facilitator_text, "structured": structured},
        state_update,
        form_patches,
        resulting_phase,
        next_action,
    )
