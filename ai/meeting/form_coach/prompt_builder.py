from __future__ import annotations

import json
from pathlib import Path


_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "ideation_form_coach_01.txt"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_prompt(
    *,
    state: dict,
    current_field: dict,
    current_phase: str,
    previous_phase: str | None,
    previous_field: dict | None,
    latest_user_answer: str,
    evidence: list[dict],
    expert_review: dict,
    answer_assessment: dict | None = None,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    contest_analysis = state.get("contest_analysis") or {}
    contest_context = {
        "title": state.get("competition_name", ""),
        "evaluation_criteria": contest_analysis.get("evaluation_criteria") or [],
        "notice_document": state.get("competition_document", ""),
    }
    form_fields = [
        {
            "id": row.get("field_id"),
            "label": row.get("field_name"),
            "description": row.get("description"),
            "char_limit": row.get("char_limit"),
            "value": row.get("value"),
            "status": row.get("status"),
        }
        for row in state.get("application_form_draft") or []
    ]
    replacements = {
        "<<COMPETITION_NAME>>": state.get("competition_name", ""),
        "<<COMPETITION_DOCUMENT>>": state.get("competition_document", ""),
        "<<SELECTED_IDEA_JSON>>": _json(state.get("selected_idea") or {}),
        "<<IDEA_CANVAS_JSON>>": _json(state.get("idea_canvas") or {}),
        "<<APPLICATION_FORM_DRAFT_JSON>>": _json(state.get("application_form_draft") or []),
        "<<CURRENT_FIELD_JSON>>": _json(current_field),
        "<<CURRENT_PHASE>>": current_phase,
        "<<PREVIOUS_PHASE>>": previous_phase or "",
        "<<PREVIOUS_FIELD_JSON>>": _json(previous_field),
        "<<CONTEST_CONTEXT_JSON>>": _json(contest_context),
        "<<FORM_FIELDS_JSON>>": _json(form_fields),
        "<<CONFIRMED_STATE_JSON>>": _json(state.get("confirmed_state") or {}),
        "<<CANDIDATE_STATE_JSON>>": _json(state.get("candidate_state") or {}),
        "<<LATEST_USER_ANSWER>>": latest_user_answer,
        "<<ANSWER_ASSESSMENT_JSON>>": _json(answer_assessment or {}),
        "<<RECENT_MESSAGES_JSON>>": _json((state.get("messages") or [])[-8:]),
        "<<EVIDENCE_JSON>>": _json(evidence),
        "<<EXPERT_REVIEW_JSON>>": _json(expert_review),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template
