from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Protocol


_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "ideation_form_coach_answer_assessment_01.txt"
)
logger = logging.getLogger(__name__)
_MENTOR_REASONS = {
    "broad_or_off_track",
    "important_direction",
    "evaluation_alignment",
    "tradeoff",
    "implementation_constraint",
}


class AnswerAssessmentProvider(Protocol):
    def assess(
        self,
        *,
        phase: str,
        current_field: dict,
        state: dict,
        previous_question: str,
        latest_user_answer: str,
    ) -> dict:
        ...


class NullAnswerAssessmentProvider:
    def assess(self, **kwargs) -> dict:
        return {}


class LLMAnswerAssessmentProvider:
    """Classifies one answer before optional expert and facilitator calls."""

    def __init__(self, llm_call: Callable[[str], str]):
        self._llm_call = llm_call

    def assess(
        self,
        *,
        phase: str,
        current_field: dict,
        state: dict,
        previous_question: str,
        latest_user_answer: str,
    ) -> dict:
        prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        values = {
            "<<PHASE>>": phase,
            "<<CURRENT_FIELD_JSON>>": json.dumps(current_field, ensure_ascii=False),
            "<<CONTEST_CONTEXT_JSON>>": json.dumps(
                {
                    "competition_name": state.get("competition_name"),
                    "competition_document": state.get("competition_document"),
                    "contest_analysis": state.get("contest_analysis") or {},
                },
                ensure_ascii=False,
            ),
            "<<APPLICATION_FORM_DRAFT_JSON>>": json.dumps(
                state.get("application_form_draft") or [],
                ensure_ascii=False,
            ),
            "<<PREVIOUS_QUESTION>>": previous_question,
            "<<LATEST_USER_ANSWER>>": latest_user_answer,
            "<<SELECTED_IDEA_JSON>>": json.dumps(state.get("selected_idea") or {}, ensure_ascii=False),
            "<<IDEA_CANVAS_JSON>>": json.dumps(state.get("idea_canvas") or {}, ensure_ascii=False),
            "<<CONFIRMED_STATE_JSON>>": json.dumps(state.get("confirmed_state") or {}, ensure_ascii=False),
            "<<CANDIDATE_STATE_JSON>>": json.dumps(state.get("candidate_state") or {}, ensure_ascii=False),
            "<<RECENT_MESSAGES_JSON>>": json.dumps((state.get("messages") or [])[-8:], ensure_ascii=False),
        }
        for marker, value in values.items():
            prompt = prompt.replace(marker, value)
        try:
            payload = json.loads(self._llm_call(prompt))
        except Exception:
            logger.exception("form_coach answer assessment failed phase=%s", phase)
            return {}
        if not isinstance(payload, dict):
            return {}
        mentor_reason = str(payload.get("mentor_reason") or "none").strip()
        payload["mentor_reason"] = mentor_reason
        if mentor_reason in _MENTOR_REASONS:
            payload["mentor_needed"] = True
        return payload
