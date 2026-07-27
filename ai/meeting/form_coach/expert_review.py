from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Protocol

from ..prompts.prompt_loader import get_persona_card, render_persona_block


_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "ideation_form_coach_expert_review_01.txt"
logger = logging.getLogger(__name__)


class ExpertReviewProvider(Protocol):
    def review(
        self,
        *,
        phase: str,
        current_field: dict,
        state: dict,
        latest_user_answer: str,
        mentor_types: list[str] | None = None,
        evidence: list[dict] | None = None,
        mentor_reason: str = "",
    ) -> dict:
        ...


class NullExpertReviewProvider:
    def review(self, **kwargs) -> dict:
        return {"planning": "", "development": ""}


class LLMExpertReviewProvider:
    """Reviews a broad or off-track user answer for post-answer guidance."""

    def __init__(self, llm_call: Callable[[str], str]):
        self._llm_call = llm_call

    def review(
        self,
        *,
        phase: str,
        current_field: dict,
        state: dict,
        latest_user_answer: str,
        mentor_types: list[str] | None = None,
        evidence: list[dict] | None = None,
        mentor_reason: str = "",
    ) -> dict:
        requested = [
            value
            for value in (mentor_types or [])
            if value in {"planning", "development"}
        ]
        if requested == ["planning"]:
            role = "planning"
            persona_ids = ["planning_expert"]
        elif requested == ["development"]:
            role = "development"
            persona_ids = ["dev_expert"]
        else:
            role = "planning_and_development"
            persona_ids = ["planning_expert", "dev_expert"]
        persona_block = "\n\n".join(
            render_persona_block(get_persona_card(persona_id))
            for persona_id in persona_ids
        )
        prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        values = {
            "<<ROLE>>": role,
            "<<PERSONA_BLOCK>>": persona_block,
            "<<PHASE>>": phase,
            "<<CURRENT_FIELD_JSON>>": json.dumps(current_field, ensure_ascii=False),
            "<<SELECTED_IDEA_JSON>>": json.dumps(state.get("selected_idea") or {}, ensure_ascii=False),
            "<<IDEA_CANVAS_JSON>>": json.dumps(state.get("idea_canvas") or {}, ensure_ascii=False),
            "<<CONFIRMED_STATE_JSON>>": json.dumps(state.get("confirmed_state") or {}, ensure_ascii=False),
            "<<CANDIDATE_STATE_JSON>>": json.dumps(state.get("candidate_state") or {}, ensure_ascii=False),
            "<<DRAFT_JSON>>": json.dumps(state.get("application_form_draft") or [], ensure_ascii=False),
            "<<LATEST_USER_ANSWER>>": latest_user_answer,
            "<<MENTOR_REASON>>": mentor_reason or "broad_or_off_track",
            "<<RETRIEVED_EVIDENCE_JSON>>": json.dumps(evidence or [], ensure_ascii=False),
        }
        for marker, value in values.items():
            prompt = prompt.replace(marker, value)
        try:
            payload = json.loads(self._llm_call(prompt))
        except Exception:
            # Expert review is advisory. A provider/API/JSON failure must not prevent the
            # facilitator from opening the form-writing turn.
            logger.exception("form_coach expert review failed phase=%s role=%s", phase, role)
            return {"planning": "", "development": ""}
        if not isinstance(payload, dict):
            return {"planning": "", "development": ""}
        summary = str(payload.get("summary") or "").strip()
        planning = str(payload.get("planning") or "").strip()
        development = str(payload.get("development") or "").strip()
        return {
            "planning": planning or (summary if role == "planning" else ""),
            "development": development or (summary if role == "development" else ""),
        }
