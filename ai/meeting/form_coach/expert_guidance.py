from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Protocol

from ..prompts.prompt_loader import get_persona_card, render_persona_block


_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "ideation_form_coach_expert_guidance_01.txt"
)
logger = logging.getLogger(__name__)


class ExpertGuidanceProvider(Protocol):
    def guide(
        self,
        *,
        phase: str,
        state: dict,
        facilitator_text: str,
        question: str,
        choices: list[dict],
        mentor_types: list[str],
        guidance_reason: str,
        evidence: list[dict],
    ) -> dict:
        ...


class NullExpertGuidanceProvider:
    def guide(self, **kwargs) -> dict:
        return {"planning": "", "development": ""}


class LLMExpertGuidanceProvider:
    """Generates pre-answer guidance using the existing expert personas."""

    def __init__(self, llm_call: Callable[[str], str]):
        self._llm_call = llm_call

    def guide(
        self,
        *,
        phase: str,
        state: dict,
        facilitator_text: str,
        question: str,
        choices: list[dict],
        mentor_types: list[str],
        guidance_reason: str,
        evidence: list[dict],
    ) -> dict:
        requested = [
            value for value in mentor_types if value in {"planning", "development"}
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
        prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        replacements = {
            "<<ROLE>>": role,
            "<<PERSONA_BLOCK>>": "\n\n".join(
                render_persona_block(get_persona_card(persona_id))
                for persona_id in persona_ids
            ),
            "<<PHASE>>": phase,
            "<<SELECTED_IDEA_JSON>>": json.dumps(
                state.get("selected_idea") or {},
                ensure_ascii=False,
            ),
            "<<CONFIRMED_STATE_JSON>>": json.dumps(
                state.get("confirmed_state") or {},
                ensure_ascii=False,
            ),
            "<<FACILITATOR_TEXT>>": facilitator_text,
            "<<QUESTION>>": question,
            "<<CHOICES_JSON>>": json.dumps(choices, ensure_ascii=False),
            "<<GUIDANCE_REASON>>": guidance_reason,
            "<<RETRIEVED_EVIDENCE_JSON>>": json.dumps(
                evidence,
                ensure_ascii=False,
            ),
        }
        for marker, value in replacements.items():
            prompt = prompt.replace(marker, value)
        try:
            payload = json.loads(self._llm_call(prompt))
        except Exception:
            logger.exception("form_coach pre-answer expert guidance failed phase=%s", phase)
            return {"planning": "", "development": ""}
        if not isinstance(payload, dict):
            return {"planning": "", "development": ""}
        return {
            "planning": str(payload.get("planning") or "").strip(),
            "development": str(payload.get("development") or "").strip(),
        }
