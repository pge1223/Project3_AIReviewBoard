from __future__ import annotations

import json
import logging
from typing import Callable, Protocol

from ai.meeting.prompts import build_ideation_conv_synthesis_prompt


logger = logging.getLogger(__name__)


def _parse_object(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _visible_messages(state: dict) -> list[dict]:
    allowed = {"user", "ideation_facilitator", "planning_expert", "dev_expert"}
    visible = []
    for message in state.get("messages") or []:
        if (
            message.get("speaker_id") in allowed
            and str(message.get("content") or "").strip()
        ):
            visible.append(
                {
                    "speaker_id": message.get("speaker_id"),
                    "message_type": message.get("message_type"),
                    "content": message.get("content"),
                }
            )
        structured = message.get("structured")
        ui_message = structured.get("ui_message") if isinstance(structured, dict) else {}
        mentors = (
            ui_message.get("post_answer_mentor_messages")
            if isinstance(ui_message, dict)
            else []
        )
        for mentor in mentors if isinstance(mentors, list) else []:
            expert_type = str(mentor.get("expert_type") or "")
            content = str(mentor.get("text") or "").strip()
            if content:
                visible.append(
                    {
                        "speaker_id": (
                            "dev_expert" if expert_type == "development" else "planning_expert"
                        ),
                        "message_type": "expert_advice",
                        "content": content,
                    }
                )
    return visible


class SynthesisProvider(Protocol):
    def synthesize(self, state: dict) -> dict:
        ...


class NullSynthesisProvider:
    def synthesize(self, state: dict) -> dict:
        return {}


class LLMSynthesisProvider:
    """Calls legacy synthesis with only confirmed V2 state and visible dialogue."""

    def __init__(self, llm_call: Callable[[str], str]):
        self._llm_call = llm_call

    def synthesize(self, state: dict) -> dict:
        confirmed = state.get("confirmed_state") or {}
        draft = {
            str(row.get("field_name")): row.get("value")
            for row in state.get("application_form_draft") or []
            if str(row.get("value") or "").strip()
        }
        user_idea = {
            "selected_idea": state.get("selected_idea") or {},
            "confirmed_state": confirmed,
            "application_form_draft": draft,
        }
        consensus = [
            {"field": key, "value": value}
            for key, value in confirmed.items()
            if key != "selected_idea" and str(value or "").strip()
        ]
        unresolved = [
            key
            for key in (
                "problem",
                "target_user",
                "value",
                "solution",
                "ai_role",
                "mvp_scope",
                "expected_effect",
                "differentiation",
                "risk",
            )
            if not str(confirmed.get(key) or "").strip()
        ]
        discovery_history = {
            "original_candidates": state.get("original_idea_candidates") or [],
            "selected_idea": state.get("selected_idea") or {},
            "selection_reason": state.get("selection_reason") or "",
        }
        prompt = build_ideation_conv_synthesis_prompt(
            {
                "competition_name": state.get("competition_name"),
                "competition_document": state.get("competition_document"),
                "contest_analysis": state.get("contest_analysis") or {},
            },
            user_idea,
            _visible_messages(state),
            consensus,
            unresolved,
            discovery_history,
        )
        try:
            return _parse_object(self._llm_call(prompt))
        except Exception:
            logger.exception("form_coach V2 synthesis failed")
            return {}
