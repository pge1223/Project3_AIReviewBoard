from __future__ import annotations

import json
import logging
from typing import Callable, Protocol

from ai.meeting.prompts import (
    build_ideation_conv_expert_delegation_facilitator_prompt,
    build_ideation_conv_expert_delegation_prompt,
    build_ideation_conv_expert_delegation_review_prompt,
)


logger = logging.getLogger(__name__)

_PLANNING_PHASES = {
    "problem_definition",
    "target_and_context",
    "purpose_and_value",
    "solution_scenario",
    "effect_and_validation",
    "differentiation_and_risk",
    "application_drafting",
}
_REVISION_STANCES = {"조건부_동의", "반박", "대안_제시"}
_CROSS_REVIEW_PHASES = {
    "solution_scenario",
    "ai_role_and_mvp",
    "differentiation_and_risk",
    "application_drafting",
}


def _parse_object(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def owner_for_phase(phase: str) -> tuple[str, str]:
    if phase in _PLANNING_PHASES:
        return "planning", "planning_expert"
    return "development", "dev_expert"


def counterpart_review_required(phase: str) -> bool:
    return phase in _CROSS_REVIEW_PHASES


class ExpertDelegationProvider(Protocol):
    def delegate(
        self,
        *,
        phase: str,
        pending_question: str,
        state: dict,
        evidence_by_expert: dict[str, list[dict]],
    ) -> dict:
        ...


class NullExpertDelegationProvider:
    def delegate(self, **kwargs) -> dict:
        return {}


class LLMExpertDelegationProvider:
    """V2 adapter over the existing proposal/review/facilitator prompts."""

    def __init__(self, llm_call: Callable[[str], str]):
        self._llm_call = llm_call

    def delegate(
        self,
        *,
        phase: str,
        pending_question: str,
        state: dict,
        evidence_by_expert: dict[str, list[dict]],
    ) -> dict:
        owner_type, owner_persona = owner_for_phase(phase)
        counterpart_type = "development" if owner_type == "planning" else "planning"
        counterpart_persona = "dev_expert" if owner_type == "planning" else "planning_expert"
        notice = {
            "competition_name": state.get("competition_name"),
            "competition_document": state.get("competition_document"),
            "contest_analysis": state.get("contest_analysis") or {},
        }
        user_idea = {
            "selected_idea": state.get("selected_idea") or {},
            "idea_canvas": state.get("idea_canvas") or {},
            "confirmed_state": state.get("confirmed_state") or {},
            "candidate_state": state.get("candidate_state") or {},
        }
        context = {
            "phase": phase,
            "application_form_draft": state.get("application_form_draft") or [],
            "recent_messages": (state.get("messages") or [])[-8:],
        }

        proposal = _parse_object(
            self._llm_call(
                build_ideation_conv_expert_delegation_prompt(
                    owner_persona,
                    notice,
                    user_idea,
                    evidence_by_expert.get(owner_type) or [],
                    context,
                    pending_question,
                )
            )
        )
        if not str(proposal.get("proposal") or "").strip():
            return {}

        review = {}
        if counterpart_review_required(phase):
            review = _parse_object(
                self._llm_call(
                    build_ideation_conv_expert_delegation_review_prompt(
                        counterpart_persona,
                        notice,
                        user_idea,
                        evidence_by_expert.get(counterpart_type) or [],
                        context,
                        pending_question,
                        proposal,
                    )
                )
            )
        revision = None
        if str(review.get("stance") or "").strip() in _REVISION_STANCES:
            revision = _parse_object(
                self._llm_call(
                    build_ideation_conv_expert_delegation_prompt(
                        owner_persona,
                        notice,
                        user_idea,
                        evidence_by_expert.get(owner_type) or [],
                        context,
                        pending_question,
                        stage="revision",
                        counterpart_review=review,
                    )
                )
            )

        facilitator = _parse_object(
            self._llm_call(
                build_ideation_conv_expert_delegation_facilitator_prompt(
                    notice,
                    pending_question,
                    proposal,
                    review,
                    revision,
                )
            )
        )
        suggested_value = str(
            (revision or {}).get("proposal")
            or (revision or {}).get("revision")
            or proposal.get("proposal")
            or facilitator.get("final_recommendation")
            or ""
        ).strip()
        return {
            "owner_expert_type": owner_type,
            "counterpart_expert_type": counterpart_type,
            "proposal": proposal,
            "review": review,
            "revision": revision,
            "facilitator": facilitator,
            "suggested_value": suggested_value,
        }
