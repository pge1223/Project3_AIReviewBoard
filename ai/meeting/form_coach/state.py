from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from .phase_resolver import PHASE_ORDER, first_phase


_FIELD_ID_SAFE = re.compile(r"[^a-z0-9_]+")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_message(
    *,
    speaker_id: str,
    speaker_name: str,
    role: str,
    message_type: str,
    content: str,
    structured: dict | None = None,
) -> dict:
    return {
        "message_id": f"MSG-{uuid.uuid4().hex[:10]}",
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "role": role,
        "round": 0,
        "message_type": message_type,
        "content": content.strip(),
        "referenced_message_ids": [],
        "evidence": [],
        "created_at": now_iso(),
        "structured": structured,
    }


def initialize_draft(items: list[dict] | None) -> list[dict]:
    rows: list[dict] = []
    used_ids: set[str] = set()
    for index, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        field_name = str(item.get("field_name") or "").strip()
        if not field_name:
            continue
        provided = str(item.get("field_id") or "").strip().lower()
        field_id = _FIELD_ID_SAFE.sub("_", provided).strip("_") if provided else ""
        if not field_id or field_id in used_ids:
            field_id = f"form_field_{index + 1}"
        used_ids.add(field_id)
        rows.append(
            {
                "field_id": field_id,
                "field_name": field_name,
                "description": str(item.get("description") or "").strip(),
                "char_limit": item.get("char_limit"),
                "value": "",
                "provisional_value": "",
                "source": None,
                "status": "in_progress",
                "confidence": None,
                "editable": True,
            }
        )
    return rows


def new_session(
    *,
    session_id: str,
    competition_name: str,
    competition_document: str,
    selected_idea: dict,
    idea_canvas: dict | None,
    application_form_items: list[dict],
    legacy_context: dict | None = None,
    project_id: str | None = None,
    use_rag: bool = False,
) -> dict:
    legacy = legacy_context or {}
    seeded_canvas = {
        **dict(selected_idea),
        **dict(idea_canvas or {}),
    }
    legacy_messages = list(legacy.get("messages") or [])
    last_user_index = max(
        (index for index, message in enumerate(legacy_messages) if message.get("speaker_id") == "user"),
        default=-1,
    )
    preserved_messages = legacy_messages[: last_user_index + 1] if last_user_index >= 0 else []
    return {
        "conversation_flow": "form_coach_v2",
        "session_id": session_id,
        "project_id": project_id,
        "use_rag": bool(use_rag and project_id),
        "selected_idea_document_id": legacy.get("selected_idea_document_id"),
        "phase": "awaiting_form_answer",
        "round": 0,
        "max_rounds": len(PHASE_ORDER),
        "competition_name": competition_name,
        "competition_document": competition_document,
        "contest_analysis": dict(legacy.get("contest_analysis") or {}),
        "selected_idea": dict(selected_idea),
        "idea_canvas": seeded_canvas,
        "application_form_items": [dict(item) for item in application_form_items],
        "application_form_draft": initialize_draft(application_form_items),
        "confirmed_state": {
            "selected_idea": selected_idea.get("title") or selected_idea.get("idea_name") or "",
            "problem": "",
            "target_user": "",
            "operator": "",
            "context": "",
            "value": "",
            # Compatibility aliases for existing canvas/finalization consumers.
            "target": "",
            "purpose": "",
            "solution": "",
            "ai_role": "",
            "mvp_scope": "",
            "required_data": "",
            "expected_effect": "",
            "validation_method": "",
            "differentiation": "",
            "risk": "",
        },
        "candidate_state": {
            "solution_candidate": "",
            "target_candidate": "",
            "operator_candidate": "",
            "context_candidate": "",
            "value_candidate": "",
            "delegation_candidate": "",
        },
        "pending_delegation": None,
        "current_field_id": None,
        "current_phase": first_phase(),
        # Keep candidate history through the selection. A legacy facilitator turn may have
        # been emitted in that same request, so intentionally drop everything after the user.
        "messages": preserved_messages,
        "consensus": [],
        "unresolved_issues": [],
        "idea_proposal": None,
        "ideation_mode": legacy.get("ideation_mode", "discovery"),
        "idea_candidates": legacy.get("idea_candidates", []),
        "original_idea_candidates": legacy.get("original_idea_candidates", []),
        "selection_reason": legacy.get("selection_reason"),
        "selection_intent": legacy.get("selection_intent"),
        "user_selection_message": legacy.get("user_selection_message"),
        "source_candidates": legacy.get("source_candidates", []),
        "merge_analysis": legacy.get("merge_analysis"),
        "error": None,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
