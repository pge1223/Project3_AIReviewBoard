"""Application-form draft helpers for the facilitator v02 prompt.

Rollback boundary:
- Switching prompt_loader.IDEATION_CONV_DISCUSSION_FACILITATOR_TEMPLATE back to
  ideation_conv_discussion_facilitator.txt disables new draft patches.
- These helpers are additive and keep legacy sessions without draft state valid.
"""

from __future__ import annotations

import re
from typing import Any


_FIELD_ID_SAFE = re.compile(r"[^a-z0-9_]+")


def _field_id(item: dict, index: int) -> str:
    provided = str(item.get("field_id") or "").strip().lower()
    if provided:
        normalized = _FIELD_ID_SAFE.sub("_", provided).strip("_")
        if normalized:
            return normalized
    return f"form_field_{index + 1}"


def initialize_application_form_draft(items: list[dict] | None) -> list[dict]:
    """Build stable, empty draft rows without modifying extracted form items."""
    rows: list[dict] = []
    for index, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        field_name = str(item.get("field_name") or "").strip()
        if not field_name:
            continue
        rows.append(
            {
                "field_id": _field_id(item, index),
                "field_name": field_name,
                "description": str(item.get("description") or "").strip(),
                "char_limit": item.get("char_limit"),
                "value": "",
                "status": "empty",
            }
        )
    return rows


def apply_application_form_draft_patch(
    current_draft: list[dict] | None,
    raw_patch: Any,
    confirmable_field_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Apply only patches targeting known fields and return sanitized applied rows."""
    draft = [dict(row) for row in (current_draft or []) if isinstance(row, dict)]
    by_id = {str(row.get("field_id")): row for row in draft}
    applied: list[dict] = []
    if not isinstance(raw_patch, list):
        return draft, applied

    for patch in raw_patch:
        if not isinstance(patch, dict):
            continue
        field_id = str(patch.get("field_id") or "").strip()
        value = str(patch.get("value") or "").strip()
        row = by_id.get(field_id)
        if row is None or not value:
            continue

        char_limit = row.get("char_limit")
        if isinstance(char_limit, int) and char_limit > 0:
            value = value[:char_limit]

        status = str(patch.get("status") or "draft").strip().lower()
        if status not in {"draft", "confirmed"}:
            status = "draft"
        if (
            status == "confirmed"
            and confirmable_field_ids is not None
            and field_id not in confirmable_field_ids
        ):
            status = "draft"
        row["value"] = value
        row["status"] = status
        applied.append({"field_id": field_id, "value": value, "status": status})

    return draft, applied
