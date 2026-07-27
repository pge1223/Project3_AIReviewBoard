from __future__ import annotations

import logging
from typing import Callable, Protocol


logger = logging.getLogger(__name__)


class EvidenceProvider(Protocol):
    """Boundary for adding API search or RAG without changing coach logic."""

    def retrieve(
        self,
        *,
        query: str,
        context: dict,
        expert_types: list[str] | None = None,
    ) -> list[dict]:
        ...


class NullEvidenceProvider:
    """Default provider used before the form coach is connected to RAG."""

    def retrieve(self, **kwargs) -> list[dict]:
        return []


class LookupEvidenceProvider:
    """Adapts the existing ideation RAG lookup to the form-coach boundary."""

    _PERSONA_BY_EXPERT = {
        "planning": "planning_expert",
        "development": "dev_expert",
    }

    def __init__(
        self,
        lookup: Callable,
        *,
        session_id: str,
        selected_candidate_document_id: str | None = None,
        limit: int = 8,
    ):
        self._lookup = lookup
        self._runtime_scope = {
            "session_id": session_id,
            "selected_candidate_document_id": selected_candidate_document_id,
        }
        self._limit = limit

    def retrieve(
        self,
        *,
        query: str,
        context: dict,
        expert_types: list[str] | None = None,
    ) -> list[dict]:
        requested = expert_types or ["planning"]
        results: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for expert_type in requested:
            persona_id = self._PERSONA_BY_EXPERT.get(expert_type)
            if not persona_id:
                continue
            try:
                items = self._lookup(
                    persona_id,
                    query,
                    runtime_scope=self._runtime_scope,
                )
            except Exception:
                logger.exception(
                    "form_coach RAG lookup failed persona_id=%s",
                    persona_id,
                )
                continue
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                key = (
                    str(item.get("document_id") or ""),
                    str(item.get("chunk_id") or item.get("ref") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                results.append(dict(item))
                if len(results) >= self._limit:
                    return results
        return results
