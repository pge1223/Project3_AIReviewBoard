# 작성자: 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편)
# 목적: discovery(아이디어 발굴) 모드가 candidate_generation(이제 "압축된 provisional 후보
#       생성"으로 재정의)보다 먼저 거치는 단계 — 문제 발견(problem_discovery) -> 문제 초점
#       선택(problem_focus_selection) -> 문제 정의(problem_definition) -> 아이디어 발산
#       (idea_divergence) -> 반론·결합(idea_conflict_and_merge, 라운드 상한까지 자동 반복) ->
#       (조건 미충족 시) 사용자에게 결합/방향추가/방향폐기/검증진행 요청(conflict_resolution).
#       조건이 충족되면 기존 candidate_planning/candidate_feasibility(ideation_conv_discovery.py,
#       전혀 수정하지 않음)로 자연스럽게 이어지고, 그 후보 선택 결과는 기존
#       candidate_selection 노드를 그대로(블랙박스로) 재사용하되 provisional_idea/
#       idea_validation으로 재해석한다(make_provisional_selection_node) — "선택 즉시
#       selected_idea 확정"을 막기 위해서다. idea_validation을 거쳐야만 concept_confirmation
#       에서 사용자가 최종 확정(idea_locked=True)할 수 있다.
# import: 표준 라이브러리 uuid/re/typing, prompts 패키지, 같은 패키지의
#         ideation_conv_nodes(_blank/_build_message/_last_user_answer/_safe_call_structured_json)/
#         ideation_conv_discovery(_contest_query/_runtime_scope_for/make_candidate_selection_node/
#         is_regenerate_request)/ideation_conv_state/ideation_nodes/ideation_trace/llm.

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from prompts import (
    build_ideation_conv_conflict_merge_prompt,
    build_ideation_conv_idea_divergence_prompt,
    build_ideation_conv_idea_validation_planning_prompt,
    build_ideation_conv_idea_validation_technical_prompt,
    build_ideation_conv_problem_definition_prompt,
    build_ideation_conv_problem_discovery_prompt,
)

from .ideation_conv_discovery import (
    ExternalEvidenceLookupFn,
    _call_external_evidence_lookup,
    _contest_query,
    _external_evidence_query,
    _index_selected_candidate,
    _merge_external_evidence_results,
    is_regenerate_request,
    make_candidate_selection_node,
    solution_direction_to_idea,
)
from .ideation_conv_nodes import (
    ClaimGroundingFn,
    _blank,
    _build_message,
    _last_user_answer,
    _safe_call_structured_json,
)
from .ideation_conv_state import (
    MAX_CONFLICT_ROUNDS,
    MAX_PROBLEM_REGENERATIONS,
    MAX_VALIDATION_REVISE_ROUNDS,
    IdeationConvState,
    critique_count,
    meets_conflict_and_merge_min_conditions,
    merge_or_revision_count,
    planning_validation_completed,
    solution_direction_count,
    technical_validation_completed,
)
from .ideation_nodes import EvidenceLookup, call_evidence_lookup
from .ideation_trace import sanitize_preview, trace_event
from .llm import LLMCall

IndexTargetEvidenceFn = Callable[[str, dict], dict]

_VALIDATION_STATUSES = {"passed", "passed_with_caution", "needs_revision"}
_VALIDATION_SEVERITIES = {"minor", "major", "blocking"}
_RAW_DOCUMENT_MESSAGE_RE = re.compile(
    r"\.(?:hwpx?|pdf|docx?)\b|<\s*작성\s*요령\s*>|(?:^|\n)\s*(?:○|※|-)\s*",
    re.IGNORECASE,
)

_PROBLEM_FOCUS_QUESTION = (
    "공고문과 최근 이슈를 바탕으로, 이번 공모전에서 해결할 수 있는 문제 후보를 찾았습니다. "
    "먼저 해결하고 싶은 문제를 선택해 주세요. 이 선택은 최종 아이디어 확정이 아닙니다. "
    "번호 하나(예: '1번') 또는 최대 두 개(예: '1번과 2번 결합')를 고르거나, "
    "'다른 문제 제안', 또는 직접 문제를 설명해 주셔도 됩니다."
)

_OTHER_PROBLEM_KEYWORDS = ("다른 문제", "다른 영역", "다른 거", "다시 제안")

_NUMERIC_TOKEN_RE = re.compile(r"[1-4]")


def _runtime_scope_for(state: IdeationConvState) -> dict[str, Any]:
    return {
        "session_id": state.get("session_id"),
        "selected_candidate_document_id": state.get("selected_idea_document_id"),
        "phase": state.get("phase"),
    }


def _normalize(text: str) -> str:
    return (text or "").strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _problem_discovery_external_query(
    state: IdeationConvState,
    previous_problem_areas: list[dict] | None = None,
) -> str:
    """Build a short, issue-oriented query for the initial NAVER news lookup."""
    parts: list[str] = []
    notice = state.get("notice_and_criteria")
    if isinstance(notice, dict):
        for field in ("competition_name", "purpose", "objective", "theme", "summary"):
            value = notice.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    elif notice:
        parts.append(str(notice).strip()[:120])

    for area in previous_problem_areas or []:
        if not isinstance(area, dict):
            continue
        fragment = " ".join(
            str(area.get(field) or "")
            for field in ("title", "summary", "who_is_affected")
        ).strip()
        if fragment:
            parts.append(fragment)

    parts.append("최근 뉴스 이슈 문제 현황")
    return " ".join(" ".join(part.split()) for part in parts if part)[:300]


def _new_evolution_record(
    state: IdeationConvState,
    *,
    stage: str,
    action_type: str,
    title: str,
    content: str,
    changed_by: str,
    target_direction_ids: list[str] | None = None,
    before: Any = None,
    after: Any = None,
    result_direction_id: str | None = None,
) -> dict:
    record = {
        "record_id": f"EVOL-{uuid.uuid4().hex[:10]}",
        "stage": stage,
        "action_type": action_type,
        "title": title,
        "content": content,
        "changed_by": changed_by,
        "target_direction_ids": target_direction_ids or [],
        "created_at": _now_iso(),
    }
    if before is not None:
        record["before"] = before
    if after is not None:
        record["after"] = after
    if result_direction_id:
        record["result_direction_id"] = result_direction_id
    return record


# ============================================================================
# 1. problem_discovery — 문제 영역 2~4개 생성(완성된 서비스/기능 없음)
# ============================================================================


def _validate_problem_discovery_response(raw: dict) -> str | None:
    areas = raw.get("problem_areas")
    if not isinstance(areas, list) or not (2 <= len(areas) <= 4):
        return "problem_areas_count_invalid"
    seen: set[str] = set()
    for area in areas:
        if not isinstance(area, dict):
            return "problem_area_not_object"
        area_id = area.get("area_id")
        if _blank(area_id) or area_id in seen:
            return "area_id_missing_or_duplicate"
        seen.add(area_id)
        for field in ("title", "summary", "who_is_affected"):
            if _blank(area.get(field)):
                return f"missing_or_empty_field:{field}"
    return None


def make_problem_discovery_node(
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
    external_evidence_lookup: ExternalEvidenceLookupFn | None = None,
) -> Callable[[IdeationConvState], dict]:
    """기획 전문가가 완성된 아이디어가 아니라 "해결할 가치가 있는 문제 영역" 2~4개를
    만드는 노드. discovery 모드의 새 진입점(요청: "발산 전에 완성되는" 문제를 막기 위해
    candidate_generation보다 앞에 둔다)."""

    def node(state: IdeationConvState) -> dict:
        retrieved = call_evidence_lookup(
            evidence_lookup, "planning_expert", _contest_query(state), runtime_scope=_runtime_scope_for(state)
        )
        previous_areas = state.get("problem_areas") or []
        regeneration_reason = None
        if previous_areas:
            last_answer = _last_user_answer(state["messages"])
            regeneration_reason = (last_answer or {}).get("content")
        external_result = _call_external_evidence_lookup(
            external_evidence_lookup,
            "planning_expert",
            _problem_discovery_external_query(state, previous_areas),
        )
        prompt = build_ideation_conv_problem_discovery_prompt(
            state["notice_and_criteria"],
            retrieved,
            previous_areas,
            regeneration_reason,
            external_research=external_result.get("external_evidence"),
        )
        raw, ok, attempts = _safe_call_structured_json(
            llm_call, prompt, _validate_problem_discovery_response, "problem_discovery"
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"phase": "failed", "failed_node": "problem_discovery", "llm_calls_used": used}
        question_message = _build_message(
            persona_id="ideation_facilitator",
            round_number=state["round"],
            message_type="question",
            content=_PROBLEM_FOCUS_QUESTION,
            referenced_message_ids=[],
            evidence=[],
        )
        merged_external = _merge_external_evidence_results(
            {
                "external_evidence": state.get("external_evidence", []),
                **(state.get("external_evidence_meta") or {}),
            },
            external_result,
        )
        return {
            "problem_areas": raw["problem_areas"],
            "messages": [question_message],
            "phase": "awaiting_problem_focus_selection",
            "llm_calls_used": used,
            "external_evidence": merged_external["external_evidence"],
            "external_evidence_meta": {
                "used_dataset_search": merged_external["used_dataset_search"],
                "used_public_api_search": merged_external["used_public_api_search"],
                "warnings": merged_external["warnings"],
            },
        }

    return node


# ============================================================================
# 2. problem_focus_selection — 사용자의 문제 영역 선택/결합/재요청/직접입력 해석
# ============================================================================


def _extract_selected_areas(text: str, areas: list[dict]) -> list[dict]:
    """번호(최대 2개, 예: "1", "1번", "1,2", "1번과 2번")만 결정적으로 처리한다 —
    candidate_selection의 _match_single_candidate와 같은 원칙(단순 선택은 코드로)."""
    tokens = _NUMERIC_TOKEN_RE.findall(text)
    indices: list[int] = []
    for token in tokens:
        idx = int(token) - 1
        if 0 <= idx < len(areas) and idx not in indices:
            indices.append(idx)
        if len(indices) == 2:
            break
    return [areas[i] for i in indices]


def _is_other_problem_request(text: str) -> bool:
    return any(keyword in text for keyword in _OTHER_PROBLEM_KEYWORDS) or is_regenerate_request(text)


def _select_areas_by_action_payload(areas: list[dict], payload: dict) -> list[dict]:
    """select_problem_focus/combine_problem_focus의 payload({"indices": [1-based, ...]}
    또는 {"area_ids": [...]})로 최대 2개까지 문제 영역을 고른다. 잘못된 인덱스/id는
    조용히 걸러내고, 하나도 못 골랐으면 빈 리스트를 반환해 호출부가 자연어 폴백으로
    넘어가게 한다."""
    indices = payload.get("indices")
    if isinstance(indices, list):
        selected = []
        for raw_index in indices[:2]:
            if not isinstance(raw_index, int):
                continue
            idx = raw_index - 1
            if 0 <= idx < len(areas) and areas[idx] not in selected:
                selected.append(areas[idx])
        if selected:
            return selected
    area_ids = payload.get("area_ids")
    if isinstance(area_ids, list):
        by_id = {a.get("area_id"): a for a in areas if isinstance(a, dict)}
        selected = [by_id[aid] for aid in area_ids[:2] if aid in by_id]
        if selected:
            return selected
    return []


def make_problem_focus_selection_node(
    llm_call: LLMCall, evidence_lookup: EvidenceLookup | None = None
) -> Callable[[IdeationConvState], dict]:
    """사용자의 문제 영역 선택(1개/2개 결합)/다른 문제 요청/직접 입력을 해석한다. 번호
    선택과 재요청 키워드는 LLM 없이 코드가 결정적으로 처리한다(candidate_selection_node와
    동일한 원칙). 성공하면 problem_focus를 채우고 다음 노드(problem_definition)로 그래프
    간선이 자동으로 이어간다(별도 정지 없음)."""

    def node(state: IdeationConvState) -> dict:
        areas = state.get("problem_areas") or []

        # 용준/Claude(2026-07-27, 후속 요청 4번: 구조화된 사용자 액션 지원) — 2차 프론트가
        # action code(select_problem_focus/combine_problem_focus)를 함께 보내면 그것을
        # 우선 사용한다. payload는 {"indices": [1] 또는 [1,2]}(1-based, UI 표시 번호와
        # 동일) 또는 {"area_ids": ["area_1", ...]} 중 하나를 받는다 — 텍스트 정규식보다
        # 결정적이고 자연어 표현 변화에 흔들리지 않는다. 매칭에 실패하면(잘못된 인덱스 등)
        # 조용히 무시하지 않고 기존 자연어 파싱으로 폴백한다(하위 호환).
        action = state.get("pending_user_action") or {}
        action_code = action.get("code")
        if action_code in ("select_problem_focus", "combine_problem_focus"):
            payload = action.get("payload") or {}
            selected = _select_areas_by_action_payload(areas, payload)
            if selected:
                evolution_action_type = (
                    "problem_focus_merged" if action_code == "combine_problem_focus" else "problem_focus_selected"
                )
                evolution_title = " · ".join(a.get("title", "") for a in selected)
                evolution_record = _new_evolution_record(
                    state,
                    stage="problem_focus_selection",
                    action_type=evolution_action_type,
                    title=evolution_title,
                    content="\n".join(a.get("summary", "") for a in selected),
                    changed_by="user",
                    target_direction_ids=[a.get("area_id") for a in selected if a.get("area_id")],
                )
                return {"problem_focus": selected, "idea_evolution": [evolution_record]}

        last_answer = _last_user_answer(state["messages"])
        text = _normalize((last_answer or {}).get("content", ""))

        if _is_other_problem_request(text):
            regen = state.get("problem_regeneration_count", 0)
            if regen >= MAX_PROBLEM_REGENERATIONS:
                notice = _build_message(
                    persona_id="ideation_facilitator",
                    round_number=state["round"],
                    message_type="summary",
                    content=(
                        f"문제 영역 재요청은 최대 {MAX_PROBLEM_REGENERATIONS}회까지 가능합니다. "
                        "현재 제시된 문제 중에서 선택하거나 직접 입력해 주세요."
                    ),
                    referenced_message_ids=[],
                    evidence=[],
                )
                return {"messages": [notice], "phase": "awaiting_problem_focus_selection"}
            return {"phase": "problem_discovery", "problem_regeneration_count": regen + 1}

        selected = _extract_selected_areas(text, areas)
        if selected:
            return {"problem_focus": selected}

        # 번호로 특정할 수 없는 텍스트 — 요청: "직접 문제 입력"을 허용한다. 목록에 없는
        # 문제라도 회의 대상으로 받아들인다(사용자의 초기 선택을 강제로 목록 안으로만
        # 제한하지 않는다).
        custom_area = {
            "area_id": "custom",
            "title": text[:80] if text else "사용자가 제시한 문제",
            "summary": text or "사용자가 구체적인 설명 없이 새 문제를 제시했습니다.",
            "who_is_affected": "",
        }
        return {"problem_focus": [custom_area]}

    return node


# ============================================================================
# 3. problem_definition — 선택된 문제를 구체화(해결책 없음)
# ============================================================================


def _validate_problem_definition_response(raw: dict) -> str | None:
    definition = raw.get("problem_definition")
    if not isinstance(definition, dict):
        return "problem_definition_missing"
    for field in ("problem", "target_user", "user_context", "root_cause", "existing_solution", "existing_limitations"):
        if _blank(definition.get(field)):
            return f"missing_or_empty_field:{field}"
    return None


def make_problem_definition_node(
    llm_call: LLMCall, evidence_lookup: EvidenceLookup | None = None
) -> Callable[[IdeationConvState], dict]:
    """problem_focus를 problem_definition(문제/대상 사용자/상황/원인/기존 방식/기존 한계)
    구조로 구체화한다. 정지 없이 바로 idea_divergence로 이어진다(요청 흐름에 사용자
    체크포인트가 명시돼 있지 않음 — candidate_planning->candidate_feasibility가 정지 없이
    이어지는 것과 같은 원칙)."""

    def node(state: IdeationConvState) -> dict:
        retrieved = call_evidence_lookup(
            evidence_lookup, "planning_expert", _contest_query(state), runtime_scope=_runtime_scope_for(state)
        )
        focus = state.get("problem_focus") or []
        prompt = build_ideation_conv_problem_definition_prompt(state["notice_and_criteria"], retrieved, focus)
        raw, ok, attempts = _safe_call_structured_json(
            llm_call, prompt, _validate_problem_definition_response, "problem_definition"
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"phase": "failed", "failed_node": "problem_definition", "llm_calls_used": used}
        definition = raw["problem_definition"]
        summary_message = _build_message(
            persona_id="ideation_facilitator",
            round_number=state["round"],
            message_type="summary",
            content=f"{definition['problem']}\n(대상: {definition['target_user']})",
            referenced_message_ids=[],
            evidence=[],
        )
        return {
            "problem_definition": definition,
            "messages": [summary_message],
            "llm_calls_used": used,
        }

    return node


# ============================================================================
# 4. idea_divergence — 해결 원리가 서로 다른 방향 3개 이상 생성
# ============================================================================


def _validate_idea_divergence_response(raw: dict) -> str | None:
    directions = raw.get("solution_directions")
    if not isinstance(directions, list) or len(directions) < 3:
        return "solution_directions_count_invalid"
    seen: set[str] = set()
    for direction in directions:
        if not isinstance(direction, dict):
            return "solution_direction_not_object"
        direction_id = direction.get("direction_id")
        if _blank(direction_id) or direction_id in seen:
            return "direction_id_missing_or_duplicate"
        seen.add(direction_id)
        for field in ("title", "core_principle", "mechanism", "target_user_fit"):
            if _blank(direction.get(field)):
                return f"missing_or_empty_field:{field}"
    return None


def make_idea_divergence_node(
    llm_call: LLMCall, evidence_lookup: EvidenceLookup | None = None
) -> Callable[[IdeationConvState], dict]:
    """정의된 문제 하나에 대해 해결 원리가 서로 다른 방향을 3개 이상 만든다. 정지 없이
    바로 idea_conflict_and_merge 1라운드로 이어진다."""

    def node(state: IdeationConvState) -> dict:
        retrieved = call_evidence_lookup(
            evidence_lookup, "planning_expert", _contest_query(state), runtime_scope=_runtime_scope_for(state)
        )
        prompt = build_ideation_conv_idea_divergence_prompt(
            state["notice_and_criteria"], retrieved, state.get("problem_definition")
        )
        raw, ok, attempts = _safe_call_structured_json(
            llm_call, prompt, _validate_idea_divergence_response, "idea_divergence"
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"phase": "failed", "failed_node": "idea_divergence", "llm_calls_used": used}
        directions = [
            {**direction, "status": "active", "origin_direction_ids": []} for direction in raw["solution_directions"]
        ]
        evolution_records = [
            _new_evolution_record(
                state,
                stage="idea_divergence",
                action_type="divergence",
                title=direction["title"],
                content=direction["core_principle"],
                changed_by="planning_expert",
                target_direction_ids=[direction["direction_id"]],
            )
            for direction in directions
        ]
        listing = "\n".join(f"- {d['title']}" for d in directions)
        message = _build_message(
            persona_id="planning_expert",
            round_number=state["round"],
            message_type="opinion",
            content=f"이 문제를 풀 수 있는 서로 다른 방향을 정리했습니다.\n{listing}",
            referenced_message_ids=[],
            evidence=[],
        )
        return {
            "solution_directions": directions,
            "idea_evolution": evolution_records,
            "messages": [message],
            "llm_calls_used": used,
        }

    return node


# ============================================================================
# 5. idea_conflict_and_merge — 기획/개발 위원의 반론·결합 라운드(상한까지 자동 반복)
# ============================================================================

_VALID_EVENT_SPEAKERS = {"planning_expert", "dev_expert"}
_VALID_ACTION_TYPES = {"critique", "merge", "revision", "drop", "agreement"}


def _validate_conflict_merge_response(raw: dict, active_direction_ids: set[str] | None = None) -> str | None:
    events = raw.get("round_events")
    if not isinstance(events, list) or len(events) < 2:
        return "round_events_count_invalid"
    critique_indexes: list[int] = []
    planning_change_indexes: list[int] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            return "round_event_not_object"
        if event.get("issued_by") not in _VALID_EVENT_SPEAKERS:
            return "invalid_issued_by"
        action_type = event.get("action_type")
        if action_type not in _VALID_ACTION_TYPES:
            return "invalid_action_type"
        if action_type == "critique" and event.get("issued_by") == "dev_expert":
            critique_indexes.append(index)
        if action_type in ("merge", "revision") and event.get("issued_by") == "planning_expert":
            planning_change_indexes.append(index)
        if _blank(event.get("spoken_text")):
            return "missing_or_empty_field:spoken_text"
        target_ids = event.get("target_direction_ids")
        if not isinstance(target_ids, list):
            return "target_direction_ids_not_list"
        if action_type in ("critique", "merge", "revision", "drop") and not target_ids:
            return "target_direction_ids_empty"
        if active_direction_ids is not None and any(target_id not in active_direction_ids for target_id in target_ids):
            return "target_direction_id_not_active"
    if not critique_indexes:
        return "developer_critique_missing"
    if len(planning_change_indexes) != 1:
        return "planning_revision_or_merge_missing_or_ambiguous"
    if critique_indexes[0] > planning_change_indexes[0]:
        return "planning_change_precedes_developer_critique"
    if planning_change_indexes[0] != len(events) - 1:
        return "planning_change_must_be_final_expert_event"

    change_event = events[planning_change_indexes[0]]
    change_type = change_event["action_type"]
    change_targets = change_event["target_direction_ids"]
    if change_type == "merge" and len(change_targets) < 2:
        return "merge_requires_multiple_targets"
    if change_type == "revision" and len(change_targets) != 1:
        return "revision_requires_single_target"
    critique_targets = {
        target_id
        for index in critique_indexes
        for target_id in (events[index].get("target_direction_ids") or [])
    }
    dropped_targets = {
        target_id
        for event in events
        if event.get("action_type") == "drop"
        for target_id in (event.get("target_direction_ids") or [])
    }
    if not critique_targets.issubset(set(change_targets) | dropped_targets):
        return "developer_critique_not_resolved"

    resulting = raw.get("resulting_direction")
    if not isinstance(resulting, dict):
        return "resulting_direction_missing"
    for field in ("direction_id", "title", "core_principle", "mechanism", "target_user_fit"):
        if _blank(resulting.get(field)):
            return f"missing_or_empty_field:resulting_direction.{field}"
    if active_direction_ids is not None and resulting["direction_id"] in active_direction_ids:
        return "resulting_direction_id_must_be_new"
    parent_ids = resulting.get("parent_direction_ids") or resulting.get("origin_direction_ids")
    if not isinstance(parent_ids, list) or set(parent_ids) != set(change_targets):
        return "resulting_direction_parent_ids_mismatch"
    for field in ("strengths", "open_assumptions"):
        value = resulting.get(field)
        if value is not None and not isinstance(value, list):
            return f"resulting_direction.{field}_not_list"
    updated_status = raw.get("updated_direction_status")
    if updated_status is not None and not isinstance(updated_status, list):
        return "updated_direction_status_not_list"
    return None


def _materialize_direction_change(
    directions: list[dict], events: list[dict], resulting: dict
) -> tuple[list[dict], dict, list[dict]]:
    """검증된 기획 revision/merge를 실제 방향 버전 전이로 반영한다.
    LLM의 updated_direction_status는 신뢰하지 않고 action/target으로 상태를 결정한다."""
    change_event = next(
        event
        for event in events
        if event.get("issued_by") == "planning_expert" and event.get("action_type") in ("revision", "merge")
    )
    change_type = change_event["action_type"]
    parent_ids = list(change_event["target_direction_ids"])
    dropped_ids = {
        target_id
        for event in events
        if event.get("action_type") == "drop"
        for target_id in (event.get("target_direction_ids") or [])
    }
    before = [direction for direction in directions if direction.get("direction_id") in parent_ids]
    updated = []
    for direction in directions:
        direction_id = direction.get("direction_id")
        if direction_id in parent_ids:
            status = "merged" if change_type == "merge" else "superseded"
            updated.append({**direction, "status": status})
        elif direction_id in dropped_ids:
            updated.append({**direction, "status": "dropped"})
        else:
            updated.append(direction)
    new_direction = {
        "direction_id": resulting["direction_id"],
        "title": resulting["title"],
        "core_principle": resulting["core_principle"],
        "mechanism": resulting["mechanism"],
        "target_user_fit": resulting["target_user_fit"],
        "status": "active",
        "origin_direction_ids": parent_ids,
        "parent_direction_ids": parent_ids,
        "strengths": list(resulting.get("strengths") or []),
        "open_assumptions": list(resulting.get("open_assumptions") or []),
    }
    return updated + [new_direction], change_event, before


_EVENT_ACTION_TO_MESSAGE_TYPE = {
    "critique": "disagreement",
    "agreement": "agreement",
}


def make_idea_conflict_and_merge_node(
    llm_call: LLMCall, evidence_lookup: EvidenceLookup | None = None
) -> Callable[[IdeationConvState], dict]:
    """기획/개발 위원이 현재 유효한(status="active") 해결 방향들을 놓고 반론·수정·결합하는
    라운드 하나를 만든다. 그래프 조건부 간선(_route_after_conflict_merge)이 이 노드를
    최소 조건 충족 또는 MAX_CONFLICT_ROUNDS 도달까지 반복 호출한다 — 이 노드 자신은 몇
    라운드째인지만 알 뿐 반복 여부를 스스로 정하지 않는다(기존 discussion_facilitator/
    _route_next_expert_turn과 같은 원칙: 반복 제어는 라우터가, 콘텐츠 생성은 노드가)."""

    def node(state: IdeationConvState) -> dict:
        retrieved = call_evidence_lookup(
            evidence_lookup, "dev_expert", _contest_query(state), runtime_scope=_runtime_scope_for(state)
        )
        all_directions = state.get("solution_directions") or []
        active_directions = [d for d in all_directions if d.get("status") == "active"]
        round_number = state.get("conflict_round_count", 0) + 1
        prompt = build_ideation_conv_conflict_merge_prompt(
            state["notice_and_criteria"],
            retrieved,
            state.get("problem_definition"),
            active_directions,
            state.get("idea_evolution") or [],
            round_number,
            MAX_CONFLICT_ROUNDS,
            provisional_idea=state.get("provisional_idea"),
            validation_result=state.get("validation_result"),
            required_changes=(state.get("validation_result") or {}).get("required_changes") or [],
        )
        active_direction_ids = {direction["direction_id"] for direction in active_directions}
        raw, ok, attempts = _safe_call_structured_json(
            llm_call,
            prompt,
            lambda payload: _validate_conflict_merge_response(payload, active_direction_ids),
            "idea_conflict_and_merge",
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            # 구조화 응답에 기획 revision/merge가 끝내 없으면 후보 생성으로 조용히
            # 넘어가지 않는다. 라운드 상한으로 보내 사용자 조정 선택지를 노출한다.
            return {"conflict_round_count": MAX_CONFLICT_ROUNDS, "llm_calls_used": used}

        events = raw["round_events"]
        updated_directions, change_event, before_directions = _materialize_direction_change(
            all_directions, events, raw["resulting_direction"]
        )
        resulting_direction = updated_directions[-1]
        messages = []
        evolution_records = []
        for event in events:
            speaker = event["issued_by"]
            action_type = event["action_type"]
            message_type = _EVENT_ACTION_TO_MESSAGE_TYPE.get(action_type, "opinion")
            messages.append(
                _build_message(
                    persona_id=speaker,
                    round_number=state["round"],
                    message_type=message_type,
                    content=event["spoken_text"],
                    referenced_message_ids=[],
                    evidence=[],
                )
            )
            is_material_change = event is change_event
            evolution_records.append(
                _new_evolution_record(
                    state,
                    stage="idea_conflict_and_merge",
                    action_type=action_type,
                    title=sanitize_preview(event["spoken_text"], limit=40),
                    content=event.get("detail") or event["spoken_text"],
                    changed_by=speaker,
                    target_direction_ids=[i for i in (event.get("target_direction_ids") or []) if not _blank(i)],
                    before=before_directions if is_material_change else None,
                    after=resulting_direction if is_material_change else None,
                    result_direction_id=resulting_direction["direction_id"] if is_material_change else None,
                )
            )

        # 반론·수정 결과가 canonical 회의록 안에서도 한눈에 닫히도록, 실제 round_events를
        # 근거로 진행자 요약을 덧붙인다. 별도 LLM 추론이나 프런트 가짜 메시지가 아니라
        # 방금 검증을 통과한 서버 이벤트를 결정론적으로 요약한 메시지다.
        before_titles = ", ".join(direction["title"] for direction in before_directions)
        change_verb = "결합" if change_event["action_type"] == "merge" else "수정"
        change_summary = (
            f"'{before_titles}' 방향이 기획위원의 {change_verb}을 거쳐 "
            f"'{resulting_direction['title']}' 방향으로 변경됐습니다."
        )
        messages.append(
            _build_message(
                persona_id="ideation_facilitator",
                round_number=state["round"],
                message_type="summary",
                content=change_summary,
                referenced_message_ids=[message["message_id"] for message in messages],
                evidence=[],
            )
        )

        return {
            "solution_directions": updated_directions,
            "idea_evolution": evolution_records,
            "messages": messages,
            "conflict_round_count": round_number,
            "llm_calls_used": used,
        }

    return node


def _route_after_conflict_merge(state: IdeationConvState) -> str:
    """요청 6번 — 최소 조건(해결 방향 3개 이상/반론 1회 이상/수정·결합 1회 이상)을
    충족하면 곧바로 검증 단계로 자동 진행한다. 라운드 상한
    (MAX_CONFLICT_ROUNDS)에 도달했는데도 조건을 못 채우면 자동 진행하지 않고 사용자에게
    결합/방향추가/방향폐기/검증진행 중 하나를 요청한다(awaiting_conflict_resolution) —
    상한 도달만으로 조건 없이 다음 단계로 넘기지 않는다.

    용준/Claude(2026-07-28, 요청: "위원들이 결합하는 방식으로" 카드 선택 단계 제거) —
    "proceed"의 그래프 목적지가 기존 candidate_planning(후보 카드 나열 → 사용자 선택)에서
    provisional_from_merge(위원들이 이미 결합한 방향을 카드 없이 바로 검증 대상으로
    채택)로 바뀌었다. 이 함수 자체의 반환값("proceed" 문자열)은 바뀌지 않는다 — 그래프
    배선(ideation_conv_build.py)만 그 값을 다른 노드로 매핑한다."""
    if state.get("phase") == "failed":
        route = "failed"
    elif meets_conflict_and_merge_min_conditions(state):
        route = "proceed"
    elif state.get("conflict_round_count", 0) >= MAX_CONFLICT_ROUNDS:
        route = "ask_user"
    else:
        route = "continue"
    trace_event(
        "IDEATION_GRAPH_ROUTE",
        source="idea_conflict_and_merge",
        target={
            "failed": "end",
            "proceed": "provisional_from_merge",
            "ask_user": "await_conflict_resolution",
            "continue": "idea_conflict_and_merge",
        }[route],
        phase=state.get("phase"),
        solution_direction_count=solution_direction_count(state),
        critique_count=critique_count(state),
        merge_or_revision_count=merge_or_revision_count(state),
        conflict_round_count=state.get("conflict_round_count", 0),
    )
    return route


# ============================================================================
# 5.5. provisional_from_merge — 위원들이 결합한 해결 방향을 카드 선택 없이 곧바로
#      provisional_idea로 채택(결정론적, LLM 미사용)
# ============================================================================


def make_provisional_from_merge_node(
    index_target_evidence: IndexTargetEvidenceFn | None = None,
) -> Callable[[IdeationConvState], dict]:
    """용준/Claude(2026-07-28, 요청: "예전에는 후보 카드를 나열해서 사용자가 고르면
    회의를 했는데, 이제는 위원들이 결합하는 방식으로") — idea_conflict_and_merge가
    최소 조건(해결 방향 3개 이상/반론 1회 이상/수정·결합 1회 이상)을 충족해 "proceed"로
    라우팅되면, 더 이상 candidate_planning/candidate_feasibility로 후보 카드를 나열하고
    사용자 선택을 기다리지 않는다 — 위원들이 이미 반론·결합까지 끝낸 결과를 그대로
    검증 대상(provisional_idea)으로 채택하고 곧바로 validate_planning으로 이어간다.

    active 방향이 라운드 반복(1회 초과, conflict_resolution의 "결합/추가" 요청 등)으로
    여러 개 남을 수 있다 — `_materialize_direction_change`(ideation_conv_problem.py)가
    매 라운드 결합/수정 결과를 항상 solution_directions 리스트 끝에 append하므로, 리스트의
    마지막 active 방향이 항상 "가장 최근에 결합·수정된 방향"이다. 사용자 확인: 그 방향만
    자동 채택하고 나머지 active 방향은 별도 라운드 반복이나 사용자 개입 없이 조용히
    버린다(다만 idea_evolution에 기록은 남긴다 — 감사 이력 보존)."""

    def node(state: IdeationConvState) -> dict:
        all_directions = state.get("solution_directions") or []
        active_directions = [d for d in all_directions if d.get("status") == "active"]
        if not active_directions:
            # 이론상 idea_divergence가 항상 3개 이상을 만들고 나서만 이 노드에 도달하므로
            # 발생하지 않아야 하지만(방어적 처리), 안전하게 실패 처리한다.
            return {"phase": "failed", "failed_node": "provisional_from_merge"}

        adopted = active_directions[-1]
        remaining = active_directions[:-1]
        problem_definition = state.get("problem_definition")
        idea = solution_direction_to_idea(problem_definition, adopted, label=adopted.get("title") or "채택된 방향")
        idea["source"] = "committee_merge"
        idea["source_direction_ids"] = [adopted.get("direction_id")] if adopted.get("direction_id") else []

        evolution_records = []
        if remaining:
            evolution_records.append(
                _new_evolution_record(
                    state,
                    stage="provisional_from_merge",
                    action_type="auto_adopted",
                    title=f"'{idea['title']}' 방향을 검증 대상으로 자동 채택",
                    content=(
                        "결합 조건 충족 후 가장 최근에 결합·수정된 방향을 검증 대상으로 자동 "
                        f"채택했습니다. 함께 남아있던 다른 방향({', '.join(d.get('title', '') for d in remaining)})은 "
                        "이번 검증에는 반영하지 않습니다."
                    ),
                    changed_by="ideation_facilitator",
                    target_direction_ids=[adopted.get("direction_id")] if adopted.get("direction_id") else [],
                )
            )

        selected_idea_document_id = _index_selected_candidate(
            state=state,
            idea=idea,
            source_ids=idea["source_direction_ids"],
            index_target_evidence=index_target_evidence,
        )

        bridge_message = _build_message(
            persona_id="ideation_facilitator",
            round_number=state["round"],
            message_type="summary",
            content=f"위원회가 결합한 '{idea['title']}' 방향을 검증 대상으로 채택했습니다.",
            referenced_message_ids=[],
            evidence=[],
        )

        return {
            "provisional_idea": idea,
            "selected_idea_document_id": selected_idea_document_id,
            "selected_idea": None,
            "idea_locked": False,
            "validation_result": None,
            # 용준/Claude(2026-07-28) — 여기서 validation_revise_count를 리셋하지 않는다.
            # 이 노드는 idea_conflict_and_merge가 "proceed"할 때마다 실행되는데, 그 "proceed"가
            # 검증 실패 후 재결합 라운드에서 온 것일 수도 있다(정확히 그 경우를 위해 이
            # 카운터가 존재한다) — 여기서 0으로 되돌리면 make_technical_validation_node가
            # 늘려놓은 값이 매 재결합마다 지워져 자동 반복 상한이 무력화된다.
            "idea_evolution": evolution_records,
            "messages": [bridge_message],
            "phase": "idea_validation",
        }

    return node


# ============================================================================
# 6. conflict_resolution — 라운드 상한 도달 시 사용자의 결합/추가/폐기/검증진행 요청 해석
# ============================================================================

_CONFLICT_RESOLUTION_QUESTION_TEMPLATE = (
    "아직 {reason} 회의가 자동으로 정리되지 않았습니다. 다음 중 하나를 알려주세요: "
    "① 두 방향 결합(예: '1번과 2번 결합') ② 새 해결 방향 추가 요청 "
    "③ 특정 방향 폐기(예: '2번 제외') ④ 현재 방향으로 검증 진행."
)

_PROCEED_KEYWORDS = ("검증 진행", "진행해", "이대로 진행", "그대로 진행")
_ADD_DIRECTION_KEYWORDS = ("방향 추가", "새로운 방향", "다른 방향 추가", "방향을 추가")
_DROP_KEYWORDS = ("제외", "폐기", "빼줘", "빼주세요")
_COMBINE_KEYWORDS = ("결합", "합쳐", "합치")


def _conflict_resolution_reason(state: IdeationConvState) -> str:
    if solution_direction_count(state) < 3:
        return "해결 방향이 3개 미만이라"
    if critique_count(state) < 1:
        return "아직 실질적인 반론이 없어서"
    if merge_or_revision_count(state) < 1:
        return "아직 방향이 수정·결합된 적이 없어서"
    return "조건이 아직 충족되지 않아서"


def _drop_directions(
    state: IdeationConvState, directions: list[dict], direction_ids: list[str], text: str
) -> dict | None:
    """direction_ids로 명시된 방향들을 status="dropped"로 바꾼다. 유효한 id가 하나도
    없으면 None을 반환해 호출부가 다른 해석(자연어 폴백 등)으로 넘어가게 한다."""
    valid_ids = {d.get("direction_id") for d in directions}
    dropped_ids = [d for d in direction_ids if d in valid_ids]
    if not dropped_ids:
        return None
    updated = [{**d, "status": "dropped"} if d.get("direction_id") in dropped_ids else d for d in directions]
    evolution_record = _new_evolution_record(
        state,
        stage="conflict_resolution",
        action_type="drop",
        title="사용자 요청으로 방향 폐기",
        content=text or f"{', '.join(dropped_ids)} 폐기",
        changed_by="user",
        target_direction_ids=dropped_ids,
    )
    return {"solution_directions": updated, "idea_evolution": [evolution_record], "next_route": "continue"}


def _request_another_conflict_round(state: IdeationConvState, content: str) -> dict:
    """결합/방향추가 요청 — 다음 idea_conflict_and_merge 라운드가 이 사용자 개입을
    idea_evolution_so_far 맥락으로 보고 반영하도록 기록만 남기고 라운드를 한 번 더
    돌린다. 라운드 상한을 그대로 두면 다음 라운드에서 즉시 다시 상한에 걸리므로, 사용자가
    명시적으로 방향을 지시한 이번 한 라운드만 상한을 유예한다(무한 반복 방지는
    conflict_round_count가 계속 증가하고 _route_after_conflict_merge가 매번 다시
    검사하므로 유지된다)."""
    evolution_record = _new_evolution_record(
        state, stage="conflict_resolution", action_type="revision", title="사용자 개입", content=content, changed_by="user"
    )
    return {
        "idea_evolution": [evolution_record],
        "conflict_round_count": max(0, state.get("conflict_round_count", 0) - 1),
        "next_route": "continue",
    }


def _return_to_problem_definition(state: IdeationConvState, content: str) -> dict:
    """요청 4번 — return_to_problem_definition 액션. 해결 방향/반론·결합 이력을 비우고
    problem_definition 노드로 되돌아간다(같은 problem_focus로 재구체화 — 문제 영역 선택
    자체를 다시 하지는 않는다, 그건 problem_discovery/problem_focus_selection의 역할).
    idea_evolution에는 되돌아간 사실을 기록으로 남긴다(요청: 아이디어 변화 이력 보존)."""
    evolution_record = _new_evolution_record(
        state,
        stage="conflict_resolution",
        action_type="revision",
        title="문제 정의 단계로 복귀",
        content=content,
        changed_by="user",
    )
    return {
        "idea_evolution": [evolution_record],
        "solution_directions": [],
        "conflict_round_count": 0,
        "next_route": "return_to_problem_definition",
    }


def make_conflict_resolution_node(llm_call: LLMCall) -> Callable[[IdeationConvState], dict]:
    """idea_conflict_and_merge가 라운드 상한에 도달했는데도 최소 조건을 못 채웠을 때만
    진입한다. 결합/방향추가/방향폐기/검증진행 요청을 결정적 키워드로 해석한다(LLM 호출
    없음 — 4가지 행동만 구분하면 되므로 결정적 판단으로 충분하다, 요청 5번 원칙과 동일)."""

    def node(state: IdeationConvState) -> dict:
        last_answer = _last_user_answer(state["messages"])
        text = _normalize((last_answer or {}).get("content", ""))
        directions = state.get("solution_directions") or []

        # 용준/Claude(2026-07-27, 후속 요청 4번: 구조화된 사용자 액션 지원) — action code가
        # 있으면 우선 사용하고, 없거나 처리할 수 없으면 아래 기존 자연어 키워드 판정으로
        # 폴백한다(하위 호환). direction_ids 기반 처리는 텍스트에서 번호를 다시 추출하는
        # _extract_selected_areas보다 결정적이다 — 프론트가 이미 알고 있는 정확한
        # direction_id를 그대로 넘기기 때문이다.
        action = state.get("pending_user_action") or {}
        action_code = action.get("code")
        action_payload = action.get("payload") or {}

        if action_code == "proceed_to_validation":
            return {"next_route": "proceed"}

        if action_code == "return_to_problem_definition":
            return _return_to_problem_definition(state, text or "사용자가 문제 정의로 돌아가기를 요청했습니다.")

        if action_code == "drop_direction":
            direction_ids = [d for d in action_payload.get("direction_ids", []) if isinstance(d, str)]
            result = _drop_directions(state, directions, direction_ids, text)
            if result is not None:
                return result

        if action_code == "merge_directions":
            direction_ids = [d for d in action_payload.get("direction_ids", []) if isinstance(d, str)]
            if len(direction_ids) >= 2:
                return _request_another_conflict_round(
                    state, text or f"사용자가 {', '.join(direction_ids)} 결합을 요청했습니다."
                )

        if action_code == "add_solution_direction":
            return _request_another_conflict_round(
                state, text or "사용자가 새로운 해결 방향 추가를 요청했습니다."
            )

        if any(keyword in text for keyword in _PROCEED_KEYWORDS):
            # 사용자가 명시적으로 "이대로 검증 진행"을 요청했다 — 조건 미충족이어도 사용자
            # 판단을 존중해 후보 압축 단계로 보낸다.
            return {"next_route": "proceed"}

        if any(keyword in text for keyword in _DROP_KEYWORDS):
            selected = _extract_selected_areas(text, directions)  # 재사용: 번호 추출 로직은 동일
            if selected:
                dropped_ids = {d.get("direction_id") for d in selected}
                updated = [
                    {**d, "status": "dropped"} if d.get("direction_id") in dropped_ids else d for d in directions
                ]
                evolution_record = _new_evolution_record(
                    state,
                    stage="conflict_resolution",
                    action_type="drop",
                    title="사용자 요청으로 방향 폐기",
                    content=text,
                    changed_by="user",
                    target_direction_ids=list(dropped_ids),
                )
                return {
                    "solution_directions": updated,
                    "idea_evolution": [evolution_record],
                    "next_route": "continue",
                }

        if any(keyword in text for keyword in _COMBINE_KEYWORDS) or any(
            keyword in text for keyword in _ADD_DIRECTION_KEYWORDS
        ):
            # 결합/방향 추가 요청 — 다음 idea_conflict_and_merge 라운드가 이 사용자 발언을
            # idea_evolution_so_far 맥락으로 보고 반영하도록, 사용자 개입 자체를 기록만
            # 남기고 라운드를 한 번 더 돌린다(사용자 발언 자체는 이미 messages에 있으므로
            # 프롬프트의 conversation_context로도 참조된다).
            evolution_record = _new_evolution_record(
                state,
                stage="conflict_resolution",
                action_type="revision",
                title="사용자 개입",
                content=text,
                changed_by="user",
            )
            # 라운드 상한을 그대로 유지하면 다음 라운드에서 즉시 다시 상한에 걸려 사용자
            # 개입이 반영될 기회가 없다 — 사용자가 명시적으로 방향을 지시한 이번 한 라운드만
            # 상한을 유예한다(무한 반복 방지: conflict_round_count는 계속 증가하고,
            # _route_after_conflict_merge는 다음 라운드 결과에서 다시 상한을 검사한다).
            return {
                "idea_evolution": [evolution_record],
                "conflict_round_count": max(0, state.get("conflict_round_count", 0) - 1),
                "next_route": "continue",
            }

        # 해석할 수 없는 응답 — 다시 묻는다.
        notice = _build_message(
            persona_id="ideation_facilitator",
            round_number=state["round"],
            message_type="question",
            content=_CONFLICT_RESOLUTION_QUESTION_TEMPLATE.format(reason=_conflict_resolution_reason(state)),
            referenced_message_ids=[],
            evidence=[],
        )
        return {"messages": [notice], "phase": "awaiting_conflict_resolution"}

    return node


# ============================================================================
# 7. provisional_selection — 기존 candidate_selection 노드를 그대로 재사용하되, 결과를
#    selected_idea가 아니라 provisional_idea로 재해석한다(선택 즉시 확정 방지).
# ============================================================================


def make_provisional_selection_node(
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
    index_target_evidence: IndexTargetEvidenceFn | None = None,
) -> Callable[[IdeationConvState], dict]:
    """요청 3·4번 — "잠정 후보 선택"과 "최종 아이디어 확정"을 상태상 분리한다. 실제 선택/
    결합/재추천/전문가추천 해석 로직은 전혀 새로 만들지 않고 기존
    make_candidate_selection_node()를 블랙박스로 그대로 호출한다(그 노드의 프롬프트·검증·
    단위 테스트는 전혀 변경되지 않는다) — 그 결과에 "selected_idea"가 나타나면(=선택/결합이
    확정된 경우) provisional_idea로 옮기고 phase를 "idea_validation"으로 바꿔치기할 뿐이다."""
    inner_node = make_candidate_selection_node(llm_call, evidence_lookup, index_target_evidence=index_target_evidence)

    def node(state: IdeationConvState) -> dict:
        result = dict(inner_node(state))
        if "selected_idea" not in result:
            # 재추천(phase="candidate_generation")/재질문(awaiting_candidate_selection 유지)/
            # 실패(phase="failed") — 그대로 통과시킨다.
            return result

        provisional_idea = result.pop("selected_idea")
        # selected_idea_document_id는 그대로 둔다 — 색인된 target evidence(_resolve_selection이
        # index_target_evidence로 이미 만든 문서)는 확정 여부와 무관하게 잠정 후보에 대한
        # RAG 검색(idea_validation 등)이 계속 참조해야 한다. user_idea/initial_idea는
        # 팝(제거)한다 — 이 필드들은 "현재 작업 중인 아이디어"라는 의미로 refinement 쪽
        # 코드가 읽는데, 아직 확정 전이므로 여기서 채우지 않는다(concept_confirmation의
        # 확정 분기가 실제 확정 시점에 채운다, 요청 3번).
        result.pop("user_idea", None)
        result.pop("initial_idea", None)
        result["provisional_idea"] = provisional_idea
        result["selected_idea"] = None
        result["idea_locked"] = False
        result["validation_result"] = None
        result["validation_revise_count"] = 0
        result["phase"] = "idea_validation"
        return result

    return node


# ============================================================================
# 8. idea_validation — 잠정 후보를 기획/개발 관점에서 검증
# ============================================================================


def _validate_validation_section_response(raw: dict) -> str | None:
    """용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다") —
    기획/개발 검증 프롬프트가 이제 각각 flat한 단일 섹션(status/passed_items/issues/
    revision_suggestions/message/claims)을 반환한다. 세부 필드 누락과 허용값 이탈은
    아래 정규화 단계(_normalize_validation_section)에서 안전하게 보정하므로, 응답 자체가
    JSON 객체가 아닌 경우에만 재시도한다."""
    if not isinstance(raw, dict):
        return "response_not_object"
    return None


def _normalize_validation_section(raw: dict, section: str) -> dict:
    if any(field in raw for field in ("status", "passed_items", "issues", "revision_suggestions", "message")):
        normalized_issues = []
        for index, issue in enumerate(raw.get("issues") or []):
            if not isinstance(issue, dict):
                continue
            description = str(issue.get("description") or "").strip()
            if not description:
                continue
            normalized_issues.append(
                {
                    "code": str(issue.get("code") or f"{section}_issue_{index + 1}").strip(),
                    "description": description,
                    "severity": (
                        issue.get("severity")
                        if issue.get("severity") in _VALIDATION_SEVERITIES
                        else "major"
                    ),
                }
            )
        supplied_status = raw.get("status")
        if supplied_status in _VALIDATION_STATUSES:
            status = supplied_status
        elif normalized_issues:
            # 용준/Claude(2026-07-28, 요청: "minor/caution 이슈로 idea_conflict_and_merge를
            # 반복하지 말고 blocking일 때만 되돌아가라") — 모델이 status 필드를 안 채워서
            # 여기 폴백으로 오는 경우, 이슈 목록에 severity="blocking"이 하나도 없으면
            # (minor/major만 있으면) needs_revision이 아니라 passed_with_caution으로 본다.
            # status를 모델이 직접 명시한 경우(위 supplied_status 분기)는 이 규칙과 무관하게
            # 그대로 존중한다 — 여기는 "이슈만 있고 판정이 없을 때"의 기본값만 바꾼다.
            status = (
                "needs_revision"
                if any(issue["severity"] == "blocking" for issue in normalized_issues)
                else "passed_with_caution"
            )
        elif raw.get("passed_items"):
            status = "passed_with_caution"
        else:
            status = "needs_revision"
        return {
            "status": status,
            "passed_items": [str(item).strip() for item in raw.get("passed_items") or [] if str(item).strip()],
            "issues": normalized_issues,
            "revision_suggestions": [
                str(item).strip()
                for item in raw.get("revision_suggestions") or []
                if str(item).strip()
            ],
            "message": str(raw.get("message") or "검증 결과의 세부 형식을 보완해야 합니다.").strip(),
        }

    concerns = [str(item).strip() for item in raw.get("concerns") or [] if str(item).strip()]
    legacy_fields = (
        (
            "value_worth_solving",
            "target_user_clarity",
            "differentiation",
            "usage_motivation",
            "contest_alignment",
        )
        if section == "planning"
        else (
            "data_availability",
            "feasibility",
            "ai_necessity",
            "privacy_or_security_risks",
            "prototype_feasibility",
        )
    )
    passed_items = [str(raw.get(field) or "").strip() for field in legacy_fields if str(raw.get(field) or "").strip()]
    issues = [
        {
            "code": f"legacy_{section}_concern_{index + 1}",
            "description": concern,
            "severity": "major",
        }
        for index, concern in enumerate(concerns)
    ]
    return {
        "status": "needs_revision" if issues else "passed",
        "passed_items": passed_items,
        "issues": issues,
        "revision_suggestions": list(concerns),
        "message": concerns[0] if concerns else (passed_items[0] if passed_items else "검증을 완료했습니다."),
    }


def _safe_validation_message(message: str, fallback: str, retrieved: list[dict]) -> str:
    text = str(message or "").strip()
    copied_raw = False
    normalized_text = " ".join(text.split())
    for item in retrieved:
        evidence_text = str(item.get("quote") or item.get("text") or "").strip()
        if len(evidence_text) >= 32 and " ".join(evidence_text.split())[:32] in normalized_text:
            copied_raw = True
            break
    if not text or copied_raw or _RAW_DOCUMENT_MESSAGE_RE.search(text):
        return fallback
    return text


def _idea_validation_retry_note(reason: str) -> str:
    if reason.startswith("missing_or_empty_field:") or reason.endswith("_missing"):
        return (
            "\n\n[재시도 지시] 이전 응답에서 planning/technical 각 섹션에 status"
            "(passed/passed_with_caution/needs_revision), passed_items, issues, "
            "revision_suggestions, message 필드가 누락되었거나 비어 있었습니다. "
            "두 섹션 모두 이 다섯 필드를 빠짐없이 채워 다시 응답하세요."
        )
    if reason.startswith("invalid_status:"):
        return "\n\n[재시도 지시] status는 passed/passed_with_caution/needs_revision 중 하나만 사용하세요."
    if reason.startswith("invalid_issue:"):
        return (
            "\n\n[재시도 지시] issues의 각 항목에 code, description, "
            "severity(minor/major/blocking)를 모두 채우세요."
        )
    return "\n\n[재시도 지시] 출력 규칙의 JSON 스키마를 정확히 지켜 다시 응답하세요."


def _route_after_idea_validation(state: IdeationConvState) -> str:
    if state.get("phase") == "failed":
        return "failed"
    return "revise" if state.get("phase") == "idea_conflict_and_merge" else "confirm"


def _effective_validation_status(section: dict) -> str:
    """용준/Claude(2026-07-28, 요청: "blocking 이슈가 있을 때만 수정 단계로 복귀") — 모델이
    명시한 status(needs_revision 등)를 severity와 무관하게 그대로 존중하면, 검증 프롬프트가
    "major도 반드시 수정해야 하는 문제"라고 가르쳐서 모델이 major 이슈만 있어도 스스로
    status를 needs_revision으로 써버린다. LLM이 지시를 어겨도 결과가 흔들리지 않도록 여기서
    severity를 최종 판단 기준으로 삼는다 — 모델이 뭐라고 쓰든 blocking 이슈가 없으면
    needs_revision으로 취급하지 않는다. 이 override가 곧 "표시되는 status"이기도 해서,
    blocking이 없는 이슈(minor/major)는 passed_with_caution으로 내려가 프론트의 "남은
    주의사항"에 그대로 노출된다(status==='passed_with_caution'일 때만 issues를 보여주는
    기존 프론트 로직과 맞물림)."""
    if any(issue.get("severity") == "blocking" for issue in section["issues"]):
        return "needs_revision"
    if section["issues"] or section["status"] == "passed_with_caution":
        return "passed_with_caution"
    return "passed"


def _validation_unavailable_fallback(section: str, label: str) -> dict:
    return {
        "status": "needs_revision",
        "passed_items": [],
        "issues": [
            {
                "code": f"{section}_validation_unavailable",
                "description": f"{label} 검증 응답 형식을 확인할 수 없어 재검증이 필요합니다.",
                # 용준/Claude(2026-07-28) — 검증 자체가 불가능했던 경우라 severity를 major가
                # 아니라 blocking으로 둔다. needs_revision 판정이 blocking 유무만 보므로,
                # major였다면 이 안전장치가 무력화되어 파싱 실패에도 그냥 주제 확정으로
                # 넘어가 버린다.
                "severity": "blocking",
            }
        ],
        "revision_suggestions": [f"{label} 검증 항목을 유지한 상태로 다시 검증"],
        "message": f"{label} 검증 응답 형식을 확인하지 못해 안전하게 수정 필요로 판정했습니다.",
    }


def make_planning_validation_node(
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
    external_evidence_lookup: ExternalEvidenceLookupFn | None = None,
    ground_claims: ClaimGroundingFn | None = None,
) -> Callable[[IdeationConvState], dict]:
    """용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다" — 기존
    make_idea_validation_node(기획+개발 동시 1회 호출)를 두 노드로 분리) — provisional_idea를
    기획 관점에서 검증하고 회의 시작 발언 + 기획위원 발언을 만든다. phase는 그대로
    "idea_validation"에 머문다 — make_technical_validation_node가 이어서 실행되어야 다음
    phase(idea_conflict_and_merge/awaiting_concept_confirmation)로 넘어간다.

    ground_claims는 옵션이다(None이면 use_rag=False 세션 등 — grounding 없이 메시지를
    만든다, 하위 호환)."""

    def node(state: IdeationConvState) -> dict:
        retrieved = call_evidence_lookup(
            evidence_lookup, "planning_expert", _contest_query(state), runtime_scope=_runtime_scope_for(state)
        )
        provisional_idea = state.get("provisional_idea") or {}
        external_query = _external_evidence_query(state, [provisional_idea])
        planning_external_result = _call_external_evidence_lookup(
            external_evidence_lookup, "planning_expert", external_query
        )
        planning_external_evidence = [
            item for item in planning_external_result.get("external_evidence") or [] if isinstance(item, dict)
        ]
        prompt = build_ideation_conv_idea_validation_planning_prompt(
            state["notice_and_criteria"],
            retrieved,
            provisional_idea,
            external_research=planning_external_evidence,
        )
        raw, ok, attempts = _safe_call_structured_json(
            llm_call,
            prompt,
            _validate_validation_section_response,
            "idea_validation_planning",
            retry_note_for=_idea_validation_retry_note,
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            raw = {**_validation_unavailable_fallback("planning", "기획"), "unresolved_assumptions": ["기획 검증 응답을 다시 확인해야 합니다."]}

        planning = _normalize_validation_section(raw, "planning")
        normalized_evidence = [
            {
                **item,
                "quote": str(item.get("quote") or item.get("text") or "").strip(),
            }
            for item in retrieved
            if isinstance(item, dict)
        ]
        normalized_planning_external = [
            {**item, "quote": str(item.get("quote") or item.get("text") or "").strip()}
            for item in planning_external_evidence
        ]
        # 용준/Claude(2026-07-28, 요청: "위원들이 RAG를 근거로 회의를 진행" + 화면 "근거 보기"
        # 복구) — claims(위 프롬프트 [근거 인용 규칙]로 요구)를 실제 검색된 근거
        # (normalized_evidence, "ref" 보존됨)와 대조 검증한다. ground_claims가 None이면
        # (use_rag=False 세션 등) 기존과 동일하게 grounding 없이 진행한다 — 외부 참고자료는
        # 이 claims 검증 대상이 아니다.
        planning_grounding = ground_claims("planning_expert", raw.get("claims"), normalized_evidence) if ground_claims else None
        planning["message"] = _safe_validation_message(
            planning["message"],
            "기획 관점의 검증 항목과 보완 필요 여부를 확인했습니다.",
            normalized_evidence + normalized_planning_external,
        )
        planning["status"] = _effective_validation_status(planning)
        unresolved = [a for a in (raw.get("unresolved_assumptions") or []) if a]
        start_message = _build_message(
            persona_id="ideation_facilitator",
            round_number=state["round"],
            message_type="summary",
            content=(
                "선택한 후보는 아직 최종 확정되지 않았습니다. "
                "기획 관점과 개발 관점에서 차례로 검증하겠습니다."
            ),
            referenced_message_ids=[],
            evidence=[],
        )
        planning_message = _build_message(
            persona_id="planning_expert",
            round_number=state["round"],
            message_type="opinion",
            content=planning["message"],
            referenced_message_ids=[start_message["message_id"]],
            evidence=normalized_evidence + normalized_planning_external,
            structured={"validation": planning},
            grounding=planning_grounding,
        )
        merged_external = _merge_external_evidence_results(
            {"external_evidence": state.get("external_evidence", []), **(state.get("external_evidence_meta") or {})},
            planning_external_result,
        )
        return {
            "validation_result": {"planning": planning},
            "unresolved_issues": list(state["unresolved_issues"]) + [a for a in unresolved if a not in state["unresolved_issues"]],
            "messages": [start_message, planning_message],
            "selected_idea": None,
            "idea_locked": False,
            "llm_calls_used": used,
            "external_evidence": merged_external["external_evidence"],
            "external_evidence_meta": {
                "used_dataset_search": merged_external["used_dataset_search"],
                "used_public_api_search": merged_external["used_public_api_search"],
                "warnings": merged_external["warnings"],
            },
        }

    return node


def make_technical_validation_node(
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
    external_evidence_lookup: ExternalEvidenceLookupFn | None = None,
    ground_claims: ClaimGroundingFn | None = None,
) -> Callable[[IdeationConvState], dict]:
    """용준/Claude(2026-07-28) — make_planning_validation_node 직후 실행되어 provisional_idea를
    개발 관점에서 검증한다. 방금 만들어진 planning 검증 결과(state["validation_result"]
    ["planning"])를 프롬프트에 주입해 "기획위원 발언을 보고 이어 말하는" 순서를 만든다. 두
    관점이 모두 준비된 뒤에야(이 노드가 끝난 뒤에야) needs_revision/확정 대기 라우팅을
    결정한다 — _route_after_idea_validation은 이 노드의 조건부 엣지로만 연결한다."""

    def node(state: IdeationConvState) -> dict:
        retrieved = call_evidence_lookup(
            evidence_lookup, "dev_expert", _contest_query(state), runtime_scope=_runtime_scope_for(state)
        )
        provisional_idea = state.get("provisional_idea") or {}
        planning = (state.get("validation_result") or {}).get("planning") or {}
        external_query = _external_evidence_query(state, [provisional_idea])
        technical_external_result = _call_external_evidence_lookup(
            external_evidence_lookup, "dev_expert", external_query
        )
        technical_external_evidence = [
            item for item in technical_external_result.get("external_evidence") or [] if isinstance(item, dict)
        ]
        prompt = build_ideation_conv_idea_validation_technical_prompt(
            state["notice_and_criteria"],
            retrieved,
            provisional_idea,
            planning,
            external_research=technical_external_evidence,
        )
        raw, ok, attempts = _safe_call_structured_json(
            llm_call,
            prompt,
            _validate_validation_section_response,
            "idea_validation_technical",
            retry_note_for=_idea_validation_retry_note,
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            raw = {**_validation_unavailable_fallback("technical", "개발"), "unresolved_assumptions": ["개발 검증 응답을 다시 확인해야 합니다."]}

        technical = _normalize_validation_section(raw, "technical")
        normalized_evidence = [
            {
                **item,
                "quote": str(item.get("quote") or item.get("text") or "").strip(),
            }
            for item in retrieved
            if isinstance(item, dict)
        ]
        normalized_technical_external = [
            {**item, "quote": str(item.get("quote") or item.get("text") or "").strip()}
            for item in technical_external_evidence
        ]
        technical_grounding = ground_claims("dev_expert", raw.get("claims"), normalized_evidence) if ground_claims else None
        technical["message"] = _safe_validation_message(
            technical["message"],
            "개발 관점의 구현 가능성과 기술 위험을 확인했습니다.",
            normalized_evidence + normalized_technical_external,
        )
        technical["status"] = _effective_validation_status(technical)

        sections = (planning, technical)
        needs_revision = any(section["status"] == "needs_revision" for section in sections)
        required_changes: list[str] = []
        for section in sections:
            candidates = section["revision_suggestions"] + [
                issue["description"]
                for issue in section["issues"]
                if issue["severity"] == "blocking"
            ]
            for item in candidates:
                if item and item not in required_changes:
                    required_changes.append(item)
        overall_status = "needs_revision" if needs_revision else (
            "passed_with_caution"
            if any(section["status"] == "passed_with_caution" for section in sections)
            else "passed"
        )
        # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거로 사라진 자연 정지점 대체) —
        # provisional_from_merge가 카드 선택 없이 바로 이 노드까지 이어지면서, "검증 실패 ->
        # idea_conflict_and_merge 재실행 -> 조건 재충족 -> 검증 -> 실패 -> ..."가 사용자
        # 개입 없이 한 그래프 호출 안에서 무한 반복될 수 있다(이전에는 candidate_feasibility가
        # 재실행마다 항상 멈춰줬다). validation_revise_count가 MAX_VALIDATION_REVISE_ROUNDS에
        # 도달하면 자동으로 되돌리지 않고 사용자 확인을 기다린다(overall_status는
        # needs_revision 그대로 노출해 화면에서 확인 가능하게 한다).
        revise_count = state.get("validation_revise_count", 0)
        auto_revise_capped = needs_revision and revise_count >= MAX_VALIDATION_REVISE_ROUNDS
        next_phase = (
            "idea_conflict_and_merge" if needs_revision and not auto_revise_capped else "awaiting_concept_confirmation"
        )
        if auto_revise_capped:
            summary_message = (
                "기획·개발 검증에서 보완이 필요한 항목을 다시 확인했습니다. "
                f"가장 중요한 수정 방향은 {required_changes[0] if required_changes else '검증 지적 사항 반영'}입니다. "
                "자동 재검토 한도에 도달해 이번에는 직접 확정하거나 재검토를 요청해 주세요."
            )
        elif needs_revision:
            summary_message = (
                "기획·개발 검증에서 보완이 필요한 항목을 확인했습니다. "
                f"가장 중요한 수정 방향은 {required_changes[0] if required_changes else '검증 지적 사항 반영'}입니다. "
                "검증 결과를 반영해 아이디어를 수정하겠습니다."
            )
        else:
            summary_message = (
                "기획·개발 검증을 통과했습니다. "
                "남은 주의사항을 확인한 뒤 이 아이디어의 최종 확정 여부를 선택해 주세요."
            )
        validation_result = {
            "planning": planning,
            "technical": technical,
            "overall_status": overall_status,
            "planning_status": planning["status"],
            "technical_status": technical["status"],
            "required_changes": required_changes,
            "summary_message": summary_message,
            "next_phase": next_phase,
        }
        unresolved = [a for a in (raw.get("unresolved_assumptions") or []) if a]
        evolution_record = _new_evolution_record(
            state,
            stage="idea_validation",
            action_type="validation",
            title="기획·개발 검증",
            content=summary_message,
            changed_by="ideation_facilitator",
            before=provisional_idea,
            after=validation_result,
        )
        planning_message_id = state["messages"][-1]["message_id"] if state.get("messages") else None
        technical_message = _build_message(
            persona_id="dev_expert",
            round_number=state["round"],
            message_type="opinion",
            content=technical["message"],
            referenced_message_ids=[planning_message_id] if planning_message_id else [],
            evidence=normalized_evidence + normalized_technical_external,
            structured={"validation": technical},
            grounding=technical_grounding,
        )
        summary_message_record = _build_message(
            persona_id="ideation_facilitator",
            round_number=state["round"],
            message_type="summary",
            content=summary_message,
            referenced_message_ids=[m for m in [planning_message_id, technical_message["message_id"]] if m],
            evidence=[],
            structured={"validation_summary": validation_result},
        )
        merged_external = _merge_external_evidence_results(
            {"external_evidence": state.get("external_evidence", []), **(state.get("external_evidence_meta") or {})},
            technical_external_result,
        )
        return {
            "validation_result": validation_result,
            "unresolved_issues": list(state["unresolved_issues"]) + [a for a in unresolved if a not in state["unresolved_issues"]],
            "idea_evolution": [evolution_record],
            "messages": [technical_message, summary_message_record],
            "phase": next_phase,
            "selected_idea": None,
            "idea_locked": False,
            # 용준/Claude(2026-07-28) — next_phase가 "idea_conflict_and_merge"일 때만(자동
            # 재검토 루프를 실제로 한 번 더 도는 경우) 늘린다. 확정 대기로 넘어가면(자동
            # 반복 없이 멈췄든 통과했든) 이번 provisional_idea에 대한 검증은 끝난 것이므로
            # 늘리지 않는다 — 사용자가 concept_confirmation에서 수동으로 재검토를 요청하면
            # 그건 이 자동 루프 카운터와 무관하다(그 경로는 conflict_round_count만 되돌린다).
            "validation_revise_count": (
                revise_count + 1 if next_phase == "idea_conflict_and_merge" else revise_count
            ),
            "llm_calls_used": used,
            "forced_next_speaker": None,
            "external_evidence": merged_external["external_evidence"],
            "external_evidence_meta": {
                "used_dataset_search": merged_external["used_dataset_search"],
                "used_public_api_search": merged_external["used_public_api_search"],
                "warnings": merged_external["warnings"],
            },
        }

    return node


# ============================================================================
# 9. concept_confirmation — 사용자의 최종 확정/재수정 요청 해석(결정적, LLM 미사용)
# ============================================================================

_CONFIRM_KEYWORDS = ("확정", "이걸로", "이대로", "좋아요", "네 진행", "동의")
_REVISE_KEYWORDS = ("다시", "수정", "바꿔", "변경", "재검토")

# 용준/Claude(2026-07-27, 요청 2번: "idea_locked=True 이후 기존 expert_discussion이
# 문제/대상 사용자/해결 원리/핵심 가치를 다시 바꿀 수 있는지 확인하고, 더 안전한 방향으로
# 수정") — B안(정리 전용 제한)을 최소 침습으로 구현한다. selected_idea 객체 자체는
# ideation_conv_nodes.py의 어떤 노드도 재할당하지 않으므로(코드 검증 완료, "selected_idea":
# state.get("selected_idea")로 참조만 함) 구조적으로 안전하지만, expert_discussion이 다음에
# 열 쟁점(active_issue)을 자동으로 고를 때(_select_next_issue_family가 remaining_topics_for
# 기준으로 TOPIC_PRIORITY를 순회) problem/target_user/core_value/differentiation을 "아직
# 안 다룬 주제"로 오인해 다시 논의 주제로 여는 것을 막는다 — concept_confirmation에서
# 확정되는 순간 이미 problem_definition+검증을 거쳐 확정된 값이므로, 이 네 주제는
# "이미 해결됨"으로 표시해 라운드테이블이 구현 세부사항(mvp/data/ai_role/roadmap)에
# 집중하게 한다. 사용자가 자유 발언으로 직접 재론의를 요청하는 것까지 막지는 않는다
# (그것은 사용자의 명시적 의도이지 시스템의 자동 드리프트가 아니다).
_CONFIRMED_CORE_TOPICS: tuple[str, ...] = ("problem", "target_user", "core_value", "differentiation")


def make_concept_confirmation_node(llm_call: LLMCall) -> Callable[[IdeationConvState], dict]:
    """사용자의 확정/재수정 응답을 해석한다. 확정 여부는 코드가 결정적으로 판단하고
    (요청 5번 — LLM이 idea_locked를 직접 세팅하지 않는다), 애매하면 안전하게 "확정 아님"
    으로 처리해 다시 명확히 묻는다."""

    def node(state: IdeationConvState) -> dict:
        last_answer = _last_user_answer(state["messages"])
        text = _normalize((last_answer or {}).get("content", ""))

        # 용준/Claude(2026-07-27, 후속 요청 4번: 구조화된 사용자 액션 지원) — action code가
        # 있으면 텍스트 키워드 판정보다 우선한다.
        action = state.get("pending_user_action") or {}
        action_code = action.get("code")
        if action_code == "return_to_problem_definition":
            return _return_to_problem_definition(
                state, text or "사용자가 문제 정의로 돌아가기를 요청했습니다."
            )
        if action_code == "choose_another_candidate":
            # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — idea_candidates가 없는
            # 세션(provisional_from_merge를 거쳐온 새 discovery 플로우)은 고를 카드 자체가
            # 없으므로 awaiting_candidate_selection으로 보내면 막다른 길이 된다. 이 경우
            # "다른 후보를 보고 싶다"는 요청을 반론·결합 라운드로 되돌아가 다시 결합하라는
            # 의미로 해석한다(concept_confirmation의 기존 "재검토" 분기와 동일한 처리).
            # idea_candidates가 있는 레거시 세션(candidate_planning을 실제로 거친 경우)은
            # 기존 동작(카드 선택 화면으로 복귀)을 그대로 유지한다.
            if not state.get("idea_candidates"):
                return {
                    "idea_evolution": [
                        _new_evolution_record(
                            state,
                            stage="concept_confirmation",
                            action_type="revision",
                            title="사용자 요청으로 다른 방향 재검토",
                            content="사용자가 다른 방향을 다시 보고 싶다고 요청했습니다.",
                            changed_by="user",
                        )
                    ],
                    "conflict_round_count": max(0, state.get("conflict_round_count", 0) - 1),
                    "next_route": "continue",
                }
            return {
                "provisional_idea": None,
                "validation_result": None,
                "selected_idea": None,
                "idea_locked": False,
                "phase": "awaiting_candidate_selection",
                "next_route": "choose_another_candidate",
            }
        if action_code == "confirm_concept":
            is_confirm = True
        elif action_code == "revise_candidate":
            is_confirm = False
        else:
            is_confirm = any(keyword in text for keyword in _CONFIRM_KEYWORDS) and not any(
                keyword in text for keyword in _REVISE_KEYWORDS
            )
        if is_confirm:
            provisional_idea = state.get("provisional_idea") or {}
            evolution_record = _new_evolution_record(
                state,
                stage="concept_confirmation",
                action_type="confirmation",
                title="사용자 최종 확정",
                content=provisional_idea.get("title", ""),
                changed_by="user",
            )
            confirm_message = _build_message(
                persona_id="ideation_facilitator",
                round_number=state["round"],
                message_type="summary",
                content=f"'{provisional_idea.get('title', '')}' 방향을 최종 주제로 확정했습니다. 이제 문서 작성을 도와드리겠습니다.",
                referenced_message_ids=[],
                evidence=[],
            )
            resolved_topics = list(state.get("resolved_topics") or [])
            for topic in _CONFIRMED_CORE_TOPICS:
                if topic not in resolved_topics:
                    resolved_topics.append(topic)
            return {
                "selected_idea": provisional_idea,
                "user_idea": provisional_idea,
                "initial_idea": provisional_idea.get("title"),
                "user_confirmed": True,
                "idea_locked": True,
                "idea_evolution": [evolution_record],
                "messages": [confirm_message],
                "next_route": "to_refinement",
                "phase": "expert_discussion",
                # 위 _CONFIRMED_CORE_TOPICS 주석 참고 — 확정된 핵심 주제를 라운드테이블
                # 자동 의제 로테이션에서 제외한다.
                "resolved_topics": resolved_topics,
            }

        # 확정하지 않음 — 더 수정하고 싶다는 뜻이므로 반론·결합 라운드로 되돌아간다.
        # conflict_round_count를 리셋하지 않는다(사용자가 원하면 다시 상한까지 반복될 수
        # 있게, 그러나 무한 루프는 여전히 MAX_CONFLICT_ROUNDS가 막는다).
        evolution_record = _new_evolution_record(
            state,
            stage="concept_confirmation",
            action_type="revision",
            title="사용자 재검토 요청",
            content=text or "사용자가 확정 대신 추가 수정을 요청했습니다.",
            changed_by="user",
        )
        return {
            "idea_evolution": [evolution_record],
            "conflict_round_count": max(0, state.get("conflict_round_count", 0) - 1),
            "next_route": "continue",
        }

    return node
