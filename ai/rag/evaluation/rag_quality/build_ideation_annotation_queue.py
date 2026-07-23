"""운영 trace에서 다중 문서 RAG 사람 검수 대기열을 만든다.

자동으로 `human_verified=true`를 만들지 않는다. 로그는 시스템이 무엇을 검색·선택·연결했는지
보여줄 뿐, 그것이 정답인지는 사람이 원문과 대조해야 하기 때문이다.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

_EVENT_RE = re.compile(r"\[(IDEATION_[A-Z0-9_]+)\]")
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_DECODER = json.JSONDecoder()


def _value(line: str, name: str, default: Any = None) -> Any:
    match = re.search(rf"(?:^|\s){re.escape(name)}=", line)
    if not match:
        return default
    raw = line[match.end() :].lstrip()
    try:
        return _DECODER.raw_decode(raw)[0]
    except (json.JSONDecodeError, TypeError):
        token = raw.split(maxsplit=1)[0] if raw else ""
        return token or default


def _actor_key(line: str) -> tuple[str, str, str]:
    return (
        str(_value(line, "session", "")),
        str(_value(line, "request", "")),
        str(_value(line, "speaker", "")),
    )


def collect_observations(lines: Iterable[str]) -> list[dict[str, Any]]:
    current: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_plan_id: dict[str, dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []

    for line in lines:
        event_match = _EVENT_RE.search(line)
        if not event_match:
            continue
        event = event_match.group(1)
        key = _actor_key(line)
        timestamp_match = _TIMESTAMP_RE.match(line)
        timestamp = timestamp_match.group(1) if timestamp_match else ""

        if event == "IDEATION_EVIDENCE_LOOKUP" and _value(line, "project_id"):
            observation = {
                "observed_at": timestamp,
                "session_id": key[0],
                "request_id": key[1],
                "project_id": str(_value(line, "project_id")),
                "persona_id": key[2],
                "issue_id": str(_value(line, "issue", "")),
                "query": str(_value(line, "query", "")),
                "retrieved_chunk_ids": list(_value(line, "chunk_ids", []) or []),
                "selected_chunk_ids": [],
                "selected_evidence_observed": [],
                "planner_fallback": False,
                "issue_match": None,
                "response_text": "",
                "observed_claim_count": 0,
                "observed_grounded_claim_count": 0,
                "observed_expert_judgment_count": 0,
                "observed_unsupported_claim_count": 0,
                "linked_claims_observed": [],
            }
            current[key] = observation
            continue

        observation = current.get(key)
        if observation is None:
            continue

        if event == "IDEATION_EVIDENCE_PLAN_SHADOW_CREATED":
            plan_id = str(_value(line, "plan_id", ""))
            observation["plan_id"] = plan_id
            effective_issue_id = str(_value(line, "effective_issue_id", ""))
            if effective_issue_id:
                observation["issue_id"] = effective_issue_id
            observation["effective_issue_title"] = str(
                _value(line, "effective_issue_title", "")
            )
            selected = list(_value(line, "selected_evidence", []) or [])
            observation["selected_evidence_observed"] = selected
            observation["selected_chunk_ids"] = [
                item.get("chunk_id") for item in selected if isinstance(item, dict) and item.get("chunk_id")
            ]
            if plan_id:
                by_plan_id[plan_id] = observation
        elif event in ("IDEATION_EVIDENCE_PLAN_FALLBACK", "IDEATION_EVIDENCE_PLAN_SHADOW_FAILED"):
            observation["planner_fallback"] = True
        elif event == "IDEATION_EVIDENCE_LINKED":
            observation["linked_claims_observed"].append(
                {
                    "claim_id": str(_value(line, "claim_id", "")),
                    "evidence_refs": list(_value(line, "evidence_refs", []) or []),
                    "linked_chunk_ids": list(_value(line, "chunk_ids", []) or []),
                }
            )
        elif event == "IDEATION_EVIDENCE_PLAN_COMPLIANCE":
            plan_id = str(_value(line, "plan_id", ""))
            observation = by_plan_id.get(plan_id, observation)
            effective_issue_id = str(_value(line, "effective_issue_id", ""))
            if effective_issue_id:
                observation["issue_id"] = effective_issue_id
            observation["issue_match"] = _value(line, "issue_match")
            observation["observed_claim_count"] = int(_value(line, "claim_count", 0) or 0)
            observation["observed_grounded_claim_count"] = int(
                _value(line, "grounded_claim_count", 0) or 0
            )
            observation["linked_chunk_ids_observed"] = list(
                _value(line, "linked_chunk_ids", []) or []
            )
        elif event == "IDEATION_TURN_END":
            observation["response_text"] = str(_value(line, "text", ""))
            observation["observed_claim_count"] = int(
                _value(line, "accepted_claim_count", observation["observed_claim_count"]) or 0
            )
            observation["observed_grounded_claim_count"] = int(
                _value(
                    line,
                    "grounded_claim_count",
                    observation["observed_grounded_claim_count"],
                )
                or 0
            )
            observation["observed_expert_judgment_count"] = int(
                _value(line, "expert_judgment_count", 0) or 0
            )
            observation["observed_unsupported_claim_count"] = int(
                _value(line, "unsupported_claim_count", 0) or 0
            )
            if observation not in observations:
                observations.append(observation)

    return observations


def build_annotation_queue(
    observations: list[dict[str, Any]],
    *,
    project_count: int = 3,
    cases_per_project: int = 5,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        # Planner 도입 전 구버전 로그나 중간에 끊긴 턴은 selected/issue_match가 비어 있어
        # 사람이 추측으로 채워야 한다. 그런 관측은 품질 점수용 대기열에서 제외한다.
        complete_planner_observation = (
            observation.get("plan_id")
            and observation.get("issue_id")
            and observation.get("issue_match") in (True, False)
        )
        if (
            observation.get("project_id")
            and observation.get("retrieved_chunk_ids")
            and complete_planner_observation
        ):
            grouped[str(observation["project_id"])].append(observation)

    eligible_projects = sorted(
        (project_id for project_id, items in grouped.items() if len(items) >= cases_per_project),
        key=lambda project_id: (-len(grouped[project_id]), project_id),
    )[:project_count]
    cases: list[dict[str, Any]] = []
    for project_id in eligible_projects:
        for index, item in enumerate(grouped[project_id][:cases_per_project], start=1):
            linked_claims = item.get("linked_claims_observed") or []
            selected_type_by_chunk = {
                evidence.get("chunk_id"): evidence.get("claim_type")
                for evidence in item.get("selected_evidence_observed") or []
                if isinstance(evidence, dict)
            }
            claims = [
                {
                    "claim_id": link.get("claim_id") or f"claim_{claim_index}",
                    "text": "",
                    "claim_type": next(
                        (
                            selected_type_by_chunk.get(chunk_id)
                            for chunk_id in link.get("linked_chunk_ids") or []
                            if selected_type_by_chunk.get(chunk_id)
                        ),
                        "",
                    ),
                    "expected_claim_type": "",
                    "supported": None,
                    "linked_chunk_ids": link.get("linked_chunk_ids") or [],
                }
                for claim_index, link in enumerate(linked_claims, start=1)
            ]
            for expert_index in range(int(item.get("observed_expert_judgment_count") or 0)):
                claims.append(
                    {
                        "claim_id": f"expert_claim_{expert_index + 1}",
                        "text": "",
                        "claim_type": "expert_judgment",
                        "expected_claim_type": "",
                        "supported": None,
                        "linked_chunk_ids": [],
                    }
                )
            cases.append(
                {
                    "case_id": (
                        f"{project_id}-{item.get('session_id')}-{item.get('plan_id') or index}"
                    ),
                    "project_id": project_id,
                    "session_id": item.get("session_id"),
                    "request_id": item.get("request_id"),
                    "observed_at": item.get("observed_at"),
                    "persona_id": item.get("persona_id"),
                    "issue_id": item.get("issue_id"),
                    "effective_issue_title": item.get("effective_issue_title", ""),
                    "query": item.get("query"),
                    "gold_relevant_chunk_ids": [],
                    "retrieved_chunk_ids": item.get("retrieved_chunk_ids") or [],
                    "selected_chunk_ids": item.get("selected_chunk_ids") or [],
                    "selected_evidence_observed": item.get("selected_evidence_observed") or [],
                    "response_text_observed": item.get("response_text", ""),
                    "observed_claim_count": item.get("observed_claim_count", 0),
                    "observed_grounded_claim_count": item.get("observed_grounded_claim_count", 0),
                    "observed_expert_judgment_count": item.get(
                        "observed_expert_judgment_count", 0
                    ),
                    "observed_unsupported_claim_count": item.get(
                        "observed_unsupported_claim_count", 0
                    ),
                    "claims": claims,
                    "issue_match": item.get("issue_match"),
                    "planner_fallback": bool(item.get("planner_fallback")),
                    "human_verified": False,
                    "reviewer_id": "",
                    "reviewed_at": "",
                    "reviewer_notes": "",
                    "annotation_instructions": (
                        "원문에서 gold_relevant_chunk_ids를 확정하고 각 claim의 text/type/"
                        "expected_claim_type/supported를 채운 뒤 검수자 정보를 기록해야 "
                        "human_verified=true로 변경할 수 있습니다."
                    ),
                }
            )

    return {
        "dataset_name": "ideation_multi_document_quality_v1",
        "source": "ideation_trace_logs",
        "project_count_requested": project_count,
        "cases_per_project_requested": cases_per_project,
        "project_ids": eligible_projects,
        "case_count": len(cases),
        "ready_for_scoring": False,
        "cases": cases,
    }


def build_from_files(paths: list[Path], *, project_count: int, cases_per_project: int) -> dict[str, Any]:
    lines: list[str] = []
    for path in paths:
        lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return build_annotation_queue(
        collect_observations(lines),
        project_count=project_count,
        cases_per_project=cases_per_project,
    )


def apply_assistant_review(
    payload: dict[str, Any],
    review_payload: dict[str, Any],
) -> dict[str, Any]:
    """Codex 등 비인간 검수 결과를 병합하되 human_verified로 승격하지 않는다."""
    decisions = review_payload.get("cases")
    if not isinstance(decisions, dict):
        raise ValueError("assistant review에는 case_id를 키로 하는 cases 객체가 필요합니다")
    queue_ids = {case.get("case_id") for case in payload.get("cases") or []}
    missing = sorted(case_id for case_id in queue_ids if case_id not in decisions)
    extra = sorted(case_id for case_id in decisions if case_id not in queue_ids)
    if missing or extra:
        raise ValueError(f"assistant review case 불일치: missing={missing}, extra={extra}")

    for case in payload.get("cases") or []:
        decision = decisions[case["case_id"]]
        for field in ("gold_relevant_chunk_ids", "claims", "issue_match", "reviewer_notes"):
            case[field] = decision[field]
        if "planner_relevant_selected_chunk_ids" in decision:
            case["planner_relevant_selected_chunk_ids"] = decision[
                "planner_relevant_selected_chunk_ids"
            ]
        case["assistant_reviewed"] = True
        case["human_verified"] = False
        case["reviewer_id"] = review_payload.get("reviewer_id", "")
        case["reviewer_type"] = review_payload.get("reviewer_type", "")
        case["reviewed_at"] = review_payload.get("reviewed_at", "")
    payload["assistant_review_complete"] = True
    payload["assistant_review_methodology"] = review_payload.get("methodology", "")
    payload["ready_for_scoring"] = False
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="아이디어 회의 trace → 다중 문서 검수 대기열")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-count", type=int, default=3)
    parser.add_argument("--cases-per-project", type=int, default=5)
    parser.add_argument(
        "--assistant-review",
        type=Path,
        help="비인간 수동 검수 JSON을 병합한다(human_verified는 false 유지)",
    )
    args = parser.parse_args()
    payload = build_from_files(
        args.logs,
        project_count=args.project_count,
        cases_per_project=args.cases_per_project,
    )
    if args.assistant_review:
        review_payload = json.loads(args.assistant_review.read_text(encoding="utf-8"))
        payload = apply_assistant_review(payload, review_payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("project_ids", "case_count", "ready_for_scoring")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
