"""아이디어 회의 RAG의 다중 문서 품질 게이트.

운영 로그 또는 수동 검수 결과를 같은 JSON 형식으로 넣어 retrieval → planner → claim
grounding 단계별 품질을 분리해 측정한다. 사람 검수 완료 케이스만 정식 점수에 사용한다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_QUALITY_THRESHOLDS = {
    "project_count": 3,
    "verified_case_count": 15,
    "min_verified_cases_per_project": 5,
    "retrieval_recall_at_5": 0.80,
    "planner_precision": 0.80,
    "planner_coverage": 0.80,
    "citation_precision": 0.90,
    "claim_type_accuracy": 0.90,
    "unsupported_document_fact_rate": 0.05,
    "issue_match_rate": 0.90,
    "planner_fallback_rate": 0.10,
}

_VALID_CLAIM_TYPES = {"document_fact", "expert_judgment", "user_provided_fact"}


def validate_verified_case(case: dict[str, Any]) -> list[str]:
    """human_verified=true로 점수에 포함할 수 있는 최소 검수 계약을 확인한다."""
    errors: list[str] = []
    for field in ("case_id", "project_id", "persona_id", "issue_id", "query"):
        if not isinstance(case.get(field), str) or not case[field].strip():
            errors.append(f"missing_{field}")
    if not case.get("gold_relevant_chunk_ids"):
        errors.append("missing_gold_relevant_chunk_ids")
    if not isinstance(case.get("retrieved_chunk_ids"), list):
        errors.append("missing_retrieved_chunk_ids")
    if not isinstance(case.get("selected_chunk_ids"), list):
        errors.append("missing_selected_chunk_ids")
    planner_relevant = case.get("planner_relevant_selected_chunk_ids")
    if planner_relevant is not None:
        if not isinstance(planner_relevant, list):
            errors.append("invalid_planner_relevant_selected_chunk_ids")
        elif not set(planner_relevant).issubset(set(case.get("selected_chunk_ids") or [])):
            errors.append("planner_relevant_selected_chunk_ids_not_selected")
    if case.get("issue_match") not in (True, False):
        errors.append("missing_issue_match")
    if case.get("planner_fallback") not in (True, False):
        errors.append("missing_planner_fallback")
    for field in ("reviewer_id", "reviewed_at", "reviewer_notes"):
        if not isinstance(case.get(field), str) or not case[field].strip():
            errors.append(f"missing_{field}")

    claims = case.get("claims")
    if not isinstance(claims, list):
        errors.append("missing_claims")
    elif claims:
        for index, claim in enumerate(claims):
            prefix = f"claim_{index + 1}"
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str) or not claim["text"].strip():
                errors.append(f"{prefix}_missing_text")
                continue
            if claim.get("claim_type") not in _VALID_CLAIM_TYPES:
                errors.append(f"{prefix}_invalid_claim_type")
            if claim.get("expected_claim_type") not in _VALID_CLAIM_TYPES:
                errors.append(f"{prefix}_invalid_expected_claim_type")
            if claim.get("supported") not in (True, False):
                errors.append(f"{prefix}_missing_supported")
            if not isinstance(claim.get("linked_chunk_ids"), list):
                errors.append(f"{prefix}_missing_linked_chunk_ids")
    return errors


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_multi_document_cases(
    cases: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
    verification_field: str = "human_verified",
) -> dict[str, Any]:
    """사람이 검수한 3개 이상 프로젝트의 단계별 품질과 통과 여부를 계산한다."""
    limits = {**DEFAULT_QUALITY_THRESHOLDS, **(thresholds or {})}
    if verification_field not in ("human_verified", "assistant_reviewed"):
        raise ValueError("verification_field는 human_verified 또는 assistant_reviewed여야 합니다")
    declared_verified = [case for case in cases if case.get(verification_field) is True]
    duplicate_case_ids = {
        case_id
        for case_id in {case.get("case_id") for case in declared_verified}
        if case_id and sum(1 for case in declared_verified if case.get("case_id") == case_id) > 1
    }
    invalid_verified_cases: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for case in declared_verified:
        errors = validate_verified_case(case)
        if case.get("case_id") in duplicate_case_ids:
            errors.append("duplicate_case_id")
        if errors:
            invalid_verified_cases.append({"case_id": case.get("case_id"), "errors": errors})
        else:
            verified.append(case)
    project_ids = sorted({str(case.get("project_id")) for case in verified if case.get("project_id")})
    per_project_counts = {
        project_id: sum(1 for case in verified if str(case.get("project_id")) == project_id)
        for project_id in project_ids
    }

    retrieval_recalls: list[float] = []
    selected_total = selected_relevant = covered = 0
    planner_cases = 0
    document_claims = supported_document_claims = 0
    typed_claims = correctly_typed_claims = 0
    issue_matches = fallback_count = 0

    for case in verified:
        gold = set(case.get("gold_relevant_chunk_ids") or [])
        retrieved = set((case.get("retrieved_chunk_ids") or [])[:5])
        selected = set(case.get("selected_chunk_ids") or [])
        # Chunk 단위 gold만으로 Planner precision을 계산하면, 큰 평가표 chunk 안에서
        # 현재 쟁점과 무관한 quote를 골라도 같은 chunk의 다른 문장 때문에 정답 처리된다.
        # 검수자가 선택 quote 자체를 판정한 경우 그 결과를 우선하고, 구버전 데이터만
        # 기존 chunk 교집합 계산으로 폴백한다.
        reviewed_planner_relevant = case.get("planner_relevant_selected_chunk_ids")
        selected_relevant_for_case = (
            set(reviewed_planner_relevant)
            if isinstance(reviewed_planner_relevant, list)
            else selected & gold
        )
        if gold:
            retrieval_recalls.append(len(gold & retrieved) / len(gold))
            planner_cases += 1
            covered += int(bool(selected_relevant_for_case))
        selected_total += len(selected)
        selected_relevant += len(selected_relevant_for_case)
        issue_matches += int(case.get("issue_match") is True)
        fallback_count += int(case.get("planner_fallback") is True)

        for claim in case.get("claims") or []:
            actual_type = claim.get("claim_type")
            expected_type = claim.get("expected_claim_type")
            if expected_type:
                typed_claims += 1
                correctly_typed_claims += int(actual_type == expected_type)
            if actual_type == "document_fact":
                document_claims += 1
                supported_document_claims += int(claim.get("supported") is True)

    metrics: dict[str, float | int | None] = {
        "project_count": len(project_ids),
        "verified_case_count": len(verified),
        "min_verified_cases_per_project": min(per_project_counts.values()) if per_project_counts else 0,
        "retrieval_recall_at_5": _mean(retrieval_recalls),
        "planner_precision": _ratio(selected_relevant, selected_total),
        "planner_coverage": _ratio(covered, planner_cases),
        "citation_precision": _ratio(supported_document_claims, document_claims),
        "claim_type_accuracy": _ratio(correctly_typed_claims, typed_claims),
        "unsupported_document_fact_rate": _ratio(
            document_claims - supported_document_claims, document_claims
        ),
        "issue_match_rate": _ratio(issue_matches, len(verified)),
        "planner_fallback_rate": _ratio(fallback_count, len(verified)),
    }

    lower_is_better = {"unsupported_document_fact_rate", "planner_fallback_rate"}
    gate_results: dict[str, bool] = {}
    for name, limit in limits.items():
        value = metrics.get(name)
        gate_results[name] = value is not None and (value <= limit if name in lower_is_better else value >= limit)

    failure_reasons = [name for name, passed in gate_results.items() if not passed]
    if invalid_verified_cases:
        failure_reasons.append("invalid_verified_cases")

    return {
        "metrics": metrics,
        "verification_field": verification_field,
        "official_human_evaluation": verification_field == "human_verified",
        "thresholds": limits,
        "gate_results": gate_results,
        "passed": all(gate_results.values()) and not invalid_verified_cases,
        "project_ids": project_ids,
        "verified_cases_per_project": per_project_counts,
        "excluded_unverified_case_count": len(cases) - len(declared_verified),
        "invalid_verified_cases": invalid_verified_cases,
        "failure_reasons": failure_reasons,
    }


def evaluate_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    verification_field: str = "human_verified",
) -> dict[str, Any]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("평가 입력은 cases 배열 또는 배열 자체여야 합니다")
    report = evaluate_multi_document_cases(cases, verification_field=verification_field)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="아이디어 회의 다중 문서 RAG 품질 게이트")
    parser.add_argument("input", help="검수 결과 JSON")
    parser.add_argument("--output", help="리포트 JSON 저장 경로")
    parser.add_argument(
        "--verification-field",
        choices=("human_verified", "assistant_reviewed"),
        default="human_verified",
        help="기본값 human_verified. assistant_reviewed는 잠정 점수만 산출",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_file(
                args.input,
                args.output,
                verification_field=args.verification_field,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
