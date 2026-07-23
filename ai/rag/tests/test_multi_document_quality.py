from ai.rag.evaluation.rag_quality.multi_document_quality import evaluate_multi_document_cases


def _case(project: str, index: int, *, supported: bool = True) -> dict:
    chunk = f"{project}-chunk-{index}"
    return {
        "case_id": f"{project}-{index}",
        "project_id": project,
        "persona_id": "planning_expert",
        "issue_id": "problem",
        "query": f"{project} 문제 정의 검토 {index}",
        "gold_relevant_chunk_ids": [chunk],
        "retrieved_chunk_ids": [chunk],
        "selected_chunk_ids": [chunk],
        "claims": [
            {
                "text": f"{project}의 문제 정의가 공모전 평가 대상이다.",
                "claim_type": "document_fact",
                "expected_claim_type": "document_fact",
                "supported": supported,
                "linked_chunk_ids": [chunk],
            }
        ],
        "issue_match": True,
        "planner_fallback": False,
        "human_verified": True,
        "reviewer_id": "reviewer-1",
        "reviewed_at": "2026-07-23T17:30:00+09:00",
        "reviewer_notes": "원문 청크와 주장을 대조함",
    }


def test_quality_gate_passes_only_with_three_projects_and_fifteen_verified_cases():
    cases = [_case(project, index) for project in ("P1", "P2", "P3") for index in range(5)]
    report = evaluate_multi_document_cases(cases)
    assert report["passed"] is True
    assert report["metrics"]["project_count"] == 3
    assert report["metrics"]["verified_case_count"] == 15
    assert report["metrics"]["min_verified_cases_per_project"] == 5
    assert report["metrics"]["citation_precision"] == 1.0


def test_quality_gate_excludes_unverified_and_reports_stage_specific_failures():
    cases = [_case("P1", index, supported=False) for index in range(5)]
    cases.append({**_case("P2", 1), "human_verified": False})
    report = evaluate_multi_document_cases(cases)
    assert report["passed"] is False
    assert report["excluded_unverified_case_count"] == 1
    assert "project_count" in report["failure_reasons"]
    assert "citation_precision" in report["failure_reasons"]
    assert "unsupported_document_fact_rate" in report["failure_reasons"]


def test_declared_verified_case_without_annotation_evidence_is_rejected():
    case = _case("P1", 1)
    case["reviewer_id"] = ""
    case["gold_relevant_chunk_ids"] = []
    report = evaluate_multi_document_cases([case])
    assert report["passed"] is False
    assert report["metrics"]["verified_case_count"] == 0
    assert report["invalid_verified_cases"][0]["case_id"] == "P1-1"
    assert "missing_reviewer_id" in report["invalid_verified_cases"][0]["errors"]
    assert "missing_gold_relevant_chunk_ids" in report["invalid_verified_cases"][0]["errors"]


def test_fifteen_cases_cannot_pass_when_distribution_is_thirteen_one_one():
    cases = (
        [_case("P1", index) for index in range(13)]
        + [_case("P2", 1)]
        + [_case("P3", 1)]
    )
    report = evaluate_multi_document_cases(cases)
    assert report["metrics"]["project_count"] == 3
    assert report["metrics"]["verified_case_count"] == 15
    assert report["metrics"]["min_verified_cases_per_project"] == 1
    assert report["passed"] is False
    assert "min_verified_cases_per_project" in report["failure_reasons"]


def test_assistant_review_can_be_scored_but_is_marked_non_official():
    cases = [
        {**_case(project, index), "human_verified": False, "assistant_reviewed": True}
        for project in ("P1", "P2", "P3")
        for index in range(5)
    ]
    report = evaluate_multi_document_cases(cases, verification_field="assistant_reviewed")
    assert report["passed"] is True
    assert report["official_human_evaluation"] is False


def test_planner_precision_uses_reviewed_quote_relevance_not_coarse_chunk_gold():
    case = _case("P1", 1)
    # 검색 chunk에는 쟁점 관련 문장이 있지만 Planner가 그 안의 무관한 quote를 선택한 경우.
    case["planner_relevant_selected_chunk_ids"] = []
    report = evaluate_multi_document_cases(
        [{**case, "human_verified": False, "assistant_reviewed": True}],
        verification_field="assistant_reviewed",
    )
    assert report["metrics"]["retrieval_recall_at_5"] == 1.0
    assert report["metrics"]["planner_precision"] == 0.0
    assert report["metrics"]["planner_coverage"] == 0.0


def test_reviewed_planner_relevance_must_be_subset_of_selected_chunks():
    case = _case("P1", 1)
    case["planner_relevant_selected_chunk_ids"] = ["not-selected"]
    report = evaluate_multi_document_cases([case])
    assert report["metrics"]["verified_case_count"] == 0
    assert "planner_relevant_selected_chunk_ids_not_selected" in report[
        "invalid_verified_cases"
    ][0]["errors"]
