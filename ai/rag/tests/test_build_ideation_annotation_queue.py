from ai.rag.evaluation.rag_quality.build_ideation_annotation_queue import (
    apply_assistant_review,
    build_annotation_queue,
    collect_observations,
)


def _lines(project: str, session: str, plan: str) -> list[str]:
    return [
        (
            f'2026-07-23 10:00:00 | INFO | trace | [IDEATION_EVIDENCE_LOOKUP] '
            f'session="{session}" request="REQ-1" speaker="planning_expert" '
            f'project_id="{project}" issue="problem" query="문제 정의" '
            f'chunk_ids=["{project}-C1","{project}-C2"]'
        ),
        (
            f'2026-07-23 10:00:01 | INFO | trace | [IDEATION_EVIDENCE_PLAN_SHADOW_CREATED] '
            f'session="{session}" request="REQ-1" speaker="planning_expert" '
            f'plan_id="{plan}" effective_issue_title="문제 정의" '
            f'selected_evidence=[{{"ref":"E1","chunk_id":"{project}-C1","quote_preview":"근거"}}]'
        ),
        (
            f'2026-07-23 10:00:02 | INFO | trace | [IDEATION_EVIDENCE_LINKED] '
            f'session="{session}" request="REQ-1" speaker="planning_expert" '
            f'claim_id="claim_1" evidence_refs=["E1"] chunk_ids=["{project}-C1"]'
        ),
        (
            f'2026-07-23 10:00:03 | INFO | trace | [IDEATION_EVIDENCE_PLAN_COMPLIANCE] '
            f'session="{session}" request="REQ-1" speaker="planning_expert" '
            f'plan_id="{plan}" issue_match=true claim_count=1 grounded_claim_count=1 '
            f'linked_chunk_ids=["{project}-C1"]'
        ),
        (
            f'2026-07-23 10:00:04 | INFO | trace | [IDEATION_TURN_END] '
            f'session="{session}" request="REQ-1" speaker="planning_expert" '
            f'text="전문가 발언" accepted_claim_count=1 grounded_claim_count=1 '
            f'expert_judgment_count=0 unsupported_claim_count=0'
        ),
    ]


def test_trace_observation_becomes_unverified_annotation_case():
    observations = collect_observations(_lines("P1", "S1", "EP-1"))
    payload = build_annotation_queue(observations, project_count=1, cases_per_project=1)
    assert payload["case_count"] == 1
    case = payload["cases"][0]
    assert case["retrieved_chunk_ids"] == ["P1-C1", "P1-C2"]
    assert case["selected_chunk_ids"] == ["P1-C1"]
    assert case["claims"][0]["linked_chunk_ids"] == ["P1-C1"]
    assert case["human_verified"] is False
    assert payload["ready_for_scoring"] is False


def test_queue_requires_enough_cases_per_project():
    observations = collect_observations(_lines("P1", "S1", "EP-1"))
    payload = build_annotation_queue(observations, project_count=1, cases_per_project=2)
    assert payload["project_ids"] == []
    assert payload["case_count"] == 0


def test_queue_excludes_legacy_observation_without_issue_or_compliance():
    observations = collect_observations(_lines("P1", "S1", "EP-1"))
    observations[0]["issue_id"] = ""
    observations[0]["issue_match"] = None
    payload = build_annotation_queue(observations, project_count=1, cases_per_project=1)
    assert payload["case_count"] == 0


def test_assistant_review_does_not_impersonate_human_verification():
    observations = collect_observations(_lines("P1", "S1", "EP-1"))
    payload = build_annotation_queue(observations, project_count=1, cases_per_project=1)
    case_id = payload["cases"][0]["case_id"]
    reviewed = apply_assistant_review(
        payload,
        {
            "reviewer_id": "codex",
            "reviewer_type": "ai_assistant_manual_source_review",
            "reviewed_at": "2026-07-23T18:30:00+09:00",
            "cases": {
                case_id: {
                    "gold_relevant_chunk_ids": ["P1-C1"],
                    "planner_relevant_selected_chunk_ids": [],
                    "claims": [],
                    "issue_match": True,
                    "reviewer_notes": "원문 대조",
                }
            },
        },
    )
    assert reviewed["cases"][0]["assistant_reviewed"] is True
    assert reviewed["cases"][0]["human_verified"] is False
    assert reviewed["cases"][0]["planner_relevant_selected_chunk_ids"] == []
