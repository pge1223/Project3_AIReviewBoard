# 작성자: 용준/Claude(2026-07-29, 요청: target/criteria/외부근거/전문가판단 분리)
# 목적: _classify_linked_evidence_buckets(ai/meeting/graph/ideation_conv_nodes.py)가
#       실제로 claim에 인용·검증된 근거만 target/criteria/external 버킷으로 나누고,
#       인용되지 않은 검색 결과나 근거 없는 claim을 올바르게 분류하는지 검증한다.
#       ai/meeting은 ai.rag를 import하지 않으므로(TestScopeBoundary), source_type이
#       이미 태깅된 plain dict를 그대로 손으로 구성해 순수 함수만 테스트한다.

import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_conv_nodes import _classify_linked_evidence_buckets  # noqa: E402


def _pool_item(
    chunk_id: str, source_type: str, ref: str | None = None, *, ideation_source_type: str | None = None
) -> dict:
    item = {"chunk_id": chunk_id, "source_type": source_type}
    if ref:
        item["ref"] = ref
    if ideation_source_type:
        item["ideation_source_type"] = ideation_source_type
    return item


def _grounding(claims, claim_evidence_links):
    return {"claims": claims, "claim_evidence_links": claim_evidence_links}


def test_target_evidence_is_not_counted_as_external():
    pool = [_pool_item("T1", "target")]
    grounding = _grounding(
        claims=[{"claim_id": "claim_1", "claim_type": "user_provided_fact"}],
        claim_evidence_links=[{"claim_id": "claim_1", "evidence_refs": ["T1"], "chunk_ids": ["T1"]}],
    )
    buckets = _classify_linked_evidence_buckets(grounding, pool)

    assert buckets["reviewed_target_refs"] == ["T1"]
    assert buckets["linked_external_evidence_refs"] == []
    assert buckets["linked_criteria_refs"] == []


def test_user_session_answer_target_is_excluded_from_every_bucket():
    """용준/Claude(2026-07-30, 요청: "사용자 답변은 EvidenceToggle에 절대 표시하지
    마세요") — 사용자 채팅 답변(ideation_source_type=user_session_answer)은 선택 아이디어와
    같은 document_role=target으로 색인되지만, claim이 실제로 인용해도 reviewed_target_refs를
    포함한 어느 버킷에도 노출되면 안 된다."""
    pool = [_pool_item("A1", "target", ideation_source_type="user_session_answer")]
    grounding = _grounding(
        claims=[{"claim_id": "claim_1", "claim_type": "user_provided_fact"}],
        claim_evidence_links=[{"claim_id": "claim_1", "evidence_refs": ["A1"], "chunk_ids": ["A1"]}],
    )
    buckets = _classify_linked_evidence_buckets(grounding, pool)

    assert buckets["reviewed_target_refs"] == []
    assert buckets["linked_criteria_refs"] == []
    assert buckets["linked_external_evidence_refs"] == []


def test_selected_idea_target_still_shown_alongside_excluded_user_answer():
    pool = [
        _pool_item("T1", "target", ideation_source_type="ideation_candidate"),
        _pool_item("A1", "target", ideation_source_type="user_session_answer"),
    ]
    grounding = _grounding(
        claims=[
            {"claim_id": "claim_1", "claim_type": "user_provided_fact"},
            {"claim_id": "claim_2", "claim_type": "user_provided_fact"},
        ],
        claim_evidence_links=[
            {"claim_id": "claim_1", "evidence_refs": ["T1"], "chunk_ids": ["T1"]},
            {"claim_id": "claim_2", "evidence_refs": ["A1"], "chunk_ids": ["A1"]},
        ],
    )
    buckets = _classify_linked_evidence_buckets(grounding, pool)

    assert buckets["reviewed_target_refs"] == ["T1"]


def test_criteria_and_official_statistics_are_classified_correctly():
    pool = [_pool_item("C1", "criteria"), _pool_item("E1", "official_statistics")]
    grounding = _grounding(
        claims=[
            {"claim_id": "claim_1", "claim_type": "document_fact"},
            {"claim_id": "claim_2", "claim_type": "document_fact"},
        ],
        claim_evidence_links=[
            {"claim_id": "claim_1", "evidence_refs": ["C1"], "chunk_ids": ["C1"]},
            {"claim_id": "claim_2", "evidence_refs": ["E1"], "chunk_ids": ["E1"]},
        ],
    )
    buckets = _classify_linked_evidence_buckets(grounding, pool)

    assert buckets["linked_criteria_refs"] == ["C1"]
    assert buckets["linked_external_evidence_refs"] == ["E1"]


def test_retrieved_but_uncited_pool_items_are_excluded_from_every_bucket():
    # E2는 검색 풀에는 있지만 어떤 claim의 claim_evidence_links에도 등장하지 않는다.
    pool = [_pool_item("E1", "official_statistics"), _pool_item("E2", "news")]
    grounding = _grounding(
        claims=[{"claim_id": "claim_1", "claim_type": "document_fact"}],
        claim_evidence_links=[{"claim_id": "claim_1", "evidence_refs": ["E1"], "chunk_ids": ["E1"]}],
    )
    buckets = _classify_linked_evidence_buckets(grounding, pool)

    assert buckets["linked_external_evidence_refs"] == ["E1"]
    assert "E2" not in buckets["linked_external_evidence_refs"]


def test_claim_without_any_linked_evidence_goes_to_expert_judgment_bucket():
    pool: list[dict] = []
    grounding = _grounding(
        claims=[{"claim_id": "claim_1", "claim_type": "expert_judgment"}],
        claim_evidence_links=[],
    )
    buckets = _classify_linked_evidence_buckets(grounding, pool)

    assert buckets["expert_judgment_without_external_evidence"] == ["claim_1"]
    assert buckets["reviewed_target_refs"] == []
    assert buckets["linked_criteria_refs"] == []
    assert buckets["linked_external_evidence_refs"] == []


def test_unsupported_document_fact_claim_also_counts_as_expert_judgment_bucket():
    # claim_grounding.py가 target 근거로는 document_fact를 절대 grounding시키지 않으므로
    # (claim_type_document_role_mismatch), 그런 claim은 claim_evidence_links에 없다 — 이
    # 버킷 분류 함수는 그 결과를 그대로 존중해 expert_judgment 버킷에 둔다(새 grounding
    # 규칙을 추가하지 않았음을 고정하는 회귀 테스트).
    pool = [_pool_item("T1", "target")]
    grounding = _grounding(
        claims=[{"claim_id": "claim_1", "claim_type": "document_fact"}],
        claim_evidence_links=[],
    )
    buckets = _classify_linked_evidence_buckets(grounding, pool)

    assert buckets["expert_judgment_without_external_evidence"] == ["claim_1"]
    assert buckets["reviewed_target_refs"] == []
