"""
Unit Tests for ai.rag.evaluation.rag_quality.{schemas,dataset,retrieval_eval}
(FakeRoleAwareRetrievalService만 사용 — 실제 KURE, Chroma, OpenAI API 없음)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai.rag.evaluation.rag_quality.dataset import extract_review_sample, filter_cases, load_cases
from ai.rag.evaluation.rag_quality.retrieval_eval import aggregate_retrieval, run_retrieval_eval
from ai.rag.evaluation.rag_quality.schemas import RagEvalCase, RagEvalFilters
from ai.rag.role_retrieval.schemas import RoleSearchResponse, RoleSearchResult


class FakeRoleAwareRetrievalService:
    """RoleAwareRetrievalService.search_by_role()과 정확히 같은 시그니처를 갖는 fake —
    이 시그니처가 실제 서비스와 어긋나면 ai.rag.orchestration.ideation_evidence_service.
    search_ideation_evidence()가 실제로 호출할 때 TypeError가 나므로, 이 테스트들이
    간접적으로 '재구현하지 않고 실제 함수를 그대로 호출하는지'를 검증한다."""

    def __init__(self, responses_by_query: dict[str, RoleSearchResponse]):
        self._responses = responses_by_query
        self.calls: list[dict] = []

    def search_by_role(self, query, project_id, role_id=None, document_id=None, top_k=5, candidate_k=None):
        self.calls.append(
            {"query": query, "project_id": project_id, "role_id": role_id, "top_k": top_k}
        )
        return self._responses.get(
            query,
            RoleSearchResponse(
                query=query, expanded_query=query, role_id=role_id, project_id=project_id, results=[], result_count=0
            ),
        )


def _result(document_id: str, chunk_id: str, score: float) -> RoleSearchResult:
    return RoleSearchResult(
        record_id=f"{document_id}::{chunk_id}",
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content {chunk_id}",
        role_score=score,
        final_score=score,
        metadata={"document_name": f"{document_id}.pdf"},
    )


def _response(query: str, project_id: str, role_id: str, results: list[RoleSearchResult]) -> RoleSearchResponse:
    return RoleSearchResponse(
        query=query,
        expanded_query=query,
        role_id=role_id,
        role_name=role_id,
        project_id=project_id,
        results=results,
        result_count=len(results),
    )


def _case(
    case_id="c1",
    query="문제 정의",
    persona_id="planning_expert",
    project_id="proj-1",
    gold=("doc-a",),
    expect_no_evidence=False,
    human_verified=False,
) -> RagEvalCase:
    return RagEvalCase(
        id=case_id,
        query=query,
        persona_id=persona_id,
        filters=RagEvalFilters(project_id=project_id),
        gold_document_ids=list(gold) if not expect_no_evidence else [],
        expect_no_evidence=expect_no_evidence,
        human_verified=human_verified,
    )


# --------------------------------------------------------------------------
# 스키마 검증
# --------------------------------------------------------------------------


def test_case_requires_gold_document_ids_unless_expect_no_evidence():
    with pytest.raises(ValidationError):
        RagEvalCase(id="c1", query="q", persona_id="planning_expert", filters=RagEvalFilters(project_id="p1"))


def test_case_forbids_gold_document_ids_when_expect_no_evidence():
    with pytest.raises(ValidationError):
        RagEvalCase(
            id="c1",
            query="q",
            persona_id="planning_expert",
            filters=RagEvalFilters(project_id="p1"),
            gold_document_ids=["doc-a"],
            expect_no_evidence=True,
        )


def test_case_human_verified_defaults_to_false():
    case = _case()
    assert case.human_verified is False


def test_dataset_rejects_duplicate_case_ids(tmp_path):
    path = tmp_path / "dup.jsonl"
    path.write_text(
        '{"id":"c1","query":"q","persona_id":"planning_expert","filters":{"project_id":"p"},'
        '"gold_document_ids":["d1"]}\n'
        '{"id":"c1","query":"q2","persona_id":"dev_expert","filters":{"project_id":"p"},'
        '"gold_document_ids":["d2"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_cases(path)


def test_load_cases_reads_dataset_header_meta(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text(
        '{"dataset_name":"my_ds","version":"9.9.9"}\n'
        '{"id":"c1","query":"q","persona_id":"planning_expert","filters":{"project_id":"p"},'
        '"gold_document_ids":["d1"]}\n',
        encoding="utf-8",
    )
    ds = load_cases(path)
    assert ds.dataset_name == "my_ds"
    assert ds.version == "9.9.9"
    assert len(ds.cases) == 1


def test_filter_cases_by_persona_and_limit():
    cases = [_case(case_id="a", persona_id="planning_expert"), _case(case_id="b", persona_id="dev_expert")]
    filtered = filter_cases(cases, persona_id="dev_expert")
    assert [c.id for c in filtered] == ["b"]
    assert len(filter_cases(cases, limit=1)) == 1


def test_extract_review_sample_only_pulls_unverified_and_is_deterministic():
    cases = [_case(case_id=f"c{i}", human_verified=(i % 5 == 0)) for i in range(20)]
    sample1 = extract_review_sample(cases, fraction=0.5)
    sample2 = extract_review_sample(cases, fraction=0.5)
    assert all(not c.human_verified for c in sample1)
    assert [c.id for c in sample1] == [c.id for c in sample2]  # 결정적(seed 고정)


# --------------------------------------------------------------------------
# Recall@K / Hit@K
# --------------------------------------------------------------------------


def test_recall_at_k_single_gold_document_hit():
    service = FakeRoleAwareRetrievalService(
        {"q1": _response("q1", "p1", "planning", [_result("doc-a", "c1", 0.9), _result("doc-b", "c2", 0.5)])}
    )
    results = run_retrieval_eval([_case(query="q1", gold=("doc-a",))], role_retrieval_service=service, top_k=5)
    assert results[0].recall_at_k == 1.0
    assert results[0].hit_at_k == 1.0
    assert results[0].reciprocal_rank == 1.0
    assert results[0].retrieved_document_ids == ["doc-a", "doc-b"]


def test_recall_at_k_multiple_gold_documents_partial_hit():
    service = FakeRoleAwareRetrievalService(
        {"q1": _response("q1", "p1", "planning", [_result("doc-a", "c1", 0.9)])}
    )
    results = run_retrieval_eval(
        [_case(query="q1", gold=("doc-a", "doc-missing"))], role_retrieval_service=service, top_k=5
    )
    assert results[0].recall_at_k == 0.5  # 정답 2개 중 1개만 상위 K에 있음
    assert results[0].hit_at_k == 1.0  # 하나라도 있으면 hit


def test_recall_at_k_dedupes_multiple_chunks_of_same_document():
    """같은 document_id의 청크 두 개가 상위권에 있어도 문서 하나로 접혀야 한다."""
    service = FakeRoleAwareRetrievalService(
        {
            "q1": _response(
                "q1", "p1", "planning", [_result("doc-a", "c1", 0.9), _result("doc-a", "c2", 0.8), _result("doc-b", "c3", 0.7)]
            )
        }
    )
    results = run_retrieval_eval([_case(query="q1", gold=("doc-a",))], role_retrieval_service=service, top_k=5)
    assert results[0].retrieved_document_ids == ["doc-a", "doc-b"]


def test_reciprocal_rank_uses_first_relevant_document_rank():
    service = FakeRoleAwareRetrievalService(
        {
            "q1": _response(
                "q1",
                "p1",
                "planning",
                [_result("doc-x", "c1", 0.9), _result("doc-a", "c2", 0.8)],
            )
        }
    )
    results = run_retrieval_eval(
        [_case(query="q1", gold=("doc-a",), human_verified=True)],
        role_retrieval_service=service,
        top_k=5,
    )
    assert results[0].reciprocal_rank == 0.5
    assert aggregate_retrieval(results, k=5).mrr_macro == 0.5


def test_expect_no_evidence_case_excluded_from_recall_and_tracks_empty_result():
    service = FakeRoleAwareRetrievalService({"q1": _response("q1", "p1", "planning", [])})
    case = _case(query="q1", expect_no_evidence=True)
    results = run_retrieval_eval([case], role_retrieval_service=service, top_k=5)
    assert results[0].recall_at_k == 0.0  # 계산 자체가 의미 없음 — aggregate에서 제외됨
    assert results[0].empty_result is True

    agg = aggregate_retrieval(results, k=5)
    assert agg.recall_at_k_macro is None  # scored 케이스가 없음(no-evidence만 있음)
    assert agg.no_evidence_case_count == 1
    assert agg.no_evidence_accuracy == 1.0


def test_aggregate_separates_human_verified_from_reference_scores():
    service = FakeRoleAwareRetrievalService(
        {
            "verified_q": _response("verified_q", "p1", "planning", [_result("doc-a", "c1", 0.9)]),
            "unverified_q": _response("unverified_q", "p1", "planning", []),
        }
    )
    verified_case = _case(case_id="v1", query="verified_q", gold=("doc-a",), human_verified=True)
    unverified_case = _case(case_id="u1", query="unverified_q", gold=("doc-a",), human_verified=False)

    results = run_retrieval_eval([verified_case, unverified_case], role_retrieval_service=service, top_k=5)
    agg = aggregate_retrieval(results, k=5)

    assert agg.human_verified_case_count == 1
    assert agg.recall_at_k_macro == 1.0  # 검수된 케이스만
    assert agg.reference_recall_at_k_macro == 0.5  # 전체(검수+미검수) 평균 = (1.0+0.0)/2


def test_retrieval_failure_rate_counts_empty_results_when_gold_expected():
    service = FakeRoleAwareRetrievalService({"q1": _response("q1", "p1", "planning", [])})
    results = run_retrieval_eval([_case(query="q1", gold=("doc-a",))], role_retrieval_service=service, top_k=5)
    agg = aggregate_retrieval(results, k=5)
    assert agg.retrieval_failure_rate == 1.0
