# 작성자: 용준/Claude(2026-07-22)
# 목적: Recall@K/Hit@K 검색 평가. LLM 호출 없이 실행된다(요청 8번).
#       ai.rag.orchestration.ideation_evidence_service.search_ideation_evidence()를 그대로
#       호출한다 — planning_expert/dev_expert의 evidence_lookup이 실제로 쓰는 그 함수다.
#       재구현하지 않는다. Recall@K/Hit@K 계산은 ai/rag/evaluation/metrics.py의 순수 함수를
#       그대로 import해 쓴다(재구현 금지 원칙).
from __future__ import annotations

import time
from typing import Optional

from ai.rag.evaluation.metrics import hit_rate_at_k, recall_at_k
from ai.rag.evaluation.rag_quality.schemas import (
    RagEvalCase,
    RetrievalAggregate,
    RetrievalCaseResult,
    RetrievedDocumentHit,
)
from ai.rag.orchestration.ideation_evidence_service import (
    resolve_ideation_role_id,
    search_ideation_evidence,
)
from ai.rag.role_retrieval.service import RoleAwareRetrievalService


def _dedupe_by_document(items: list[dict]) -> list[RetrievedDocumentHit]:
    """search_ideation_evidence()가 반환하는 청크 단위 결과를 문서 단위로 접는다 — 같은
    document_id가 여러 청크로 상위권에 반복되면 최초(가장 점수가 높은) 등장 순위만 남긴다.
    요청 스키마가 gold_document_ids(문서 단위)를 쓰므로, "문서가 상위 K개 안에 있는가"로
    맞춘 것 — 청크 단위 원본 정보(chunk_id/score)는 그대로 보존한다."""
    seen: set[str] = set()
    hits: list[RetrievedDocumentHit] = []
    for item in items:
        document_id = item.get("document_id")
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        hits.append(
            RetrievedDocumentHit(
                document_id=document_id,
                chunk_id=item.get("chunk_id") or "",
                rank=len(hits) + 1,
                score=float(item.get("score") or 0.0),
                document_name=item.get("document_name"),
            )
        )
    return hits


def run_retrieval_eval(
    cases: list[RagEvalCase],
    *,
    role_retrieval_service: RoleAwareRetrievalService,
    top_k: int = 5,
) -> list[RetrievalCaseResult]:
    results: list[RetrievalCaseResult] = []
    for case in cases:
        role_id = case.filters.role_id or resolve_ideation_role_id(case.persona_id)

        started = time.perf_counter()
        items = search_ideation_evidence(
            case.persona_id,
            case.query,
            case.filters.project_id,
            role_retrieval_service,
            top_k=top_k,
            phase=case.filters.phase,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        hits = _dedupe_by_document(items)
        retrieved_document_ids = [h.document_id for h in hits]
        gold_ids = set(case.gold_document_ids)

        recall = recall_at_k(retrieved_document_ids, gold_ids, top_k) if gold_ids else 0.0
        hit = hit_rate_at_k(retrieved_document_ids, gold_ids, top_k) if gold_ids else 0.0
        first_relevant_rank = next(
            (rank for rank, document_id in enumerate(retrieved_document_ids[:top_k], start=1) if document_id in gold_ids),
            None,
        )
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0

        results.append(
            RetrievalCaseResult(
                case_id=case.id,
                query=case.query,
                persona_id=case.persona_id,
                project_id=case.filters.project_id,
                role_id=role_id,
                gold_document_ids=sorted(gold_ids),
                retrieved=hits,
                retrieved_document_ids=retrieved_document_ids,
                recall_at_k=recall,
                hit_at_k=hit,
                reciprocal_rank=reciprocal_rank,
                expect_no_evidence=case.expect_no_evidence,
                empty_result=len(items) == 0,
                human_verified=case.human_verified,
                retrieval_time_ms=elapsed_ms,
            )
        )
    return results


def aggregate_retrieval(results: list[RetrievalCaseResult], *, k: int) -> RetrievalAggregate:
    """요청 3번 — 정식 점수(human_verified=true & expect_no_evidence=false)는 macro
    average, 미검수 항목은 reference_*로 분리, expect_no_evidence 케이스는 recall 계산에서
    빼고 "검색 결과 없음"의 정확성을 별도로 집계한다."""
    scored = [r for r in results if not r.expect_no_evidence]
    verified = [r for r in scored if r.human_verified]
    no_evidence = [r for r in results if r.expect_no_evidence]

    def _mean(values: list[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    retrieval_failures = [r for r in scored if r.empty_result]

    return RetrievalAggregate(
        k=k,
        case_count=len(results),
        human_verified_case_count=len(verified),
        recall_at_k_macro=_mean([r.recall_at_k for r in verified]),
        hit_at_k_macro=_mean([r.hit_at_k for r in verified]),
        mrr_macro=_mean([r.reciprocal_rank for r in verified]),
        reference_recall_at_k_macro=_mean([r.recall_at_k for r in scored]),
        reference_hit_at_k_macro=_mean([r.hit_at_k for r in scored]),
        reference_mrr_macro=_mean([r.reciprocal_rank for r in scored]),
        no_evidence_case_count=len(no_evidence),
        no_evidence_accuracy=_mean([1.0 if r.empty_result else 0.0 for r in no_evidence]),
        retrieval_failure_rate=(len(retrieval_failures) / len(scored)) if scored else None,
        avg_retrieval_time_ms=_mean([r.retrieval_time_ms for r in results]) or 0.0,
    )
