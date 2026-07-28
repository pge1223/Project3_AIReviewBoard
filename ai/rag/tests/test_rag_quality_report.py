"""
Unit Tests for ai.rag.evaluation.rag_quality.report (JSON/CSV/Markdown 생성물 형태).
외부 의존성 없음 — EvalReport를 직접 조립해서 파일로 내보낸 뒤 다시 읽어 검증한다.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from ai.rag.evaluation.rag_quality.report import write_csv, write_json, write_markdown
from ai.rag.evaluation.rag_quality.schemas import (
    ClaimVerdict,
    EvalReport,
    EvalSettingsSnapshot,
    GenerationAggregate,
    GenerationCaseResult,
    MessageEvalResult,
    PersonaFitResult,
    RetrievalAggregate,
    RetrievalCaseResult,
    RetrievedDocumentHit,
)


def _report() -> EvalReport:
    retrieval = [
        RetrievalCaseResult(
            case_id="rag_eval_001",
            query="q1",
            persona_id="planning_expert",
            project_id="p1",
            role_id="planning",
            gold_document_ids=["doc-a"],
            retrieved=[RetrievedDocumentHit(document_id="doc-b", chunk_id="c1", rank=1, score=0.5)],
            retrieved_document_ids=["doc-b"],
            recall_at_k=0.0,
            hit_at_k=0.0,
            human_verified=True,
            empty_result=True,
        )
    ]
    generation = [
        GenerationCaseResult(
            case_id="rag_eval_001",
            query="q1",
            messages=[
                MessageEvalResult(
                    case_id="rag_eval_001",
                    message_id="m1",
                    persona_id="planning_expert",
                    round=1,
                    content_preview="발언 미리보기",
                    claims=[ClaimVerdict(claim="c", verdict="unsupported", reason="근거 없음", confidence=0.4)],
                    faithfulness_score=0.0,
                    hallucination_rate=1.0,
                    unsupported_count=1,
                    persona_fit=PersonaFitResult(
                        persona_id="planning_expert", message_id="m1", score=1, normalized_score=0.25
                    ),
                    judge_error=None,
                )
            ],
        )
    ]
    return EvalReport(
        run_id="run-1",
        executed_at=datetime.now(timezone.utc),
        settings=EvalSettingsSnapshot(
            dataset_name="rag_eval_v1",
            dataset_version="1.0.0",
            dataset_path="ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl",
            mode="all",
            top_k=5,
            generation_model="gpt-4o-mini",
            eval_model="gpt-4o-mini",
            eval_prompt_version="faithfulness_judge_v1",
            chroma_collection="project_documents_kure_v1",
            human_verified_only=False,
            cache_enabled=True,
        ),
        retrieval_results=retrieval,
        retrieval_aggregate=RetrievalAggregate(k=5, case_count=1, human_verified_case_count=1, recall_at_k_macro=0.0),
        generation_results=generation,
        generation_aggregate=GenerationAggregate(case_count=1, message_count=1, faithfulness_macro=0.0),
    )


def test_write_json_round_trips(tmp_path):
    report = _report()
    path = write_json(report, tmp_path / "out" / "report.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run-1"
    assert data["retrieval_results"][0]["case_id"] == "rag_eval_001"
    assert data["generation_results"][0]["messages"][0]["claims"][0]["verdict"] == "unsupported"


def test_write_csv_has_one_row_per_case_with_four_metrics_and_failure_reason(tmp_path):
    report = _report()
    path = write_csv(report, tmp_path / "report.csv")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["case_id"] == "rag_eval_001"
    assert row["recall_at_k"] == "0.0"
    assert row["reciprocal_rank"] == "0.0"
    assert row["faithfulness"] == "0.0"
    assert row["hallucination_rate"] == "1.0"
    assert row["persona_evidence_fit"] == "0.25"
    assert "retrieval_empty" in row["failure_reasons"]


def test_write_markdown_includes_summary_and_worst_cases_sections(tmp_path):
    report = _report()
    path = write_markdown(report, tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    assert "# RAG Evaluation" in text
    assert "## Summary" in text
    assert "Recall@5" in text
    assert "MRR@5" in text
    assert "Faithfulness" in text
    assert "Hallucination Rate" in text
    assert "Persona Evidence Fit" in text
    assert "## Worst Cases" in text
    assert "rag_eval_001" in text


def test_write_markdown_shows_na_for_missing_scores(tmp_path):
    report = _report()
    report.retrieval_aggregate.recall_at_k_macro = None
    path = write_markdown(report, tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    assert "N/A" in text
