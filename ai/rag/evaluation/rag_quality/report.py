# 작성자: 용준/Claude(2026-07-22)
# 목적: EvalReport를 JSON(전체 원시 결과)/CSV(케이스별 4지표+실패 사유)/Markdown(요약
#       리포트) 세 형태로 저장한다(요청 9번).
from __future__ import annotations

import csv
import json
from pathlib import Path

from ai.rag.evaluation.rag_quality.schemas import EvalReport


def write_json(report: EvalReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(json.loads(report.model_dump_json()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def write_csv(report: EvalReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    retrieval_by_case = {r.case_id: r for r in report.retrieval_results}
    generation_by_case: dict[str, list] = {}
    for g in report.generation_results:
        generation_by_case.setdefault(g.case_id, []).extend(g.messages)

    case_ids = sorted(set(retrieval_by_case) | set(generation_by_case))
    for case_id in case_ids:
        retrieval = retrieval_by_case.get(case_id)
        messages = generation_by_case.get(case_id, [])
        faith_values = [m.faithfulness_score for m in messages if m.faithfulness_score is not None]
        halluc_values = [m.hallucination_rate for m in messages if m.hallucination_rate is not None]
        fit_values = [m.persona_fit.normalized_score for m in messages if m.persona_fit is not None]
        failure_reasons = []
        if retrieval and retrieval.empty_result and not retrieval.expect_no_evidence:
            failure_reasons.append("retrieval_empty")
        failure_reasons.extend(m.judge_error for m in messages if m.judge_error)

        rows.append(
            {
                "case_id": case_id,
                "recall_at_k": retrieval.recall_at_k if retrieval else "",
                "hit_at_k": retrieval.hit_at_k if retrieval else "",
                "reciprocal_rank": retrieval.reciprocal_rank if retrieval else "",
                "faithfulness": (sum(faith_values) / len(faith_values)) if faith_values else "",
                "hallucination_rate": (sum(halluc_values) / len(halluc_values)) if halluc_values else "",
                "persona_evidence_fit": (sum(fit_values) / len(fit_values)) if fit_values else "",
                "human_verified": retrieval.human_verified if retrieval else "",
                "failure_reasons": ";".join(failure_reasons),
            }
        )

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "recall_at_k",
                "hit_at_k",
                "reciprocal_rank",
                "faithfulness",
                "hallucination_rate",
                "persona_evidence_fit",
                "human_verified",
                "failure_reasons",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(report: EvalReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# RAG Evaluation", ""]
    lines.append(f"- 평가 일시: {report.executed_at.isoformat()}")
    lines.append(f"- 데이터셋: {report.settings.dataset_name} v{report.settings.dataset_version}")
    lines.append(f"- 데이터셋 경로: {report.settings.dataset_path}")
    lines.append(f"- 검색 설정: top_k={report.settings.top_k}, collection={report.settings.chroma_collection}")
    lines.append(f"- 생성 모델: {report.settings.generation_model or 'N/A'}")
    lines.append(f"- 평가 모델: {report.settings.eval_model or 'N/A'} (prompt_version={report.settings.eval_prompt_version})")
    verified_count = sum(1 for r in report.retrieval_results if r.human_verified)
    lines.append(f"- 검수 완료 케이스 수: {verified_count} / {len(report.retrieval_results) or len(report.generation_results)}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    if report.retrieval_aggregate:
        agg = report.retrieval_aggregate
        lines.append(f"- Recall@{agg.k}: {_fmt(agg.recall_at_k_macro)} (참고: {_fmt(agg.reference_recall_at_k_macro)}, 검수 {agg.human_verified_case_count}건 기준)")
        lines.append(f"- Hit@{agg.k}: {_fmt(agg.hit_at_k_macro)} (참고: {_fmt(agg.reference_hit_at_k_macro)})")
        lines.append(f"- MRR@{agg.k}: {_fmt(agg.mrr_macro)} (참고: {_fmt(agg.reference_mrr_macro)})")
        lines.append(f"- 검색 실패율(예상 근거 있는데 결과 0건): {_fmt(agg.retrieval_failure_rate)}")
        lines.append(f"- 근거 없음 예상 케이스 정확도: {_fmt(agg.no_evidence_accuracy)} ({agg.no_evidence_case_count}건)")
        lines.append(f"- 평균 검색 시간: {agg.avg_retrieval_time_ms:.1f}ms")
    if report.generation_aggregate:
        agg = report.generation_aggregate
        lines.append(f"- Faithfulness: {_fmt(agg.faithfulness_macro)}")
        lines.append(f"- Hallucination Rate: {_fmt(agg.hallucination_rate_macro)}")
        lines.append(f"- Persona Evidence Fit: {_fmt(agg.persona_evidence_fit_macro)} ({_fmt(agg.persona_evidence_fit_percent)}%)")
        lines.append(f"- 평가 실패율(judge 오류): {_fmt(agg.eval_failure_rate)}")
        lines.append(f"- 생성 실패율: {_fmt(agg.generation_failure_rate)}")
        lines.append(f"- 평균 생성 시간: {agg.avg_generation_time_ms:.1f}ms")
    lines.append(f"- 예상 비용: {_fmt(report.estimated_cost_usd)} USD" if report.estimated_cost_usd is not None else "- 예상 비용: 미측정")
    lines.append("")

    lines.append("## Worst Cases")
    lines.append("")
    low_recall = sorted(
        (r for r in report.retrieval_results if not r.expect_no_evidence),
        key=lambda r: r.recall_at_k,
    )[:5]
    if low_recall:
        lines.append("### Recall@K가 낮은 질문")
        for r in low_recall:
            lines.append(f"- `{r.case_id}` recall={r.recall_at_k:.2f} query={r.query!r}")
        lines.append("")

    all_messages = [m for g in report.generation_results for m in g.messages]
    unsupported_heavy = sorted(all_messages, key=lambda m: m.unsupported_count, reverse=True)[:5]
    if unsupported_heavy and any(m.unsupported_count for m in unsupported_heavy):
        lines.append("### unsupported 주장이 많은 답변")
        for m in unsupported_heavy:
            if m.unsupported_count:
                lines.append(f"- `{m.case_id}`/`{m.message_id}` unsupported={m.unsupported_count} content={m.content_preview!r}")
        lines.append("")

    contradicted = [m for m in all_messages if m.contradicted_count]
    if contradicted:
        lines.append("### contradicted 주장")
        for m in contradicted[:5]:
            lines.append(f"- `{m.case_id}`/`{m.message_id}` contradicted={m.contradicted_count}")
        lines.append("")

    low_fit = sorted((m for m in all_messages if m.persona_fit), key=lambda m: m.persona_fit.score)[:5]
    if low_fit:
        lines.append("### Persona Evidence Fit이 낮은 발언")
        for m in low_fit:
            lines.append(f"- `{m.case_id}`/`{m.message_id}` score={m.persona_fit.score}/4 persona={m.persona_id}")
        lines.append("")

    empty_but_asserted = [
        r for r in report.retrieval_results if r.empty_result and not r.expect_no_evidence
    ]
    if empty_but_asserted:
        lines.append("### 검색 결과가 없는데 사실을 단정한 사례 후보(검색 0건, 정답 기대)")
        for r in empty_but_asserted[:5]:
            lines.append(f"- `{r.case_id}` query={r.query!r}")
        lines.append("")

    if report.notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(report.notes)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
