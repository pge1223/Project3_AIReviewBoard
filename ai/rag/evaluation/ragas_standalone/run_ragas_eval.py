# 작성자: 용준/Claude(2026-07-29)
# 목적: ai/rag/evaluation/rag_quality/export_ragas_dataset.py가 만든 JSONL(+project_id/
#       question_type을 덧붙인 파일럿/전체 서브셋)을 읽어 실제 Ragas 패키지로
#       Faithfulness / Answer Relevancy / Context Precision / Context Recall을 계산하고,
#       message 단위 평균 -> case 단위 macro 평균 -> project_id/persona/질문유형별 평균까지
#       집계한다.
#
#       이 스크립트는 이 저장소의 앱 코드(ai.*, backend.*)를 절대 import하지 않는다 —
#       ragas가 요구하는 langchain-community 버전이 앱이 쓰는 최신 langchain-core/
#       langchain-community와 충돌해서(ragas가 이미 삭제된
#       langchain_community.chat_models.vertexai를 무조건 import함) 별도 conda 환경
#       (environment-ragas.yml/requirements-ragas.txt)에서만 이 스크립트를 실행한다.
#       사용법은 README.md 참고.
#
#       reference(정답 발언)가 없는 샘플은 Context Precision/Context Recall을 계산하지
#       않고 "not_measurable"로 명시한다 — 두 지표 모두 Ragas 원 정의상 reference가
#       필수다(LLMContextPrecisionWithReference/LLMContextRecall의 required_columns).
#       값을 만들어 채우거나 다른 알고리즘으로 대체해 같은 이름을 붙이지 않는다.
#
#       한 행(row) 채점이 실패해도 전체 실행은 죽지 않는다 — 실패는 항상 그 행의
#       *_error 필드에 남기고 다음 행으로 넘어간다.
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ragas 오프라인 평가 실행(별도 환경 전용)")
    parser.add_argument("--input", required=True, help="평가 대상 JSONL 경로")
    parser.add_argument("--output-dir", required=True, help="결과 파일들을 저장할 디렉터리(이미 있으면 파일명 충돌 시 에러)")
    parser.add_argument("--output-prefix", default="ragas", help="출력 파일명 접두사")
    parser.add_argument("--model", default="gpt-4o-mini", help="ragas judge LLM (OpenAI chat 모델)")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--temperature", type=float, default=0.0, help="judge LLM temperature(재현성을 위해 기본 0)")
    parser.add_argument("--limit", type=int, default=None, help="평가할 최대 행 수")
    parser.add_argument("--api-key", default=None, help="미지정 시 환경변수 OPENAI_API_KEY를 사용")
    return parser.parse_args()


def _load_rows(path: Path, limit: Optional[int]) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _reserve_output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    """기존 결과 파일을 덮어쓰지 않는다 — 이미 파일이 있으면 즉시 에러로 멈춘다(요청:
    '입력 데이터셋과 기존 결과 파일을 덮어쓰지 않음')."""
    paths = {
        "samples": output_dir / f"{prefix}_samples.jsonl",
        "full_csv": output_dir / f"{prefix}_full.csv",
        "case_agg_csv": output_dir / f"{prefix}_case_agg.csv",
        "summary_md": output_dir / f"{prefix}_summary.md",
    }
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise SystemExit(
            "출력 파일이 이미 존재합니다(덮어쓰지 않음). --output-dir 또는 --output-prefix를 "
            f"바꿔주세요: {existing}"
        )
    return paths


@dataclass
class RowResult:
    case_id: str
    message_id: str
    persona_id: str
    project_id: str
    question_type: str
    faithfulness: Optional[float] = None
    faithfulness_error: Optional[str] = None
    answer_relevancy: Optional[float] = None
    answer_relevancy_error: Optional[str] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision_status: str = "not_measurable"  # measured | not_measurable | error
    context_recall_status: str = "not_measurable"
    context_precision_error: Optional[str] = None
    context_recall_error: Optional[str] = None
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "message_id": self.message_id,
            "persona_id": self.persona_id,
            "project_id": self.project_id,
            "question_type": self.question_type,
            "faithfulness": self.faithfulness,
            "faithfulness_error": self.faithfulness_error,
            "answer_relevancy": self.answer_relevancy,
            "answer_relevancy_error": self.answer_relevancy_error,
            "context_precision": self.context_precision,
            "context_precision_status": self.context_precision_status,
            "context_precision_error": self.context_precision_error,
            "context_recall": self.context_recall,
            "context_recall_status": self.context_recall_status,
            "context_recall_error": self.context_recall_error,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


async def _score_row(row: dict, *, metrics: dict) -> RowResult:
    from ragas.dataset_schema import SingleTurnSample

    started = time.perf_counter()
    result = RowResult(
        case_id=row.get("case_id", ""),
        message_id=row.get("message_id", ""),
        persona_id=row.get("persona_id", ""),
        project_id=row.get("project_id", ""),
        question_type=row.get("question_type", ""),
    )

    user_input = row.get("user_input") or ""
    response = row.get("response") or ""
    retrieved_contexts = row.get("retrieved_contexts") or []
    reference = row.get("reference") or None

    missing = [f for f in ("user_input", "response", "retrieved_contexts") if not row.get(f)]
    if missing:
        msg = f"입력 필드 누락: {missing}"
        result.faithfulness_error = msg
        result.answer_relevancy_error = msg
        result.context_precision_error = msg
        result.context_recall_error = msg
        result.context_precision_status = "error"
        result.context_recall_status = "error"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    sample_no_ref = SingleTurnSample(
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
    )

    try:
        result.faithfulness = await metrics["faithfulness"].single_turn_ascore(sample_no_ref)
    except Exception as exc:  # noqa: BLE001 - 한 행 실패로 전체 실행을 죽이지 않는다
        result.faithfulness_error = str(exc)

    try:
        result.answer_relevancy = await metrics["answer_relevancy"].single_turn_ascore(sample_no_ref)
    except Exception as exc:  # noqa: BLE001
        result.answer_relevancy_error = str(exc)

    if reference:
        sample_with_ref = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )
        try:
            result.context_precision = await metrics["context_precision"].single_turn_ascore(sample_with_ref)
            result.context_precision_status = "measured"
        except Exception as exc:  # noqa: BLE001
            result.context_precision_status = "error"
            result.context_precision_error = str(exc)

        try:
            result.context_recall = await metrics["context_recall"].single_turn_ascore(sample_with_ref)
            result.context_recall_status = "measured"
        except Exception as exc:  # noqa: BLE001
            result.context_recall_status = "error"
            result.context_recall_error = str(exc)
    else:
        result.context_precision_error = "reference 없음 - Ragas LLMContextPrecisionWithReference는 reference가 필수"
        result.context_recall_error = "reference 없음 - Ragas LLMContextRecall은 reference가 필수"

    result.elapsed_ms = (time.perf_counter() - started) * 1000
    return result


def _mean(values: list[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _group_mean(results: list[RowResult], key_fn, metric: str) -> dict[str, Optional[float]]:
    groups: dict[str, list[float]] = {}
    for r in results:
        v = getattr(r, metric)
        if v is None:
            continue
        groups.setdefault(key_fn(r), []).append(v)
    return {k: _mean(v) for k, v in groups.items()}


def _case_macro(results: list[RowResult], metric: str) -> tuple[Optional[float], dict[str, Optional[float]]]:
    """같은 case_id의 메시지 점수를 먼저 평균(case별 값) -> 그 case별 값들을 다시
    평균(macro, case마다 동일 가중치). 요청 2번 집계 방식."""
    per_case: dict[str, list[float]] = {}
    for r in results:
        v = getattr(r, metric)
        if v is None:
            continue
        per_case.setdefault(r.case_id, []).append(v)
    case_means = {cid: _mean(v) for cid, v in per_case.items()}
    macro = _mean([v for v in case_means.values() if v is not None])
    return macro, case_means


def _worst_n(results: list[RowResult], metric: str, n: int = 5) -> list[dict]:
    scored = [(getattr(r, metric), r) for r in results if getattr(r, metric) is not None]
    scored.sort(key=lambda t: t[0])
    return [
        {"case_id": r.case_id, "message_id": r.message_id, "persona_id": r.persona_id,
         "project_id": r.project_id, "question_type": r.question_type, "score": round(v, 4)}
        for v, r in scored[:n]
    ]


def _failure_counts(results: list[RowResult]) -> dict[str, int]:
    return {
        "faithfulness_error_count": sum(1 for r in results if r.faithfulness_error),
        "answer_relevancy_error_count": sum(1 for r in results if r.answer_relevancy_error),
        "context_precision_error_count": sum(1 for r in results if r.context_precision_status == "error"),
        "context_precision_not_measurable_count": sum(
            1 for r in results if r.context_precision_status == "not_measurable"
        ),
        "context_recall_error_count": sum(1 for r in results if r.context_recall_status == "error"),
        "context_recall_not_measurable_count": sum(
            1 for r in results if r.context_recall_status == "not_measurable"
        ),
    }


def _build_aggregate(results: list[RowResult]) -> dict:
    message_level = {m: _mean([getattr(r, m) for r in results]) for m in METRIC_NAMES}

    case_macro = {}
    case_means_by_metric = {}
    for m in METRIC_NAMES:
        macro, case_means = _case_macro(results, m)
        case_macro[m] = macro
        case_means_by_metric[m] = case_means

    project_level = {m: _group_mean(results, lambda r: r.project_id, m) for m in METRIC_NAMES}
    persona_level = {m: _group_mean(results, lambda r: r.persona_id, m) for m in METRIC_NAMES}
    qtype_level = {m: _group_mean(results, lambda r: r.question_type, m) for m in METRIC_NAMES}
    worst = {m: _worst_n(results, m, 5) for m in METRIC_NAMES}

    return {
        "message_level_mean": message_level,
        "case_macro_mean": case_macro,
        "case_level_means": case_means_by_metric,
        "project_level_mean": project_level,
        "persona_level_mean": persona_level,
        "question_type_level_mean": qtype_level,
        "worst_5_by_metric": worst,
        "failure_counts": _failure_counts(results),
        "distinct_case_count": len({r.case_id for r in results}),
        "message_count": len(results),
    }


def _write_summary_md(path: Path, *, settings: dict, aggregate: dict, elapsed_seconds: float) -> None:
    lines: list[str] = []
    lines.append(f"# Ragas 평가 결과 — {settings['output_prefix']}")
    lines.append("")
    lines.append(f"- 실행 시각(UTC): {settings['executed_at']}")
    lines.append(f"- 입력: `{settings['input_path']}`")
    lines.append(f"- judge 모델: `{settings['judge_model']}` (temperature={settings['temperature']})")
    lines.append(f"- 임베딩 모델: `{settings['embedding_model']}`")
    lines.append(f"- ragas 버전: `{settings['ragas_version']}`, langchain-openai 버전: `{settings['langchain_openai_version']}`")
    lines.append(f"- 샘플 수: {aggregate['message_count']}건 (distinct case: {aggregate['distinct_case_count']}건)")
    lines.append(f"- 소요 시간: {elapsed_seconds:.1f}초 (건당 평균 {elapsed_seconds / max(aggregate['message_count'],1):.1f}초)")
    lines.append(f"- 근사 비용: {settings['approx_cost_note']}")
    lines.append("")

    lines.append("## 1. 메시지 단위 평균")
    lines.append("")
    lines.append("| 지표 | 평균 |")
    lines.append("|---|---:|")
    for m in METRIC_NAMES:
        v = aggregate["message_level_mean"][m]
        lines.append(f"| {m} | {v:.4f} |" if v is not None else f"| {m} | N/A |")
    lines.append("")

    lines.append("## 2. case_id 단위 macro 평균 (case마다 동일 가중치)")
    lines.append("")
    lines.append("| 지표 | macro 평균 |")
    lines.append("|---|---:|")
    for m in METRIC_NAMES:
        v = aggregate["case_macro_mean"][m]
        lines.append(f"| {m} | {v:.4f} |" if v is not None else f"| {m} | N/A |")
    lines.append("")

    lines.append("## 3. project_id별 평균")
    lines.append("")
    for m in METRIC_NAMES:
        lines.append(f"**{m}**")
        lines.append("")
        lines.append("| project_id | 평균 |")
        lines.append("|---|---:|")
        for pid, v in aggregate["project_level_mean"][m].items():
            lines.append(f"| {pid} | {v:.4f} |")
        lines.append("")

    lines.append("## 4. persona별 평균")
    lines.append("")
    lines.append("| 지표 | planning_expert | dev_expert |")
    lines.append("|---|---:|---:|")
    for m in METRIC_NAMES:
        pv = aggregate["persona_level_mean"][m]
        p = pv.get("planning_expert")
        d = pv.get("dev_expert")
        p_s = f"{p:.4f}" if p is not None else "N/A"
        d_s = f"{d:.4f}" if d is not None else "N/A"
        lines.append(f"| {m} | {p_s} | {d_s} |")
    lines.append("")

    lines.append("## 5. criteria vs target 질문 유형별 평균")
    lines.append("")
    lines.append("| 지표 | criteria | target |")
    lines.append("|---|---:|---:|")
    for m in METRIC_NAMES:
        qv = aggregate["question_type_level_mean"][m]
        c = qv.get("criteria")
        t = qv.get("target")
        c_s = f"{c:.4f}" if c is not None else "N/A"
        t_s = f"{t:.4f}" if t is not None else "N/A"
        lines.append(f"| {m} | {c_s} | {t_s} |")
    lines.append("")

    lines.append("## 6. 지표별 최저 점수 사례 5건")
    lines.append("")
    for m in METRIC_NAMES:
        lines.append(f"**{m}**")
        lines.append("")
        lines.append("| case_id | message_id | persona | project_id | 유형 | 점수 |")
        lines.append("|---|---|---|---|---|---:|")
        for w in aggregate["worst_5_by_metric"][m]:
            lines.append(
                f"| {w['case_id']} | {w['message_id']} | {w['persona_id']} | {w['project_id']} | "
                f"{w['question_type']} | {w['score']:.4f} |"
            )
        lines.append("")

    lines.append("## 7. 평가 실패·누락 건수")
    lines.append("")
    for k, v in aggregate["failure_counts"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


async def _run(args: argparse.Namespace) -> None:
    import langchain_openai
    import ragas
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY가 없습니다. --api-key로 넘기거나 환경변수를 설정하세요.")

    input_path = Path(args.input)
    rows = _load_rows(input_path, args.limit)
    if not rows:
        raise SystemExit(f"입력 JSONL에 평가할 행이 없습니다: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = _reserve_output_paths(output_dir, args.output_prefix)

    llm = LangchainLLMWrapper(ChatOpenAI(model=args.model, api_key=api_key, temperature=args.temperature))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=args.embedding_model, api_key=api_key))

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": LLMContextPrecisionWithReference(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
    }

    wall_started = time.perf_counter()
    results: list[RowResult] = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] case={row.get('case_id')} message={row.get('message_id')}", file=sys.stderr)
        results.append(await _score_row(row, metrics=metrics))
    elapsed_seconds = time.perf_counter() - wall_started

    aggregate = _build_aggregate(results)

    # 정확한 토큰 사용량은 재지 않는다(요청 09번의 "예상 비용" 수준) - Faithfulness는 클레임
    # 추출+검증 2회, Answer Relevancy는 합성 질문 생성 N회, Context Precision/Recall은
    # reference가 있을 때 각 1회 이상 LLM을 호출한다. 정확한 값은 OpenAI 사용량 대시보드에서
    # 확인해야 한다.
    approx_calls = len(results) * 6
    approx_cost_note = (
        f"정확한 토큰 사용량 미계측. 대략 {approx_calls}회 내외의 LLM 호출(건당 faithfulness"
        f" 2회 + answer_relevancy 1~N회 + context_precision/recall 최대 2회)이 발생했을 것으로"
        f" 추정 — 정확한 값은 OpenAI 사용량 대시보드에서 확인 필요."
    )

    settings = {
        "output_prefix": args.output_prefix,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "judge_model": args.model,
        "temperature": args.temperature,
        "embedding_model": args.embedding_model,
        "ragas_version": ragas.__version__,
        "langchain_openai_version": langchain_openai.__version__,
        "elapsed_seconds": elapsed_seconds,
        "approx_cost_note": approx_cost_note,
    }

    out_paths["samples"].write_text(
        "\n".join(json.dumps(r.as_dict(), ensure_ascii=False) for r in results),
        encoding="utf-8",
    )

    with out_paths["full_csv"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].as_dict().keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_dict())

    with out_paths["case_agg_csv"].open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["case_id"] + list(METRIC_NAMES)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        case_ids = sorted({r.case_id for r in results})
        for cid in case_ids:
            row_out = {"case_id": cid}
            for m in METRIC_NAMES:
                row_out[m] = aggregate["case_level_means"][m].get(cid)
            writer.writerow(row_out)

    _write_summary_md(out_paths["summary_md"], settings=settings, aggregate=aggregate, elapsed_seconds=elapsed_seconds)

    # settings + aggregate를 JSON으로도 남긴다(사람이 읽는 md와 별개로 기계가 읽기 쉬운 원본).
    json_path = output_dir / f"{args.output_prefix}_report.json"
    json_path.write_text(
        json.dumps({"settings": settings, "aggregate": aggregate}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"samples JSONL: {out_paths['samples']}")
    print(f"full CSV:      {out_paths['full_csv']}")
    print(f"case-agg CSV:  {out_paths['case_agg_csv']}")
    print(f"summary MD:    {out_paths['summary_md']}")
    print(f"report JSON:   {json_path}")
    print(
        f"message-level: faithfulness={aggregate['message_level_mean']['faithfulness']} "
        f"answer_relevancy={aggregate['message_level_mean']['answer_relevancy']} "
        f"context_precision={aggregate['message_level_mean']['context_precision']} "
        f"context_recall={aggregate['message_level_mean']['context_recall']}"
    )
    print(f"elapsed: {elapsed_seconds:.1f}s for {len(results)} rows")
    print(f"failures: {aggregate['failure_counts']}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
