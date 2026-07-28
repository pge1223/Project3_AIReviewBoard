# 작성자: 용준/Claude(2026-07-22)
# 목적: RAG 품질 평가 CLI. 실제 색인된 Chroma 데이터와 실제 OpenAI 호출을 쓴다(개발자가
#       직접 실행하는 도구 — 이 파일을 import만 해서는 어떤 외부 호출도 일어나지 않는다).
#
# 실행 예:
#   python -m ai.rag.evaluation.rag_quality.cli \
#       --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl \
#       --mode retrieval --top-k 5 --output reports/rag_eval
#
#   python -m ai.rag.evaluation.rag_quality.cli \
#       --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl \
#       --mode generation --limit 10 --output reports/rag_eval
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

_BACKEND_DIR = Path(__file__).resolve().parents[4] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.rag.evaluation.rag_quality.cache import JudgeCache
from ai.rag.evaluation.rag_quality.dataset import filter_cases, load_cases
from ai.rag.evaluation.rag_quality.generation_eval import aggregate_generation, run_generation_eval
from ai.rag.evaluation.rag_quality.report import write_csv, write_json, write_markdown
from ai.rag.evaluation.rag_quality.retrieval_eval import aggregate_retrieval, run_retrieval_eval
from ai.rag.evaluation.rag_quality.schemas import EvalReport, EvalSettingsSnapshot
from ai.rag.evaluation.runner import _build_real_retriever

from ._meeting_path import ensure_meeting_on_path

ensure_meeting_on_path()
from graph.llm import make_openai_llm_call  # noqa: E402

FAITHFULNESS_PROMPT_VERSION_LABEL = "faithfulness_judge_v1 / persona_fit_judge_v1"


class _CallCounter:
    """LLM 호출 횟수를 세는 얇은 래퍼 — 비용 추정에 쓴다(정확한 토큰 사용량은 재지 않는다,
    요청 9번의 "예상 비용"에 해당)."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return self._inner(prompt)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 품질 오프라인 평가 도구")
    parser.add_argument("--dataset", required=True, help="평가셋 JSONL 경로")
    parser.add_argument("--mode", choices=["retrieval", "generation", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="평가할 최대 케이스 수")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--persona", choices=["planning_expert", "dev_expert"], default=None)
    parser.add_argument("--output", required=True, help="결과를 저장할 디렉터리")
    parser.add_argument("--no-cache", action="store_true", help="judge 결과 캐시를 쓰지 않는다")
    parser.add_argument("--human-verified-only", action="store_true")
    parser.add_argument(
        "--chroma-path", default=None, help="미지정 시 backend 설정(CHROMA_PERSIST_DIR)을 사용"
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--notes", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> EvalReport:
    args = _parse_args(argv)

    from app.config import resolve_chroma_persist_dir, settings
    from ai.rag.domain.config import DEFAULT_COLLECTION_NAME

    dataset = load_cases(args.dataset)
    cases = filter_cases(
        dataset.cases,
        case_id=args.case_id,
        persona_id=args.persona,
        human_verified_only=args.human_verified_only,
        limit=args.limit,
    )
    if not cases:
        raise SystemExit("필터 조건에 맞는 케이스가 없습니다.")

    # FastAPI와 평가 CLI가 반드시 같은 실제 저장소를 사용하게 공통 resolver를 쓴다.
    chroma_path = args.chroma_path or settings.CHROMA_PERSIST_DIR
    chroma_path = resolve_chroma_persist_dir(chroma_path)
    role_retrieval_service, _ = _build_real_retriever(chroma_path, DEFAULT_COLLECTION_NAME)

    generation_model = settings.DEV_LLM_REVIEWER_MODEL
    eval_model = settings.EVAL_LLM_MODEL

    retrieval_results = []
    retrieval_aggregate = None
    generation_results = []
    generation_aggregate = None

    generation_call_counter: Optional[_CallCounter] = None
    judge_call_counter: Optional[_CallCounter] = None

    if args.mode in ("retrieval", "all"):
        retrieval_results = run_retrieval_eval(cases, role_retrieval_service=role_retrieval_service, top_k=args.top_k)
        retrieval_aggregate = aggregate_retrieval(retrieval_results, k=args.top_k)

    if args.mode in ("generation", "all"):
        llm_call = _CallCounter(make_openai_llm_call(generation_model, api_key=settings.OPENAI_API_KEY or None))
        judge_llm_call = _CallCounter(make_openai_llm_call(eval_model, api_key=settings.OPENAI_API_KEY or None))
        generation_call_counter = llm_call
        judge_call_counter = judge_llm_call
        cache = JudgeCache(enabled=not args.no_cache)

        generation_results = run_generation_eval(
            cases,
            llm_call=llm_call,
            judge_llm_call=judge_llm_call,
            judge_model=eval_model,
            role_retrieval_service=role_retrieval_service,
            top_k=args.top_k,
            cache=cache,
        )
        generation_aggregate = aggregate_generation(generation_results)

    # 요청 9번 "예상 또는 실제 평가 비용" — 정확한 토큰 사용량을 재지 않으므로, 호출
    # 1건당 평균 1,500 토큰을 가정한 근사치다(gpt-4o-mini 기준 대략적인 자릿수 확인용 —
    # 정확한 값이 필요하면 OpenAI 사용량 대시보드를 확인해야 한다는 것을 리포트에 명시).
    _APPROX_USD_PER_CALL = 0.001
    estimated_cost = None
    if generation_call_counter is not None or judge_call_counter is not None:
        total_calls = (generation_call_counter.calls if generation_call_counter else 0) + (
            judge_call_counter.calls if judge_call_counter else 0
        )
        estimated_cost = round(total_calls * _APPROX_USD_PER_CALL, 4)

    report = EvalReport(
        run_id=args.run_id or str(uuid.uuid4()),
        executed_at=datetime.now(timezone.utc),
        settings=EvalSettingsSnapshot(
            dataset_name=dataset.dataset_name,
            dataset_version=dataset.version,
            dataset_path=str(args.dataset),
            mode=args.mode,
            top_k=args.top_k,
            generation_model=generation_model if args.mode in ("generation", "all") else None,
            eval_model=eval_model if args.mode in ("generation", "all") else None,
            eval_prompt_version=FAITHFULNESS_PROMPT_VERSION_LABEL,
            chroma_collection=DEFAULT_COLLECTION_NAME,
            human_verified_only=args.human_verified_only,
            cache_enabled=not args.no_cache,
        ),
        retrieval_results=retrieval_results,
        retrieval_aggregate=retrieval_aggregate,
        generation_results=generation_results,
        generation_aggregate=generation_aggregate,
        estimated_cost_usd=estimated_cost,
        notes=args.notes,
    )

    output_dir = Path(args.output)
    json_path = write_json(report, output_dir / "report.json")
    csv_path = write_csv(report, output_dir / "report.csv")
    md_path = write_markdown(report, output_dir / "report.md")

    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")
    if retrieval_aggregate:
        print(
            f"Recall@{retrieval_aggregate.k}={retrieval_aggregate.recall_at_k_macro} "
            f"(참고={retrieval_aggregate.reference_recall_at_k_macro}) "
            f"Hit@{retrieval_aggregate.k}={retrieval_aggregate.hit_at_k_macro}"
        )
    if generation_aggregate:
        print(
            f"Faithfulness={generation_aggregate.faithfulness_macro} "
            f"HallucinationRate={generation_aggregate.hallucination_rate_macro} "
            f"PersonaEvidenceFit={generation_aggregate.persona_evidence_fit_macro}"
        )
    return report


if __name__ == "__main__":
    main()
