# 작성자: 용준/Claude(2026-07-29)
# 목적: 실제 위원 발언 + 그 발언을 만들 때 쓰인 검색 근거를 Ragas 평가용 JSONL로 내보낸다.
#       LLM judge(judge_faithfulness/judge_persona_fit) 호출은 하지 않는다 — 이 앱 환경에는
#       ragas 패키지를 설치하지 않는다(요청: langchain-community 버전 충돌로 이 환경과
#       분리된 별도 conda 환경(ai/rag/evaluation/ragas_standalone/)에서만 ragas를 돌린다).
#       발언 생성 자체는 generation_eval.py의 _make_capturing_evidence_lookup과
#       start_ideation_conversation을 그대로 재사용한다 — 검색/생성 로직을 재구현하지 않는다.
#
# 실행 예:
#   python -m ai.rag.evaluation.rag_quality.export_ragas_dataset \
#       --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v1.jsonl \
#       --top-k 5 --limit 10 \
#       --output reports/ragas_export/dataset.jsonl
#
# 출력 JSONL 한 줄 = 위원 발언 1건 = Ragas SingleTurnSample 1건:
#   {"case_id", "message_id", "persona_id", "round",
#    "user_input", "response", "retrieved_contexts": [..text..], "reference": str|null}
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

_BACKEND_DIR = Path(__file__).resolve().parents[4] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.rag.evaluation.rag_quality.dataset import filter_cases, load_cases
from ai.rag.evaluation.rag_quality.generation_eval import (
    _initial_state_overrides,
    _make_capturing_evidence_lookup,
    _selected_candidate_document_id,
)
from ai.rag.evaluation.rag_quality.schemas import RagEvalCase
from ai.rag.evaluation.runner import _build_real_retriever

from ._meeting_path import ensure_meeting_on_path

ensure_meeting_on_path()
from graph import start_ideation_conversation  # noqa: E402
from graph.llm import make_openai_llm_call  # noqa: E402

_MAX_ROUNDS = 1  # generation_eval.py와 동일하게 케이스당 비용을 짧게 유지한다.


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="위원 발언 + 검색 근거를 Ragas 평가용 JSONL로 내보낸다(judge 호출 없음)"
    )
    parser.add_argument("--dataset", required=True, help="평가셋 JSONL 경로")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--persona", choices=["planning_expert", "dev_expert"], default=None)
    parser.add_argument("--output", required=True, help="출력 JSONL 파일 경로")
    parser.add_argument(
        "--chroma-path", default=None, help="미지정 시 backend 설정(CHROMA_PERSIST_DIR)을 사용"
    )
    return parser.parse_args(argv)


def _export_case(
    case: RagEvalCase,
    *,
    llm_call,
    role_retrieval_service,
    top_k: int,
) -> list[dict]:
    wrapped_lookup, captured = _make_capturing_evidence_lookup(
        case.filters.project_id,
        role_retrieval_service,
        top_k,
        selected_candidate_document_id=_selected_candidate_document_id(case),
    )
    try:
        state = start_ideation_conversation(
            session_id=f"RAGAS-EXPORT-{uuid.uuid4().hex[:8]}",
            notice_and_criteria={
                "competition_name": "Ragas 평가용 내보내기 세션",
                "notice_document": "이 세션은 Ragas 평가 데이터 내보내기 도구가 생성한 것입니다.",
            },
            user_idea={"description": case.query},
            llm_call=llm_call,
            max_rounds=_MAX_ROUNDS,
            evidence_lookup=wrapped_lookup,
            initial_state_overrides=_initial_state_overrides(case),
        )
    except Exception as exc:  # noqa: BLE001 - 한 케이스 실패로 전체 내보내기를 죽이지 않는다
        print(f"[WARN] case={case.id} 생성 실패: {exc}", file=sys.stderr)
        return []

    persona_cursor = {"planning_expert": 0, "dev_expert": 0}
    rows: list[dict] = []
    for message in state.get("messages", []):
        persona_id = message.get("speaker_id")
        if persona_id not in ("planning_expert", "dev_expert"):
            continue
        idx = persona_cursor[persona_id]
        persona_cursor[persona_id] += 1
        retrieved = captured.get(persona_id, [])
        context_for_message = retrieved[idx] if idx < len(retrieved) else []
        context_texts = [item.get("text") or "" for item in context_for_message if item.get("text")]

        rows.append(
            {
                "case_id": case.id,
                "message_id": message.get("message_id", ""),
                "persona_id": persona_id,
                "round": message.get("round", 1),
                "user_input": case.query,
                "response": message.get("content", ""),
                "retrieved_contexts": context_texts,
                "reference": case.reference_answer,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = _parse_args(argv)

    from app.config import resolve_chroma_persist_dir, settings
    from ai.rag.domain.config import DEFAULT_COLLECTION_NAME

    dataset = load_cases(args.dataset)
    cases = filter_cases(
        dataset.cases,
        case_id=args.case_id,
        persona_id=args.persona,
        limit=args.limit,
    )
    if not cases:
        raise SystemExit("필터 조건에 맞는 케이스가 없습니다.")

    chroma_path = args.chroma_path or settings.CHROMA_PERSIST_DIR
    chroma_path = resolve_chroma_persist_dir(chroma_path)
    role_retrieval_service, _ = _build_real_retriever(chroma_path, DEFAULT_COLLECTION_NAME)

    generation_model = settings.DEV_LLM_REVIEWER_MODEL
    llm_call = make_openai_llm_call(generation_model, api_key=settings.OPENAI_API_KEY or None)

    all_rows: list[dict] = []
    for case in cases:
        all_rows.extend(
            _export_case(case, llm_call=llm_call, role_retrieval_service=role_retrieval_service, top_k=args.top_k)
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset.dataset_name,
                "dataset_version": dataset.version,
                "dataset_path": str(args.dataset),
                "top_k": args.top_k,
                "case_count": len(cases),
                "message_count": len(all_rows),
                "generation_model": generation_model,
                "chroma_collection": DEFAULT_COLLECTION_NAME,
                "reference_answer_count": sum(1 for r in all_rows if r.get("reference")),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"JSONL: {output_path} ({len(all_rows)}건)")
    print(f"매니페스트: {manifest_path}")
    print(
        f"reference 있는 발언: {sum(1 for r in all_rows if r.get('reference'))}/{len(all_rows)} "
        "(없는 발언은 Ragas 환경에서 Context Precision/Context Recall이 '측정 불가'로 표시됩니다)"
    )
    return output_path


if __name__ == "__main__":
    main()
