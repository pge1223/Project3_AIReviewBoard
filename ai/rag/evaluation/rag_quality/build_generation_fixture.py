# 작성자: 용준/Claude(2026-07-29, 요청: 생성 품질 개선 §1 — 비교 조건 고정)
# 목적: 검색이 이미 개선된 상태에서, 생성 품질 개선 전후 비교에 검색 변동이 섞이지 않도록
#       현재(개선된) 검색 결과를 딱 한 번만 실행해 고정 fixture로 저장한다. 이후
#       generation_regression.py는 이 fixture의 retrieved_contexts를 그대로 재생(replay)만
#       하고 검색을 다시 타지 않는다.
#
#       generation_eval.py의 _make_capturing_evidence_lookup/_initial_state_overrides/
#       _selected_candidate_document_id와 start_ideation_conversation(max_rounds=1)을
#       그대로 재사용한다(export_ragas_dataset.py와 동일한 재사용 패턴) — 검색/생성 로직을
#       재구현하지 않는다.
#
# 실행 예:
#   python -m ai.rag.evaluation.rag_quality.build_generation_fixture \
#       --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v6_regression_subset.jsonl \
#       --top-k 5 \
#       --output ai/rag/evaluation/rag_quality/datasets/generation_fixture_6case.jsonl
from __future__ import annotations

import argparse
import json
import sys
import uuid
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

_MAX_ROUNDS = 1  # generation_eval.py/export_ragas_dataset.py와 동일하게 비용을 짧게 유지한다.


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="검색 결과를 고정한 generation regression fixture를 생성한다(검색은 이번 1회만 실행)"
    )
    parser.add_argument("--dataset", required=True, help="평가셋 JSONL 경로")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output", required=True, help="출력 JSONL 경로")
    parser.add_argument("--chroma-path", default=None)
    return parser.parse_args(argv)


def _build_fixture_rows(
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
            session_id=f"GEN-FIXTURE-{uuid.uuid4().hex[:8]}",
            notice_and_criteria={
                "competition_name": "생성 회귀용 고정 fixture 세션",
                "notice_document": "이 세션은 generation regression fixture 생성 도구가 만든 것입니다.",
            },
            user_idea={"description": case.query},
            llm_call=llm_call,
            max_rounds=_MAX_ROUNDS,
            evidence_lookup=wrapped_lookup,
            initial_state_overrides=_initial_state_overrides(case),
        )
    except Exception as exc:  # noqa: BLE001 - 한 케이스 실패로 전체 fixture 생성을 죽이지 않는다
        print(f"[WARN] case={case.id} 생성 실패(fixture에서 제외): {exc}", file=sys.stderr)
        return []

    session_state_dict = case.session_state.model_dump() if case.session_state else None

    persona_cursor = {"planning_expert": 0, "dev_expert": 0}
    rows: list[dict] = []
    for message_index, message in enumerate(state.get("messages", [])):
        persona_id = message.get("speaker_id")
        if persona_id not in ("planning_expert", "dev_expert"):
            continue
        idx = persona_cursor[persona_id]
        persona_cursor[persona_id] += 1
        retrieved = captured.get(persona_id, [])
        context_for_message = retrieved[idx] if idx < len(retrieved) else []

        structured = message.get("structured") or {}
        current_issue = {
            "active_issue_id": structured.get("active_issue_id") or message.get("active_issue_id"),
            "active_issue_title": structured.get("active_issue_title"),
        }

        rows.append(
            {
                "case_id": case.id,
                "message_index": message_index,
                "message_id": message.get("message_id", ""),
                "user_input": case.query,
                # 요청: 검색을 다시 타지 않기 위해 원문 evidence 항목(dict) 전체를 그대로
                # 담는다(ref/chunk_id/document_role 등 replay에 필요한 필드를 잃지 않는다) —
                # ragas export의 텍스트만 남기는 축약본과는 다른 용도다.
                "retrieved_contexts": context_for_message,
                "session_state": session_state_dict,
                "persona_id": persona_id,
                "current_issue": current_issue,
                "reference": case.reference_answer,
                "expected_source_chunk_ids": case.expected_source_chunk_ids,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = _parse_args(argv)

    from app.config import resolve_chroma_persist_dir, settings
    from ai.rag.domain.config import DEFAULT_COLLECTION_NAME

    dataset = load_cases(args.dataset)
    cases = filter_cases(dataset.cases, case_id=args.case_id)
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
            _build_fixture_rows(case, llm_call=llm_call, role_retrieval_service=role_retrieval_service, top_k=args.top_k)
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"고정 fixture: {output_path} ({len(all_rows)}건, {len(cases)}케이스)")
    return output_path


if __name__ == "__main__":
    main()
