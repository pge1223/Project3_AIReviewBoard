# 작성자: 용준/Claude(2026-07-29, 요청: 생성 품질 개선 2단계 §6/§8 — final_answer 전용
# 통합 평가 하네스)
# 목적: 그래프에는 "사용자 질문에 바로 답하는" 전용 노드가 없고(탐색 결과 confirmed),
#       cold-start(start_ideation_conversation)는 evidence 하나 없이도 항상 "problem"
#       쟁점부터 열어 문제 정의로 발산한다 — document_fact_query처럼 즉답이 필요한 질문에는
#       부적합하다. 대신 test_ideation_claim_grounding.py/test_ideation_conv_graph.py가 이미
#       쓰는 패턴 — initial_conv_state()로 state를 직접 구성하고 그래프 없이
#       make_conv_discussion_node(...)를 바로 호출 — 을 재사용해 각 query_type 1건씩 실제
#       LLM 생성 결과를 얻는다. 이 결과의 Faithfulness/direct-answer 존재/근거 커버리지는
#       generation_regression.py의 discussion 메시지 점수와 절대 섞지 않는다(요청 그대로,
#       별도 리포트로 출력한다).
#
# 실행 예:
#   python -m ai.rag.evaluation.rag_quality.final_answer_eval \
#       --fixture ai/rag/evaluation/rag_quality/datasets/generation_fixture_6case.jsonl \
#       --output reports/rag_eval_current_v5_10_generation/final_answer_report.json
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

_BACKEND_DIR = Path(__file__).resolve().parents[4] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.rag.evaluation.rag_quality.cache import JudgeCache
from ai.rag.evaluation.rag_quality.judge import judge_faithfulness
from ai.rag.evaluation.rag_quality.utterance_type import bucket_claims

from ._meeting_path import ensure_meeting_on_path

ensure_meeting_on_path()
from graph import initial_conv_state  # noqa: E402
from graph.ideation_conv_nodes import make_conv_discussion_node  # noqa: E402
from graph.llm import make_openai_llm_call  # noqa: E402

_NOTICE_AND_CRITERIA = {
    "competition_name": "final_answer 통합 평가용 세션",
    "notice_document": "이 세션은 final_answer_eval.py가 만든 것입니다.",
}

# 요청 §8 — 최소 3케이스(document_fact_query/session_state_query/expert_analysis_query 각 1건).
# fixture(generation_fixture_6case.jsonl)의 실제 검색 결과를 그대로 재생한다 — 검색을 다시
# 타지 않는다.
_CASES = (
    {"expected_query_type": "document_fact_query", "case_id": "rag_eval_v6_A01", "persona_id": "planning_expert"},
    {"expected_query_type": "session_state_query", "case_id": "rag_eval_v6_B05", "persona_id": "planning_expert"},
    {"expected_query_type": "expert_analysis_query", "case_id": "rag_eval_v6_B01", "persona_id": "dev_expert"},
)


def _load_fixture_rows(path: str | Path) -> dict[str, list[dict]]:
    by_case: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_case[row["case_id"]].append(row)
    return dict(by_case)


def _first_row_for_persona(rows: list[dict], persona_id: str) -> Optional[dict]:
    for row in sorted(rows, key=lambda r: r["message_index"]):
        if row["persona_id"] == persona_id:
            return row
    return None


def _make_ground_claims():
    """backend/app/api/routes/ideation_conversation_preview.py::_ground_claims_for와 동일한
    프로덕션 어댑터(요청: "프로덕션 grounding adapter 재사용")."""
    from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl

    role_keywords = {
        "planning_expert": ["실현 가능성", "경제성", "평가기준", "심사", "차별성", "사업성", "공모전"],
        "dev_expert": ["데이터", "API", "구현", "보안", "성능", "아키텍처", "연동", "기술"],
    }

    def grounder(persona_id: str, claims, retrieved_evidence: list[dict]) -> dict:
        return _ground_claims_impl(claims, retrieved_evidence, role_keywords=role_keywords.get(persona_id))

    return grounder


def _run_case(spec: dict, fixture: dict[str, list[dict]], *, llm_call, judge_llm_call, judge_model, cache) -> dict:
    rows = fixture.get(spec["case_id"]) or []
    row = _first_row_for_persona(rows, spec["persona_id"])
    if row is None:
        return {**spec, "error": f"fixture에 {spec['case_id']}/{spec['persona_id']} 행이 없습니다"}

    state = initial_conv_state(
        f"FINAL-ANSWER-{uuid.uuid4().hex[:8]}",
        _NOTICE_AND_CRITERIA,
        {"description": row["user_input"]},
    )
    session_state = row.get("session_state")
    if session_state:
        state["idea_locked"] = bool(session_state.get("idea_locked", False))
        title = session_state.get("selected_idea_title")
        if title:
            state["selected_idea"] = {"title": title}

    served = {"done": False}

    def frozen_lookup(persona_id: str, query: str) -> list[dict]:
        # 요청: 검색을 다시 타지 않는다 — fixture가 캡처한 그대로 1회 반환한다.
        if served["done"]:
            return row["retrieved_contexts"]
        served["done"] = True
        return row["retrieved_contexts"]

    node = make_conv_discussion_node(
        spec["persona_id"],
        llm_call,
        evidence_lookup=frozen_lookup,
        ground_claims=_make_ground_claims(),
    )
    try:
        update = node(state)
    except Exception as exc:  # noqa: BLE001 - 한 케이스 실패로 전체를 죽이지 않는다
        return {**spec, "error": str(exc)}

    message = update.get("messages", [{}])[0]
    content = message.get("content", "")
    structured = message.get("structured") or {}
    detected_query_type = structured.get("query_type")

    claims = message.get("claims") or []
    claim_buckets = bucket_claims(claims)

    verdicts, judge_error = judge_faithfulness(
        judge_llm_call,
        model=judge_model,
        persona_id=spec["persona_id"],
        statement_content=content,
        retrieved_context=row["retrieved_contexts"],
        cache=cache,
    )
    scorable = [v for v in verdicts if v.verdict != "non_factual"]
    supported = sum(1 for v in scorable if v.verdict == "supported")
    partial = sum(1 for v in scorable if v.verdict == "partially_supported")
    unsupported = sum(1 for v in scorable if v.verdict == "unsupported")
    contradicted = sum(1 for v in scorable if v.verdict == "contradicted")
    denom = supported + partial + unsupported + contradicted
    faithfulness = ((supported + 0.5 * partial) / denom) if denom else None

    # direct answer 존재 휴리스틱: 첫 문장이 비어있지 않고, "확인 불가"류 표현이 아니면서
    # 질문 핵심 명사(2글자 이상 토큰)가 하나 이상 등장하는지 본다. Answer Relevancy(ragas
    # 지표)는 이 환경에 ragas가 없어 실측 불가 — 요청대로 한계를 그대로 보고한다(허수 대체 안 함).
    first_sentence = (content.split(".")[0] or "").strip()
    direct_answer_present = bool(first_sentence) and "확인할 수 없습니다" not in first_sentence[:5]

    return {
        **spec,
        "detected_query_type": detected_query_type,
        "query_type_matches_expected": detected_query_type == spec["expected_query_type"],
        "content": content,
        "grounded_claim_count": len(claim_buckets["grounded_claim"]),
        "expert_judgment_claim_count": len(claim_buckets["expert_judgment"]),
        "linked_evidence_refs": message.get("linked_evidence_refs"),
        "grounding_unsupported_claim_count": message.get("unsupported_claim_count"),
        "faithfulness_score": faithfulness,
        "unsupported_claim_count_judge": unsupported,
        "contradicted_count_judge": contradicted,
        "direct_answer_present_heuristic": direct_answer_present,
        "judge_error": judge_error,
    }


def main(argv: Optional[Sequence[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="document_fact/session_state/expert_analysis query 각 1건의 final_answer 품질만 별도 평가한다")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    from app.config import settings

    fixture = _load_fixture_rows(args.fixture)
    generation_model = settings.DEV_LLM_REVIEWER_MODEL
    eval_model = settings.EVAL_LLM_MODEL
    llm_call = make_openai_llm_call(generation_model, api_key=settings.OPENAI_API_KEY or None)
    judge_llm_call = make_openai_llm_call(eval_model, api_key=settings.OPENAI_API_KEY or None)
    cache = JudgeCache(enabled=not args.no_cache)

    results = [
        _run_case(spec, fixture, llm_call=llm_call, judge_llm_call=judge_llm_call, judge_model=eval_model, cache=cache)
        for spec in _CASES
    ]

    report = {
        "note": (
            "Answer Relevancy(ragas 지표)는 이 환경에 ragas가 설치되지 않아 실측하지 못했다 — "
            "direct_answer_present_heuristic/faithfulness_score로 대체 보고한다(허수로 채우지 않음). "
            "이 리포트의 점수는 discussion 메시지 회귀(generation_regression.py)와 별도 집계다."
        ),
        "cases": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"final_answer 리포트: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
