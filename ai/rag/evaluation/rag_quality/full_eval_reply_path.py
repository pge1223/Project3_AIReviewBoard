# 작성자: 용준/Claude(2026-07-29, 요청: 88건 전체 평가 — 실제 최상위 라우팅 경로로 재실행)
# 목적: 기존 export_ragas_dataset.py/generation_eval.py는 case.query를
#       start_ideation_conversation(user_idea={"description": case.query})로 "세션의 원래
#       아이디어"인 것처럼 밀어넣는다 — 이렇게 하면 session_state_query가 실제로 겪는 버그
#       (B05: 결정론적 답변이 _topic_query를 보느라 최신 질문을 놓침)가 우연히 가려진다.
#       이 스크립트는 실제 프로덕션 경로 그대로 재현한다:
#         1) start_ideation_conversation으로 세션을 만들되(사건과 무관한 중립 아이디어 +
#            case.session_state 기반 initial_state_overrides로 "이미 진행된 세션" 재현)
#         2) reply_ideation_conversation(user_message=case.query)로 최신 질문을 진짜
#            reply로 흘려보낸다 — classify_query_type/current_user_input/
#            _SINGLE_TURN_REQUEST_TYPES/_deterministic_session_state_answer가 전부
#            실제 호출 경로 그대로 실행된다.
#       검색은 실제 로컬 Chroma(project_documents_kure_v6)를 그대로 사용한다(프레임 고정
#       재생이 아니다 — 이 환경에 실제 색인된 데이터가 있음을 확인 후 사용).
#
# 산출물:
#   --output-jsonl: Ragas SingleTurnSample 형식(case_id/message_id/persona_id/round/
#     user_input/response/retrieved_contexts/reference) — ragas-eval 환경에 그대로 넘긴다.
#   --output-report: request_type/phase/메시지 수/unsupported claim/제목 정확도 등 이
#     환경(review-board)에서 계산 가능한 모든 커스텀 지표.
#
# 실행 예:
#   python -m ai.rag.evaluation.rag_quality.full_eval_reply_path \
#       --dataset ai/rag/evaluation/rag_quality/datasets/rag_eval_v6_grounded.jsonl \
#       --top-k 5 \
#       --output-jsonl reports/ragas_eval/postfix_e214a40/postfix_samples.jsonl \
#       --output-report reports/ragas_eval/postfix_e214a40/postfix_report.json
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

_BACKEND_DIR = Path(__file__).resolve().parents[4] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.rag.evaluation.rag_quality.cache import JudgeCache
from ai.rag.evaluation.rag_quality.dataset import load_cases
from ai.rag.evaluation.rag_quality.generation_eval import (
    _check_forbidden_claims,
    _initial_state_overrides,
    _make_capturing_evidence_lookup,
    _selected_candidate_document_id,
)
from ai.rag.evaluation.rag_quality.judge import judge_faithfulness
from ai.rag.evaluation.rag_quality.schemas import RagEvalCase
from ai.rag.evaluation.rag_quality.utterance_type import bucket_claims
from ai.rag.evaluation.runner import _build_real_retriever

from ._meeting_path import ensure_meeting_on_path

ensure_meeting_on_path()
from graph import initial_conv_state, reply_ideation_conversation  # noqa: E402
from graph.ideation_conv_nodes import classify_query_type  # noqa: E402
from graph.llm import make_openai_llm_call  # noqa: E402

# 용준/Claude(2026-07-29, 실측 발견) — 처음에는 start_ideation_conversation으로 "무관한
# 중립 아이디어"를 실제로 한 라운드 토론시킨 뒤 reply로 진짜 질문을 흘려보내려 했으나,
# 실측 결과 그 중립 아이디어(예: "손님 문의 챗봇")에 대한 실제 토론 메시지 9건이
# state["messages"]/problem_definition/idea_canvas 등에 남아 프롬프트 컨텍스트를 오염시켜,
# 실제 질문("실증·PoC 수행보고서 데이터 항목")에 답해야 할 위원이 계속 챗봇 얘기로
# 새는 문제가 실측됐다(A04 드라이런에서 응답이 질문과 전혀 무관한 챗봇 얘기로 나옴).
# small_regression_llm.py에서 이미 검증된 방식(initial_conv_state로 state를 직접 구성,
# messages=[], phase="discussion_complete")을 그대로 재사용한다 — 이것도 "질문을
# user_idea에 넣어 우연히 통과"시키는 방식이 아니다(user_idea는 그대로 중립이고, 질문은
# 오직 reply_ideation_conversation(user_message=...)로만 들어간다). 대신 불필요한 라운드
# 생성 비용과 무관한 대화 이력에 의한 오염을 없앤다.
_NEUTRAL_SESSION_IDEA = "소상공인이 손님 문의에 자동으로 답하는 챗봇"
_NOTICE_AND_CRITERIA = {
    "competition_name": "88건 재평가용 세션(실제 라우팅 경로)",
    "notice_document": "이 세션은 full_eval_reply_path.py가 만든 것입니다.",
}
_FALLBACK_MARKERS = ("현재 자료로 확인할 수 없", "확인할 수 없습니다")


def _make_ground_claims():
    from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl

    role_keywords = {
        "planning_expert": ["실현 가능성", "경제성", "평가기준", "심사", "차별성", "사업성", "공모전"],
        "dev_expert": ["데이터", "API", "구현", "보안", "성능", "아키텍처", "연동", "기술"],
    }

    def grounder(persona_id: str, claims, retrieved_evidence: list[dict]) -> dict:
        return _ground_claims_impl(claims, retrieved_evidence, role_keywords=role_keywords.get(persona_id))

    return grounder


def _run_case(
    case: RagEvalCase,
    *,
    llm_call,
    judge_llm_call,
    judge_model: str,
    role_retrieval_service,
    top_k: int,
    cache: JudgeCache,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": case.id,
        "query": case.query,
        "expected_persona_id": case.persona_id,
        "project_id": case.filters.project_id,
    }
    started = time.perf_counter()

    # 1) 세션 생성 — 그래프로 실제 라운드테이블을 돌리지 않고 state를 직접 구성한다
    # (이유는 위 _NEUTRAL_SESSION_IDEA 주석 참고: 중립 아이디어를 실제로 토론시키면 그
    # 대화 이력이 프롬프트를 오염시켜 실제 질문에 대한 답이 엉뚱한 주제로 샌다 — 실측
    # 확인). user_idea는 여전히 평가 질문과 무관한 중립 문구다(요청: "질문을 user_idea에
    # 넣어서 우연히 통과시키는 기존 평가 방식은 사용하지 마세요" — 이 방식은 질문을
    # user_idea에 넣지 않는다, reply_ideation_conversation에만 넣는다). session_state가
    # 있으면 "이미 그 세션이 후보를 확정한 상태"를 재현한다(검색 입력 전용 필드만 사용 —
    # 채점 필드는 여기 들어가지 않는다).
    state = dict(
        initial_conv_state(
            f"FULLEVAL-{uuid.uuid4().hex[:8]}",
            _NOTICE_AND_CRITERIA,
            {"description": _NEUTRAL_SESSION_IDEA},
            max_rounds=3,
        )
    )
    state["phase"] = "discussion_complete"
    state["messages"] = []
    overrides = _initial_state_overrides(case)
    if overrides:
        state.update(overrides)

    baseline_count = 0
    bootstrap_message_count = 0

    # 2) 실제 질문을 진짜 reply로 흘려보낸다 — classify_query_type이 이 원문으로 판정하고,
    # current_user_input도 이 원문으로 갱신된다(최상위 라우터 실제 경로).
    wrapped_lookup, captured = _make_capturing_evidence_lookup(
        case.filters.project_id, role_retrieval_service, top_k,
        selected_candidate_document_id=_selected_candidate_document_id(case),
    )
    try:
        reply_state = reply_ideation_conversation(
            previous_state=state,
            user_message=case.query,
            llm_call=llm_call,
            evidence_lookup=wrapped_lookup,
            ground_claims=_make_ground_claims(),
            stop_after_expert_turn=False,  # 요청: 임의로 강제 정지/강제 다회 실행 금지 —
            # 라우터(_SINGLE_TURN_REQUEST_TYPES)가 스스로 판단하게 둔다.
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"reply_failed: {type(exc).__name__}: {exc}"
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        result["bootstrap_message_count"] = bootstrap_message_count
        return result
    elapsed_ms = (time.perf_counter() - started) * 1000

    new_messages = reply_state.get("messages", [])[baseline_count:]
    speakers = [m.get("speaker_id") for m in new_messages]
    committee_messages = [m for m in new_messages if m.get("speaker_id") not in ("user", None)]

    result.update(
        {
            "request_type": reply_state.get("request_type"),
            "classify_query_type_direct": classify_query_type(case.query),
            "phase_final": reply_state.get("phase"),
            "bootstrap_message_count": bootstrap_message_count,
            "reply_message_count": len(new_messages),
            "reply_speaker_sequence": speakers,
            "elapsed_ms": elapsed_ms,
        }
    )

    persona_cursor = {"planning_expert": 0, "dev_expert": 0}
    message_rows: list[dict[str, Any]] = []
    for message in committee_messages:
        persona_id = message.get("speaker_id")
        if persona_id not in ("planning_expert", "dev_expert"):
            continue  # 진행자 발언은 ragas 표본에서 제외(기존 export_ragas_dataset.py와 동일)
        idx = persona_cursor[persona_id]
        persona_cursor[persona_id] += 1
        retrieved = captured.get(persona_id, [])
        context_for_message = retrieved[idx] if idx < len(retrieved) else []
        context_texts = [item.get("text") or "" for item in context_for_message if item.get("text")]
        retrieved_chunk_ids = [item.get("chunk_id") for item in context_for_message if item.get("chunk_id")]
        retrieved_document_ids = [item.get("document_id") for item in context_for_message if item.get("document_id")]

        content = message.get("content", "")
        claims = message.get("claims") or []
        claim_buckets = bucket_claims(claims)
        forbidden_hits = _check_forbidden_claims(content, case.forbidden_claims)

        verdicts, judge_error = judge_faithfulness(
            judge_llm_call,
            model=judge_model,
            persona_id=persona_id,
            statement_content=content,
            retrieved_context=context_for_message,
            cache=cache,
        )
        scorable = [v for v in verdicts if v.verdict != "non_factual"]
        supported = sum(1 for v in scorable if v.verdict == "supported")
        partial = sum(1 for v in scorable if v.verdict == "partially_supported")
        unsupported = sum(1 for v in scorable if v.verdict == "unsupported")
        contradicted = sum(1 for v in scorable if v.verdict == "contradicted")
        denom = supported + partial + unsupported + contradicted
        faithfulness = ((supported + 0.5 * partial) / denom) if denom else None

        expected_source_hit = bool(
            case.expected_source_chunk_ids and set(case.expected_source_chunk_ids) & set(retrieved_chunk_ids)
        )
        gold_doc_hit = bool(case.gold_document_ids and set(case.gold_document_ids) & set(retrieved_document_ids))

        message_rows.append(
            {
                "message_id": message.get("message_id", ""),
                "persona_id": persona_id,
                "round": message.get("round", 1),
                "content": content,
                "has_fallback_phrase": any(marker in content for marker in _FALLBACK_MARKERS),
                "retrieved_contexts": context_texts,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_document_ids": retrieved_document_ids,
                "expected_source_hit": expected_source_hit,
                "gold_document_hit": gold_doc_hit,
                "grounded_claim_count": len(claim_buckets["grounded_claim"]),
                "expert_judgment_claim_count": len(claim_buckets["expert_judgment"]),
                "linked_evidence_refs": message.get("linked_evidence_refs"),
                "grounding_unsupported_claim_count": message.get("unsupported_claim_count"),
                "forbidden_claim_hits": forbidden_hits,
                "judge_faithfulness_score": faithfulness,
                "judge_unsupported_claim_count": unsupported,
                "judge_contradicted_claim_count": contradicted,
                "judge_error": judge_error,
            }
        )
    result["messages"] = message_rows

    # session_state_query 전용: 확정 제목이 정확히 포함됐는지(제목 변형·요약 없이).
    if case.session_state and case.session_state.selected_idea_title:
        title = case.session_state.selected_idea_title
        result["expected_title"] = title
        result["title_exact_match_any_message"] = any(
            title in (m.get("content") or "") for m in committee_messages
        )

    return result


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="88건(16 case) 평가 — 실제 최상위 라우팅 경로(start+reply)로 재실행"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output-jsonl", required=True, help="ragas-eval 환경에 넘길 SingleTurnSample JSONL")
    parser.add_argument("--output-report", required=True, help="이 환경에서 계산 가능한 커스텀 지표 리포트")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    from app.config import resolve_chroma_persist_dir, settings
    from ai.rag.domain.config import DEFAULT_COLLECTION_NAME

    if not settings.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY가 설정되지 않았습니다 — 키 값은 여기 출력하지 않습니다.")

    dataset = load_cases(args.dataset)
    cases = dataset.cases
    if args.case_id:
        cases = [c for c in cases if c.id == args.case_id]
    if not cases:
        raise SystemExit("조건에 맞는 케이스가 없습니다.")

    chroma_path = resolve_chroma_persist_dir(settings.CHROMA_PERSIST_DIR)
    role_retrieval_service, retrieval_settings = _build_real_retriever(chroma_path, DEFAULT_COLLECTION_NAME)

    generation_model = settings.reviewer_model()
    judge_model = settings.EVAL_LLM_MODEL
    llm_call = make_openai_llm_call(generation_model, api_key=settings.OPENAI_API_KEY or None)
    judge_llm_call = make_openai_llm_call(judge_model, api_key=settings.OPENAI_API_KEY or None)
    cache = JudgeCache(enabled=not args.no_cache)

    case_results = []
    for case in cases:
        print(f"[{case.id}] 실행 중...", file=sys.stderr)
        case_results.append(
            _run_case(
                case,
                llm_call=llm_call,
                judge_llm_call=judge_llm_call,
                judge_model=judge_model,
                role_retrieval_service=role_retrieval_service,
                top_k=args.top_k,
                cache=cache,
            )
        )

    # ragas-eval 환경으로 넘길 JSONL(SingleTurnSample 1건 = 위원 발언 1건).
    ragas_rows = []
    for case, result in zip(cases, case_results):
        if "error" in result:
            continue
        for m in result.get("messages", []):
            ragas_rows.append(
                {
                    "case_id": case.id,
                    "message_id": m["message_id"],
                    "persona_id": m["persona_id"],
                    "round": m["round"],
                    "user_input": case.query,
                    "response": m["content"],
                    "retrieved_contexts": m["retrieved_contexts"],
                    "reference": case.reference_answer,
                }
            )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in ragas_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_reply_messages = sum(r.get("reply_message_count", 0) for r in case_results if "error" not in r)
    report = {
        "commit": None,  # main()이 호출부에서 채운다(git 정보를 코드가 임의로 지어내지 않음)
        "generation_model": generation_model,
        "judge_model": judge_model,
        "retrieval_collection": DEFAULT_COLLECTION_NAME,
        "case_count": len(cases),
        "case_count_ok": sum(1 for r in case_results if "error" not in r),
        "case_count_error": sum(1 for r in case_results if "error" in r),
        "total_reply_message_count": total_reply_messages,
        "total_ragas_row_count": len(ragas_rows),
        "cases": case_results,
    }
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"JSONL(ragas용): {output_jsonl} ({len(ragas_rows)}행)", file=sys.stderr)
    print(f"리포트: {output_report}", file=sys.stderr)
    print(f"case={len(cases)} ok={report['case_count_ok']} error={report['case_count_error']} reply_messages={total_reply_messages}", file=sys.stderr)


if __name__ == "__main__":
    main()
