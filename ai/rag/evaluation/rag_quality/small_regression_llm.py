# 작성자: 용준/Claude(2026-07-29, 요청: 88건 전체 평가 전 실제 LLM 소규모 통합 회귀)
# 목적: 88건 평가(정확히는 case_id 16개 × 메시지 다건 = 88행, reports/ragas_eval/full_88
#       참고)를 다시 돌리기 전에, expert_analysis_query 전용 단일 응답 경로 추가가 기존
#       document_fact_query/session_state_query/ideation_discussion_request 흐름을
#       깨지 않았는지 실제 OpenAI 호출로 확인한다.
#
#       그래프 실행은 review-board 앱 환경에서만 가능하다(ai.meeting.graph/backend.app을
#       직접 import) — ragas-eval 환경은 의도적으로 이 코드를 import하지 않으므로(README
#       참고) 여기서는 실행하지 않는다. Ragas 지표(Faithfulness/Answer Relevancy)가 필요하면
#       이 스크립트의 출력과 별개로 export_ragas_dataset.py → run_ragas_eval.py(ragas-eval
#       환경) 경로를 추가로 타야 한다.
#
#       reply_ideation_conversation()을 직접 호출한다(final_answer_eval.py처럼 discussion
#       노드를 단독 호출하지 않음) — classify_query_type -> _SINGLE_TURN_REQUEST_TYPES ->
#       _drive_graph(stop_after_expert_turn=) -> 실제 그래프 노드까지 프로덕션과 동일한
#       라우팅 경로를 그대로 태워야 "expert_analysis_query가 위원 발언 1건 후 종료되는지"를
#       의미 있게 검증할 수 있기 때문이다.
#
#       A04/B05는 ai/rag/evaluation/rag_quality/datasets/rag_eval_v6_grounded.jsonl에 정의된
#       실제 검수 케이스이고, generation_fixture_6case.jsonl에 그 케이스의 실제 검색 결과가
#       고정(frozen)돼 있다 — 로컬에 Chroma 벡터DB가 없어(.env에 CHROMA_PERSIST_DIR 미설정)
#       실시간 재검색은 불가능하므로, 검색만 이 프로젝트의 기존 관례(final_answer_eval.py/
#       generation_regression.py)대로 재생(replay)하고 생성은 100% 실제 LLM 호출로 한다.
#       expert_analysis_query 2건은 이번에 새로 추가된 라우팅이라 기존 평가셋에 없어 새로
#       작성했다 — A04/A01의 검증된 검색 결과 위에 실제 위원 판단이 필요한 질문을 얹었다.
#
# 보안: API 키는 settings.OPENAI_API_KEY로만 읽고 OpenAI 클라이언트 생성자에만 전달한다.
#       로그·리포트·표준출력 어디에도 키 값을 출력하지 않는다(존재 여부만 별도로 확인).
#
# 실행 예:
#   cd backend && python -m ai.rag.evaluation.rag_quality.small_regression_llm \
#       --output ../reports/small_regression_llm_20260729/report.json --repeats 3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

_BACKEND_DIR = Path(__file__).resolve().parents[4] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.rag.evaluation.rag_quality.cache import JudgeCache
from ai.rag.evaluation.rag_quality.judge import judge_faithfulness
from ai.rag.evaluation.rag_quality.utterance_type import bucket_claims

from ._meeting_path import ensure_meeting_on_path

ensure_meeting_on_path()
from graph import initial_conv_state, reply_ideation_conversation  # noqa: E402
from graph.llm import make_openai_llm_call  # noqa: E402

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "datasets" / "generation_fixture_6case.jsonl"
)
_GROUNDED_DATASET_PATH = (
    Path(__file__).resolve().parent / "datasets" / "rag_eval_v6_grounded.jsonl"
)

_NOTICE_AND_CRITERIA = {
    "competition_name": "소규모 통합 회귀용 세션",
    "notice_document": "이 세션은 small_regression_llm.py가 만든 것입니다.",
}

_FALLBACK_MARKERS = ("현재 자료로 확인할 수 없", "확인할 수 없습니다")


def _load_fixture_contexts(case_id: str) -> list[dict]:
    """generation_fixture_6case.jsonl에서 case_id의 첫 행(message_index 최소)의
    retrieved_contexts를 그대로 재생한다 — 검색을 다시 타지 않는다(기존 관례)."""
    rows = []
    for line in _FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("case_id") == case_id:
            rows.append(row)
    if not rows:
        raise ValueError(f"fixture에 case_id={case_id} 행이 없습니다")
    rows.sort(key=lambda r: r["message_index"])
    return rows[0]["retrieved_contexts"]


def _load_grounded_case(case_id: str) -> dict:
    for line in _GROUNDED_DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") == case_id:
            return row
    raise ValueError(f"rag_eval_v6_grounded.jsonl에 id={case_id} 행이 없습니다")


def _make_ground_claims():
    """backend/app/api/routes/ideation_conversation_preview.py::_ground_claims_for와 동일한
    프로덕션 어댑터(요청: "기존과 동일한 조건")."""
    from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl

    role_keywords = {
        "planning_expert": ["실현 가능성", "경제성", "평가기준", "심사", "차별성", "사업성", "공모전"],
        "dev_expert": ["데이터", "API", "구현", "보안", "성능", "아키텍처", "연동", "기술"],
    }

    def grounder(persona_id: str, claims, retrieved_evidence: list[dict]) -> dict:
        return _ground_claims_impl(claims, retrieved_evidence, role_keywords=role_keywords.get(persona_id))

    return grounder


def _frozen_evidence_lookup(contexts: list[dict]):
    def lookup(persona_id: str, query: str) -> list[dict]:
        return contexts

    return lookup


def _empty_evidence_lookup(persona_id: str, query: str) -> list[dict]:
    return []


class _Case:
    def __init__(
        self,
        key: str,
        expected_request_type: str,
        user_message: str,
        *,
        evidence_lookup=None,
        session_state_overrides: Optional[dict] = None,
        gold_reference: Optional[dict] = None,
        judge_against_evidence: bool = False,
    ):
        self.key = key
        self.expected_request_type = expected_request_type
        self.user_message = user_message
        self.evidence_lookup = evidence_lookup or _empty_evidence_lookup
        self.session_state_overrides = session_state_overrides or {}
        self.gold_reference = gold_reference or {}
        self.judge_against_evidence = judge_against_evidence


def _build_cases() -> list[_Case]:
    a04 = _load_grounded_case("rag_eval_v6_A04")
    b05 = _load_grounded_case("rag_eval_v6_B05")
    a04_contexts = _load_fixture_contexts("rag_eval_v6_A04")
    a01_contexts = _load_fixture_contexts("rag_eval_v6_A01")

    return [
        _Case(
            "A04_document_fact",
            "document_fact_query",
            a04["query"],
            evidence_lookup=_frozen_evidence_lookup(a04_contexts),
            gold_reference={
                "reference_answer": a04["reference_answer"],
                "expected_source_chunk_ids": a04["expected_source_chunk_ids"],
                "expected_evidence_topics": a04["expected_evidence_topics"],
            },
            judge_against_evidence=True,
        ),
        _Case(
            "B05_session_state",
            "session_state_query",
            b05["query"],
            evidence_lookup=_frozen_evidence_lookup(a04_contexts),  # 사실상 사용 안 됨(결정론적 응답)
            session_state_overrides={
                "idea_locked": True,
                "selected_idea": {"title": b05["session_state"]["selected_idea_title"]},
            },
            gold_reference={"expected_title": b05["session_state"]["selected_idea_title"]},
        ),
        _Case(
            "EA1_expert_analysis_data_risk",
            "expert_analysis_query",
            "이 실증·PoC 사업에서 데이터 확보 방식의 구현 가능성과 기술적 위험 요소를 분석해 주세요",
            evidence_lookup=_frozen_evidence_lookup(a04_contexts),
            judge_against_evidence=True,
        ),
        _Case(
            "EA2_expert_analysis_differentiation",
            "expert_analysis_query",
            "이 아이디어가 심사에서 차별성을 인정받으려면 어떤 개선 제안이 필요할까요?",
            evidence_lookup=_frozen_evidence_lookup(a01_contexts),
            judge_against_evidence=True,
        ),
        _Case(
            "D1_ideation_discussion_request",
            "ideation_discussion_request",
            "이 부분에 대해 위원들이 좀 더 토론해 주세요",
            evidence_lookup=_empty_evidence_lookup,
        ),
    ]


def _initial_replyable_state(case: _Case) -> dict:
    state = dict(
        initial_conv_state(
            f"SMALLREG-{uuid.uuid4().hex[:8]}",
            _NOTICE_AND_CRITERIA,
            {"description": "회귀 테스트용 베이스 세션"},
            max_rounds=1,
        )
    )
    state["phase"] = "discussion_complete"
    state["messages"] = []
    state.update(case.session_state_overrides)
    return state


def _has_fallback_phrase(text: str) -> bool:
    return any(marker in text for marker in _FALLBACK_MARKERS)


def _run_single(case: _Case, *, llm_call, judge_llm_call, judge_model, cache) -> dict:
    state = _initial_replyable_state(case)
    try:
        result_state = reply_ideation_conversation(
            previous_state=state,
            user_message=case.user_message,
            llm_call=llm_call,
            evidence_lookup=case.evidence_lookup,
            ground_claims=_make_ground_claims(),
        )
    except Exception as exc:  # noqa: BLE001 - 한 반복 실패로 전체를 죽이지 않는다
        return {"error": f"{type(exc).__name__}: {exc}"}

    new_messages = result_state["messages"]
    speakers = [m["speaker_id"] for m in new_messages]
    committee_messages = [m for m in new_messages if m["speaker_id"] != "user"]
    last_message = committee_messages[-1] if committee_messages else {}
    content = last_message.get("content", "")
    claims = last_message.get("claims") or []
    claim_buckets = bucket_claims(claims)

    row: dict[str, Any] = {
        "request_type": result_state.get("request_type"),
        "expected_request_type": case.expected_request_type,
        "request_type_matches": result_state.get("request_type") == case.expected_request_type,
        "phase": result_state.get("phase"),
        "phase_is_discussion_complete": result_state.get("phase") == "discussion_complete",
        "speaker_sequence": speakers,
        "committee_message_count": len(committee_messages),
        "content": content,
        "content_first_sentence": (content.split(".")[0] or "").strip(),
        "has_fallback_phrase": _has_fallback_phrase(content),
        "grounded_claim_count": len(claim_buckets["grounded_claim"]),
        "expert_judgment_claim_count": len(claim_buckets["expert_judgment"]),
        "linked_evidence_refs": last_message.get("linked_evidence_refs"),
        "unsupported_claim_count_grounding": last_message.get("unsupported_claim_count"),
        "message_ids": [m.get("message_id") for m in new_messages],
        "message_ids_unique": len({m.get("message_id") for m in new_messages}) == len(new_messages),
    }

    if case.judge_against_evidence and content:
        evidence = case.evidence_lookup("planning_expert", case.user_message)
        verdicts, judge_error = judge_faithfulness(
            judge_llm_call,
            model=judge_model,
            persona_id=last_message.get("speaker_id", "planning_expert"),
            statement_content=content,
            retrieved_context=evidence,
            cache=cache,
        )
        scorable = [v for v in verdicts if v.verdict != "non_factual"]
        unsupported = sum(1 for v in scorable if v.verdict == "unsupported")
        contradicted = sum(1 for v in scorable if v.verdict == "contradicted")
        row["judge_unsupported_claim_count"] = unsupported
        row["judge_contradicted_claim_count"] = contradicted
        row["judge_error"] = judge_error
    else:
        row["judge_unsupported_claim_count"] = None
        row["judge_contradicted_claim_count"] = None
        row["judge_error"] = None

    if case.key == "B05_session_state":
        expected_title = case.gold_reference.get("expected_title", "")
        row["title_exact_match"] = expected_title in content
        row["mentions_other_candidate_as_final"] = False  # 별도 후보 title이 없어 오인 케이스 없음(단일 후보 세션)

    return row


def _aggregate(case_key: str, repeats: list[dict]) -> dict:
    ok_repeats = [r for r in repeats if "error" not in r]
    n = len(repeats)
    success = len(ok_repeats)
    agg: dict[str, Any] = {
        "case_key": case_key,
        "repeats_total": n,
        "repeats_ok": success,
        "success_rate": success / n if n else 0.0,
        "errors": [r["error"] for r in repeats if "error" in r],
    }
    if ok_repeats:
        agg["request_type_match_rate"] = sum(1 for r in ok_repeats if r["request_type_matches"]) / len(ok_repeats)
        agg["discussion_complete_rate"] = sum(1 for r in ok_repeats if r["phase_is_discussion_complete"]) / len(ok_repeats)
        agg["fallback_phrase_rate"] = sum(1 for r in ok_repeats if r["has_fallback_phrase"]) / len(ok_repeats)
        agg["message_ids_unique_rate"] = sum(1 for r in ok_repeats if r["message_ids_unique"]) / len(ok_repeats)
        counts = [r["committee_message_count"] for r in ok_repeats]
        agg["committee_message_count_mean"] = statistics.mean(counts)
        agg["committee_message_count_stdev"] = statistics.stdev(counts) if len(counts) > 1 else 0.0
        judge_unsupported = [r["judge_unsupported_claim_count"] for r in ok_repeats if r["judge_unsupported_claim_count"] is not None]
        if judge_unsupported:
            agg["judge_unsupported_claim_count_mean"] = statistics.mean(judge_unsupported)
            agg["judge_unsupported_claim_count_stdev"] = (
                statistics.stdev(judge_unsupported) if len(judge_unsupported) > 1 else 0.0
            )
        agg["sample_content"] = ok_repeats[0]["content"]
        agg["sample_speaker_sequence"] = ok_repeats[0]["speaker_sequence"]
    return agg


def main(argv: Optional[list[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="88건 평가 전 실제 LLM 소규모 통합 회귀(A04/B05/expert_analysis 2건/discussion 1건, 각 N회)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    from app.config import settings

    if not settings.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY가 설정되지 않았습니다(backend/.env 확인) — 키 값은 여기 출력하지 않습니다.")

    generation_model = settings.reviewer_model()
    judge_model = settings.EVAL_LLM_MODEL
    llm_call = make_openai_llm_call(generation_model, api_key=settings.OPENAI_API_KEY or None)
    judge_llm_call = make_openai_llm_call(judge_model, api_key=settings.OPENAI_API_KEY or None)
    cache = JudgeCache(enabled=not args.no_cache)

    cases = _build_cases()
    report: dict[str, Any] = {
        "note": (
            "실제 OpenAI 호출(생성 모델="
            + generation_model
            + ", 판정 모델="
            + judge_model
            + ") 결과 — API 키 값은 어디에도 기록하지 않았다. 검색은 A04/B05/EA1/EA2 모두 "
            "generation_fixture_6case.jsonl에 고정된 실제 검색 결과를 재생했다(로컬에 Chroma "
            "벡터DB가 없어 재검색 불가) — 생성·판정만 100% 실시간 LLM 호출이다."
        ),
        "generation_model": generation_model,
        "judge_model": judge_model,
        "repeats_per_case": args.repeats,
        "cases": [],
    }

    for case in cases:
        repeats = [
            _run_single(case, llm_call=llm_call, judge_llm_call=judge_llm_call, judge_model=judge_model, cache=cache)
            for _ in range(args.repeats)
        ]
        report["cases"].append({
            "case_key": case.key,
            "expected_request_type": case.expected_request_type,
            "user_message": case.user_message,
            "repeats": repeats,
            "aggregate": _aggregate(case.key, repeats),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"소규모 회귀 리포트: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
