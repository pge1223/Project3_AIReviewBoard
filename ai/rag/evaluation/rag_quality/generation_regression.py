# 작성자: 용준/Claude(2026-07-29, 요청: 생성 품질 개선 §1/§7 — 검색 고정, 생성만 반복 비교)
# 목적: build_generation_fixture.py가 저장한 고정 retrieved_contexts를 그대로 재생(replay)
#       하는 evidence_lookup 스텁을 만들어 start_ideation_conversation을 다시 실행한다 —
#       검색은 절대 다시 타지 않는다. 같은 6케이스를 3회 반복 생성해 평균/표준편차를 낸다.
#
#       judge_faithfulness/judge_persona_fit/judge_expert_judgment, aggregate_generation_by_type
#       (generation_eval.py, judge.py, utterance_type.py)를 그대로 재사용한다 — 판정 로직을
#       새로 만들지 않는다.
#
# 실행 예:
#   python -m ai.rag.evaluation.rag_quality.generation_regression \
#       --fixture ai/rag/evaluation/rag_quality/datasets/generation_fixture_6case.jsonl \
#       --repeats 3 \
#       --output reports/rag_eval_current_v5_10_generation/regression_report.json
from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

_BACKEND_DIR = Path(__file__).resolve().parents[4] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ai.rag.evaluation.rag_quality.cache import JudgeCache
from ai.rag.evaluation.rag_quality.generation_eval import (
    _check_forbidden_claims,
    aggregate_generation_by_type,
)
from ai.rag.evaluation.rag_quality.judge import judge_expert_judgment, judge_faithfulness, judge_persona_fit
from ai.rag.evaluation.rag_quality.schemas import GenerationCaseResult, MessageEvalResult
from ai.rag.evaluation.rag_quality.utterance_type import bucket_claims, classify_message

from ._meeting_path import ensure_meeting_on_path

ensure_meeting_on_path()
from graph import start_ideation_conversation  # noqa: E402
from graph.llm import make_openai_llm_call  # noqa: E402

_MAX_ROUNDS = 1

# 용준/Claude(2026-07-29, 요청: 생성 품질 개선 §5/§7 회귀 실행 중 발견) — generation_eval.py도
# 원래 ground_claims를 주입하지 않아서(ai/meeting/graph는 ai.rag를 직접 import하지 않는
# 경계 때문에 항상 backend 레이어가 주입한다) 이 오프라인 하네스는 실제 세션과 달리 근거
# 게이팅(_ground_and_finalize_claims)이 전혀 실행되지 않는 채로 생성돼 왔다. 실제 프로덕션
# 주입부(backend/app/api/routes/ideation_conversation_preview.py::_ground_claims_for)와
# 동일한 어댑터를 그대로 재사용해, 이 회귀 실행에서는 실제 세션과 같은 근거 게이팅이 돌게
# 한다.
_ROLE_RELEVANCE_KEYWORDS = {
    "planning_expert": ["실현 가능성", "경제성", "평가기준", "심사", "차별성", "사업성", "공모전"],
    "dev_expert": ["데이터", "API", "구현", "보안", "성능", "아키텍처", "연동", "기술"],
}


def _make_ground_claims():
    from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl

    def grounder(persona_id: str, claims, retrieved_evidence: list[dict]) -> dict:
        return _ground_claims_impl(claims, retrieved_evidence, role_keywords=_ROLE_RELEVANCE_KEYWORDS.get(persona_id))

    return grounder


def _load_fixture(path: str | Path) -> dict[str, list[dict]]:
    """case_id -> 그 케이스의 fixture row 리스트(message_index 순)."""
    by_case: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_case[row["case_id"]].append(row)
    for rows in by_case.values():
        rows.sort(key=lambda r: r["message_index"])
    return dict(by_case)


def _make_frozen_lookup(case_rows: list[dict]):
    """build_generation_fixture.py가 캡처한 순서 그대로 재생하는 스텁. 실제 검색은 절대
    호출하지 않는다 — persona_id별로 fixture에 기록된 순서대로 retrieved_contexts를
    돌려준다. 새 실행이 원본 fixture보다 같은 persona 호출을 더 많이 하면(회의 라우팅이
    이번 생성에서 달라진 경우) 마지막 캡처값을 반복 재사용한다 — 검색을 다시 타지 않기
    위한 의도적 선택이다."""
    per_persona: dict[str, list[list[dict]]] = defaultdict(list)
    for row in case_rows:
        per_persona[row["persona_id"]].append(row["retrieved_contexts"])
    cursor = defaultdict(int)

    def frozen(persona_id: str, topic_query: str) -> list[dict]:
        items = per_persona.get(persona_id) or []
        if not items:
            return []
        idx = min(cursor[persona_id], len(items) - 1)
        cursor[persona_id] += 1
        return items[idx]

    return frozen


def _initial_state_overrides_from_session_state(session_state: Optional[dict]) -> Optional[dict]:
    if not session_state:
        return None
    overrides: dict = {"idea_locked": bool(session_state.get("idea_locked", False))}
    title = session_state.get("selected_idea_title")
    if title:
        overrides["selected_idea"] = {"title": title}
    return overrides


def _run_one_repeat(
    case_id: str,
    case_rows: list[dict],
    *,
    llm_call,
    judge_llm_call,
    judge_model: str,
    cache: Optional[JudgeCache],
    evaluate_expert_judgment: bool,
) -> GenerationCaseResult:
    user_input = case_rows[0]["user_input"]
    session_state = case_rows[0].get("session_state")
    reference = case_rows[0].get("reference")
    forbidden_claims: list[str] = []  # fixture는 채점 필드(forbidden_claims)를 담지 않는다(§1 요청)

    frozen_lookup = _make_frozen_lookup(case_rows)
    try:
        state = start_ideation_conversation(
            session_id=f"GEN-REGRESSION-{uuid.uuid4().hex[:8]}",
            notice_and_criteria={
                "competition_name": "생성 회귀 재실행 세션(검색 고정)",
                "notice_document": "이 세션은 generation_regression.py가 고정 fixture로 재실행한 것입니다.",
            },
            user_idea={"description": user_input},
            llm_call=llm_call,
            max_rounds=_MAX_ROUNDS,
            evidence_lookup=frozen_lookup,
            ground_claims=_make_ground_claims(),
            initial_state_overrides=_initial_state_overrides_from_session_state(session_state),
        )
    except Exception as exc:  # noqa: BLE001
        return GenerationCaseResult(case_id=case_id, query=user_input, generation_error=str(exc))

    persona_cursor = {"planning_expert": 0, "dev_expert": 0}
    per_persona_context: dict[str, list[list[dict]]] = defaultdict(list)
    for row in case_rows:
        per_persona_context[row["persona_id"]].append(row["retrieved_contexts"])

    message_results: list[MessageEvalResult] = []
    for message in state.get("messages", []):
        persona_id = message.get("speaker_id")
        if persona_id not in ("planning_expert", "dev_expert"):
            continue
        idx = persona_cursor[persona_id]
        persona_cursor[persona_id] += 1
        contexts = per_persona_context.get(persona_id) or []
        context_for_message = contexts[min(idx, len(contexts) - 1)] if contexts else []

        content = message.get("content", "")
        claims, faith_error = judge_faithfulness(
            judge_llm_call,
            model=judge_model,
            persona_id=persona_id,
            statement_content=content,
            retrieved_context=context_for_message,
            cache=cache,
        )
        persona_fit, fit_error = judge_persona_fit(
            judge_llm_call,
            model=judge_model,
            persona_id=persona_id,
            message_id=message.get("message_id", ""),
            statement_content=content,
            retrieved_context=context_for_message,
            cache=cache,
        )

        scorable = [c for c in claims if c.verdict != "non_factual"]
        supported = sum(1 for c in scorable if c.verdict == "supported")
        partial = sum(1 for c in scorable if c.verdict == "partially_supported")
        unsupported = sum(1 for c in scorable if c.verdict == "unsupported")
        contradicted = sum(1 for c in scorable if c.verdict == "contradicted")
        denom = supported + partial + unsupported + contradicted

        utterance_type = classify_message(message)
        source_claim_buckets = bucket_claims(message.get("claims") or [])
        expert_judgments = []
        if evaluate_expert_judgment and source_claim_buckets["expert_judgment"]:
            ej_text = " ".join(c.get("text", "") for c in source_claim_buckets["expert_judgment"] if c.get("text"))
            ej_result, ej_error = judge_expert_judgment(
                judge_llm_call,
                model=judge_model,
                persona_id=persona_id,
                message_id=message.get("message_id", ""),
                claim_id=None,
                statement_content=ej_text or content,
                grounded_claims=source_claim_buckets["grounded_claim"],
                cache=cache,
            )
            if ej_result is not None:
                expert_judgments.append(ej_result)
            elif ej_error:
                faith_error = faith_error or ej_error

        message_results.append(
            MessageEvalResult(
                case_id=case_id,
                message_id=message.get("message_id", ""),
                persona_id=persona_id,
                round=message.get("round", 1),
                content_preview=content[:200],
                claims=claims,
                faithfulness_score=((supported + 0.5 * partial) / denom) if denom else None,
                hallucination_rate=((unsupported + contradicted) / denom) if denom else None,
                unsupported_count=unsupported,
                contradicted_count=contradicted,
                non_factual_count=len(claims) - len(scorable),
                forbidden_claim_hits=_check_forbidden_claims(content, forbidden_claims),
                persona_fit=persona_fit,
                judge_error=faith_error or fit_error,
                utterance_type=utterance_type,
                grounded_claim_count=message.get("grounded_claim_count"),
                expert_judgment_claim_count=message.get("expert_judgment_count"),
                grounding_unsupported_claim_count=message.get("unsupported_claim_count"),
                linked_evidence_refs=list(message.get("linked_evidence_refs") or []),
                expert_judgments=expert_judgments,
            )
        )

    return GenerationCaseResult(case_id=case_id, query=user_input, messages=message_results)


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "stdev": None, "n": 0}
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "stdev": stdev, "n": len(values)}


def main(argv: Optional[Sequence[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="고정 fixture로 생성만 N회 반복 재실행해 평균/표준편차를 낸다")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-expert-judgment", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    from app.config import settings

    fixture = _load_fixture(args.fixture)
    generation_model = settings.DEV_LLM_REVIEWER_MODEL
    eval_model = settings.EVAL_LLM_MODEL
    llm_call = make_openai_llm_call(generation_model, api_key=settings.OPENAI_API_KEY or None)
    judge_llm_call = make_openai_llm_call(eval_model, api_key=settings.OPENAI_API_KEY or None)
    cache = JudgeCache(enabled=not args.no_cache)

    per_case_repeats: dict[str, list[GenerationCaseResult]] = defaultdict(list)
    all_repeat_results: list[GenerationCaseResult] = []
    for case_id, case_rows in fixture.items():
        for repeat_idx in range(args.repeats):
            result = _run_one_repeat(
                case_id,
                case_rows,
                llm_call=llm_call,
                judge_llm_call=judge_llm_call,
                judge_model=eval_model,
                cache=cache,
                evaluate_expert_judgment=not args.no_expert_judgment,
            )
            per_case_repeats[case_id].append(result)
            all_repeat_results.append(result)
            print(f"[{case_id}] repeat {repeat_idx + 1}/{args.repeats} done "
                  f"(messages={len(result.messages)}, error={result.generation_error})")

    per_case_summary = {}
    for case_id, repeats in per_case_repeats.items():
        faith_means = [
            statistics.mean([m.faithfulness_score for m in r.messages if m.faithfulness_score is not None] or [0.0])
            for r in repeats
            if r.generation_error is None
        ]
        per_case_summary[case_id] = {
            "repeats": len(repeats),
            "generation_errors": sum(1 for r in repeats if r.generation_error is not None),
            "faithfulness_across_repeats": _mean_std(faith_means),
        }

    overall_by_type = aggregate_generation_by_type(all_repeat_results)

    report = {
        "fixture": str(args.fixture),
        "repeats": args.repeats,
        "case_count": len(fixture),
        "per_case_summary": per_case_summary,
        "aggregate_by_utterance_type": overall_by_type,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    raw_path = output_path.with_name(output_path.stem + "_raw.json")
    raw_path.write_text(
        json.dumps(
            {cid: [r.model_dump() for r in repeats] for cid, repeats in per_case_repeats.items()},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"요약: {output_path}")
    print(f"원본: {raw_path}")
    return output_path


if __name__ == "__main__":
    main()
