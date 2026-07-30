# 작성자: 용준/Claude(2026-07-22)
# 목적: 실제 페르소나 회의(ai.meeting.graph.start_ideation_conversation)를 케이스별로
#       짧게(max_rounds=1) 실행해 진짜 planning_expert/dev_expert ConvMessage를 만들고,
#       그 발언이 생성될 때 실제로 evidence_lookup이 반환한 retrieved_context를 캡처해
#       judge.py로 Faithfulness/Hallucination/Persona Evidence Fit을 판정한다.
#
#       회의 그래프(ai/meeting) 자체는 전혀 수정하지 않는다 — 여기서는 그 함수를 그대로
#       호출만 한다. evidence_lookup은 make_ideation_evidence_lookup()이 만든 실제 콜백을
#       감싸기만 하고(호출 인자/반환값을 가로채 기록), 검색 로직 자체를 바꾸지 않는다.
from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

from ai.rag.evaluation.rag_quality.cache import JudgeCache
from ai.rag.evaluation.rag_quality.judge import judge_expert_judgment, judge_faithfulness, judge_persona_fit
from ai.rag.evaluation.rag_quality.schemas import (
    GenerationAggregate,
    GenerationCaseResult,
    MessageEvalResult,
    RagEvalCase,
)
from ai.rag.evaluation.rag_quality.utterance_type import bucket_claims, classify_message
from ai.rag.orchestration.ideation_evidence_service import make_ideation_evidence_lookup
from ai.rag.role_retrieval.service import RoleAwareRetrievalService

from ._meeting_path import ensure_meeting_on_path

ensure_meeting_on_path()
from graph import start_ideation_conversation  # noqa: E402
from graph.llm import LLMCall  # noqa: E402

_MAX_ROUNDS = 1  # 케이스당 비용을 짧게 유지한다(요청: --limit과 함께 비용 제어) — 진짜
# 발언은 1라운드(안건 제시 -> 기획 최초 의견 -> 개발 검토 -> [선택적 수정] -> 진행자 정리)만
# 만들어도 Faithfulness/Persona Fit 판정에는 충분하다.


def _make_capturing_evidence_lookup(
    project_id: str,
    role_retrieval_service: RoleAwareRetrievalService,
    top_k: int,
    *,
    selected_candidate_document_id: Optional[str] = None,
) -> tuple[Callable[[str, str], list[dict]], dict[str, list[list[dict]]]]:
    """make_ideation_evidence_lookup()이 만든 실제 콜백을 그대로 호출하되, persona_id별로
    호출 순서대로 retrieved_context를 기록해 둔다 — 이후 같은 persona_id의 메시지가
    state["messages"]에 나타나는 순서와 1:1로 대응한다(노드가 evidence_lookup을 부른 직후
    바로 그 결과로 메시지를 만들기 때문).

    용준/Claude(2026-07-29, 요청: B05 회귀) — 이 평가 하네스는 candidate_selection 노드를
    거치지 않는 단발성 start_ideation_conversation 호출이라, 실제 세션이라면
    ai/meeting의 _runtime_scope_for()가 state["selected_idea_document_id"]에서 읽어
    evidence_lookup에 실어 보냈을 정보(selected_candidate_document_id)가 여기서는
    원래 전혀 전달되지 않았다 — B05가 "최종 확정한 방향"을 검색 근거 없이 답하다 후보
    목록을 최종안으로 오인한 원인이 이것이었다(실측: retrieved_contexts에 선택된 아이디어
    문서가 아예 없었음). 이 파라미터는 실제 프로덕션 state가 아는 것과 같은 종류의 정보
    (문서 id 하나)만 전달한다 — reference_answer 텍스트 자체는 넘기지 않는다."""
    real_lookup = make_ideation_evidence_lookup(
        project_id,
        role_retrieval_service,
        top_k=top_k,
        selected_candidate_document_id=selected_candidate_document_id,
    )
    captured: dict[str, list[list[dict]]] = {"planning_expert": [], "dev_expert": []}

    def wrapped(persona_id: str, topic_query: str) -> list[dict]:
        result = real_lookup(persona_id, topic_query)
        captured.setdefault(persona_id, []).append(result)
        return result

    return wrapped, captured


def _selected_candidate_document_id(case: RagEvalCase) -> Optional[str]:
    """용준/Claude(2026-07-29, 요청: 평가 정합성 — 정답 누출 분리). case.session_state(검색
    입력 전용 필드)에서만 읽는다 — gold_document_ids/expected_source_chunk_ids/
    reference_answer(채점 전용 필드)는 절대 검색 입력으로 쓰지 않는다. session_state가 없는
    케이스는 실제 세션이 아직 후보를 확정하지 않은 것과 동일하게 처리한다(None -> 검색은
    순수 semantic 검색으로 fallback, 값을 지어내지 않는다)."""
    if case.session_state is None:
        return None
    return case.session_state.selected_idea_document_id


def _initial_state_overrides(case: RagEvalCase) -> Optional[dict]:
    """용준/Claude(2026-07-29, 요청: RAG 품질 개선 3단계 — 정확한 상태값은 코드가 결정적으로
    출력). ai/meeting의 _deterministic_final_direction_prefix()는 state["idea_locked"]와
    state["selected_idea"]["title"]을 직접 읽는다 — 이 오프라인 평가 하네스는
    candidate_selection 노드를 거치지 않으므로, 실제 세션이라면 그 노드가 채웠을 값을
    case.session_state(검색/생성 입력 전용, 채점 필드 아님)에서 그대로 재현해
    start_ideation_conversation(initial_state_overrides=...)에 넘긴다."""
    if case.session_state is None:
        return None
    overrides: dict = {"idea_locked": case.session_state.idea_locked}
    if case.session_state.selected_idea_title:
        overrides["selected_idea"] = {"title": case.session_state.selected_idea_title}
    return overrides


def _check_forbidden_claims(content: str, forbidden_claims: list[str]) -> list[str]:
    """forbidden_claims는 LLM 판정이 아니라 결정적 부분 문자열 매칭으로 확인한다 —
    "이런 표현이 나오면 안 된다"는 검증은 판정자 재량이 아니라 명확한 금칙어 검사이기
    때문이다."""
    return [claim for claim in forbidden_claims if claim and claim in content]


def run_generation_eval(
    cases: list[RagEvalCase],
    *,
    llm_call: LLMCall,
    judge_llm_call: LLMCall,
    judge_model: str,
    role_retrieval_service: RoleAwareRetrievalService,
    top_k: int = 5,
    cache: Optional[JudgeCache] = None,
    evaluate_expert_judgment: bool = False,
) -> list[GenerationCaseResult]:
    """evaluate_expert_judgment=True면 메시지별 expert_judgment claim에 대해
    judge_expert_judgment(G-Eval)를 추가로 호출한다 — 용준/Claude(2026-07-29, 요청: 생성 품질
    개선 §6). 기존 88건 전체 평가(cli.py) 호출부는 이 인자를 넘기지 않으므로 기본 동작(비용,
    출력 스키마)은 변하지 않는다 — 새 필드는 모두 선택 필드다."""
    results: list[GenerationCaseResult] = []

    for case in cases:
        wrapped_lookup, captured = _make_capturing_evidence_lookup(
            case.filters.project_id,
            role_retrieval_service,
            top_k,
            selected_candidate_document_id=_selected_candidate_document_id(case),
        )
        started = time.perf_counter()
        try:
            state = start_ideation_conversation(
                session_id=f"RAG-EVAL-{uuid.uuid4().hex[:8]}",
                notice_and_criteria={
                    "competition_name": "RAG 품질 평가용 세션",
                    "notice_document": "이 세션은 RAG 품질 오프라인 평가 도구가 생성한 것입니다.",
                },
                user_idea={"description": case.query},
                llm_call=llm_call,
                max_rounds=_MAX_ROUNDS,
                evidence_lookup=wrapped_lookup,
                initial_state_overrides=_initial_state_overrides(case),
            )
        except Exception as exc:  # noqa: BLE001 - 평가 도구는 한 케이스 실패로 전체를 죽이지 않는다
            results.append(
                GenerationCaseResult(
                    case_id=case.id,
                    query=case.query,
                    generation_error=str(exc),
                    generation_time_ms=(time.perf_counter() - started) * 1000,
                )
            )
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000

        persona_cursor = {"planning_expert": 0, "dev_expert": 0}
        message_results: list[MessageEvalResult] = []
        for message in state.get("messages", []):
            persona_id = message.get("speaker_id")
            if persona_id not in ("planning_expert", "dev_expert"):
                continue
            idx = persona_cursor[persona_id]
            persona_cursor[persona_id] += 1
            retrieved_context = captured.get(persona_id, [])
            context_for_message = retrieved_context[idx] if idx < len(retrieved_context) else []

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

            # 용준/Claude(2026-07-29, 요청: 생성 품질 개선 §2/§6) — 새 필드로 저장하지 않고
            # ConvMessage에 이미 있는 claim_type 기반 grounding 결과(ai/rag/evidence_linking/
            # claim_grounding.py::ground_claims 출력)를 그대로 읽어 유형별 통계만 덧붙인다.
            utterance_type = classify_message(message)
            source_claim_buckets = bucket_claims(message.get("claims") or [])
            expert_judgment_results: list = []
            if evaluate_expert_judgment and source_claim_buckets["expert_judgment"]:
                ej_text = " ".join(
                    c.get("text", "") for c in source_claim_buckets["expert_judgment"] if c.get("text")
                )
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
                    expert_judgment_results.append(ej_result)
                elif ej_error:
                    faith_error = faith_error or ej_error

            message_results.append(
                MessageEvalResult(
                    case_id=case.id,
                    message_id=message.get("message_id", ""),
                    persona_id=persona_id,  # type: ignore[arg-type]
                    round=message.get("round", 1),
                    content_preview=content[:200],
                    claims=claims,
                    faithfulness_score=((supported + 0.5 * partial) / denom) if denom else None,
                    hallucination_rate=((unsupported + contradicted) / denom) if denom else None,
                    unsupported_count=unsupported,
                    contradicted_count=contradicted,
                    non_factual_count=len(claims) - len(scorable),
                    forbidden_claim_hits=_check_forbidden_claims(content, case.forbidden_claims),
                    persona_fit=persona_fit,
                    judge_error=faith_error or fit_error,
                    utterance_type=utterance_type,
                    grounded_claim_count=message.get("grounded_claim_count"),
                    expert_judgment_claim_count=message.get("expert_judgment_count"),
                    grounding_unsupported_claim_count=message.get("unsupported_claim_count"),
                    linked_evidence_refs=list(message.get("linked_evidence_refs") or []),
                    expert_judgments=expert_judgment_results,
                )
            )

        results.append(
            GenerationCaseResult(
                case_id=case.id,
                query=case.query,
                messages=message_results,
                generation_time_ms=elapsed_ms,
            )
        )

    return results


def aggregate_generation(results: list[GenerationCaseResult]) -> GenerationAggregate:
    all_messages = [m for r in results for m in r.messages]
    scored_faith = [m.faithfulness_score for m in all_messages if m.faithfulness_score is not None]
    scored_halluc = [m.hallucination_rate for m in all_messages if m.hallucination_rate is not None]
    scored_fit = [m.persona_fit.normalized_score for m in all_messages if m.persona_fit is not None]
    failed_generations = [r for r in results if r.generation_error is not None]
    failed_judges = [m for m in all_messages if m.judge_error is not None]

    def _mean(values: list[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    fit_macro = _mean(scored_fit)
    severe = [
        f"{m.case_id}/{m.message_id}: {c.claim} ({c.verdict})"
        for m in all_messages
        for c in m.claims
        if c.verdict == "contradicted"
    ][:10]

    return GenerationAggregate(
        case_count=len(results),
        message_count=len(all_messages),
        faithfulness_macro=_mean(scored_faith),
        hallucination_rate_macro=_mean(scored_halluc),
        persona_evidence_fit_macro=fit_macro,
        persona_evidence_fit_percent=(fit_macro * 100 if fit_macro is not None else None),
        generation_failure_rate=(len(failed_generations) / len(results)) if results else None,
        eval_failure_rate=(len(failed_judges) / len(all_messages)) if all_messages else None,
        avg_generation_time_ms=_mean([r.generation_time_ms for r in results]) or 0.0,
        severe_hallucination_examples=severe,
    )


def aggregate_generation_by_type(results: list[GenerationCaseResult]) -> dict:
    """용준/Claude(2026-07-29, 요청: 생성 품질 개선 §6). aggregate_generation()의 기존 전체
    평균은 그대로 두고, utterance_type(final_answer/discussion — facilitator는 이 하네스가
    애초에 planning_expert/dev_expert 메시지만 다루므로 구조적으로 이미 제외돼 있다) 별로
    별도 집계를 만든다. 기존 결과 리스트를 변형하지 않고 읽기만 한다."""

    def _mean(values: list[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    all_messages = [m for r in results for m in r.messages]

    by_type: dict[str, list] = {"final_answer": [], "discussion": [], "facilitator_message": []}
    for m in all_messages:
        by_type.setdefault(m.utterance_type or "discussion", []).append(m)

    def _bucket_summary(messages: list) -> dict:
        faith = [m.faithfulness_score for m in messages if m.faithfulness_score is not None]
        grounded_counts = [m.grounded_claim_count for m in messages if m.grounded_claim_count is not None]
        linked_counts = [len(m.linked_evidence_refs) for m in messages]
        unsupported_counts = [
            m.grounding_unsupported_claim_count for m in messages if m.grounding_unsupported_claim_count is not None
        ]
        evidence_link_rate = (
            sum(linked_counts) / sum(grounded_counts) if grounded_counts and sum(grounded_counts) > 0 else None
        )
        return {
            "message_count": len(messages),
            "faithfulness_mean": _mean(faith),
            "grounded_claim_total": sum(grounded_counts) if grounded_counts else 0,
            "linked_evidence_ref_total": sum(linked_counts),
            "evidence_link_rate": evidence_link_rate,
            "unsupported_claim_total": sum(unsupported_counts) if unsupported_counts else 0,
        }

    grounded_claim_bucket_messages = [m for m in all_messages if (m.grounded_claim_count or 0) > 0]
    expert_judgments = [ej for m in all_messages for ej in m.expert_judgments]

    return {
        "grounded_claim": _bucket_summary(grounded_claim_bucket_messages),
        "final_answer": _bucket_summary(by_type["final_answer"]),
        "discussion": _bucket_summary(by_type["discussion"]),
        "facilitator_message": _bucket_summary(by_type["facilitator_message"]),
        "expert_judgment": {
            "count": len(expert_judgments),
            "role_fit_mean": _mean([ej.role_fit for ej in expert_judgments]),
            "logic_mean": _mean([ej.logic for ej in expert_judgments]),
            "actionability_mean": _mean([ej.actionability for ej in expert_judgments]),
            "normalized_score_mean": _mean([ej.normalized_score for ej in expert_judgments]),
            "conflicts_with_grounded_fact_count": sum(1 for ej in expert_judgments if ej.conflicts_with_grounded_fact),
        },
    }
