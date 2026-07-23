# 작성자: 용준/Claude(2026-07-23, 요청: "사용자 정보 수집형 회의"에서 "근거 기반 자율
#         토론형 회의"로 개편)
# 목적: resolve_user_input_gate/classify_user_decision_topic 등 결정론적 사용자 질문
#       게이트가 (1) 문서 근거가 있으면 사용자에게 묻지 않고, (2) 일반 기술 질문은 보완
#       검색 1회 후에도 안 되면 전문가 판단으로 자율 진행하고, (3) 예산처럼 진짜 사용자
#       결정이 필요한 주제만 선택지+기본값 형식으로 묻고, (4) 같은 결정 질문을 반복하지
#       않는지 검증한다.
# import: 표준 라이브러리 json/sys/pathlib, pytest; ai/meeting/graph 패키지,
#         ai.rag.evidence_linking.claim_grounding(실제 grounding 판정 재사용).

from __future__ import annotations

import json
import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
REPO_ROOT = MEETING_DIR.parents[1]
sys.path.insert(0, str(MEETING_DIR))
sys.path.insert(0, str(REPO_ROOT))

from graph import start_ideation_conversation  # noqa: E402
from graph.ideation_conv_nodes import (  # noqa: E402
    classify_user_decision_topic,
    resolve_user_input_gate,
)

from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl  # noqa: E402

NOTICE_AND_CRITERIA = {
    "competition_name": "IT 공공서비스 공모전",
    "notice_document": "실현가능성, 공공성을 평가한다.",
}
USER_IDEA = {
    "description": "공공기관의 정책·지원사업 문서를 RAG로 검색하고, 사용자 상황에 맞는 지원사업을 추천하는 AI 서비스를 만들고 싶습니다."
}


def _persona(prompt: str) -> str:
    if "당신은 AI Review Board의 기획 전문가입니다" in prompt:
        return "planning_expert"
    if "당신은 AI Review Board의 개발 전문가입니다" in prompt:
        return "dev_expert"
    return "ideation_facilitator"


def _ground_claims_fn(persona_id, claims, retrieved):
    return _ground_claims_impl(claims, retrieved)


def _facilitator_payload() -> dict:
    return {
        "agreements": [],
        "disagreements": [],
        "facilitator_summary": "쟁점을 정리했습니다.",
        "spoken_text": "쟁점을 정리했습니다.",
        "needs_user_decision": False,
        "user_question": None,
    }


def _canvas_payload() -> dict:
    return {
        "problem": "문제 정의",
        "target_user": "목표 사용자",
        "core_value": "핵심 가치",
        "solution": "해결 방안",
        "differentiation": "차별점",
        "contest_fit": "공모전 적합성",
        "feasibility": "medium",
        "risks": [],
    }


def _claim_payload(
    speaker: str,
    *,
    claim_text: str,
    claim_evidence_refs: list[str],
    issue_resolved: bool,
    recommended_next_speaker: str,
    is_first_message: bool,
    needs_counterpart_response: bool = True,
    needs_user_input: bool = False,
    user_question: str | None = None,
    claim_type: str = "document_fact",
) -> dict:
    responding_to = None if is_first_message else "상대 발언"
    return {
        "stance": "보완",
        "spoken_text": f"[{speaker}] {claim_text}",
        "judgment": "판단",
        "reason": "근거",
        "suggestion": "제안",
        "interim_conclusion": "임시 결론",
        "responding_to": responding_to,
        "agreement": "" if is_first_message else "일부 동의합니다",
        "concern": "우려",
        "confirmed": [],
        "unconfirmed": [],
        "referenced_message_ids": [],
        "evidence": [],
        "next_action": None,
        "active_issue_id": "hardware_spec",
        "active_issue_title": "하드웨어 사양",
        "new_information": [claim_text],
        "proposal": "제안",
        "changed_position": False,
        "needs_counterpart_response": needs_counterpart_response,
        "recommended_next_speaker": recommended_next_speaker,
        "issue_resolved": issue_resolved,
        "needs_user_input": needs_user_input,
        "user_question": user_question,
        "claims": [
            {
                "claim_id": "c1",
                "text": claim_text,
                "claim_type": claim_type,
                "evidence_refs": claim_evidence_refs,
            }
        ],
    }


# ---------------------------------------------------------------------------
# 1. 순수 게이트 함수 단위 테스트(LLM 호출 없음)
# ---------------------------------------------------------------------------


def test_classify_user_decision_topic_detects_budget_keyword():
    topic = classify_user_decision_topic(
        missing_information=["예산 상한이 확인되지 않았습니다"], issue_title="비용 계획", near_issue_cap=False
    )
    assert topic == "budget"


def test_classify_user_decision_topic_ignores_generic_technical_gap():
    """일반적인 기술 접근 방법·센서 종류 같은 주제는 사용자 질문 사유가 아니다."""
    topic = classify_user_decision_topic(
        missing_information=["센서 설치 방식이 구체적으로 문서에 없습니다"], issue_title="센서 연동", near_issue_cap=False
    )
    assert topic is None


def test_resolve_user_input_gate_tries_supplemental_retrieval_before_expert_judgment():
    gate = resolve_user_input_gate(
        missing_information=["센서 설치 방식이 구체적으로 문서에 없습니다"],
        issue_id="hardware_spec",
        issue_title="하드웨어 사양",
        issue_turn_count=1,
        supplemental_attempted_issue_ids=[],
        asked_decision_fingerprints=[],
    )
    assert gate["resolution_mode"] == "supplemental_retrieval"


def test_resolve_user_input_gate_falls_back_to_expert_judgment_after_supplemental_attempted():
    gate = resolve_user_input_gate(
        missing_information=["센서 설치 방식이 구체적으로 문서에 없습니다"],
        issue_id="hardware_spec",
        issue_title="하드웨어 사양",
        issue_turn_count=2,
        supplemental_attempted_issue_ids=["hardware_spec"],
        asked_decision_fingerprints=[],
    )
    assert gate["resolution_mode"] == "continue_with_expert_judgment"
    assert gate["blocking_reason_code"] == "expert_judgment_fallback"


def test_resolve_user_input_gate_requires_user_decision_for_budget():
    gate = resolve_user_input_gate(
        missing_information=["예산 상한이 확인되지 않았습니다"],
        issue_id="budget_cap",
        issue_title="예산 계획",
        issue_turn_count=1,
        supplemental_attempted_issue_ids=[],
        asked_decision_fingerprints=[],
    )
    assert gate["resolution_mode"] == "require_user_decision"
    assert gate["decision_topic"] == "budget"
    # 좋은 예: 왜 필요한지 + 2~3개 선택지(장단점 포함) + 기본값.
    assert "선택해 주세요" in gate["decision_question"]
    assert "기본값" in gate["decision_question"]
    assert len(gate["decision_options"]) >= 2
    assert gate["fingerprint"]


def test_resolve_user_input_gate_suppresses_duplicate_decision_question():
    first = resolve_user_input_gate(
        missing_information=["예산 상한이 확인되지 않았습니다"],
        issue_id="budget_cap",
        issue_title="예산 계획",
        issue_turn_count=1,
        supplemental_attempted_issue_ids=[],
        asked_decision_fingerprints=[],
    )
    second = resolve_user_input_gate(
        missing_information=["예산 상한이 확인되지 않았습니다"],
        issue_id="budget_cap",
        issue_title="예산 계획",
        issue_turn_count=2,
        supplemental_attempted_issue_ids=[],
        asked_decision_fingerprints=[first["fingerprint"]],
    )
    assert second["resolution_mode"] == "continue_with_expert_judgment"
    assert second["blocking_reason_code"] == "duplicate_question_suppressed"


# ---------------------------------------------------------------------------
# 2. 통합 테스트 — 실제 discussion 노드를 통해 전체 흐름 검증
# ---------------------------------------------------------------------------


class _GenericTechGapLLM:
    """일반적인 기술 구현 방법이 문서에 없는 경우 — 사용자에게 조사를 요청하지 않고
    개발 위원이 기술 대안을 제시해야 한다(요청 4번 테스트)."""

    def __call__(self, prompt: str) -> str:
        if "[의견 규칙]" in prompt:
            speaker = _persona(prompt)
            if speaker == "planning_expert":
                return json.dumps(
                    _claim_payload(
                        speaker,
                        claim_text="센서 설치 방식이 구체적으로 문서에 없습니다",
                        claim_evidence_refs=["E1"],
                        issue_resolved=False,
                        recommended_next_speaker="dev_expert",
                        is_first_message=True,
                    ),
                    ensure_ascii=False,
                )
            return json.dumps(
                _claim_payload(
                    speaker,
                    claim_text="저비용 IoT 센서와 공공 데이터를 결합하는 것이 합리적입니다",
                    claim_evidence_refs=[],
                    issue_resolved=True,
                    recommended_next_speaker="ideation_facilitator",
                    needs_counterpart_response=False,
                    is_first_message=False,
                    claim_type="expert_judgment",
                ),
                ensure_ascii=False,
            )
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(_facilitator_payload(), ensure_ascii=False)
        if "[캔버스 갱신 규칙]" in prompt:
            return json.dumps(_canvas_payload(), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")


def _irrelevant_evidence_lookup(persona_id: str, query: str) -> list[dict]:
    """검색 자체는 이뤄졌지만(retrieved_evidence 비어있지 않음) 주장과 무관한 청크만
    돌려준다 — evidence_status="ungrounded"(문서 근거 연결 실패)를 재현하기 위함이다.
    "검색 자체를 안 했다"(no_evidence_available)와는 다른 상황이다."""
    return [
        {
            "chunk_id": "chk_irrelevant_0001",
            "document_id": "DOC-CRITERIA-1",
            "document_name": "공고문",
            "section": "일반 사항",
            "document_role": "criteria",
            "text": "본 공모전은 매년 3월에 개최된다.",
        }
    ]


def test_generic_technical_gap_does_not_ask_user_and_lets_expert_propose():
    state = start_ideation_conversation(
        session_id="AUTONOMOUS-1",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=_GenericTechGapLLM(),
        max_rounds=1,
        evidence_lookup=_irrelevant_evidence_lookup,
        ground_claims=_ground_claims_fn,
    )
    planning_message = next(m for m in state["messages"] if m["speaker_id"] == "planning_expert")
    assert planning_message["structured"]["needs_user_input"] is False
    assert planning_message["structured"]["resolution_mode"] == "continue_with_expert_judgment"
    assert planning_message["structured"]["supplemental_retrieval_attempted"] is True
    # 사용자에게 조사를 요청하는 대신 개발 위원이 곧바로 기술 대안을 제시해야 한다.
    dev_message = next(m for m in state["messages"] if m["speaker_id"] == "dev_expert")
    assert "IoT" in dev_message["content"] or "센서" in dev_message["content"]
    if state["phase"] == "awaiting_user_decision":
        assert state.get("pending_question")
        assert "선택해 주세요" in state["pending_question"]


class _RealBudgetDecisionLLM:
    """예산 상한처럼 사용자만 결정할 수 있는 사업 의사결정 — 회의를 멈추고 구조화된 질문을
    던져야 한다(요청 3번 테스트)."""

    def __call__(self, prompt: str) -> str:
        if "[의견 규칙]" in prompt:
            speaker = _persona(prompt)
            return json.dumps(
                _claim_payload(
                    speaker,
                    claim_text="예산 상한이 확인되지 않아 구현 범위를 확정할 수 없습니다",
                    claim_evidence_refs=["E1"],
                    issue_resolved=False,
                    recommended_next_speaker="dev_expert" if speaker == "planning_expert" else "planning_expert",
                    is_first_message=speaker == "planning_expert",
                ),
                ensure_ascii=False,
            )
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(_facilitator_payload(), ensure_ascii=False)
        if "[캔버스 갱신 규칙]" in prompt:
            return json.dumps(_canvas_payload(), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")


def test_real_budget_decision_asks_structured_question_and_awaits_user():
    state = start_ideation_conversation(
        session_id="AUTONOMOUS-2",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=_RealBudgetDecisionLLM(),
        max_rounds=3,
        evidence_lookup=_irrelevant_evidence_lookup,
        ground_claims=_ground_claims_fn,
    )
    planning_message = next(m for m in state["messages"] if m["speaker_id"] == "planning_expert")
    assert planning_message["structured"]["needs_user_input"] is True
    assert planning_message["structured"]["decision_topic"] == "budget"
    assert planning_message["structured"]["recommended_next_speaker"] == "user"
    assert len(planning_message["structured"]["decision_options"]) >= 2
    assert state["phase"] == "awaiting_user_decision"
    assert state.get("pending_question")
    assert "기본값" in state["pending_question"]


class _SupplementalRetrievalLLM:
    """1차 검색으로는 근거를 찾지 못했지만, 보완 검색으로 실제 target 문서를 찾으면 사용자
    에게 묻지 않고 grounding이 개선돼야 한다(요청 6번 테스트)."""

    def __call__(self, prompt: str) -> str:
        if "[의견 규칙]" in prompt:
            speaker = _persona(prompt)
            if speaker == "planning_expert":
                return json.dumps(
                    _claim_payload(
                        speaker,
                        claim_text="리모컨 배터리는 자동으로 교체된다",
                        claim_evidence_refs=["E1"],
                        issue_resolved=False,
                        recommended_next_speaker="dev_expert",
                        is_first_message=True,
                    ),
                    ensure_ascii=False,
                )
            return json.dumps(
                _claim_payload(
                    speaker,
                    claim_text="검토를 마쳤습니다",
                    claim_evidence_refs=[],
                    issue_resolved=True,
                    recommended_next_speaker="ideation_facilitator",
                    needs_counterpart_response=False,
                    is_first_message=False,
                    claim_type="expert_judgment",
                ),
                ensure_ascii=False,
            )
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(_facilitator_payload(), ensure_ascii=False)
        if "[캔버스 갱신 규칙]" in prompt:
            return json.dumps(_canvas_payload(), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")


def _make_supplemental_evidence_lookup():
    calls: list[str] = []

    def lookup(persona_id: str, query: str) -> list[dict]:
        calls.append(query)
        if "리모컨 배터리" in query:
            return [
                {
                    "chunk_id": "chk_supplemental_0001",
                    "document_id": "DOC-TARGET-1",
                    "document_name": "제안서 초안",
                    "section": "하드웨어 사양",
                    "document_role": "criteria",
                    "text": "리모컨 배터리는 자동으로 교체된다.",
                }
            ]
        # 최초 검색(주제 질의)은 검색 자체는 이뤄졌지만 무관한 청크만 돌려준다 —
        # evidence_status="ungrounded"를 재현해야 보완 검색(supplemental_retrieval)
        # 분기를 탄다("no_evidence_available"이면 애초에 이 분기를 타지 않는다).
        return [
            {
                "chunk_id": "chk_irrelevant_0001",
                "document_id": "DOC-CRITERIA-1",
                "document_name": "공고문",
                "section": "일반 사항",
                "document_role": "criteria",
                "text": "본 공모전은 매년 3월에 개최된다.",
            }
        ]

    lookup.calls = calls
    return lookup


def test_supplemental_retrieval_grounds_new_evidence_without_asking_user():
    evidence_lookup = _make_supplemental_evidence_lookup()
    llm = _SupplementalRetrievalLLM()
    state = start_ideation_conversation(
        session_id="AUTONOMOUS-3",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        max_rounds=1,
        evidence_lookup=evidence_lookup,
        ground_claims=_ground_claims_fn,
    )
    planning_message = next(m for m in state["messages"] if m["speaker_id"] == "planning_expert")
    structured = planning_message["structured"]
    assert structured["needs_user_input"] is False
    assert structured["supplemental_retrieval_attempted"] is True
    assert structured["supplemental_evidence_count"] == 1
    assert structured["evidence_status"] != "ungrounded"
    # 쟁점당 보완 검색은 최대 1회다 — 같은 쟁점("hardware_spec")으로 다시 시도하지 않는다.
    assert state["supplemental_retrieval_issue_ids"].count("hardware_spec") == 1
    # 보완 검색은 LLM을 다시 호출하지 않는다(비용 제한 — 검색만 1회 추가).
    # planning 2(최초 1 + 기존 grounding 하드 실패 재시도 1, 보완 검색과는 무관한
    # 별개의 기존 메커니즘) + dev 1 + facilitator 1 + canvas_update 1 = 5.
    # 보완 검색 자체는 검색만 1회 추가할 뿐 LLM을 다시 부르지 않으므로 이 값에
    # 포함되지 않는다(요청: 비용 제한).
    assert state["llm_calls_used"] == 5
