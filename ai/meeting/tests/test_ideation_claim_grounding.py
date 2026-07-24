# 작성자: 용준/Claude(2026-07-22, 요청: RAG 근거 실제 활용 강화)
# 목적: 아이디어 회의 전문가 발언의 claims가 실제 RAG 검색 근거(retrieved_evidence)와
#       연결·검증되는지 노드 단위로 확인한다. injected_evidence_count(프롬프트에 넣었다는
#       사실)가 아니라 linked_evidence_count(실제로 연결·검증됐다는 사실)가 성공 기준이라는
#       요청의 핵심을 검증한다.
#
# ai/meeting/graph는 ai.rag를 직접 import하지 않는다(ai/rag/tests/test_meeting_evidence_
# service.py::TestScopeBoundary가 이 경계를 강제한다) — 실제 근거 연결 구현(ai.rag.
# evidence_linking.claim_grounding.ground_claims)은 이 테스트 파일이 backend와 동일한
# 방식으로 주입한다. 이 파일은 ai/meeting/graph 밖(ai/meeting/tests)이므로 import 제약이
# 없다.

import json
import sys
from pathlib import Path

import pytest

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
REPO_ROOT = MEETING_DIR.parents[1]
sys.path.insert(0, str(MEETING_DIR))
sys.path.insert(0, str(REPO_ROOT))

from graph import start_ideation_conversation  # noqa: E402

from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl  # noqa: E402

NOTICE_AND_CRITERIA = {
    "competition_name": "WSCE2026 어워즈",
    "notice_document": "실현 가능성과 경제성을 평가한다.",
}
USER_IDEA = {"description": "실시간 교통 데이터를 활용한 통학로 안전 알림 서비스"}

_ROLE_KEYWORDS = {
    "planning_expert": ["실현 가능성", "경제성", "평가기준"],
    "dev_expert": ["데이터", "API", "구현"],
}


def _real_ground_claims(persona_id, claims, retrieved):
    return _ground_claims_impl(claims, retrieved, role_keywords=_ROLE_KEYWORDS.get(persona_id))


def _evidence_lookup(persona_id: str, query: str):
    return [
        {
            "chunk_id": "C1",
            "document_id": "DOC-1",
            "document_name": "WSCE2026 공고문",
            "section": "평가 기준",
            "page": 4,
            "text": "본 사업은 실현 가능성과 경제성을 중점적으로 평가한다.",
        }
    ]


def _discussion_response(*, stance, spoken_text, claims, issue_resolved=False, needs_user_input=False, user_question=None):
    return json.dumps(
        {
            "stance": stance,
            "spoken_text": spoken_text,
            "judgment": "판단",
            "reason": "근거",
            "suggestion": "제안",
            "interim_conclusion": "잠정 결론",
            "responding_to": None,
            "agreement": "",
            "concern": "",
            "confirmed": [],
            "unconfirmed": [],
            "referenced_message_ids": [],
            "claims": claims,
            "next_action": None,
            "active_issue_id": "feasibility",
            "active_issue_title": "실현 가능성",
            "new_information": ["새로운 검토 내용"],
            "proposal": "제안 내용",
            "changed_position": False,
            "needs_counterpart_response": not issue_resolved,
            "recommended_next_speaker": "ideation_facilitator" if issue_resolved else "dev_expert",
            "issue_resolved": issue_resolved,
            "needs_user_input": needs_user_input,
            "user_question": user_question,
        },
        ensure_ascii=False,
    )


class _GroundedScriptedLLM:
    """기획 전문가가 실제 검색 근거를 정확히 인용하는 document_fact claim을 반환한다 —
    grounding이 이를 linked_evidence_refs로 연결해야 한다.

    용준/Claude(2026-07-23, 요청: RAG 근거 실제 활용 강화 — evidence 참조 안정화):
    call_evidence_lookup(ai/meeting/graph/ideation_nodes.py)이 이제 각 근거에 순번 참조
    ("ref": "E1")를 자동으로 부여하고, 프롬프트도 chunk_id 대신 그 ref를 인용하도록
    안내한다 — 실제로 그 지시를 따르는 LLM을 흉내 내려면 evidence_refs에 chunk_id("C1")가
    아니라 ref("E1")를 넣어야 한다(_evidence_lookup이 반환하는 단일 항목이 call_evidence_
    lookup을 거치며 정확히 "E1"을 받는다)."""

    def __call__(self, prompt: str) -> str:
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "agreements": ["실현 가능성 평가기준을 확인했다"],
                    "disagreements": [],
                    "facilitator_summary": "실현 가능성 평가기준을 문서로 확인했습니다.",
                    "spoken_text": "실현 가능성 평가기준을 문서로 확인했습니다.",
                    "needs_user_decision": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )
        is_dev = "당신은 AI Review Board의 개발 전문가입니다" in prompt
        if is_dev:
            return _discussion_response(
                stance="동의",
                spoken_text="기획 관점에 동의하며, 데이터 연동 방식을 추가로 검토하겠습니다.",
                claims=[
                    {
                        "claim_id": "claim_1",
                        "text": "교통 API 연동 방식은 추가 확인이 필요하다.",
                        "claim_type": "expert_judgment",
                        "evidence_refs": [],
                    }
                ],
                issue_resolved=True,
            )
        return _discussion_response(
            stance="보완",
            spoken_text="WSCE는 실현 가능성과 경제성을 평가하므로 이 기준에 맞춰 계획을 구체화하겠습니다.",
            claims=[
                {
                    "claim_id": "claim_1",
                    "text": "WSCE는 실현 가능성과 경제성을 중점적으로 평가한다.",
                    "claim_type": "document_fact",
                    "evidence_refs": ["E1"],
                }
            ],
        )


class _UngroundedScriptedLLM:
    """기획 전문가가 존재하지 않는 chunk_id를 인용하는 document_fact claim을 반환한다 —
    재생성 후에도 실패하면 안전한 fallback 발언으로 대체돼야 한다."""

    def __init__(self):
        self._planning_calls = 0

    def __call__(self, prompt: str) -> str:
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "agreements": [],
                    "disagreements": [],
                    "facilitator_summary": "추가 확인이 필요한 상태로 정리합니다.",
                    "spoken_text": "추가 확인이 필요한 상태로 정리합니다.",
                    "needs_user_decision": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )
        is_dev = "당신은 AI Review Board의 개발 전문가입니다" in prompt
        if is_dev:
            return _discussion_response(
                stance="동의",
                spoken_text="기획 관점에 동의합니다.",
                claims=[],
                issue_resolved=True,
            )
        self._planning_calls += 1
        # 재시도해도 계속 존재하지 않는 chunk_id를 인용한다(재생성이 실제로 한 번만
        # 일어나고 무한 반복하지 않는지 확인하기 위함).
        return _discussion_response(
            stance="보완",
            spoken_text="이 사업은 반드시 6개월 안에 구축해야 한다고 공고문에 명시되어 있습니다.",
            claims=[
                {
                    "claim_id": "claim_1",
                    "text": "6개월 안에 구축해야 한다.",
                    "claim_type": "document_fact",
                    "evidence_refs": ["C-does-not-exist"],
                }
            ],
        )


def _start(llm, ground_claims):
    return start_ideation_conversation(
        session_id="CONV-GROUNDING-TEST",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        max_rounds=1,
        evidence_lookup=_evidence_lookup,
        ground_claims=ground_claims,
    )


def _planning_message(state):
    return next(m for m in state["messages"] if m["speaker_id"] == "planning_expert" and m["message_type"] != "question")


def test_document_fact_with_real_chunk_id_is_linked():
    state = _start(_GroundedScriptedLLM(), _real_ground_claims)
    message = _planning_message(state)

    assert message["evidence_status"] == "grounded"
    assert message["linked_evidence_refs"] == ["C1"]
    assert message["supported_claim_count"] == 1
    assert message["unsupported_claim_count"] == 0
    assert message["claims"][0]["claim_type"] == "document_fact"


def test_unknown_chunk_id_is_not_linked_and_message_uses_safe_fallback():
    state = _start(_UngroundedScriptedLLM(), _real_ground_claims)
    message = _planning_message(state)

    assert message["evidence_status"] == "ungrounded"
    assert message["linked_evidence_refs"] == []
    assert message["unsupported_claim_count"] == 1
    # 안전한 fallback 문구로 대체되어 "반드시 6개월 안에 구축해야 한다"는 원래의 확정
    # 표현이 그대로 노출되지 않고, "확인하기 어렵다"는 유보적 표현으로 바뀐다.
    assert "반드시" not in message["content"]
    assert "확인하기 어렵습니다" in message["content"]
    assert "추가 확인이 필요합니다" in message["content"]


def test_grounding_result_does_not_stop_meeting():
    """근거가 전혀 연결되지 않아도 회의는 중단되지 않고 phase가 failed로 떨어지지 않는다."""
    state = _start(_UngroundedScriptedLLM(), _real_ground_claims)
    assert state["phase"] != "failed"
    assert len(state["messages"]) > 1


def test_no_ground_claims_fn_falls_back_to_empty_grounding_without_crashing():
    """ground_claims를 주입하지 않으면(use_rag=False 등) 검증을 건너뛰고 evidence/claims는
    비어 있는 채로 안전하게 진행된다 — 기존 evidence(retrieved 전체)는 그대로 유지된다."""
    state = _start(_GroundedScriptedLLM(), None)
    message = _planning_message(state)

    assert message["evidence_status"] == "no_evidence_available"
    assert message["linked_evidence_refs"] == []
    # evidence(기존 필드, "프롬프트에 주입된 근거 전체")는 grounding과 무관하게 유지된다.
    assert message["evidence"]
    assert message["evidence"][0]["chunk_id"] == "C1"


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-22, 요청: 반복되는 근거 없는 의견을 사용자 질문으로 전환) —
# linked_evidence_count=0(또는 expert_judgment_only)인 턴이 같은 쟁점에서 2회 연속되면,
# max_issue_turns_reached(발언 캡 6회)까지 기다리지 않고 곧바로 사용자에게 구체적으로
# 되묻는 흐름으로 전환되는지 검증한다.
# ---------------------------------------------------------------------------


class _RepeatedExpertJudgmentOnlyLLM:
    """기획/개발 위원이 같은 쟁점("feasibility")에서 각자 문서 근거 없는 expert_judgment
    claim만 반복한다(둘 다 evidence_refs=[]) — 실제 청크와 연결된 주장이 하나도 없는 상태가
    2턴 연속(기획 1회 + 개발 1회) 이어진다."""

    def __call__(self, prompt: str) -> str:
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "agreements": [],
                    "disagreements": [],
                    "facilitator_summary": "데이터 접근성과 인프라 계획을 추가로 검토해야 한다는 판단입니다.",
                    "spoken_text": "구체적인 데이터 제공기관과 기술 구성을 확인할 수 없어 추가 정보가 필요합니다.",
                    "needs_user_decision": True,
                    "user_question": "활용하려는 공공데이터 API나 협력기관이 정해져 있나요?",
                },
                ensure_ascii=False,
            )
        is_dev = "당신은 AI Review Board의 개발 전문가입니다" in prompt
        if is_dev:
            # discussion_stage="response"(기획 발언 직후이므로)는 responding_to/agreement·
            # concern 중 하나가 채워져 있어야 구조화 검증을 통과한다 — _discussion_response
            # 헬퍼는 이 필드들을 항상 비워두므로 여기서는 JSON을 직접 만든다.
            return json.dumps(
                {
                    "stance": "보완",
                    "spoken_text": "기획 관점에 동의하며, 안정적인 서비스 운영을 위해 인프라 확장 계획도 함께 검토해야 합니다.",
                    "judgment": "판단",
                    "reason": "근거",
                    "suggestion": "제안",
                    "interim_conclusion": "잠정 결론",
                    "responding_to": "기획 위원의 데이터 접근성 판단",
                    "agreement": "데이터 접근성이 중요하다는 판단에 동의합니다.",
                    "concern": "",
                    "confirmed": [],
                    "unconfirmed": [],
                    "referenced_message_ids": [],
                    "claims": [
                        {
                            "claim_id": "claim_1",
                            "text": "인프라 확장 계획을 함께 검토해야 한다.",
                            "claim_type": "expert_judgment",
                            "evidence_refs": [],
                        }
                    ],
                    "next_action": None,
                    "active_issue_id": "feasibility",
                    "active_issue_title": "실현 가능성",
                    "new_information": ["인프라 확장 관점 추가"],
                    "proposal": "제안 내용",
                    "changed_position": False,
                    "needs_counterpart_response": False,
                    "recommended_next_speaker": "planning_expert",
                    "issue_resolved": False,
                    "needs_user_input": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )
        return _discussion_response(
            stance="보완",
            spoken_text="데이터 접근성과 인프라 구축이 중요하다고 판단합니다.",
            claims=[
                {
                    "claim_id": "claim_1",
                    "text": "데이터 접근성과 인프라 구축이 중요하다.",
                    "claim_type": "expert_judgment",
                    "evidence_refs": [],
                }
            ],
            issue_resolved=False,
        )


def test_repeated_expert_judgment_turns_close_issue_without_forcing_user_research():
    """문서 인용이 없는 expert_judgment는 정상적인 전문가 역할이다.

    문서 인용이 없다는 이유만으로 사용자를 부르지는 않는다. 다만 같은 의미의 판단이 실제
    발언에서 반복되면 6턴 상한을 모두 소모하지 않고 진행자가 잠정 정리해야 한다.
    """
    llm = _RepeatedExpertJudgmentOnlyLLM()
    state = _start(llm, _real_ground_claims)

    expert_messages = [m for m in state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
    issue_message_counts: dict[str, int] = {}
    for message in expert_messages:
        issue_id = message["structured"]["active_issue_id"]
        issue_message_counts[issue_id] = issue_message_counts.get(issue_id, 0) + 1
    assert issue_message_counts
    assert all(2 <= count < 6 for count in issue_message_counts.values())

    dev_message = expert_messages[-1]
    assert dev_message["evidence_status"] == "expert_judgment_only"
    assert dev_message["linked_evidence_refs"] == []
    assert all(m["structured"]["needs_user_input"] is False for m in expert_messages)
    assert all(m["structured"]["recommended_next_speaker"] != "user" for m in expert_messages)
    assert all(m["structured"]["next_action"] == "continue_discussion" for m in expert_messages)
    assert any(m["structured"]["repetition_detected"] for m in expert_messages)

    assert state["phase"] == "discussion_complete"
    assert state["pending_question"] is None


def test_repeated_missing_information_closes_early_without_generic_user_question():
    """document_fact 주장이 매번 같은 missing_information으로 실패(unsupported)해도,
    같은 정보가 반복되면 포괄적인 자료 제공 요청을 만들지 않고 조기에 진행자가 정리한다."""

    def _make_response(prompt: str):
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "agreements": [],
                    "disagreements": [],
                    "facilitator_summary": "정리",
                    "spoken_text": "추가 확인이 필요합니다.",
                    "needs_user_decision": True,
                    "user_question": "관련 정보를 제공해 주실 수 있나요?",
                },
                ensure_ascii=False,
            )
        return _discussion_response(
            stance="보완",
            spoken_text="예상 구축 비용은 5천만 원이라고 판단됩니다.",
            claims=[
                {
                    "claim_id": "claim_1",
                    "text": "예상 구축 비용은 5천만 원이다.",
                    "claim_type": "document_fact",
                    "evidence_refs": [],
                }
            ],
            issue_resolved=False,
        )

    state = _start(_make_response, _real_ground_claims)
    expert_messages = [m for m in state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
    assert 1 <= len(expert_messages) < 6
    assert all(m["structured"]["needs_user_input"] is False for m in expert_messages)
    assert all(m["structured"]["recommended_next_speaker"] != "user" for m in expert_messages)
    assert any(m["structured"]["repetition_detected"] for m in expert_messages)
    assert state["phase"] == "discussion_complete"
    assert state["pending_question"] is None
