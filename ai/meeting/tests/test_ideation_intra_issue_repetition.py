from __future__ import annotations

import json
import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_conv_nodes import (  # noqa: E402
    _issue_evidence_exhausted,
    _looks_like_restatement,
    _recent_issue_restatement_matches,
    _safe_discussion_fallback,
    _stop_reason_for,
    make_discussion_facilitator_node,
)


def _message(message_id: str, speaker_id: str, content: str, issue_id: str = "problem") -> dict:
    return {
        "message_id": message_id,
        "speaker_id": speaker_id,
        "content": content,
        "structured": {"active_issue_id": issue_id},
    }


def test_paraphrased_citizen_participation_restatement_is_detected():
    previous = (
        "시민 참여 확대는 중요한 문제입니다. 단순한 의견 수렴 시스템 외에도 참여를 유도할 "
        "마케팅 전략과 시민 피드백의 실제 정책 반영을 보장할 체계가 필요합니다."
    )
    current = (
        "시민 참여를 늘리려면 프로모션으로 참여를 유도하고 시민 의견을 정책에 반영해야 합니다. "
        "참여율을 높이는 홍보도 필요합니다."
    )
    assert _looks_like_restatement(current, previous) is True


def test_genuinely_different_topic_is_not_restatement():
    previous = "시민 의견을 수렴하고 정책에 반영하는 참여 플랫폼이 필요합니다."
    current = "개인정보를 암호화하고 접근 권한을 분리해 보안 위험을 줄여야 합니다."
    assert _looks_like_restatement(current, previous) is False


def test_actual_energy_dialogue_is_stopped_after_second_rephrased_turn():
    previous = (
        "에너지 소비의 비효율성과 환경 문제는 분명 중요한 이슈입니다. 이 점을 더 구체화하면, "
        "사용자가 어떤 방식으로 시스템을 통해 해결 방안을 받아들일 수 있을지 명확히 정의해볼 "
        "필요가 있습니다."
    )
    current = (
        "현재 문제인 에너지 소비의 비효율성과 환경 문제는 명확히 정의되어 있습니다. 따라서, "
        "이 시스템이 사용자에게 제공할 해결 방안을 구체화해야 합니다."
    )
    matches = _recent_issue_restatement_matches(
        [_message("M1", "planning_expert", previous)],
        issue_id="problem",
        spoken_text=current,
    )
    assert [match["message_id"] for match in matches] == ["M1"]


def test_current_turn_matching_two_recent_issue_messages_is_semantic_repetition():
    messages = [
        _message(
            "M1",
            "planning_expert",
            "시민 참여 기회를 확대하고 피드백 시스템으로 시민 의견을 수렴해야 합니다.",
        ),
        _message(
            "M2",
            "dev_expert",
            "시민 참여 확대를 위해 마케팅과 프로모션으로 의견 수렴 참여를 유도해야 합니다.",
        ),
    ]
    current = "시민 참여율을 높이려면 홍보로 참여를 유도하고 시민 피드백을 계속 수렴해야 합니다."
    matches = _recent_issue_restatement_matches(messages, issue_id="problem", spoken_text=current)
    assert {match["message_id"] for match in matches} == {"M1", "M2"}


def test_actual_web_paraphrases_match_multiple_previous_turns():
    messages = [
        _message(
            "M1",
            "planning_expert",
            "시민의 참여 기회를 확대하는 것은 중요한 문제입니다. 참여를 유도하기 위해 소통 경로를 "
            "개선하고, AI 기반 피드백 시스템을 통해 시민의 의견을 효과적으로 수렴하는 것이 핵심입니다.",
        ),
        _message(
            "M2",
            "dev_expert",
            "시민 참여 확대는 중요한 문제입니다. 단순한 의견 수렴 시스템 외에도 참여를 유도할 "
            "마케팅 전략과 시민 피드백의 실제 정책 반영을 보장할 체계가 필요합니다.",
        ),
        _message(
            "M3",
            "dev_expert",
            "시민 참여 확대를 위해 데이터 분석의 정확성과 참여 유도 마케팅 전략이 필요합니다.",
        ),
    ]
    current = (
        "시민 참여를 유도하기 위한 마케팅뿐만 아니라 정책 결정 과정에서 시민 의견이 어떻게 "
        "반영될지 설계해야 합니다. 시민이 참여할 이유와 소통 경로도 마련해야 합니다."
    )
    matches = _recent_issue_restatement_matches(messages, issue_id="problem", spoken_text=current)
    assert len(matches) >= 2


def test_same_evidence_set_after_both_experts_spoke_is_exhausted():
    messages = [
        {
            **_message(
                "M1",
                "planning_expert",
                "교통혼잡 문제와 환경 오염 증가가 확인되며 문제를 구체화해야 합니다.",
            ),
            "linked_evidence_refs": ["chk_target", "chk_criteria"],
        }
    ]

    assert _issue_evidence_exhausted(
        messages,
        issue_id="problem",
        current_speaker_id="dev_expert",
        current_linked_chunk_ids=["chk_target", "chk_criteria"],
    ) is True


def test_new_evidence_keeps_discussion_open_after_both_experts_spoke():
    messages = [
        {
            **_message("M1", "planning_expert", "문제 정의를 검토합니다."),
            "linked_evidence_refs": ["chk_target"],
        }
    ]

    assert _issue_evidence_exhausted(
        messages,
        issue_id="problem",
        current_speaker_id="dev_expert",
        current_linked_chunk_ids=["chk_target", "chk_new_measurement"],
    ) is False


def test_same_evidence_does_not_close_before_counterpart_review():
    messages = [
        {
            **_message("M1", "planning_expert", "문제 정의를 검토합니다."),
            "linked_evidence_refs": ["chk_target"],
        }
    ]

    assert _issue_evidence_exhausted(
        messages,
        issue_id="problem",
        current_speaker_id="planning_expert",
        current_linked_chunk_ids=["chk_target"],
    ) is False


def test_safe_structured_fallback_routes_to_facilitator_without_fake_new_information():
    raw = _safe_discussion_fallback(
        persona_id="planning_expert",
        state={
            "active_issue_id": "problem",
            "open_issues": [{"issue_id": "problem", "title": "문제 정의", "turns": 2}],
            "resolved_issues": [],
            "resolved_topics": [],
            "unresolved_issues": [],
        },
        discussion_stage="response",
        responding_to_message_id="M1",
        responding_to_content="앞선 발언",
    )
    assert raw["safe_fallback"] is True
    assert raw["new_information"] == []
    assert raw["proposal"] is None
    assert raw["needs_counterpart_response"] is False
    assert raw["recommended_next_speaker"] == "ideation_facilitator"


class _FacilitatorLLM:
    def __init__(self) -> None:
        self.prompt = ""

    def __call__(self, prompt: str) -> str:
        self.prompt = prompt
        return json.dumps(
            {
                "agreements": [],
                "disagreements": [],
                "facilitator_summary": "반복된 쟁점을 잠정 정리하고 다음 공식 쟁점으로 이동합니다.",
                "spoken_text": "같은 판단이 반복되어 다음 쟁점인 목표 사용자로 넘어가겠습니다.",
                "needs_user_decision": False,
                "user_question": None,
            },
            ensure_ascii=False,
        )


class _SuppressedQuestionFacilitatorLLM(_FacilitatorLLM):
    def __call__(self, prompt: str) -> str:
        self.prompt = prompt
        return json.dumps(
            {
                "agreements": [],
                "disagreements": [],
                "facilitator_summary": "추가 근거가 필요한 상태입니다.",
                "spoken_text": "관련 정보를 제공해 주실 수 있나요?",
                "needs_user_decision": True,
                "user_question": "관련 정보를 제공해 주실 수 있나요?",
            },
            ensure_ascii=False,
        )


def _repetition_state() -> dict:
    return {
        "session_id": "REPETITION-E2E",
        "phase": "expert_discussion",
        "round": 1,
        "max_rounds": 5,
        "messages": [
            {
                **_message("M-LAST", "dev_expert", "같은 판단이 반복됐습니다."),
                "structured": {
                    "active_issue_id": "problem",
                    "needs_user_input": False,
                    "repetition_detected": True,
                },
            }
        ],
        "consensus": [],
        "unresolved_issues": [],
        "notice_and_criteria": {},
        "active_issue_id": "problem",
        "open_issues": [
            {
                "issue_id": "problem",
                "title": "문제 정의",
                "status": "open",
                "planning_position": "시민 참여",
                "development_position": "시민 참여",
                "resolution": None,
                "turns": 3,
                "family": "problem",
                "closed_reason": None,
                "resolution_kind": None,
            }
        ],
        "resolved_issues": [],
        "resolved_topics": [],
        "expert_turn_count": 3,
        "llm_calls_used": 0,
    }


def test_repetition_stop_reason_closes_issue_and_rotates_before_turn_cap():
    state = _repetition_state()
    assert _stop_reason_for(state) == "semantic_repetition_detected"

    llm = _FacilitatorLLM()
    update = make_discussion_facilitator_node(llm)(state)

    assert update["active_issue_id"] == "topic_target_user"
    assert any(issue["issue_id"] == "topic_target_user" for issue in update["open_issues"])
    parked = next(issue for issue in update["resolved_issues"] if issue["issue_id"] == "problem")
    assert parked["closed_reason"] == "semantic_repetition_detected"
    assert parked["resolution_kind"] == "parked_expert_judgment"
    assert "목표 사용자" in llm.prompt


def test_max_round_without_actionable_question_continues_to_remaining_issue():
    state = _repetition_state()
    state["round"] = 1
    state["max_rounds"] = 1

    update = make_discussion_facilitator_node(_SuppressedQuestionFacilitatorLLM())(state)

    assert update["phase"] == "expert_discussion"
    assert update["next_route"] == "continue_round"
    assert update["pending_question"] is None
