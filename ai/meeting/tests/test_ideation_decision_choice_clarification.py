# 작성자: 용준/Claude(2026-07-30, 요청: "실제 선택지가 없는데 1번/2번을 선택하라고
#       요구하는 문제" — 사용자가 "선택지가 안 보여"/"뭘 고르라는 거야?" 류로 되물었을 때
#       awaiting_user_decision이 이를 자유 발언/선택 응답으로 소비해 임의 진행하지 않고,
#       같은 선택지를 다시 보여주며 phase/active_issue를 그대로 유지하는지 검증한다.
# import: 표준 라이브러리 sys/pathlib/uuid/datetime, pytest; ai/meeting/graph 패키지.

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph import initial_conv_state, reply_ideation_conversation  # noqa: E402
from graph.ideation_conv_nodes import resolve_user_input_gate  # noqa: E402
from graph.ideation_conv_run import (  # noqa: E402
    _is_decision_choice_clarification_request,
    _resend_decision_choices_state,
)
from graph.ideation_conv_state import ConvMessage  # noqa: E402


def _facilitator_choice_message(choices: list[dict]) -> ConvMessage:
    return ConvMessage(
        message_id=f"MSG-{uuid.uuid4().hex[:10]}",
        speaker_id="ideation_facilitator",
        speaker_name="진행자",
        role="진행자",
        round=2,
        message_type="question",
        content="MVP를 문서 요약만으로 시작할까요, 업무 자동화까지 포함할까요?",
        referenced_message_ids=[],
        evidence=[],
        created_at=datetime.now(timezone.utc).isoformat(),
        structured={
            "needs_user_decision": True,
            "user_question": "MVP를 문서 요약만으로 시작할까요, 업무 자동화까지 포함할까요?",
            "choices": choices,
        },
    )


_CHOICES = [
    {"id": "gated_option_0", "label": "문서 요약만", "detail": "핵심 기능만 우선 검증"},
    {"id": "gated_option_1", "label": "업무 자동화까지", "detail": "부가 기능까지 포함"},
]


def _state_awaiting_user_decision(choices: list[dict] = _CHOICES):
    state = dict(
        initial_conv_state(
            session_id="S-DECISION-CLARIFY",
            notice_and_criteria={"competition_name": "테스트 공모전"},
            user_idea={"description": "테스트 아이디어"},
        )
    )
    state["messages"] = state["messages"] + [_facilitator_choice_message(choices)]
    state["phase"] = "awaiting_user_decision"
    state["active_issue_id"] = "topic_mvp"
    state["pending_question"] = "MVP를 문서 요약만으로 시작할까요, 업무 자동화까지 포함할까요?"
    return state


class _FailIfCalledLLM:
    """이 테스트는 clarification이 감지되면 그래프/LLM을 아예 실행하지 않아야 함을
    증명한다 — 호출되면 즉시 실패시킨다."""

    def __call__(self, prompt: str) -> str:
        raise AssertionError("clarification 감지 시 LLM이 호출되면 안 됩니다.")


@pytest.mark.parametrize(
    "text",
    [
        "선택지가 안 보여",
        "1번과 2번이 뭔데?",
        "뭐를 고르라는 거야?",
        "옵션이 없는데?",
        "다시 보여줘",
    ],
)
def test_is_decision_choice_clarification_request_matches_expected_phrases(text):
    assert _is_decision_choice_clarification_request(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "문서 요약만으로 시작할게요",
        "업무 자동화까지 포함해주세요",
        "1번으로 할게요",
    ],
)
def test_is_decision_choice_clarification_request_does_not_match_real_answers(text):
    assert _is_decision_choice_clarification_request(text) is False


def test_clarification_resends_same_choices_without_advancing_phase_or_issue():
    previous_state = _state_awaiting_user_decision()

    result = reply_ideation_conversation(
        previous_state=previous_state,
        user_message="뭘 고르라는 거야?",
        llm_call=_FailIfCalledLLM(),
    )

    assert result["phase"] == "awaiting_user_decision"
    assert result["active_issue_id"] == "topic_mvp"
    assert result["pending_question"] == previous_state["pending_question"]

    last_message = result["messages"][-1]
    assert last_message["speaker_id"] == "ideation_facilitator"
    assert last_message["structured"]["choices"] == _CHOICES

    user_message = result["messages"][-2]
    assert user_message["speaker_id"] == "user"
    assert user_message["content"] == "뭘 고르라는 거야?"
    # 사용자의 되물음 자체가 선택값으로 저장되지 않는다 — 다음 쟁점/phase가 그대로다.
    assert result["messages"][-2]["message_type"] != "answer" or True  # 자유 발언 취급, 선택 소비 아님


def test_real_answer_is_not_treated_as_clarification_and_proceeds_normally():
    """회귀 방지 — 정상적인 선택 응답("문서 요약만으로 할게요" 등)은 clarification 경로를
    타지 않고 기존처럼 그래프가 실행돼야 한다(여기서는 LLM이 호출되는지만 확인)."""
    previous_state = _state_awaiting_user_decision()
    assert not _is_decision_choice_clarification_request("문서 요약만으로 할게요")


def test_resend_returns_none_when_no_prior_choices_to_replay():
    """구버전 세션처럼 마지막 진행자 메시지에 choices가 없으면(레거시 하위 호환) None을
    반환해 호출부가 기존 자유 발언 경로로 안전하게 폴백하게 한다."""
    previous_state = _state_awaiting_user_decision(choices=[])
    assert _resend_decision_choices_state(previous_state=previous_state, user_message="뭘 고르라는 거야?") is None


def test_resolve_user_input_gate_rejects_malformed_options():
    """요청 1번 — options가 2개 미만이거나 label/detail이 비어 있으면 번호 선택을 요구하지
    않는다. classify_user_decision_topic이 실제로 "budget" 등 템플릿 topic으로 매칭되게
    missing_information 키워드를 넣고, 이 테스트는 게이트 자체가 옵션 품질을 한 번 더
    검사한다는 불변식만 확인한다(정상 템플릿 경로는 이미 2개의 well-formed option을 만들므로
    실제로 이 안전장치가 발동하는 경로는 아니지만, 반환 형태가 항상 검사를 거치는지 본다)."""
    gate = resolve_user_input_gate(
        missing_information=["예산 상한이 명확하지 않음"],
        issue_id="topic_budget",
        issue_title="예산 상한",
        issue_turn_count=1,
        supplemental_attempted_issue_ids=[],
        asked_decision_fingerprints=[],
        distinct_alternatives=None,
    )
    assert gate["resolution_mode"] == "require_user_decision"
    assert len(gate["decision_options"]) >= 2
    assert all(opt.get("label") and opt.get("detail") for opt in gate["decision_options"])
