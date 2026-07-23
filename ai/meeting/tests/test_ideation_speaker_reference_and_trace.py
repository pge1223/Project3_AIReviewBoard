from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

MEETING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_conv_nodes import (  # noqa: E402
    _validate_discussion_response,
    validate_spoken_text_speaker_reference,
)
from graph.ideation_trace import (  # noqa: E402
    bind_trace_context,
    configure_ideation_trace,
    is_late_request_event,
    reset_trace_context,
    sanitize_preview,
    stream_delta_trace_enabled,
    trace_event,
)
from prompts import build_ideation_conv_discussion_prompt  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_trace_config():
    configure_ideation_trace(enabled=None, content_max_chars=None, stream_deltas=None)
    yield
    configure_ideation_trace(enabled=None, content_max_chars=None, stream_deltas=None)


@pytest.mark.parametrize(
    ("speaker", "target", "spoken_text"),
    [
        ("planning_expert", "dev_expert", "기획 전문가가 언급한 사용자 피드백을 반영하겠습니다."),
        ("dev_expert", "planning_expert", "개발 위원이 제안한 기술 구성을 다시 검토하겠습니다."),
    ],
)
def test_current_speaker_cannot_refer_to_own_role_as_counterpart(speaker, target, spoken_text):
    assert validate_spoken_text_speaker_reference(speaker, target, spoken_text) == "spoken_text_self_role_reference"


def test_current_speaker_can_refer_to_actual_counterpart_naturally():
    assert validate_spoken_text_speaker_reference(
        "planning_expert",
        "dev_expert",
        "말씀하신 데이터 갱신 비용을 고려하면 MVP에서는 주간 동기화로 범위를 줄이는 게 좋겠습니다.",
    ) is None


def test_role_reference_must_match_actual_responding_target():
    assert validate_spoken_text_speaker_reference(
        "planning_expert",
        "user",
        "개발 위원이 제안한 범위를 적용하겠습니다.",
    ) == "spoken_text_role_reference_target_mismatch"


def test_response_that_only_repeats_previous_message_is_rejected():
    raw = {
        "spoken_text": "피드백 기능이 필요하다는 의견에 동의하며 피드백 기능이 필요합니다.",
        "judgment": "동의",
        "reason": "피드백 기능이 필요합니다.",
        "suggestion": "피드백 기능을 둡니다.",
        "interim_conclusion": "피드백 기능이 필요합니다.",
        "responding_to": "피드백 기능이 필요하다는 주장",
        "agreement": "피드백 기능 필요성",
        "concern": "",
        "active_issue_id": "feedback",
        "new_information": ["피드백 기능이 필요합니다."],
        "needs_user_input": False,
    }
    assert _validate_discussion_response(
        raw,
        "response",
        current_speaker_id="planning_expert",
        responding_to_speaker_id="dev_expert",
        responding_to_content="피드백 기능이 필요합니다.",
    ) in {"spoken_text_restates_responding_message", "new_information_only_repeats_responding_message"}


def _valid_user_response(spoken_text: str) -> dict:
    return {
        "spoken_text": spoken_text,
        "judgment": "보완 필요",
        "reason": "운영 단계의 관리 계획이 필요합니다.",
        "suggestion": "정기 점검 주기와 담당자를 정합니다.",
        "interim_conclusion": "유지보수 계획을 MVP 범위에 포함합니다.",
        "responding_to": None,
        "agreement": "",
        "concern": "",
        "active_issue_id": "problem",
        "new_information": ["정기 점검 주기와 장애 대응 담당자를 정의해야 합니다."],
        "needs_user_input": False,
    }


def test_first_response_to_user_must_address_question_focus():
    reason = _validate_discussion_response(
        _valid_user_response("IoT 센서 설치 비용을 줄이기 위해 기본 기능부터 시작합니다."),
        "initial_position",
        current_speaker_id="dev_expert",
        responding_to_speaker_id="user",
        responding_to_content="유지보수 문제를 어떻게 해결해야 하지?",
        require_user_question_focus=True,
    )
    assert reason == "spoken_text_does_not_answer_user_question"


def test_first_response_to_user_passes_when_it_directly_addresses_question():
    reason = _validate_discussion_response(
        _valid_user_response("유지보수 문제는 센서 정기 점검과 장애 대응 담당자를 지정해 해결할 수 있습니다."),
        "initial_position",
        current_speaker_id="dev_expert",
        responding_to_speaker_id="user",
        responding_to_content="유지보수 문제를 어떻게 해결해야 하지?",
        require_user_question_focus=True,
    )
    assert reason is None


def test_discussion_response_must_keep_deterministic_effective_issue():
    reason = _validate_discussion_response(
        _valid_user_response("문제의 영향을 구체적으로 정의해야 합니다."),
        expected_issue_id="target_user",
    )
    assert reason == "active_issue_id_mismatch"


def test_problem_issue_rejects_implementation_only_spoken_text():
    raw = _valid_user_response("실시간 센서 데이터를 수집하고 API로 연동해야 합니다.")
    reason = _validate_discussion_response(
        raw,
        expected_issue_id="problem",
        expected_issue_title="문제 정의",
    )
    assert reason == "spoken_text_issue_mismatch"


def test_problem_issue_accepts_spoken_text_about_user_problem():
    raw = _valid_user_response("도시 오염으로 시민이 겪는 피해와 기존 대응의 부족함을 먼저 정의해야 합니다.")
    reason = _validate_discussion_response(
        raw,
        expected_issue_id="problem",
        expected_issue_title="문제 정의",
    )
    assert reason is None


def test_problem_issue_rejects_token_problem_followed_by_implementation_drift():
    raw = _valid_user_response(
        "침수 문제를 해결하려면 데이터 확보와 센서 연동, 운영 비용을 먼저 정해야 합니다."
    )
    reason = _validate_discussion_response(
        raw,
        expected_issue_id="problem",
        expected_issue_title="문제 정의",
    )
    assert reason == "spoken_text_issue_drift"


def test_problem_issue_allows_implementation_term_when_problem_is_concrete():
    raw = _valid_user_response(
        "침수 위험으로 시민 대피가 늦어지는 피해와 경보 부족이 핵심 문제이며, 센서 연동은 이후 쟁점입니다."
    )
    reason = _validate_discussion_response(
        raw,
        expected_issue_id="problem",
        expected_issue_title="문제 정의",
    )
    assert reason is None


def test_claim_type_must_match_selected_evidence_document_role():
    raw = _valid_user_response("아이디어에는 실시간 센서 수집 기능이 포함됩니다.")
    raw["claims"] = [
        {
            "claim_id": "claim_1",
            "text": "아이디어에는 실시간 센서 수집 기능이 포함됩니다.",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        }
    ]
    reason = _validate_discussion_response(
        raw,
        expected_issue_id="problem",
        require_issue_content_focus=False,
        evidence_claim_types_by_ref={"E1": "user_provided_fact"},
    )
    assert reason == "claim_type_evidence_role_mismatch"


def test_expert_judgment_cannot_borrow_document_citation():
    raw = _valid_user_response("운영 비용을 줄이려면 수집 주기를 제한하는 편이 좋습니다.")
    raw["claims"] = [
        {
            "claim_id": "claim_1",
            "text": "운영 비용을 줄이려면 수집 주기를 제한하는 편이 좋습니다.",
            "claim_type": "expert_judgment",
            "evidence_refs": ["E1"],
        }
    ]
    reason = _validate_discussion_response(
        raw,
        expected_issue_id="problem",
        require_issue_content_focus=False,
        evidence_claim_types_by_ref={"E1": "user_provided_fact"},
    )
    assert reason == "claim_type_evidence_role_mismatch"


def test_criteria_fact_cannot_silently_become_negative_evaluation():
    raw = _valid_user_response(
        "현재 아이디어는 도시 문제의 설정이 구체적이지 않아 보완이 필요합니다."
    )
    raw["claims"] = [
        {
            "claim_id": "claim_1",
            "text": "도시 문제의 설정이 구체적인가?",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        }
    ]
    reason = _validate_discussion_response(
        raw,
        expected_issue_id="problem",
        require_issue_content_focus=False,
        evidence_claim_types_by_ref={"E1": "document_fact"},
    )
    assert reason == "spoken_evaluation_missing_expert_judgment_claim"


def test_negative_evaluation_passes_when_explicitly_typed_as_expert_judgment():
    raw = _valid_user_response(
        "평가 기준은 도시 문제의 구체성을 확인하며, 현재 설명은 피해 범위를 더 명확히 할 필요가 있습니다."
    )
    raw["claims"] = [
        {
            "claim_id": "claim_1",
            "text": "도시 문제의 설정이 구체적인가?",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        },
        {
            "claim_id": "claim_2",
            "text": "현재 설명은 피해 범위를 더 명확히 할 필요가 있다.",
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        },
    ]
    reason = _validate_discussion_response(
        raw,
        expected_issue_id="problem",
        require_issue_content_focus=False,
        evidence_claim_types_by_ref={"E1": "document_fact"},
    )
    assert reason is None


def test_discussion_prompt_contains_code_verified_current_speaker_and_target():
    prompt = build_ideation_conv_discussion_prompt(
        "planning_expert",
        {},
        {},
        [],
        {},
        speaks_second=True,
        discussion_stage="response",
        current_speaker={"speaker_id": "planning_expert", "role_name": "기획 위원"},
        responding_to_message={
            "speaker_id": "dev_expert",
            "role_name": "개발 위원",
            "message_id": "MSG-DEV-1",
            "spoken_text": "실시간 갱신은 운영 비용이 큽니다.",
        },
    )
    assert '"speaker_id": "planning_expert"' in prompt
    assert '"speaker_id": "dev_expert"' in prompt
    assert '"message_id": "MSG-DEV-1"' in prompt
    assert "자신의 역할을 제3자로" in prompt
    assert "부르지 않는다" in prompt


def test_trace_is_off_by_default(caplog):
    configure_ideation_trace(enabled=False, content_max_chars=500, stream_deltas=False)
    with caplog.at_level(logging.INFO, logger="ai.meeting.ideation_trace"):
        trace_event("IDEATION_TURN_END", text="보이면 안 됨")
    assert "IDEATION_TURN_END" not in caplog.text


def test_trace_masks_pii_secrets_and_truncates(caplog):
    configure_ideation_trace(enabled=True, content_max_chars=45, stream_deltas=False)
    tokens = bind_trace_context("SESSION-1", "REQUEST-1")
    try:
        preview = sanitize_preview("user@example.com 010-1234-5678 sk-abcdefghijklmnop 아주 긴 발언입니다")
        with caplog.at_level(logging.INFO, logger="ai.meeting.ideation_trace"):
            trace_event(
                "IDEATION_TURN_END",
                speaker="planning_expert",
                target="dev_expert",
                issue="user@example.com",
                text=preview,
            )
    finally:
        reset_trace_context(tokens)

    assert "SESSION-1" in caplog.text and "REQUEST-1" in caplog.text
    assert 'speaker="planning_expert"' in caplog.text and 'target="dev_expert"' in caplog.text
    assert "user@example.com" not in caplog.text
    assert "010-1234-5678" not in caplog.text
    assert "sk-abcdefghijklmnop" not in caplog.text
    assert "[EMAIL]" in caplog.text and "[PHONE]" in caplog.text and "[SECRET]" in caplog.text


def test_stream_delta_trace_requires_both_flags():
    configure_ideation_trace(enabled=True, content_max_chars=500, stream_deltas=False)
    assert stream_delta_trace_enabled() is False
    configure_ideation_trace(enabled=True, content_max_chars=500, stream_deltas=True)
    assert stream_delta_trace_enabled() is True
    configure_ideation_trace(enabled=False, content_max_chars=500, stream_deltas=True)
    assert stream_delta_trace_enabled() is False


def test_late_request_event_is_warned_and_detected(caplog):
    configure_ideation_trace(enabled=True, content_max_chars=500, stream_deltas=False)
    with caplog.at_level(logging.WARNING, logger="ai.meeting.ideation_trace"):
        assert is_late_request_event("REQUEST-OLD", "REQUEST-NOW") is True
    assert "IDEATION_LATE_REQUEST_EVENT" in caplog.text
    assert "REQUEST-OLD" in caplog.text and "REQUEST-NOW" in caplog.text
