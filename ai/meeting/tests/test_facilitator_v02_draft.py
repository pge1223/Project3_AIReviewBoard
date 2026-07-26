import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEETING_DIR))

from graph.application_form_draft import (
    apply_application_form_draft_patch,
    initialize_application_form_draft,
)
from graph.ideation_conv_build import _route_after_candidate_selection
from graph.ideation_conv_nodes import (
    _compose_form_facilitator_text,
    _facilitator_context_anchors,
    _form_facilitator_fallback_payload,
    _make_validate_form_facilitator_response,
    _route_next_expert_turn,
)
from prompts import build_ideation_conv_discussion_facilitator_prompt


def test_draft_patch_updates_only_known_form_fields_and_respects_char_limit():
    draft = initialize_application_form_draft(
        [
            {"field_name": "첫 번째 항목", "description": "", "char_limit": 5},
            {"field_name": "두 번째 항목", "description": "", "char_limit": None},
        ]
    )

    updated, applied = apply_application_form_draft_patch(
        draft,
        [
            {"field_id": "form_field_1", "value": "1234567", "status": "draft"},
            {"field_id": "unknown", "value": "무시", "status": "confirmed"},
        ],
    )

    assert updated[0]["value"] == "12345"
    assert updated[1]["value"] == ""
    assert applied == [{"field_id": "form_field_1", "value": "12345", "status": "draft"}]


def test_candidate_hypothesis_cannot_be_confirmed_without_answering_that_field():
    draft = initialize_application_form_draft([{"field_name": "동적 양식 항목"}])

    updated, applied = apply_application_form_draft_patch(
        draft,
        [{"field_id": "form_field_1", "value": "초기 가설", "status": "confirmed"}],
        confirmable_field_ids=set(),
    )

    assert updated[0]["status"] == "draft"
    assert applied[0]["status"] == "draft"


def test_answered_current_field_can_be_promoted_to_confirmed():
    draft = initialize_application_form_draft([{"field_name": "동적 양식 항목"}])

    updated, _ = apply_application_form_draft_patch(
        draft,
        [{"field_id": "form_field_1", "value": "사용자가 확정한 내용", "status": "confirmed"}],
        confirmable_field_ids={"form_field_1"},
    )

    assert updated[0]["status"] == "confirmed"


def _valid_problem_facilitator_payload() -> dict:
    return {
        "phase": "problem_definition",
        "current_field_id": "form_field_1",
        "confirmed_content": "아직 없음",
        "decision_reason": "공고가 구체적인 문제 설정을 평가하므로 먼저 정해야 합니다.",
        "spoken_text": "문제 상황을 먼저 정리하겠습니다.",
        "user_question": "이동 과정에서 가장 먼저 해결할 불편은 무엇인가요?",
        "choices": [
            {"id": "focus_1", "label": "이동 지연이 반복되는 상황"},
            {"id": "direct_input", "label": "다른 문제를 직접 입력"},
        ],
        "expert_insight_summary": {"planning": "", "development": ""},
        "draft_patch": [],
        "next_action": "ask_user",
        "agreements": [],
        "disagreements": [],
        "facilitator_summary": "문제 정의를 먼저 확인합니다.",
        "needs_user_decision": True,
    }


def test_form_facilitator_contract_requires_all_four_visible_parts():
    draft = initialize_application_form_draft([{"field_name": "사업 목적"}])
    validate = _make_validate_form_facilitator_response(
        draft,
        focus_phase="form_field_1",
        next_phase="form_field_2",
        prior_field_id=None,
    )
    payload = _valid_problem_facilitator_payload()

    assert validate(payload) is None

    del payload["decision_reason"]
    assert validate(payload) == "missing_or_empty_field:decision_reason"


def test_early_facilitator_phase_rejects_downstream_technical_question():
    draft = initialize_application_form_draft([{"field_name": "사업 목적"}])
    validate = _make_validate_form_facilitator_response(
        draft,
        focus_phase="form_field_1",
        next_phase="form_field_2",
        prior_field_id=None,
        is_session_first_turn=True,
    )
    payload = _valid_problem_facilitator_payload()
    payload["user_question"] = "데이터 수집 인프라는 어떻게 구축할 예정인가요?"

    assert validate(payload) == "downstream_technical_topic_before_problem_target_context"


def test_form_facilitator_text_hides_empty_confirmation_marker():
    content = _compose_form_facilitator_text(
        "사업 목적",
        "아직 없음",
        "구체적인 문제 설정이 평가 기준이기 때문에 지금 결정해야 합니다.",
        "누가 어떤 상황에서 가장 큰 불편을 겪나요?",
        [],
    )

    assert "사업 목적" in content
    assert "지금까지 확정된 내용: 아직 없음" not in content
    assert "아직 없음" not in content
    assert "평가 기준" in content
    assert "누가 어떤 상황" in content
    assert len(content.splitlines()) == 3


def test_facilitator_context_anchors_are_derived_from_selected_idea():
    anchors = _facilitator_context_anchors(
        {
            "title": "AI 기반 스마트 모빌리티 서비스",
            "problem": "출퇴근 교통 혼잡으로 이동 시간이 길다",
            "target_user": "대중교통 이용 직장인",
        },
        None,
    )

    assert "교통" in anchors
    assert "모빌리티" in anchors
    assert "서비스" not in anchors


def test_facilitator_rejects_generic_question_reused_across_ideas():
    draft = initialize_application_form_draft([{"field_name": "사업 목적"}])
    validate = _make_validate_form_facilitator_response(
        draft,
        focus_phase="form_field_1",
        next_phase="form_field_2",
        prior_field_id=None,
        context_anchors=["교통"],
    )
    payload = _valid_problem_facilitator_payload()
    payload["decision_reason"] = "교통 문제를 구체화해야 공고 평가 기준과 연결할 수 있습니다."
    payload["user_question"] = "교통 서비스의 목적이 무엇인지 자세히 설명해 주실 수 있나요?"

    assert validate(payload) == "vague_facilitator_question"


def test_facilitator_question_must_name_selected_idea_context():
    draft = initialize_application_form_draft([{"field_name": "사업 목적"}])
    validate = _make_validate_form_facilitator_response(
        draft,
        focus_phase="form_field_1",
        next_phase="form_field_2",
        prior_field_id=None,
        context_anchors=["교통"],
    )
    payload = _valid_problem_facilitator_payload()

    assert validate(payload) == "question_not_grounded_in_selected_idea"

    payload["decision_reason"] = "교통 문제를 먼저 좁혀야 평가 기준과 연결할 수 있습니다."
    payload["user_question"] = "교통 이동 중 가장 반복적으로 지연되는 상황은 언제인가요?"
    assert validate(payload) is None


def test_facilitator_validation_failure_has_contextual_non_failing_fallback():
    draft = initialize_application_form_draft(
        [
            {"field_name": "담당자 성명"},
            {"field_name": "사업 목적"},
            {"field_name": "필요 데이터"},
        ]
    )
    fallback = _form_facilitator_fallback_payload(
        draft,
        focus_phase="form_field_2",
        prior_field_id=None,
        context_anchors=["교통"],
        selected_idea={"problem": "출퇴근 교통 혼잡으로 이동 시간이 길어지는 문제"},
        is_session_first_turn=True,
    )
    validate = _make_validate_form_facilitator_response(
        draft,
        focus_phase="form_field_2",
        next_phase="form_field_3",
        prior_field_id=None,
        context_anchors=["교통"],
        is_session_first_turn=True,
    )

    assert fallback["current_field_id"] == "form_field_2"
    assert fallback["confirmed_content"] == "아직 없음"
    assert "교통" in fallback["user_question"]
    assert len(fallback["choices"]) == 2
    assert "출퇴근 교통 혼잡" in fallback["choices"][0]["label"]
    assert fallback["safe_fallback"] is True
    assert validate(fallback) is None


def test_candidate_selection_starts_with_form_coach_when_form_exists():
    state = {
        "phase": "expert_discussion",
        "next_route": "to_refinement",
        "application_form_items": [{"field_name": "사업 목적"}],
    }

    assert _route_after_candidate_selection(state) == "to_form_coach"

    state["application_form_items"] = []
    assert _route_after_candidate_selection(state) == "to_refinement"


def test_form_phase_label_no_longer_gates_dev_expert_participation():
    """2026-07-26 라운드테이블 재설계: 예전엔 진행자의 structured.phase가 고정 9단계 중
    앞의 두 단계("problem_definition"/"target_user")일 때 기획 검토 뒤 dev_expert를
    건너뛰고 바로 facilitator로 돌려보냈다. 이제 그 게이트는 삭제됐다 — "전문가 발언 없이
    진행자 혼자 묻는" 턴(고정 문제정의 확인 턴)은 라우터가 호출되기도 전에
    discussion_facilitator가 처리하고 끝내므로, 이 라우터는 항상 "전문가가 이번 주제에
    대해 최소 1회는 말한 뒤"에만 불린다 — phase 라벨과 무관하게 recommended_next_speaker를
    그대로 따라야 한다."""
    state = {
        "session_id": "TEST-SESSION",
        "phase": "expert_discussion",
        "application_form_items": [{"field_name": "사업 목적"}],
        "messages": [
            {
                "speaker_id": "ideation_facilitator",
                "structured": {"current_field_id": "form_field_1"},
            },
            {
                "speaker_id": "planning_expert",
                "structured": {
                    "recommended_next_speaker": "dev_expert",
                    "needs_user_input": False,
                },
            },
        ],
        "expert_turn_count": 1,
        "open_issues": [{"issue_id": "issue-1", "turns": 1}],
        "active_issue_id": "issue-1",
        "required_counterpart_speaker_id": None,
        "counterpart_review_completed": True,
    }

    assert _route_next_expert_turn(state) == "dev_expert"


def test_facilitator_v02_prompt_receives_form_draft_and_latest_user_answer():
    prompt = build_ideation_conv_discussion_facilitator_prompt(
        notice_and_criteria={},
        planning_position={},
        development_review={},
        revised_proposal=None,
        consensus_so_far=[],
        unresolved_issues=[],
        decided_next_action="await_user_decision",
        round_number=1,
        max_rounds=3,
        application_form_items=[{"field_name": "동적 양식 항목"}],
        application_form_draft=[
            {
                "field_id": "form_field_1",
                "field_name": "동적 양식 항목",
                "value": "",
                "status": "empty",
            }
        ],
        latest_user_answer="사용자가 방금 제공한 내용",
        remaining_form_fields=[
            {"field_id": "form_field_1", "field_name": "동적 양식 항목", "description": ""}
        ],
        meeting_stage_hint="form_filling",
        context_anchors=["교통", "출퇴근"],
        recent_messages=[
            {
                "message_id": "MSG-USER-1",
                "speaker_id": "user",
                "message_type": "answer",
                "content": "사용자가 앞에서 말한 내용",
            },
            {
                "message_id": "MSG-FACILITATOR-1",
                "speaker_id": "ideation_facilitator",
                "message_type": "summary",
                "content": "진행자가 직전에 정리한 내용",
            },
            {
                "message_id": "MSG-PLANNING-1",
                "speaker_id": "planning_expert",
                "message_type": "opinion",
                "content": "기획 전문가가 최근 대화에서 수정한 의견",
            },
            {
                "message_id": "MSG-DEV-1",
                "speaker_id": "dev_expert",
                "message_type": "opinion",
                "content": "개발 전문가가 최근 대화에서 덧붙인 조건",
            },
        ],
    )

    assert '"draft_patch"' in prompt
    assert '"confirmed_content"' in prompt
    assert '"decision_reason"' in prompt
    assert '정확히 "아직 없음"' in prompt
    assert "현재 단계보다 앞선 기술 검토" in prompt
    assert "초기 가설" in prompt
    assert "답변 범위가 불명확한 표현" in prompt
    assert '"field_id": "form_field_1"' in prompt
    assert "사용자가 방금 제공한 내용" in prompt
    assert "사용자가 앞에서 말한 내용" in prompt
    assert "진행자가 직전에 정리한 내용" in prompt
    assert "기획 전문가가 최근 대화에서 수정한 의견" in prompt
    assert "개발 전문가가 최근 대화에서 덧붙인 조건" in prompt
    assert "어느 하나만 보고 전문가 의견을 요약하지 않는다" in prompt
    assert "<<APPLICATION_FORM_DRAFT_JSON>>" not in prompt
    assert "<<RECENT_MESSAGES_JSON>>" not in prompt
    assert "<<REMAINING_FORM_FIELDS_JSON>>" not in prompt
    assert "<<MEETING_STAGE_HINT>>" not in prompt
    assert "meeting_stage_hint" in prompt.lower() or "form_filling" in prompt
    assert '"교통"' in prompt
    assert "<<CONTEXT_ANCHORS_JSON>>" not in prompt
