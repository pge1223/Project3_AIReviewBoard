import json
from pathlib import Path

from ai.meeting.form_coach import (
    LLMAnswerAssessmentProvider,
    finalize_session,
    reply_to_session,
    start_session,
)
from ai.meeting.form_coach.expert_review import LLMExpertReviewProvider
from ai.meeting.form_coach.prompt_builder import build_prompt
from ai.meeting.form_coach.response_validator import validate_response


def _response(
    *,
    phase: str,
    update: dict | None = None,
    draft_patch: list[dict] | None = None,
    choices: list[dict] | None = None,
    next_action: str = "ask_user",
    assessment: dict | None = None,
    candidate_update: dict | None = None,
    pre_guidance: dict | None = None,
) -> str:
    should_advance = bool(update) or next_action == "generate_application_draft"
    return json.dumps(
        {
            "internal_state": {
                "phase": phase,
                "current_form_fields": ["form_field_1"],
                "confirmed_summary": {},
                "decision_reason": "다음 기획 결정을 구체화해야 합니다.",
                "pre_answer_guidance": pre_guidance or {
                    "needed": False,
                    "reason": "none",
                    "mentor_types": [],
                },
                "answer_assessment": assessment or {
                    "fits_current_phase": True,
                    "specificity": "specific" if should_advance else "broad",
                    "needs_clarification": not should_advance,
                    "should_advance_phase": should_advance,
                    "mentor_needed": not should_advance,
                    "mentor_types": ["planning"] if not should_advance else [],
                },
                "draft_patch": draft_patch or [],
                "confirmed_state_update": update or {},
                "candidate_state_update": candidate_update or {},
                "next_action": next_action,
            },
            "ui_message": {
                "facilitator_text": "다음 기획 결정을 한 단계 더 구체화해볼게요.",
                "mentor_messages": [
                    {
                        "expert_type": "planning",
                        "name": "기획 전문가",
                        "text": "확정한 내용을 바탕으로 다음 기획 결정을 좁혀야 합니다.",
                    },
                    {
                        "expert_type": "development",
                        "name": "개발 전문가",
                        "text": "확정한 내용을 바탕으로 다음 구현 범위를 좁혀야 합니다.",
                    },
                ],
                "question": "이번 단계에서 우선할 방향을 골라주세요.",
                "choices": choices or [],
            },
        },
        ensure_ascii=False,
    )


def _items():
    return [
        {"field_name": "AI 솔루션 적용 기술 설명", "description": "기술과 적용 내용을 설명"},
        {"field_name": "혁신성", "description": "아이디어의 혁신 정도"},
    ]


def _start(first_response=None):
    return start_session(
        session_id="FORM-COACH-TEST",
        competition_name="테스트 공모전",
        competition_document="혁신성과 적용성을 평가한다.",
        selected_idea={"title": "운영 최적화", "problem": "운영 비효율"},
        idea_canvas={"problem": "운영 비효율로 이용자가 불편하다."},
        application_form_items=_items(),
        llm_call=lambda _: first_response or _response(phase="problem_definition"),
    )


def test_first_turn_contract_is_problem_even_when_evaluation_field_is_first():
    choices = [
        {"id": "focus_a", "label": "첫 번째 문제 초점", "description": "후보 문맥에서 도출한 설명"},
        {"id": "custom", "label": "직접 입력 또는 결합", "description": "원하는 상황을 직접 설명"},
    ]
    state = _start(_response(phase="problem_definition", choices=choices))
    structured = state["messages"][-1]["structured"]

    assert state["current_phase"] == "problem_definition"
    assert state["application_form_draft"][0]["status"] == "in_progress"
    assert structured["current_field_ids"] == ["form_field_1"]
    assert structured["choices"][0]["description"] == "후보 문맥에서 도출한 설명"
    assert len([choice for choice in structured["choices"] if choice["id"] != "custom"]) >= 2
    assert structured["choices"][-1]["id"] == "custom"
    assert structured["confirmed_state_update"] == {}


def test_selected_candidate_seeds_canvas_when_legacy_canvas_is_empty():
    selected = {
        "title": "생활 건강 지원",
        "problem": "주민이 필요한 건강 정보를 찾기 어렵다.",
        "target_user": "도시 주민",
        "solution": "맞춤형 정보 안내",
        "core_value": "건강 정보 접근성 개선",
    }
    state = start_session(
        session_id="FORM-COACH-CANVAS",
        competition_name="테스트 공모전",
        competition_document="",
        selected_idea=selected,
        idea_canvas={},
        application_form_items=_items(),
        llm_call=lambda _: _response(phase="problem_definition"),
    )

    assert state["idea_canvas"]["problem"] == selected["problem"]
    assert state["idea_canvas"]["target_user"] == selected["target_user"]
    assert state["idea_canvas"]["solution"] == selected["solution"]
    assert state["idea_canvas"]["core_value"] == selected["core_value"]


def test_short_answer_is_saved_as_facilitator_summary_and_advances():
    state = _start()
    state = reply_to_session(
        state,
        answer="운행 간격이 길다",
        llm_call=lambda _: _response(
            phase="target_and_context",
            update={
                "problem": "수요 변화에 비해 운행 간격 조정이 늦어 이용자의 대기 시간이 길어진다."
            },
        ),
    )

    assert state["current_phase"] == "target_and_context"
    assert state["confirmed_state"]["problem"].startswith("수요 변화에 비해")
    assert state["confirmed_state"]["problem"] != "운행 간격이 길다"
    assert state["idea_canvas"]["problem"] == state["confirmed_state"]["problem"]


def test_ambiguous_answer_without_state_update_stays_on_same_phase():
    state = _start()
    state = reply_to_session(
        state,
        answer="여러 문제가 있습니다.",
        llm_call=lambda _: _response(phase="problem_definition", update={}),
    )

    assert state["current_phase"] == "problem_definition"
    assert state["confirmed_state"]["problem"] == ""


def test_off_track_solution_is_persisted_as_candidate_not_confirmed_problem():
    state = start_session(
        session_id="FORM-COACH-CANDIDATE",
        competition_name="테스트 공모전",
        competition_document="",
        selected_idea={"title": "생활 지원"},
        idea_canvas={},
        application_form_items=[
            {"field_name": "현황 및 문제점", "description": "해결할 문제"},
            {"field_name": "제안 내용", "description": "해결 방식과 기능"},
        ],
        llm_call=lambda _: _response(phase="problem_definition"),
    )
    state = reply_to_session(
        state,
        answer="온라인 플랫폼",
        llm_call=lambda _: _response(
            phase="problem_definition",
            assessment={
                "fits_current_phase": False,
                "off_track_category": "solution_candidate",
                "specificity": "specific",
                "needs_clarification": True,
                "should_advance_phase": False,
                "mentor_needed": True,
                "mentor_types": ["planning"],
            },
            candidate_update={"solution_candidate": "온라인 플랫폼"},
        ),
    )

    assert state["current_phase"] == "problem_definition"
    assert state["confirmed_state"]["problem"] == ""
    assert state["candidate_state"]["solution_candidate"] == "온라인 플랫폼"
    assert state["application_form_draft"][0]["provisional_value"] == ""
    assert state["application_form_draft"][1]["status"] == "in_progress"
    assert "온라인 플랫폼" in state["application_form_draft"][1]["provisional_value"]
    mentor_text = state["messages"][-1]["structured"]["mentor_messages"][0]["text"]
    assert "온라인 플랫폼" in mentor_text
    assert "해결 방식 후보" in mentor_text


def test_application_fields_are_filled_incrementally_without_llm_patches():
    state = start_session(
        session_id="FORM-COACH-INCREMENTAL",
        competition_name="테스트 공모전",
        competition_document="",
        selected_idea={"title": "생활 지원"},
        idea_canvas={},
        application_form_items=[
            {"field_name": "현황 및 문제점", "description": "해결할 문제"},
            {"field_name": "제안 대상 및 적용 현장", "description": "대상 사용자와 현장"},
            {"field_name": "제안 목적", "description": "목표와 핵심 가치"},
            {"field_name": "제안 내용 및 추진 방법", "description": "해결 방식"},
        ],
        llm_call=lambda _: _response(phase="problem_definition"),
    )
    turns = [
        ("problem", "주민이 필요한 정보를 제때 찾기 어렵다.", "target_and_context", 0),
        ("target_user", "지역 주민", "purpose_and_value", 1),
        ("value", "필요한 정보에 대한 접근성을 높인다.", "solution_scenario", 2),
        ("solution", "상황에 맞는 정보를 안내한다.", "ai_role_and_mvp", 3),
    ]
    for key, value, phase, row_index in turns:
        state = reply_to_session(
            state,
            answer=value,
            llm_call=lambda _, key=key, value=value, phase=phase: _response(
                phase=phase,
                update={key: value},
            ),
        )
        row = state["application_form_draft"][row_index]
        assert row["value"]
        assert row["status"] == "draft"
        assert row["source"] == "conversation"
        assert row["confidence"] == "medium"
        assert row["editable"] is True

    assert state["application_form_draft"][0]["value"] == "주민이 필요한 정보를 제때 찾기 어렵다."
    assert "지역 주민" in state["application_form_draft"][1]["value"]
    assert state["application_form_draft"][2]["value"] == "필요한 정보에 대한 접근성을 높인다."
    assert state["application_form_draft"][3]["value"] == "상황에 맞는 정보를 안내한다."


def test_full_eight_phase_flow_can_finalize():
    state = _start()
    turns = [
        ({"problem": "문제"}, "target_and_context", "ask_user"),
        ({"target": "대상"}, "purpose_and_value", "ask_user"),
        ({"purpose": "목적"}, "solution_scenario", "ask_user"),
        ({"solution": "해결 방식"}, "ai_role_and_mvp", "ask_user"),
        ({"ai_role": "판단 보조", "mvp_scope": "입력 보조", "required_data": "운영 기록"}, "effect_and_validation", "ask_user"),
        ({"expected_effect": "대기 감소", "validation_method": "처리 시간 비교"}, "differentiation_and_risk", "ask_user"),
        ({"differentiation": "현장 중심", "risk": "데이터 편향"}, "application_drafting", "confirm_summary"),
        ({}, "application_drafting", "generate_application_draft"),
    ]
    for update, phase, action in turns:
        state = reply_to_session(
            state,
            answer="사용자 답변",
            llm_call=lambda _, update=update, phase=phase, action=action: _response(
                phase=phase,
                update=update,
                next_action=action,
            ),
        )

    assert state["phase"] == "discussion_complete"
    finalized = finalize_session(state)
    assert finalized["phase"] == "finalized"
    assert finalized["idea_proposal"]["problem_definition"] == "문제"


def test_prompt_does_not_hardcode_public_transport_example():
    prompt = (
        Path(__file__).resolve().parents[1] / "prompts" / "ideation_form_coach_01.txt"
    ).read_text(encoding="utf-8")
    assert "대중교통" not in prompt
    assert "출퇴근 시간대" not in prompt


def test_facilitator_prompt_receives_full_selected_idea_each_turn():
    prompt = build_prompt(
        state={
            "competition_name": "테스트 공모전",
            "selected_idea": {
                "title": "선택 후보",
                "problem": "후보에 담긴 문제",
                "target": "후보에 담긴 대상",
                "solution": "후보에 담긴 해결 방식",
            },
            "idea_canvas": {},
            "application_form_draft": [],
            "confirmed_state": {},
            "candidate_state": {"solution_candidate": "온라인 플랫폼"},
            "messages": [],
        },
        current_field={"field_id": "form_field_1"},
        current_phase="problem_definition",
        previous_phase=None,
        previous_field=None,
        latest_user_answer="",
        evidence=[],
        expert_review={"planning": "", "development": ""},
    )

    assert "[selected_idea]" in prompt
    assert "후보에 담긴 문제" in prompt
    assert "후보에 담긴 대상" in prompt
    assert "후보에 담긴 해결 방식" in prompt
    assert "[candidate_state]" in prompt
    assert "온라인 플랫폼" in prompt


def test_expert_review_failure_is_non_blocking():
    review = LLMExpertReviewProvider(lambda _: (_ for _ in ()).throw(RuntimeError("unavailable"))).review(
        phase="problem_definition",
        current_field={"field_id": "form_field_1", "field_name": "기술 설명"},
        state={"selected_idea": {}, "idea_canvas": {}, "application_form_draft": []},
        latest_user_answer="",
    )
    assert review == {"planning": "", "development": ""}


def _validated_ui(
    *,
    phase: str,
    answer: str = "",
    assessment: dict | None = None,
    update: dict | None = None,
    candidate_update: dict | None = None,
    mentor_messages: list[dict] | None,
    post_answer_mentor_messages: list[dict] | None = None,
    expert_review: dict,
    choices: list[dict] | None = None,
    confirmed_state: dict | None = None,
    confirmed_summary: dict | None = None,
    question: str = "가장 먼저 적용할 방향을 골라주세요.",
    previous_question: str = "",
):
    raw = json.dumps(
        {
            "internal_state": {
                "phase": phase,
                "current_form_fields": ["form_field_1"],
                "confirmed_summary": confirmed_summary or {"problem": "내부 문제 요약"},
                "decision_reason": "이 값은 사용자 화면에 노출하지 않습니다.",
                "answer_assessment": assessment or {},
                "confirmed_state_update": update or {},
                "candidate_state_update": candidate_update or {},
                "draft_patch": [],
                "next_action": "ask_user",
            },
            "ui_message": {
                "facilitator_text": "다음 선택의 범위를 좁혀볼게요.",
                "mentor_messages": mentor_messages or [],
                "post_answer_mentor_messages": post_answer_mentor_messages,
                "question": question,
                "choices": choices or [],
            },
        },
        ensure_ascii=False,
    )
    return validate_response(
        raw,
        current_phase=phase,
        confirmed_state=confirmed_state or {"selected_idea": "테스트 아이디어"},
        connected_form_fields=[{"field_id": "form_field_1", "field_name": "기술 설명"}],
        latest_user_answer=answer,
        expert_review=expert_review,
        application_form_draft=[{"field_id": "form_field_1", "field_name": "기술 설명"}],
        previous_question=previous_question,
    )


def test_first_phase_entry_does_not_expose_expert_without_user_answer():
    message, *_ = _validated_ui(
        phase="problem_definition",
        mentor_messages=[],
        expert_review={"planning": "내부 검토", "development": ""},
    )

    assert message["content"] == "다음 선택의 범위를 좁혀볼게요."
    assert "내부 문제 요약" not in message["content"]
    assert "decision_reason" not in message["content"]
    assert message["structured"]["ui_message"]["mentor_messages"] == []
    assert message["structured"]["internal_state"]["answer_assessment"]["specificity"] == "none"
    assert "mentor_message" not in message["structured"]["ui_message"]
    assert len(message["structured"]["choices"]) == 4
    assert message["structured"]["choices"][-1]["id"] == "custom"


def test_broad_problem_answer_stays_in_phase_and_requires_planning_mentor():
    message, state_update, _, resulting_phase, _ = _validated_ui(
        phase="problem_definition",
        answer="소비 데이터 부족",
        assessment={
            "fits_current_phase": True,
            "specificity": "broad",
            "needs_clarification": True,
            "should_advance_phase": False,
            "mentor_needed": True,
            "mentor_types": ["planning"],
        },
        update={"problem": "모델이 잘못 확정하려 한 문장"},
        mentor_messages=[],
        expert_review={"planning": "어떤 소비 상황의 데이터가 부족한지 먼저 좁혀야 합니다.", "development": ""},
    )

    assert resulting_phase == "problem_definition"
    assert state_update == {}
    assert message["structured"]["mentor_messages"][0]["expert_type"] == "planning"


def test_specific_problem_answer_is_confirmed_and_advances_without_mentor():
    message, state_update, _, resulting_phase, _ = _validated_ui(
        phase="problem_definition",
        answer="퇴근 시간대 버스 배차 조정이 늦어 정류장 대기 시간이 길어집니다.",
        assessment={
            "fits_current_phase": True,
            "specificity": "specific",
            "needs_clarification": False,
            "should_advance_phase": True,
            "mentor_needed": False,
            "mentor_types": [],
        },
        update={"problem": "퇴근 시간대 배차 조정 지연으로 정류장 대기 시간이 길어진다."},
        mentor_messages=[
            {
                "expert_type": "planning",
                "text": "퇴근 시간대 배차 조정 지연은 발생 상황과 영향이 충분히 드러납니다.",
            }
        ],
        expert_review={"planning": "문제가 충분히 구체적입니다.", "development": ""},
    )

    assert resulting_phase == "target_and_context"
    assert state_update["problem"].startswith("퇴근 시간대")
    assert message["structured"]["mentor_messages"] == []


def test_purpose_turn_cannot_overwrite_confirmed_target():
    message, state_update, _, resulting_phase, _ = _validated_ui(
        phase="purpose_and_value",
        answer="개인 맞춤형 관리",
        assessment={
            "fits_current_phase": True,
            "specificity": "specific",
            "needs_clarification": False,
            "should_advance_phase": True,
            "mentor_needed": False,
            "mentor_types": [],
        },
        update={
            "target": "헬스케어 서비스 제공 기관",
            "purpose": "도시 주민이 개인 맞춤형 건강 관리를 받을 수 있게 한다.",
        },
        mentor_messages=[],
        expert_review={"planning": "", "development": ""},
        confirmed_state={
            "selected_idea": "AI 기반 헬스케어 플랫폼",
            "problem": "도시 주민의 건강 관리 공백",
            "target": "도시 주민",
            "purpose": "",
            "solution": "",
        },
        confirmed_summary={"target": "헬스케어 서비스 제공 기관"},
    )

    assert resulting_phase == "solution_scenario"
    assert state_update == {
        "purpose": "도시 주민이 개인 맞춤형 건강 관리를 받을 수 있게 한다.",
        "value": "도시 주민이 개인 맞춤형 건강 관리를 받을 수 있게 한다.",
    }
    assert message["structured"]["internal_state"]["confirmed_summary"]["target"] == "도시 주민"


def test_repeated_question_is_replaced_with_phase_specific_follow_up():
    repeated = "AI 기반 헬스케어 플랫폼을 통해 어떤 변화를 이루고 싶으신가요?"
    message, *_ = _validated_ui(
        phase="purpose_and_value",
        answer="개인 맞춤형 관리",
        assessment={
            "fits_current_phase": True,
            "specificity": "broad",
            "needs_clarification": True,
            "should_advance_phase": False,
            "mentor_needed": True,
            "mentor_types": ["planning"],
        },
        update={},
        mentor_messages=[],
        expert_review={
            "planning": "개인 맞춤형 관리가 만드는 변화를 한 가지로 좁히면 핵심 가치가 선명해집니다.",
            "development": "",
        },
        question=repeated,
        previous_question=repeated,
    )

    next_question = message["structured"]["ui_message"]["question"]
    assert next_question != repeated
    assert "가장 우선해야 할 변화 하나" in next_question


def test_off_phase_solution_answer_is_not_saved_as_problem():
    message, state_update, _, resulting_phase, _ = _validated_ui(
        phase="problem_definition",
        answer="커뮤니티 플랫폼을 만들고 싶어요.",
        assessment={
            "fits_current_phase": False,
            "off_track_category": "solution_candidate",
            "specificity": "specific",
            "needs_clarification": True,
            "should_advance_phase": True,
            "mentor_needed": True,
            "mentor_types": ["planning"],
        },
        update={"solution": "커뮤니티 플랫폼"},
        mentor_messages=[],
        expert_review={"planning": "플랫폼은 해결 방식 후보이므로 먼저 문제 상황을 정해야 합니다.", "development": ""},
    )

    assert resulting_phase == "problem_definition"
    assert state_update == {}
    assert message["structured"]["internal_state"]["answer_assessment"]["should_advance_phase"] is False
    assert message["structured"]["candidate_state_update"] == {
        "solution_candidate": "커뮤니티 플랫폼을 만들고 싶어요."
    }


def test_mixed_target_answer_requires_planning_clarification():
    message, _, _, resulting_phase, _ = _validated_ui(
        phase="target_and_context",
        answer="시민, 지자체, 운영기관 전부 다요.",
        assessment={
            "fits_current_phase": True,
            "specificity": "broad",
            "needs_clarification": True,
            "should_advance_phase": False,
            "mentor_needed": True,
            "mentor_types": ["planning"],
        },
        update={},
        mentor_messages=[],
        expert_review={"planning": "수혜자와 실제 도입·운영 주체를 나누어 우선순위를 정해야 합니다.", "development": ""},
    )

    assert resulting_phase == "target_and_context"
    assert message["structured"]["mentor_messages"][0]["expert_type"] == "planning"


def test_mentor_needed_with_generic_review_gets_answer_specific_fallback():
    message, *_ = _validated_ui(
        phase="ai_role_and_mvp",
        answer="AI가 다 해요.",
        assessment={
            "fits_current_phase": True,
            "specificity": "broad",
            "needs_clarification": True,
            "should_advance_phase": False,
            "mentor_needed": True,
            "mentor_types": ["development"],
        },
        update={},
        mentor_messages=[],
        expert_review={"planning": "", "development": "상세 설명이 필요합니다."},
    )

    mentor = message["structured"]["mentor_messages"][0]
    assert mentor["expert_type"] == "development"
    assert "AI가 다 해요" in mentor["text"]
    assert "첫 버전" in mentor["text"]


def test_two_optional_experts_are_limited_to_one_sentence_each():
    message, *_ = _validated_ui(
        phase="solution_scenario",
        answer="신고부터 자동 출동까지 모두 처리합니다.",
        assessment={
            "fits_current_phase": True,
            "specificity": "specific",
            "needs_clarification": False,
            "should_advance_phase": False,
            "mentor_needed": False,
            "mentor_types": [],
        },
        update={},
        mentor_messages=[
            {"expert_type": "planning", "text": "자동 출동은 사용자 가치 범위를 정해야 합니다. 기획 둘째 문장입니다."},
            {"expert_type": "development", "text": "자동 출동은 구현 범위와 제약을 정해야 합니다. 개발 둘째 문장입니다."},
        ],
        expert_review={"planning": "기획 내부 검토", "development": "개발 내부 검토"},
    )

    mentors = message["structured"]["mentor_messages"]
    assert [item["expert_type"] for item in mentors] == ["planning", "development"]
    assert "둘째 문장" not in mentors[0]["text"]
    assert "둘째 문장" not in mentors[1]["text"]


def test_first_question_does_not_expose_pre_answer_mentor_guidance():
    message, *_ = _validated_ui(
        phase="problem_definition",
        mentor_messages=[],
        expert_review={"planning": "", "development": ""},
    )

    ui_message = message["structured"]["ui_message"]
    assert ui_message["pre_answer_mentor_messages"] == []
    assert ui_message["post_answer_mentor_messages"] == []


def test_universal_target_is_confirmed_without_forced_clarification():
    message, state_update, _, resulting_phase, _ = _validated_ui(
        phase="target_and_context",
        answer="전 연령 누구나",
        assessment={
            "fits_current_phase": True,
            "specificity": "broad",
            "needs_clarification": True,
            "should_advance_phase": False,
            "mentor_needed": True,
            "mentor_types": ["planning"],
        },
        update={"target_user": "전 연령의 모든 사용자"},
        mentor_messages=[],
        expert_review={"planning": "연령을 더 좁혀야 합니다.", "development": ""},
    )

    assessment = message["structured"]["internal_state"]["answer_assessment"]
    assert state_update["target_user"] == "전 연령의 모든 사용자"
    assert assessment["needs_clarification"] is False
    assert assessment["mentor_needed"] is False
    assert resulting_phase == "purpose_and_value"


class _CountingExpertReviewProvider:
    def __init__(self):
        self.calls = 0

    def review(self, **kwargs):
        self.calls += 1
        return {
            "planning": f"'{kwargs['latest_user_answer']}'에서 빠진 상황 하나를 먼저 정해야 합니다.",
            "development": "",
        }


class _SequenceAssessmentProvider:
    def __init__(self, assessments):
        self.assessments = list(assessments)

    def assess(self, **kwargs):
        return self.assessments.pop(0)


def test_service_calls_post_answer_review_only_for_clarification():
    provider = _CountingExpertReviewProvider()
    assessment_provider = _SequenceAssessmentProvider(
        [
            {
                "fits_current_phase": True,
                "specificity": "specific",
                "needs_clarification": False,
                "should_advance_phase": True,
                "mentor_needed": False,
                "mentor_types": [],
            },
            {
                "fits_current_phase": True,
                "specificity": "broad",
                "needs_clarification": True,
                "should_advance_phase": False,
                "mentor_needed": True,
                "mentor_types": ["planning"],
            },
        ]
    )
    state = _start()
    state = reply_to_session(
        state,
        answer="퇴근 시간대 배차 조정이 늦어 대기 시간이 길어집니다.",
        llm_call=lambda _: _response(
            phase="target_and_context",
            update={"problem": "퇴근 시간대 배차 조정 지연으로 대기 시간이 길어진다."},
        ),
        answer_assessment_provider=assessment_provider,
        expert_review_provider=provider,
    )
    assert provider.calls == 0

    state = reply_to_session(
        state,
        answer="모두 다요.",
        llm_call=lambda _: _response(
            phase="target_and_context",
            assessment={
                "fits_current_phase": True,
                "specificity": "broad",
                "needs_clarification": True,
                "should_advance_phase": False,
                "mentor_needed": True,
                "mentor_types": ["planning"],
            },
        ),
        answer_assessment_provider=assessment_provider,
        expert_review_provider=provider,
    )
    assert provider.calls == 1
    post_messages = state["messages"][-1]["structured"]["ui_message"]["post_answer_mentor_messages"]
    assert post_messages[0]["expert_type"] == "planning"


class _DelegationProvider:
    def delegate(self, **kwargs):
        return {
            "owner_expert_type": "planning",
            "counterpart_expert_type": "development",
            "proposal": {
                "proposal": "출퇴근 시간대 배차 조정 지연으로 시민 대기 시간이 길어지는 문제",
                "spoken_text": "기획 관점에서는 출퇴근 시간대의 배차 지연을 우선 문제로 잡는 것이 좋습니다.",
            },
            "review": {
                "stance": "동의",
                "spoken_text": "개발 관점에서도 시간대별 운행 기록으로 검증 가능한 범위입니다.",
            },
            "facilitator": {
                "final_recommendation": "출퇴근 시간대 배차 조정 지연 문제를 우선 다룹니다.",
                "spoken_text": "두 의견을 반영해 우선 문제를 하나의 상황으로 정리했습니다.",
            },
            "suggested_value": "출퇴근 시간대 배차 조정 지연으로 시민 대기 시간이 길어지는 문제",
        }


def test_expert_delegation_stays_provisional_until_user_confirms():
    state = _start()
    state = reply_to_session(
        state,
        answer="잘 모르겠어요. 전문가가 추천해 주세요.",
        llm_call=lambda _: (_ for _ in ()).throw(AssertionError("facilitator should not run")),
        expert_delegation_provider=_DelegationProvider(),
    )

    assert state["phase"] == "awaiting_delegation_confirmation"
    assert state["confirmed_state"]["problem"] == ""
    assert state["candidate_state"]["delegation_candidate"]
    assert state["application_form_draft"][0]["source"] == "expert_delegation"
    assert state["application_form_draft"][0]["provisional_value"]
    ui = state["messages"][-1]["structured"]["ui_message"]
    assert [choice["id"] for choice in ui["choices"]] == [
        "apply_delegation",
        "reject_delegation",
    ]

    state = reply_to_session(
        state,
        answer="권고안 적용",
        llm_call=lambda _: _response(phase="target_and_context"),
        expert_delegation_provider=_DelegationProvider(),
    )

    assert state["phase"] == "awaiting_form_answer"
    assert state["current_phase"] == "target_and_context"
    assert state["confirmed_state"]["problem"].startswith("출퇴근 시간대")
    assert state["candidate_state"]["delegation_candidate"] == ""
    assert state["pending_delegation"] is None
    assert state["application_form_draft"][0]["status"] == "draft"
    assert state["application_form_draft"][0]["provisional_value"] == ""


class _SynthesisProvider:
    def synthesize(self, state):
        return {
            "problem_definition": "합성기가 바꾸려 한 문제",
            "one_line_pitch": "확정된 기획을 바탕으로 만든 한 줄 설명",
            "next_actions": ["신청서 문장 검토"],
            "final_recommendation": "추천",
        }


def test_synthesis_adds_summary_but_cannot_override_confirmed_v2_state():
    state = _start()
    state["phase"] = "discussion_complete"
    state["confirmed_state"]["problem"] = "사용자와 확정한 문제"
    state["application_form_draft"][0]["value"] = "사용자가 수정한 신청서 문장"

    finalized = finalize_session(state, synthesis_provider=_SynthesisProvider())

    assert finalized["idea_proposal"]["problem_definition"] == "사용자와 확정한 문제"
    assert finalized["idea_proposal"]["one_line_pitch"] == "확정된 기획을 바탕으로 만든 한 줄 설명"
    assert finalized["idea_proposal"]["application_form"][
        finalized["application_form_draft"][0]["field_name"]
    ] == "사용자가 수정한 신청서 문장"


def test_expert_timing_reason_forces_review_without_phase_requirement():
    provider = LLMAnswerAssessmentProvider(
        lambda _: json.dumps(
            {
                "fits_current_phase": True,
                "classification": "on_track",
                "specificity": "specific",
                "needs_clarification": False,
                "should_advance_phase": True,
                "mentor_needed": False,
                "mentor_reason": "tradeoff",
                "mentor_types": ["planning", "development"],
            }
        )
    )

    assessment = provider.assess(
        phase="solution_scenario",
        current_field={"field_id": "solution", "field_name": "제안 내용"},
        state={
            "competition_name": "공모전",
            "competition_document": "적용성과 실현 가능성을 평가한다.",
            "contest_analysis": {},
            "selected_idea": {},
            "idea_canvas": {},
            "confirmed_state": {},
            "candidate_state": {},
            "application_form_draft": [],
            "messages": [],
        },
        previous_question="초기 해결 방식을 골라주세요.",
        latest_user_answer="자동화 범위를 넓히고 싶어요.",
    )

    assert assessment["mentor_needed"] is True
    assert assessment["mentor_reason"] == "tradeoff"


def test_reply_call_order_is_assessment_rag_expert_then_facilitator():
    events = []

    class Assessment:
        def assess(self, **kwargs):
            events.append("assessment")
            return {
                "fits_current_phase": True,
                "specificity": "broad",
                "needs_clarification": True,
                "should_advance_phase": False,
                "mentor_needed": True,
                "mentor_types": ["planning"],
            }

    class Evidence:
        def retrieve(self, **kwargs):
            events.append("rag")
            assert kwargs["expert_types"] == ["planning"]
            return [{"document_id": "criteria-1", "chunk_id": "chunk-1", "text": "공고 근거"}]

    class Expert:
        def review(self, **kwargs):
            events.append("expert")
            assert kwargs["mentor_types"] == ["planning"]
            assert kwargs["evidence"][0]["text"] == "공고 근거"
            return {
                "planning": "'여러 문제'에는 우선 해결할 발생 상황 하나가 빠져 있습니다.",
                "development": "",
            }

    state = _start()

    def facilitator(prompt):
        events.append("facilitator")
        assert "우선 해결할 발생 상황 하나" in prompt
        assert '"mentor_needed": true' in prompt
        return _response(
            phase="problem_definition",
            assessment={
                "fits_current_phase": True,
                "specificity": "broad",
                "needs_clarification": True,
                "should_advance_phase": False,
                "mentor_needed": True,
                "mentor_types": ["planning"],
            },
        )

    state = reply_to_session(
        state,
        answer="여러 문제가 있습니다.",
        llm_call=facilitator,
        answer_assessment_provider=Assessment(),
        evidence_provider=Evidence(),
        expert_review_provider=Expert(),
    )

    assert events == ["assessment", "rag", "expert", "facilitator"]
    post_messages = state["messages"][-1]["structured"]["ui_message"]["post_answer_mentor_messages"]
    assert post_messages[0]["expert_type"] == "planning"


def test_next_question_can_attach_pre_answer_expert_after_facilitator():
    events = []

    class Assessment:
        def assess(self, **kwargs):
            events.append("assessment")
            return {
                "fits_current_phase": True,
                "specificity": "specific",
                "needs_clarification": False,
                "should_advance_phase": True,
                "mentor_needed": False,
                "mentor_reason": "none",
                "mentor_types": [],
            }

    class Evidence:
        def retrieve(self, **kwargs):
            events.append("rag")
            assert kwargs["expert_types"] == ["planning"]
            return [{"document_id": "criteria-1", "chunk_id": "chunk-1"}]

    class Guidance:
        def guide(self, **kwargs):
            events.append("guidance")
            assert kwargs["phase"] == "target_and_context"
            assert kwargs["guidance_reason"] == "important_direction"
            return {
                "planning": "수혜자와 실제 도입 주체를 나누어 선택하면 적용 구조가 선명해집니다.",
                "development": "",
            }

    state = _start()

    def facilitator(_):
        events.append("facilitator")
        return _response(
            phase="target_and_context",
            update={"problem": "출퇴근 시간대 시민의 대기 시간이 길어지는 문제"},
            pre_guidance={
                "needed": True,
                "reason": "important_direction",
                "mentor_types": ["planning"],
            },
        )

    state = reply_to_session(
        state,
        answer="출퇴근 시간대 대기 문제",
        llm_call=facilitator,
        answer_assessment_provider=Assessment(),
        evidence_provider=Evidence(),
        expert_guidance_provider=Guidance(),
    )

    assert events == ["assessment", "facilitator", "rag", "guidance"]
    ui_message = state["messages"][-1]["structured"]["ui_message"]
    assert ui_message["post_answer_mentor_messages"] == []
    assert ui_message["pre_answer_mentor_messages"][0]["expert_type"] == "planning"
