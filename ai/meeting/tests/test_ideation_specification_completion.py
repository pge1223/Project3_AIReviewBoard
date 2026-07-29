# 작성자: 용준/Claude(2026-07-30, 요청: "미확정 필드 확인 -> 필드별 논의 -> 구조화 저장 ->
#       검증 -> 사용자 확인" 흐름 추가) — make_specification_completion_node가 main_features/
#       required_data/technical_approach/mvp_scope/risks_and_mitigations/success_metrics/
#       assumptions_to_validate를 unknown으로 남기지 않고 proposed로 채우는지, 라운드
#       상한/게이트/사용자 승인 전 상태 보호가 실제로 지켜지는지 검증한다.

import json
import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_conv_problem import (  # noqa: E402
    _SPEC_FIELD_ORDER,
    _SPEC_REQUIRED_FIELDS,
    make_specification_completion_node,
    spec_gate_passed,
)


def _provisional_idea():
    return {
        "title": "직무별 AI 활용 교육 프로그램",
        "problem": "직원들이 AI 도구 활용법을 몰라 업무에 적용하지 못한다",
        "target_user": "공공기관 실무 직원",
        "solution": "직무별 맞춤 AI 활용 교육과 실습을 제공한다",
        "core_value": "직원이 AI 도구를 실제 업무에 적용하게 한다",
        "differentiation": "직무별 맞춤형 콘텐츠로 차별화한다",
    }


def _base_state(**overrides):
    state = {
        "session_id": "S-SPEC-1",
        "round": 1,
        "notice_and_criteria": {"competition_name": "테스트 공모전"},
        "provisional_idea": _provisional_idea(),
        "idea_spec": None,
        "spec_completion_rounds": {},
        "llm_calls_used": 0,
        "phase": "specification_completion",
        "selected_idea_document_id": None,
    }
    state.update(overrides)
    return state


def _value_response(*values, claim_type="expert_judgment", evidence_refs=None):
    return json.dumps(
        {
            "value": list(values),
            "claim_type": claim_type,
            "evidence_refs": evidence_refs or [],
            "reason": "테스트 근거",
        },
        ensure_ascii=False,
    )


def test_main_features_becomes_proposed_after_one_turn():
    def llm(prompt):
        assert "주요 기능" in prompt
        assert "기획위원" in prompt
        return _value_response("직무별 AI 활용 사례 추천", "단계별 교육 콘텐츠 제공")

    node = make_specification_completion_node(llm)
    result = node(_base_state())

    spec = result["idea_spec"]["main_features"]
    assert spec["status"] == "proposed"
    assert spec["value"] == ["직무별 AI 활용 사례 추천", "단계별 교육 콘텐츠 제공"]
    assert spec["source_turn_ids"]
    assert result["phase"] == "specification_completion"  # 아직 다른 unknown 필드가 남아있음


def test_required_data_uses_dev_expert_and_is_not_left_null():
    def llm(prompt):
        assert "필요한 데이터" in prompt
        assert "개발위원" in prompt
        return _value_response("직무별 업무 유형 데이터", "교육 콘텐츠 이용 기록")

    state = _base_state(idea_spec={
        "core_user_value": {"value": "v", "status": "proposed", "source_turn_ids": [], "evidence_refs": [], "updated_by": "facilitator", "updated_at": "", "expert_judgment": True},
        "differentiation": {"value": "d", "status": "proposed", "source_turn_ids": [], "evidence_refs": [], "updated_by": "facilitator", "updated_at": "", "expert_judgment": True},
        "main_features": {"value": ["f"], "status": "proposed", "source_turn_ids": [], "evidence_refs": [], "updated_by": "planning_expert", "updated_at": "", "expert_judgment": True},
        "required_data": {"value": None, "status": "unknown", "source_turn_ids": [], "evidence_refs": [], "updated_by": "facilitator", "updated_at": "", "expert_judgment": False},
        "technical_approach": {"value": None, "status": "unknown", "source_turn_ids": [], "evidence_refs": [], "updated_by": "facilitator", "updated_at": "", "expert_judgment": False},
        "mvp_scope": {"value": None, "status": "unknown", "source_turn_ids": [], "evidence_refs": [], "updated_by": "facilitator", "updated_at": "", "expert_judgment": False},
        "risks_and_mitigations": {"value": None, "status": "unknown", "source_turn_ids": [], "evidence_refs": [], "updated_by": "facilitator", "updated_at": "", "expert_judgment": False},
        "success_metrics": {"value": None, "status": "unknown", "source_turn_ids": [], "evidence_refs": [], "updated_by": "facilitator", "updated_at": "", "expert_judgment": False},
        "assumptions_to_validate": {"value": None, "status": "unknown", "source_turn_ids": [], "evidence_refs": [], "updated_by": "facilitator", "updated_at": "", "expert_judgment": False},
    })
    node = make_specification_completion_node(llm)
    result = node(state)

    spec = result["idea_spec"]["required_data"]
    assert spec["status"] == "proposed"
    assert spec["value"] is not None
    assert spec["value"] != []


def test_document_fact_with_valid_ref_marks_not_expert_judgment_and_stores_chunk_id():
    def evidence_lookup(persona_id, query, **kwargs):
        return [{"chunk_id": "chk_1", "document_role": "criteria", "text": "공모전은 실증 데이터 활용을 우대한다."}]

    def llm(prompt):
        assert "E1" in prompt or "ref" in prompt
        return _value_response("공모전 우대 항목에 맞춘 데이터 활용", claim_type="document_fact", evidence_refs=["E1"])

    node = make_specification_completion_node(llm, evidence_lookup)
    result = node(_base_state())
    spec = result["idea_spec"]["main_features"]
    assert spec["status"] == "proposed"
    assert spec["expert_judgment"] is False
    assert spec["evidence_refs"] == ["chk_1"]


def test_document_fact_without_matching_ref_is_downgraded_to_expert_judgment():
    def evidence_lookup(persona_id, query, **kwargs):
        return [{"chunk_id": "chk_1", "document_role": "criteria", "text": "무관한 조항."}]

    def llm(prompt):
        return _value_response("근거 없이 주장", claim_type="document_fact", evidence_refs=["E99"])

    node = make_specification_completion_node(llm, evidence_lookup)
    result = node(_base_state())
    spec = result["idea_spec"]["main_features"]
    assert spec["expert_judgment"] is True
    assert spec["evidence_refs"] == []


def test_round_cap_generates_fallback_proposed_value_after_two_failed_rounds():
    def broken_llm(prompt):
        return "이것은 JSON이 아닙니다"

    node = make_specification_completion_node(broken_llm)

    state = _base_state()
    result1 = node(state)
    assert result1["idea_spec"]["main_features"]["status"] == "unknown"
    assert result1["phase"] == "specification_completion"
    assert result1["spec_completion_rounds"]["main_features"] == 1

    state2 = {**state, "idea_spec": result1["idea_spec"], "spec_completion_rounds": result1["spec_completion_rounds"]}
    result2 = node(state2)
    spec = result2["idea_spec"]["main_features"]
    assert spec["status"] == "proposed"  # 2회 상한 도달 -> 강제로 proposed 확정(요청 8번)
    assert spec["value"]
    assert result2["spec_completion_rounds"]["main_features"] == 2


def test_already_proposed_field_is_not_reprocessed_or_overwritten():
    proposed_spec = {
        "value": ["기존 값"], "status": "proposed", "source_turn_ids": ["MSG-old"],
        "evidence_refs": [], "updated_by": "planning_expert", "updated_at": "t0", "expert_judgment": True,
    }
    idea_spec = {
        "core_user_value": {**proposed_spec, "value": "v"},
        "differentiation": {**proposed_spec, "value": "d"},
        "main_features": proposed_spec,
        **{f: {"value": None, "status": "unknown", "source_turn_ids": [], "evidence_refs": [], "updated_by": "facilitator", "updated_at": "", "expert_judgment": False} for f in _SPEC_FIELD_ORDER[1:]},
    }

    def llm(prompt):
        assert "주요 기능" not in prompt  # main_features는 이미 proposed라 다시 논의되지 않는다
        return _value_response("필요한 데이터 값")

    node = make_specification_completion_node(llm)
    result = node(_base_state(idea_spec=idea_spec))
    assert result["idea_spec"]["main_features"] == proposed_spec


def test_gate_blocks_until_all_required_fields_non_unknown():
    empty_field = {"value": None, "status": "unknown", "source_turn_ids": [], "evidence_refs": [], "updated_by": "facilitator", "updated_at": "", "expert_judgment": False}
    idea_spec = {field: {**empty_field} for field in _SPEC_REQUIRED_FIELDS}
    assert spec_gate_passed(idea_spec) is False

    for field in _SPEC_REQUIRED_FIELDS:
        idea_spec[field]["status"] = "proposed"
        idea_spec[field]["value"] = "x"
    assert spec_gate_passed(idea_spec) is True


def test_gate_passes_through_legacy_sessions_without_idea_spec():
    assert spec_gate_passed(None) is True


def test_node_never_sets_user_confirmed_or_idea_locked():
    def llm(prompt):
        return _value_response("초안")

    node = make_specification_completion_node(llm)
    result = node(_base_state())
    assert "user_confirmed" not in result
    assert "idea_locked" not in result
    assert "selected_idea" not in result


def test_all_seven_fields_reach_proposed_across_repeated_turns():
    """요청: 필수 E2E 성공 조건 — 7개 필드 모두 최소 proposed 이상, value가 비어있지 않음."""

    def llm(prompt):
        return _value_response("초안 항목 1", "초안 항목 2")

    node = make_specification_completion_node(llm)
    state = _base_state()
    seen_phases = []
    for _ in range(len(_SPEC_FIELD_ORDER) * 2):  # 여유 있게 반복(단일 위원 필드는 1콜, 공동 필드는 2콜)
        result = node(state)
        seen_phases.append(result["phase"])
        state = {
            **state,
            "idea_spec": result["idea_spec"],
            "spec_completion_rounds": result["spec_completion_rounds"],
            "phase": result["phase"],
        }
        if result["phase"] == "idea_validation":
            break

    assert state["phase"] == "idea_validation"
    for field in _SPEC_FIELD_ORDER:
        spec = state["idea_spec"][field]
        assert spec["status"] == "proposed", field
        assert spec["value"], field
        assert spec["source_turn_ids"], field
    assert spec_gate_passed(state["idea_spec"]) is True
