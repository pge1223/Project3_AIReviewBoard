# 작성자: 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편)
# 목적: discovery(아이디어 발굴) 모드가 candidate_generation 이전에 거치는 신규 단계
#       (problem_discovery -> problem_focus_selection -> problem_definition -> idea_divergence
#       -> idea_conflict_and_merge -> [candidate_planning/candidate_feasibility(기존 재사용)]
#       -> provisional_selection -> idea_validation -> concept_confirmation)의 단계 전환
#       가드와 사용자 필수 시나리오 8개를 검증한다.
#       기존 test_ideation_discovery_graph.py::DiscoveryScriptedLLM/_start_discovery를
#       재사용해 "1번" 문제 영역 선택까지는 동일한 stub 응답으로 진행하고, 이 파일은 라운드
#       상한/최소 조건 미충족/사전 잠금 콘텐츠 가드처럼 그 표준 stub이 다루지 않는 경로만
#       전용 stub으로 검증한다.
# import: 표준 라이브러리 json/sys/pathlib, pytest; ai/meeting/graph 패키지, 같은 tests
#         패키지의 test_ideation_discovery_graph.

import json
import sys
from pathlib import Path

import pytest

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph import (  # noqa: E402
    reply_ideation_conversation,
    start_ideation_conversation,
)
from graph.ideation_conv_nodes import _select_next_issue_family  # noqa: E402
from graph.ideation_conv_problem import (  # noqa: E402
    _route_after_conflict_merge,
    _route_after_idea_validation,
    make_planning_validation_node,
    make_technical_validation_node,
)
from graph.ideation_conv_run import _guard_pre_lock_messages, finalize_ideation_conversation  # noqa: E402
from graph.ideation_conv_state import (  # noqa: E402
    MAX_CONFLICT_ROUNDS,
    MIN_SOLUTION_DIRECTIONS,
    active_stage_for,
    apply_user_answer,
    contains_pre_lock_banned_content,
    critique_count,
    existing_limitations_defined,
    initial_conv_state,
    meets_conflict_and_merge_min_conditions,
    merge_or_revision_count,
    planning_validation_completed,
    problem_defined,
    ready_for_concept_confirmation,
    request_finalize,
    solution_direction_count,
    technical_validation_completed,
    target_user_defined,
)

from test_ideation_discovery_graph import (  # noqa: E402
    NOTICE_AND_CRITERIA,
    DiscoveryScriptedLLM,
    _start_discovery,
)


# ---------------------------------------------------------------------------
# 결정론적 파생 함수 단위 테스트 — 요청 5번("LLM이 boolean이나 count를 임의로 직접
# 결정하도록 만들지 마세요")을 상태 데이터만으로 검증한다(LLM 호출 없음).
# ---------------------------------------------------------------------------


def test_problem_defined_helpers_read_from_problem_definition_fields():
    state = {"problem_definition": None}
    assert problem_defined(state) is False
    assert target_user_defined(state) is False
    assert existing_limitations_defined(state) is False

    state = {
        "problem_definition": {
            "problem": "국민이 필요한 서비스를 찾기 어렵다",
            "target_user": "",
            "existing_limitations": "기존 검색은 키워드 기반이라 한계가 있다",
        }
    }
    assert problem_defined(state) is True
    assert target_user_defined(state) is False  # 빈 문자열은 정의되지 않은 것으로 취급.
    assert existing_limitations_defined(state) is True


def test_solution_direction_count_ignores_dropped_and_merged():
    state = {
        "solution_directions": [
            {"direction_id": "d1", "status": "active"},
            {"direction_id": "d2", "status": "dropped"},
            {"direction_id": "d3", "status": "merged"},
            {"direction_id": "d4", "status": "active"},
        ]
    }
    assert solution_direction_count(state) == 2


def test_critique_and_merge_counts_come_from_idea_evolution_records_only():
    state = {
        "idea_evolution": [
            {"action_type": "divergence"},
            {"action_type": "critique"},
            {"action_type": "critique"},
            {"action_type": "merge"},
            {"action_type": "validation"},
        ]
    }
    assert critique_count(state) == 2
    assert merge_or_revision_count(state) == 1


def _materialized_min_condition_state():
    return {
        "solution_directions": [
            {"direction_id": "d0", "status": "merged"},
            {"direction_id": "d1", "status": "active"},
            {"direction_id": "d2", "status": "active"},
            {"direction_id": "d3", "status": "active", "parent_direction_ids": ["d0"]},
        ],
        "idea_evolution": [
            {"action_type": "critique"},
            {
                "action_type": "merge",
                "changed_by": "planning_expert",
                "target_direction_ids": ["d0"],
                "result_direction_id": "d3",
            },
        ],
        "messages": [
            {"speaker_id": "dev_expert", "message_type": "disagreement"},
            {"speaker_id": "planning_expert", "message_type": "opinion"},
            {"speaker_id": "ideation_facilitator", "message_type": "summary"},
        ],
    }


def test_meets_conflict_and_merge_min_conditions_requires_all_three():
    base = _materialized_min_condition_state()
    assert meets_conflict_and_merge_min_conditions(base) is True

    only_two_directions = {
        **base,
        "solution_directions": [
            base["solution_directions"][0],
            base["solution_directions"][1],
            base["solution_directions"][3],
        ],
    }
    assert meets_conflict_and_merge_min_conditions(only_two_directions) is False

    no_critique = {**base, "idea_evolution": base["idea_evolution"][1:]}
    assert meets_conflict_and_merge_min_conditions(no_critique) is False

    no_merge_or_revision = {**base, "idea_evolution": base["idea_evolution"][:1]}
    assert meets_conflict_and_merge_min_conditions(no_merge_or_revision) is False


def test_ready_for_concept_confirmation_requires_both_validation_sections():
    assert ready_for_concept_confirmation({"validation_result": None}) is False
    assert (
        ready_for_concept_confirmation({"validation_result": {"planning": {"a": "b"}, "technical": None}}) is False
    )
    assert (
        ready_for_concept_confirmation(
            {"validation_result": {"planning": {"a": "b"}, "technical": {"c": "d"}}}
        )
        is True
    )


def test_contains_pre_lock_banned_content_detects_document_drafting_language():
    assert contains_pre_lock_banned_content("이제 신청서 작성 방법을 안내하겠습니다") is True
    assert contains_pre_lock_banned_content("사업계획서 목차는 다음과 같습니다") is True
    assert contains_pre_lock_banned_content("이 문제를 해결하는 방향을 더 찾아봅시다") is False
    assert contains_pre_lock_banned_content(None) is False


def test_contains_pre_lock_banned_content_distinguishes_review_question_from_drafting_action():
    """후속 요청 3번 — "idea_locked=False라는 이유만으로 모든 신청서/사업계획서 키워드
    발화를 교체하지 마세요. 검토 질문과 실제 문서 작성 요청을 구분하세요." 우연히 문서
    명사구를 포함하지만 실제로는 검토 질문인 발화는 차단하지 않는다."""
    # "사업계획서 작성"이 부분 문자열로 들어있지만, 질문 억양(?)과 검토 표현이 있어
    # 실제로는 검토 질문이다 — 차단하지 않는다.
    assert contains_pre_lock_banned_content("이 방향이 사업계획서 작성 관점에서 설득력이 있는가?") is False
    assert contains_pre_lock_banned_content("신청서 작성 항목과 비교했을 때 이 아이디어가 충분한가요?") is False
    # 명확한 작성 행위 표현은 여전히 차단한다.
    assert contains_pre_lock_banned_content("이제 사업계획서 작성을 시작하겠습니다") is True


def test_contains_pre_lock_banned_content_phase_aware_for_validation_stage():
    """idea_validation/concept_confirmation처럼 문서 종류를 언급하는 것 자체가 자연스러운
    검증 단계에서는, 명확한 작성 행위 마커가 없으면 단순 언급만으로 차단하지 않는다 —
    반대로 그 외 단계(예: expert_discussion)에서는 기존처럼 보수적으로 차단한다."""
    mention_only = "이 아이디어는 사업계획서 작성 항목들을 대부분 충족합니다"

    assert contains_pre_lock_banned_content(mention_only, phase="idea_validation") is False
    assert contains_pre_lock_banned_content(mention_only, phase="awaiting_concept_confirmation") is False
    # 검증 단계가 아닌 일반 논의 단계에서는 여전히 보수적으로 차단한다.
    assert contains_pre_lock_banned_content(mention_only, phase="expert_discussion") is True
    assert contains_pre_lock_banned_content(mention_only, phase=None) is True

    # 검증 단계라도 명확한 작성 행위 마커가 있으면 차단한다.
    drafting_action = "사업계획서 작성을 시작하겠습니다"
    assert contains_pre_lock_banned_content(drafting_action, phase="idea_validation") is True


def test_pre_lock_guard_discovery_mode_blocks_only_before_lock_and_outside_validation_stage():
    """discovery 모드 — idea_validation 단계에서는 검토성 문서 언급을 통과시키고,
    expert_discussion(확정 이후에도 idea_locked=False로 남는 refinement 진입 초반과 달리,
    discovery는 concept_confirmation에서 확정되므로 idea_locked=True인 뒤에는 애초에
    가드가 동작하지 않는다) 단계에서는 명확한 작성 행위만 차단한다."""
    baseline = [_message("user", "1번")]

    validation_mention = [
        _message(
            "dev_expert",
            "이 아이디어는 사업계획서 작성 항목들을 대부분 충족합니다.",
        )
    ]
    state = {
        "session_id": "DISC-GUARD",
        "phase": "idea_validation",
        "idea_locked": False,
        "messages": baseline + validation_mention,
    }
    guarded = _guard_pre_lock_messages(state, baseline_message_count=len(baseline))
    assert guarded["messages"][1]["content"] == "이 아이디어는 사업계획서 작성 항목들을 대부분 충족합니다."

    drafting_action = [_message("ideation_facilitator", "이제 사업계획서 작성을 시작하겠습니다.")]
    state2 = {
        "session_id": "DISC-GUARD-2",
        "phase": "idea_validation",
        "idea_locked": False,
        "messages": baseline + drafting_action,
    }
    guarded2 = _guard_pre_lock_messages(state2, baseline_message_count=len(baseline))
    assert "사업계획서 작성을 시작" not in guarded2["messages"][1]["content"]


def test_pre_lock_guard_refinement_mode_blocks_before_finalize_regardless_of_review_mentions():
    """refinement 모드 — idea_locked=False가 finalize 전까지 계속 유지되므로, 명확한 작성
    행위 발화는 어느 단계(phase="expert_discussion")든 계속 차단된다."""
    baseline = [_message("user", "동네 가게 챗봇")]
    drafting_action = [
        _message("dev_expert", "신청서 작성 방법을 지금 안내해 드리겠습니다.")
    ]
    state = {
        "session_id": "REF-GUARD",
        "phase": "expert_discussion",
        "idea_locked": False,
        "messages": baseline + drafting_action,
    }
    guarded = _guard_pre_lock_messages(state, baseline_message_count=len(baseline))
    assert "신청서 작성 방법을 지금 안내" not in guarded["messages"][1]["content"]

    # 순수 검토 질문(반론·검증)은 refinement 단계에서도 차단되지 않는다.
    review_question = [_message("planning_expert", "신청서 작성 항목과 비교했을 때 이 방향이 충분한가요?")]
    state2 = {
        "session_id": "REF-GUARD-2",
        "phase": "expert_discussion",
        "idea_locked": False,
        "messages": baseline + review_question,
    }
    guarded2 = _guard_pre_lock_messages(state2, baseline_message_count=len(baseline))
    assert guarded2["messages"][1]["content"] == "신청서 작성 항목과 비교했을 때 이 방향이 충분한가요?"


# ---------------------------------------------------------------------------
# 시나리오 1 — 초기 문제 영역 선택 직후에도 idea_locked=False.
# ---------------------------------------------------------------------------


def test_scenario1_initial_problem_focus_selection_does_not_lock_idea():
    llm = DiscoveryScriptedLLM()
    state = start_ideation_conversation(
        session_id="SCN-1",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": ""},
        llm_call=llm,
    )
    assert state["phase"] == "awaiting_problem_focus_selection"
    assert state["idea_locked"] is False

    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["problem_focus"]
    assert state["idea_locked"] is False
    assert state["selected_idea"] is None


def test_single_turn_pacing_does_not_stop_discovery_before_developer_critique():
    """아바타 화면이 보내는 single_turn=true는 legacy expert_discussion의 발언 페이싱
    옵션이다. discovery의 정지 없는 문제 정의→발산→반론·결합 경로에는 적용되면 안 된다."""
    llm = DiscoveryScriptedLLM()
    state = start_ideation_conversation(
        session_id="SCN-SINGLE-TURN",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": ""},
        llm_call=llm,
    )
    baseline = len(state["messages"])

    state = reply_ideation_conversation(
        previous_state=state,
        user_message="1번",
        llm_call=llm,
        stop_after_expert_turn=True,
    )

    new_messages = state["messages"][baseline:]
    speakers = [message["speaker_id"] for message in new_messages]
    assert "dev_expert" in speakers
    assert speakers.index("dev_expert") > speakers.index("planning_expert")
    assert any(
        message["speaker_id"] == "planning_expert" and message["message_type"] == "opinion"
        for message in new_messages[speakers.index("dev_expert") + 1 :]
    )
    assert any(
        message["speaker_id"] == "ideation_facilitator" and message["message_type"] == "summary"
        for message in new_messages[speakers.index("dev_expert") + 1 :]
    )
    assert critique_count(state) >= 1
    assert merge_or_revision_count(state) >= 1
    # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — 결합 조건 충족 후에는 더 이상
    # candidate_planning/candidate_feasibility를 거쳐 awaiting_candidate_selection에서
    # 멈추지 않는다. provisional_from_merge가 결합 결과를 곧바로 검증 대상으로 채택해
    # validate_planning까지 이어지고, 그 기획위원 발언에서 단일 턴 페이싱이 멈춘다
    # (idea_validation도 expert_discussion과 같은 페이싱 정지 지점을 공유한다,
    # ideation_conv_run.py::_drive_graph 참고).
    assert state["phase"] == "idea_validation"


# ---------------------------------------------------------------------------
# 시나리오 2 — problem_definition 완료 전에는 idea_divergence로 넘어가지 않는다.
# problem_definition 노드 자체가 검증 실패 시 phase="failed"로 멈추므로(빈 필드를 만들지
# 않는다), 이 가드는 "완료되지 않은 정의로는 발산 단계 진입 조건이 성립하지 않는다"는
# 의미를 problem_defined() 등 파생 함수로 직접 검증한다.
# ---------------------------------------------------------------------------


def test_scenario2_incomplete_problem_definition_blocks_divergence_readiness():
    incomplete_state = {
        "problem_definition": {"problem": "국민이 서비스를 찾기 어렵다", "target_user": "", "existing_limitations": ""}
    }
    assert problem_defined(incomplete_state) is True
    assert target_user_defined(incomplete_state) is False
    assert existing_limitations_defined(incomplete_state) is False
    # 세 조건이 모두 참이어야 문제 정의가 완료된 것으로 다룬다(요청 5번 예시 상태값 의미
    # 보존) — 하나라도 비어 있으면 아직 발산으로 넘어갈 준비가 안 된 것이다.
    ready = problem_defined(incomplete_state) and target_user_defined(incomplete_state) and existing_limitations_defined(
        incomplete_state
    )
    assert ready is False


# ---------------------------------------------------------------------------
# 시나리오 3·4 — 해결 방향 3개 미만이거나 반론/수정이 없으면 candidate_generation(후보
# 압축)으로 자동 진행하지 않고, 라운드 상한에서 사용자에게 결합/추가/폐기/검증진행을
# 요청한다.
# ---------------------------------------------------------------------------


class _NeverSatisfiesConflictConditionsLLM(DiscoveryScriptedLLM):
    """idea_divergence는 정상적으로 3개를 만들지만, idea_conflict_and_merge는 매 라운드
    critique만 반복하고(merge/revision 없음) 절대 최소 조건을 채우지 못하는 stub —
    MAX_CONFLICT_ROUNDS에 도달해도 자동으로 candidate_generation에 못 간다는 것을
    검증하기 위함이다."""

    def __call__(self, prompt: str) -> str:
        if "[반론·결합 규칙]" in prompt:
            return json.dumps(
                {
                    "round_events": [
                        {
                            "issued_by": "dev_expert",
                            "action_type": "critique",
                            "target_direction_ids": ["direction_1"],
                            "spoken_text": "아직 실현 가능성이 불확실합니다.",
                            "detail": "상세",
                        },
                        {
                            "issued_by": "planning_expert",
                            "action_type": "agreement",
                            "target_direction_ids": ["direction_1"],
                            "spoken_text": "확인하겠습니다.",
                            "detail": "상세",
                        },
                    ],
                    "updated_direction_status": [],
                    "resulting_direction": None,
                },
                ensure_ascii=False,
            )
        return super().__call__(prompt)


class _RevisionConflictLLM(DiscoveryScriptedLLM):
    def __call__(self, prompt: str) -> str:
        if "[반론·결합 규칙]" in prompt:
            return json.dumps(
                {
                    "round_events": [
                        {
                            "issued_by": "dev_expert",
                            "action_type": "critique",
                            "target_direction_ids": ["direction_1"],
                            "spoken_text": "현장 중심 방식은 장소와 시간 제약이 큽니다.",
                            "detail": "현장 운영 제약",
                        },
                        {
                            "issued_by": "planning_expert",
                            "action_type": "revision",
                            "target_direction_ids": ["direction_1"],
                            "spoken_text": "온라인 지역 문제 제안형 체험으로 수정하겠습니다.",
                            "detail": "온라인 방식으로 수정",
                        },
                    ],
                    "resulting_direction": {
                        "direction_id": "direction_revised_1",
                        "title": "지역 문제 연계형 온라인 AI 시민 체험",
                        "core_principle": "시민이 지역 문제를 제안하고 AI 해결 과정을 온라인으로 체험",
                        "mechanism": "온라인 제안과 AI 설명을 결합",
                        "target_user_fit": "현장 방문이 어려운 시민도 참여 가능",
                        "parent_direction_ids": ["direction_1"],
                        "strengths": ["현장 방문 없이 참여 가능"],
                        "open_assumptions": ["시민 참여 의향"],
                    },
                },
                ensure_ascii=False,
            )
        return super().__call__(prompt)


def test_revision_supersedes_original_and_creates_new_active_direction():
    llm = _RevisionConflictLLM()
    state = start_ideation_conversation(
        session_id="REVISION-MATERIALIZED",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": ""},
        llm_call=llm,
    )
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)

    directions = {direction["direction_id"]: direction for direction in state["solution_directions"]}
    assert directions["direction_1"]["status"] == "superseded"
    assert directions["direction_revised_1"]["status"] == "active"
    assert directions["direction_revised_1"]["parent_direction_ids"] == ["direction_1"]
    revision = next(record for record in state["idea_evolution"] if record["action_type"] == "revision")
    assert revision["changed_by"] == "planning_expert"
    assert revision["before"][0]["direction_id"] == "direction_1"
    assert revision["after"]["direction_id"] == "direction_revised_1"
    # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — 조건 충족 즉시
    # provisional_from_merge가 결합 결과(마지막 active 방향)를 카드 선택 없이 바로 검증
    # 대상으로 채택하고, validate_planning/validate_technical까지 정지 없이 이어져
    # awaiting_concept_confirmation에서 멈춘다.
    assert state["phase"] == "awaiting_concept_confirmation"
    assert state["provisional_idea"]["source"] == "committee_merge"
    assert state["provisional_idea"]["source_direction_ids"] == ["direction_revised_1"]


def test_scenario3_4_round_cap_without_min_conditions_asks_user_instead_of_auto_advancing():
    llm = _NeverSatisfiesConflictConditionsLLM()
    state = start_ideation_conversation(
        session_id="SCN-3-4",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": ""},
        llm_call=llm,
    )
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)

    # 최소 조건(해결 방향 3개 이상은 만족하지만 merge/revision이 한 번도 없음)을 못 채운
    # 채 라운드 상한에 도달했으므로, candidate_generation으로 자동 진행하지 않고 사용자
    # 개입을 기다린다.
    assert solution_direction_count(state) >= MIN_SOLUTION_DIRECTIONS
    assert merge_or_revision_count(state) == 0
    assert meets_conflict_and_merge_min_conditions(state) is False
    assert state["conflict_round_count"] >= MAX_CONFLICT_ROUNDS
    assert state["phase"] == "awaiting_conflict_resolution"
    assert state["idea_locked"] is False
    # candidate_planning(후보 압축)이 한 번도 호출되지 않았다 — 조건 미충족 상태에서
    # 자동으로 넘어가지 않았다는 것을 프롬프트 마커로도 재확인한다.
    assert not any("[후보 생성 규칙]" in p for p in llm.captured_prompts)


def test_scenario3_4_user_can_explicitly_proceed_despite_unmet_conditions():
    """awaiting_conflict_resolution에서 사용자가 "검증 진행"을 명시하면(요청 6번 선택지
    ④), 조건 미충족이어도 사용자 판단을 존중해 후보 압축 단계로 넘어간다."""
    llm = _NeverSatisfiesConflictConditionsLLM()
    state = start_ideation_conversation(
        session_id="SCN-3-4-PROCEED",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": ""},
        llm_call=llm,
    )
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_conflict_resolution"

    state = reply_ideation_conversation(previous_state=state, user_message="이대로 검증 진행", llm_call=llm)
    # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — "이대로 검증 진행"도 더 이상
    # candidate_planning(후보 압축)을 거치지 않는다. provisional_from_merge가 현재 active
    # 방향 중 마지막 하나를 카드 없이 바로 채택하고, validate_planning/validate_technical
    # 까지 정지 없이 이어져 awaiting_concept_confirmation에서 멈춘다.
    assert state["phase"] == "awaiting_concept_confirmation"
    assert not any("[후보 생성 규칙]" in p for p in llm.captured_prompts)
    assert state["provisional_idea"]["source"] == "committee_merge"


# ---------------------------------------------------------------------------
# 시나리오 5 — 확정 전(idea_locked=False) 단계에서 LLM이 신청서/사업계획서 작성으로
# 새려고 하면 진행자가 이를 가로채 안내 문구로 교체한다.
# ---------------------------------------------------------------------------


def _message(speaker_id: str, content: str, structured: dict | None = None) -> dict:
    return {
        "message_id": f"MSG-{speaker_id}-{hash(content) & 0xFFFF}",
        "speaker_id": speaker_id,
        "speaker_name": speaker_id,
        "role": speaker_id,
        "round": 1,
        "message_type": "opinion",
        "content": content,
        "referenced_message_ids": [],
        "evidence": [],
        "created_at": "2026-07-27T00:00:00+00:00",
        "structured": structured,
    }


def test_scenario5_pre_lock_document_drafting_content_is_intercepted():
    """요청 4·5번 — idea_locked=False인 동안 LLM 발화에 신청서/사업계획서 작성 언어가
    섞이면, 그 발화를 그대로 노출하지 않고 진행자 안내 문구로 교체한다(baseline 이후에
    새로 추가된 발화만 대상, 사용자 발화는 건드리지 않는다)."""
    baseline_messages = [_message("user", "1번")]
    new_messages = [
        _message("planning_expert", "이 방향이 좋아 보입니다."),
        _message(
            "dev_expert",
            "이제 신청서 작성 방법을 안내하겠습니다. 사업계획서 목차부터 정리하죠.",
            structured={"spoken_text": "이제 신청서 작성 방법을 안내하겠습니다."},
        ),
    ]
    state = {
        "session_id": "SCN-5",
        "phase": "expert_discussion",
        "idea_locked": False,
        "messages": baseline_messages + new_messages,
    }

    guarded = _guard_pre_lock_messages(state, baseline_message_count=len(baseline_messages))

    guarded_messages = guarded["messages"]
    assert guarded_messages[0]["content"] == "1번"  # baseline(사용자 발화)은 그대로.
    assert guarded_messages[1]["content"] == "이 방향이 좋아 보입니다."  # 정상 발화도 그대로.
    redirected = guarded_messages[2]
    assert "신청서 작성" not in redirected["content"]
    assert redirected["content"] == (
        "아직 아이디어를 확정하는 단계가 아닙니다. 문서를 작성하기 전에 해결하려는 문제와 "
        "가능한 해결 방향을 더 탐색하겠습니다."
    )
    assert redirected["structured"]["pre_lock_guard_triggered"] is True


def test_pre_lock_guard_is_noop_once_idea_locked():
    """idea_locked=True(확정 이후)면 신청서/사업계획서 언급이 있어도 건드리지 않는다 —
    확정 이후에는 문서 작성 단계로 정상적으로 넘어가야 한다."""
    messages = [
        _message("user", "1번"),
        _message("ideation_facilitator", "신청서 작성 초안을 함께 준비하겠습니다."),
    ]
    state = {"session_id": "SCN-5-LOCKED", "phase": "finalizing", "idea_locked": True, "messages": messages}
    guarded = _guard_pre_lock_messages(state, baseline_message_count=1)
    assert guarded is state  # 변경 없이 그대로 반환(불필요한 복사도 하지 않는다).


# ---------------------------------------------------------------------------
# 시나리오 7 — 사용자가 명시적으로 확정해야만 idea_locked=True가 된다(그 전 모든
# 중간 단계에서는 계속 False).
# ---------------------------------------------------------------------------


def test_scenario7_only_explicit_user_confirmation_sets_idea_locked_true():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)  # awaiting_candidate_selection까지 자동 진행.
    assert state["idea_locked"] is False

    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_concept_confirmation"
    assert state["provisional_idea"] is not None
    assert state["selected_idea"] is None
    assert state["idea_locked"] is False  # 잠정 선택만으로는 잠기지 않는다.
    assert state["user_confirmed"] is False
    assert planning_validation_completed(state) is True
    assert technical_validation_completed(state) is True

    # 확정 대신 재검토를 요청하면 반론·결합 라운드로 되돌아간다. 용준/Claude(2026-07-28,
    # 요청: 카드 선택 단계 제거) — 조건이 다시 충족되면 더 이상 카드 재압축·재선택을
    # 거치지 않고, provisional_from_merge가 새로 결합된 방향을 곧바로 검증 대상으로
    # 채택해 정지 없이 awaiting_concept_confirmation까지 이어간다(같은 호출 안에서).
    state = reply_ideation_conversation(previous_state=state, user_message="다시 검토해줘", llm_call=llm)
    assert state["idea_locked"] is False
    assert state["phase"] == "awaiting_concept_confirmation"

    state = reply_ideation_conversation(previous_state=state, user_message="확정할게요", llm_call=llm)
    assert state["idea_locked"] is True
    assert state["user_confirmed"] is True
    assert state["selected_idea"] is not None


# ---------------------------------------------------------------------------
# 시나리오 8 — idea_locked=True가 된 이후에만 문서 작성(finalize/document_drafting) 단계에
# 진입할 수 있다. request_finalize는 awaiting_user_decision/discussion_complete가 아니면
# 예외를 던지므로, "잠기기 전에는 애초에 그 phase에 도달하지 않는다"는 것 자체가 순서
# 보장이다 — 이 테스트는 그 경계를 명시적으로 확인한다.
# ---------------------------------------------------------------------------


def test_scenario8_document_drafting_requires_idea_locked_first():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_concept_confirmation"
    assert state["idea_locked"] is False

    # 확정 전(awaiting_concept_confirmation)에는 request_finalize를 호출할 수 없다.
    with pytest.raises(ValueError):
        request_finalize(state)

    state = reply_ideation_conversation(previous_state=state, user_message="확정할게요", llm_call=llm)
    assert state["idea_locked"] is True
    assert state["phase"] in ("discussion_complete", "awaiting_user_decision")

    # 확정 이후에는 정상적으로 finalize 진입(phase="finalizing")이 가능하다.
    finalizing_state = request_finalize(state)
    assert finalizing_state["phase"] == "finalizing"
    assert finalizing_state["idea_locked"] is True


# ---------------------------------------------------------------------------
# apply_user_answer 전환 가드 — 새 awaiting_* phase 3개가 올바른 다음 phase로만 전이되는지
# (LLM 판단 없이 phase 값만으로 결정론적으로 결정된다는 것을 직접 확인).
# ---------------------------------------------------------------------------


def _blank_answer_message():
    return {
        "message_id": "MSG-TEST",
        "speaker_id": "user",
        "speaker_name": "사용자",
        "role": "사용자",
        "round": 1,
        "message_type": "answer",
        "content": "테스트",
        "referenced_message_ids": [],
        "evidence": [],
        "created_at": "2026-07-27T00:00:00+00:00",
        "structured": None,
    }


@pytest.mark.parametrize(
    "prev_phase,expected_next_phase",
    [
        ("awaiting_problem_focus_selection", "problem_focus_selection"),
        ("awaiting_conflict_resolution", "conflict_resolution"),
        ("awaiting_concept_confirmation", "concept_confirmation"),
    ],
)
def test_apply_user_answer_transitions_new_awaiting_phases_deterministically(prev_phase, expected_next_phase):
    state = initial_conv_state("APPLY-TEST", NOTICE_AND_CRITERIA, {"description": ""})
    state = {**state, "phase": prev_phase}
    next_state = apply_user_answer(state, _blank_answer_message())
    assert next_state["phase"] == expected_next_phase


def test_initial_conv_state_discovery_mode_starts_at_problem_discovery_with_idea_unlocked():
    state = initial_conv_state("INIT-TEST", NOTICE_AND_CRITERIA, {"description": ""})
    assert state["ideation_mode"] == "discovery"
    assert state["phase"] == "problem_discovery"
    assert state["idea_locked"] is False
    assert state["problem_areas"] == []
    assert state["solution_directions"] == []
    assert state["provisional_idea"] is None


def test_initial_conv_state_refinement_mode_unaffected_by_new_phases():
    state = initial_conv_state("INIT-TEST-2", NOTICE_AND_CRITERIA, {"description": "동네 가게 챗봇"})
    assert state["ideation_mode"] == "refinement"
    assert state["phase"] == "expert_discussion"  # 기존 동작 그대로 — 새 phase를 거치지 않는다.
    assert state["idea_locked"] is False


def test_problem_discovery_uses_planning_news_trends_and_preserves_external_evidence():
    class _PromptRecordingDiscoveryLLM(DiscoveryScriptedLLM):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def __call__(self, prompt):
            self.prompts.append(prompt)
            return super().__call__(prompt)

    calls = []

    def external_lookup(persona_id, query):
        calls.append({"persona_id": persona_id, "query": query})
        return {
            "external_evidence": [
                {
                    "source_id": "NAVER-ISSUE-1",
                    "document_id": "NAVER-ISSUE-1",
                    "chunk_id": "NAVER-ISSUE-1-01",
                    "title": "공공서비스 이용 불편 관련 최신 뉴스",
                    "publisher": "example.com",
                    "source_url": "https://example.com/issues/1",
                    "published_at": "2026-07-28",
                    "quote": "여러 기관에서 반복되는 이용 불편과 민원이 보도됐다.",
                    "provider": "naver_api_hub",
                }
            ],
            "used_dataset_search": False,
            "used_public_api_search": True,
            "warnings": [],
        }

    llm = _PromptRecordingDiscoveryLLM()
    state = start_ideation_conversation(
        session_id="PROBLEM-NEWS-TRENDS",
        notice_and_criteria={
            **NOTICE_AND_CRITERIA,
            "purpose": "공공서비스 접근성과 이용 편의 개선",
        },
        user_idea={"description": ""},
        llm_call=llm,
        external_evidence_lookup=external_lookup,
    )

    assert state["phase"] == "awaiting_problem_focus_selection"
    assert [call["persona_id"] for call in calls] == ["planning_expert"]
    assert "공공서비스 접근성과 이용 편의 개선" in calls[0]["query"]
    assert "최근 뉴스 이슈 문제 현황" in calls[0]["query"]
    assert len(calls[0]["query"]) <= 300
    assert "공공서비스 이용 불편 관련 최신 뉴스" in llm.prompts[0]
    assert state["external_evidence"][0]["provider"] == "naver_api_hub"
    assert state["external_evidence_meta"]["used_public_api_search"] is True
    assert "공공서비스 이용 불편 관련 최신 뉴스" not in state["messages"][0]["content"]


def test_problem_discovery_rag_path_runs_planning_then_dev_then_facilitator_and_links_only_grounded_refs():
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        if "[problem_discovery 기획위원 1회 분석]" in prompt:
            return json.dumps(
                {
                    "spoken_text": "공모 목적과 국민 체감 효과 기준을 확인했습니다.",
                    "claims": [
                        {
                            "claim_id": "planning_claim_1",
                            "text": "공모문은 대국민 체감 효과를 평가합니다.",
                            "claim_type": "document_fact",
                            "evidence_refs": ["P1"],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        if "[problem_discovery 개발위원 1회 분석]" in prompt:
            return json.dumps(
                {
                    "spoken_text": "공모 기간 안의 실증 가능성과 개인정보 위험을 검토했습니다.",
                    "claims": [
                        {
                            "claim_id": "technical_claim_1",
                            "text": "공모문은 실증 가능성을 평가합니다.",
                            "claim_type": "document_fact",
                            "evidence_refs": ["D1"],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "spoken_text": "두 관점을 종합해 문제 후보를 정리했습니다.",
                "problem_areas": [
                    {
                        "area_id": "area_1",
                        "problem_title": "공공 AI 서비스의 낮은 체감 효과",
                        "problem_description": "국민이 실제 변화를 체감하기 어렵습니다.",
                        "affected_users": "공공서비스 이용 국민",
                        "planning_view": "대국민 체감 효과를 높일 필요가 있습니다.",
                        "technical_view": "기간 내 실증 범위를 정해야 합니다.",
                        "evidence_refs": ["P1", "D1", "X9"],
                    },
                    {
                        "area_id": "area_2",
                        "problem_title": "개인정보 위험으로 인한 도입 지연",
                        "problem_description": "안전한 데이터 활용 기준이 불명확합니다.",
                        "affected_users": "공공기관 담당자",
                        "planning_view": "전문가 판단: 정책 수용성 검토가 필요합니다.",
                        "technical_view": "전문가 판단: 최소 수집 설계가 필요합니다.",
                        "evidence_refs": [],
                    },
                ],
            },
            ensure_ascii=False,
        )

    calls = []

    def lookup(persona_id, query, runtime_scope=None):
        calls.append((persona_id, query, runtime_scope))
        suffix = "planning" if persona_id == "planning_expert" else "technical"
        return [
            {
                "ref": "E1",
                "chunk_id": f"chunk-{suffix}",
                "document_id": f"doc-{suffix}",
                "document_name": f"{suffix}.pdf",
                "document_role": "criteria",
                "source_type": "criteria",
                "page": 3,
                "section": "평가기준",
                "quote": "공모문은 대국민 체감 효과와 실증 가능성을 평가합니다.",
                "text": "공모문은 대국민 체감 효과와 실증 가능성을 평가합니다.",
            }
        ]

    def grounder(persona_id, claims, evidence):
        claim = claims[0]
        item = evidence[0]
        return {
            "claims": claims,
            "linked_evidence_refs": [item["chunk_id"]],
            "claim_evidence_links": [
                {
                    "claim_id": claim["claim_id"],
                    "evidence_refs": claim["evidence_refs"],
                    "chunk_ids": [item["chunk_id"]],
                }
            ],
            "unsupported_claims": [],
            "supported_claim_count": 1,
            "unsupported_claim_count": 0,
            "accepted_claim_count": 1,
            "grounded_claim_count": 1,
            "expert_judgment_count": 0,
            "linked_evidence_count": 1,
            "missing_information": [],
            "evidence_status": "grounded",
            "prompt_guard": "",
            "allow_definitive_judgment": True,
        }

    state = start_ideation_conversation(
        session_id="PROBLEM-THREE-TURNS",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": ""},
        llm_call=llm,
        evidence_lookup=lookup,
        ground_claims=grounder,
    )

    assert [call[0] for call in calls] == ["planning_expert", "dev_expert"]
    assert len(prompts) == 3
    assert [message["speaker_id"] for message in state["messages"]] == [
        "planning_expert",
        "dev_expert",
        "ideation_facilitator",
    ]
    assert state["phase"] == "awaiting_problem_focus_selection"
    assert state["messages"][0]["structured"]["problem_discovery"]["retrieval_results"][0]["page"] == 3
    assert state["messages"][1]["structured"]["problem_discovery"]["retrieval_results"][0]["section"] == "평가기준"
    assert state["problem_areas"][0]["evidence_refs"] == ["chunk-planning", "chunk-technical"]
    assert state["messages"][-1]["linked_criteria_refs"] == ["chunk-planning", "chunk-technical"]
    assert len(state["messages"][-1]["evidence"]) == 2
    assert all("X9" not in area["evidence_refs"] for area in state["problem_areas"])


def _validation_state() -> dict:
    state = initial_conv_state("VALIDATION-TEST", NOTICE_AND_CRITERIA, {"description": ""})
    return {
        **state,
        "phase": "idea_validation",
        "provisional_idea": {
            "candidate_id": "candidate_1",
            "title": "민원 안내 도우미",
            "problem": "필요한 행정 정보를 찾기 어렵다",
            "target_user": "디지털 취약계층",
            "solution": "질문 맥락에 맞는 행정 정보를 안내한다",
        },
    }


def _validation_payload(*, planning_status="passed", technical_status="passed", technical_issues=None):
    return {
        "planning": {
            "status": planning_status,
            "passed_items": ["대상 사용자가 명확함", "공모전 주제와 연결됨"],
            "issues": [],
            "revision_suggestions": [],
            "message": "대상 사용자가 명확하고 해결할 불편이 구체적입니다.",
        },
        "technical": {
            "status": technical_status,
            "passed_items": ["제한된 범위에서 프로토타입 구현 가능"],
            "issues": technical_issues or [],
            "revision_suggestions": (
                ["외부 기관 연계 없이 공개 데이터 범위로 축소"]
                if technical_issues
                else []
            ),
            "message": "공개 데이터 범위로 제한하면 기간 내 MVP 구현이 가능합니다.",
        },
        "unresolved_assumptions": [],
    }


_PLANNING_ROLE_MARKER = "당신은 AI Review Board의 기획 전문가입니다"
_DEV_ROLE_MARKER = "당신은 AI Review Board의 개발 전문가입니다"


class _ValidationLLM:
    """용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다") —
    idea_validation이 기획/개발 순차 프롬프트 2개(각자 역할 마커 포함)로 분리되면서, 이
    스텁도 어느 쪽 호출인지 역할 마커로 구분해 payload["planning"]/payload["technical"]
    (여전히 flat한 단일 섹션 dict, 기존 테스트 payload 구성 방식과 동일)을 그대로
    돌려준다 — payload 자체의 모양은 바꾸지 않아 기존 테스트의 payload 조립 코드
    (_validation_payload, payload["planning"]["claims"] = [...] 등)를 그대로 재사용한다."""

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        section = "technical" if _DEV_ROLE_MARKER in prompt else "planning"
        return json.dumps(self.payload[section], ensure_ascii=False)


def _apply_node_update(state: dict, update: dict) -> dict:
    """용준/Claude(2026-07-28) — LangGraph의 실제 병합 규칙(ideation_conv_state.py의
    Annotated[..., operator.add] 필드는 누적, 그 외는 교체)을 테스트에서 그대로 재현한다.
    idea_validation이 두 노드로 나뉘면서, 각 노드가 반환하는 부분 업데이트를 순서대로
    이어 적용해야 기존처럼 "최종 4개 메시지·validation_result" 형태를 검증할 수 있다."""
    merged = dict(state)
    for key, value in update.items():
        if key in ("messages", "idea_evolution"):
            merged[key] = list(state.get(key) or []) + list(value)
        else:
            merged[key] = value
    return merged


def _run_idea_validation(
    payload: dict,
    *,
    evidence_lookup=None,
    external_evidence_lookup=None,
    ground_claims=None,
) -> tuple[dict, _ValidationLLM]:
    """make_planning_validation_node -> make_technical_validation_node를 실제 그래프와
    같은 순서로(기획 먼저, 그 결과를 본 개발이 이어서) 실행하고 병합된 최종 state를
    반환한다. 반환값이 예전 make_idea_validation_node(...)(state) 한 번 호출의 update와
    똑같은 모양(messages 4개, validation_result 등)이 되도록 해서 기존 테스트 단언을
    최대한 그대로 재사용할 수 있게 한다."""
    llm = _ValidationLLM(payload)
    state = _validation_state()
    planning_node = make_planning_validation_node(llm, evidence_lookup, external_evidence_lookup, ground_claims=ground_claims)
    state = _apply_node_update(state, planning_node(state))
    technical_node = make_technical_validation_node(llm, evidence_lookup, external_evidence_lookup, ground_claims=ground_claims)
    state = _apply_node_update(state, technical_node(state))
    return state, llm


def _run_idea_validation_with_llm(llm) -> dict:
    """페이로드 기반이 아니라 커스텀 LLM 스텁(예: 항상 잘못된 JSON을 반환)을 두 노드에
    똑같이 재사용해야 하는 테스트용 — _run_idea_validation과 병합 로직은 같지만 LLM을
    새로 만들지 않는다."""
    state = _validation_state()
    planning_node = make_planning_validation_node(llm)
    state = _apply_node_update(state, planning_node(state))
    technical_node = make_technical_validation_node(llm)
    state = _apply_node_update(state, technical_node(state))
    return state


class _RevisionValidationDiscoveryLLM(DiscoveryScriptedLLM):
    def __call__(self, prompt):
        if "[검증 규칙]" in prompt:
            # 용준/Claude(2026-07-28, 요청: "blocking 이슈가 있을 때만 수정 단계로 복귀") —
            # severity가 major만 되어도 idea_conflict_and_merge로 돌아가던 예전 동작을 검증하던
            # 픽스처였다. 이제 major/minor는 주의사항으로만 남고 blocking만 복귀 사유이므로,
            # 이 테스트가 실제로 확인하려는 것("검증 결과가 반론·결합 단계로 제대로 전달되는가")을
            # 그대로 유지하려면 severity를 blocking으로 바꿔야 한다.
            issue = {
                "code": "scope_too_large",
                "description": "기관 전체 데이터 연계 범위가 너무 큼",
                "severity": "blocking",
            }
            payload = _validation_payload(
                planning_status="passed",
                technical_status="needs_revision",
                technical_issues=[issue],
            )
            section = "technical" if _DEV_ROLE_MARKER in prompt else "planning"
            return json.dumps(payload[section], ensure_ascii=False)
        return super().__call__(prompt)


def test_idea_validation_stores_structured_results_and_four_ordered_messages():
    update, _llm = _run_idea_validation(_validation_payload())

    assert update["phase"] == "awaiting_concept_confirmation"
    assert update["selected_idea"] is None
    assert update["idea_locked"] is False
    assert update["validation_result"]["planning"]["status"] == "passed"
    assert update["validation_result"]["technical"]["status"] == "passed"
    assert [message["speaker_id"] for message in update["messages"]] == [
        "ideation_facilitator",
        "planning_expert",
        "dev_expert",
        "ideation_facilitator",
    ]
    assert _route_after_idea_validation(update) == "confirm"


def test_idea_validation_warns_when_criteria_retrieval_is_empty(monkeypatch):
    """용준/Claude(2026-07-30, 요청 §8·14 — preflight) — notice_and_criteria가 채워진
    세션에서 evidence_lookup이 criteria 역할 청크를 하나도 못 찾으면, 조용히 target/전문가
    판단만으로 넘어가지 않고 trace에 criteria_retrieval_empty 경고를 남겨야 한다."""
    import graph.ideation_conv_problem as problem_module

    events: list[dict] = []

    def fake_trace_event(event, **fields):
        events.append({"event": event, **fields})

    monkeypatch.setattr(problem_module, "trace_event", fake_trace_event)

    def evidence_lookup(_persona_id, _query, **_kwargs):
        return [{"chunk_id": "T1", "document_role": "target", "text": "target only"}]

    _run_idea_validation(_validation_payload(), evidence_lookup=evidence_lookup)

    warning_events = [e for e in events if e["event"] == "IDEATION_CRITERIA_RETRIEVAL_EMPTY"]
    assert len(warning_events) == 2  # planning_validation + technical_validation
    assert {e["node"] for e in warning_events} == {"planning_validation", "technical_validation"}
    assert all(e["criteria_retrieval_empty"] is True for e in warning_events)


def test_idea_validation_uses_role_specific_external_evidence():
    calls = []

    def external_lookup(persona_id, query):
        calls.append({"persona_id": persona_id, "query": query})
        role_label = "기획 최신 뉴스" if persona_id == "planning_expert" else "개발 기술 뉴스"
        return {
            "external_evidence": [
                {
                    "source_id": f"NEWS-{persona_id}",
                    "document_id": f"NEWS-{persona_id}",
                    "chunk_id": f"NEWS-{persona_id}-1",
                    "title": role_label,
                    "publisher": "example.com",
                    "source_url": f"https://example.com/{persona_id}",
                    "published_at": "2026-07-28",
                    "quote": f"{role_label}의 요약입니다.",
                    "reference_only": True,
                }
            ],
            "used_dataset_search": False,
            "used_public_api_search": True,
            "warnings": [],
        }

    update, llm = _run_idea_validation(
        _validation_payload(),
        external_evidence_lookup=external_lookup,
    )

    assert [call["persona_id"] for call in calls] == ["planning_expert", "dev_expert"]
    assert "민원 안내 도우미" in calls[0]["query"]
    # 용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다") — 이제
    # 기획/개발이 각자 별도 프롬프트를 받으므로, 서로의 외부 근거가 상대방 프롬프트에 섞여
    # 들어가지 않는다(예전 결합 프롬프트 때는 llm.prompts[0] 하나에 둘 다 있었다).
    assert "기획 최신 뉴스" in llm.prompts[0]
    assert "개발 기술 뉴스" not in llm.prompts[0]
    assert "개발 기술 뉴스" in llm.prompts[1]
    assert "기획 최신 뉴스" not in llm.prompts[1]
    assert {item["title"] for item in update["messages"][1]["evidence"]} == {"기획 최신 뉴스"}
    assert {item["title"] for item in update["messages"][2]["evidence"]} == {"개발 기술 뉴스"}
    assert len(update["external_evidence"]) == 2
    assert update["external_evidence_meta"]["used_public_api_search"] is True


def test_idea_validation_retrieval_persona_matches_grounding_persona():
    """용준/Claude(2026-07-29, 요청: persona_id 불일치 버그 수정) — 실측(웹 UI): 기획위원
    검증 턴인데 RAG 검색(evidence_lookup)이 "dev_expert" role로 호출되고 있었다(바로 아래
    ground_claims는 "planning_expert"로 호출돼 서로 어긋남) — make_technical_validation_node
    코드를 복붙하면서 planning 쪽 persona_id를 안 고친 버그. 검색 role이 실제 화자와
    일치해야 criteria/target 쿼터(ai/rag/orchestration/ideation_evidence_service.py::
    _DOCUMENT_ROLE_QUOTAS)가 그 위원에게 맞게 적용된다."""
    calls: list[str] = []

    def evidence_lookup(persona_id: str, query: str, **_kwargs) -> list[dict]:
        calls.append(persona_id)
        return []

    _run_idea_validation(_validation_payload(), evidence_lookup=evidence_lookup)

    assert calls == ["planning_expert", "dev_expert"], (
        "기획 검증 노드는 planning_expert로, 개발 검증 노드는 dev_expert로 순서대로 "
        f"검색을 호출해야 하는데 실제로는 {calls}였습니다"
    )


def test_idea_validation_links_claims_to_retrieved_evidence_when_ground_claims_provided():
    """용준/Claude(2026-07-28, 요청: "위원들이 RAG를 근거로 회의를 진행" + 화면 "근거 보기"
    복구) — ground_claims가 주어지면 planning/technical claims가 실제 검색 근거(ref="E1")와
    연결·검증되어 message.linked_evidence_refs에 실제 chunk_id로 채워지는지 확인한다. 이
    필드가 비어있으면 프론트(IdeationConversationScreen.jsx::EvidenceToggle)가 근거를 아예
    렌더링하지 않는다."""
    from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl

    def _ground_claims(persona_id, claims, retrieved):
        return _ground_claims_impl(claims, retrieved)

    def _evidence_lookup(_persona_id, _query):
        return [
            {
                "chunk_id": "CHUNK-1",
                "document_id": "DOC-1",
                "document_name": "행정 서비스 안내 공모전 공고문",
                "section": "평가 기준",
                "text": "행정 정보 접근성을 개선하는 서비스를 우대한다.",
            }
        ]

    payload = _validation_payload()
    payload["planning"]["claims"] = [
        {
            "claim_id": "claim_1",
            "text": "공고문은 행정 정보 접근성을 개선하는 서비스를 우대한다고 명시한다.",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        }
    ]
    payload["technical"]["claims"] = [
        {
            "claim_id": "claim_1",
            "text": "이 정도 범위면 기간 내 프로토타입 구현이 가능해 보인다.",
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        }
    ]

    update, _llm = _run_idea_validation(
        payload,
        evidence_lookup=_evidence_lookup,
        ground_claims=_ground_claims,
    )

    planning_message, technical_message = update["messages"][1], update["messages"][2]
    assert planning_message["speaker_id"] == "planning_expert"
    assert planning_message["linked_evidence_refs"] == ["CHUNK-1"]
    assert planning_message["claims"][0]["claim_type"] == "document_fact"
    # expert_judgment claim은 문서 근거로 연결되지 않는다(evidence_refs=[]) — linked는 비어야 한다.
    assert technical_message["speaker_id"] == "dev_expert"
    assert technical_message["linked_evidence_refs"] == []
    assert technical_message["claims"][0]["claim_type"] == "expert_judgment"


def test_idea_validation_never_exposes_user_session_answer_as_grounded_evidence():
    """용준/Claude(2026-07-30, 요청: "사용자 답변은 문서 기반 grounded evidence로 사용하지
    마세요") — evidence_lookup이 사용자 채팅 답변 청크(ideation_source_type=
    user_session_answer, document_role=target)를 검색 상위에 반환해도, call_evidence_lookup
    단계에서 제외되어 message.evidence/claims/evidence_buckets 어디에도 등장하지 않아야
    한다. 선택 아이디어(target, ideation_source_type=ideation_candidate에 준하는 실제
    criteria 근거)는 그대로 인용 가능해야 한다(회귀 없음)."""
    from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl

    def _ground_claims(persona_id, claims, retrieved):
        return _ground_claims_impl(claims, retrieved)

    def _evidence_lookup(_persona_id, _query):
        return [
            {
                "chunk_id": "CHUNK-ANSWER",
                "document_id": "ideation-answer::P1::S1::MSG-1",
                "ideation_source_type": "user_session_answer",
                "document_name": "[사용자 추가 답변] 회의 답변",
                "text": "사용자가 회의 중 직접 말한 답변 원문입니다.",
            },
            {
                "chunk_id": "CHUNK-CRITERIA",
                "document_id": "DOC-CRITERIA",
                # source_type만 명시한다(document_role은 비워 둔다) — claim_grounding.py의
                # criteria_scope_overreach 검사는 document_role만 보고, 화면 버킷 분류
                # (_classify_linked_evidence_buckets)는 source_type을 우선 본다. 이 테스트가
                # 확인하려는 건 그 overreach 휴리스틱이 아니라 user_session_answer 배제이므로
                # document_role은 굳이 채우지 않는다.
                "source_type": "criteria",
                "document_name": "행정 서비스 안내 공모전 공고문",
                "section": "평가 기준",
                "text": "행정 정보 접근성을 개선하는 서비스를 우대한다.",
            },
        ]

    payload = _validation_payload()
    payload["planning"]["claims"] = [
        {
            "claim_id": "claim_1",
            "text": "공고문은 행정 정보 접근성을 개선하는 서비스를 우대한다고 명시한다.",
            "claim_type": "document_fact",
            # 검색 순서상 사용자 답변이 필터링되지 않았다면 "E1"이었을 자리 — 필터링 후에는
            # criteria 청크가 "E1"이 된다.
            "evidence_refs": ["E1"],
        }
    ]
    payload["technical"]["claims"] = [
        {"claim_id": "claim_1", "text": "구현은 가능해 보인다.", "claim_type": "expert_judgment", "evidence_refs": []}
    ]

    update, _llm = _run_idea_validation(
        payload,
        evidence_lookup=_evidence_lookup,
        ground_claims=_ground_claims,
    )

    planning_message = update["messages"][1]
    evidence_chunk_ids = {item.get("chunk_id") for item in planning_message["evidence"]}
    assert "CHUNK-ANSWER" not in evidence_chunk_ids
    assert "CHUNK-CRITERIA" in evidence_chunk_ids
    assert planning_message["linked_evidence_refs"] == ["CHUNK-CRITERIA"]
    assert "CHUNK-ANSWER" not in planning_message["reviewed_target_refs"]
    assert "CHUNK-ANSWER" not in planning_message["linked_criteria_refs"]
    assert "CHUNK-ANSWER" not in planning_message["linked_external_evidence_refs"]
    assert planning_message["linked_criteria_refs"] == ["CHUNK-CRITERIA"]


def test_idea_validation_links_rag007_external_evidence_and_excludes_uncited_items():
    from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl

    def _ground_claims(_persona_id, claims, retrieved):
        return _ground_claims_impl(claims, retrieved)

    def _external_lookup(persona_id, _query):
        return {
            "external_evidence": [
                {
                    "chunk_id": f"EXT-{persona_id}-1",
                    "document_id": f"EXT-{persona_id}",
                    "title": "공공부문 AI 안전성 가이드",
                    "publisher": "공식기관",
                    "source_url": "https://example.go.kr/ai-guide",
                    "reference_date": "2026-07-01",
                    "quote": "공공부문 AI 서비스는 개인정보 보호와 안전성 점검 절차를 마련해야 한다.",
                    "source_type": "official_report",
                    "allow_grounded_claim": True,
                },
                {
                    "chunk_id": f"UNUSED-{persona_id}-2",
                    "document_id": f"UNUSED-{persona_id}",
                    "title": "관련 없는 외부 자료",
                    "publisher": "공식기관",
                    "source_url": "https://example.go.kr/unrelated",
                    "reference_date": "2026-07-01",
                    "quote": "이 자료는 다른 주제를 다룬다.",
                    "source_type": "official_report",
                    "allow_grounded_claim": True,
                },
            ],
            "used_dataset_search": True,
            "used_public_api_search": False,
            "warnings": [],
        }

    payload = _validation_payload()
    payload["planning"]["claims"] = [
        {
            "claim_id": "claim_1",
            "text": "공공부문 AI 서비스에는 개인정보 보호와 안전성 점검 절차가 필요하다.",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        }
    ]
    payload["technical"]["claims"] = [
        {
            "claim_id": "claim_1",
            "text": "구현 범위는 단계적으로 좁힐 필요가 있다.",
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        }
    ]

    update, _llm = _run_idea_validation(
        payload,
        external_evidence_lookup=_external_lookup,
        ground_claims=_ground_claims,
    )

    planning_message = update["messages"][1]
    assert planning_message["linked_external_evidence_refs"] == ["EXT-planning_expert-1"]
    assert planning_message["linked_evidence_refs"] == ["EXT-planning_expert-1"]
    assert "UNUSED-planning_expert-2" not in planning_message["linked_external_evidence_refs"]


def test_idea_validation_does_not_ground_summary_only_external_evidence():
    from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl

    def _ground_claims(_persona_id, claims, retrieved):
        return _ground_claims_impl(claims, retrieved)

    def _external_lookup(persona_id, _query):
        return {
            "external_evidence": [
                {
                    "chunk_id": f"DPG-{persona_id}-1",
                    "document_id": f"DPG-{persona_id}",
                    "title": "DPG 소개 페이지",
                    "publisher": "디지털플랫폼정부위원회",
                    "source_url": "https://example.go.kr/dpg",
                    "reference_date": "2026-07-01",
                    "quote": "공공 AI 서비스의 확산 효과를 소개한다.",
                    "source_type": "official_page_summary",
                    "summary_only": True,
                    "allow_grounded_claim": False,
                }
            ],
            "used_dataset_search": True,
            "used_public_api_search": False,
            "warnings": [],
        }

    payload = _validation_payload()
    payload["planning"]["claims"] = [
        {
            "claim_id": "claim_1",
            "text": "공공 AI 서비스가 확산 효과를 냈다.",
            "claim_type": "document_fact",
            "evidence_refs": ["E1"],
        }
    ]

    update, _llm = _run_idea_validation(
        payload,
        external_evidence_lookup=_external_lookup,
        ground_claims=_ground_claims,
    )

    planning_message = update["messages"][1]
    assert planning_message["linked_external_evidence_refs"] == []
    assert planning_message["unsupported_claim_count"] == 1
    assert planning_message["evidence_status"] == "ungrounded"


def test_idea_validation_without_ground_claims_keeps_empty_grounding_backward_compatible():
    """ground_claims를 안 넘기면(use_rag=False 세션 등) 기존과 동일하게 grounding 없이
    동작해야 한다 — 하위 호환 확인."""
    update, _llm = _run_idea_validation(_validation_payload())
    planning_message, technical_message = update["messages"][1], update["messages"][2]
    assert planning_message["linked_evidence_refs"] == []
    assert technical_message["linked_evidence_refs"] == []


def test_idea_validation_needs_revision_and_blocking_issue_route_back_to_merge():
    issue = {
        "code": "scope_too_large",
        "description": "기관 전체 데이터 연계 범위가 너무 큼",
        "severity": "blocking",
    }
    update, _llm = _run_idea_validation(
        _validation_payload(
            planning_status="passed",
            technical_status="passed",
            technical_issues=[issue],
        )
    )

    assert update["phase"] == "idea_conflict_and_merge"
    assert update["validation_result"]["overall_status"] == "needs_revision"
    assert "기관 전체 데이터 연계 범위가 너무 큼" in update["validation_result"]["required_changes"]
    assert _route_after_idea_validation(update) == "revise"


def test_idea_validation_normalizes_missing_status_without_failing():
    payload = _validation_payload()
    payload["planning"].pop("status")
    payload["technical"].pop("status")

    update, _llm = _run_idea_validation(payload)

    assert update["phase"] == "awaiting_concept_confirmation"
    assert update["validation_result"]["planning"]["status"] == "passed_with_caution"
    assert update["validation_result"]["technical"]["status"] == "passed_with_caution"


def test_idea_validation_invalid_json_uses_safe_revision_fallback():
    class _InvalidValidationLLM:
        def __call__(self, _prompt):
            return "JSON 형식이 아닌 응답"

    update = _run_idea_validation_with_llm(_InvalidValidationLLM())

    assert update["phase"] == "idea_conflict_and_merge"
    assert update["validation_result"]["overall_status"] == "needs_revision"
    assert update["validation_result"]["planning"]["issues"][0]["code"] == "planning_validation_unavailable"
    assert [message["speaker_id"] for message in update["messages"]] == [
        "ideation_facilitator",
        "planning_expert",
        "dev_expert",
        "ideation_facilitator",
    ]


def test_idea_validation_keeps_hwpx_raw_text_only_in_evidence():
    raw_chunk = (
        "'붙임3_2026_공공기관_AI_혁신_챌린지_참가_신청_서식.hwpx'\n"
        "○\n-\n※\n<작성 요령> 지속 운영 계획을 구체적으로 작성"
    )
    payload = _validation_payload()
    payload["planning"]["message"] = raw_chunk

    def evidence_lookup(_persona_id, _query):
        return [
            {
                "document_id": "DOC-HWPX",
                "document_name": "붙임3_2026_공공기관_AI_혁신_챌린지_참가_신청_서식.hwpx",
                "chunk_id": "CHK-HWPX",
                "section": "지속 운영 계획",
                "text": "<작성 요령> 지속 운영 계획과 예산 확보 방안을 작성",
            }
        ]

    update, _llm = _run_idea_validation(payload, evidence_lookup=evidence_lookup)
    planning_message = update["messages"][1]

    assert ".hwpx" not in planning_message["content"]
    assert "<작성 요령>" not in planning_message["content"]
    assert planning_message["evidence"][0]["document_name"].endswith(".hwpx")
    assert "<작성 요령>" in planning_message["evidence"][0]["quote"]


def test_failed_validation_is_forwarded_to_conflict_merge_without_full_divergence():
    llm = _RevisionValidationDiscoveryLLM()
    state = _start_discovery(llm)
    divergence_calls_before_validation = sum(
        "[발산 규칙]" in prompt for prompt in llm.captured_prompts
    )

    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)

    # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — 카드 선택 정지점이 사라지면서,
    # 검증 실패 -> idea_conflict_and_merge 재실행 -> 조건 재충족 -> 재검증이 한 그래프
    # 호출 안에서 반복될 수 있게 됐다. 이 mock은 검증을 계속 needs_revision으로만
    # 응답하므로, MAX_VALIDATION_REVISE_ROUNDS(1)에 도달할 때까지 한 번 더 자동으로
    # 재결합 라운드를 거친 뒤(validation_revise_count) 더 자동으로 되돌리지 않고
    # awaiting_concept_confirmation에서 멈춘다(overall_status는 needs_revision 그대로
    # 노출해 사용자가 직접 확정/재검토를 선택하게 한다).
    assert state["phase"] == "awaiting_concept_confirmation"
    assert state["validation_result"]["overall_status"] == "needs_revision"
    assert state["validation_result"]["next_phase"] == "awaiting_concept_confirmation"
    assert state["validation_revise_count"] == 1
    assert "기관 전체 데이터 연계 범위가 너무 큼" in state["validation_result"]["required_changes"]
    assert sum("[발산 규칙]" in prompt for prompt in llm.captured_prompts) == divergence_calls_before_validation
    conflict_prompts = [prompt for prompt in llm.captured_prompts if "[반론·결합 규칙]" in prompt]
    assert "기관 전체 데이터 연계 범위가 너무 큼" in conflict_prompts[-1]
    assert any(
        record.get("action_type") == "validation"
        for record in state["idea_evolution"]
    )
    assert any(
        record.get("action_type") in {"revision", "merge"}
        and record.get("stage") == "idea_conflict_and_merge"
        for record in state["idea_evolution"]
    )


def test_idea_validation_stops_after_planning_turn_when_single_turn_requested():
    """용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다") —
    프론트가 항상 singleTurn=true로 보내므로(IdeationConversationScreen.jsx), 후보 확정
    직후 reply_ideation_conversation(stop_after_expert_turn=True)이 validate_planning
    실행 직후 곧바로 멈추는지 확인한다(기획위원 발언까지만, 개발위원은 아직). 이어서
    continue_ideation_validation_turn이 그 상태를 이어받아 개발위원 발언까지 만들고
    awaiting_concept_confirmation으로 넘어가는지도 함께 확인한다 — 실제 백엔드
    /reply/stream -> /continue-turn/stream 흐름을 그래프 함수 수준에서 재현한다."""
    from graph import continue_ideation_validation_turn

    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)  # awaiting_candidate_selection까지 자동 진행.

    state = reply_ideation_conversation(
        previous_state=state,
        user_message="1번",
        llm_call=llm,
        stop_after_expert_turn=True,
    )

    assert state["phase"] == "idea_validation"
    assert [m["speaker_id"] for m in state["messages"][-2:]] == ["ideation_facilitator", "planning_expert"]
    assert planning_validation_completed(state) is True
    assert technical_validation_completed(state) is False

    state = continue_ideation_validation_turn(previous_state=state, llm_call=llm)

    assert state["phase"] == "awaiting_concept_confirmation"
    assert [m["speaker_id"] for m in state["messages"][-2:]] == ["dev_expert", "ideation_facilitator"]
    assert technical_validation_completed(state) is True
    assert state["forced_next_speaker"] is None


def test_continue_ideation_validation_turn_rejects_wrong_phase_or_speaker():
    from graph import continue_ideation_validation_turn

    with pytest.raises(ValueError):
        continue_ideation_validation_turn(
            previous_state={**_validation_state(), "phase": "expert_discussion"},
            llm_call=_ValidationLLM(_validation_payload()),
        )

    state_without_planning_turn = {**_validation_state(), "messages": []}
    with pytest.raises(ValueError):
        continue_ideation_validation_turn(
            previous_state=state_without_planning_turn,
            llm_call=_ValidationLLM(_validation_payload()),
        )


# ---------------------------------------------------------------------------
# _route_after_conflict_merge 라우팅 단위 테스트 — 그래프 없이 순수 함수로 검증.
# ---------------------------------------------------------------------------


def test_route_after_conflict_merge_continues_until_cap_or_conditions_met():
    directions = [{"direction_id": f"d{i}", "status": "active"} for i in range(3)]
    not_met_state = {"solution_directions": directions, "idea_evolution": [], "conflict_round_count": 0}
    assert _route_after_conflict_merge(not_met_state) == "continue"

    capped_not_met_state = {**not_met_state, "conflict_round_count": MAX_CONFLICT_ROUNDS}
    assert _route_after_conflict_merge(capped_not_met_state) == "ask_user"

    met_state = {**_materialized_min_condition_state(), "conflict_round_count": 1}
    assert _route_after_conflict_merge(met_state) == "proceed"

    assert _route_after_conflict_merge({"phase": "failed"}) == "failed"


# ---------------------------------------------------------------------------
# 후속 요청(2026-07-27) — candidate_planning이 problem_definition/solution_directions/
# idea_evolution을 실제로 프롬프트에 압축 재료로 주입하는지(다시 처음부터 발명하지 않는지)
# 확인한다. 이전 구현은 solution_directions만 부가 정보로 넘기고 problem_definition/
# idea_evolution은 아예 넘기지 않아, 프롬프트의 1순위 지시가 여전히 "공모전 공고문만으로
# 새 아이디어 발명"이었다 — 이번 수정으로 프롬프트 자체가 두 값의 존재 여부로 압축 모드/
# 레거시 발명 모드를 명시적으로 분기한다(ideation_conv_candidate_planning.txt [모드 판단]).
# ---------------------------------------------------------------------------


def test_candidate_planning_prompt_is_grounded_in_prior_meeting_results_not_reinvented():
    """용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — candidate_planning은 더 이상
    새 discovery 플로우의 그래프 배선에서 자동 실행되지 않는다(idea_conflict_and_merge
    "proceed"는 이제 provisional_from_merge로 간다). 다만 이 노드 자체는 레거시 재개용으로
    삭제되지 않았고, "이전 회의 결과(problem_definition/solution_directions/idea_evolution)를
    실제로 압축 재료로 쓰는지"는 여전히 유효한 검증 대상이므로, 그래프를 통해 자동
    도달시키는 대신 real 회의 상태를 만든 뒤(stop_after_expert_turn=True로 idea_validation
    직전까지 진행) 노드 함수를 직접 호출해 프롬프트를 검증한다(레거시 helper 패턴,
    test_ideation_discovery_graph.py::_legacy_resolve_selection_then_ask_planning_question과
    동일한 원칙)."""
    from graph.ideation_conv_discovery import make_candidate_planning_node

    llm = DiscoveryScriptedLLM()
    state = start_ideation_conversation(
        session_id="COMPRESS-TEST",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": ""},
        llm_call=llm,
    )
    state = reply_ideation_conversation(
        previous_state=state, user_message="1번", llm_call=llm, stop_after_expert_turn=True
    )
    assert state["phase"] == "idea_validation"

    candidate_planning_result = make_candidate_planning_node(llm)(state)
    state = {**state, **candidate_planning_result}

    planning_prompts = [p for p in llm.captured_prompts if "[후보 생성 규칙 — 압축 모드" in p]
    assert planning_prompts, "candidate_planning이 압축 모드 규칙 섹션을 포함해야 한다"
    prompt = planning_prompts[0]

    # problem_definition 실제 값이 프롬프트에 주입됐다(발명이 아니라 압축 대상으로 참조).
    assert "문제 정의" in prompt or "problem_definition" in prompt
    # DiscoveryScriptedLLM의 idea_divergence stub이 만든 방향 제목이 그대로 프롬프트에
    # 전달됐다 — 프롬프트가 실제 회의 결과를 압축 재료로 받았는지 문자열 수준에서 확인한다.
    assert "방향 1" in prompt or "결합 방향" in prompt
    # idea_evolution(반론·결합 기록)도 함께 전달된다.
    assert "critique" in prompt or "merge" in prompt
    assert '"status": "merged"' in prompt
    active_ids = {
        direction["direction_id"] for direction in state["solution_directions"] if direction["status"] == "active"
    }
    assert active_ids
    assert all(
        set(candidate["source_direction_ids"]).issubset(active_ids)
        for candidate in state["idea_candidates"]
    )

    # [모드 판단] 섹션이 실제로 존재해, "solution_directions가 있으면 압축 모드를 강제한다"는
    # 지시 자체가 프롬프트에 포함돼 있는지 확인한다(단순 참고 데이터가 아니라 규칙임을 보장).
    assert "solution_directions가 비어 있지 않으면 반드시 압축 모드로 동작" in prompt


def test_candidate_planning_legacy_invention_mode_prompt_omits_compression_directive_when_no_prior_context():
    """solution_directions/problem_definition이 모두 비어 있으면(문제 발견 단계를 거치지
    않은 예외적 직접 진입) 레거시 발명 모드 규칙만 적용된다는 것을, 프롬프트 빌더 단위로
    직접 확인한다(그래프를 통해서는 이 경로에 도달하기 어렵다 — initial_conv_state가 항상
    problem_discovery부터 시작하기 때문)."""
    from prompts import build_ideation_conv_candidate_planning_prompt

    prompt = build_ideation_conv_candidate_planning_prompt({"competition_name": "데모"}, [], [], None)
    assert "solution_directions가 비어 있지 않으면 반드시 압축 모드로 동작" in prompt
    # 빈 값이면 두 후보 생성 규칙 섹션 모두 프롬프트에 존재하지만(모드 판단은 실행 시점에
    # LLM이 입력값으로 직접 판단), 압축 대상 데이터 자체는 비어 있다("null"/"[]").
    assert "<<" not in prompt  # 모든 토큰이 치환됐다.


# ---------------------------------------------------------------------------
# 후속 요청(2026-07-27) 2번 — idea_locked=True 이후 expert_discussion이 확정된 핵심 주제
# (문제/대상 사용자/핵심 가치/차별성)를 자동으로 다시 여는지 확인하고, B안(정리 전용 제한)을
# 검증한다.
# ---------------------------------------------------------------------------


def test_select_next_issue_family_skips_confirmed_core_topics_after_concept_confirmation():
    """concept_confirmation이 seed하는 resolved_topics(problem/target_user/core_value/
    differentiation)가 실제로 discussion_facilitator의 다음 쟁점 자동 선택
    (_select_next_issue_family, ideation_conv_nodes.py — expert_discussion 라운드 사이
    자동 로테이션에 쓰이는 바로 그 함수)에서 제외되는지 확인한다."""
    post_confirmation_resolved_topics = ["problem", "target_user", "core_value", "differentiation"]

    next_family = _select_next_issue_family(
        excluded_family=None,
        open_issues=[],
        resolved_issues=[],
        resolved_topics=post_confirmation_resolved_topics,
    )
    # 문제/대상 사용자/핵심 가치/차별성이 아니라, 확정 이후에도 여전히 다듬을 여지가 있는
    # 구현 세부사항(mvp)이 다음 의제로 선택된다.
    assert next_family == "mvp"
    assert next_family not in post_confirmation_resolved_topics


def test_select_next_issue_family_would_reopen_core_topics_without_seeding():
    """대조군 — resolved_topics를 seed하지 않으면(이전 구현) "problem"이 여전히 다음
    의제로 선택될 수 있었다는 것을 보여준다(회귀 방지용 대조 테스트)."""
    next_family = _select_next_issue_family(
        excluded_family=None, open_issues=[], resolved_issues=[], resolved_topics=[]
    )
    assert next_family == "problem"


def test_concept_confirmation_seeds_resolved_topics_with_confirmed_core():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_concept_confirmation"
    assert "problem" not in (state.get("resolved_topics") or [])

    confirmed_idea_snapshot = dict(state["provisional_idea"])
    state = reply_ideation_conversation(previous_state=state, user_message="확정할게요", llm_call=llm)
    assert state["idea_locked"] is True
    for topic in ("problem", "target_user", "core_value", "differentiation"):
        assert topic in state["resolved_topics"]

    # selected_idea 객체 자체는 라운드테이블 한 라운드가 실행된 뒤에도 확정 시점 값과
    # 완전히 동일하다(요청 2번 — 확정한 아이디어와 최종 결과가 달라지면 안 된다). 어떤
    # discussion/facilitator/canvas_update 노드도 selected_idea를 재할당하지 않는다
    # (ideation_conv_nodes.py 전체에서 "selected_idea": state.get("selected_idea")로만
    # 참조된다 — 코드 검토로 확인, 이 테스트는 그 불변성을 회귀 테스트로 고정한다).
    assert state["selected_idea"] == confirmed_idea_snapshot


# ---------------------------------------------------------------------------
# 후속 요청(2026-07-27) 4번 — 구조화된 사용자 액션(action_code/action_payload) 지원.
# action code가 자연어 키워드 판정보다 우선 사용되고, 없으면 기존 자연어 파싱으로
# 폴백하는지 확인한다.
# ---------------------------------------------------------------------------


def test_action_code_select_problem_focus_bypasses_text_parsing():
    llm = DiscoveryScriptedLLM()
    state = start_ideation_conversation(
        session_id="ACTION-1", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea={"description": ""}, llm_call=llm
    )
    assert state["phase"] == "awaiting_problem_focus_selection"

    # 자연어로는 절대 "1번"으로 해석되지 않을 문장이지만, action_code가 우선 적용된다.
    state = reply_ideation_conversation(
        previous_state=state,
        user_message="음... 뭘 골라야 할지 모르겠지만 일단 이걸로 할게요",
        llm_call=llm,
        action_code="select_problem_focus",
        action_payload={"indices": [1]},
    )
    assert state["problem_focus"][0]["area_id"] == "area_1"


def test_action_code_combine_problem_focus_selects_two_areas():
    llm = DiscoveryScriptedLLM()
    state = start_ideation_conversation(
        session_id="ACTION-2", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea={"description": ""}, llm_call=llm
    )
    state = reply_ideation_conversation(
        previous_state=state,
        user_message="",
        llm_call=llm,
        action_code="combine_problem_focus",
        action_payload={"indices": [1, 2]},
    )
    assert {a["area_id"] for a in state["problem_focus"]} == {"area_1", "area_2"}


def test_action_code_with_invalid_payload_raises_explicit_validation_error():
    """용준/Claude(2026-07-27, 후속 요청 2번) — action_code가 명시적으로 전달됐는데
    payload가 잘못됐으면(범위 밖 인덱스) 조용히 자연어 파싱으로 폴백하지 않고 명시적인
    ValueError를 낸다(프론트 버그가 숨겨지지 않아야 한다는 요청)."""
    llm = DiscoveryScriptedLLM()
    state = start_ideation_conversation(
        session_id="ACTION-3", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea={"description": ""}, llm_call=llm
    )
    with pytest.raises(ValueError, match="action_payload"):
        reply_ideation_conversation(
            previous_state=state,
            user_message="1번",
            llm_call=llm,
            action_code="select_problem_focus",
            action_payload={"indices": [99]},  # 범위 밖 — 명시적으로 거부돼야 한다.
        )


def test_action_code_unsupported_raises_explicit_validation_error():
    llm = DiscoveryScriptedLLM()
    state = start_ideation_conversation(
        session_id="ACTION-UNSUPPORTED", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea={"description": ""}, llm_call=llm
    )
    with pytest.raises(ValueError, match="지원하지 않는 action_code"):
        reply_ideation_conversation(
            previous_state=state, user_message="1번", llm_call=llm, action_code="not_a_real_action"
        )


def test_action_code_wrong_phase_raises_explicit_validation_error():
    """confirm_concept은 awaiting_concept_confirmation 전용인데 problem_focus_selection
    대기 중에 보내면(프론트 버그 시나리오) 명시적으로 거부돼야 한다."""
    llm = DiscoveryScriptedLLM()
    state = start_ideation_conversation(
        session_id="ACTION-WRONG-PHASE", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea={"description": ""}, llm_call=llm
    )
    assert state["phase"] == "awaiting_problem_focus_selection"
    with pytest.raises(ValueError, match="허용되지 않습니다"):
        reply_ideation_conversation(
            previous_state=state, user_message="확정", llm_call=llm, action_code="confirm_concept"
        )


def test_choose_another_candidate_returns_to_candidate_selection_unlocked():
    """용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — idea_candidates가 없는 세션
    (카드 선택을 거치지 않고 provisional_from_merge로 온 새 discovery 플로우)에서
    "choose_another_candidate"는 더 이상 존재하지 않는 카드 목록으로 돌아갈 수 없다.
    이 경우 반론·결합 라운드로 되돌아가 다시 결합하라는 의미로 처리한다
    (concept_confirmation의 "재검토" 분기와 동일 — make_concept_confirmation_node 참고).
    idea_candidates가 있는 레거시 세션의 기존 동작(카드 선택 화면 복귀)은
    test_old_saved_discovery_session_can_still_reply_at_awaiting_candidate_selection에서
    별도로 검증한다."""
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_concept_confirmation"
    assert not state.get("idea_candidates")

    state = reply_ideation_conversation(
        previous_state=state,
        user_message="다른 후보를 선택하겠습니다.",
        llm_call=llm,
        action_code="choose_another_candidate",
        action_payload={},
    )

    # 조건이 다시 충족되면 카드 없이 곧바로 새 provisional_idea가 채택되고 검증까지
    # 정지 없이 이어져 다시 awaiting_concept_confirmation에서 멈춘다.
    assert state["phase"] == "awaiting_concept_confirmation"
    assert state["provisional_idea"] is not None
    assert state["idea_locked"] is False


def test_action_code_drop_direction_missing_payload_raises():
    llm = _NeverSatisfiesConflictConditionsLLM()
    state = start_ideation_conversation(
        session_id="ACTION-DROP-MISSING", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea={"description": ""}, llm_call=llm
    )
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_conflict_resolution"
    with pytest.raises(ValueError, match="direction_ids"):
        reply_ideation_conversation(
            previous_state=state, user_message="", llm_call=llm, action_code="drop_direction", action_payload={}
        )


def test_action_code_merge_directions_already_dropped_target_raises():
    llm = _NeverSatisfiesConflictConditionsLLM()
    state = start_ideation_conversation(
        session_id="ACTION-MERGE-DROPPED", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea={"description": ""}, llm_call=llm
    )
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_conflict_resolution"
    direction_ids = [d["direction_id"] for d in state["solution_directions"]]
    state = reply_ideation_conversation(
        previous_state=state,
        user_message="",
        llm_call=llm,
        action_code="drop_direction",
        action_payload={"direction_ids": [direction_ids[0]]},
    )
    assert state["phase"] == "awaiting_conflict_resolution"
    with pytest.raises(ValueError, match="폐기되거나 결합된"):
        reply_ideation_conversation(
            previous_state=state,
            user_message="",
            llm_call=llm,
            action_code="merge_directions",
            action_payload={"direction_ids": [direction_ids[0], direction_ids[1]]},
        )


def test_action_code_proceed_to_validation_skips_unmet_conditions_like_text_keyword():
    llm = _NeverSatisfiesConflictConditionsLLM()
    state = start_ideation_conversation(
        session_id="ACTION-4", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea={"description": ""}, llm_call=llm
    )
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_conflict_resolution"

    state = reply_ideation_conversation(
        previous_state=state, user_message="", llm_call=llm, action_code="proceed_to_validation"
    )
    # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — "proceed_to_validation"도 더 이상
    # candidate_planning(후보 압축·카드 나열)을 거치지 않는다. provisional_from_merge가
    # 현재 active 방향을 카드 없이 바로 채택하고, 검증까지 정지 없이 이어진다.
    assert state["phase"] == "awaiting_concept_confirmation"


def test_action_code_drop_direction_by_explicit_direction_id():
    llm = _NeverSatisfiesConflictConditionsLLM()
    state = start_ideation_conversation(
        session_id="ACTION-5", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea={"description": ""}, llm_call=llm
    )
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_conflict_resolution"

    state = reply_ideation_conversation(
        previous_state=state,
        user_message="",
        llm_call=llm,
        action_code="drop_direction",
        action_payload={"direction_ids": ["direction_1"]},
    )
    dropped = next(d for d in state["solution_directions"] if d["direction_id"] == "direction_1")
    assert dropped["status"] == "dropped"


def test_action_code_confirm_concept_locks_idea():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_concept_confirmation"

    # "음..."은 _CONFIRM_KEYWORDS 어디에도 안 걸리지만 action_code가 우선한다.
    state = reply_ideation_conversation(
        previous_state=state, user_message="음...", llm_call=llm, action_code="confirm_concept"
    )
    assert state["idea_locked"] is True


def test_action_code_revise_candidate_does_not_lock_even_with_confirm_like_text():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_concept_confirmation"

    # 텍스트만 보면 "확정"이 들어있어 자연어 파싱으로는 confirm으로 오인될 수 있지만,
    # action_code="revise_candidate"가 우선하므로 잠기지 않는다.
    state = reply_ideation_conversation(
        previous_state=state,
        user_message="이걸로 확정하기 전에 다른 것도 보고 싶어요",
        llm_call=llm,
        action_code="revise_candidate",
    )
    assert state["idea_locked"] is False
    # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — 재검토는 반론·결합 라운드로
    # 되돌아간다. 조건이 다시 충족되면 카드 없이 곧바로 새 provisional_idea가 채택되고
    # 검증까지 정지 없이 이어져 다시 awaiting_concept_confirmation에서 멈춘다.
    assert state["phase"] == "awaiting_concept_confirmation"


def test_action_code_return_to_problem_definition_resets_solution_directions_and_replays_definition():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)  # awaiting_candidate_selection까지 자동 진행.
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_concept_confirmation"
    assert state["provisional_idea"] is not None

    state = reply_ideation_conversation(
        previous_state=state,
        user_message="",
        llm_call=llm,
        action_code="return_to_problem_definition",
    )
    # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — problem_definition ->
    # idea_divergence -> idea_conflict_and_merge를 다시 거쳐 provisional_from_merge까지
    # 카드 없이 자동으로 이어지므로(DiscoveryScriptedLLM stub이 매번 조건을 만족시킴),
    # 검증까지 정지 없이 진행돼 결국 다시 awaiting_concept_confirmation에 도달한다 — 문제
    # 정의부터 다시 시작했다는 뜻이다.
    assert state["phase"] == "awaiting_concept_confirmation"
    assert any(r.get("action_type") == "revision" and "문제 정의" in r.get("title", "") for r in state["idea_evolution"])


# ---------------------------------------------------------------------------
# 후속 요청(2026-07-27) 5번 — 누락 테스트 4종.
# ---------------------------------------------------------------------------


class _RecordingEvidenceLookup:
    """test_ideation_target_evidence_wiring.py::_RecordingEvidenceLookup와 동일한 fake —
    이 파일은 신규 problem 단계 노드들이 실제로 evidence_lookup(RAG-006)을 호출하는지
    검증하는 데 쓴다."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, persona_id, query, *, runtime_scope=None):
        self.calls.append({"persona_id": persona_id, "query": query, "runtime_scope": runtime_scope})
        return []


def test_evidence_lookup_is_called_across_all_new_problem_stage_nodes():
    """요청 5-2번 — 신규 problem 단계 전체(problem_discovery/problem_definition/
    idea_divergence/idea_conflict_and_merge/idea_validation)에서 기존 evidence_lookup
    (RAG-006, project_id/use_rag로 만들어지는 콜백)이 계속 호출되는지 확인한다. 이
    노드들을 새로 추가하면서 evidence_lookup 배선을 빠뜨리는 회귀를 막는다."""
    lookup = _RecordingEvidenceLookup()
    llm = DiscoveryScriptedLLM()

    state = start_ideation_conversation(
        session_id="RAG-CONTINUITY",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": ""},
        llm_call=llm,
        evidence_lookup=lookup,
    )
    assert any(c["persona_id"] == "planning_expert" for c in lookup.calls), "problem_discovery"

    lookup.calls.clear()
    state = reply_ideation_conversation(
        previous_state=state, user_message="1번", llm_call=llm, evidence_lookup=lookup
    )
    # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — problem_definition/
    # idea_divergence(둘 다 planning_expert) + idea_conflict_and_merge(dev_expert) +
    # provisional_from_merge(evidence_lookup 없음, 결정론적) + validate_planning
    # (planning_expert) + validate_technical(dev_expert)까지, 카드 선택 정지 없이 한 번의
    # /reply 안에서 전부 자동으로 이어져 곧바로 awaiting_concept_confirmation에 도달한다.
    assert state["phase"] == "awaiting_concept_confirmation"
    called_personas = {c["persona_id"] for c in lookup.calls}
    assert "planning_expert" in called_personas
    assert "dev_expert" in called_personas


def test_idea_conflict_and_merge_produces_new_merged_direction_from_two_directions():
    """요청 5-1번 — 사용자가 아니라 회의(반론·결합) 과정에서 두 해결 방향이 결합되면
    새로운 방향이 실제로 solution_directions에 추가되는지 확인한다(DiscoveryScriptedLLM의
    기본 stub이 direction_1+direction_3을 결합해 "direction_merged_1"을 만든다)."""
    llm = DiscoveryScriptedLLM()
    state = start_ideation_conversation(
        session_id="MERGE-CREATES-DIRECTION",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": ""},
        llm_call=llm,
    )
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)

    merged = [d for d in state["solution_directions"] if d["direction_id"] == "direction_merged_1"]
    assert len(merged) == 1
    assert merged[0]["status"] == "active"
    assert merged[0]["origin_direction_ids"] == ["direction_1", "direction_3"]
    directions = {d["direction_id"]: d for d in state["solution_directions"]}
    assert directions["direction_1"]["status"] == "merged"
    assert directions["direction_3"]["status"] == "merged"
    assert directions["direction_2"]["status"] == "active"
    assert directions["direction_4"]["status"] == "active"
    assert merged[0]["parent_direction_ids"] == ["direction_1", "direction_3"]
    merge_records = [r for r in state["idea_evolution"] if r["action_type"] == "merge"]
    assert merge_records and merge_records[0]["target_direction_ids"] == ["direction_1", "direction_3"]


def test_action_code_merge_directions_requests_another_round_with_explicit_ids():
    """요청 5-1번(사용자 결합 요청 경로) — merge_directions action code로 사용자가 명시한
    두 후보/방향이 다음 idea_conflict_and_merge 라운드에서 실제로 결합 재료로 쓰이는지
    (스텁이 항상 direction_1+direction_3을 결합하므로, 최종적으로 다시 direction_merged_1
    이 생성되는지)로 확인한다."""
    llm = _NeverSatisfiesConflictConditionsLLM()
    state = start_ideation_conversation(
        session_id="ACTION-MERGE", notice_and_criteria=NOTICE_AND_CRITERIA, user_idea={"description": ""}, llm_call=llm
    )
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_conflict_resolution"

    # 다음 라운드부터는 기본 DiscoveryScriptedLLM(항상 조건을 만족시키는 stub)으로 바꿔
    # merge_directions 요청이 실제로 다음 라운드 실행까지 이어지는지 확인한다.
    llm2 = DiscoveryScriptedLLM()
    state = reply_ideation_conversation(
        previous_state=state,
        user_message="",
        llm_call=llm2,
        action_code="merge_directions",
        action_payload={"direction_ids": ["direction_1", "direction_3"]},
    )
    # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — 결합 조건 충족 후 카드 없이
    # 곧바로 provisional_from_merge -> 검증까지 정지 없이 이어져 awaiting_concept_confirmation
    # 에 도달한다.
    assert state["phase"] == "awaiting_concept_confirmation"
    assert any(r.get("action_type") == "revision" and r.get("changed_by") == "user" for r in state["idea_evolution"])


# ---------------------------------------------------------------------------
# 후속 요청(2026-07-27) 7번 — 기존 저장 세션(신규 필드가 없는 구버전 state) 호환성.
# ---------------------------------------------------------------------------


def test_old_saved_discovery_session_without_new_fields_is_handled_safely():
    """신규 필드(problem_areas/solution_directions/idea_evolution/provisional_idea/
    idea_locked/pending_user_action 등)가 아예 없는, 이번 개편 이전에 저장된 discovery
    세션을 흉내낸다 — 순수 조회 함수(active_stage_for/각 가드 함수)가 KeyError 없이
    안전한 기본값으로 동작하는지 확인한다."""
    legacy_state = {
        "session_id": "LEGACY-1",
        "phase": "awaiting_candidate_selection",
        "ideation_mode": "discovery",
        "idea_candidates": [{"candidate_id": "candidate_1", "title": "구버전 후보"}],
        "selected_idea": None,
        # 아래 신규 필드들은 의도적으로 전부 빠뜨린다.
    }
    # active_stage_for는 phase 문자열만 보고 판단하므로 신규 필드 유무와 무관하게 안전하다.
    assert active_stage_for(legacy_state["phase"]) == "candidate_discovery"
    # 가드 함수들은 전부 .get()으로 읽으므로 키가 아예 없어도 예외 없이 결정론적 기본값을
    # 돌려준다(문제/방향/검증 데이터가 없으니 "아직 안 됨"으로 안전하게 판정).
    assert problem_defined(legacy_state) is False
    assert target_user_defined(legacy_state) is False
    assert existing_limitations_defined(legacy_state) is False
    assert solution_direction_count(legacy_state) == 0
    assert critique_count(legacy_state) == 0
    assert merge_or_revision_count(legacy_state) == 0
    assert meets_conflict_and_merge_min_conditions(legacy_state) is False
    assert ready_for_concept_confirmation(legacy_state) is False


_LEGACY_CANDIDATES = [
    {
        "candidate_id": "candidate_1",
        "title": "구버전 후보 1",
        "problem": "구버전 세션이 저장했던 문제 정의",
        "target_user": "구버전 세션의 대상 사용자",
        "usage_scenario": "구버전 세션의 사용 시나리오",
        "core_value": "구버전 세션의 핵심 가치",
        "solution": "구버전 세션의 해결 방식",
        "main_features": ["구버전 기능 1"],
        "differentiation": "구버전 차별점",
        "contest_fit": "구버전 공모전 적합성",
        "success_metrics": ["구버전 성공 지표"],
        "feasibility": "medium",
        "technical_approach": "구버전 기술 접근",
        "required_data": [],
        "risks": [],
    },
    {
        "candidate_id": "candidate_2",
        "title": "구버전 후보 2",
        "problem": "구버전 세션이 저장했던 다른 문제 정의",
        "target_user": "구버전 세션의 다른 대상 사용자",
        "usage_scenario": "구버전 세션의 다른 사용 시나리오",
        "core_value": "구버전 세션의 다른 핵심 가치",
        "solution": "구버전 세션의 다른 해결 방식",
        "main_features": ["구버전 기능 2"],
        "differentiation": "구버전 다른 차별점",
        "contest_fit": "구버전 다른 공모전 적합성",
        "success_metrics": ["구버전 다른 성공 지표"],
        "feasibility": "medium",
        "technical_approach": "구버전 다른 기술 접근",
        "required_data": [],
        "risks": [],
    },
]


def test_old_saved_discovery_session_can_still_reply_at_awaiting_candidate_selection():
    """구버전 discovery 세션(신규 problem 단계가 생기기 전, candidate_generation부터
    바로 시작해 카드를 나열하고 phase가 이미 "awaiting_candidate_selection"인 상태)을
    이어받아도 /reply가 정상 동작한다. 용준/Claude(2026-07-28, 요청: 카드 선택 단계
    제거) — start_ideation_conversation은 이제 discovery 모드에서 항상 problem_discovery
    부터 시작하므로(카드 선택 단계 자체를 더 이상 거치지 않는다), 이 시나리오는 더 이상
    실제 호출로 재현할 수 없다. 대신 initial_conv_state로 만든 빈 상태에 "이미
    awaiting_candidate_selection에 저장돼 있던 구버전 세션"을 직접 흉내내(phase +
    idea_candidates만 채운다) apply_user_answer -> candidate_selection 경로(레거시
    노드, 그래프 배선 그대로 유지됨)가 여전히 정상 동작하는지 확인한다."""
    llm = DiscoveryScriptedLLM()
    legacy_state = {
        **initial_conv_state("LEGACY-2", NOTICE_AND_CRITERIA, {"description": ""}),
        "phase": "awaiting_candidate_selection",
        "idea_candidates": _LEGACY_CANDIDATES,
    }
    state = reply_ideation_conversation(previous_state=legacy_state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_concept_confirmation"
    assert state["provisional_idea"]["candidate_id"] == "candidate_1"

    # 신규 필드가 아예 없는 구버전 저장 데이터(이번 개편 이전에 저장된, phase가 여전히
    # awaiting_candidate_selection인 세션)를 재현해 같은 경로가 안전하게 동작하는지도
    # 확인한다.
    simulated_legacy = dict(legacy_state)
    for legacy_missing_key in (
        "problem_definition",
        "solution_directions",
        "idea_evolution",
        "provisional_idea",
        "idea_locked",
        "pending_user_action",
        "user_confirmed",
    ):
        simulated_legacy.pop(legacy_missing_key, None)

    result = reply_ideation_conversation(previous_state=simulated_legacy, user_message="1번", llm_call=llm)
    assert result["phase"] == "awaiting_concept_confirmation"
    assert result["provisional_idea"] is not None
    # idea_locked 키 자체가 없었던 구버전 state를 거치면 내부 그래프 state에는 False가
    # 아니라 None으로 남을 수 있다(어떤 노드도 이 키를 없다고 해서 강제로 채우지 않는다) —
    # 다만 모든 실제 사용처(_guard_pre_lock_messages 등)는 `.get("idea_locked")`를
    # 진위값으로만 검사하므로 None/False가 동일하게 "아직 확정 아님"으로 안전하게
    # 처리된다. API 응답 계층(_serialize_state)은 `.get("idea_locked", False)`로 항상
    # 명시적 False를 노출한다 — 여기서는 내부 state의 falsy 동작만 확인한다.
    assert not result.get("idea_locked")


# ---------------------------------------------------------------------------
# 후속 요청(2026-07-27) 3번 — candidate_planning 동작 수준 테스트. 기존 테스트는 프롬프트에
# 특정 문자열이 포함됐는지만 확인했다(문자열 주입 테스트). 여기서는 idea_conflict_and_merge가
# 실제로 만든 solution_directions(방향 1/3 active, 방향 2 dropped, 결합 방향 active)와
# problem_definition을 "규칙을 지키는" 결정론적 LLM이 그대로 압축한다고 가정했을 때, 그
# 결과가 idea_candidates에 구조화된 필드 수준으로 반영되는지 확인한다 — 새 아이디어를
# 임의로 발명하지 않고, drop된 방향이 부활하지 않으며, merge된 방향의 핵심 요소가 반영되고,
# 후보의 problem/target_user가 problem_definition을 그대로 유지하는지가 핵심이다.
# ---------------------------------------------------------------------------

_ACTIVE_DIRECTION_MARKERS = ("방향 2", "원리 2", "방향 4", "원리 4", "결합 방향", "결합 원리")
_DROPPED_DIRECTION_MARKERS = ("방향 1", "원리 1", "방향 3", "원리 3")


class _CompressionCompliantLLM(DiscoveryScriptedLLM):
    """candidate_planning 호출에만 개입해, idea_conflict_and_merge가 실제로 남긴 active
    solution_directions(방향 1/방향 3/결합 방향)와 problem_definition만 재료로 써서 후보
    2개를 압축하는 "규칙을 지키는 LLM"을 흉내낸다. 다른 모든 노드는 기존
    DiscoveryScriptedLLM 표준 stub(문제 영역 생성/문제 정의/발산/반론·결합/검토/해석 규칙)을
    그대로 재사용한다 — 방향 2가 dropped되고 방향 1·3이 결합 방향으로 병합되는 것은 그
    표준 stub의 [반론·결합 규칙] 응답이 이미 결정하므로 이 서브클래스가 다시 정의하지
    않는다."""

    def __call__(self, prompt: str) -> str:
        if "[후보 생성 규칙" in prompt:
            self.captured_prompts.append(prompt)
            self.call_counts["candidate_planning"] += 1
            return json.dumps(
                {
                    "contest_analysis": {
                        "purpose": "목적",
                        "key_criteria": ["기준1"],
                        "required_tech_or_theme": ["기술1"],
                        "suitable_problem_domains": ["영역1"],
                        "constraints": ["제약1"],
                        "unknown_from_notice": ["미상1"],
                    },
                    "candidates": [
                        {
                            "candidate_id": "candidate_1",
                            "title": "방향 2 압축안",
                            "problem": "문제 정의",
                            "target_user": "대상 사용자",
                            "usage_scenario": "방향 2 원리 2를 활용한 이용 시나리오",
                            "core_value": "방향 2 압축안 핵심 가치",
                            "solution": "원리 2를 중심으로 방향 2를 구현합니다.",
                            "main_features": ["방향 2 기능1"],
                            "differentiation": "방향 2 압축안 차별성",
                            "contest_fit": "방향 2 압축안 공모전 적합성",
                            "source_direction_ids": ["direction_2"],
                            "reflected_evolution_ids": [],
                        },
                        {
                            "candidate_id": "candidate_2",
                            "title": "결합 방향 압축안",
                            "problem": "문제 정의",
                            "target_user": "대상 사용자",
                            "usage_scenario": "결합 방향(결합 원리)을 활용한 이용 시나리오",
                            "core_value": "결합 방향 압축안 핵심 가치",
                            "solution": "결합 원리를 반영해 결합 방향을 구현합니다.",
                            "main_features": ["결합 방향 기능1"],
                            "differentiation": "결합 방향 압축안 차별성",
                            "contest_fit": "결합 방향 압축안 공모전 적합성",
                            "source_direction_ids": ["direction_merged_1"],
                            "reflected_evolution_ids": [],
                        },
                    ],
                },
                ensure_ascii=False,
            )
        return super().__call__(prompt)


def test_candidate_planning_compresses_prior_meeting_results_without_reinventing():
    """용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — candidate_planning은 새
    discovery 플로우의 그래프 배선에서 더 이상 자동 실행되지 않으므로(위
    test_candidate_planning_prompt_is_grounded_in_prior_meeting_results_not_reinvented와
    동일한 이유), 실제 회의 상태를 만든 뒤(stop_after_expert_turn=True) 노드 함수를 직접
    호출해 압축 동작 자체는 그대로 검증한다."""
    from graph.ideation_conv_discovery import make_candidate_planning_node

    llm = _CompressionCompliantLLM()
    state = start_ideation_conversation(
        session_id="COMPRESS-BEHAVIOR",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": ""},
        llm_call=llm,
    )
    state = reply_ideation_conversation(
        previous_state=state, user_message="1번", llm_call=llm, stop_after_expert_turn=True
    )
    assert state["phase"] == "idea_validation"
    state = {**state, **make_candidate_planning_node(llm)(state)}

    # idea_conflict_and_merge가 실제로 만든 방향 상태 — 방향 1·3은 merged로 비활성화되고,
    # 방향 2·4와 새 결합 방향만 active로 후보 생성에 전달된다.
    directions_by_title = {d["title"]: d for d in state["solution_directions"]}
    assert directions_by_title["방향 2"]["status"] == "active"
    assert directions_by_title["방향 4"]["status"] == "active"
    assert directions_by_title["방향 1"]["status"] == "merged"
    assert directions_by_title["방향 3"]["status"] == "merged"
    assert directions_by_title["결합 방향"]["status"] == "active"

    candidates = state["idea_candidates"]
    assert 2 <= len(candidates) <= 3
    # 후보 2~3개가 기존 solution_directions의 압축·조합 결과다(생성기가 필요한 개수를
    # 스스로 결정할 수 있으므로 정확히 2개로 단정하지 않는다).

    for candidate in candidates:
        # 후보의 problem/target_user가 problem_definition을 그대로 유지한다.
        assert candidate["problem"] == state["problem_definition"]["problem"]
        assert candidate["target_user"] == state["problem_definition"]["target_user"]

        combined_text = " ".join(
            str(candidate.get(field, "")) for field in ("title", "solution", "usage_scenario", "differentiation")
        )
        # merge로 supersede된 이전 방향(방향 1·3)이 최종 후보에서 부활하지 않는다.
        assert not any(marker in combined_text for marker in _DROPPED_DIRECTION_MARKERS)
        # 완전히 새로운 해결 방향을 임의로 생성하지 않는다 — 후보 내용은 실제 active
        # solution_directions(방향 1/방향 3/결합 방향)의 제목·원리에서만 파생된다.
        assert any(marker in combined_text for marker in _ACTIVE_DIRECTION_MARKERS)

    # merge된 방향(결합 방향/결합 원리)의 핵심 요소가 최소 한 후보에 반영된다.
    assert any("결합 방향" in c["solution"] or "결합 원리" in c["solution"] for c in candidates)


# ---------------------------------------------------------------------------
# 후속 요청(2026-07-27) 4번 — idea_locked=True 이후 expert_discussion/finalizing을 거친
# 최종 회의 결과에서도 확정된 아이디어의 핵심 필드(problem/target_user/core_value/
# differentiation/solution)가 달라지지 않는지 검증한다. selected_idea 객체 자체가
# 재할당되지 않는다는 것(코드 주석으로만 남아 있던 것)과, 그 핵심 필드가 최종
# idea_proposal까지 훼손 없이 전달되는지(파이프라인 무결성)를 함께 확인한다.
# ---------------------------------------------------------------------------


class _CoreFieldPreservingSynthesisLLM(DiscoveryScriptedLLM):
    """synthesis(최종 종합) 호출에만 개입해, 확정된 아이디어의 핵심 필드를 그대로
    반영하는 "규칙을 지키는 LLM"을 흉내낸다. 다른 모든 노드는 기존 DiscoveryScriptedLLM
    표준 stub을 그대로 쓴다.

    용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — 확정되는 아이디어가 더 이상
    candidate_planning이 만든 candidate_1이 아니라, provisional_from_merge가 채택한
    결합 방향(problem="문제 정의", target_user="결합 적합성", differentiation은 채택된
    방향 제목 기반 문구)이므로, 이 mock의 하드코딩된 값도 그 실제 값과 일치하도록
    맞춘다 — 이 테스트는 "LLM이 똑똑하게 반영하는지"가 아니라 "확정된 값이 파이프라인
    끝까지 훼손 없이 전달되는지"를 검증하므로, mock 자체는 여전히 프롬프트 내용과 무관한
    고정 응답이어도 된다."""

    def __call__(self, prompt: str) -> str:
        if '"idea_name"' in prompt:
            self.captured_prompts.append(prompt)
            return json.dumps(
                {
                    "idea_name": "결합 방향 기반 아이디어",
                    "one_line_pitch": "위원회가 결합한 방향으로 문제를 해결한다",
                    "problem_definition": "문제 정의",
                    "target_user": "결합 적합성",
                    "core_user_value": "결합 원리",
                    "key_features": ["결합 작동 방식"],
                    "required_data": ["committee_merge 데이터"],
                    "tech_direction": "committee_merge 기술 접근",
                    "mvp_scope": ["committee_merge MVP"],
                    "differentiation": "'수정 방향 2' 해결 방향의 핵심 원리를 그대로 반영한 안전 후보입니다.",
                    "risks_and_mitigations": [{"risk": "위험1", "mitigation": "대응1"}],
                    "success_metrics": ["검증 단계에서 확정 예정"],
                    "expert_final_opinions": {"planning_expert": "기획 판단", "dev_expert": "개발 판단"},
                    "unverified_assumptions": [],
                    "final_recommendation": "추천",
                    "final_recommendation_reason": "근거",
                    "next_actions": ["다음 작업1"],
                },
                ensure_ascii=False,
            )
        return super().__call__(prompt)


def test_confirmed_idea_core_fields_survive_expert_discussion_and_finalizing():
    llm = _CoreFieldPreservingSynthesisLLM(dev_next_action="await_user_decision")
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "awaiting_concept_confirmation"

    state = reply_ideation_conversation(previous_state=state, user_message="확정할게요", llm_call=llm)
    assert state["idea_locked"] is True
    confirmed_idea = state["selected_idea"]
    assert confirmed_idea is not None
    core_fields = ("problem", "target_user", "core_value", "differentiation", "solution")
    confirmed_snapshot = {field: confirmed_idea.get(field) for field in core_fields}
    assert state["phase"] == "discussion_complete"

    state = finalize_ideation_conversation(previous_state=state, llm_call=llm)
    assert state["phase"] == "finalized"

    # selected_idea는 idea_locked=True 이후 expert_discussion/finalizing을 거치는 동안 어떤
    # 노드도 재할당하지 않는다 — 핵심 필드가 전부 그대로 남아 있어야 한다.
    for field in core_fields:
        assert state["selected_idea"].get(field) == confirmed_snapshot[field]

    # 확정된 아이디어의 핵심 의미가 최종 회의 결과(idea_proposal)에서도 달라지지 않는다.
    proposal = state["idea_proposal"]
    assert proposal["problem_definition"] == confirmed_snapshot["problem"]
    assert proposal["target_user"] == confirmed_snapshot["target_user"]
    assert proposal["differentiation"] == confirmed_snapshot["differentiation"]
