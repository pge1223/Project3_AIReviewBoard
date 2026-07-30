# 작성자: 용준/Claude(2026-07-20, 2026-07-22 동적 전문가 회의로 개편)
# 목적: 대화형 아이디어 발전 회의(ideation-conversation) LangGraph 그래프 조립.
#
#       배치형(ideation_build.py)과의 핵심 차이: 배치형은 START에서 시작해 한 번의
#       graph.stream()이 여러 노드를 연달아 통과한다. 이 그래프는 반대로 "한 번의
#       graph.stream() 호출에서 보통 딱 하나의 정지 지점까지만 간다" — START 진입 자체를
#       state["phase"]로 분기해서(_route_entry), HTTP 요청 한 번에 딱 필요한 만큼만
#       실행한다. 그래서 이 그래프는 매 HTTP 요청마다 backend가 새로 assemble해서 쓴다.
#
#       2026-07-22 개편: "기획 1회 → 개발 1회 → [조건부 수정 1회] → 진행자 정리(무조건)"로
#       고정돼 있던 라운드 구조를 쟁점(issue) 기반 동적 라우팅으로 바꿨다 —
#       planning_expert_discussion과 dev_expert_discussion이 서로를 직접 호출할 수 있고
#       (_route_next_expert_turn), discussion_facilitator는 라우터가 "이제 정리할 시점"이라고
#       판단했을 때만 실행된다. 유일하게 그래프 내부에서 "정지 없이" 이어지는 구간은 전문가
#       발언들 사이(발언 캡까지)와, 다음 라운드로 넘어갈 때(discussion_facilitator가
#       decided_next_action="continue_round"를 판단해 곧바로 planning_expert_discussion으로
#       되돌아가는 것)이다 — 사용자 입력 없이 시스템이 스스로 진행해도 되는 구간이기
#       때문이다. 최종 확정(synthesis)은 이 루프 안에 들어있지 않다 — 오직 별도 API 호출
#       (ideation_conv_run.py::finalize_ideation_conversation)로만 phase="finalizing"을
#       만들어 진입할 수 있다(임의 확정 금지).
# import: langgraph.graph.StateGraph/START/END, 같은 패키지의 ideation_conv_nodes/state.

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .ideation_conv_discovery import (
    make_candidate_feasibility_node,
    make_candidate_planning_node,
    make_candidate_selection_node,
)
from .ideation_conv_nodes import (
    _route_next_expert_turn,
    make_canvas_update_node,
    make_conv_discussion_node,
    make_conv_question_node,
    make_conv_synthesis_node,
    make_discussion_facilitator_node,
)
from .ideation_conv_problem import (
    _route_after_conflict_merge,
    _route_after_idea_validation,
    _route_after_specification_completion,
    make_concept_confirmation_node,
    make_conflict_resolution_node,
    make_idea_conflict_and_merge_node,
    make_idea_divergence_node,
    make_planning_validation_node,
    make_specification_completion_node,
    make_technical_validation_node,
    make_problem_definition_node,
    make_problem_discovery_node,
    make_problem_focus_selection_node,
    make_provisional_from_merge_node,
    make_provisional_selection_node,
)
from .ideation_conv_state import IdeationConvState
from .ideation_trace import trace_event
from .llm import LLMCall

_ENTRY_NODES = {
    # 용준/Claude(2026-07-21): discovery(아이디어 발굴) 모드 진입점 2개 — refinement 전용
    # 진입점(아래 4개)은 값 하나도 바꾸지 않는다.
    "candidate_generation": "candidate_planning",
    "candidate_selection": "candidate_selection",
    "planning_question": "planning_question",
    "developer_question": "developer_question",
    "expert_discussion": "planning_expert_discussion",
    "finalizing": "synthesis",
    # 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — discovery
    # 모드가 candidate_generation보다 먼저 거치는 문제 발견/정의/발산/반론·결합 단계의
    # API 재진입점. "candidate_selection"(기존, 즉시 확정)은 위에 그대로 남겨두되(직접
    # 노드를 호출하는 기존 단위 테스트 보존용) apply_user_answer는 더 이상 그 값으로
    # 전이시키지 않는다 — 대신 "provisional_selection"로 전이한다.
    "problem_discovery": "problem_discovery",
    "problem_focus_selection": "problem_focus_selection",
    "conflict_resolution": "conflict_resolution",
    "provisional_selection": "provisional_selection",
    # 용준/Claude(2026-07-30, 요청: specification_completion 흐름 추가) — 정상 흐름에서는
    # provisional_from_merge/provisional_selection이 같은 그래프 호출 안에서 바로 이
    # 노드로 이어지므로 별도 API 재진입이 필요 없지만, 방어적으로(예: 예외 복구 후 재개)
    # phase="specification_completion"으로 그래프가 다시 시작될 수 있으므로 등록해 둔다.
    "specification_completion": "specification_completion",
    # 용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다") —
    # idea_validation은 이제 항상 기획위원부터(validate_planning) 진입한다. 개발위원 차례로
    # 이어서 재진입하는 경우는 expert_discussion과 동일하게 forced_next_speaker로
    # 처리한다(아래 _route_entry의 idea_validation 분기 참고).
    "idea_validation": "validate_planning",
    "concept_confirmation": "concept_confirmation",
}

_FORCED_SPEAKER_TO_NODE = {
    "planning_expert": "planning_expert_discussion",
    "dev_expert": "dev_expert_discussion",
    # 재인/Claude(2026-07-23, 아바타 페이싱 연동): ideation_conv_run.py::continue_ideation_expert_turn이
    # 라운드를 한 턴씩 끊어 진행할 때, 다음 차례가 진행자면 여기로 강제 진입한다. 이 매핑을
    # 추가한 것 자체는 "누가 다음에 말할지" 판단(_route_next_expert_turn)과 무관하다 — 그
    # 함수가 이미 "facilitator"를 반환할 수 있었고(_expert_turn_targets 참고), 여기서는 그
    # 결과를 재진입 지점으로도 쓸 수 있게 진입 테이블만 넓힌 것뿐이다.
    "facilitator": "discussion_facilitator",
}


def _route_entry(state: IdeationConvState) -> str:
    """START 직후 phase를 보고 이번 그래프 호출에서 실행할 노드를 고른다. 이 그래프는
    awaiting_*/awaiting_user_decision/finalized/failed phase로는 절대 진입하지 않는다
    (호출부가 그 phase에서는 그래프를 아예 부르지 않아야 한다 — ideation_conv_run.py가
    보장한다).

    재인/Claude(2026-07-23, 아바타 페이싱 연동): phase가 "expert_discussion"이고
    forced_next_speaker가 설정돼 있으면(continue_ideation_expert_turn이 한 턴씩 끊어 진행할
    때 다음 화자를 강제 지정한 경우) 기본값(planning_expert_discussion) 대신 그 화자의
    노드로 바로 진입한다. forced_next_speaker=planning_expert/dev_expert는 그 노드가
    실행되자마자 리셋되므로(make_conv_discussion_node 참고) 다음 라운드에는 잔류하지 않는다.

    forced_next_speaker="facilitator"(위와 같은
    이유로 discussion_facilitator로 강제 진입)는 discussion_facilitator_node 자체에는 리셋
    로직이 없다 — 그 노드가 원래 forced 진입 대상이 아니었기 때문이다. 대신 이 값을 쓰는
    쪽(ideation_conv_run.py::continue_ideation_expert_turn)이 호출 뒤 직접 지운다."""
    phase = state.get("phase")
    if phase not in _ENTRY_NODES:
        raise ValueError(
            f"이 phase에서는 그래프를 시작할 수 없습니다: {phase!r}. "
            "awaiting_*/awaiting_user_decision/finalized/failed는 API 레이어가 걸러야 합니다."
        )
    if phase == "expert_discussion":
        forced = state.get("forced_next_speaker")
        forced_node = _FORCED_SPEAKER_TO_NODE.get(forced) if forced else None
        if forced_node:
            return forced_node
    # 용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다") —
    # idea_validation은 순서가 항상 "기획→개발" 고정이라(사용자 개입/라운드 반복 없음)
    # expert_discussion의 라운터(_route_next_expert_turn)는 쓰지 않는다.
    # continue_ideation_validation_turn(ideation_conv_run.py)이 기획위원 발언 직후에만
    # forced_next_speaker="dev_expert"를 설정해 호출하므로, 여기서는 그 값 하나만 본다.
    if phase == "idea_validation" and state.get("forced_next_speaker") == "dev_expert":
        return "validate_technical"
    return _ENTRY_NODES[phase]


def _route_after_facilitator(state: IdeationConvState) -> str:
    """discussion_facilitator가 직접 다음 라우팅을 결정한다(더 이상 전문가 노드가 정하지
    않는다 — make_discussion_facilitator_node 참고). 용준/Claude(2026-07-22, 요청: "잠시만"
    취소 중 phase 오염 수정) — "다음 라운드로 자동 진행" 여부는 phase가 아니라 별도 필드
    next_route("continue_round")로만 판단한다(1:1 인터뷰 노드가 아니라 라운드테이블로
    돌아간다) — phase는 이 시점에도 항상 실제 canonical 상태("expert_discussion")를 유지해,
    이 라우팅 직후(다음 노드 실행 중) 취소되어도 저장되는 phase가 그래프 밖에서 의미 없는
    내부 신호값이 아니라 항상 재개 가능한 값이 되도록 한다. 그 외(awaiting_user_decision
    등)는 END로 멈춘다."""
    phase = state.get("phase")
    if phase == "failed":
        return "failed"
    if state.get("next_route") == "continue_round":
        return "continue_round"
    if state.get("phase") == "discussion_complete":
        return "discussion_complete"
    return "await_user_decision"


def _route_after_candidate_planning(state: IdeationConvState) -> str:
    """용준/Claude(2026-07-28, 요청: candidate_planning 실패 시 안전 폴백) — 정상 성공
    경로는 candidate_planning이 phase를 건드리지 않으므로(이전 phase 그대로 유지) 기본값
    "ok"로 candidate_feasibility(개발위원 실현가능성 검토)로 이어진다. 노드가 안전 폴백을
    쓴 경우(phase="awaiting_candidate_selection", make_candidate_planning_node 참고)는
    candidate_feasibility를 거치지 않고 바로 멈춘다 — 폴백 경로에 또 다른 LLM 실패
    지점을 두지 않기 위함이다. active direction 부족으로 폴백도 못 만든 경우
    (phase="awaiting_conflict_resolution")는 기존 라운드 상한 도달 시와 동일한 사용자
    조정 화면으로 보낸다."""
    phase = state.get("phase")
    if phase == "failed":
        return "failed"
    if phase == "awaiting_candidate_selection":
        return "fallback"
    if phase == "awaiting_conflict_resolution":
        return "insufficient"
    return "ok"


def _route_after_candidate_selection(state: IdeationConvState) -> str:
    """후보 선택 결과에 따라 다음 노드를 고른다.

    2026-07-26 라운드테이블 재설계: 신청 양식 유무와 무관하게 후보 선택 직후에는 항상
    진행자가 먼저 선택 후보를 요약하고 문제정의를 확인하는 고정 1턴을 연다(목표 루프의
    "고정 1턴" 요건) — "to_form_coach"/"to_refinement" 두 키 모두 discussion_facilitator로
    간다(아래 엣지 매핑 참고). 신청 양식 유무는 이후 discussion_facilitator 내부에서
    (필드 채우기로 전환할지) 판단할 뿐, 진입 노드 자체를 가르지 않는다. 재추천 요청은
    후보 생성으로 돌아가며 나머지는 입력을 기다린다.
    """
    phase = state.get("phase")
    if phase == "failed":
        return "failed"
    if state.get("next_route") == "to_refinement" and state.get("application_form_items"):
        return "to_form_coach"
    if state.get("next_route") == "to_refinement":
        return "to_refinement"
    if phase == "candidate_generation":
        return "regenerate"
    return "await_selection"


def _route_after_problem_focus_selection(state: IdeationConvState) -> str:
    """용준/Claude(2026-07-27) — problem_focus_selection 노드 실행 직후 분기.
    "다른 문제 제안" 요청이면 phase를 "problem_discovery"로 되돌려 놨으므로(노드 자체가
    설정) 같은 요청 안에서 재생성으로 돌아간다. 재요청 상한에 걸리면 phase가
    "awaiting_problem_focus_selection"으로 그대로 남아 있다(노드가 안내 메시지만 붙이고
    멈춘 경우). 그 외(문제 초점이 정해진 성공 경로)는 phase가 그대로 "problem_focus_selection"
    이므로(노드가 phase를 건드리지 않음) problem_definition으로 이어간다."""
    phase = state.get("phase")
    if phase == "failed":
        return "failed"
    if phase == "problem_discovery":
        return "regenerate"
    if phase == "awaiting_problem_focus_selection":
        return "capped"
    return "ok"


def _route_after_problem_definition(state: IdeationConvState) -> str:
    route = "failed" if state.get("phase") == "failed" else "ok"
    trace_event(
        "IDEATION_GRAPH_ROUTE",
        source="problem_definition",
        target="end" if route == "failed" else "idea_divergence",
        phase=state.get("phase"),
    )
    return route


def _route_after_idea_divergence(state: IdeationConvState) -> str:
    route = "failed" if state.get("phase") == "failed" else "ok"
    trace_event(
        "IDEATION_GRAPH_ROUTE",
        source="idea_divergence",
        target="end" if route == "failed" else "idea_conflict_and_merge",
        phase=state.get("phase"),
        solution_direction_count=len(state.get("solution_directions") or []),
    )
    return route


def _route_after_conflict_resolution(state: IdeationConvState) -> str:
    """conflict_resolution 노드 실행 직후 분기. 노드가 해석할 수 없어 다시 물어야 하면
    phase="awaiting_conflict_resolution"으로 스스로 멈춘다(reask). 그 외에는 next_route로
    "이대로 검증 진행"(proceed) / "문제 정의로 복귀"(return_to_problem_definition, 요청
    4번 액션) / "결합/추가/폐기 반영해 라운드 재실행"(continue)인지 구분한다."""
    if state.get("phase") == "awaiting_conflict_resolution":
        return "reask"
    if state.get("next_route") == "proceed":
        return "proceed"
    if state.get("next_route") == "return_to_problem_definition":
        return "return_to_problem_definition"
    return "continue"


def _route_after_provisional_selection(state: IdeationConvState) -> str:
    """용준/Claude(2026-07-27) — make_provisional_selection_node는 기존
    make_candidate_selection_node를 그대로 감싸므로, 그 결과 phase 값 그대로 분기한다.
    "specification_completion"(선택/결합 확정 -> 설계 필드 보완 단계로, 용준/Claude
    (2026-07-30) — 예전에는 곧바로 idea_validation으로 갔다) / "candidate_generation"
    (재추천) / 그 외(재질문·결합 적합도 낮음 등, awaiting_candidate_selection 유지) /
    "failed"."""
    phase = state.get("phase")
    if phase == "failed":
        return "failed"
    if phase == "specification_completion":
        return "validate"
    if phase == "candidate_generation":
        return "regenerate"
    return "await_selection"


def _route_after_concept_confirmation(state: IdeationConvState) -> str:
    if state.get("next_route") == "to_refinement":
        return "to_refinement"
    if state.get("next_route") == "return_to_problem_definition":
        return "return_to_problem_definition"
    if state.get("next_route") == "choose_another_candidate":
        return "choose_another_candidate"
    return "continue"


def _await_conflict_resolution_node(_state: IdeationConvState) -> dict:
    """idea_conflict_and_merge가 라운드 상한에 도달했는데도 최소 조건을 못 채웠을 때만
    거치는 얇은 정지 노드 — 라우팅 함수(_route_after_conflict_merge, ideation_conv_problem.py)
    자체는 state를 변경할 수 없으므로, "사용자에게 물어야 한다"는 라우팅 결정을 실제
    phase 전이로 옮기는 역할만 한다."""
    return {"phase": "awaiting_conflict_resolution"}


def assemble_ideation_conversation_graph(
    llm_call: LLMCall,
    checkpointer: Any | None = None,
    evidence_lookup=None,
    ground_claims=None,
    index_target_evidence=None,
    evidence_planner=None,
    external_evidence_lookup=None,
    similar_case_lookup=None,
    compose_evidence_pool=None,
):
    """대화형 아이디어 발전 회의 그래프를 조립한다.

    노드:
      - planning_question / developer_question: 질문 하나 만들고 바로 END(각각
        awaiting_planning_answer / awaiting_developer_answer로 멈춤).
      - planning_expert_discussion <-> dev_expert_discussion(용준/Claude(2026-07-22, 요청:
        동적 전문가 회의로 개편)): 쟁점·반론 여부에 따라 서로를 직접 호출할 수 있는 양방향
        루프(_route_next_expert_turn이 매 발언 후 다음 발언자를 계산) — 더 이상 "기획 1회 →
        개발 1회"로 고정되지 않는다.
      - discussion_facilitator: 라우터가 "이제 정리할 시점"이라고 판단했을 때만 실행되어
        다음 라운드로 자동 진행할지(continue_round) 사용자 결정을 기다릴지
        (await_user_decision) 직접 결정한다.
      - synthesis: 사용자가 확정 버튼을 눌렀을 때만(phase="finalizing") 진입.

    발언 수 캡(무한 루프 방지)은 ideation_conv_nodes.py::_route_next_expert_turn/
    MAX_EXPERT_TURNS_PER_ROUND/MAX_EXPERT_TURNS_PER_ISSUE가 state를 직접 재계산해 판단한다
    (배치형 facilitator와 같은 원칙 — 그래프 구조가 아니라 라우터가 State를 신뢰의 근거로
    삼는다).
    """
    graph = StateGraph(IdeationConvState)

    planning_question_node = make_conv_question_node(
        "planning_expert", "awaiting_planning_answer", llm_call, evidence_lookup, ground_claims
    )
    developer_question_node = make_conv_question_node(
        "dev_expert", "awaiting_developer_answer", llm_call, evidence_lookup, ground_claims
    )
    # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — speaks_second/discussion_stage를
    # 더 이상 빌드 시점에 고정하지 않는다(각 노드가 매 실행마다 state로부터 계산한다,
    # make_conv_discussion_node 참고). 두 전문가 모두 서로를 직접 호출할 수 있는 대칭 노드다.
    # 용준/Claude(2026-07-23, Phase 1 "Shadow Deterministic Evidence Planner") — evidence_planner는
    # 오직 이 두 discussion 노드에만 주입한다(요청: 질문/후보 생성/후보 검토/synthesis/
    # facilitator에는 Phase 1 planner를 적용하지 않는다).
    planning_discussion_node = make_conv_discussion_node(
        "planning_expert",
        llm_call=llm_call,
        evidence_lookup=evidence_lookup,
        ground_claims=ground_claims,
        evidence_planner=evidence_planner,
        external_evidence_lookup=external_evidence_lookup,
        similar_case_lookup=similar_case_lookup,
        compose_evidence_pool=compose_evidence_pool,
    )
    dev_discussion_node = make_conv_discussion_node(
        "dev_expert",
        llm_call=llm_call,
        evidence_lookup=evidence_lookup,
        ground_claims=ground_claims,
        evidence_planner=evidence_planner,
        external_evidence_lookup=external_evidence_lookup,
        similar_case_lookup=similar_case_lookup,
        compose_evidence_pool=compose_evidence_pool,
    )
    discussion_facilitator_node = make_discussion_facilitator_node(llm_call)
    canvas_update_node = make_canvas_update_node(llm_call)
    synthesis_node = make_conv_synthesis_node(llm_call)

    # 용준/Claude(2026-07-21): discovery(아이디어 발굴) 모드 노드 3종.
    # 용준/Claude(2026-07-27, RAG-007 연결) — external_evidence_lookup은 candidate_planning/
    # candidate_feasibility에만 주입한다(요청 4번). 다른 노드(질문/토론/synthesis)는 이
    # 파라미터를 받지 않는다 — discovery 후보 생성에만 외부 통계·정책 참고자료가 필요하다.
    candidate_planning_node = make_candidate_planning_node(llm_call, evidence_lookup, external_evidence_lookup)
    candidate_feasibility_node = make_candidate_feasibility_node(llm_call, evidence_lookup, external_evidence_lookup)
    candidate_selection_node = make_candidate_selection_node(
        llm_call, evidence_lookup, index_target_evidence=index_target_evidence
    )

    # 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — discovery
    # 모드가 candidate_planning보다 먼저 거치는 문제 발견/정의/발산/반론·결합/검증 단계
    # 노드 8종(ideation_conv_problem.py). candidate_planning/candidate_feasibility/
    # candidate_selection(위) 자체는 전혀 수정하지 않았다 — provisional_selection_node가
    # candidate_selection_node를 감싸 재사용할 뿐이다.
    problem_discovery_node = make_problem_discovery_node(
        llm_call,
        evidence_lookup,
        external_evidence_lookup,
        ground_claims,
    )
    problem_focus_selection_node = make_problem_focus_selection_node(llm_call, evidence_lookup)
    problem_definition_node = make_problem_definition_node(llm_call, evidence_lookup)
    idea_divergence_node = make_idea_divergence_node(llm_call, evidence_lookup)
    idea_conflict_and_merge_node = make_idea_conflict_and_merge_node(llm_call, evidence_lookup)
    conflict_resolution_node = make_conflict_resolution_node(llm_call)
    provisional_selection_node = make_provisional_selection_node(
        llm_call, evidence_lookup, index_target_evidence=index_target_evidence
    )
    # 용준/Claude(2026-07-28, 요청: "위원들이 결합하는 방식으로" 카드 선택 단계 제거) —
    # idea_conflict_and_merge가 조건을 충족(proceed)하면 더 이상 candidate_planning/
    # candidate_feasibility/candidate_selection(카드 나열 → 사용자 선택)을 거치지 않고,
    # 위원들이 이미 결합한 방향을 바로 provisional_idea로 채택한다(LLM 미사용,
    # ideation_conv_problem.py 참고). candidate_planning 등 기존 노드는 레거시 세션
    # 재개용으로 그대로 남겨둔다(아래 등록/엣지 변경 없음).
    provisional_from_merge_node = make_provisional_from_merge_node(index_target_evidence)
    # 용준/Claude(2026-07-30, 요청: specification_completion 흐름 추가) — evidence_lookup만
    # 주입한다(ground_claims는 이 노드가 claim 단위 grounding 대신 ref 존재 여부만 직접
    # 검사하므로 필요 없다, ideation_conv_problem.py 참고).
    specification_completion_node = make_specification_completion_node(llm_call, evidence_lookup)
    # 용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다") — 기존
    # 단일 idea_validation 노드(기획+개발 동시 1회 호출)를 expert_discussion과 같은 패턴
    # (위원마다 별도 노드 + 별도 LLM 호출)으로 분리했다.
    planning_validation_node = make_planning_validation_node(
        llm_call, evidence_lookup, external_evidence_lookup, ground_claims=ground_claims
    )
    technical_validation_node = make_technical_validation_node(
        llm_call, evidence_lookup, external_evidence_lookup, ground_claims=ground_claims
    )
    concept_confirmation_node = make_concept_confirmation_node(llm_call)

    graph.add_node("planning_question", planning_question_node)
    graph.add_node("developer_question", developer_question_node)
    graph.add_node("planning_expert_discussion", planning_discussion_node)
    graph.add_node("dev_expert_discussion", dev_discussion_node)
    graph.add_node("discussion_facilitator", discussion_facilitator_node)
    graph.add_node("canvas_update", canvas_update_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("candidate_planning", candidate_planning_node)
    graph.add_node("candidate_feasibility", candidate_feasibility_node)
    graph.add_node("candidate_selection", candidate_selection_node)
    graph.add_node("problem_discovery", problem_discovery_node)
    graph.add_node("problem_focus_selection", problem_focus_selection_node)
    graph.add_node("problem_definition", problem_definition_node)
    graph.add_node("idea_divergence", idea_divergence_node)
    graph.add_node("idea_conflict_and_merge", idea_conflict_and_merge_node)
    graph.add_node("await_conflict_resolution", _await_conflict_resolution_node)
    graph.add_node("conflict_resolution", conflict_resolution_node)
    graph.add_node("provisional_selection", provisional_selection_node)
    graph.add_node("provisional_from_merge", provisional_from_merge_node)
    graph.add_node("specification_completion", specification_completion_node)
    graph.add_node("validate_planning", planning_validation_node)
    graph.add_node("validate_technical", technical_validation_node)
    graph.add_node("concept_confirmation", concept_confirmation_node)

    graph.set_conditional_entry_point(
        _route_entry,
        {
            "candidate_planning": "candidate_planning",
            "candidate_selection": "candidate_selection",
            "planning_question": "planning_question",
            "developer_question": "developer_question",
            "planning_expert_discussion": "planning_expert_discussion",
            "dev_expert_discussion": "dev_expert_discussion",
            # 재인/Claude(2026-07-23, 아바타 페이싱 연동): _FORCED_SPEAKER_TO_NODE에 "facilitator"를
            # 추가한 것과 짝을 이루는 목적지 등록 — LangGraph의 conditional entry point는 라우터
            # 함수(_route_entry)가 반환할 수 있는 값마다 여기 등록된 목적지가 있어야 한다(없으면
            # KeyError). _route_entry 자체의 판단 로직은 그대로다.
            "discussion_facilitator": "discussion_facilitator",
            "synthesis": "synthesis",
            "problem_discovery": "problem_discovery",
            "problem_focus_selection": "problem_focus_selection",
            "conflict_resolution": "conflict_resolution",
            "provisional_selection": "provisional_selection",
            "specification_completion": "specification_completion",
            # 용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야
            # 한다") — idea_validation 기본 진입점(validate_planning)과, 위 주석과 같은
            # 이유로 forced_next_speaker="dev_expert"일 때의 강제 진입점
            # (validate_technical) 둘 다 등록해야 한다(_route_entry가 둘 중 하나를 반환).
            "validate_planning": "validate_planning",
            "validate_technical": "validate_technical",
            "concept_confirmation": "concept_confirmation",
        },
    )

    graph.add_edge("planning_question", END)
    graph.add_edge("developer_question", END)

    # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — 두 전문가 노드 모두 같은
    # 라우터(_route_next_expert_turn)로 조건부 엣지를 건다: 서로를 직접 다시 부를 수도,
    # 같은 화자가 이어 말할 수도(라우터가 검증), 진행자에게 넘길 수도 있다.
    _expert_turn_targets = {
        "planning_expert": "planning_expert_discussion",
        "dev_expert": "dev_expert_discussion",
        "facilitator": "discussion_facilitator",
        "failed": END,
    }
    graph.add_conditional_edges(
        "planning_expert_discussion", _route_next_expert_turn, _expert_turn_targets
    )
    graph.add_conditional_edges(
        "dev_expert_discussion", _route_next_expert_turn, _expert_turn_targets
    )
    graph.add_conditional_edges(
        "discussion_facilitator",
        lambda state: "failed" if state.get("phase") == "failed" else "update_canvas",
        {
            "update_canvas": "canvas_update",
            "failed": END,
        },
    )
    graph.add_conditional_edges(
        "canvas_update",
        _route_after_facilitator,
        {
            "continue_round": "planning_expert_discussion",
            "discussion_complete": END,
            "await_user_decision": END,
            "failed": END,
        },
    )
    graph.add_edge("synthesis", END)

    # discovery: candidate_planning -> candidate_feasibility(성공 시, 정지 없이 이어짐) ->
    # END(awaiting_candidate_selection으로 멈춤). candidate_selection은 선택 확정 시
    # planning_question으로(같은 요청 안에서 refinement 첫 질문까지 생성), 재추천 요청 시
    # candidate_planning으로 되돌아가고(같은 요청 안에서 새 후보 생성), 그 외에는 END.
    graph.add_conditional_edges(
        "candidate_planning",
        _route_after_candidate_planning,
        {
            "ok": "candidate_feasibility",
            # 용준/Claude(2026-07-28): 안전 폴백/방향 부족 두 경우 모두 이미 정지 phase를
            # 스스로 설정했으므로(awaiting_candidate_selection/awaiting_conflict_resolution)
            # 그래프는 더 실행할 노드 없이 END로 멈춘다.
            "fallback": END,
            "insufficient": END,
            "failed": END,
        },
    )
    graph.add_edge("candidate_feasibility", END)
    graph.add_conditional_edges(
        "candidate_selection",
        _route_after_candidate_selection,
        {
            # 신청 양식 유무와 무관하게 후보 확정 직후에는 항상 진행자가 먼저 선택 후보를
            # 요약하고 문제정의를 확인한다(목표 루프의 고정 1턴) — 두 키 모두 같은 목적지.
            "to_form_coach": "discussion_facilitator",
            "to_refinement": "discussion_facilitator",
            "regenerate": "candidate_planning",
            "await_selection": END,
            "failed": END,
        },
    )

    # 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — discovery
    # 전용 신규 경로. problem_discovery(문제 영역 생성, 정지) -> problem_focus_selection
    # (사용자 선택 해석, 정지 없이 이어짐) -> problem_definition(정지 없이 이어짐) ->
    # idea_divergence(정지 없이 이어짐) -> idea_conflict_and_merge(라운드 상한/최소 조건까지
    # 자기 자신으로 루프) -> 조건 충족 시 provisional_from_merge(정지 없이 이어짐, 아래
    # 2026-07-28 갱신 참고)로 합류, 미충족+상한 도달 시 await_conflict_resolution(정지) ->
    # 사용자 응답 -> conflict_resolution(정지 없이 이어짐, 사용자가 "검증 진행"을 명시하면
    # provisional_from_merge로, 아니면 idea_conflict_and_merge로 재진입) -> idea_validation
    # (정지 없이 이어짐, awaiting_concept_confirmation으로 멈춤) -> 사용자 응답 ->
    # concept_confirmation(정지 없이 이어짐 — 확정이면 discussion_facilitator로 합류해
    # 기존 refinement 고정 1턴을 그대로 타고, 재검토면 idea_conflict_and_merge로 되돌아간다).
    #
    # 용준/Claude(2026-07-28, 요청: "위원들이 결합하는 방식으로" 카드 선택 단계 제거) —
    # 위 경로에서 candidate_planning/candidate_feasibility/candidate_selection/
    # provisional_selection(후보 카드 나열 -> 사용자 선택)은 더 이상 거치지 않는다.
    # 이 4개 노드와 그 진입점(candidate_generation/candidate_selection phase)은 삭제하지
    # 않고 레거시 세션 재개용으로만 남겨둔다 — 아래에서 그대로 등록·배선한다.
    graph.add_edge("problem_discovery", END)
    graph.add_conditional_edges(
        "problem_focus_selection",
        _route_after_problem_focus_selection,
        {
            "regenerate": "problem_discovery",
            "capped": END,
            "ok": "problem_definition",
            "failed": END,
        },
    )
    graph.add_conditional_edges(
        "problem_definition",
        _route_after_problem_definition,
        {"ok": "idea_divergence", "failed": END},
    )
    graph.add_conditional_edges(
        "idea_divergence",
        _route_after_idea_divergence,
        {"ok": "idea_conflict_and_merge", "failed": END},
    )
    graph.add_conditional_edges(
        "idea_conflict_and_merge",
        _route_after_conflict_merge,
        {
            "continue": "idea_conflict_and_merge",
            "proceed": "provisional_from_merge",
            "ask_user": "await_conflict_resolution",
            "failed": END,
        },
    )
    graph.add_edge("await_conflict_resolution", END)
    graph.add_conditional_edges(
        "conflict_resolution",
        _route_after_conflict_resolution,
        {
            "reask": END,
            "proceed": "provisional_from_merge",
            "continue": "idea_conflict_and_merge",
            "return_to_problem_definition": "problem_definition",
        },
    )
    # 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거) — provisional_from_merge는 LLM을
    # 쓰지 않는 결정론적 노드라 정상 경로는 항상 성공하지만, active 방향이 하나도 없는
    # 방어적 케이스에서만 phase="failed"를 스스로 설정한다(그런 경우까지 validate_planning을
    # 실행하지 않도록 problem_definition/idea_divergence와 동일한 패턴으로 조건부 엣지를
    # 건다).
    graph.add_conditional_edges(
        "provisional_from_merge",
        lambda state: "failed" if state.get("phase") == "failed" else "ok",
        {"ok": "specification_completion", "failed": END},
    )
    graph.add_conditional_edges(
        "provisional_selection",
        _route_after_provisional_selection,
        {
            "validate": "specification_completion",
            "regenerate": "candidate_planning",
            "await_selection": END,
            "failed": END,
        },
    )
    # 용준/Claude(2026-07-30, 요청: "미확정 필드 확인 -> 필드별 논의 -> 구조화 저장 -> 검증
    # -> 사용자 확인" 흐름 추가) — main_features 등 설계 필드가 unknown인 동안은 이 노드가
    # 자기 자신으로 반복해서 한 번에 하나씩만 채운다(요청 4번). 모든 필수 필드가
    # proposed 이상이 되면(phase가 "idea_validation"으로 바뀜) validate_planning으로
    # 진행한다 — idea_conflict_and_merge와 동일한 "라우터가 반복 여부를, 노드가 콘텐츠
    # 생성을 담당" 원칙.
    graph.add_conditional_edges(
        "specification_completion",
        _route_after_specification_completion,
        {
            "continue": "specification_completion",
            "proceed": "validate_planning",
            "failed": END,
        },
    )
    # 용준/Claude(2026-07-28, 요청: "위원들이 순서대로 대화하는 것처럼 보여야 한다") —
    # 기획위원 검증 직후 곧바로 개발위원 검증으로 이어진다(끊김 없이 도는 호출 기준 — 실제
    # HTTP 스트리밍 경로에서 "기획위원 발언 1건에서 멈추는" 동작은 그래프 엣지가 아니라
    # ideation_conv_run.py::_drive_graph의 stop_after_expert_turn이 담당한다,
    # expert_discussion과 동일한 방식). 라우팅 결정(_route_after_idea_validation)은 두
    # 관점이 모두 준비된 뒤에만 의미가 있으므로 validate_technical에만 건다.
    graph.add_edge("validate_planning", "validate_technical")
    graph.add_conditional_edges(
        "validate_technical",
        _route_after_idea_validation,
        {
            "revise": "idea_conflict_and_merge",
            "confirm": END,
            "failed": END,
        },
    )
    graph.add_conditional_edges(
        "concept_confirmation",
        _route_after_concept_confirmation,
        {
            "to_refinement": "discussion_facilitator",
            "continue": "idea_conflict_and_merge",
            "choose_another_candidate": END,
            "return_to_problem_definition": "problem_definition",
        },
    )

    return graph.compile(checkpointer=checkpointer)
