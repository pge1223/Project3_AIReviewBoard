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
from .ideation_conv_state import IdeationConvState
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

    용준/Claude(2026-07-22, 요청: "잠시만" 재개 — 지정 위원 우선 응답): phase가
    "expert_discussion"이고 forced_next_speaker가 설정돼 있으면(reply_to_interjection이
    사용자가 지정한 위원을 강제 지정한 경우) 기본값(planning_expert_discussion) 대신 그
    위원의 노드로 바로 진입한다. forced_next_speaker=planning_expert/dev_expert는 그 노드가
    실행되자마자 리셋되므로(make_conv_discussion_node 참고) 다음 라운드에는 잔류하지 않는다.

    재인/Claude(2026-07-23, 아바타 페이싱 연동): forced_next_speaker="facilitator"(위와 같은
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
    return "failed" if state.get("phase") == "failed" else "ok"


def _route_after_candidate_selection(state: IdeationConvState) -> str:
    """candidate_selection 노드가 반환한 phase/next_route를 보고 이번 요청 안에서 더
    진행할지 정한다.
    - next_route="to_refinement": 선택/결합/추천이 확정됐다 — 같은 요청 안에서 곧바로
      refinement의 라운드테이블(planning_expert_discussion)까지 만든다(요청 4번, 사용자
      왕복을 하나 아낀다). 용준/Claude(2026-07-22, 요청: "잠시만" 취소 중 phase 오염 수정) —
      이전에는 이 신호를 phase="planning_question"으로 표현했지만, candidate_selection 직후
      바로 이어지는 노드 실행 중 취소되면 그 내부 신호값이 그대로 세션에 저장돼 재개를
      막았다. 이제 phase는 항상 canonical 상태("expert_discussion")를 유지한다.
    - "candidate_generation"(phase 자체가 이미 정확한 canonical 진입 phase다): "다시 추천" —
      같은 요청 안에서 후보를 다시 만든다.
    - 그 외("awaiting_candidate_selection" 그대로거나 재추천 상한 도달 안내): 정지(END).
    """
    phase = state.get("phase")
    if phase == "failed":
        return "failed"
    if state.get("next_route") == "to_refinement":
        return "to_refinement"
    if phase == "candidate_generation":
        return "regenerate"
    return "await_selection"


def assemble_ideation_conversation_graph(
    llm_call: LLMCall,
    checkpointer: Any | None = None,
    evidence_lookup=None,
    ground_claims=None,
    index_target_evidence=None,
    evidence_planner=None,
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
    )
    dev_discussion_node = make_conv_discussion_node(
        "dev_expert",
        llm_call=llm_call,
        evidence_lookup=evidence_lookup,
        ground_claims=ground_claims,
        evidence_planner=evidence_planner,
    )
    discussion_facilitator_node = make_discussion_facilitator_node(llm_call)
    canvas_update_node = make_canvas_update_node(llm_call)
    synthesis_node = make_conv_synthesis_node(llm_call)

    # 용준/Claude(2026-07-21): discovery(아이디어 발굴) 모드 노드 3종.
    candidate_planning_node = make_candidate_planning_node(llm_call, evidence_lookup)
    candidate_feasibility_node = make_candidate_feasibility_node(llm_call, evidence_lookup)
    candidate_selection_node = make_candidate_selection_node(
        llm_call, evidence_lookup, index_target_evidence=index_target_evidence
    )

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
        {"ok": "candidate_feasibility", "failed": END},
    )
    graph.add_edge("candidate_feasibility", END)
    graph.add_conditional_edges(
        "candidate_selection",
        _route_after_candidate_selection,
        {
            # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 후보 확정 직후에도
            # 1:1 인터뷰가 아니라 라운드테이블로 바로 들어간다(refinement 시작과 동일한
            # 원칙). 안건 제시 메시지는 ideation_conv_discovery.py의 후보 확정 지점이
            # build_roundtable_opening_message로 미리 붙여 둔다.
            "to_refinement": "planning_expert_discussion",
            "regenerate": "candidate_planning",
            "await_selection": END,
            "failed": END,
        },
    )

    return graph.compile(checkpointer=checkpointer)
