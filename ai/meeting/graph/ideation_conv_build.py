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
    make_candidate_selection_node,
    make_keyword_recommendation_node,
    make_keyword_selection_node,
    make_topic_generation_node,
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
    # pge/Claude(2026-07-27, 주제 브레인스토밍 재설계): discovery(아이디어 발굴) 모드 진입점
    # 3개 — refinement 전용 진입점(아래 4개)은 값 하나도 바꾸지 않는다.
    "keyword_generation": "keyword_recommendation",
    "keyword_selection": "keyword_selection",
    "topic_generation": "topic_generation",
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


def _route_after_keyword_selection(state: IdeationConvState) -> str:
    """pge/Claude(2026-07-27, 주제 브레인스토밍 재설계): keyword_selection 노드가 결정한
    phase에 따라 다음 노드를 고른다 — 재추천 요청이면 키워드 추천으로 되돌아가고("regenerate"),
    선택이 파싱됐으면 주제 생성으로 이어지며("continue"), 선택 0개/재추천 상한 도달이면
    안내 메시지만 남기고 멈춘다("await_selection")."""
    phase = state.get("phase")
    if phase == "keyword_generation":
        return "regenerate"
    if phase == "topic_generation":
        return "continue"
    return "await_selection"


def _route_after_candidate_selection(state: IdeationConvState) -> str:
    """후보(주제) 선택 결과에 따라 다음 노드를 고른다.

    2026-07-26 라운드테이블 재설계: 신청 양식 유무와 무관하게 후보 선택 직후에는 항상
    진행자가 먼저 선택 후보를 요약하고 문제정의를 확인하는 고정 1턴을 연다(목표 루프의
    "고정 1턴" 요건) — "to_form_coach"/"to_refinement" 두 키 모두 discussion_facilitator로
    간다(아래 엣지 매핑 참고). 신청 양식 유무는 이후 discussion_facilitator 내부에서
    (필드 채우기로 전환할지) 판단할 뿐, 진입 노드 자체를 가르지 않는다. "다시 추천"은
    주제 생성으로 돌아가며(선택된 키워드는 그대로 유지) "키워드 다시 선택"(고정 문구
    버튼)은 키워드 추천부터 다시 시작한다(pge/Claude(2026-07-27) 실측 요청) — 나머지는
    입력을 기다린다.
    """
    phase = state.get("phase")
    if phase == "failed":
        return "failed"
    if state.get("next_route") == "to_refinement" and state.get("application_form_items"):
        return "to_form_coach"
    if state.get("next_route") == "to_refinement":
        return "to_refinement"
    if phase == "keyword_generation":
        return "reselect_keywords"
    if phase == "topic_generation":
        return "regenerate"
    return "await_selection"


def assemble_ideation_conversation_graph(
    llm_call: LLMCall,
    checkpointer: Any | None = None,
    evidence_lookup=None,
    ground_claims=None,
    index_target_evidence=None,
    evidence_planner=None,
    external_evidence_lookup=None,
    trend_search_lookup=None,
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

    # pge/Claude(2026-07-27, 주제 브레인스토밍 재설계): discovery(아이디어 발굴) 모드 노드 4종.
    # keyword_recommendation에만 evidence_lookup(RAG-006)/trend_search_lookup(네이버 검색)을
    # 준다 — "무엇을 다룰지" 정하는 이 시점에만 근거 검색이 필요하다. keyword_selection은
    # LLM을 호출하지 않으므로 llm_call조차 필요 없다. topic_generation은 이미 키워드에 실린
    # 근거(rationale)만 보고 만들므로 evidence_lookup/trend_search_lookup을 받지 않는다.
    # external_evidence_lookup(RAG-007)은 discovery 흐름 어디에도 더 이상 쓰지 않는다(요청:
    # 공모전 적합성 근거는 RAG-006만으로 충분) — candidate_selection에도 안 준다(실현
    # 가능성 검토는 RAG-006만 사용, discovery.py::_apply_feasibility_review 참고).
    keyword_recommendation_node = make_keyword_recommendation_node(llm_call, evidence_lookup, trend_search_lookup)
    keyword_selection_node = make_keyword_selection_node()
    topic_generation_node = make_topic_generation_node(llm_call)
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
    graph.add_node("keyword_recommendation", keyword_recommendation_node)
    graph.add_node("keyword_selection", keyword_selection_node)
    graph.add_node("topic_generation", topic_generation_node)
    graph.add_node("candidate_selection", candidate_selection_node)

    graph.set_conditional_entry_point(
        _route_entry,
        {
            "keyword_recommendation": "keyword_recommendation",
            "keyword_selection": "keyword_selection",
            "topic_generation": "topic_generation",
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

    # pge/Claude(2026-07-27, 주제 브레인스토밍 재설계): discovery 흐름 —
    # keyword_recommendation -> END(awaiting_keyword_selection으로 멈춤, 성공/실패 모두
    # 이 노드가 이미 최종 phase를 정했으므로 조건 분기가 필요 없다) -> (사용자 응답)
    # keyword_selection -> [재추천이면 keyword_recommendation으로, 선택 파싱됐으면
    # topic_generation으로(같은 요청 안에서 이어짐), 그 외(선택 0개/재추천 상한)엔 END] ->
    # topic_generation -> END(awaiting_candidate_selection으로 멈춤, 역시 조건 분기 불필요).
    # candidate_selection은 선택 확정 시 discussion_facilitator로(같은 요청 안에서
    # refinement 첫 라운드까지 이어짐), 재추천 요청 시 topic_generation으로 되돌아가고
    # (키워드는 그대로 유지, 주제만 다시 생성), 그 외에는 END.
    graph.add_edge("keyword_recommendation", END)
    graph.add_conditional_edges(
        "keyword_selection",
        _route_after_keyword_selection,
        {
            "regenerate": "keyword_recommendation",
            "continue": "topic_generation",
            "await_selection": END,
        },
    )
    graph.add_edge("topic_generation", END)
    graph.add_conditional_edges(
        "candidate_selection",
        _route_after_candidate_selection,
        {
            # 신청 양식 유무와 무관하게 후보 확정 직후에는 항상 진행자가 먼저 선택 후보를
            # 요약하고 문제정의를 확인한다(목표 루프의 고정 1턴) — 두 키 모두 같은 목적지.
            "to_form_coach": "discussion_facilitator",
            "to_refinement": "discussion_facilitator",
            "regenerate": "topic_generation",
            # pge/Claude(2026-07-27, 실측 요청: "주제 후보 카드에서 키워드 다시 선택") —
            # 주제만 다시 만드는 "regenerate"와 달리 키워드 추천부터 다시 시작한다.
            "reselect_keywords": "keyword_recommendation",
            "await_selection": END,
            "failed": END,
        },
    )

    return graph.compile(checkpointer=checkpointer)
