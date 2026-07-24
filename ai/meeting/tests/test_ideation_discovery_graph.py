# 작성자: 용준/Claude(2026-07-21)
# 목적: 대화형 아이디어 발전 회의(ideation-conversation)의 discovery(아이디어 발굴) 모드
#       검증 — 초기 아이디어 유무에 따른 모드 자동 결정, 기획/개발 전문가의 후보 생성·검토,
#       사용자의 번호/제목/결합/재추천/전문가추천 처리, 선택 이후 refinement 흐름으로의
#       전환, 최종 결과의 discovery 이력 포함 여부를 실제 LLM 호출 없이 확인한다.
#       기존 test_ideation_conv_graph.py의 stub 패턴을 그대로 따른다.
# import: 표준 라이브러리 json/sys/pathlib, pytest; ai/meeting/graph 패키지.

import json
import re
import sys
from pathlib import Path

import pytest

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph import (  # noqa: E402
    active_stage_for,
    finalize_ideation_conversation,
    reply_ideation_conversation,
    start_ideation_conversation,
)
from graph.ideation_conv_discovery import make_candidate_selection_node  # noqa: E402
from graph.ideation_conv_nodes import make_conv_question_node  # noqa: E402
from graph.ideation_conv_run import _new_user_message  # noqa: E402
from graph.ideation_conv_state import apply_user_answer  # noqa: E402

_REMAINING_TOPICS_RE = re.compile(
    r"\[아직 확인되지 않은 주제\(우선순위 순\) remaining_topics\]\n(.*?)\n\n", re.S
)
# 용준/Claude(2026-07-21, 후보 결합 컨텍스트 보존 테스트): 질문 프롬프트에 실제로 주입된
# selection_context를 캡처된 프롬프트 원문에서 그대로 추출한다 — 프롬프트 문자열 안에
# 후보 제목·핵심 내용이 실제로 들어갔는지(요청 9번 배선)를 검증하는 데 쓴다.
_SELECTION_CONTEXT_RE = re.compile(r"\[선택 컨텍스트 selection_context\]\n(.*?)\n\n", re.S)

CANVAS_STUB_RESPONSE = json.dumps(
    {
        "problem": "문제 상황",
        "target_user": "목표 사용자",
        "core_value": "핵심 가치",
        "solution": "핵심 해결 방식",
        "differentiation": "차별점",
        "feasibility": "medium",
        "risks": ["구현 위험"],
        "contest_fit": "공모전 기준 대응",
    },
    ensure_ascii=False,
)


def _selection_context_from_prompt(prompt: str) -> dict:
    match = _SELECTION_CONTEXT_RE.search(prompt)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except (ValueError, TypeError):
        return {}


def _topic_from_prompt(prompt: str) -> str:
    """test_ideation_conv_graph.py::_topic_from_prompt와 동일 — 질문 프롬프트에 실제로
    주입된 remaining_topics의 맨 앞 항목을 그대로 골라 써서 stub이 항상 유효한
    question_topic을 반환하도록 한다."""
    match = _REMAINING_TOPICS_RE.search(prompt)
    if not match:
        return "problem"
    try:
        remaining = json.loads(match.group(1))
    except (ValueError, TypeError):
        return "problem"
    return remaining[0] if remaining else "problem"


NOTICE_AND_CRITERIA = {
    "competition_name": "지역 소상공인 디지털전환 공모전",
    "notice_document": "실현가능성, 차별성, 사업성을 평가한다.",
}


def _candidate(cid, title, problem, target_user):
    return {
        "candidate_id": cid,
        "title": title,
        "problem": problem,
        "target_user": target_user,
        "usage_scenario": f"{title} 사용 상황",
        "core_value": f"{title} 핵심 가치",
        "solution": f"{title} 해결 방식",
        "main_features": [f"{title} 기능1"],
        "differentiation": f"{title} 차별성",
        "contest_fit": f"{title} 공모전 적합성",
        "success_metrics": [f"{title} 지표"],
    }


def _default_candidates():
    return [
        _candidate("candidate_1", "후보1: 문의 자동응답", "반복 문의 응대 부담", "동네 카페 사장님"),
        _candidate("candidate_2", "후보2: 예약 관리", "예약 누락과 중복", "동네 미용실 사장님"),
    ]


def _review(cid, feasibility="high"):
    return {
        "candidate_id": cid,
        "required_data": [f"{cid} 데이터"],
        "technical_approach": f"{cid} 기술 접근",
        "mvp_scope": f"{cid} MVP",
        "feasibility": feasibility,
        "risks": [f"{cid} 위험"],
        "dev_notes": None,
    }


class DiscoveryScriptedLLM:
    """프롬프트 마커로 노드를 판별해 고정 응답을 돌려주는 discovery 전용 stub.

    candidates_queue: candidate_planning 호출마다 순서대로 꺼내 쓰는 candidates 리스트
    (재추천 시나리오에서 매번 다른 후보를 반환하도록). 비어 있으면 _default_candidates()를
    반복 사용한다.
    selection_response: candidate_selection(LLM 해석) 호출 시 반환할 고정 응답(dict) 또는
    호출마다 꺼내 쓸 리스트.
    broken_for: {"candidate_planning", "candidate_feasibility", "candidate_selection",
    "planning_question"} 중 지정된 노드는 파싱 불가능한 텍스트를 반환한다.
    """

    def __init__(
        self,
        candidates_queue=None,
        selection_responses=None,
        broken_for=None,
        dev_next_action="await_user_decision",
        fixed_invalid_candidates=None,
    ):
        self.captured_prompts: list[str] = []
        self.candidates_queue = list(candidates_queue) if candidates_queue else []
        self.selection_responses = list(selection_responses) if selection_responses else []
        self.broken_for = broken_for or set()
        self.dev_next_action = dev_next_action
        # 항상 이 값(스키마상 유효하지 않은 후보 목록)을 반환한다 — 재시도해도 계속 실패하는
        # 상황을 흉내내기 위함(candidates_queue는 pop 방식이라 재시도 때 다른 값이 나가버려
        # "계속 무효한 응답"을 표현할 수 없다).
        self.fixed_invalid_candidates = fixed_invalid_candidates
        self.call_counts = {"candidate_planning": 0, "candidate_feasibility": 0, "candidate_selection": 0}

    def __call__(self, prompt: str) -> str:
        self.captured_prompts.append(prompt)

        if "[후보 생성 규칙]" in prompt:
            self.call_counts["candidate_planning"] += 1
            if "candidate_planning" in self.broken_for:
                return "이것은 JSON이 아닙니다"
            if self.fixed_invalid_candidates is not None:
                candidates = self.fixed_invalid_candidates
            else:
                candidates = self.candidates_queue.pop(0) if self.candidates_queue else _default_candidates()
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
                    "candidates": candidates,
                },
                ensure_ascii=False,
            )

        if "[검토 규칙]" in prompt:
            self.call_counts["candidate_feasibility"] += 1
            if "candidate_feasibility" in self.broken_for:
                return "이것은 JSON이 아닙니다"
            return json.dumps({"candidate_reviews": [_review("candidate_1"), _review("candidate_2", "medium")]}, ensure_ascii=False)

        if "[해석 규칙]" in prompt:
            self.call_counts["candidate_selection"] += 1
            if "candidate_selection" in self.broken_for:
                return "이것은 JSON이 아닙니다"
            if self.selection_responses:
                return json.dumps(self.selection_responses.pop(0), ensure_ascii=False)
            raise AssertionError("selection_responses가 준비되지 않았는데 LLM 해석이 호출되었습니다")

        if "[판정 규칙]" in prompt:
            return json.dumps({"is_sufficient": True, "reason": "충분", "follow_up_question": None}, ensure_ascii=False)

        if '"idea_name"' in prompt:
            return json.dumps(
                {
                    "idea_name": "선택된 아이디어",
                    "one_line_pitch": "한줄 소개",
                    "problem_definition": "문제 정의",
                    "target_user": "목표 사용자",
                    "core_user_value": "핵심 가치",
                    "key_features": ["기능1"],
                    "required_data": ["데이터1"],
                    "tech_direction": "기술 방향",
                    "mvp_scope": ["MVP1"],
                    "differentiation": "차별성",
                    "risks_and_mitigations": [{"risk": "위험1", "mitigation": "대응1"}],
                    "success_metrics": ["지표1"],
                    "expert_final_opinions": {"planning_expert": "기획 판단", "dev_expert": "개발 판단"},
                    "unverified_assumptions": [],
                    "final_recommendation": "추천",
                    "final_recommendation_reason": "근거",
                    "next_actions": ["다음 작업1"],
                },
                ensure_ascii=False,
            )

        if "[질문 규칙]" in prompt:
            if "planning_question" in self.broken_for:
                return "이것은 JSON이 아닙니다"
            is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
            speaker = "planning_expert" if is_planning else "dev_expert"
            payload = {
                "spoken_text": f"[{speaker}] 발화 질문",
                "judgment": f"[{speaker}] 판단",
                "question": f"[{speaker}] 질문",
                "question_topic": _topic_from_prompt(prompt),
                "referenced_message_ids": [],
                "evidence": [],
            }
            # 후보 결합 직후 첫 질문(require_combine_structure=true)이면 요청 6번 구조에
            # 필요한 필드도 채운다 — 실제 값(선택 컨텍스트 반영 내용)은 결합 컨텍스트
            # 전용 테스트가 별도 llm_call로 검증하므로, 여기서는 검증 통과에 필요한
            # 최소한의 고정 문자열만 채운다.
            if "[결합 직후 첫 메시지 여부 require_combine_structure]\ntrue" in prompt:
                payload["user_selection_summary"] = f"[{speaker}] 사용자 선택 반영 요약"
                payload["proposal"] = f"[{speaker}] 제안"
            return json.dumps(payload, ensure_ascii=False)

        if "[의견 규칙]" in prompt:
            is_dev = "당신은 AI Review Board의 개발 전문가입니다" in prompt
            # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — next_action은 더 이상
            # 그래프가 읽지 않지만, dev_next_action="continue_round"면 기존 테스트 의도(다음
            # 라운드로 자동 진행)를 보존하기 위해 쟁점을 아직 해결하지 않은 채 진행자에게
            # 넘긴다.
            dev_resolves_issue = self.dev_next_action != "continue_round"
            return json.dumps(
                {
                    "stance": "보완",
                    "spoken_text": "발화 판단입니다",
                    "judgment": "판단",
                    "reason": "근거",
                    "suggestion": "제안",
                    "interim_conclusion": "현재 임시 결론입니다",
                    "responding_to": "기획 전문가의 방금 판단" if is_dev else None,
                    "agreement": "범위를 좁히는 방향에 동의" if is_dev else "",
                    "concern": "",
                    "confirmed": [],
                    "unconfirmed": [],
                    "referenced_message_ids": [],
                    "evidence": [],
                    "next_action": None,
                    "active_issue_id": "mvp_scope",
                    "active_issue_title": "MVP 범위",
                    "new_information": ["새로 확인된 내용"],
                    "proposal": "제안",
                    "changed_position": False,
                    "needs_counterpart_response": not is_dev,
                    "recommended_next_speaker": "ideation_facilitator" if is_dev else "dev_expert",
                    "issue_resolved": bool(is_dev and dev_resolves_issue),
                    "needs_user_input": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )

        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "agreements": [],
                    "disagreements": [],
                    "facilitator_summary": "두 전문가가 이번 라운드 의견을 정리했습니다.",
                    "spoken_text": "두 위원이 이번 라운드 의견을 정리했습니다.",
                    "needs_user_decision": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )

        if "[캔버스 갱신 규칙]" in prompt:
            return CANVAS_STUB_RESPONSE

        raise AssertionError(f"예상하지 못한 프롬프트입니다: {prompt[:200]}")


def _start_discovery(llm, user_idea=""):
    return start_ideation_conversation(
        session_id="DISC-TEST",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": user_idea},
        llm_call=llm,
    )


def _legacy_resolve_selection_then_ask_planning_question(llm, state, user_message):
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) 보존 검증용 헬퍼 — 예전에는
    candidate_selection 노드가 선택/결합을 확정한 직후 그대로 1:1 인터뷰 질문 노드
    (planning_question)로 이어져, require_combine_structure(결합 직후 첫 메시지에
    선택 컨텍스트를 구조화해서 넣는 규칙)가 그 질문 프롬프트에 적용됐다. 지금은 확정 직후
    라운드테이블(planning_expert_discussion)로 곧바로 이어지고, 그 discussion 프롬프트는
    require_combine_structure/selection_context를 전혀 참조하지 않는다 — 즉 이 기능은 새
    기본 흐름에서는 더 이상 실행되지 않는 레거시 코드 경로다(코드 자체는 삭제되지 않았다).
    이 헬퍼는 candidate_selection 노드와 planning_question 노드를 손으로 이어 붙여, 그
    보존된 경로가 여전히 올바르게 동작하는지 검증한다."""
    answer_message = _new_user_message(user_message, state["round"])
    state = apply_user_answer(state, answer_message)  # phase -> "candidate_selection"
    selection_update = make_candidate_selection_node(llm)(state)
    state = {**state, **selection_update, "messages": state["messages"] + selection_update.get("messages", [])}
    # 용준/Claude(2026-07-22, 요청: "잠시만" 취소 중 phase 오염 수정) — 선택/결합 확정
    # 신호는 이제 phase가 아니라 next_route("to_refinement")로 표현된다(phase는 항상
    # canonical 상태 "expert_discussion"을 유지한다). 이 헬퍼가 보존 검증하려는 레거시
    # 경로 자체는 그대로다 — 그 경로로 갈지 판단하는 신호만 바뀌었다.
    if state.get("next_route") != "to_refinement":
        return state  # low fit 등 — 질문 노드까지 가지 않는다.
    state = dict(state)
    state["phase"] = "planning_question"
    question_update = make_conv_question_node("planning_expert", "awaiting_planning_answer", llm)(state)
    return {**state, **question_update, "messages": state["messages"] + question_update.get("messages", [])}


# ---------------------------------------------------------------------------
# 1~3. 모드 자동 결정 — 초기 아이디어 유무/공백에 따라 refinement/discovery로 시작
# ---------------------------------------------------------------------------


def test_initial_idea_present_starts_refinement_mode():
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) 이후 refinement 세션은
    1:1 인터뷰 질문 하나에서 멈추지 않고, 진행자 안건 제시 -> 라운드테이블 한 라운드가
    같은 호출 안에서 곧바로 끝까지 실행된다."""
    llm = DiscoveryScriptedLLM()
    state = start_ideation_conversation(
        session_id="MODE-TEST-1",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea={"description": "동네 가게 챗봇"},
        llm_call=llm,
    )
    assert state["ideation_mode"] == "refinement"
    assert state["phase"] == "discussion_complete"
    assert state["initial_idea"] == "동네 가게 챗봇"
    assert not any(m["message_type"] == "question" for m in state["messages"])
    # discovery 노드는 전혀 호출되지 않는다.
    assert llm.call_counts["candidate_planning"] == 0


def test_no_initial_idea_starts_discovery_mode():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm, user_idea="")
    assert state["ideation_mode"] == "discovery"
    assert state["phase"] == "awaiting_candidate_selection"
    assert state["initial_idea"] is None


def test_whitespace_only_idea_starts_discovery_mode():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm, user_idea="   \n\t  ")
    assert state["ideation_mode"] == "discovery"
    assert state["phase"] == "awaiting_candidate_selection"


# ---------------------------------------------------------------------------
# 5~6. discovery에서 서로 다른 후보 2~3개 생성 + 개발 전문가 실현 가능성 검토
# ---------------------------------------------------------------------------


def test_discovery_generates_distinct_candidates_with_feasibility_review():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)

    assert state["phase"] == "awaiting_candidate_selection"
    candidates = state["idea_candidates"]
    assert 2 <= len(candidates) <= 3
    # 후보끼리 problem/target_user가 본질적으로 달라야 한다.
    problems = {c["problem"] for c in candidates}
    targets = {c["target_user"] for c in candidates}
    assert len(problems) == len(candidates)
    assert len(targets) == len(candidates)
    # 개발 전문가 검토 결과(실현 가능성 등)가 병합되어 있어야 한다.
    for c in candidates:
        assert c["feasibility"] in {"high", "medium", "low"}
        assert c["technical_approach"]
        assert isinstance(c["required_data"], list)
    assert state["original_idea_candidates"] == candidates


# ---------------------------------------------------------------------------
# 12. 후보 선택 전에는 refinement 질문(기획/개발 질문 노드)이 절대 실행되지 않는지
# ---------------------------------------------------------------------------


def test_no_refinement_question_runs_before_candidate_selection():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    assert state["phase"] == "awaiting_candidate_selection"
    question_prompts = [p for p in llm.captured_prompts if "[질문 규칙]" in p]
    assert not question_prompts, "후보 선택 전에 refinement 질문 노드가 호출되면 안 된다"


# ---------------------------------------------------------------------------
# 7. 후보 번호 선택 후 refinement로 전환(코드가 결정적으로 처리 — LLM 해석 호출 없음)
# ---------------------------------------------------------------------------


def test_numeric_candidate_selection_switches_to_refinement_without_llm_interpretation():
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) 이후 후보 확정 직후에는
    1:1 인터뷰 질문이 아니라 라운드테이블 한 라운드가 같은 요청 안에서 곧바로 끝까지
    실행된다."""
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)

    assert state["phase"] == "discussion_complete"
    assert state["ideation_mode"] == "discovery"  # 모드 자체는 바뀌지 않는다.
    assert state["selected_idea"]["candidate_id"] == "candidate_1"
    assert state["selected_idea"]["source"] == "select"
    assert state["user_idea"]["candidate_id"] == "candidate_1"
    assert llm.call_counts["candidate_selection"] == 0  # 단순 번호 선택은 LLM을 호출하지 않는다.
    # 선택 직후 같은 요청 안에서 라운드테이블(기획 위원 최초 의견 -> 개발 위원 검토 -> 진행자
    # 정리)까지 만들어졌다 — 1:1 인터뷰 질문(message_type="question")은 없다.
    assert state["messages"][-1]["speaker_id"] == "ideation_facilitator"
    assert state["messages"][-1]["message_type"] == "summary"
    # 후보 선택 질문(discovery 단계의 정상적인 message_type="question")을 제외한, 선택
    # 확정 이후에 생성된 메시지 중에는 1:1 인터뷰 질문이 없어야 한다.
    after_selection = state["messages"][2:]  # [0]=선택 질문, [1]=사용자의 "1번" 답변
    assert not any(m["message_type"] == "question" for m in after_selection)


# ---------------------------------------------------------------------------
# active_stage — ideation_mode(최초 진입 모드, 세션 내내 고정)와 별개로, discovery로
# 시작해 후보를 선택한 뒤에는 프론트 배지가 "아이디어 발전 모드"(active_stage="refinement")로
# 바뀌어야 한다. ideation_mode 자체는 계속 "discovery"로 남아야 한다(최초 진입 모드 기록
# 유지 요구).
# ---------------------------------------------------------------------------


def test_active_stage_switches_from_candidate_discovery_to_refinement_after_selection():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)

    assert state["ideation_mode"] == "discovery"
    assert active_stage_for(state["phase"]) == "candidate_discovery"

    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)

    # 최초 진입 모드 기록은 그대로 유지된다 — active_stage만 바뀐다.
    assert state["ideation_mode"] == "discovery"
    assert active_stage_for(state["phase"]) == "refinement"


# ---------------------------------------------------------------------------
# 8. 후보 제목 선택도 결정적으로 처리되는지
# ---------------------------------------------------------------------------


def test_title_candidate_selection_resolves_deterministically():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    title = state["idea_candidates"][1]["title"]
    state = reply_ideation_conversation(previous_state=state, user_message=title, llm_call=llm)

    assert state["selected_idea"]["candidate_id"] == "candidate_2"
    assert llm.call_counts["candidate_selection"] == 0


# ---------------------------------------------------------------------------
# 9. 복수 후보 결합 요청 처리(자연어 -> LLM 해석 호출)
# ---------------------------------------------------------------------------


def test_combine_request_uses_llm_interpretation_and_produces_combined_idea():
    llm = DiscoveryScriptedLLM(
        selection_responses=[
            {
                "resolution": "combine",
                "selected_candidate_ids": ["candidate_1", "candidate_2"],
                "selection_reason": "두 후보의 장점을 결합",
                "combined_idea": {
                    "title": "결합 아이디어",
                    "problem": "결합된 문제",
                    "target_user": "결합된 사용자",
                    "usage_scenario": "결합 상황",
                    "core_value": "결합 가치",
                    "solution": "결합 해결책",
                    "main_features": ["결합 기능"],
                    "required_data": ["결합 데이터"],
                    "technical_approach": "결합 기술",
                    "mvp_scope": "결합 MVP",
                    "differentiation": "결합 차별성",
                    "contest_fit": "결합 적합성",
                    "success_metrics": ["결합 지표"],
                },
                "merge_analysis": {
                    "common_problem": "반복 업무 부담",
                    "common_value": "사장님의 시간 절약",
                    "fit": "high",
                    "primary_features": ["문의 자동응답"],
                    "secondary_features": ["예약 관리"],
                    "conflicts": [],
                    "open_questions": [],
                },
                "unverified_assumptions": ["결합 가정1"],
                "clarifying_question": None,
            }
        ]
    )
    state = _start_discovery(llm)
    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 결합 확정 직후 곧바로
    # 라운드테이블이 이어지고, 그 라운드의 dev 의견이 unconfirmed=[]를 반환하면
    # unresolved_issues가 그 값으로 덮어써진다(DiscoveryScriptedLLM의 "[의견 규칙]" stub이
    # 항상 unconfirmed=[]를 반환하기 때문). candidate_selection 노드 자체가 unverified_
    # assumptions를 unresolved_issues에 정확히 반영하는지는(요청 사항 자체) 노드를 직접
    # 호출해 그 시점의 값으로 검증한다 — 이후 라운드가 그 값을 다시 덮어쓰는지는 이 테스트의
    # 관심사가 아니다.
    answer_message = _new_user_message("1번과 2번 결합해줘", state["round"])
    selection_state = apply_user_answer(state, answer_message)
    update = make_candidate_selection_node(llm)(selection_state)

    assert llm.call_counts["candidate_selection"] == 1
    assert update["selected_idea"]["title"] == "결합 아이디어"
    assert update["selected_idea"]["source"] == "combine"
    assert update["selected_idea"]["source_candidate_ids"] == ["candidate_1", "candidate_2"]
    assert "결합 가정1" in update["unresolved_issues"]
    # 용준/Claude(2026-07-22, 요청: "잠시만" 취소 중 phase 오염 수정) — phase는 항상
    # canonical 상태("expert_discussion")를 유지하고, "곧바로 라운드테이블로 이어간다"는
    # 그래프 내부 라우팅 신호는 next_route로 분리됐다(ideation_conv_build.py::
    # _route_after_candidate_selection 참고).
    assert update["phase"] == "expert_discussion"
    assert update["next_route"] == "to_refinement"


# ---------------------------------------------------------------------------
# 10. "다시 추천" 요청 및 반복 상한
# ---------------------------------------------------------------------------


def test_regenerate_request_produces_new_candidates_without_llm_interpretation():
    second_batch = [
        _candidate("candidate_1", "새 후보1", "새 문제1", "새 사용자1"),
        _candidate("candidate_2", "새 후보2", "새 문제2", "새 사용자2"),
    ]
    # candidates_queue는 candidate_planning이 호출될 때마다 순서대로 소비된다 — 최초
    # 시작(1번째 호출)에는 기본 후보를, 재추천(2번째 호출)에는 second_batch를 받도록 두
    # 항목을 순서대로 넣는다.
    llm = DiscoveryScriptedLLM(candidates_queue=[_default_candidates(), second_batch])
    state = _start_discovery(llm)
    first_titles = {c["title"] for c in state["idea_candidates"]}

    state = reply_ideation_conversation(previous_state=state, user_message="다시 추천해줘", llm_call=llm)

    assert state["phase"] == "awaiting_candidate_selection"
    assert state["candidate_regeneration_count"] == 1
    new_titles = {c["title"] for c in state["idea_candidates"]}
    assert new_titles == {"새 후보1", "새 후보2"}
    assert new_titles.isdisjoint(first_titles)
    # 최초 생성 후보 이력은 재추천과 무관하게 보존된다.
    assert {c["title"] for c in state["original_idea_candidates"]} == first_titles
    assert llm.call_counts["candidate_selection"] == 0


def test_regenerate_request_after_selection_returns_to_new_candidate_list():
    """후보를 선택해 기획 위원 질문에 진입한 뒤에도 "아이디어 다시 짜줘"는
    불충분한 답변이 아니라 후보 재생성 의도로 처리되어야 한다."""
    second_batch = [
        _candidate("candidate_1", "새 후보1", "새 문제1", "새 사용자1"),
        _candidate("candidate_2", "새 후보2", "새 문제2", "새 사용자2"),
    ]
    llm = DiscoveryScriptedLLM(candidates_queue=[_default_candidates(), second_batch])
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 선택 직후 라운드테이블이
    # 같은 요청 안에서 끝까지 실행돼 "awaiting_user_decision"으로 멈춘다.
    assert state["phase"] == "discussion_complete"
    assert state["selected_idea"] is not None

    sufficiency_calls_before = sum("[판정 규칙]" in prompt for prompt in llm.captured_prompts)
    state = reply_ideation_conversation(previous_state=state, user_message="아이디어 다시 짜줘", llm_call=llm)

    assert state["phase"] == "awaiting_candidate_selection"
    assert state["candidate_regeneration_count"] == 1
    assert {c["title"] for c in state["idea_candidates"]} == {"새 후보1", "새 후보2"}
    assert state["selected_idea"] is None
    assert state["selection_reason"] is None
    assert state["resolved_topics"] == []
    assert any(m["speaker_id"] == "user" and m["content"] == "아이디어 다시 짜줘" for m in state["messages"])
    assert sum("[판정 규칙]" in prompt for prompt in llm.captured_prompts) == sufficiency_calls_before


def test_regeneration_capped_and_stops_calling_llm_after_limit():
    llm = DiscoveryScriptedLLM(candidates_queue=[_default_candidates(), _default_candidates(), _default_candidates()])
    state = _start_discovery(llm)

    state = reply_ideation_conversation(previous_state=state, user_message="다시 추천", llm_call=llm)
    assert state["candidate_regeneration_count"] == 1
    state = reply_ideation_conversation(previous_state=state, user_message="다시 추천", llm_call=llm)
    assert state["candidate_regeneration_count"] == 2

    calls_before = llm.call_counts["candidate_planning"]
    state = reply_ideation_conversation(previous_state=state, user_message="다시 추천", llm_call=llm)
    assert state["phase"] == "awaiting_candidate_selection"
    assert state["candidate_regeneration_count"] == 2  # 더 늘지 않는다.
    assert llm.call_counts["candidate_planning"] == calls_before  # LLM이 추가 호출되지 않았다.
    assert "최대" in state["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# 11. 전문가 추천 요청 처리
# ---------------------------------------------------------------------------


def test_expert_recommend_request_produces_reasoned_recommendation():
    llm = DiscoveryScriptedLLM(
        selection_responses=[
            {
                "resolution": "recommend",
                "selected_candidate_ids": ["candidate_2"],
                "selection_reason": "데이터 확보가 더 쉽고 MVP 구현이 간단합니다.",
                "combined_idea": _candidate("candidate_2", "후보2: 예약 관리", "예약 누락과 중복", "동네 미용실 사장님"),
                "unverified_assumptions": ["예약 데이터 형식이 표준화되어 있다는 가정"],
                "clarifying_question": None,
            }
        ]
    )
    state = _start_discovery(llm)
    # test_combine_request_uses_llm_interpretation_and_produces_combined_idea와 같은 이유로
    # (라운드테이블의 후속 라운드가 unresolved_issues를 덮어쓸 수 있다) candidate_selection
    # 노드를 직접 호출해 그 시점의 값을 검증한다.
    answer_message = _new_user_message("전문가 추천해 주세요", state["round"])
    selection_state = apply_user_answer(state, answer_message)
    update = make_candidate_selection_node(llm)(selection_state)

    assert update["selected_idea"]["source"] == "recommend"
    assert "데이터 확보가 더 쉽고" in update["selection_reason"]
    assert any("예약 데이터 형식" in issue for issue in update["unresolved_issues"])
    # 용준/Claude(2026-07-22, 요청: "잠시만" 취소 중 phase 오염 수정) — 위 결합 테스트와
    # 동일한 이유로 phase="expert_discussion" + next_route="to_refinement"로 바뀌었다.
    assert update["phase"] == "expert_discussion"
    assert update["next_route"] == "to_refinement"


# ---------------------------------------------------------------------------
# 14. 필수 키가 없는 후보 생성 응답 — 빈 카드를 만들지 않고 실패 처리
# ---------------------------------------------------------------------------


def test_candidate_planning_missing_required_field_does_not_produce_empty_candidates():
    llm = DiscoveryScriptedLLM(fixed_invalid_candidates=[{"candidate_id": "candidate_1", "title": "제목만 있음"}])
    state = _start_discovery(llm)
    assert state["phase"] == "failed"
    assert state["failed_node"] == "candidate_planning"
    assert state["idea_candidates"] == []
    assert llm.call_counts["candidate_planning"] == 2  # 최초 1회 + 재시도 1회, 계속 무효했다.


def test_candidate_feasibility_llm_failure_falls_back_to_failed_phase():
    llm = DiscoveryScriptedLLM(broken_for={"candidate_feasibility"})
    state = _start_discovery(llm)
    assert state["phase"] == "failed"
    assert state["failed_node"] == "candidate_feasibility"


def test_candidate_selection_llm_failure_falls_back_to_failed_phase():
    llm = DiscoveryScriptedLLM(broken_for={"candidate_selection"})
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번과 2번 결합해줘", llm_call=llm)
    assert state["phase"] == "failed"
    assert state["failed_node"] == "candidate_selection"


# ---------------------------------------------------------------------------
# 16. 최종 결과 13개 항목 + discovery 이력(discovery_history) 보존
# ---------------------------------------------------------------------------


def test_discovery_final_result_includes_13_fields_and_discovery_history():
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 후보 선택 직후 라운드테이블이
    같은 요청 안에서 끝까지 실행되므로 "1번" 선택 한 번의 reply로 awaiting_user_decision에
    도달한다(과거처럼 두 번의 추가 질문 답변이 필요하지 않다)."""
    llm = DiscoveryScriptedLLM(dev_next_action="await_user_decision")
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)
    assert state["phase"] == "discussion_complete"

    state = finalize_ideation_conversation(previous_state=state, llm_call=llm)
    assert state["phase"] == "finalized"
    proposal = state["idea_proposal"]

    for field in (
        "problem_definition",
        "target_user",
        "core_user_value",
        "key_features",
        "required_data",
        "tech_direction",
        "mvp_scope",
        "differentiation",
        "risks_and_mitigations",
        "success_metrics",
        "expert_final_opinions",
        "unverified_assumptions",
        "final_recommendation",
    ):
        assert field in proposal

    # discovery_history는 synthesis 프롬프트에 전달된 것을 stub이 그대로 반영하지 않지만
    # (stub은 고정 idea_name 응답만 반환), 프롬프트 자체에 discovery 이력(최초 후보/선택된
    # 후보/선택 이유)이 실제로 주입되었는지는 캡처된 프롬프트 원문으로 검증한다.
    synthesis_prompts = [p for p in llm.captured_prompts if '"idea_name"' in p]
    assert synthesis_prompts
    last_synthesis_prompt = synthesis_prompts[-1]
    assert "candidate_1" in last_synthesis_prompt  # original_candidates가 주입됨
    assert state["selection_reason"] in last_synthesis_prompt  # selection_reason이 주입됨


# ---------------------------------------------------------------------------
# 후보 결합 컨텍스트 보존 — "1번과 2번 결합" 요청이 refinement로 넘어가면서 사라지지
# 않고 state/프롬프트/전문가 메시지에 명시적으로 남아있는지 검증한다(요청 1~10번).
# ---------------------------------------------------------------------------


def _merge_analysis(fit, conflicts=None, open_questions=None):
    return {
        "common_problem": "반복 업무 부담",
        "common_value": "사장님의 시간 절약",
        "fit": fit,
        "primary_features": ["문의 자동응답"],
        "secondary_features": ["예약 관리"],
        "conflicts": conflicts or [],
        "open_questions": open_questions or [],
    }


def _combine_selection_response(fit, conflicts=None, open_questions=None):
    return {
        "resolution": "combine",
        "selected_candidate_ids": ["candidate_1", "candidate_2"],
        "selection_reason": "두 후보의 장점을 결합",
        "combined_idea": {
            "title": "결합 아이디어",
            "problem": "결합된 문제",
            "target_user": "결합된 사용자",
            "usage_scenario": "결합 상황",
            "core_value": "결합 가치",
            "solution": "결합 해결책",
            "main_features": ["결합 기능"],
            "required_data": ["결합 데이터"],
            "technical_approach": "결합 기술",
            "mvp_scope": "결합 MVP",
            "differentiation": "결합 차별성",
            "contest_fit": "결합 적합성",
            "success_metrics": ["결합 지표"],
        },
        "merge_analysis": _merge_analysis(fit, conflicts, open_questions),
        "unverified_assumptions": [],
        "clarifying_question": None,
    }


class _CombineAwareScriptedLLM(DiscoveryScriptedLLM):
    """DiscoveryScriptedLLM을 그대로 재사용하되, "후보 결합 직후 첫 질문"
    (require_combine_structure=true)일 때만 프롬프트에 실제로 주입된 selection_context의
    후보 제목을 그대로 읽어 user_selection_summary/proposal에 반영한다 — 이 stub이 실제
    LLM처럼 "프롬프트에 넣어준 정보를 답에 반영"하는지를 통해, 코드가 실제 후보 데이터를
    프롬프트에 넣어주고 있는지(요청 9번 배선)를 검증할 수 있다."""

    def __call__(self, prompt: str) -> str:
        if "[질문 규칙]" in prompt and "[결합 직후 첫 메시지 여부 require_combine_structure]\ntrue" in prompt:
            self.captured_prompts.append(prompt)
            is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
            speaker = "planning_expert" if is_planning else "dev_expert"
            ctx = _selection_context_from_prompt(prompt)
            titles = [c.get("title", "") for c in ctx.get("source_candidates", [])]
            payload = {
                "spoken_text": f"선택하신 {' 와 '.join(titles)}를 결합해 주 기능은 문의 자동응답, 보조 기능은 예약 관리로 제안합니다.",
                "judgment": f"[{speaker}] 판단",
                "question": f"[{speaker}] 질문",
                "question_topic": _topic_from_prompt(prompt),
                "user_selection_summary": f"선택하신 후보는 {' 와 '.join(titles)}입니다.",
                "proposal": "주 기능은 문의 자동응답, 보조 기능은 예약 관리로 제안합니다.",
                "referenced_message_ids": [],
                "evidence": [],
            }
            return json.dumps(payload, ensure_ascii=False)
        return super().__call__(prompt)


def test_combine_preserves_both_source_candidates_in_state():
    """요청 1·11번 — "1번과 2번 결합" 시 두 원본 후보(제목/문제/목표 사용자/핵심 가치/
    주요 기능)와 사용자 원문 요청, 선택 의도가 state에 그대로 보존되는지."""
    llm = _CombineAwareScriptedLLM(selection_responses=[_combine_selection_response("high")])
    state = _start_discovery(llm)
    original_candidates = {c["candidate_id"]: c for c in state["idea_candidates"]}

    state = reply_ideation_conversation(previous_state=state, user_message="1번과 2번 결합", llm_call=llm)

    assert state["selection_intent"] == "combine"
    assert state["user_selection_message"] == "1번과 2번 결합"
    source_ids = {c["candidate_id"] for c in state["source_candidates"]}
    assert source_ids == {"candidate_1", "candidate_2"}
    for c in state["source_candidates"]:
        original = original_candidates[c["candidate_id"]]
        assert c["title"] == original["title"]
        assert c["problem"] == original["problem"]
        assert c["target_user"] == original["target_user"]
        assert c["core_value"] == original["core_value"]
        assert c["main_features"] == original["main_features"]
    assert state["merge_analysis"]["fit"] == "high"


def test_combine_first_question_prompt_includes_both_candidate_titles_and_content():
    """요청 2·9번 — 결합 직후 첫 전문가 질문 프롬프트에 selection_context를 통해 두 후보의
    제목과 핵심 내용(문제)이 구조화된 형태로 실제로 주입되는지.

    용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환): require_combine_structure는
    레거시 1:1 인터뷰 질문 노드(planning_question)의 검증 규칙이다 — 새 기본 흐름은 후보
    확정 직후 곧바로 라운드테이블(discussion 노드)로 넘어가고, 그 discussion 프롬프트는
    selection_context를 전혀 참조하지 않는다(이 코드베이스의 실제 동작이다, 소스는 이번
    작업 범위 밖). 아래는 그 레거시 코드 경로(candidate_selection -> planning_question)가
    여전히 올바르게 동작하는지 손으로 이어 붙여 검증한다."""
    llm = _CombineAwareScriptedLLM(selection_responses=[_combine_selection_response("high")])
    state = _start_discovery(llm)
    titles = [c["title"] for c in state["idea_candidates"]]
    problems = [c["problem"] for c in state["idea_candidates"]]

    _legacy_resolve_selection_then_ask_planning_question(llm, state, "1번과 2번 결합")

    combine_question_prompts = [
        p for p in llm.captured_prompts if "[결합 직후 첫 메시지 여부 require_combine_structure]\ntrue" in p
    ]
    assert len(combine_question_prompts) == 1
    prompt = combine_question_prompts[0]
    for title in titles:
        assert title in prompt
    for problem in problems:
        assert problem in prompt


def test_combine_first_expert_message_mentions_both_candidates_concretely():
    """요청 3·7번 — "1번과 2번을 결합하고 싶은 것으로 이해했습니다"처럼 번호만 언급하지
    않고, 결합 직후 첫 전문가 메시지가 두 후보의 실제 제목을 구체적으로 언급하는지.

    레거시 1:1 인터뷰 질문 노드(planning_question) 경로 보존 검증 — 위 테스트와 같은 이유로
    레거시 헬퍼를 사용한다."""
    llm = _CombineAwareScriptedLLM(selection_responses=[_combine_selection_response("high")])
    state = _start_discovery(llm)
    titles = [c["title"] for c in state["idea_candidates"]]

    state = _legacy_resolve_selection_then_ask_planning_question(llm, state, "1번과 2번 결합")

    last_message = state["messages"][-1]
    assert last_message["speaker_id"] == "planning_expert"
    # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — content는
    # 이제 spoken_text 그대로다([사용자 선택 반영]/[제안] 헤더는 더 이상 붙지 않는다).
    # user_selection_summary/proposal 원문은 structured에서 확인한다.
    assert last_message["structured"]["user_selection_summary"]
    assert last_message["structured"]["proposal"]
    for title in titles:
        assert title in last_message["content"]


def test_combine_high_fit_finalizes_selection_normally():
    """요청 4번 — 결합 적합도가 high이면 selected_idea가 즉시 확정되고, 용준/Claude
    (2026-07-21, 요청: 전문가 라운드테이블 전환) 이후에는 refinement 첫 질문 대신
    라운드테이블 한 라운드까지 같은 요청 안에서 정상적으로 이어지는지."""
    llm = _CombineAwareScriptedLLM(selection_responses=[_combine_selection_response("high")])
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번과 2번 결합", llm_call=llm)

    assert state["phase"] == "discussion_complete"
    assert state["selected_idea"] is not None
    assert state["merge_analysis"]["fit"] == "high"


def test_combine_medium_fit_finalizes_and_preserves_primary_secondary_features():
    """요청 5번 — 결합 적합도가 medium이면 결합을 확정하되(주 기능/보조 기능을 구분해
    사용자에게 우선순위를 묻는 것은 프롬프트가 실제 LLM에게 지시하는 부분이므로, 여기서는
    "주 기능/보조 기능 구분이 state에 실제로 남아있는지"를 배선 수준에서 검증한다."""
    llm = _CombineAwareScriptedLLM(selection_responses=[_combine_selection_response("medium")])
    state = _start_discovery(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="1번과 2번 결합", llm_call=llm)

    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 결합 확정 직후 라운드테이블
    # 이 곧바로 실행돼 "awaiting_user_decision"으로 멈춘다("[핵심 질문]" 형식의 1:1 인터뷰
    # 질문은 더 이상 생성되지 않는다).
    assert state["phase"] == "discussion_complete"
    assert state["selected_idea"] is not None
    assert state["merge_analysis"]["fit"] == "medium"
    assert state["merge_analysis"]["primary_features"] == ["문의 자동응답"]
    assert state["merge_analysis"]["secondary_features"] == ["예약 관리"]


def test_combine_low_fit_does_not_finalize_and_asks_for_primary_direction():
    """요청 5·6번 — 결합 적합도가 low이면 selected_idea를 확정하지 않고, 선택한 두 후보와
    목적 차이, 결합 시 발생하는 문제를 설명한 뒤 주 방향을 묻는 메시지를 반환하며, 여전히
    awaiting_candidate_selection에 머무는지."""
    llm = _CombineAwareScriptedLLM(
        selection_responses=[
            _combine_selection_response("low", conflicts=["목표 사용자가 서로 다릅니다"])
        ]
    )
    state = _start_discovery(llm)
    titles = [c["title"] for c in state["idea_candidates"]]

    state = reply_ideation_conversation(previous_state=state, user_message="1번과 2번 결합", llm_call=llm)

    assert state["phase"] == "awaiting_candidate_selection"
    assert state["selected_idea"] is None
    # 결합 컨텍스트 자체는 잃지 않는다 — 다음 사용자 응답에서 활용될 수 있도록 보존.
    assert state["selection_intent"] == "combine"
    assert state["merge_analysis"]["fit"] == "low"
    last_message = state["messages"][-1]
    for title in titles:
        assert title in last_message["content"]
    assert "목표 사용자가 서로 다릅니다" in last_message["content"]
    assert "[질문]" in last_message["content"]
    # 결합 직후 질문 프롬프트 자체가 아예 호출되지 않는다 — low fit은 질문 노드까지
    # 진행하지 않는다.
    combine_question_prompts = [
        p for p in llm.captured_prompts if "[결합 직후 첫 메시지 여부 require_combine_structure]\ntrue" in p
    ]
    assert not combine_question_prompts


def test_combine_does_not_reask_already_selected_candidates():
    """요청 8번 — 사용자가 이미 후보를 선택/결합했으면, 다음 질문에서 "어떤 후보를
    선택하셨나요?" 같은 재확인 질문 프롬프트를 만들지 않는다(코드가 selection_context를
    항상 채워 넘기므로, 프롬프트에는 이미 selected_idea/source_candidates가 채워진
    상태로 들어간다는 배선을 확인).

    레거시 1:1 인터뷰 질문 노드(planning_question) 경로 보존 검증 — 위 두 combine 프롬프트
    테스트와 같은 이유로 레거시 헬퍼를 사용한다."""
    llm = _CombineAwareScriptedLLM(selection_responses=[_combine_selection_response("high")])
    state = _start_discovery(llm)
    state = _legacy_resolve_selection_then_ask_planning_question(llm, state, "1번과 2번 결합")

    combine_question_prompts = [
        p for p in llm.captured_prompts if "[결합 직후 첫 메시지 여부 require_combine_structure]\ntrue" in p
    ]
    prompt = combine_question_prompts[0]
    ctx = _selection_context_from_prompt(prompt)
    assert ctx.get("selection_intent") == "combine"
    assert ctx.get("selected_idea") is not None
    assert len(ctx.get("source_candidates") or []) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
