# 작성자: 용준/Claude(2026-07-20, 2026-07-21 개편)
# 목적: 대화형 아이디어 발전 회의(ideation-conversation) 그래프 검증. 2026-07-21 "전문가
#       라운드테이블 전환" 이후 새 세션은 더 이상 1:1 인터뷰(기획 질문 -> 답변 -> 개발
#       질문 -> 답변)로 시작하지 않고, 진행자 안건 제시 직후 곧바로 기획/개발 위원이 서로
#       토론하는 라운드테이블로 시작한다(phase="expert_discussion"). 1:1 인터뷰 노드
#       (planning_question/developer_question, awaiting_planning_answer/
#       awaiting_developer_answer)는 코드에서 삭제되지 않고 그대로 남아 있지만, 새 세션의
#       기본 진입점에서는 더 이상 도달하지 않는다 — 이 파일의 일부 테스트는 그 "레거시지만
#       보존된" 경로를 손으로(직접 노드를 호출하거나 state를 구성해) 계속 검증한다(아래
#       _legacy_start_at_awaiting_planning_answer 참고). 나머지는 새 기본 흐름(라운드테이블)
#       자체를 검증하도록 다시 작성됐다.
# import: 표준 라이브러리 json/sys/pathlib, pytest; ai/meeting/graph 패키지.

import json
import re
import sys
from pathlib import Path

import pytest

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph import (  # noqa: E402
    finalize_ideation_conversation,
    initial_conv_state,
    reply_ideation_conversation,
    start_ideation_conversation,
)
from graph.ideation_conv_nodes import make_conv_discussion_node, make_conv_question_node  # noqa: E402

NOTICE_AND_CRITERIA = {
    "competition_name": "지역 소상공인 디지털전환 공모전",
    "notice_document": "실현가능성, 차별성을 평가한다.",
}
USER_IDEA = {"description": "소상공인이 손님 문의에 자동으로 답하는 챗봇"}

CANVAS_STUB_RESPONSE = json.dumps(
    {
        "problem": "손님 문의 응대 부담",
        "target_user": "동네 소상공인",
        "core_value": "응대 시간 절감",
        "solution": "FAQ 자동 응답 챗봇",
        "differentiation": "저비용 구축",
        "feasibility": "medium",
        "risks": ["오답 응대 위험"],
        "contest_fit": "실현가능성·차별성 기준에 대응",
    },
    ensure_ascii=False,
)

_REMAINING_TOPICS_RE = re.compile(
    r"\[아직 확인되지 않은 주제\(우선순위 순\) remaining_topics\]\n(.*?)\n\n", re.S
)


def _topic_from_prompt(prompt: str) -> str:
    """질문 프롬프트에 실제로 주입된 remaining_topics(우선순위 순, 이미 resolved_topics를
    반영해 필터링된 값)의 맨 앞 항목을 그대로 골라 쓴다 — stub이 실제 코드가 계산한 값과
    항상 일치하는 question_topic을 반환하도록 보장하기 위함이다(고정된 topic을 하드코딩하면
    두 번째 질문 호출부터 "이미 해결된 주제" 검증에 걸릴 수 있다)."""
    match = _REMAINING_TOPICS_RE.search(prompt)
    if not match:
        return "problem"
    try:
        remaining = json.loads(match.group(1))
    except (ValueError, TypeError):
        return "problem"
    return remaining[0] if remaining else "problem"


class ScriptedLLM:
    """프롬프트 내용을 보고 어느 노드가 호출했는지 판별해 고정 응답을 돌려주는 stub.

    sufficiency_queue: "[판정 규칙]"(answer_sufficiency 판정 프롬프트) 호출마다 순서대로
    꺼내 쓰는 응답 목록. 비어 있으면 항상 충분(is_sufficient=true)으로 응답한다 — 기존
    라운드 진행 테스트가 재질문 게이트 때문에 깨지지 않도록 하는 기본값이다.
    """

    def __init__(self, dev_next_action="await_user_decision", broken_for=None, sufficiency_queue=None):
        self.captured_prompts: list[str] = []
        self.dev_next_action = dev_next_action
        self.broken_for = broken_for or set()
        self.sufficiency_queue = list(sufficiency_queue) if sufficiency_queue else []

    def __call__(self, prompt: str) -> str:
        self.captured_prompts.append(prompt)

        is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
        is_dev = "당신은 AI Review Board의 개발 전문가입니다" in prompt

        if '"idea_name"' in prompt:
            return json.dumps(
                {
                    "idea_name": "동네 가게 챗봇",
                    "one_line_pitch": "소상공인 손님 문의 자동 응대",
                    "problem_definition": "반복 문의 응대 부담",
                    "target_user": "동네 소상공인",
                    "core_user_value": "문의 응대 시간 절감",
                    "key_features": ["자주 묻는 질문 자동 응답"],
                    "required_data": ["자주 묻는 질문 목록"],
                    "tech_direction": "카카오톡 챗봇빌더",
                    "mvp_scope": ["FAQ 자동 응답"],
                    "differentiation": "저비용 구축",
                    "risks_and_mitigations": [{"risk": "오답 응대", "mitigation": "FAQ 범위 밖 질문은 사람에게 이관"}],
                    "success_metrics": ["자동 응답률"],
                    "expert_final_opinions": {"planning_expert": "적합", "dev_expert": "구현 가능"},
                    "unverified_assumptions": [],
                    "final_recommendation": "추천",
                    "final_recommendation_reason": "MVP로 검증 가능",
                    "next_actions": ["FAQ 정리"],
                },
                ensure_ascii=False,
            )

        if "[판정 규칙]" in prompt:
            if "sufficiency" in self.broken_for:
                return "이것은 JSON이 아닙니다"
            if self.sufficiency_queue:
                return json.dumps(self.sufficiency_queue.pop(0), ensure_ascii=False)
            return json.dumps(
                {"is_sufficient": True, "reason": "핵심 질문에 구체적으로 답했습니다", "follow_up_question": None},
                ensure_ascii=False,
            )

        if "[제안 규칙]" in prompt:
            # 용준/Claude(2026-07-21, 요청: "모르겠다" UX 개선) — 전문가 위임(expert_delegation)
            # 제안 생성 프롬프트. stage="revision"(용준/Claude(2026-07-21, 요청: 위원 간
            # 상호 검토로 확장)이면 responding_to/revision도 채워야 검증을 통과한다.
            if "expert_delegation" in self.broken_for:
                return "이것은 JSON이 아닙니다"
            speaker = "planning_expert" if is_planning else "dev_expert"
            is_revision_stage = "[stage]\nrevision" in prompt
            payload = {
                "spoken_text": f"[{speaker}] 발화 제안 내용입니다",
                "proposal": f"[{speaker}] 임시 제안 내용입니다",
                "reason": f"[{speaker}] 제안 이유입니다",
                "assumption": f"[{speaker}] 이 방향을 기준으로 진행하겠습니다",
                "responding_to": "상대 전문가가 방금 제기한 우려" if is_revision_stage else None,
                "revision": f"[{speaker}] 그 우려를 반영해 범위를 조정했습니다" if is_revision_stage else None,
                "referenced_message_ids": [],
                "evidence": [],
            }
            return json.dumps(payload, ensure_ascii=False)

        if "[위임 검토 규칙]" in prompt:
            # 용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장) —
            # 반대 위원의 검토 stub. stance="보완"(REVISION_TRIGGER_STANCES 밖)이라 기본
            # 흐름에서는 수정 턴이 추가로 실행되지 않는다.
            reviewer = "dev_expert" if is_dev else "planning_expert"
            return json.dumps(
                {
                    "stance": "보완",
                    "spoken_text": f"[{reviewer}] 발화 검토 내용입니다",
                    "judgment": f"[{reviewer}] 검토 판단입니다",
                    "reason": f"[{reviewer}] 검토 근거입니다",
                    "responding_to": "상대 전문가의 임시 제안 내용",
                    "agreement": f"[{reviewer}] 동의 지점입니다",
                    "concern": "",
                    "recommendation": f"[{reviewer}] 이 방향을 채택해도 좋습니다",
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )

        if "[위임 정리 규칙]" in prompt:
            # 용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장) —
            # 진행자 최종 권고안 stub. 스키마에 재질문 필드가 없다.
            return json.dumps(
                {
                    "agreements": ["제안 방향에 합의"],
                    "considerations": ["추후 세부 사항은 계속 조정 가능"],
                    "final_recommendation": "이 방향으로 진행하겠습니다.",
                    "spoken_text": "이 방향으로 진행하겠습니다.",
                },
                ensure_ascii=False,
            )

        if "[질문 규칙]" in prompt:
            if is_planning and "planning_question" in self.broken_for:
                return "이것은 JSON이 아닙니다"
            if is_dev and "developer_question" in self.broken_for:
                return "이것은 JSON이 아닙니다"
            speaker = "planning_expert" if is_planning else "dev_expert"
            return json.dumps(
                {
                    "spoken_text": f"[{speaker}] 발화 핵심 질문입니다",
                    "judgment": f"[{speaker}] 현재까지 확인된 내용입니다",
                    "question": f"[{speaker}] 핵심 질문입니다",
                    "question_topic": _topic_from_prompt(prompt),
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )

        if "[의견 규칙]" in prompt:
            speaker = "dev" if is_dev else "planning"
            # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — dev_next_action은 더
            # 이상 그래프가 직접 읽지 않지만(다음 라운드 진행은 discussion_facilitator가
            # stop_reason/open_issues로 결정), 이 stub은 기존 테스트 의도를 최대한 보존하기
            # 위해 dev_next_action="continue_round"면 쟁점을 아직 해결하지 않은 채(issue_
            # resolved=False) 진행자에게 넘기고("아직 할 일이 남았다" 신호 — facilitator가
            # round<max_rounds라면 continue_round로 이어간다), 그 외에는 쟁점을 해결한 채
            # 넘긴다(진행자가 await_user_decision으로 멈춘다).
            dev_resolves_issue = self.dev_next_action != "continue_round"
            return json.dumps(
                {
                    "stance": "보완",
                    "spoken_text": f"[{speaker}] 발화 핵심 판단입니다",
                    "judgment": f"[{speaker}] 핵심 판단입니다",
                    "reason": f"[{speaker}] 판단 근거입니다",
                    "suggestion": f"[{speaker}] 개선 제안입니다",
                    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 매 발언 필수인
                    # "현재 임시 결론"(interim_conclusion).
                    "interim_conclusion": f"[{speaker}] 현재 임시 결론입니다",
                    # 용준/Claude(2026-07-21, 요청: 위원 간 실제 회의로 개편) — review 단계
                    # (is_dev) 검증이 responding_to/agreement 또는 concern 중 하나를 요구하므로
                    # 항상 채워둔다. stance="보완"은 REVISION_TRIGGER_STANCES에 없어
                    # 기획 위원의 추가 수정 응답은 필요하지 않다(기본 stub 흐름 유지).
                    "responding_to": "기획 전문가가 방금 말한 핵심 판단" if is_dev else None,
                    "agreement": f"[{speaker}] 동의 지점입니다" if is_dev else "",
                    "concern": "",
                    "confirmed": ["소상공인 손님 응대 자동화로 범위를 좁힌다"],
                    "unconfirmed": ["결제 연동 필요 여부"],
                    "referenced_message_ids": [],
                    "evidence": [],
                    "next_action": None,
                    "active_issue_id": "mvp_scope",
                    "active_issue_title": "MVP 범위",
                    "new_information": [f"[{speaker}] 새로 확인된 내용"],
                    "proposal": f"[{speaker}] 제안",
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
            # 용준/Claude(2026-07-21, 요청: 위원 간 실제 회의로 개편) — discussion_facilitator
            # 노드용 stub. needs_user_decision은 항상 false로 둬 기존 테스트의 phase 기대값
            # (dev_next_action이 그대로 결정)에 영향을 주지 않는다 — 이 노드는 phase를 절대
            # 바꾸지 않는다.
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


def _start(llm, max_rounds=3, evidence_lookup=None, application_form_items=None):
    return start_ideation_conversation(
        session_id="CONV-TEST",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        max_rounds=max_rounds,
        evidence_lookup=evidence_lookup,
        application_form_items=application_form_items,
    )


def _legacy_planning_question_state(max_rounds=3):
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환): 레거시 1:1 인터뷰 진입
    phase("planning_question")로 손수 되돌린 state. initial_conv_state()는 이제 refinement
    세션을 곧바로 "expert_discussion"으로 시작하므로, 새 세션은 더 이상 이 phase를 거치지
    않는다 — 하지만 planning_question 노드 자체는 하위 호환을 위해 삭제하지 않았으므로, 이
    state로 그 노드를 직접 호출해 여전히 올바르게 동작하는지 검증한다."""
    state = dict(initial_conv_state("CONV-TEST-LEGACY", NOTICE_AND_CRITERIA, USER_IDEA, max_rounds=max_rounds))
    state["phase"] = "planning_question"
    state["messages"] = []
    return state


def _legacy_start_at_awaiting_planning_answer(llm, max_rounds=3, evidence_lookup=None):
    """레거시 인터뷰 진입점(planning_question 노드)을 직접 호출해 phase=
    "awaiting_planning_answer"에 도달한 state를 만든다 — 과거
    start_ideation_conversation()이 만들던 것과 동일한 결과다. 새 세션의 기본 진입점에서는
    더 이상 도달하지 않는 경로지만(라운드테이블로 즉시 진입), 노드 자체는 보존돼 있으므로
    이 헬퍼로 그 보존된 경로를 계속 검증한다."""
    state = _legacy_planning_question_state(max_rounds=max_rounds)
    node = make_conv_question_node("planning_expert", "awaiting_planning_answer", llm, evidence_lookup)
    update = node(state)
    return {**state, **update, "messages": state["messages"] + update.get("messages", [])}


# ---------------------------------------------------------------------------
# 1. 세션을 시작하면 라운드테이블(진행자 안건 제시 -> 기획 위원 최초 의견 -> 개발 위원
#    검토 -> 진행자 정리)이 같은 호출 안에서 곧바로 실행되는지(전문가 라운드테이블 전환)
# ---------------------------------------------------------------------------


def test_start_runs_roundtable_immediately_without_interview_question():
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) 이후 start_ideation_conversation()은
    더 이상 기획 전문가의 1:1 질문 하나만 만들고 멈추지 않는다 — dev_next_action 기본값이
    "await_user_decision"이라 첫 라운드가 끝나면 멈춘다."""
    llm = ScriptedLLM()
    state = _start(llm)
    assert state["phase"] == "discussion_complete"
    speakers_and_types = [(m["speaker_id"], m["message_type"]) for m in state["messages"]]
    assert speakers_and_types == [
        ("ideation_facilitator", "summary"),
        ("planning_expert", "opinion"),
        ("dev_expert", "opinion"),
        ("ideation_facilitator", "summary"),
    ]
    assert state["messages"][0]["content"].startswith("오늘은 '")
    # 사용자에게 직접 던지는 인터뷰 질문(message_type="question")은 하나도 없어야 한다.
    assert not any(m["message_type"] == "question" for m in state["messages"])
    dev_prompts = [p for p in llm.captured_prompts if "당신은 AI Review Board의 개발 전문가입니다" in p]
    assert dev_prompts, "라운드테이블에서는 사용자 답변을 기다리지 않고 개발 위원도 곧바로 실행돼야 한다"


def test_roundtable_updates_idea_canvas_without_changing_meeting_phase():
    """진행자 정리 뒤 캔버스가 갱신되고 기존 회의 종료 phase는 그대로 유지된다."""
    llm = ScriptedLLM()
    state = _start(llm)

    assert state["phase"] == "discussion_complete"
    assert state["idea_canvas"] == json.loads(CANVAS_STUB_RESPONSE)
    canvas_prompts = [prompt for prompt in llm.captured_prompts if "[캔버스 갱신 규칙]" in prompt]
    assert len(canvas_prompts) == 1
    assert "[기획 전문가 최초 의견 planning_position]" in canvas_prompts[0]


# ---------------------------------------------------------------------------
# 1-2. 신청양식 항목 약한 주입 — 가은/Claude(2026-07-22, 요청: "프롬프트에서 이 방향으로
#      너무 집중되면 안 됨" — 참고 자료로만 주입되고, 항목이 없으면 discussion 프롬프트가
#      바뀌지 않는지 확인한다)
# ---------------------------------------------------------------------------

_SAMPLE_FORM_ITEMS = [
    {"field_name": "문제 정의", "description": "해결하려는 문제를 서술", "char_limit": 300},
    {"field_name": "차별성", "description": None, "char_limit": None},
]


def test_application_form_items_injected_into_discussion_prompt_when_present():
    """세션 시작 시 application_form_items를 넘기면, 라운드테이블 discussion 프롬프트
    (기획/개발 전문가 모두)에 그 항목이 JSON으로 실제로 주입돼야 한다."""
    llm = ScriptedLLM()
    _start(llm, application_form_items=_SAMPLE_FORM_ITEMS)

    discussion_prompts = [p for p in llm.captured_prompts if "[의견 규칙]" in p]
    assert discussion_prompts, "discussion 프롬프트가 하나도 호출되지 않았습니다"
    for prompt in discussion_prompts:
        assert "[신청양식 항목 application_form_items" in prompt
        assert "문제 정의" in prompt
        assert "차별성" in prompt
        assert '"char_limit": 300' in prompt


def test_application_form_items_absent_renders_null_and_does_not_change_topic_flow():
    """application_form_items를 넘기지 않으면(기존 호출부와 동일) 데이터 블록이 null로
    렌더되고, 회의 흐름(라운드 진행/phase)은 항목이 있을 때와 완전히 동일해야 한다 —
    "질문 주제·순서를 바꾸지 않는다"는 규칙이 실제로 구조적으로 지켜지는지 확인한다."""
    llm_without = ScriptedLLM()
    state_without = _start(llm_without)

    llm_with = ScriptedLLM()
    state_with = _start(llm_with, application_form_items=_SAMPLE_FORM_ITEMS)

    discussion_prompts_without = [p for p in llm_without.captured_prompts if "[의견 규칙]" in p]
    assert discussion_prompts_without
    for prompt in discussion_prompts_without:
        assert "[신청양식 항목 application_form_items" in prompt
        # 데이터 블록 값이 그대로 "null" 리터럴로 렌더돼야 한다(다른 선택적 데이터 필드와
        # 같은 관례 — REVISED_PROPOSAL_JSON 등도 없으면 "null"로 렌더된다).
        header_and_value = prompt.split("[신청양식 항목 application_form_items", 1)[1]
        value_line = header_and_value.split("]\n", 1)[1].split("\n", 1)[0].strip()
        assert value_line == "null"

    # 항목 유무와 무관하게 phase/메시지 순서(질문 주제·라운드 진행)는 동일하다.
    assert state_without["phase"] == state_with["phase"]
    types_without = [(m["speaker_id"], m["message_type"]) for m in state_without["messages"]]
    types_with = [(m["speaker_id"], m["message_type"]) for m in state_with["messages"]]
    assert types_without == types_with


# ---------------------------------------------------------------------------
# 2. 레거시 인터뷰 경로 보존: 기획 질문에 답하면 개발 전문가가 그 답변을 참조해 질문 하나만
#    만들고 멈추는지(새 세션의 기본 진입점에서는 더 이상 도달하지 않지만, 코드는 보존됨)
# ---------------------------------------------------------------------------


def test_reply_to_planning_question_triggers_only_developer_question():
    """레거시 인터뷰 경로(planning_question 노드) 보존 검증."""
    llm = ScriptedLLM()
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="타깃은 동네 카페 사장님입니다", llm_call=llm)

    assert state["phase"] == "awaiting_developer_answer"
    assert state["messages"][-2]["speaker_id"] == "user"
    assert state["messages"][-2]["content"] == "타깃은 동네 카페 사장님입니다"
    assert state["messages"][-1]["speaker_id"] == "dev_expert"
    assert state["messages"][-1]["message_type"] == "question"

    dev_prompt = next(p for p in llm.captured_prompts if "당신은 AI Review Board의 개발 전문가입니다" in p)
    assert "타깃은 동네 카페 사장님입니다" in dev_prompt


# ---------------------------------------------------------------------------
# 3. 레거시 인터뷰 경로 보존: 개발 질문에 답하면 두 전문가가 순서대로 보완 의견을 말하는지
# ---------------------------------------------------------------------------


def test_reply_to_developer_question_runs_both_experts_in_order():
    """레거시 인터뷰 경로 보존 검증 — 두 질문에 모두 답한 뒤에는(새 세션에서는 이 경로 자체를
    타지 않지만) 여전히 두 전문가가 순서대로 보완 의견을 말해야 한다."""
    llm = ScriptedLLM(dev_next_action="await_user_decision")
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="답변1", llm_call=llm)
    state = reply_ideation_conversation(previous_state=state, user_message="카카오톡 채널 API를 쓰려 합니다", llm_call=llm)

    assert state["phase"] == "discussion_complete"
    speakers = [m["speaker_id"] for m in state["messages"]]
    # dev_next_action="await_user_decision"이고 stance="보완"(REVISION_TRIGGER_STANCES
    # 밖)이라 planning_expert_revision은 끼지 않는다. 진행자 정리 메시지가 항상 마지막에
    # 추가된다.
    assert speakers[-3:] == ["planning_expert", "dev_expert", "ideation_facilitator"]
    assert state["messages"][-3]["message_type"] == "opinion"
    assert state["messages"][-2]["message_type"] == "opinion"
    assert state["messages"][-1]["message_type"] == "summary"
    assert "결제 연동 필요 여부" in state["unresolved_issues"]


# ---------------------------------------------------------------------------
# 4. 개발 위원이 continue_round를 판단하면 같은 호출 안에서 다음 라운드 discussion까지
#    자동 이어지는지(1:1 인터뷰 질문으로 돌아가지 않는지) — 전문가 라운드테이블 전환으로
#    동작 자체가 바뀐 지점.
# ---------------------------------------------------------------------------


def test_continue_round_auto_advances_to_next_discussion_round_without_stopping():
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 개발 위원이
    next_action="continue_round"를 반환하면(그리고 아직 max_rounds에 도달하지 않았으면)
    같은 그래프 호출 안에서 곧바로 다음 라운드의 기획 위원 최초 의견까지 자동 생성된다
    (1:1 인터뷰 질문으로는 절대 돌아가지 않는다). dev_next_action이 항상 "continue_round"
    이므로 max_rounds(3)에 도달한 뒤 남은 쟁점을 한 번 이월하고 정상 종료한다."""
    llm = ScriptedLLM(dev_next_action="continue_round")
    state = _start(llm, max_rounds=3)

    assert state["phase"] == "discussion_complete"
    assert state["round"] == 4
    assert len(state["discussion_rounds"]) == 4
    assert not any(m["message_type"] == "question" for m in state["messages"])


# ---------------------------------------------------------------------------
# 5. max_rounds에 도달하면 LLM이 continue_round를 반환해도 강제로 사용자 대기로 가는지
# ---------------------------------------------------------------------------


def test_max_rounds_completes_discussion_without_empty_user_wait():
    llm = ScriptedLLM(dev_next_action="continue_round")
    state = _start(llm, max_rounds=1)

    assert state["phase"] == "discussion_complete"
    assert state["round"] == 2
    assert len(state["discussion_rounds"]) == 2
    assert state["pending_question"] is None


# ---------------------------------------------------------------------------
# 6. 사용자가 확정하기 전에는 idea_proposal이 생기지 않는지 + finalize를 불러야만 생기는지
# ---------------------------------------------------------------------------


def test_idea_proposal_only_exists_after_explicit_finalize_call():
    llm = ScriptedLLM(dev_next_action="await_user_decision")
    state = _start(llm)
    assert state["phase"] == "discussion_complete"
    assert state["idea_proposal"] is None
    assert not any('"idea_name"' in p for p in llm.captured_prompts)

    state = finalize_ideation_conversation(previous_state=state, llm_call=llm)
    assert state["phase"] == "finalized"
    assert state["idea_proposal"]["idea_name"] == "동네 가게 챗봇"


# ---------------------------------------------------------------------------
# 7. awaiting_user_decision이 아닌 상태에서 finalize를 부르면 거부되는지
# ---------------------------------------------------------------------------


def test_finalize_rejected_before_awaiting_user_decision():
    llm = ScriptedLLM()
    state = _legacy_start_at_awaiting_planning_answer(llm)
    with pytest.raises(ValueError):
        finalize_ideation_conversation(previous_state=state, llm_call=llm)


# ---------------------------------------------------------------------------
# 8. 잘못된 phase(finalized/failed)에서 reply를 부르면 거부되는지
# ---------------------------------------------------------------------------


def test_reply_rejected_when_conversation_already_finalized():
    llm = ScriptedLLM(dev_next_action="await_user_decision")
    state = _start(llm)
    state = finalize_ideation_conversation(previous_state=state, llm_call=llm)

    with pytest.raises(ValueError):
        reply_ideation_conversation(previous_state=state, user_message="더 할 말 있어요", llm_call=llm)


# ---------------------------------------------------------------------------
# 9. 레거시 인터뷰 경로 보존: JSON 파싱 실패 시 폴백이 동작하는지(질문 턴)
# ---------------------------------------------------------------------------


def test_json_parse_failure_falls_back_to_failed_phase():
    """레거시 인터뷰 경로(planning_question 노드) 보존 검증 — 구조화 검증 실패 시 여전히
    phase="failed"로 안전하게 끝난다."""
    llm = ScriptedLLM(broken_for={"planning_question"})
    state = _legacy_start_at_awaiting_planning_answer(llm)
    assert state["phase"] == "failed"
    assert state["failed_node"] == "question__planning_expert"


# ---------------------------------------------------------------------------
# 10. 기획/개발 두 전문가가 서로 다른 역할 지시를 받는지(프롬프트 경계) — 라운드테이블에서는
#     사용자 답변을 기다리지 않고 한 번의 start() 호출로 둘 다 실행된다.
# ---------------------------------------------------------------------------


def test_planning_and_dev_prompts_carry_different_role_instructions():
    llm = ScriptedLLM(dev_next_action="await_user_decision")
    _start(llm)

    planning_prompts = [p for p in llm.captured_prompts if "당신은 AI Review Board의 기획 전문가입니다" in p]
    dev_prompts = [p for p in llm.captured_prompts if "당신은 AI Review Board의 개발 전문가입니다" in p]
    assert planning_prompts and dev_prompts
    assert all("당신은 AI Review Board의 개발 전문가입니다" not in p for p in planning_prompts)
    assert all("당신은 AI Review Board의 기획 전문가입니다" not in p for p in dev_prompts)


# ---------------------------------------------------------------------------
# 11. 레거시 인터뷰 경로 보존: 답변이 불충분하면 다음 전문가로 넘어가지 않고 재질문이
#     생성되는지(요청 3번) — answer_sufficiency 게이트는 여전히 awaiting_planning_answer/
#     awaiting_developer_answer(레거시 인터뷰 phase)에서만 동작한다.
# ---------------------------------------------------------------------------


def test_insufficient_answer_triggers_follow_up_instead_of_next_question():
    llm = ScriptedLLM(
        sufficiency_queue=[
            {"is_sufficient": False, "reason": "구체성이 부족합니다", "follow_up_question": "정확히 누구를 위한 서비스인가요?"}
        ]
    )
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="그런 것 같아요", llm_call=llm)

    # 재질문 후에도 여전히 기획 질문 답변을 기다리는 상태여야 하고, 개발 전문가는 아직 호출되면 안 된다.
    assert state["phase"] == "awaiting_planning_answer"
    assert state["answer_retry_count"] == 1
    assert state["messages"][-1]["speaker_id"] == "planning_expert"
    assert state["messages"][-1]["message_type"] == "question"
    assert "정확히 누구를 위한 서비스인가요?" in state["messages"][-1]["content"]
    dev_prompts = [p for p in llm.captured_prompts if "당신은 AI Review Board의 개발 전문가입니다" in p]
    assert not dev_prompts, "재질문 상황에서는 개발 전문가가 호출되면 안 된다"


# ---------------------------------------------------------------------------
# 12. 레거시 인터뷰 경로 보존: 같은 쟁점의 재질문이 상한(2회)에 도달하면 판정과 무관하게
#     다음 단계로 강제 진행되는지(요청 5번)
# ---------------------------------------------------------------------------


def test_repeated_insufficient_answers_force_progress_after_retry_cap():
    llm = ScriptedLLM(
        sufficiency_queue=[
            {"is_sufficient": False, "reason": "여전히 모호합니다", "follow_up_question": "다시 설명해 주세요"},
            {"is_sufficient": False, "reason": "여전히 모호합니다", "follow_up_question": "다시 설명해 주세요"},
            {"is_sufficient": False, "reason": "세 번째도 불충분합니다", "follow_up_question": "다시 설명해 주세요"},
        ]
    )
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="그런 것 같아요", llm_call=llm)
    assert state["phase"] == "awaiting_planning_answer"
    assert state["answer_retry_count"] == 1

    state = reply_ideation_conversation(previous_state=state, user_message="음... 아마도요", llm_call=llm)
    assert state["phase"] == "awaiting_planning_answer"
    assert state["answer_retry_count"] == 2

    # 세 번째 판정도 불충분이지만 retry_count(2)가 상한에 도달했으므로 강제로 다음 단계(개발 전문가
    # 질문)로 넘어가야 한다.
    state = reply_ideation_conversation(previous_state=state, user_message="여전히 애매해요", llm_call=llm)
    assert state["phase"] == "awaiting_developer_answer"
    assert state["answer_retry_count"] == 0
    assert any("불명확하여 다음 가정으로 진행합니다" in issue for issue in state["unresolved_issues"])


# ---------------------------------------------------------------------------
# 13. 전문가 의견이 구조화된(역할별 헤더) 형식으로 출력되는지(요청 4번) — 라운드테이블이
#     사용자 답변을 기다리지 않고 곧바로 실행되므로 start() 한 번으로 충분하다.
# ---------------------------------------------------------------------------


def test_sufficient_answer_advances_and_opinion_is_structured():
    """용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — 채팅에
    보이는 content는 이제 spoken_text 그대로이고 보고서형 헤더([기획 관점]/[근거]/[제안]/
    [임시 결론]/[확정 사항]/[미확정 사항] 등)를 전혀 포함하지 않는다. 그 필드들(judgment/
    reason/suggestion/interim_conclusion/confirmed/unconfirmed)은 여전히 structured에
    그대로 저장되어 내부 상태로 쓰인다."""
    llm = ScriptedLLM(dev_next_action="await_user_decision")
    state = _start(llm)
    assert state["phase"] == "discussion_complete"

    planning_opinion = next(
        m for m in state["messages"] if m["speaker_id"] == "planning_expert" and m["message_type"] == "opinion"
    )
    for header in ("[기획 관점]", "[기술 검토]", "[근거]", "[제안]", "[임시 결론]", "[확정 사항]", "[미확정 사항]"):
        assert header not in planning_opinion["content"]
    assert planning_opinion["content"] == "[planning] 발화 핵심 판단입니다"
    for field in ("judgment", "reason", "suggestion", "interim_conclusion", "confirmed", "unconfirmed"):
        assert field in planning_opinion["structured"]
        assert planning_opinion["structured"][field] not in (None, "")


# ---------------------------------------------------------------------------
# 14. 레거시 인터뷰 경로 보존: answer_retry_count가 전역 누적이 아니라 "현재 쟁점"에만
#     스코프되는지(리뷰 지적 1번)
# ---------------------------------------------------------------------------


def test_retry_count_is_scoped_to_current_pending_question_not_global():
    llm = ScriptedLLM(
        sufficiency_queue=[
            {"is_sufficient": False, "reason": "불명확(기획1)", "follow_up_question": "다시 설명해주세요(기획)"},
            {"is_sufficient": False, "reason": "불명확(기획2)", "follow_up_question": "다시 설명해주세요(기획)"},
            # retry_count가 상한(2)에 도달한 뒤의 세 번째 판정 — 결과와 무관하게 강제 진행되므로
            # is_sufficient=false를 반환해도 다음 단계로 넘어가야 한다.
            {"is_sufficient": False, "reason": "세 번째도 불명확", "follow_up_question": None},
            {"is_sufficient": False, "reason": "불명확(개발1)", "follow_up_question": "다시 설명해주세요(개발)"},
        ]
    )
    state = _legacy_start_at_awaiting_planning_answer(llm)

    state = reply_ideation_conversation(previous_state=state, user_message="모호1", llm_call=llm)
    assert state["phase"] == "awaiting_planning_answer"
    assert state["answer_retry_count"] == 1

    state = reply_ideation_conversation(previous_state=state, user_message="모호2", llm_call=llm)
    assert state["phase"] == "awaiting_planning_answer"
    assert state["answer_retry_count"] == 2

    # 상한 도달 -> 판정 결과와 무관하게 개발 전문가 질문으로 강제 진행 + 카운터 리셋.
    state = reply_ideation_conversation(previous_state=state, user_message="모호3", llm_call=llm)
    assert state["phase"] == "awaiting_developer_answer"
    assert state["answer_retry_count"] == 0

    # 새 쟁점(개발 전문가 질문)의 첫 모호한 답변 — 기획 쟁점에서 쌓인 누적치(2)를 이어받지
    # 않고 1부터 다시 시작해야 한다(전역 카운터가 아니라 쟁점별 카운터임을 검증).
    state = reply_ideation_conversation(previous_state=state, user_message="모호(개발)", llm_call=llm)
    assert state["phase"] == "awaiting_developer_answer"
    assert state["answer_retry_count"] == 1


# ---------------------------------------------------------------------------
# expected_answer_type 플러밍 — 실제 모델의 판정 품질(생활비 절감 답변이 부당하게
# 재질문당한 문제)은 scripts/run_ideation_sufficiency_scenarios.py(실제 LLM)로 확인한다.
# 여기서는 stub으로 "질문 노드가 만든 expected_answer_type이 state에 저장되고, sufficiency
# 프롬프트에 실제로 전달되는지"라는 배선(plumbing)만 결정적으로 검증한다. 이 값은
# 레거시 인터뷰 질문 노드(planning_question/developer_question)가 만드는 값이므로 아래
# 테스트들은 모두 레거시 경로를 사용한다.
# ---------------------------------------------------------------------------


def _llm_with_question_expected_answer_type(expected_answer_type):
    """planning_question 노드가 주어진 expected_answer_type을 반환하고, 그 외(sufficiency
    포함)는 항상 충분 판정을 내리는 최소 stub."""
    captured_sufficiency_prompts: list[str] = []

    def llm_call(prompt: str) -> str:
        if "[질문 규칙]" in prompt:
            return json.dumps(
                {
                    "spoken_text": "발화 핵심 질문입니다",
                    "judgment": "현재까지 확인된 내용입니다",
                    "question": "핵심 질문입니다",
                    "question_topic": _topic_from_prompt(prompt),
                    "expected_answer_type": expected_answer_type,
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )
        if "[판정 규칙]" in prompt:
            captured_sufficiency_prompts.append(prompt)
            return json.dumps(
                {"is_sufficient": True, "reason": "충분", "follow_up_question": None}, ensure_ascii=False
            )
        raise AssertionError(f"예상하지 못한 프롬프트입니다: {prompt[:200]}")

    llm_call.captured_sufficiency_prompts = captured_sufficiency_prompts
    return llm_call


def test_question_node_stores_expected_answer_type_in_state():
    llm = _llm_with_question_expected_answer_type("preference")
    state = _legacy_start_at_awaiting_planning_answer(llm)
    assert state["pending_expected_answer_type"] == "preference"


def test_expected_answer_type_is_forwarded_to_sufficiency_prompt():
    llm = _llm_with_question_expected_answer_type("selection")
    state = _legacy_start_at_awaiting_planning_answer(llm)
    reply_ideation_conversation(previous_state=state, user_message="1번이 더 좋다", llm_call=llm)

    assert llm.captured_sufficiency_prompts
    assert "[expected_answer_type]\nselection" in llm.captured_sufficiency_prompts[-1]


def test_invalid_expected_answer_type_falls_back_to_unknown_in_sufficiency_prompt():
    """LLM이 허용값 밖의 값(스키마 오류)을 반환해도 질문 생성 자체는 실패시키지 않고,
    sufficiency 프롬프트에는 "미상"으로 전달돼야 한다(요청 사항의 하위 호환 원칙)."""
    llm = _llm_with_question_expected_answer_type("이상한값")
    state = _legacy_start_at_awaiting_planning_answer(llm)
    assert state["pending_expected_answer_type"] is None

    reply_ideation_conversation(previous_state=state, user_message="답변", llm_call=llm)
    assert "[expected_answer_type]\n미상(알 수 없음)" in llm.captured_sufficiency_prompts[-1]


def test_pending_expected_answer_type_resets_after_advancing_to_next_question():
    """기획 질문은 expected_answer_type="preference"를 만들지만, 개발 질문은 이 키 자체를
    주지 않는다(구버전 응답을 흉내) — 이전 질문의 값이 다음 질문으로 새어 나오지 않고
    None으로 정확히 리셋되는지 확인한다."""
    question_call_count = {"n": 0}

    def llm_call(prompt: str) -> str:
        if "[질문 규칙]" in prompt:
            question_call_count["n"] += 1
            payload = {
                "spoken_text": "발화 핵심 질문입니다",
                "judgment": "현재까지 확인된 내용입니다",
                "question": "핵심 질문입니다",
                "question_topic": _topic_from_prompt(prompt),
                "referenced_message_ids": [],
                "evidence": [],
            }
            if question_call_count["n"] == 1:
                payload["expected_answer_type"] = "preference"
            return json.dumps(payload, ensure_ascii=False)
        if "[판정 규칙]" in prompt:
            return json.dumps(
                {"is_sufficient": True, "reason": "충분", "follow_up_question": None}, ensure_ascii=False
            )
        raise AssertionError(f"예상하지 못한 프롬프트입니다: {prompt[:200]}")

    state = _legacy_start_at_awaiting_planning_answer(llm_call)
    assert state["pending_expected_answer_type"] == "preference"

    state = reply_ideation_conversation(previous_state=state, user_message="1번이 더 좋다", llm_call=llm_call)
    assert state["phase"] == "awaiting_developer_answer"
    assert state.get("pending_expected_answer_type") is None


# ---------------------------------------------------------------------------
# clarification_request — "가치의 종류에는 무엇이 있나요?" 같은 용어/예시/선택지 설명
# 요청을 불충분한 답변으로 오판해 같은 질문을 반복하지 않는지 검증한다(실제 대화에서
# 확인된 문제). 여기서는 stub으로 "answer_type이 clarification_request일 때 배선이
# 정확히 동작하는지"(재질문 카운터 미증가, pending_question 유지, 설명 메시지 추가)만
# 결정적으로 검증한다 — 실제 모델이 이 세 메시지를 clarification_request로 올바르게
# *분류*하는지는 scripts/run_ideation_sufficiency_scenarios.py(실제 LLM)로 확인한다.
# 레거시 인터뷰 질문 노드를 통해 pending_question이 세팅된 상태를 전제로 하므로 레거시
# 경로를 사용한다.
# ---------------------------------------------------------------------------


def _llm_with_sufficiency_response(sufficiency_response: dict):
    """[질문 규칙] 호출에는 고정 질문을, [판정 규칙] 호출에는 주어진 sufficiency_response를
    그대로 반환하는 최소 stub. captured_prompts로 호출 여부(특히 개발 전문가 호출 여부)를
    검사할 수 있다."""
    captured_prompts: list[str] = []

    def llm_call(prompt: str) -> str:
        captured_prompts.append(prompt)
        if "[질문 규칙]" in prompt:
            return json.dumps(
                {
                    "spoken_text": "발화 핵심 질문입니다",
                    "judgment": "현재까지 확인된 내용입니다",
                    "question": "핵심 질문입니다",
                    "question_topic": _topic_from_prompt(prompt),
                    "expected_answer_type": "preference",
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )
        if "[판정 규칙]" in prompt:
            return json.dumps(sufficiency_response, ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트입니다: {prompt[:200]}")

    llm_call.captured_prompts = captured_prompts
    return llm_call


@pytest.mark.parametrize("clarification_message", ["가치의 종류에는 무엇이 있나요?", "예시를 들어주세요", "질문이 무슨 뜻인가요?"])
def test_clarification_request_message_types_are_handled_without_reasking_raw(clarification_message):
    """세 표현 모두(LLM이 clarification_request로 분류했다고 가정) 재질문 카운터를 늘리지
    않고, 설명 메시지를 추가하며, 원래 pending_question을 그대로 유지해야 한다."""
    llm = _llm_with_sufficiency_response(
        {
            "answer_type": "clarification_request",
            "reason": "사용자가 설명을 요청했습니다",
            "follow_up_question": None,
            "clarification_response": (
                "선택 가능한 핵심 가치에는 생활비 절감, 안전 향상, 편의성, 맞춤형 정보 제공, "
                "정보 접근성 등이 있습니다. 현재 두 후보를 고려하면 생활비 절감과 안전 향상이 "
                "주요 선택지입니다. 둘 중 MVP에서 가장 우선할 가치 하나를 선택해 주세요."
            ),
        }
    )
    state = _legacy_start_at_awaiting_planning_answer(llm)
    original_pending_question = state["pending_question"]

    state = reply_ideation_conversation(previous_state=state, user_message=clarification_message, llm_call=llm)

    assert state["phase"] == "awaiting_planning_answer"  # 같은 질문을 여전히 기다린다.
    assert state["pending_question"] == original_pending_question  # 원래 질문이 유지된다.
    assert state["answer_retry_count"] == 0  # 요청: 설명 요청은 재질문 카운터를 늘리지 않는다.
    assert state["messages"][-1]["speaker_id"] == "planning_expert"
    assert "생활비 절감" in state["messages"][-1]["content"]
    # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — [설명]
    # 헤더는 더 이상 붙지 않는다(clarification_response 자체가 완결된 자연스러운 응답).
    assert "[설명]" not in state["messages"][-1]["content"]
    dev_prompts = [p for p in llm.captured_prompts if "당신은 AI Review Board의 개발 전문가입니다" in p]
    assert not dev_prompts, "설명 요청 상황에서는 개발 전문가가 호출되면 안 된다"


def test_clarification_request_then_sufficient_answer_advances_normally():
    """설명을 들은 뒤 사용자가 실제로 답하면(다음 reply 호출에서 answer_type="answer")
    정상적으로 다음 단계(개발 전문가 질문)까지 진행돼야 한다 — 요청: 설명 후 선택 질문
    하나만 다시 제시하고, 그 질문에 답하면 정상 진행."""
    responses = [
        {
            "answer_type": "clarification_request",
            "reason": "설명 요청",
            "follow_up_question": None,
            "clarification_response": "선택지는 생활비 절감과 안전 향상입니다. 어느 쪽을 우선하시겠어요?",
        },
        {"answer_type": "answer", "reason": "충분", "follow_up_question": None, "clarification_response": None},
    ]

    def llm_call(prompt: str) -> str:
        if "[질문 규칙]" in prompt:
            return json.dumps(
                {
                    "spoken_text": "발화 핵심 질문입니다",
                    "judgment": "현재까지 확인된 내용입니다",
                    "question": "핵심 질문입니다",
                    "question_topic": _topic_from_prompt(prompt),
                    "expected_answer_type": "preference",
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )
        if "[판정 규칙]" in prompt:
            return json.dumps(responses.pop(0), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트입니다: {prompt[:200]}")

    state = _legacy_start_at_awaiting_planning_answer(llm_call)
    state = reply_ideation_conversation(previous_state=state, user_message="가치의 종류가 뭔가요?", llm_call=llm_call)
    assert state["phase"] == "awaiting_planning_answer"
    assert state["answer_retry_count"] == 0

    state = reply_ideation_conversation(previous_state=state, user_message="생활비 절감이 중요합니다", llm_call=llm_call)
    assert state["phase"] == "awaiting_developer_answer"  # 설명 후 정상적으로 다음 단계로 진행.


def test_answer_type_answer_advances_without_retry():
    llm = _llm_with_sufficiency_response(
        {"answer_type": "answer", "reason": "명확히 선택함", "follow_up_question": None, "clarification_response": None}
    )
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="생활비 절감이 중요합니다", llm_call=llm)
    assert state["phase"] == "awaiting_developer_answer"
    assert state["answer_retry_count"] == 0


def test_answer_type_insufficient_answer_triggers_follow_up_and_increments_retry():
    llm = _llm_with_sufficiency_response(
        {
            "answer_type": "insufficient_answer",
            "reason": "우선순위를 정하지 않았습니다",
            "follow_up_question": "둘 중 무엇을 더 우선하시겠어요?",
            "clarification_response": None,
        }
    )
    state = _legacy_start_at_awaiting_planning_answer(llm)
    # "잘 모르겠습니다"류 표현은 이제 결정적 규칙으로 expert_delegation으로 먼저 분류되므로
    # (별도 테스트로 검증한다), 순수한 insufficient_answer 경로를 확인하려면 위임 표현이
    # 아닌 모호한 답변을 보낸다.
    state = reply_ideation_conversation(previous_state=state, user_message="그런 것 같기도 하고 아닌 것 같기도 해요", llm_call=llm)
    assert state["phase"] == "awaiting_planning_answer"
    assert state["answer_retry_count"] == 1
    assert "둘 중 무엇을 더 우선하시겠어요?" in state["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# 15. 레거시 인터뷰 경로 보존: answer_retry_count 필드가 없는 구버전 저장 state를 재개해도
#     KeyError 없이 동작하는지(리뷰 지적 2번 — 배포 전에 저장된 세션을 재개하는 상황을 흉내낸다)
# ---------------------------------------------------------------------------


def test_reply_works_when_previous_state_missing_answer_retry_count_field():
    llm = ScriptedLLM()
    state = _legacy_start_at_awaiting_planning_answer(llm)
    legacy_state = dict(state)
    del legacy_state["answer_retry_count"]

    new_state = reply_ideation_conversation(
        previous_state=legacy_state, user_message="타깃은 동네 카페 사장님입니다", llm_call=llm
    )
    assert new_state["phase"] == "awaiting_developer_answer"
    assert new_state["answer_retry_count"] == 0


# ---------------------------------------------------------------------------
# 16. 구조화 응답이 깨졌을 때(리뷰 지적 4번): 필수 키 누락 / 배열 대신 문자열 / 빈 응답 /
#     JSON 앞뒤 설명문 / sufficiency 호출 실패
# ---------------------------------------------------------------------------


def test_question_node_rejects_missing_keys_instead_of_producing_empty_card():
    """레거시 인터뷰 경로(planning_question 노드) 보존 검증 — judgment/question 키가 아예
    없는 응답은 빈 카드를 만들지 않고(요청 7번) 재시도 1회 후에도 여전히 비어 있으면
    phase="failed"로 안전하게 끝난다. 이전 실사용에서 빈 카드가 화면에 보였던 문제(요청
    7번 배경)를 재현하지 않는지 검증한다."""

    call_count = {"n": 0}

    def llm(prompt: str) -> str:
        if "[질문 규칙]" in prompt:
            call_count["n"] += 1
            return json.dumps({"referenced_message_ids": [], "evidence": []}, ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:100]}")

    state = _legacy_start_at_awaiting_planning_answer(llm)
    assert state["phase"] == "failed"
    assert state["failed_node"] == "question__planning_expert"
    assert state["messages"] == []  # 빈 content의 메시지가 저장되지 않았다.
    assert state["pending_question"] is None  # 빈 pending_question이 저장되지 않았다.
    assert call_count["n"] == 2  # 최초 1회 + 재시도 1회, 총 2회 시도했다.


def test_discussion_node_handles_non_list_confirmed_and_unconfirmed_safely():
    """confirmed/unconfirmed가 배열이 아니라 문자열로 오면(타입 오류) consensus/unresolved_issues가
    그 문자열의 글자 단위로 쪼개져 오염되면 안 된다. discussion 노드는 이제 라운드테이블
    진입 즉시 실행되므로 start() 한 번으로 충분하다."""

    def llm(prompt: str) -> str:
        if "[의견 규칙]" in prompt:
            is_dev = "당신은 AI Review Board의 개발 전문가입니다" in prompt
            return json.dumps(
                {
                    "stance": "보완",
                    "spoken_text": "발화 판단입니다",
                    "judgment": "판단",
                    "reason": "근거",
                    "suggestion": "제안",
                    "interim_conclusion": "임시 결론",
                    "responding_to": "기획 전문가의 방금 판단" if is_dev else None,
                    "agreement": "범위를 좁히는 방향에 동의" if is_dev else "",
                    "concern": "",
                    "confirmed": "이것은 배열이 아니라 문자열입니다",
                    "unconfirmed": "이것도 배열이 아니라 문자열입니다",
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
                    "issue_resolved": bool(is_dev),
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
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:100]}")

    state = _start(llm)

    assert state["phase"] == "discussion_complete"
    assert state["consensus"] == []
    assert state["unresolved_issues"] == []
    # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) —
    # confirmed/unconfirmed는 더 이상 content에 [확정 사항]/[미확정 사항] 헤더로 붙지 않고
    # structured에만 저장된다. 문자열이 잘못된 타입으로 와도(비-배열) 글자 단위로 쪼개져
    # 오염되지 않고 안전하게 빈 배열로 정규화되는지는 structured로 확인한다.
    for m in state["messages"]:
        if m["message_type"] == "opinion":
            assert m["structured"]["confirmed"] == []
            assert m["structured"]["unconfirmed"] == []


def test_node_recovers_when_json_is_wrapped_in_explanatory_prose():
    """레거시 인터뷰 경로(planning_question 노드) 보존 검증 — 코드블록 없이 JSON 앞뒤로
    설명문이 붙어 있어도(parse_json_response의 방어적 추출로) 질문 노드가 실패 처리되지
    않고 정상적으로 질문을 만든다."""

    def llm(prompt: str) -> str:
        if "[질문 규칙]" in prompt:
            payload = json.dumps(
                {
                    "spoken_text": "발화 핵심 질문입니다",
                    "judgment": "판단",
                    "question": "핵심 질문",
                    "question_topic": "problem",
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )
            return f"물론이죠! 요청하신 질문은 다음과 같습니다:\n{payload}\n도움이 되었길 바랍니다."
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:100]}")

    state = _legacy_start_at_awaiting_planning_answer(llm)
    assert state["phase"] == "awaiting_planning_answer"
    assert state["failed_node"] is None
    assert "핵심 질문" in state["messages"][0]["content"]


def test_empty_discussion_response_falls_back_to_safe_expert_judgment():
    """discussion의 빈 응답은 회의 실패가 아니라 안전한 전문가 판단으로 복구한다."""

    def llm(prompt: str) -> str:
        return ""

    state = initial_conv_state(
        session_id="CONV-TEST-EMPTY-RESPONSE",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
    )
    update = make_conv_discussion_node("planning_expert", llm)(state)
    assert update.get("phase") != "failed"
    assert update["previous_speaker"] == "planning_expert"
    assert "전문가 판단으로 진행" in update["messages"][0]["content"]


def test_sufficiency_call_failure_fails_open_and_conversation_still_progresses():
    """레거시 인터뷰 경로 보존 검증 — sufficiency 판정 호출 자체가 파싱 실패하면(인프라
    문제) 회의를 막지 않고 충분함으로 간주해 정상 진행한다(요청: 판정 호출 실패가 핵심
    콘텐츠 생성 실패처럼 전체를 막지 않아야 함)."""
    llm = ScriptedLLM(broken_for={"sufficiency"})
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="타깃은 동네 카페 사장님입니다", llm_call=llm)

    assert state["phase"] == "awaiting_developer_answer"
    assert state["failed_node"] is None
    assert state["answer_retry_count"] == 0


def test_discussion_node_rejects_blank_judgment_instead_of_producing_empty_card():
    """빈 LLM 카드는 저장하지 않고 서버가 만든 안전 메시지로 교체한다."""

    def llm(prompt: str) -> str:
        if "[의견 규칙]" in prompt:
            is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
            if is_planning:
                # 기획 전문가 응답만 의도적으로 비운다.
                return json.dumps(
                    {"stance": "보완", "judgment": "", "reason": "   ", "referenced_message_ids": [], "evidence": []},
                    ensure_ascii=False,
                )
            raise AssertionError("직접 호출 테스트에서는 개발 전문가를 호출하지 않는다")
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:100]}")

    state = initial_conv_state("BLANK-JUDGMENT", NOTICE_AND_CRITERIA, USER_IDEA)
    update = make_conv_discussion_node("planning_expert", llm)(state)
    message = update["messages"][0]
    assert update.get("phase") != "failed"
    assert message["content"]
    assert "문제 정의를 구체화" in message["structured"]["judgment"]


# ---------------------------------------------------------------------------
# 17. 종합 응답의 13개 항목 + 부가 필드가 노드/상태를 거치며 하나도 누락되지 않는지
#     (리뷰 지적 3번 — make_conv_synthesis_node는 raw JSON을 그대로 idea_proposal에
#     저장하므로, 필드 화이트리스트나 고정 Pydantic 스키마가 중간에 끼어들지 않는지 검증)
# ---------------------------------------------------------------------------

_SYNTHESIS_13_FIELDS = [
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
]


def test_idea_proposal_preserves_all_13_required_fields_verbatim():
    llm = ScriptedLLM(dev_next_action="await_user_decision")
    state = _start(llm)
    state = finalize_ideation_conversation(previous_state=state, llm_call=llm)

    assert state["phase"] == "finalized"
    proposal = state["idea_proposal"]
    for field in _SYNTHESIS_13_FIELDS:
        assert field in proposal, f"{field} 가 idea_proposal에서 누락되었습니다"
    # ScriptedLLM이 넣어둔 부가 필드(idea_name 등)도 그대로 남아 있어야 한다 — 중간에
    # 화이트리스트/고정 스키마가 필드를 걸러내지 않는다는 증거.
    assert proposal["idea_name"] == "동네 가게 챗봇"
    assert proposal["expert_final_opinions"] == {"planning_expert": "적합", "dev_expert": "구현 가능"}
    assert proposal["risks_and_mitigations"] == [{"risk": "오답 응대", "mitigation": "FAQ 범위 밖 질문은 사람에게 이관"}]


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-21, 요청: "모르겠다" UX 개선) — 사용자가 전문가 질문에 "모르겠다"류
# 표현으로 답했을 때 같은 질문을 반복하지 않고, 담당 전문가가 자신의 평가 범위 안에서
# 임시 가정을 제안한 뒤 다음 단계로 진행하는지 검증한다. 이 위임 흐름은 여전히
# awaiting_planning_answer/awaiting_developer_answer(레거시 인터뷰 phase)에서만 동작하므로
# (PHASE_TO_PENDING_PERSONA), 아래 테스트들은 모두 레거시 경로로 시작한다.
# ---------------------------------------------------------------------------


def test_expert_delegation_on_planning_question_produces_proposal_and_advances():
    """시나리오 1 — 기획 위원 질문 -> "잘 모르겠어": 같은 질문을 반복하지 않고, 기획
    전문가의 제안 메시지가 생성되며, 개발 위원 질문으로 진행되는지."""
    llm = ScriptedLLM(dev_next_action="await_user_decision")
    state = _legacy_start_at_awaiting_planning_answer(llm)
    assert state["phase"] == "awaiting_planning_answer"
    original_question_content = state["messages"][-1]["content"]

    state = reply_ideation_conversation(previous_state=state, user_message="잘 모르겠어", llm_call=llm)

    assert state["phase"] == "awaiting_developer_answer"
    assert state["answer_retry_count"] == 0
    # 용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장) — 단일
    # 위원 제안으로 끝나지 않고 [사용자 답, 담당(기획) 제안, 반대(개발) 검토, 진행자 권고안,
    # 다음 질문(개발)] 순서로 이어진다. stance="보완"(REVISION_TRIGGER_STANCES 밖)이라
    # planning_expert의 수정 턴은 끼지 않는다.
    speakers_and_types = [(m["speaker_id"], m["message_type"]) for m in state["messages"]]
    assert speakers_and_types[-5:] == [
        ("user", "answer"),
        ("planning_expert", "opinion"),
        ("dev_expert", "opinion"),
        ("ideation_facilitator", "summary"),
        ("dev_expert", "question"),
    ]
    proposal_message = next(
        m for m in state["messages"] if m["speaker_id"] == "planning_expert" and m["message_type"] == "opinion"
    )
    # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — content는
    # 이제 spoken_text + 고정 안내 문구다. proposal/reason/assumption은 structured에서
    # 확인한다(예전에는 [기획 전문가 제안]/[제안 이유]/[임시 가정] 헤더로 content에 있었다).
    assert proposal_message["content"].startswith("[planning_expert] 발화 제안 내용입니다")
    assert "다른 방향을 제시하면" in proposal_message["content"]
    assert proposal_message["structured"]["proposal"] == "[planning_expert] 임시 제안 내용입니다"
    assert proposal_message["structured"]["reason"] == "[planning_expert] 제안 이유입니다"
    review_message = next(
        m for m in state["messages"] if m["speaker_id"] == "dev_expert" and m["message_type"] == "opinion"
    )
    assert review_message["content"] == "[dev_expert] 발화 검토 내용입니다"
    assert review_message["structured"]["recommendation"] == "[dev_expert] 이 방향을 채택해도 좋습니다"
    facilitator_message = next(m for m in state["messages"] if m["speaker_id"] == "ideation_facilitator")
    assert facilitator_message["content"].startswith("이 방향으로 진행하겠습니다.")
    assert facilitator_message["message_type"] == "summary"
    # 결정적 규칙으로 먼저 감지되어 sufficiency(LLM) 판정 호출 자체를 건너뛰었다.
    assert not [p for p in llm.captured_prompts if "[판정 규칙]" in p]
    # 동일한 기획 질문이 그대로 반복되지 않고 새 질문(개발 전문가)이 생성됐다.
    assert state["messages"][-1]["content"] != original_question_content
    assert any("불명확하여 다음 가정으로 진행합니다" not in issue for issue in state["unresolved_issues"])


def test_expert_delegation_on_developer_question_produces_mvp_proposal_and_advances():
    """시나리오 2 — 개발 위원 질문 -> "어떤 기술이 좋은지 모르겠어요": 개발 위원이 MVP
    기술 방향을 제안하고, 같은 질문을 반복하지 않고 다음 단계(두 전문가 의견)로 진행되는지.
    이 표현은 결정적 규칙("어떤 ~이 좋은지 모르겠")으로 바로 잡히므로 sufficiency LLM
    호출 자체가 생략된다."""
    llm = ScriptedLLM(dev_next_action="await_user_decision")
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="답변1", llm_call=llm)
    assert state["phase"] == "awaiting_developer_answer"
    original_question_content = state["messages"][-1]["content"]
    # "답변1"(기획 질문에 대한 정상 답변)은 그 자체로 한 번의 sufficiency 판정을 거친다 —
    # 이번 검증 대상은 "두 번째" 답변(개발 질문에 대한 위임 표현)이 sufficiency LLM을
    # 새로 호출하지 않는지이므로, 이 시점 이후의 호출 수만 비교한다.
    sufficiency_calls_before = len([p for p in llm.captured_prompts if "[판정 규칙]" in p])

    state = reply_ideation_conversation(previous_state=state, user_message="어떤 기술이 좋은지 모르겠어요", llm_call=llm)

    sufficiency_calls_after = len([p for p in llm.captured_prompts if "[판정 규칙]" in p])
    assert sufficiency_calls_after == sufficiency_calls_before
    assert state["phase"] == "discussion_complete"
    assert state["answer_retry_count"] == 0
    dev_delegation_messages = [
        m for m in state["messages"] if m["speaker_id"] == "dev_expert" and m["message_type"] == "opinion"
    ]
    assert dev_delegation_messages
    assert dev_delegation_messages[0]["structured"]["proposal"]
    assert state["messages"][-2]["content"] != original_question_content or state["messages"][-1]["content"] != original_question_content


def test_delegation_phrase_not_deterministically_matched_falls_back_to_llm_classification():
    """개발 위원 질문에서, 결정적 규칙으로는 잡히지 않는 위임 표현이 LLM
    판정(answer_type="expert_delegation")을 거쳐서도 정상적으로 위임 처리되는지 검증한다
    (결정적 규칙 우선 + LLM 판정 fallback 두 경로가 모두 동작함을 별도로 증명). 첫 번째
    답변("답변1")은 정상적인 answer로 처리되어야 하므로, sufficiency_queue 첫 항목을
    answer로 채워 둔다(큐는 호출 순서대로 소비된다)."""
    llm = ScriptedLLM(
        dev_next_action="await_user_decision",
        sufficiency_queue=[
            {"answer_type": "answer", "reason": "충분", "follow_up_question": None, "clarification_response": None},
            {"answer_type": "expert_delegation"},
        ],
    )
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="답변1", llm_call=llm)
    assert state["phase"] == "awaiting_developer_answer"

    # 결정적 규칙에 걸리지 않는 문장(길이 제한 초과 + 정형화되지 않은 표현)이지만 실제
    # 의미는 위임이다 — LLM 판정이 잡아야 한다.
    long_delegation_text = "이 부분은 제가 기술을 잘 몰라서 어떻게 접근해야 할지 잘 판단이 서지 않네요"
    state = reply_ideation_conversation(previous_state=state, user_message=long_delegation_text, llm_call=llm)

    assert llm.sufficiency_queue == []  # 두 항목 모두 소비됐다.
    sufficiency_prompts = [p for p in llm.captured_prompts if "[판정 규칙]" in p]
    assert sufficiency_prompts  # 이번에는 LLM 판정 호출이 실제로 일어났다.
    assert state["phase"] == "discussion_complete"
    assert state["answer_retry_count"] == 0
    assert any(
        m["speaker_id"] == "dev_expert" and m["message_type"] == "opinion" and m["structured"].get("proposal")
        for m in state["messages"]
    )


def test_mvp_meaning_question_is_clarification_request_not_expert_delegation():
    """시나리오 3 — "MVP가 무슨 뜻인지 모르겠어요"는 전문가 위임이 아니라 기존
    clarification_request로 처리되어 원래 질문을 유지해야 한다."""
    llm = ScriptedLLM(
        sufficiency_queue=[
            {
                "answer_type": "clarification_request",
                "reason": "용어 설명 요청",
                "follow_up_question": None,
                "clarification_response": "MVP는 최소 기능 제품을 뜻합니다. 다시 핵심 질문을 여쭤봅니다: 우선순위가 무엇인가요?",
            }
        ]
    )
    state = _legacy_start_at_awaiting_planning_answer(llm)
    original_pending_question = state["pending_question"]

    state = reply_ideation_conversation(previous_state=state, user_message="MVP가 무슨 뜻인지 모르겠어요", llm_call=llm)

    assert state["phase"] == "awaiting_planning_answer"  # 같은 질문을 여전히 기다린다.
    assert state["pending_question"] == original_pending_question
    assert state["answer_retry_count"] == 0
    assert "MVP는 최소 기능 제품" in state["messages"][-1]["content"]
    # sufficiency LLM은 실제로 호출됐다(결정적 규칙이 용어 질문 표지를 보고 위임으로 보지
    # 않았으므로 정상적으로 LLM 판정으로 넘어갔다) — 위임 제안 프롬프트는 호출되지 않았다.
    assert not [p for p in llm.captured_prompts if "[제안 규칙]" in p]


def test_explicit_expert_recommendation_request_triggers_delegation():
    """시나리오 4 — "잘 모르겠으니 전문가가 추천해줘"는 expert_delegation으로 처리된다."""
    llm = ScriptedLLM(dev_next_action="await_user_decision")
    state = _legacy_start_at_awaiting_planning_answer(llm)

    state = reply_ideation_conversation(
        previous_state=state, user_message="잘 모르겠으니 전문가가 추천해줘", llm_call=llm
    )

    assert state["phase"] == "awaiting_developer_answer"
    assert any(
        m["speaker_id"] == "planning_expert" and m["message_type"] == "opinion" and m["structured"].get("proposal")
        for m in state["messages"]
    )


def test_meaningless_answer_still_treated_as_insufficient_answer():
    """시나리오 5 — 의미 없는 답변("그냥")은 기존 insufficient_answer 재질문 로직을
    그대로 유지해야 한다(전문가 위임으로 오판하지 않는다)."""
    llm = ScriptedLLM(
        sufficiency_queue=[
            {"answer_type": "insufficient_answer", "reason": "질문과 무관한 답변입니다", "follow_up_question": "다시 여쭤봅니다 — 핵심 사용자가 누구인가요?"}
        ]
    )
    state = _legacy_start_at_awaiting_planning_answer(llm)

    state = reply_ideation_conversation(previous_state=state, user_message="그냥", llm_call=llm)

    assert state["phase"] == "awaiting_planning_answer"
    assert state["answer_retry_count"] == 1
    assert "다시 여쭤봅니다 — 핵심 사용자가 누구인가요?" in state["messages"][-1]["content"]
    assert not [p for p in llm.captured_prompts if "[제안 규칙]" in p]


def test_expert_delegation_content_is_visible_to_next_question_and_synthesis_prompts():
    """시나리오 7 — 전문가가 제안한 내용을 이후 질문 프롬프트(conversation_context)와
    최종 종합(synthesis) 프롬프트가 참조할 수 있는지 확인한다."""
    llm = ScriptedLLM(dev_next_action="await_user_decision")
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="잘 모르겠어", llm_call=llm)
    assert state["phase"] == "awaiting_developer_answer"

    dev_question_prompts = [p for p in llm.captured_prompts if "[질문 규칙]" in p and "개발 전문가입니다" in p]
    assert dev_question_prompts
    # ScriptedLLM의 "[제안 규칙]" 스텁이 만든 proposal 문자열 자체를 그대로 찾는다(요청
    # 8번 — conversation_context의 recent_messages를 통해 이후 질문 프롬프트가 전문가
    # 제안 내용을 실제로 볼 수 있어야 한다).
    assert "[planning_expert] 임시 제안 내용입니다" in dev_question_prompts[-1]

    state = reply_ideation_conversation(previous_state=state, user_message="카카오톡 채널 API를 쓰려 합니다", llm_call=llm)
    assert state["phase"] == "discussion_complete"
    state = finalize_ideation_conversation(previous_state=state, llm_call=llm)
    assert state["phase"] == "finalized"

    synthesis_prompts = [p for p in llm.captured_prompts if '"idea_name"' in p]
    assert synthesis_prompts
    assert "[planning_expert] 임시 제안 내용입니다" in synthesis_prompts[-1]


def test_expert_delegation_generation_failure_falls_back_to_failed_phase():
    """제안 생성 자체가 구조화 검증에 실패하면(재시도 후에도 무효) 다른 콘텐츠 생성 노드와
    동일하게 phase="failed"로 끝나야 한다(빈 카드를 만들지 않는다는 코드베이스 전체 정책)."""
    llm = ScriptedLLM(broken_for={"expert_delegation"})
    state = _legacy_start_at_awaiting_planning_answer(llm)

    state = reply_ideation_conversation(previous_state=state, user_message="잘 모르겠어", llm_call=llm)

    assert state["phase"] == "failed"
    assert state["failed_node"] == "expert_delegation__planning_expert"


# ---------------------------------------------------------------------------
# 18. 위원 간 실제 회의로 개편(요청 2026-07-21): 상호참조 / 조건부 수정 / 진행자 정리 /
#     종료 조건 / discussion_rounds 보존. 라운드테이블 전환 이후에는 사용자 답변을 기다릴
#     필요 없이 start() 한 번으로 라운드 하나(또는 max_rounds까지 여러 라운드)가 자동으로
#     완료된다.
# ---------------------------------------------------------------------------


class _DebateScriptedLLM:
    """용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): review 단계(dev_expert)의
    stance를 자유롭게 지정할 수 있는 stub — 실제 개발 위원의 검토가 기획 위원의 발언을
    구체적으로 인용하는지, stance에 따라(REVISION_TRIGGER_STANCES) 기획 위원이 다시
    응답하는지 검증하기 위함이다. discussion_stage가 "initial_position"/"response" 두
    값뿐이라 몇 번째 발언인지는 페르소나별 호출 횟수로 추적한다."""

    def __init__(self, dev_stance="보완", dev_next_action="await_user_decision", revision_concern="개발 관점 우려"):
        self.captured_prompts: list[str] = []
        self.dev_stance = dev_stance
        self.dev_next_action = dev_next_action
        self.revision_concern = revision_concern
        self._planning_calls = 0
        self._dev_calls = 0
        # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — 여러 라운드에 걸쳐 이
        # stub이 재사용될 수 있으므로(예: continue_round 시나리오), "이번 기획 발언이
        # 수정(revision) 응답이어야 하는가"는 누적 호출 횟수가 아니라 "직전 개발 위원 발언이
        # 수정을 요구했는가" 플래그로 판단한다.
        self._awaiting_revision = False

    def __call__(self, prompt: str) -> str:
        self.captured_prompts.append(prompt)
        is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt

        if "[판정 규칙]" in prompt:
            return json.dumps({"is_sufficient": True, "reason": "충분", "follow_up_question": None}, ensure_ascii=False)

        if "[질문 규칙]" in prompt:
            speaker = "planning_expert" if is_planning else "dev_expert"
            return json.dumps(
                {
                    "spoken_text": f"[{speaker}] 발화 질문",
                    "judgment": f"[{speaker}] 판단",
                    "question": f"[{speaker}] 질문",
                    "question_topic": _topic_from_prompt(prompt),
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )

        if "[의견 규칙]" in prompt:
            is_dev = "당신은 AI Review Board의 개발 전문가입니다" in prompt
            needs_revision = self.dev_stance in {"반박", "조건부_동의", "대안_제시"}

            if is_dev:
                self._dev_calls += 1
                # review — 방금 나온 기획 전문가의 최초 의견을 구체적으로 검토한다.
                # needs_revision이면 기획 위원이 다시 응답해야 하므로 아직 쟁점을 닫지
                # 않는다(recommended_next_speaker="planning_expert"). 그렇지 않으면 진행자로
                # 넘기되, dev_next_action="continue_round"면 쟁점을 아직 해결하지 않은 채
                # 넘겨 facilitator가 다음 라운드로 자동 진행하게 한다(기존 테스트 의도 보존).
                dev_resolves_issue = not needs_revision and self.dev_next_action != "continue_round"
                self._awaiting_revision = needs_revision
                return json.dumps(
                    {
                        "stance": self.dev_stance,
                        "spoken_text": f"발화: {self.revision_concern}",
                        "judgment": "기획 방향에는 동의하지만 범위가 큽니다",
                        "reason": "업무 자동화 전체를 MVP에 넣으면 남은 라운드 안에 검증하기 어렵습니다",
                        "suggestion": "문서 요약과 학습 추천부터 검증하는 편이 현실적입니다",
                        "interim_conclusion": "문서 요약부터 검증하는 방향으로 잠정 결론짓습니다",
                        "responding_to": self.revision_concern,
                        "agreement": "문제 정의 자체에는 동의합니다",
                        "concern": self.revision_concern,
                        "confirmed": [],
                        "unconfirmed": [],
                        "referenced_message_ids": [],
                        "evidence": [],
                        "next_action": None,
                        "active_issue_id": "mvp_scope",
                        "active_issue_title": "MVP 범위",
                        "new_information": [self.revision_concern],
                        "proposal": "문서 요약과 학습 추천부터 검증",
                        "changed_position": False,
                        "needs_counterpart_response": needs_revision,
                        "recommended_next_speaker": "planning_expert" if needs_revision else "ideation_facilitator",
                        "issue_resolved": dev_resolves_issue,
                        "needs_user_input": False,
                        "user_question": None,
                    },
                    ensure_ascii=False,
                )

            self._planning_calls += 1
            if self._awaiting_revision:
                self._awaiting_revision = False
                # 직전 dev 발언이 수정을 요구했다 — dev의 concern에 구체적으로 응답(수정)한다.
                return json.dumps(
                    {
                        "stance": "조건부_동의",
                        "spoken_text": f"발화: '{self.revision_concern}'을 반영해 범위를 문서 요약 하나로 좁혔습니다",
                        "judgment": "기획 전문가가 개발 전문가의 우려를 반영해 범위를 조정합니다",
                        "reason": "MVP 범위를 좁히면 남은 라운드 안에 검증할 수 있습니다",
                        "suggestion": "핵심 기능 하나만 우선 구현합니다",
                        "interim_conclusion": "MVP 범위를 문서 요약 하나로 좁히는 방향으로 정리합니다",
                        "responding_to": self.revision_concern,
                        "agreement": "MVP 범위를 좁혀야 한다는 지적에 동의합니다",
                        "concern": "",
                        "revision": f"'{self.revision_concern}'을 반영해 범위를 자동화 전체에서 문서 요약 하나로 좁혔습니다",
                        "confirmed": [],
                        "unconfirmed": [],
                        "referenced_message_ids": [],
                        "evidence": [],
                        "next_action": None,
                        "active_issue_id": "mvp_scope",
                        "active_issue_title": "MVP 범위",
                        "new_information": [f"'{self.revision_concern}'을 반영해 범위를 문서 요약 하나로 좁힘"],
                        "proposal": "문서 요약 하나로 범위를 좁힌 MVP",
                        "changed_position": True,
                        "needs_counterpart_response": False,
                        "recommended_next_speaker": "ideation_facilitator",
                        "issue_resolved": True,
                        "needs_user_input": False,
                        "user_question": None,
                    },
                    ensure_ascii=False,
                )
            # initial_position — planning_expert의 최초 의견.
            return json.dumps(
                {
                    "stance": "보완",
                    "spoken_text": "발화: 업무 자동화 전체를 MVP 범위로 제안합니다",
                    "judgment": "두 후보를 결합하면 업무 자동화 과정의 부족한 역량을 학습으로 연결하는 서비스가 됩니다",
                    "reason": "사용자 답변에서 반복 업무 부담이 확인됐습니다",
                    "suggestion": "업무 자동화 전체를 MVP 범위로 제안합니다",
                    "interim_conclusion": "업무 자동화 전체를 MVP 범위로 잠정 제안합니다",
                    "responding_to": None,
                    "agreement": "",
                    "concern": "",
                    "confirmed": [],
                    "unconfirmed": [],
                    "referenced_message_ids": [],
                    "evidence": [],
                    "next_action": None,
                    "active_issue_id": "mvp_scope",
                    "active_issue_title": "MVP 범위",
                    "new_information": ["업무 자동화 전체를 MVP 범위로 제안"],
                    "proposal": "업무 자동화 전체를 MVP 범위로 제안",
                    "changed_position": False,
                    "needs_counterpart_response": True,
                    "recommended_next_speaker": "dev_expert",
                    "issue_resolved": False,
                    "needs_user_input": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )

        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "agreements": ["문서 업무 분석에서 시작한다"],
                    "disagreements": [],
                    "facilitator_summary": "기획 위원과 개발 위원이 문서 업무 분석부터 시작하는 방향에 합의했습니다.",
                    "spoken_text": "두 위원이 문서 업무 분석부터 시작하는 방향에 합의했습니다.",
                    "needs_user_decision": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )

        raise AssertionError(f"예상하지 못한 프롬프트입니다: {prompt[:200]}")


def _run_to_discussion(llm, max_rounds=3, evidence_lookup=None):
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) 이후에는 start() 한 번만으로
    라운드테이블(안건 제시 -> 기획 최초 의견 -> 개발 검토 -> [선택적 수정] -> 진행자 정리,
    필요하면 다음 라운드까지)이 끝까지 실행된다 — 과거에는 이 지점에 도달하려고 두 번의
    reply(질문 답변)가 필요했지만 이제는 필요 없다."""
    return start_ideation_conversation(
        session_id="CONV-TEST",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        max_rounds=max_rounds,
        evidence_lookup=evidence_lookup,
    )


def test_dev_review_prompt_actually_contains_planning_expert_real_statement():
    """개발 위원의 검토 프롬프트(conversation_context)에 기획 위원이 방금 실제로 한 발언이
    포함되는지 확인한다 — 독립적인 일반론이 아니라 실제 상호참조임을 검증한다."""
    llm = _DebateScriptedLLM(dev_stance="보완")
    _run_to_discussion(llm)

    review_prompts = [
        p for p in llm.captured_prompts if "[의견 규칙]" in p and "당신은 AI Review Board의 개발 전문가입니다" in p
    ]
    assert review_prompts
    assert "업무 자동화 전체를 MVP 범위로 제안합니다" in review_prompts[-1]


def test_planning_revision_runs_when_dev_stance_needs_response_and_references_concern():
    """dev의 stance가 REVISION_TRIGGER_STANCES(반박/조건부_동의/대안_제시)에 속하면
    planning_expert_revision이 실행되고, 그 발언이 dev의 구체적 concern을 실제로
    반영한다."""
    llm = _DebateScriptedLLM(dev_stance="반박", revision_concern="MVP 범위가 너무 큽니다")
    state = _run_to_discussion(llm)

    speakers = [m["speaker_id"] for m in state["messages"]]
    # planning(최초) -> dev(검토) -> planning(수정) -> facilitator(정리) 순서.
    assert speakers[-4:] == ["planning_expert", "dev_expert", "planning_expert", "ideation_facilitator"]

    revision_message = state["messages"][-2]
    assert revision_message["speaker_id"] == "planning_expert"
    assert "MVP 범위가 너무 큽니다" in revision_message["content"]
    assert revision_message["structured"]["revision"]

    assert state["discussion_rounds"], "discussion_rounds가 비어 있으면 안 된다"
    record = state["discussion_rounds"][-1]
    # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — revised_proposal은 더 이상
    # 별도로 추적하지 않는다(발언 횟수가 라운드마다 고정이 아니게 됐으므로). planning_position이
    # "기획 위원의 가장 최근 발언"이라 수정 발언 내용이 여기에 담긴다.
    assert record["revised_proposal"] is None
    assert "MVP 범위가 너무 큽니다" in record["planning_position"]


def test_planning_revision_skipped_when_dev_simply_agrees():
    """dev의 stance가 "동의"/"보완"이면(원래 제안을 바꿀 만한 반론이 아니면)
    planning_expert_revision이 실행되지 않아야 한다(요청 6번: 필요할 때만 수정 1회,
    비용 절감)."""
    llm = _DebateScriptedLLM(dev_stance="동의")
    state = _run_to_discussion(llm)

    speakers = [m["speaker_id"] for m in state["messages"]]
    assert speakers[-3:] == ["planning_expert", "dev_expert", "ideation_facilitator"]
    # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — discussion_stage에 "revision"이
    # 라는 별도 값은 더 이상 없다(review/revision을 "response"로 통합). 대신 기획 위원이
    # 두 번째로 불렸는지(=수정 발언을 했는지)로 직접 검증한다.
    assert llm._planning_calls == 1, "동의 stance에서는 기획 위원의 두 번째(수정) 발언 자체가 없어야 한다"


def test_facilitator_message_always_present_and_summarizes_not_repeats():
    """진행자 정리 메시지가 항상 마지막에 생성되고, 두 위원의 문장을 그대로 복사하지 않은
    별도의 요약 텍스트임을 확인한다(요청 5번)."""
    llm = _DebateScriptedLLM(dev_stance="보완")
    state = _run_to_discussion(llm)

    facilitator_messages = [m for m in state["messages"] if m["speaker_id"] == "ideation_facilitator"]
    assert facilitator_messages
    summary = facilitator_messages[-1]
    assert summary["message_type"] == "summary"
    assert "문서 업무 분석" in summary["content"]
    # 기획/개발 위원의 원문 그대로가 아니라 진행자 고유의 요약 문장이어야 한다.
    assert "두 후보를 결합하면 업무 자동화" not in summary["content"]

    # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — content는
    # 이제 spoken_text(1~2문장의 채팅용 정리)이고, discussion_rounds 아카이브는 여전히
    # facilitator_summary(3~5문장 상세 요약, raw.get("facilitator_summary")) 원문을 그대로
    # 저장한다 — 이 둘은 서로 다른 문장이므로 structured에서 원문을 직접 비교한다.
    assert state["discussion_rounds"][-1]["facilitator_summary"] == summary["structured"]["facilitator_summary"]


def test_facilitator_does_not_change_phase_decided_by_dev_review():
    """discussion_facilitator는 dev_expert_discussion(review)이 이미 정한 phase(next_action)를
    바꾸지 않는다 — continue_round면 같은 요청 안에서 곧바로 다음 라운드 discussion으로
    자동 이어지는지(1:1 인터뷰 질문으로 돌아가지 않는지) 확인한다. dev_next_action이 항상
    "continue_round"이므로 max_rounds(3)에 도달한 뒤 남은 쟁점을 한 번 이월하고
    정상 종료해야 한다."""
    llm = _DebateScriptedLLM(dev_stance="보완", dev_next_action="continue_round")
    state = _run_to_discussion(llm, max_rounds=3)

    assert state["phase"] == "discussion_complete"
    assert state["round"] == 4
    assert len(state["discussion_rounds"]) == 4
    assert not any(m["message_type"] == "question" for m in state["messages"])
    facilitator_indices = [i for i, m in enumerate(state["messages"]) if m["speaker_id"] == "ideation_facilitator"]
    # 오프닝 안건 제시 메시지(1) + 라운드별 정리(4) = 5.
    assert len(facilitator_indices) == 5


def test_dev_review_references_planning_message_id():
    """요청 4번 — 상호참조는 코드가 결정한다. dev의 review 메시지 structured에 담긴
    responding_to_message_id가 실제 planning_expert 최초 의견 메시지의 message_id와
    정확히 일치해야 한다(LLM이 만든 값이 아니라 코드가 계산한 값)."""
    llm = _DebateScriptedLLM(dev_stance="보완")
    state = _run_to_discussion(llm)

    planning_message = next(m for m in state["messages"] if m["speaker_id"] == "planning_expert")
    dev_message = next(m for m in state["messages"] if m["speaker_id"] == "dev_expert")

    assert dev_message["structured"]["responding_to_message_id"] == planning_message["message_id"]
    assert dev_message["structured"]["responding_to_speaker_id"] == "planning_expert"
    assert planning_message["message_id"] in dev_message["referenced_message_ids"]


def test_message_ids_unique_after_multi_round_auto_chain():
    """회귀 방지(중복 메시지 진단 1번) — 여러 라운드가 한 번의 그래프 실행 안에서 자동으로
    이어져도(continue_round) state["messages"]의 message_id가 전부 고유해야 한다."""
    llm = _DebateScriptedLLM(dev_stance="보완", dev_next_action="continue_round")
    state = _run_to_discussion(llm, max_rounds=3)

    ids = [m["message_id"] for m in state["messages"]]
    assert len(ids) == len(set(ids)), f"중복된 message_id가 있습니다: {ids}"


def test_no_actionable_question_completes_discussion_without_pending_question():
    """요청 5~6번 — pending_question은 진행자가 needs_user_decision=True로 실제 질문을
    던졌을 때만 설정된다. 여기서는 dev_next_action="await_user_decision"이지만
    facilitator 응답이 needs_user_decision=False이므로(고정 stub) pending_question이
    None이어야 한다."""
    llm = _DebateScriptedLLM(dev_stance="보완", dev_next_action="await_user_decision")
    state = _run_to_discussion(llm, max_rounds=3)

    assert state["phase"] == "discussion_complete"
    assert state["pending_question"] is None
    assert state["pending_question_topic"] is None


class _NeedsDecisionScriptedLLM(_DebateScriptedLLM):
    """진행자가 실제로 사용자에게 질문하는 경우(needs_user_decision=True)를 재현한다."""

    def __call__(self, prompt: str) -> str:
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "agreements": [],
                    "disagreements": ["MVP 범위를 어디까지 좁힐지"],
                    "facilitator_summary": "MVP 범위에 대해 두 위원의 의견이 갈립니다.",
                    "spoken_text": "MVP를 문서 요약만으로 시작할까요, 업무 자동화까지 포함할까요?",
                    "needs_user_decision": True,
                    "user_question": "MVP를 문서 요약만으로 시작할까요, 업무 자동화까지 포함할까요?",
                },
                ensure_ascii=False,
            )
        return super().__call__(prompt)


def test_facilitator_sets_pending_question_when_needs_user_decision():
    llm = _NeedsDecisionScriptedLLM(dev_stance="보완", dev_next_action="await_user_decision")
    state = _run_to_discussion(llm, max_rounds=3)

    assert state["phase"] == "awaiting_user_decision"
    assert state["pending_question"] == "MVP를 문서 요약만으로 시작할까요, 업무 자동화까지 포함할까요?"
    assert state["pending_question_topic"] == "facilitator_decision"
    # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — 예전에는
    # [사용자 의견이 필요한 사항] 헤더로 user_question이 content에 붙었지만, 이제
    # needs_user_decision=true일 때 spoken_text 자체가 질문을 자연스럽게 담는다.
    assert "MVP를 문서 요약만으로 시작할까요" in state["messages"][-1]["content"]
    assert state["messages"][-1]["structured"]["user_question"] == state["pending_question"]


def test_user_answer_to_facilitator_question_is_answer_type():
    """진행자가 실제로 물었을 때 사용자가 답하면 message_type이 "answer"여야 한다."""
    llm = _NeedsDecisionScriptedLLM(dev_stance="보완", dev_next_action="await_user_decision")
    state = _run_to_discussion(llm, max_rounds=3)

    state = reply_ideation_conversation(
        previous_state=state, user_message="문서 요약만으로 시작할게요.", llm_call=llm
    )
    user_messages = [m for m in state["messages"] if m["speaker_id"] == "user"]
    assert user_messages[-1]["message_type"] == "answer"


def test_user_interjection_after_discussion_complete_is_recorded_and_referenced():
    """요청 6번 — 진행자가 needs_user_decision=False였는데 사용자가 자유롭게 한 마디
    남기면 message_type="interjection"으로 기록되고, 다음 라운드에서 기획/개발 위원 중
    하나 이상이 그 메시지를 referenced_message_ids에 포함해야 한다."""
    llm = _DebateScriptedLLM(dev_stance="보완", dev_next_action="await_user_decision")
    state = _run_to_discussion(llm, max_rounds=3)
    assert state["phase"] == "discussion_complete"
    assert state["pending_question"] is None

    interjection_text = "학습 범위가 너무 좁은 것 같아. 대학생도 포함하고 싶어."
    state = reply_ideation_conversation(previous_state=state, user_message=interjection_text, llm_call=llm)

    user_messages = [m for m in state["messages"] if m["speaker_id"] == "user"]
    interjection_message = next(m for m in user_messages if m["content"] == interjection_text)
    assert interjection_message["message_type"] == "interjection"

    # continue_round 라운드에서 두 위원이 이 메시지를 상호참조 대상으로 볼 수 있어야 한다
    # (initial_position은 messages 마지막 메시지를 항상 responding_to로 삼는다 — 여기서는
    # 방금 넣은 사용자 개입 메시지).
    later_messages = state["messages"][state["messages"].index(interjection_message) + 1 :]
    expert_messages = [m for m in later_messages if m["speaker_id"] in ("planning_expert", "dev_expert")]
    assert expert_messages
    assert any(interjection_message["message_id"] in m["referenced_message_ids"] for m in expert_messages)


def test_no_direct_questions_to_user_from_experts_across_rounds():
    """요청 4번 — 기획/개발 위원은 기본 라운드테이블 경로에서 절대 message_type="question"을
    만들지 않는다(사용자에게 직접 묻지 않는다). 여러 라운드가 자동으로 이어져도 마찬가지다."""
    llm = _DebateScriptedLLM(dev_stance="보완", dev_next_action="continue_round")
    state = _run_to_discussion(llm, max_rounds=3)

    expert_messages = [m for m in state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
    assert expert_messages
    assert not any(m["message_type"] == "question" for m in expert_messages)


# ---------------------------------------------------------------------------
# 19. RAG 근거 공유 검증(요청 2026-07-21 후속): 기획/개발이 같은 공모전 분석을 역할에 맞게
#     따로 검색하는지, 상대 발언에 실린 근거를 서로 볼 수 있는지, 근거가 없을 때 지어내지
#     않는지. 라운드테이블 전환 이후에는 start() 한 번으로 두 위원이 모두 실행된다.
# ---------------------------------------------------------------------------


def _role_tagged_evidence_lookup(calls: list[tuple[str, str]]):
    """persona_id별로 서로 다른 quote를 반환하는 가짜 evidence_lookup — 호출 인자(persona_id,
    query)를 calls 리스트에 기록해, 각 페르소나가 실제로 "자신의 role로" 별도 검색을
    호출했는지 검증할 수 있게 한다."""

    def lookup(persona_id: str, query: str):
        calls.append((persona_id, query))
        return [
            {
                "document_id": f"DOC-{persona_id}",
                "document_name": f"공고문({persona_id} 검색)",
                "chunk_id": "C1",
                "page": 1,
                "section": None,
                "quote": f"[{persona_id}용 근거 원문] 평가 기준 발췌",
                "relevance": None,
            }
        ]

    return lookup


class _RagAwareScriptedLLM(_DebateScriptedLLM):
    """_DebateScriptedLLM을 상속해 discussion 노드 응답에 retrieved_evidence를 그대로
    반영한 evidence를 담아 반환한다 — conversation_context에 실려 다음 페르소나 프롬프트로
    전달되는지 확인하기 위함이다."""

    def __call__(self, prompt: str) -> str:
        raw_text = super().__call__(prompt)
        if "[의견 규칙]" in prompt and "[discussion_stage]\ninitial_position" in prompt:
            # planning_expert의 최초 의견에 실제로 검색된 근거(quote)를 그대로 인용해 반환한다.
            payload = json.loads(raw_text)
            payload["evidence"] = [
                {
                    "document_id": "DOC-planning_expert",
                    "document_name": "공고문(planning_expert 검색)",
                    "chunk_id": "C1",
                    "page": 1,
                    "section": None,
                    "quote": "[planning_expert용 근거 원문] 평가 기준 발췌",
                    "relevance": "문제 정의 근거",
                }
            ]
            return json.dumps(payload, ensure_ascii=False)
        return raw_text


def test_dev_expert_receives_own_role_specific_rag_retrieval():
    """기획/개발이 각자 자신의 persona_id로 evidence_lookup을 별도 호출하는지(공유된 같은
    공모전 분석을 역할에 맞게 각자 검색) 확인한다."""
    calls: list[tuple[str, str]] = []
    lookup = _role_tagged_evidence_lookup(calls)
    llm = _DebateScriptedLLM(dev_stance="보완")

    start_ideation_conversation(
        session_id="CONV-RAG-1",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        evidence_lookup=lookup,
    )

    personas_called = {persona_id for persona_id, _query in calls}
    assert "planning_expert" in personas_called
    assert "dev_expert" in personas_called, "개발 위원도 자신의 role로 RAG 검색을 별도로 호출해야 한다"


def test_discussion_message_evidence_reflects_rag_retrieval_not_llm_self_report():
    """용준/Claude(2026-07-22, RAG 근거 유실 수정 회귀 테스트): _DebateScriptedLLM은 항상
    "evidence": []를 반환한다(LLM이 evidence 필드를 스스로 채우지 않는 실제 상황과 동일) —
    그런데도 실제로 RoleAwareRetrievalService가 찾아 프롬프트에 주입한 근거(retrieved)가
    있었다면, 저장된 ConvMessage.evidence와 로그의 injected_evidence_count는 그 근거를
    반영해야 한다. 예전 코드는 raw.get("evidence")만 저장해 여기서 항상 0건으로 유실됐다."""
    calls: list[tuple[str, str]] = []
    lookup = _role_tagged_evidence_lookup(calls)
    llm = _DebateScriptedLLM(dev_stance="보완")

    state = start_ideation_conversation(
        session_id="CONV-RAG-EVIDENCE-LOSS",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        evidence_lookup=lookup,
    )

    planning_message = next(m for m in state["messages"] if m["speaker_id"] == "planning_expert" and m["message_type"] != "question")
    assert planning_message["evidence"], "RAG 검색 결과가 있었으므로 메시지 evidence가 비어있으면 안 된다"
    assert planning_message["evidence"][0]["chunk_id"] == "C1"
    assert planning_message["evidence"][0]["document_id"] == "DOC-planning_expert"


def test_dev_review_prompt_can_see_planning_evidence_via_conversation_context():
    """기획 위원이 실제로 인용한 RAG 근거(quote)가, 개발 위원의 검토 프롬프트
    conversation_context(상대 발언 전체)에 그대로 실려 전달되는지 확인한다 — 개발 위원이
    "상대 발언"과 "자신의 RAG 근거"를 구분할 수 있으려면 상대 발언에 실린 근거가 먼저
    프롬프트에 도달해야 한다."""
    calls: list[tuple[str, str]] = []
    lookup = _role_tagged_evidence_lookup(calls)
    llm = _RagAwareScriptedLLM(dev_stance="보완")

    start_ideation_conversation(
        session_id="CONV-RAG-2",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        evidence_lookup=lookup,
    )

    dev_review_prompts = [
        p for p in llm.captured_prompts if "[의견 규칙]" in p and "당신은 AI Review Board의 개발 전문가입니다" in p
        and "[discussion_stage]\nresponse" in p
    ]
    assert dev_review_prompts
    dev_prompt = dev_review_prompts[-1]
    # 상대(planning_expert)가 인용한 근거 원문이 개발 위원 프롬프트의 대화 맥락 섹션에
    # 실제로 도달했다.
    assert "[planning_expert용 근거 원문]" in dev_prompt
    # 자신의(dev_expert) RAG 검색 결과는 별도의 [검색 근거 retrieved_evidence] 섹션에 있다
    # — 두 출처가 서로 다른 섹션으로 구분돼 있어야 "상대 발언"과 "자신의 근거"를 혼동하지
    # 않는다.
    # "[대화 맥락 conversation_context]"라는 문구는 [경계]/[입력] 설명 섹션에도 나오므로,
    # 실제 데이터가 주입되는 지점(구분선 아래)부터만 잘라낸다.
    injected = dev_prompt.split("이하 실행 시 주입되는 컨텍스트")[1]
    context_section = injected.split("[대화 맥락 conversation_context]")[1].split("[speaks_second]")[0]
    evidence_section = injected.split("[검색 근거 retrieved_evidence]")[1].split("[대화 맥락")[0]
    assert "[planning_expert용 근거 원문]" in context_section
    assert "[dev_expert용 근거 원문]" in evidence_section
    assert "[dev_expert용 근거 원문]" not in context_section  # 자신의 새 검색 결과가 상대 발언 섹션에 섞이지 않는다.


def test_facilitator_and_discussion_prompts_instruct_against_fabricating_evidence():
    """진행자·전문가 프롬프트가 "근거 없는 사실을 지어내지 않는다"를 명시적으로 지시하는지
    확인한다(모델이 실제로 그 지시를 따르는지는 실제 LLM 평가 영역이라 이 테스트의 범위
    밖이다 — 여기서는 그 지시 자체가 프롬프트에 존재하는지만 구조적으로 검증한다)."""
    from prompts import build_ideation_conv_discussion_facilitator_prompt

    facilitator_prompt = build_ideation_conv_discussion_facilitator_prompt(
        NOTICE_AND_CRITERIA, {"judgment": "j"}, {"judgment": "j2"}, None, [], [], "await_user_decision", 1, 3
    )
    assert "근거 없는 사실" in facilitator_prompt and "만들어내지" in facilitator_prompt

    discussion_prompt_path = MEETING_DIR / "prompts" / "ideation_conv_discussion.txt"
    assert "근거 없는 사실" in discussion_prompt_path.read_text(encoding="utf-8")


def test_facilitator_prompt_instructs_not_to_summarize_expert_judgment_only_as_agreement():
    """용준/Claude(2026-07-22, 요청: linked_evidence_count=0인 의견을 문서로 확인된 합의처럼
    요약하지 않도록 수정) — evidence_status="expert_judgment_only"인 발언을 진행자가
    "합의했습니다"로만 요약하지 않고, 문서 근거가 없다는 사실을 함께 밝히도록 지시하는지
    구조적으로 확인한다."""
    facilitator_prompt_path = MEETING_DIR / "prompts" / "ideation_conv_discussion_facilitator.txt"
    text = facilitator_prompt_path.read_text(encoding="utf-8")
    assert "expert_judgment_only" in text
    assert "합의했습니다" in text and "반복하지 않습니다" in text


def test_no_evidence_available_produces_empty_evidence_section_without_crashing():
    """evidence_lookup이 없을 때(RAG 미사용) retrieved_evidence 섹션이 빈 배열로 표시되고,
    질문/의견 생성 자체는 정상적으로 완료되는지 확인한다(요청: "RAG가 없을 때 일반적인
    전문가 판단임을 구분") — 프롬프트에도 "근거가 없으면 지어내지 않는다"는 지시가
    포함돼야 한다."""
    llm = _DebateScriptedLLM(dev_stance="보완")
    state = start_ideation_conversation(
        session_id="CONV-RAG-3",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        evidence_lookup=None,  # RAG 미사용.
    )
    assert state["phase"] == "discussion_complete"
    planning_prompt = next(
        p for p in llm.captured_prompts if "[의견 규칙]" in p and "당신은 AI Review Board의 기획 전문가입니다" in p
    )
    assert "[검색 근거 retrieved_evidence]\n[]" in planning_prompt
    assert "지어내지" in planning_prompt


# ---------------------------------------------------------------------------
# 20. expert_delegation 위원 간 상호 검토 확장(요청 2026-07-21 후속) — 반박이면 수정 턴이
#     실행되고, 단순 동의면 생략되며, 어느 경우든 진행자 최종 권고안으로 끝나는지. 위임
#     흐름은 여전히 레거시 인터뷰 phase(PHASE_TO_PENDING_PERSONA)에서만 동작하므로, 레거시
#     경로로 시작한다.
# ---------------------------------------------------------------------------


def _delegation_scripted_llm(review_stance: str):
    """expert_delegation 전용 스텁 — 개발 질문에서 위임하면 planning_expert가 검토자다.
    review_stance로 검토 stance를 통제해 REVISION_TRIGGER_STANCES 게이팅을 검증한다."""

    def llm_call(prompt: str) -> str:
        is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
        is_dev = "당신은 AI Review Board의 개발 전문가입니다" in prompt
        speaker = "planning_expert" if is_planning else "dev_expert"

        if "[질문 규칙]" in prompt:
            return json.dumps(
                {
                    "spoken_text": f"[{speaker}] 발화 질문",
                    "judgment": f"[{speaker}] 판단",
                    "question": f"[{speaker}] 질문",
                    "question_topic": _topic_from_prompt(prompt),
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )
        if "[판정 규칙]" in prompt:
            return json.dumps(
                {"answer_type": "answer", "reason": "충분", "follow_up_question": None, "clarification_response": None},
                ensure_ascii=False,
            )
        if "[제안 규칙]" in prompt:
            is_revision_stage = "[stage]\nrevision" in prompt
            return json.dumps(
                {
                    "spoken_text": f"[{speaker}] 발화 제안",
                    "proposal": f"[{speaker}] 제안",
                    "reason": f"[{speaker}] 이유",
                    "assumption": f"[{speaker}] 가정",
                    "responding_to": "구현 범위가 너무 넓습니다" if is_revision_stage else None,
                    "revision": f"[{speaker}] '구현 범위가 너무 넓습니다'를 반영해 범위를 조정" if is_revision_stage else None,
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )
        if "[위임 검토 규칙]" in prompt:
            return json.dumps(
                {
                    "stance": review_stance,
                    "spoken_text": f"[{speaker}] 발화 검토",
                    "judgment": f"[{speaker}] 검토 판단",
                    "reason": f"[{speaker}] 검토 근거",
                    "responding_to": "상대의 임시 제안 내용",
                    "agreement": "제안 방향에는 동의" if review_stance != "반박" else "",
                    "concern": "구현 범위가 너무 넓습니다" if review_stance == "반박" else "",
                    "recommendation": f"[{speaker}] 검토 결론",
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )
        if "[위임 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "agreements": [],
                    "considerations": [],
                    "final_recommendation": "위임 최종 권고안입니다.",
                    "spoken_text": "위임 최종 권고안입니다.",
                },
                ensure_ascii=False,
            )
        if "[의견 규칙]" in prompt:
            # 위임이 developer_question에서 일어나면 apply_user_answer가 곧바로
            # expert_discussion으로 넘어가 같은 요청 안에서 실제 회의(기획/개발 보완 의견)까지
            # 이어진다 — 이 테스트들은 위임 흐름 자체만 검증하므로 discussion 응답은 항상
            # "보완"으로 단순하게 고정한다.
            is_response_stage = "[discussion_stage]\nresponse" in prompt
            return json.dumps(
                {
                    "stance": "보완",
                    "spoken_text": f"[{speaker}] 발화 판단",
                    "judgment": f"[{speaker}] 판단",
                    "reason": f"[{speaker}] 근거",
                    "suggestion": f"[{speaker}] 제안",
                    "interim_conclusion": f"[{speaker}] 임시 결론",
                    "responding_to": "상대 발언" if is_response_stage else None,
                    "agreement": "동의 지점" if is_response_stage else "",
                    "concern": "",
                    "confirmed": [],
                    "unconfirmed": [],
                    "referenced_message_ids": [],
                    "evidence": [],
                    "next_action": None,
                    "active_issue_id": "delegation_followup",
                    "active_issue_title": "위임 이후 후속 논의",
                    "new_information": [f"[{speaker}] 새로운 판단"],
                    "proposal": f"[{speaker}] 제안",
                    "changed_position": False,
                    "needs_counterpart_response": not is_response_stage,
                    "recommended_next_speaker": "ideation_facilitator" if is_response_stage else "dev_expert",
                    "issue_resolved": is_response_stage,
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
                    "facilitator_summary": "라운드 정리",
                    "spoken_text": "라운드 정리",
                    "needs_user_decision": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")

    return llm_call


def test_delegation_revision_runs_when_counterpart_review_disagrees():
    """개발 질문에서 위임 -> 기획 위원(반대 역할)의 검토가 "반박"이면
    planning_expert_revision과 동등한 수정 턴(같은 expert_delegation 스키마, stage="revision")
    이 실행되고, 그 발언이 검토의 구체적 우려를 반영하는지 확인한다."""
    llm = _delegation_scripted_llm(review_stance="반박")
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="답변1", llm_call=llm)
    assert state["phase"] == "awaiting_developer_answer"

    state = reply_ideation_conversation(previous_state=state, user_message="잘 모르겠어요", llm_call=llm)

    # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — discussion 노드의 structured도
    # 이제 "proposal" 키를 갖게 되어 위임(expert_delegation) 메시지와 구분이 안 되므로,
    # 위임 메시지에만 있는 "assumption" 키로 좁힌다.
    dev_messages = [m for m in state["messages"] if m["speaker_id"] == "dev_expert" and "assumption" in (m["structured"] or {})]
    assert len(dev_messages) == 2, "제안(1회) + 검토를 반영한 수정(1회) = 2건이어야 한다"
    revision_message = dev_messages[-1]
    assert "구현 범위가 너무 넓습니다" in revision_message["structured"]["revision"]
    facilitator_message = next(m for m in state["messages"] if m["speaker_id"] == "ideation_facilitator")
    assert facilitator_message["content"].startswith("위임 최종 권고안입니다.")


def test_delegation_revision_skipped_when_counterpart_review_simply_agrees():
    """검토가 "동의"면(REVISION_TRIGGER_STANCES 밖) 수정 턴 없이 곧바로 진행자 권고안으로
    끝나야 한다(비용 절감)."""
    llm = _delegation_scripted_llm(review_stance="동의")
    state = _legacy_start_at_awaiting_planning_answer(llm)
    state = reply_ideation_conversation(previous_state=state, user_message="답변1", llm_call=llm)
    state = reply_ideation_conversation(previous_state=state, user_message="잘 모르겠어요", llm_call=llm)

    # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — discussion 노드의 structured도
    # 이제 "proposal" 키를 갖게 되어 위임(expert_delegation) 메시지와 구분이 안 되므로,
    # 위임 메시지에만 있는 "assumption" 키로 좁힌다.
    dev_messages = [m for m in state["messages"] if m["speaker_id"] == "dev_expert" and "assumption" in (m["structured"] or {})]
    assert len(dev_messages) == 1, "동의면 제안(1회)만 있고 수정 턴이 추가되면 안 된다"
    facilitator_message = next(m for m in state["messages"] if m["speaker_id"] == "ideation_facilitator")
    assert facilitator_message["content"].startswith("위임 최종 권고안입니다.")


def test_delegation_never_re_asks_the_same_question_to_user():
    """요청: "다시 사용자에게 같은 질문을 넘기면 안 됩니다" — 위임 흐름이 끝난 뒤에도
    다음 단계로 실제로 진행되고(질문이 반복되지 않고), 진행자 메시지 어디에도 사용자에게
    되묻는 물음표 형태의 원래 질문 반복이 남지 않는지 확인한다."""
    llm = _delegation_scripted_llm(review_stance="조건부_동의")
    state = _legacy_start_at_awaiting_planning_answer(llm)
    original_question = state["pending_question"]
    state = reply_ideation_conversation(previous_state=state, user_message="잘 모르겠어요", llm_call=llm)

    assert state["phase"] == "awaiting_developer_answer"  # 같은 기획 질문에 머물지 않고 진행됐다.
    assert state["pending_question"] != original_question
    facilitator_message = next(m for m in state["messages"] if m["speaker_id"] == "ideation_facilitator")
    # 진행자 최종 권고안 스키마 자체에 재질문 필드가 없으므로, 원래 질문 문자열이 그대로
    # 반복될 수 없다.
    assert original_question not in facilitator_message["content"]


# ---------------------------------------------------------------------------
# 21. 보고서형 메시지 → 자연스러운 회의 발화 전환(요청 2026-07-22) — 회의 전체(라운드테이블
#     한 라운드 + 사용자 개입으로 다음 라운드까지)에서 어떤 메시지의 content에도 예전
#     보고서형 대괄호 헤더가 남아있지 않은지 한 번에 스윕 검증한다.
# ---------------------------------------------------------------------------

_LEGACY_REPORT_HEADERS = (
    "[기획 관점]", "[기술 검토]", "[상대 의견 검토]", "[동의]", "[동의하는 부분]",
    "[우려·제약]", "[우려/제약]", "[구현 대안]", "[임시 결론]", "[확정 사항]", "[미확정 사항]",
    "[현재 판단]", "[핵심 질문]", "[사용자 선택 반영]", "[합의 사항]", "[남은 쟁점]",
    "[사용자 의견이 필요한 사항]", "[재질문]", "[설명]", "[제안 이유]", "[임시 가정]",
    "[검토]", "[검토 결론]", "[제안 검토]", "[상대 검토 반영]", "[수정 내용]", "[참고 사항]",
)


def test_no_legacy_report_headers_leak_into_any_message_content_across_round_and_interjection():
    llm = _DebateScriptedLLM(dev_stance="반박", revision_concern="MVP 범위가 너무 큽니다")
    state = _run_to_discussion(llm, max_rounds=3)
    state = reply_ideation_conversation(
        previous_state=state, user_message="대학생도 포함하고 싶어요.", llm_call=llm
    )

    for message in state["messages"]:
        for header in _LEGACY_REPORT_HEADERS:
            assert header not in message["content"], (
                f"{message['speaker_id']}의 content에 보고서형 헤더 {header!r}가 남아있습니다: "
                f"{message['content']!r}"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
