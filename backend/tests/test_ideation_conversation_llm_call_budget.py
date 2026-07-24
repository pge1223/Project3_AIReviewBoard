# 작성자: 용준/Claude(2026-07-21)
# 목적: 대화형 아이디어 회의(ideation-conversation)에서 한 HTTP 요청이 실제로 몇 번의
#       LLM 호출을 만드는지(요청: "답변 충분성 판정이나 다른 내부 호출을 포함해 6회인지
#       실제 호출 경로를 다시 계산해 주세요")와, 상한(_MAX_LLM_CALLS_PER_REQUEST)이 실제로
#       무한 반복을 막는지를 검증한다.
#
# 다른 discovery/streaming 테스트와 달리 _build_llm_call() 자체를 monkeypatch로 대체하지
# 않는다 — 대신 app.api.routes.ideation_conversation_preview.OpenAI(client 생성자)를 가짜로
# 바꿔치기해서, _build_llm_call() 안의 "실제 상한 카운팅 코드"(call_count["n"] += 1 /
# > _MAX_LLM_CALLS_PER_REQUEST 검사)가 그대로 실행되게 한다. 이렇게 해야 "상한 로직 자체가
# 동작하는지"를 검증할 수 있다(다른 테스트들처럼 _build_llm_call을 통째로 대체하면 이
# 카운팅 코드 자체가 테스트에서 빠진다).
# import: fastapi.testclient, pytest; app.api.routes.ideation_conversation_preview 모듈.

import json
import re
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config import settings  # noqa: E402
import app.api.routes.ideation_conversation_preview as conv_route  # noqa: E402

_REMAINING_TOPICS_RE = re.compile(
    r"\[아직 확인되지 않은 주제\(우선순위 순\) remaining_topics\]\n(.*?)\n\n", re.S
)


def _topic_from_prompt(prompt: str) -> str:
    match = _REMAINING_TOPICS_RE.search(prompt)
    if not match:
        return "problem"
    try:
        remaining = json.loads(match.group(1))
    except (ValueError, TypeError):
        return "problem"
    return remaining[0] if remaining else "problem"


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def model_dump(self):
        return {}


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self, responder):
        self._responder = responder

    def create(self, *, model, messages, response_format=None, stream=False):
        prompt = messages[0]["content"]
        return _FakeCompletion(self._responder(prompt))


class _FakeChat:
    def __init__(self, responder):
        self.completions = _FakeCompletions(responder)


def _make_fake_openai(responder, counter: list[int] | None = None):
    """counter가 주어지면 매 실제 client.chat.completions.create() 호출마다 counter[0]을
    늘린다 — _serialize_state가 llm_calls_used를 API 응답에 노출하지 않으므로, "이번 HTTP
    요청에서 실제로 몇 번 호출됐는지"를 이 카운터로 직접 측정한다(요청 3번 — 실제 호출
    경로 재계산)."""

    def _counting_responder(prompt: str) -> str:
        if counter is not None:
            counter[0] += 1
        return responder(prompt)

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = _FakeChat(_counting_responder)

    return _FakeOpenAI


def _comprehensive_responder(*, dev_review_stance: str, discussion_review_stance: str, discussion_next_action: str):
    """실제 _build_llm_call이 호출하는 모든 마커(질문/판정/제안/위임 검토/위임 정리/의견/
    진행자 정리)에 유효한 최소 응답을 준다 — 요청받은 최대 호출 경로(위임이 반박으로 검토돼
    수정까지 이어지고, 곧바로 회의 discussion도 반박+continue_round로 이어지는 경우)를 정확히
    재현하기 위해 stance/next_action을 파라미터로 받는다."""

    # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — "이번 기획 발언이 dev의 수정
    # 요구에 응답하는 것인가"는 더 이상 고정된 라운드 순번이 아니라 "직전 dev 발언이 수정을
    # 요구했는가" 상태로 판단해야 한다(_DebateScriptedLLM과 같은 패턴). 클로저 밖에서
    # nonlocal로 갱신할 수 있게 리스트에 담는다.
    _awaiting_revision = [False]

    def llm_call(prompt: str) -> str:
        is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
        is_dev = "당신은 AI Review Board의 개발 전문가입니다" in prompt
        speaker = "planning_expert" if is_planning else "dev_expert"

        if "[판정 규칙]" in prompt:
            return json.dumps(
                {"answer_type": "answer", "reason": "충분", "follow_up_question": None, "clarification_response": None},
                ensure_ascii=False,
            )
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
        if "[제안 규칙]" in prompt:
            is_revision_stage = "[stage]\nrevision" in prompt
            return json.dumps(
                {
                    "spoken_text": f"[{speaker}] 발화 제안",
                    "proposal": f"[{speaker}] 제안",
                    "reason": f"[{speaker}] 이유",
                    "assumption": f"[{speaker}] 가정",
                    "responding_to": "상대 검토" if is_revision_stage else None,
                    "revision": f"[{speaker}] 수정" if is_revision_stage else None,
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )
        if "[위임 검토 규칙]" in prompt:
            return json.dumps(
                {
                    "stance": dev_review_stance,
                    "spoken_text": f"[{speaker}] 발화 검토",
                    "judgment": f"[{speaker}] 검토 판단",
                    "reason": f"[{speaker}] 검토 근거",
                    "responding_to": "상대 제안",
                    "agreement": "",
                    "concern": "우려 사항" if dev_review_stance != "동의" else "",
                    "recommendation": f"[{speaker}] 결론",
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
                    "final_recommendation": "위임 최종 권고안",
                    "spoken_text": "위임 최종 권고안",
                },
                ensure_ascii=False,
            )
        if "[의견 규칙]" in prompt:
            needs_revision = discussion_review_stance in {"반박", "조건부_동의", "대안_제시"}
            if is_dev:
                stance = discussion_review_stance
                is_response_stage = True  # dev는 항상 방금 나온 planning 발언에 반응한다.
                dev_resolves_issue = not needs_revision and discussion_next_action != "continue_round"
                _awaiting_revision[0] = needs_revision
                recommended_next_speaker = "planning_expert" if needs_revision else "ideation_facilitator"
                issue_resolved = dev_resolves_issue
            else:
                stance = "보완"
                is_response_stage = _awaiting_revision[0]
                if is_response_stage:
                    _awaiting_revision[0] = False
                recommended_next_speaker = "dev_expert" if not is_response_stage else "ideation_facilitator"
                issue_resolved = is_response_stage  # 수정 응답이면 이번 쟁점은 여기서 끝난다.
            return json.dumps(
                {
                    "stance": stance,
                    "spoken_text": f"[{speaker}] 발화 판단",
                    "judgment": f"[{speaker}] 판단",
                    "reason": f"[{speaker}] 근거",
                    "suggestion": f"[{speaker}] 제안",
                    "interim_conclusion": f"[{speaker}] 현재 임시 결론",
                    "responding_to": "상대 발언" if is_response_stage else None,
                    "agreement": "",
                    "concern": "우려" if is_response_stage else "",
                    "revision": "수정 내용" if (not is_dev and is_response_stage) else None,
                    "confirmed": [],
                    "unconfirmed": [],
                    "referenced_message_ids": [],
                    "evidence": [],
                    "next_action": None,
                    "active_issue_id": "mvp_scope",
                    "active_issue_title": "MVP 범위",
                    "new_information": [f"[{speaker}] 새로 확인된 내용"],
                    "proposal": f"[{speaker}] 제안",
                    "changed_position": False,
                    "needs_counterpart_response": needs_revision if is_dev else False,
                    "recommended_next_speaker": recommended_next_speaker,
                    "issue_resolved": issue_resolved,
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
        if "[캔버스 갱신 규칙]" in prompt:
            return json.dumps(
                {
                    "problem": "문의 응대 부담",
                    "target_user": "소상공인",
                    "core_value": "응대 시간 절감",
                    "solution": "FAQ 자동 응답",
                    "differentiation": "저비용 구축",
                    "feasibility": "medium",
                    "risks": ["오답 위험"],
                    "contest_fit": "실현가능성 기준 대응",
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")

    return llm_call


@pytest.fixture(autouse=True)
def _enable_preview(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_IDEATION_PREVIEW", True)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(conv_route.router)
    return TestClient(app)


def _force_session_to_awaiting_developer_answer(session_id: str, llm_call) -> None:
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) 보존 검증용 헬퍼 — 레거시
    인터뷰 경로(planning_question -> 정상 답변 -> developer_question)를 손으로 이어붙여
    phase="awaiting_developer_answer"에 도달한 state를 세션 스토어에 직접 주입한다. 새
    세션은 더 이상 이 phase에 자연스럽게 도달하지 않지만(라운드테이블이 곧바로 실행되므로),
    expert_delegation 위임 흐름은 여전히 이 phase(PHASE_TO_PENDING_PERSONA)에서만 동작하므로
    LLM 호출 예산 검증을 위해 그 경로를 재현한다. 이 헬퍼가 부르는 llm_call은
    call_counter로 세지 않는다(노드를 직접 호출하므로 conv_route.OpenAI를 거치지 않는다) —
    검증 대상은 오직 그 다음 실제 /reply HTTP 요청 한 번의 호출 수다."""
    from graph.ideation_conv_nodes import make_conv_question_node
    from graph.ideation_conv_run import _new_user_message
    from graph.ideation_conv_state import apply_user_answer, initial_conv_state

    state = dict(
        initial_conv_state(
            session_id,
            {"competition_name": "데모 공모전"},
            {"description": "소상공인이 손님 문의에 자동으로 답하는 챗봇"},
        )
    )
    state["phase"] = "planning_question"
    state["messages"] = []
    q_node = make_conv_question_node("planning_expert", "awaiting_planning_answer", llm_call)
    update = q_node(state)
    state = {**state, **update, "messages": state["messages"] + update.get("messages", [])}

    answer_message = _new_user_message("타깃은 동네 카페 사장님입니다", state["round"])
    state = apply_user_answer(state, answer_message)  # phase -> "developer_question"
    dev_q_node = make_conv_question_node("dev_expert", "awaiting_developer_answer", llm_call)
    dev_update = dev_q_node(state)
    state = {**state, **dev_update, "messages": state["messages"] + dev_update.get("messages", [])}

    conv_route._store.update(session_id, state)


def test_realistic_max_cascade_stays_comfortably_under_the_cap(client: TestClient, monkeypatch):
    """요청 3번 — "답변 충분성 판정이나 다른 내부 호출을 포함해" 실제 호출 경로를 재계산한다.

    용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) 이후 expert_delegation 위임
    흐름은 여전히 phase="awaiting_developer_answer"(레거시 인터뷰 진입점, 새 세션의 기본
    흐름에서는 더 이상 도달하지 않지만 코드는 보존됨)에서만 동작한다
    (_force_session_to_awaiting_developer_answer로 재현). 경로: "잘 모르겠어"(결정적 규칙,
    sufficiency 생략) -> expert_delegation(제안 1 + 반대 위원 검토 1 + [반박이므로] 수정
    1 + 진행자 권고안 1 = 4) -> 같은 요청 안에서 곧바로 expert_discussion 라운드까지
    이어짐(기획 최초 1 + 개발 검토 1 + [반박이므로] 기획 수정 1 + 진행자 정리 1 = 4,
    next_action="await_user_decision"이라 다음 라운드로는 이어지지 않는다) + 아이디어 캔버스
    갱신 1회 = 이번 reply 요청에서만 9회.

    이 값이 실제 호출 수와 정확히 일치하고, 상한(_MAX_LLM_CALLS_PER_REQUEST)보다 여유 있게
    낮은지 확인한다 — 요청 사항 그대로, 현실적인(재시도 없는) 최대 경로에서 상한이 조기에
    트립되지 않아야 한다. next_action="continue_round"까지 쓰지 않는 이유: 라운드테이블
    전환 이후 continue_round는 (구버전의 "다음 질문 1회"가 아니라) "다음 라운드 전체
    (3~4회)"를 같은 요청 안에서 반복하므로, 그 경로는 아래
    test_cap_trips_gracefully_instead_of_looping_forever가 "폭주 방지" 관점에서 별도로
    검증한다."""
    setup_responder = _comprehensive_responder(
        dev_review_stance="반박", discussion_review_stance="반박", discussion_next_action="await_user_decision"
    )
    call_counter = [0]
    monkeypatch.setattr(conv_route, "OpenAI", _make_fake_openai(setup_responder, call_counter))

    start_resp = client.post(
        "/ideation-conversation/start",
        json={"competition_name": "데모 공모전", "user_idea": "소상공인이 손님 문의에 자동으로 답하는 챗봇"},
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    # 세션을 레거시 awaiting_developer_answer phase로 되돌린다(노드 직접 호출 — call_counter에
    # 영향을 주지 않는다).
    _force_session_to_awaiting_developer_answer(session_id, setup_responder)

    call_counter[0] = 0  # 이 요청부터 세기 시작 — 검증 대상은 오직 이 reply 한 번이다.
    reply_resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "잘 모르겠어"})
    assert reply_resp.status_code == 200
    body = reply_resp.json()

    assert call_counter[0] == 9, (
        f"위임 4회 + 회의 라운드 4회 + 캔버스 갱신 1회 = 9회가 아니라 {call_counter[0]}회가 호출됐습니다"
    )
    assert call_counter[0] < conv_route._MAX_LLM_CALLS_PER_REQUEST, "현실적인 최대 경로가 상한을 초과하면 안 된다"

    # 위임 -> 회의 순서로 실제 메시지가 이어졌는지도 함께 확인한다(요청 8번 — speaker_id와
    # 순서 보존, 이번엔 위임+회의가 한 요청 안에서 이어지는 특수 경로).
    speakers = [m["speaker_id"] for m in body["messages"]]
    assert "ideation_facilitator" in speakers
    assert speakers.count("ideation_facilitator") >= 2  # 위임 권고안 1개 + 회의 정리 1개.


def test_cap_trips_gracefully_instead_of_looping_forever(client: TestClient, monkeypatch):
    """요청 3번 — 상한이 실제로 "끝없는 반복"을 안전하게 끊어내는지 확인한다.

    용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) 이후 라운드테이블은 개발 위원이
    next_action="continue_round"를 반환하고 round<max_rounds인 동안 같은 HTTP 요청 안에서
    자동으로 다음 라운드(기획 최초 의견 -> 개발 검토 -> 진행자 정리, 라운드당 3회)를 계속
    만든다 — 이 자기 제어(max_rounds)가 정상적으로 세션 시작 때 결정된 값을 지키는 것과
    별개로, 상한(_MAX_LLM_CALLS_PER_REQUEST)은 "그 자기 제어 로직이 고장 나거나 max_rounds가
    비정상적으로 크게 설정된 경우"에도 무한 반복에 빠지지 않는다는 마지막 방어선이다.

    정상적인 첫 요청(단일 라운드, 실행 가능한 질문이 없어 discussion_complete)은 낮은 상한에서도
    문제없이 끝나지만, 이어지는 요청에서 LLM이 계속 continue_round를 반환하는 폭주
    시나리오를 흉내내면 상한이 이를 502로 안전하게 끊어내야 한다."""
    normal_responder = _comprehensive_responder(
        dev_review_stance="동의", discussion_review_stance="동의", discussion_next_action="await_user_decision"
    )
    monkeypatch.setattr(conv_route, "OpenAI", _make_fake_openai(normal_responder))

    start_resp = client.post(
        "/ideation-conversation/start",
        json={
            "competition_name": "데모 공모전",
            "user_idea": "소상공인이 손님 문의에 자동으로 답하는 챗봇",
            "max_rounds": 50,
        },
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]
    assert start_resp.json()["phase"] == "discussion_complete"

    # 이제부터 LLM이 항상 continue_round를 반환하는 폭주 시나리오로 바꾸고, 상한을 낮춘다
    # (라운드 하나당 planning+dev+facilitator=3회 — 상한 5로는 두 번째 라운드의 진행자
    # 호출에서 확실히 넘긴다).
    runaway_responder = _comprehensive_responder(
        dev_review_stance="동의", discussion_review_stance="동의", discussion_next_action="continue_round"
    )
    monkeypatch.setattr(conv_route, "OpenAI", _make_fake_openai(runaway_responder))
    monkeypatch.setattr(conv_route, "_MAX_LLM_CALLS_PER_REQUEST", 5)

    reply_resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "계속 진행해 주세요"})
    # 무한 루프에 빠지거나 서버가 죽지 않고, 상한 초과가 502(안전한 실패)로 변환된다
    # (ideation_conversation_preview.py::reply_conversation의 `except Exception: 502` 경로).
    assert reply_resp.status_code == 502

    # 세션이 손상되지 않고 여전히 조회 가능한지(직전 유효 상태 — 폭주 이전의 마지막으로
    # 성공한 라운드 결과 — 그대로 남아있는지)도 확인한다.
    get_resp = client.get(f"/ideation-conversation/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["phase"] == "discussion_complete"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
