# 작성자: 재인/Claude(2026-07-23, 아바타 페이싱 연동)
# 목적: POST /ideation-conversation/{session_id}/continue-turn/stream(NDJSON) 검증. 새
#       사용자 발언 없이 진행 중인 라운드의 다음 위원 발언 1건만 스트리밍하는지, phase
#       가드/세션 락/취소가 기존 /reply/stream과 동일하게 동작하는지 확인한다. 그래프
#       내부 로직(_route_next_expert_turn 등)은 전혀 건드리지 않으므로 여기서는 API
#       계층(스트리밍 브리지, phase 검증, 이벤트 조립)만 검증한다.
# import: fastapi.testclient, pytest; app.api.routes.ideation_conversation_preview 모듈.
#         test_ideation_conversation_streaming.py의 스텁(_discussion_payload_for 등)을
#         그대로 재사용한다(같은 스텁이어야 두 테스트 파일의 결과를 서로 비교할 수 있다).

import json
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config import settings  # noqa: E402
import app.api.routes.ideation_conversation_preview as conv_route  # noqa: E402

from test_ideation_conversation_streaming import (  # noqa: E402
    _FakeStreamState,
    _discussion_payload_for,
    _read_ndjson_events,
    _start_session,
    _sync_stub_llm_call,
)

NOTICE_AND_CRITERIA = {"competition_name": "데모 공모전", "notice_document": ""}
USER_IDEA = {"description": "소상공인이 손님 문의에 자동으로 답하는 챗봇"}


@pytest.fixture(autouse=True)
def _enable_flags(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_IDEATION_PREVIEW", True)
    monkeypatch.setattr(settings, "ENABLE_IDEATION_STREAMING", True)
    monkeypatch.setattr(conv_route, "_build_llm_call", _sync_stub_llm_call)
    conv_route.configure_ideation_trace(enabled=False, content_max_chars=500, stream_deltas=False)
    yield
    conv_route.configure_ideation_trace(enabled=False, content_max_chars=500, stream_deltas=False)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(conv_route.router)
    return TestClient(app)


class _CancelAfterFirstDiscussionCall:
    """기획 위원의 최초 발언 1건만 완료시키고 그 다음 discussion 호출부터 취소한다 —
    test_ideation_conversation_discovery.py::_force_session_to_awaiting_planning_answer와
    같은 방식으로, 실제 그래프 노드를 그대로 실행해 진짜배기 mid-round state를 얻는다
    (수작업으로 state 필드를 흉내내지 않는다)."""

    def __init__(self):
        self.discussion_calls = 0

    def __call__(self, prompt: str) -> str:
        from graph import IdeationCancelled

        if "[의견 규칙]" in prompt:
            self.discussion_calls += 1
            if self.discussion_calls > 1:
                raise IdeationCancelled("SESSION-X", "REQ-1")
            return json.dumps(_discussion_payload_for(prompt), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")


def _force_session_mid_round(session_id: str) -> None:
    """실제 그래프를 기획 위원 발언 1건까지만 실행해(취소로 강제 정지) 얻은 진짜
    phase="expert_discussion" state를 세션 스토어에 주입한다."""
    from graph import IdeationCancelled, start_ideation_conversation

    llm = _CancelAfterFirstDiscussionCall()
    try:
        start_ideation_conversation(
            session_id=session_id,
            notice_and_criteria=NOTICE_AND_CRITERIA,
            user_idea=USER_IDEA,
            llm_call=llm,
            max_rounds=1,
        )
        raise AssertionError("취소가 발생하지 않았습니다 — 스텁 설정을 확인하세요.")
    except IdeationCancelled as exc:
        state = exc.partial_state
    assert state["phase"] == "expert_discussion"
    assert state["messages"][-1]["speaker_id"] == "planning_expert"
    conv_route._store.update(session_id, state)


def test_reply_stream_single_turn_stops_at_one_expert_message(client: TestClient, monkeypatch):
    """실측 회귀 테스트 — "아이디어 선택하고 진행자 2번·기획 1번·개발 1번이 2초 간격으로
    그냥 다 나왔다": single_turn=true 없이 보낸 reply는(라운드를 새로 여는 경우) 위원 발언이
    한 번에 다 몰려 왔었다. single_turn=true를 보내면 새 라운드를 여는 이 reply조차 첫
    위원 발언 1건에서 멈춰야 한다(나머지는 continue-turn이 아바타 페이싱에 맞춰 이어서
    요청) — ReplyRequest.single_turn -> reply_ideation_conversation(stop_after_expert_turn=)
    배선을 검증한다."""
    session_id = _start_session(client)  # /start는 라운드 1 전체(기획+개발+진행자 정리)를
    # 한 번에 끝낸다(single_turn은 reply_ideation_conversation에만 배선했고 /start가 부르는
    # start_ideation_conversation에는 손대지 않았다 — 기존 동작 그대로).
    record_before = conv_route._store.get_record(session_id)
    expert_before = [
        m for m in record_before.state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")
    ]
    assert len(expert_before) == 2  # 라운드 1의 기획 1건 + 개발 1건.

    fake = _FakeStreamState()
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda sid, m: fake.build())

    with client.stream(
        "POST",
        f"/ideation-conversation/{session_id}/reply/stream",
        json={"message": "계속 논의해주세요", "single_turn": True},
    ) as resp:
        assert resp.status_code == 200
        events = _read_ndjson_events(resp)

    assert not any(e["type"] == "error" for e in events), events
    state_events = [e for e in events if e["type"] == "state"]
    assert len(state_events) == 1
    final_state = state_events[0]["state"]

    assert final_state["phase"] == "expert_discussion"
    expert_after = [m for m in final_state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
    # 라운드 2를 여는 이 reply가 위원 발언을 딱 1건만 더 추가해야 한다(라운드 전체를 몰아서
    # 만들지 않는다) — 이게 이 테스트의 핵심 검증 대상이다.
    assert len(expert_after) == len(expert_before) + 1, expert_after
    assert expert_after[-1]["speaker_id"] == "planning_expert"
    assert final_state.get("forced_next_speaker") is None


def test_continue_turn_rejects_wrong_phase(client: TestClient):
    """라운드가 이미 끝난(awaiting_user_decision) 세션에 continue-turn을 부르면 400 —
    프론트가 avatarPacingTimer를 잘못된 시점에 걸었다는 신호이므로 방어적으로 막는다."""
    session_id = _start_session(client)  # /start는 한 번에 라운드 전체를 끝낸다(기존 동작).
    resp = client.post(f"/ideation-conversation/{session_id}/continue-turn/stream", json={})
    assert resp.status_code == 400
    assert "expert_discussion" in resp.json()["detail"]


def test_continue_turn_unknown_session_returns_404(client: TestClient):
    resp = client.post("/ideation-conversation/NOPE/continue-turn/stream", json={})
    assert resp.status_code == 404


def test_continue_turn_streams_exactly_one_more_expert_message(client: TestClient, monkeypatch):
    """핵심 검증 — 기획 위원 발언까지만 끝난 라운드에서 continue-turn을 부르면, 새
    사용자 발언 없이 개발 위원 발언 딱 1건만 추가되고 phase는 이번 범위(기획/개발 사이)라면
    "expert_discussion"을 유지해야 한다(이 stub은 개발 위원 턴에서 issue_resolved=True를
    반환하므로 실제로는 진행자로 넘어가 라운드가 끝난다 — 그래도 이 호출 자체는 "위원 발언
    1건"만 스트리밍한다는 계약은 동일하다)."""
    session_id = _start_session(client)
    _force_session_mid_round(session_id)

    fake = _FakeStreamState()
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda sid, m: fake.build())

    with client.stream(
        "POST", f"/ideation-conversation/{session_id}/continue-turn/stream", json={}
    ) as resp:
        assert resp.status_code == 200
        events = _read_ndjson_events(resp)

    assert not any(e["type"] == "error" for e in events), events
    state_events = [e for e in events if e["type"] == "state"]
    assert len(state_events) == 1
    final_state = state_events[0]["state"]

    expert_messages = [m for m in final_state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
    # 이전에는(취소 시점) 기획 위원 1건뿐이었다 — 이번 호출로 정확히 1건(개발 위원)만 늘어야 한다.
    assert len(expert_messages) == 2
    assert expert_messages[-1]["speaker_id"] == "dev_expert"

    # message_start/delta/end가 실제로 스트리밍됐는지(가짜 타이핑이 아니라 진짜 청크 전달).
    message_start_events = [e for e in events if e["type"] == "message_start"]
    assert len(message_start_events) == 1
    assert message_start_events[0]["speaker_id"] == "dev_expert"

    # forced_next_speaker가 응답 state에 새어나가지 않아야 한다(다음 라운드 진입 오염 방지).
    assert final_state.get("forced_next_speaker") is None


def test_continue_turn_respects_session_lock(client: TestClient, monkeypatch):
    """이미 처리 중인 세션에 continue-turn을 또 부르면 409 — /reply/stream과 동일한 락 계약
    (test_ideation_conversation_streaming.py::test_concurrent_reply_to_same_session_returns_409과
    똑같은 방식으로, 백그라운드 스레드가 실제로 락을 잡을 시간을 준 뒤 두 번째 요청을 보낸다)."""
    session_id = _start_session(client)
    _force_session_mid_round(session_id)

    fake = _FakeStreamState(chunk_size=3, delay=0.02)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda sid, m: fake.build())

    results = {}

    def first_call():
        with client.stream(
            "POST", f"/ideation-conversation/{session_id}/continue-turn/stream", json={}
        ) as resp:
            results["first_status"] = resp.status_code
            list(resp.iter_lines())

    thread = threading.Thread(target=first_call)
    thread.start()
    time.sleep(0.05)
    second_resp = client.post(f"/ideation-conversation/{session_id}/continue-turn/stream", json={})
    thread.join(timeout=15)

    assert results.get("first_status") == 200
    assert second_resp.status_code == 409
