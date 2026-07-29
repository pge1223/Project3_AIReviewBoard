# 작성자: 용준/Claude(2026-07-29, 요청: expert_analysis_query 전용 단일 응답 경로 +
#       /reply, /reply/stream 유형별 HTTP 엔드투엔드 검증)
# 목적: POST /ideation-conversation/{session_id}/reply(동기)와 .../reply/stream(NDJSON)이
#       document_fact_query / session_state_query / expert_analysis_query /
#       ideation_discussion_request 네 가지 request_type에 대해 실제 라우트 코드
#       (app.api.routes.ideation_conversation_preview)와 실제 그래프 코드(ai/meeting/graph)를
#       그대로 실행했을 때 계약대로 동작하는지 검증한다. 실제 OpenAI 호출 대신 기존 테스트
#       (test_ideation_conversation_streaming.py)와 동일한 스텁 LLM을 재사용한다 — 스텁만
#       교체하고 나머지 API/그래프 코드는 전혀 우회하지 않는다("실제 백엔드 실행" 요구사항을
#       외부 LLM 비용/키 없이 충족).
# import: fastapi.testclient, pytest; test_ideation_conversation_streaming.py의 스텁 재사용.

import json
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

from test_ideation_conversation_streaming import (  # noqa: E402
    _FakeStreamState,
    _read_ndjson_events,
    _start_session,
    _sync_stub_llm_call,
)


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


# request_type별 대표 사용자 발화 — ai/meeting/graph/ideation_conv_nodes.py::classify_query_type
# 이 실제로 이 값들로 분류하는지는 test_ideation_topic_query.py 단위 테스트가 이미 검증한다.
# 여기서는 "그 분류 결과가 HTTP 계층까지 그대로 전달되고, 계약대로 종료되는지"만 본다.
_CASES = {
    "document_fact_query": "이번 공모전 제출 서류가 뭔가요?",
    "session_state_query": "지금 어떤 단계인가요?",
    "expert_analysis_query": "구현 가능성과 MVP 위험을 분석해 주세요",
    "ideation_discussion_request": "이 부분에 대해 위원들이 좀 더 토론해 주세요",
}
# document_fact_query/session_state_query/expert_analysis_query는 위원 발언 1건(또는
# 결정론적 문장)으로 끝나야 하는 단일 응답 유형이다(ai/meeting/graph/ideation_conv_run.py::
# _SINGLE_TURN_REQUEST_TYPES). ideation_discussion_request만 기존 다회 라운드
# (기획->개발->진행자 정리)가 유지돼야 한다.
_SINGLE_TURN_TYPES = {"document_fact_query", "session_state_query", "expert_analysis_query"}


def _start_session_id(client: TestClient) -> str:
    return _start_session(client)


@pytest.mark.parametrize("request_type,user_message", list(_CASES.items()))
def test_reply_sync_terminates_with_correct_request_type_and_phase(client, request_type, user_message):
    session_id = _start_session_id(client)
    baseline = conv_route._store.get_record(session_id)
    baseline_count = len(baseline.state["messages"])

    resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": user_message})
    assert resp.status_code == 200
    body = resp.json()

    assert body["request_type"] == request_type
    assert body["phase"] == "discussion_complete"

    new_messages = body["messages"][baseline_count:]
    speakers = [m["speaker_id"] for m in new_messages]
    assert speakers[0] == "user"
    if request_type in _SINGLE_TURN_TYPES:
        # 위원 발언 1건 + 사용자 메시지만 추가되고, 다음 위원/진행자 정리로 이어지지 않는다
        # ("불필요한 다회 라운드가 생성되지 않음").
        assert len(speakers) == 2, f"{request_type}: 위원 발언 1건만 추가돼야 하는데 {speakers}"
        assert speakers[1] in ("planning_expert", "dev_expert")
    else:
        assert speakers == ["user", "planning_expert", "dev_expert", "ideation_facilitator"]

    # 정상 direct answer 뒤 fallback 문구가 붙지 않는지(예: "현재 자료로 확인할 수 없다" 같은
    # 근거 없음 안내가 실제 답변에 섞이지 않아야 한다) — 스텁은 항상 evidence 없이 답을
    # 만들므로, 최소한 위원 발언 content가 비어 있지 않은지만 확인한다.
    last_committee_message = new_messages[-1]
    assert last_committee_message["content"]


@pytest.mark.parametrize("request_type,user_message", list(_CASES.items()))
def test_reply_stream_ndjson_contract_per_request_type(client, request_type, user_message, monkeypatch):
    session_id = _start_session_id(client)
    baseline = conv_route._store.get_record(session_id)
    baseline_count = len(baseline.state["messages"])

    fake = _FakeStreamState(chunk_size=4)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda sid, m: fake.build())

    with client.stream(
        "POST",
        f"/ideation-conversation/{session_id}/reply/stream",
        json={"message": user_message},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        events = _read_ndjson_events(resp)  # 각 줄이 파싱 가능해야 한다(파싱 실패 시 예외).

    types = [e["type"] for e in events]
    # 마지막 done 이벤트 존재 + 그 직전에 최종 state 이벤트가 있어야 한다.
    assert types[-1] == "done"
    assert "state" in types

    final_state = next(e["state"] for e in events if e["type"] == "state")
    assert final_state["request_type"] == request_type
    assert final_state["phase"] == "discussion_complete"
    # done 이벤트의 phase가 최종 state의 phase와 정확히 일치해야 한다.
    assert events[-1]["phase"] == final_state["phase"]

    new_messages = final_state["messages"][baseline_count:]
    speakers = [m["speaker_id"] for m in new_messages]
    assert speakers[0] == "user"
    if request_type in _SINGLE_TURN_TYPES:
        assert len(speakers) == 2, f"{request_type}: 위원 발언 1건만 추가돼야 하는데 {speakers}"
    else:
        assert speakers == ["user", "planning_expert", "dev_expert", "ideation_facilitator"]

    # 메시지 중복 없음: id 기준으로 유니크해야 한다.
    message_ids = [m["id"] for m in final_state["messages"] if "id" in m]
    assert len(message_ids) == len(set(message_ids))


@pytest.mark.parametrize("request_type,user_message", list(_CASES.items()))
def test_reply_sync_and_stream_are_semantically_equivalent(client, request_type, user_message, monkeypatch):
    """같은 입력에 대해 /reply(동기)와 /reply/stream(NDJSON)의 최종 state가 의미상
    동일해야 한다 — request_type, phase, 새로 추가된 화자 순서가 정확히 같아야 한다
    (스트리밍 청크 분할로 인한 content 차이는 스텁이 양쪽에 동일 payload를 쓰므로 없다)."""
    sync_session_id = _start_session_id(client)
    sync_baseline_count = len(conv_route._store.get_record(sync_session_id).state["messages"])
    sync_resp = client.post(f"/ideation-conversation/{sync_session_id}/reply", json={"message": user_message})
    assert sync_resp.status_code == 200
    sync_state = sync_resp.json()

    stream_session_id = _start_session_id(client)
    stream_baseline_count = len(conv_route._store.get_record(stream_session_id).state["messages"])
    fake = _FakeStreamState(chunk_size=4)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda sid, m: fake.build())
    with client.stream(
        "POST",
        f"/ideation-conversation/{stream_session_id}/reply/stream",
        json={"message": user_message},
    ) as resp:
        events = _read_ndjson_events(resp)
    stream_state = next(e["state"] for e in events if e["type"] == "state")

    assert sync_state["request_type"] == stream_state["request_type"] == request_type
    assert sync_state["phase"] == stream_state["phase"] == "discussion_complete"

    sync_new_speakers = [
        m["speaker_id"] for m in sync_state["messages"][sync_baseline_count:]
    ]
    stream_new_speakers = [
        m["speaker_id"] for m in stream_state["messages"][stream_baseline_count:]
    ]
    assert sync_new_speakers == stream_new_speakers


def test_request_type_field_is_optional_for_legacy_sessions_without_it(client, monkeypatch):
    """request_type이 없던 기존 세션(구버전 state)에 대한 reply도 정상 동작해야 한다 —
    request_type은 선택 필드여야 한다는 요구사항 검증."""
    session_id = _start_session_id(client)
    record = conv_route._store.get_record(session_id)
    legacy_state = dict(record.state)
    legacy_state.pop("request_type", None)
    conv_route._store.update(session_id, legacy_state)

    resp = client.post(
        f"/ideation-conversation/{session_id}/reply",
        json={"message": _CASES["ideation_discussion_request"]},
    )
    assert resp.status_code == 200
    assert resp.json()["request_type"] == "ideation_discussion_request"


def test_reply_stream_ends_cleanly_when_llm_raises(client, monkeypatch):
    """예외 발생 시에도 스트림이 정상 종료돼야 한다(프론트의 sending 상태 해제 계약) —
    done 이벤트 없이 스트림이 그냥 끊기면 프론트가 영원히 sending 상태에 남는다."""
    session_id = _start_session_id(client)

    def _broken_backends(session_id, model):
        def stream_chat_completion(prompt: str):
            raise RuntimeError("stub 강제 오류")

        return stream_chat_completion

    monkeypatch.setattr(conv_route, "_build_streaming_backends", _broken_backends)

    with client.stream(
        "POST",
        f"/ideation-conversation/{session_id}/reply/stream",
        json={"message": _CASES["expert_analysis_query"]},
    ) as resp:
        assert resp.status_code == 200
        events = _read_ndjson_events(resp)

    types = [e["type"] for e in events]
    assert types, "예외가 나도 이벤트가 최소 하나(error/done)는 있어야 한다"
    assert types[-1] in ("done", "error")
    # done이 아니라 error로 끝났더라도, 그 뒤에 별도의 hang 없이 스트림이 닫혀야 한다는
    # 사실 자체는 _read_ndjson_events가 예외 없이 반환됐다는 것으로 이미 검증됐다.
