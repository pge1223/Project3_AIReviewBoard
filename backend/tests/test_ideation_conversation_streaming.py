# 작성자: 용준/Claude(2026-07-21)
# 목적: POST /ideation-conversation/{session_id}/reply/stream(NDJSON 실시간 스트리밍)
#       검증. 실제 OpenAI 호출 대신 app.api.routes.ideation_conversation_preview::
#       _build_streaming_backends를 monkeypatch해 제어 가능한 가짜 스트리밍 LLM을 쓴다
#       (test_ideation_conversation_discovery.py가 _build_llm_call을 monkeypatch하는
#       것과 같은 패턴). 그래프/노드 코드는 전혀 건드리지 않는다 — 이 테스트는 API 계층
#       (스트리밍 브리지, 이벤트 조립, 세션 락)만 검증한다.
# import: fastapi.testclient, pytest; app.api.routes.ideation_conversation_preview 모듈.

import json
import logging
import re
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


def _discussion_payload_for(prompt: str) -> dict:
    """[의견 규칙] 스텁 응답을 만든다 — 동기식(_sync_stub_llm_call)과 스트리밍
    (_FakeStreamState) 양쪽에서 똑같이 재사용해, "스트리밍 결과와 동기식 결과가 의미상
    같다"는 테스트가 서로 다른 스텁 문구 차이 때문에 실패하지 않게 한다. 용준/Claude
    (2026-07-21, 요청: 전문가 라운드테이블 전환) 이후 사용자에게 실시간으로 보이는 첫
    스트리밍 대상은 1:1 인터뷰 질문이 아니라 라운드테이블의 전문가 의견(discussion)이다."""
    is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
    is_dev = not is_planning
    speaker = "planning_expert" if is_planning else "dev_expert"
    return {
        "stance": "보완",
        # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) —
        # 화면·스트리밍에 실제로 노출되는 필드는 spoken_text 하나뿐이다. 한글·줄바꿈·
        # 인용부호가 청크 경계에서 잘려도 올바르게 재조립되는지 검증하려고 그 특징을 그대로
        # spoken_text에 담는다(예전에는 judgment/reason 등 여러 필드에 나눠 담았다).
        "spoken_text": f"[{speaker}] 한글 발화 내용\n줄바꿈도 있습니다 \"인용부호\" 포함",
        "judgment": f"[{speaker}] 한글 판단 내용\n줄바꿈도 있습니다",
        "reason": f"[{speaker}] 판단 근거 \"인용부호\" 포함",
        "suggestion": f"[{speaker}] 제안 내용입니다",
        "interim_conclusion": f"[{speaker}] 현재 임시 결론입니다",
        "responding_to": "상대 전문가가 방금 말한 판단" if is_dev else None,
        "agreement": "동의 지점입니다" if is_dev else "",
        "concern": "",
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
        "needs_counterpart_response": not is_dev,
        "recommended_next_speaker": "ideation_facilitator" if is_dev else "dev_expert",
        "issue_resolved": bool(is_dev),
        "needs_user_input": False,
        "user_question": None,
    }


def _facilitator_summary_payload_for(prompt: str) -> dict:
    return {
        "agreements": [],
        "disagreements": [],
        "facilitator_summary": "두 전문가가 이번 라운드 의견을 정리했습니다.",
        "spoken_text": "두 위원이 이번 라운드 의견을 정리했습니다.",
        "needs_user_decision": False,
        "user_question": None,
    }


def _canvas_payload() -> dict:
    """test_ideation_conversation_discovery.py의 캔버스 갱신 스텁과 같은 payload."""
    return {
        "problem": "문의 응대 부담",
        "target_user": "소상공인",
        "core_value": "응대 시간 절감",
        "solution": "FAQ 자동 응답",
        "differentiation": "저비용 구축",
        "feasibility": "medium",
        "risks": ["오답 위험"],
        "contest_fit": "실현가능성 기준 대응",
    }


def _sync_stub_llm_call(session_id: str, model: str):
    """기존 동기식 /start, /reply(비교용)에 쓰는 완성 응답 스텁 — 스트리밍 스텁과 정확히
    같은 내용을 반환해야 "스트리밍 결과와 동기식 결과가 의미상 같다"를 비교할 수 있다."""

    def llm_call(prompt: str) -> str:
        if "[의견 규칙]" in prompt:
            return json.dumps(_discussion_payload_for(prompt), ensure_ascii=False)
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(_facilitator_summary_payload_for(prompt), ensure_ascii=False)
        if "[질문 규칙]" in prompt:
            is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
            speaker = "planning_expert" if is_planning else "dev_expert"
            return json.dumps(
                {
                    "spoken_text": f"[{speaker}] 한글 발화 질문 \"인용부호\" 포함",
                    "judgment": f"[{speaker}] 한글 판단 내용\n줄바꿈도 있습니다",
                    "question": f"[{speaker}] 핵심 질문 \"인용부호\" 포함",
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
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")

    return llm_call


class _FakeStreamState:
    """[의견 규칙]/[진행자 정리 규칙] 스트리밍 호출을 몇 글자씩 쪼개 흉내내는 가짜 백엔드.
    invalid_first=True면 전체 스트림에서 가장 먼저 들어오는 호출(항상 기획 위원의 최초
    의견) 한 번만 검증 실패를 유도해 재시도 시나리오를 만든다."""

    def __init__(self, chunk_size=3, delay=0.0, invalid_first=False, stream_started_event=None):
        self.chunk_size = chunk_size
        self.delay = delay
        self.invalid_first = invalid_first
        self.stream_started_event = stream_started_event
        self.call_counts = {"discussion": 0, "facilitator_summary": 0}
        self._first_call_done = False

    def _discussion_payload(self, prompt: str) -> dict:
        if self.invalid_first and not self._first_call_done:
            self._first_call_done = True
            return {"judgment": "", "reason": ""}  # 검증 실패 유도 -> 재시도
        self._first_call_done = True
        return _discussion_payload_for(prompt)

    def build(self):
        def stream_chat_completion(prompt: str):
            if "[의견 규칙]" in prompt:
                self.call_counts["discussion"] += 1
                payload = self._discussion_payload(prompt)
            elif "[진행자 정리 규칙]" in prompt:
                self.call_counts["facilitator_summary"] += 1
                payload = _facilitator_summary_payload_for(prompt)
            else:
                raise AssertionError(f"스트리밍 대상이 아닌 프롬프트가 stream_chat_completion으로 왔습니다: {prompt[:100]}")
            raw = json.dumps(payload, ensure_ascii=False)
            for i in range(0, len(raw), self.chunk_size):
                if self.stream_started_event is not None:
                    self.stream_started_event.set()
                if self.delay:
                    time.sleep(self.delay)
                yield raw[i : i + self.chunk_size]

        def call_chat_completion(prompt: str) -> str:
            # 가은/Claude(2026-07-22, 캔버스 자동 갱신) — canvas_update는 화면 메시지를 만들지
            # 않아 스트리밍 대상이 아니고, 항상 이 동기식 경로로 온다.
            if "[캔버스 갱신 규칙]" in prompt:
                return json.dumps(_canvas_payload(), ensure_ascii=False)
            # 가은/Claude(2026-07-22, 회의 시작 대기 체감 개선 1단계) — /start/stream의
            # discovery 경로(후보 생성/검토)도 화면 메시지가 없는 phase-only 호출이라 이
            # 동기식 경로로 온다(test_ideation_conversation_discovery.py의 stub과 같은 payload).
            if "[후보 생성 규칙]" in prompt:
                return json.dumps(
                    {
                        "contest_analysis": {
                            "purpose": "p", "key_criteria": ["a"], "required_tech_or_theme": ["b"],
                            "suitable_problem_domains": ["c"], "constraints": ["d"], "unknown_from_notice": ["e"],
                        },
                        "candidates": [
                            {
                                "candidate_id": "candidate_1", "title": "후보1", "problem": "문제1",
                                "target_user": "사용자1", "usage_scenario": "상황1", "core_value": "가치1",
                                "solution": "해결1", "main_features": ["기능1"], "differentiation": "차별1",
                                "contest_fit": "적합1", "success_metrics": ["지표1"],
                            },
                            {
                                "candidate_id": "candidate_2", "title": "후보2", "problem": "문제2",
                                "target_user": "사용자2", "usage_scenario": "상황2", "core_value": "가치2",
                                "solution": "해결2", "main_features": ["기능2"], "differentiation": "차별2",
                                "contest_fit": "적합2", "success_metrics": ["지표2"],
                            },
                        ],
                    },
                    ensure_ascii=False,
                )
            if "[검토 규칙]" in prompt:
                return json.dumps(
                    {
                        "candidate_reviews": [
                            {
                                "candidate_id": "candidate_1", "required_data": ["d1"], "technical_approach": "t1",
                                "mvp_scope": "m1", "feasibility": "high", "risks": ["r1"], "dev_notes": None,
                            },
                            {
                                "candidate_id": "candidate_2", "required_data": ["d2"], "technical_approach": "t2",
                                "mvp_scope": "m2", "feasibility": "medium", "risks": ["r2"], "dev_notes": None,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:100]}")

        return stream_chat_completion, call_chat_completion


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


def _start_session(client: TestClient) -> str:
    resp = client.post(
        "/ideation-conversation/start",
        json={
            "competition_name": "데모 공모전",
            "user_idea": "소상공인이 손님 문의에 자동으로 답하는 챗봇",
        },
    )
    assert resp.status_code == 200
    return resp.json()["session_id"]


def _read_ndjson_events(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if not line:
            continue
        events.append(json.loads(line))
    return events


def test_streaming_disabled_returns_404(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_IDEATION_STREAMING", False)
    session_id = _start_session(client)
    resp = client.post(f"/ideation-conversation/{session_id}/reply/stream", json={"message": "답변1"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /start/stream — 가은/Claude(2026-07-22, 요청: 회의 시작 대기 체감 개선 1단계)
# ---------------------------------------------------------------------------


def test_start_stream_disabled_returns_404(client: TestClient, monkeypatch):
    """플래그가 꺼져 있으면 404 — 프론트가 이 응답을 보고 동기식 /start로 폴백한다."""
    monkeypatch.setattr(settings, "ENABLE_IDEATION_STREAMING", False)
    resp = client.post(
        "/ideation-conversation/start/stream",
        json={"competition_name": "데모 공모전", "user_idea": ""},
    )
    assert resp.status_code == 404


def test_start_stream_discovery_emits_phase_events_then_final_state(client: TestClient, monkeypatch):
    """board의 실제 경로(discovery, user_idea 없음): 후보 생성/실현 가능성 검토는 화면
    메시지를 만들지 않으므로 message_delta 없이 phase 이벤트만 나가고, 마지막에 최종
    state 이벤트가 온다. 세션도 정상 생성되어 이후 GET으로 이어받을 수 있어야 한다."""
    fake = _FakeStreamState(chunk_size=4)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())

    with client.stream(
        "POST",
        "/ideation-conversation/start/stream",
        json={"competition_name": "데모 공모전", "user_idea": ""},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        events = _read_ndjson_events(resp)

    types = [e["type"] for e in events]
    assert types[-1] == "state"
    phase_labels = [e.get("label", "") for e in events if e["type"] == "phase"]
    assert any("아이디어 후보를 만들고" in label for label in phase_labels)
    assert any("실현 가능성" in label for label in phase_labels)
    assert "message_delta" not in types  # discovery 시작은 화면 메시지를 만들지 않는다.

    final_state = events[-1]["state"]
    assert final_state["phase"] == "awaiting_candidate_selection"
    assert len(final_state["idea_candidates"]) == 2
    # 세션이 스토어에 저장돼 재접속(GET)으로 이어받을 수 있다.
    get_resp = client.get(f"/ideation-conversation/{final_state['session_id']}")
    assert get_resp.status_code == 200


def test_start_stream_refinement_streams_roundtable_messages(client: TestClient, monkeypatch):
    """refinement 모드(user_idea 있음)로 시작하면 라운드테이블 발언이 reply/stream과
    동일하게 message_delta(타이핑 효과)로 스트리밍된다 — 같은 make_streaming_llm_call을
    그대로 쓰므로 추가 구현 없이 따라오는 동작을 고정한다."""
    fake = _FakeStreamState(chunk_size=4)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())

    with client.stream(
        "POST",
        "/ideation-conversation/start/stream",
        json={"competition_name": "데모 공모전", "user_idea": "소상공인 손님 문의 자동 응대 챗봇"},
    ) as resp:
        events = _read_ndjson_events(resp)

    types = [e["type"] for e in events]
    assert "message_start" in types
    assert "message_delta" in types
    assert types[-1] == "state"
    assert events[-1]["state"]["phase"] == "discussion_complete"


def test_stream_event_order_message_start_delta_end_state(client: TestClient, monkeypatch):
    fake = _FakeStreamState(chunk_size=4)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())

    session_id = _start_session(client)
    with client.stream(
        "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "타깃은 동네 카페 사장님입니다"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        events = _read_ndjson_events(resp)

    types = [e["type"] for e in events]
    assert "message_start" in types
    assert "message_end" in types
    assert types[-1] == "state"
    delta_indices = [i for i, t in enumerate(types) if t == "message_delta"]
    assert delta_indices, "message_delta 이벤트가 하나도 없습니다"

    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 한 번의 /reply/stream 호출
    # 안에서 기획 위원 -> 개발 위원 -> 진행자 순으로 여러 메시지가 순차적으로 스트리밍된다.
    # 각 메시지는 자신의 message_start/message_delta.../message_end 세트 안에서만 등장하고
    # (겹치지 않고), 이전 메시지가 끝나야 다음 메시지가 시작돼야 한다.
    open_id = None
    seen_ids: list[str] = []
    for e in events:
        if e["type"] == "message_start":
            assert open_id is None, "이전 메시지가 끝나기 전에 새 메시지가 시작됐다"
            open_id = e["message_id"]
            seen_ids.append(open_id)
        elif e["type"] == "message_delta":
            assert e["message_id"] == open_id, "delta가 현재 열린 메시지를 가리키지 않는다"
        elif e["type"] == "message_end":
            assert e["message_id"] == open_id
            open_id = None
    assert open_id is None
    assert len(seen_ids) >= 1


def test_stream_delta_trace_requires_explicit_delta_flag(client: TestClient, monkeypatch, caplog):
    fake = _FakeStreamState(chunk_size=4)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())
    caplog.set_level(logging.DEBUG, logger="ai.meeting.ideation_trace")

    conv_route.configure_ideation_trace(enabled=True, content_max_chars=120, stream_deltas=False)
    session_id = _start_session(client)
    with client.stream(
        "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "답변1"}
    ) as response:
        _read_ndjson_events(response)
    assert "IDEATION_STREAM_DELTA" not in caplog.text

    caplog.clear()
    conv_route.configure_ideation_trace(enabled=True, content_max_chars=120, stream_deltas=True)
    session_id = _start_session(client)
    with client.stream(
        "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "답변1"}
    ) as response:
        _read_ndjson_events(response)
    assert "IDEATION_STREAM_DELTA" in caplog.text
    assert "spoken_text" not in caplog.text  # 내부 JSON 키/원문은 delta 로그에 노출하지 않는다.


def test_korean_text_reassembles_correctly_across_small_chunks(client: TestClient, monkeypatch):
    fake = _FakeStreamState(chunk_size=1)  # 극단적으로 잘게 쪼갠다(멀티바이트 경계 포함).
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())

    session_id = _start_session(client)
    with client.stream(
        "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "답변1"}
    ) as resp:
        events = _read_ndjson_events(resp)

    deltas = "".join(e["delta"] for e in events if e["type"] == "message_delta")
    # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — 스트리밍
    # 델타에는 이제 spoken_text 원문만 나타난다. 한글·줄바꿈·인용부호가 극단적으로 잘게
    # 쪼개져도(chunk_size=1) 정확히 재조립되는지 확인한다.
    assert "한글 발화 내용" in deltas
    assert "줄바꿈도 있습니다" in deltas
    assert '인용부호' in deltas
    # 예전 보고서형 헤더([기획 관점]/[임시 결론] 등)는 더 이상 어디에도 나타나지 않는다.
    assert "[기획 관점]" not in deltas
    assert "[임시 결론]" not in deltas
    assert "[기술 검토]" not in deltas
    # message_delta 텍스트 안에 JSON 구조(중괄호·필드명)가 노출되면 안 된다.
    assert "judgment" not in deltas
    assert "interim_conclusion" not in deltas
    assert "spoken_text" not in deltas
    assert "{" not in deltas and "}" not in deltas


def test_streaming_final_state_matches_sync_reply_semantically(client: TestClient, monkeypatch):
    fake = _FakeStreamState(chunk_size=5)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())

    stream_session_id = _start_session(client)
    with client.stream(
        "POST", f"/ideation-conversation/{stream_session_id}/reply/stream", json={"message": "답변1"}
    ) as resp:
        events = _read_ndjson_events(resp)
    stream_state = next(e["state"] for e in events if e["type"] == "state")

    sync_session_id = _start_session(client)
    sync_resp = client.post(f"/ideation-conversation/{sync_session_id}/reply", json={"message": "답변1"})
    assert sync_resp.status_code == 200
    sync_state = sync_resp.json()

    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — "awaiting_user_decision"
    # 상태에서 사용자가 자유 발언을 남기면 새 라운드(기획 -> 개발 -> 진행자)가 실행되고,
    # dev 스텁이 next_action="await_user_decision"을 반환하므로 다시 "awaiting_user_decision"
    # 으로 멈춘다.
    assert stream_state["phase"] == sync_state["phase"] == "discussion_complete"
    stream_last = stream_state["messages"][-1]
    sync_last = sync_state["messages"][-1]
    assert stream_last["speaker_id"] == sync_last["speaker_id"]
    assert stream_last["content"] == sync_last["content"]


def test_multiple_persona_messages_keep_order_and_speaker_id(client: TestClient, monkeypatch):
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 사용자가 "awaiting_user_decision"
    상태에서 자유 발언을 남기면 기획 위원 -> 개발 위원 -> 진행자 순으로 라운드 전체가
    스트리밍된다. 스트리밍된 message_start들의 speaker_id 순서가 실제 대화 순서와
    일치하는지 확인한다."""
    fake = _FakeStreamState(chunk_size=6)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())

    session_id = _start_session(client)
    with client.stream(
        "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "답변1"}
    ) as resp:
        events = _read_ndjson_events(resp)

    starts = [e for e in events if e["type"] == "message_start"]
    assert [s["speaker_id"] for s in starts] == ["planning_expert", "dev_expert", "ideation_facilitator"]
    state_event = next(e for e in events if e["type"] == "state")
    speakers = [m["speaker_id"] for m in state_event["state"]["messages"]]
    # user(답변1) 다음에 기획 -> 개발 -> 진행자 순으로 이어진다.
    assert speakers[-4:] == ["user", "planning_expert", "dev_expert", "ideation_facilitator"]


def test_retry_emits_message_reset_before_new_message_start(client: TestClient, monkeypatch):
    fake = _FakeStreamState(chunk_size=4, invalid_first=True)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())

    session_id = _start_session(client)
    with client.stream(
        "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "답변1"}
    ) as resp:
        events = _read_ndjson_events(resp)

    types = [e["type"] for e in events]
    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 검증 실패는 라운드의 첫
    # 발언(기획 위원)에서만 유도된다(재시도 1회로 성공). 이어지는 개발 위원(1회)·진행자
    # (1회)는 정상적으로 성공하므로, 전체 메시지 세트는 4개(기획 실패분 + 기획 재시도분 +
    # 개발 + 진행자)다.
    assert types.count("message_reset") == 1
    assert types.count("message_start") == 4
    assert types.count("message_end") == 4
    reset_event = next(e for e in events if e["type"] == "message_reset")
    starts = [e for e in events if e["type"] == "message_start"]
    first_start, retry_start = starts[0], starts[1]
    assert reset_event["message_id"] == first_start["message_id"]
    # 용준/Claude(2026-07-23, 요청: 스트리밍 UX 버그 수정) — reset은 이유/재시도 여부를
    # 함께 실어야 프런트가 "검토 중" 표시와 "곧 fallback" 표시를 구분할 수 있다. 그리고
    # 바로 다음 message_start는 이 reset된 말풍선의 자리를 이어받는다는 것을
    # supersedes_message_id로 명시해야, speaker_id만으로 추정하지 않고도 같은 자리에서
    # 교체할 수 있다.
    assert reset_event.get("reason") == "missing_or_empty_field:judgment_or_reason"
    assert reset_event.get("will_retry") is True
    assert retry_start.get("supersedes_message_id") == first_start["message_id"]
    # 재시도가 아닌 정상적인 다음 발언(개발 위원, 진행자)에는 supersedes_message_id가
    # 없어야 한다 — 다른 위원의 정상 발언을 재시도로 오인해서는 안 된다.
    assert all("supersedes_message_id" not in s for s in starts[2:])
    assert fake.call_counts["discussion"] == 3  # 기획(최초 실패 1 + 재시도 1) + 개발(1)
    assert fake.call_counts["facilitator_summary"] == 1


def test_llm_failure_emits_error_event(client: TestClient, monkeypatch):
    def broken_backends(session_id, model):
        def stream_chat_completion(prompt):
            raise RuntimeError("네트워크 오류 시뮬레이션")

        def call_chat_completion(prompt):
            raise RuntimeError("네트워크 오류 시뮬레이션")

        return stream_chat_completion, call_chat_completion

    monkeypatch.setattr(conv_route, "_build_streaming_backends", broken_backends)

    session_id = _start_session(client)
    with client.stream(
        "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "답변1"}
    ) as resp:
        events = _read_ndjson_events(resp)

    types = [e["type"] for e in events]
    assert "error" in types
    error_event = next(e for e in events if e["type"] == "error")
    assert error_event["code"] == "llm_failure"
    assert "네트워크 오류 시뮬레이션" not in error_event["message"]  # 원본 예외 메시지를 그대로 노출하지 않는다.


def test_session_state_not_corrupted_after_stream_and_llm_failure(client: TestClient, monkeypatch):
    """오류가 나도 세션 state는 실패 이전의 유효한 상태로 남아 있어야 한다(손상되지 않음).
    실패 후 다시 정상 스텁으로 answer 재전송하면 정상 진행되는지까지 확인한다."""

    def broken_backends(session_id, model):
        def stream_chat_completion(prompt):
            raise RuntimeError("boom")

        def call_chat_completion(prompt):
            raise RuntimeError("boom")

        return stream_chat_completion, call_chat_completion

    monkeypatch.setattr(conv_route, "_build_streaming_backends", broken_backends)
    session_id = _start_session(client)
    with client.stream(
        "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "답변1"}
    ) as resp:
        _read_ndjson_events(resp)

    # 세션이 여전히 조회 가능하고(손상/삭제되지 않음), phase도 실패 이전 그대로다.
    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 세션은 시작 직후 이미
    # 라운드테이블 한 라운드를 마치고 "awaiting_user_decision"으로 멈춰 있었다.
    get_resp = client.get(f"/ideation-conversation/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["phase"] == "discussion_complete"

    # 락도 정상적으로 풀려서 다음 요청(정상 스텁)이 처리된다.
    fake = _FakeStreamState(chunk_size=4)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda sid, m: fake.build())
    resp2 = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "답변1"})
    assert resp2.status_code == 200
    assert resp2.json()["phase"] == "discussion_complete"


def test_concurrent_reply_to_same_session_returns_409(client: TestClient, monkeypatch):
    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 한 번의 /reply/stream 호출이
    # 이제 기획/개발/진행자 3개 메시지를 순차 스트리밍하므로(과거 1개 메시지 대비 총 대기
    # 시간이 늘어난다), chunk당 delay를 줄이고 join 타임아웃을 넉넉히 잡는다.
    fake = _FakeStreamState(chunk_size=3, delay=0.02)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())

    session_id = _start_session(client)
    results = {}

    def first_call():
        with client.stream(
            "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "답변1"}
        ) as resp:
            results["first_status"] = resp.status_code
            list(resp.iter_lines())  # 끝까지 소비해서 워커가 마무리되게 한다.

    thread = threading.Thread(target=first_call)
    thread.start()
    time.sleep(0.05)  # 첫 요청이 락을 잡을 시간을 준다.
    second_resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "동시 요청"})
    thread.join(timeout=15)

    assert results.get("first_status") == 200
    assert second_resp.status_code == 409


def test_cancel_stops_active_stream_and_releases_lock_without_failing_phase(client: TestClient, monkeypatch, caplog):
    """용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소) — POST /cancel이 진행 중인 스트림을
    실제로 멈추고(cancelled 이벤트 수신), 세션 락을 반납해 곧바로 새 요청을 보낼 수 있어야
    한다(409가 나지 않아야 한다). 취소된 요청은 phase="failed"로 이어지면 안 된다."""
    stream_started_event = threading.Event()
    fake = _FakeStreamState(chunk_size=1, delay=0.05, stream_started_event=stream_started_event)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())
    conv_route.configure_ideation_trace(enabled=True, content_max_chars=120, stream_deltas=False)
    caplog.set_level(logging.INFO, logger="ai.meeting.ideation_trace")

    session_id = _start_session(client)
    results = {}

    def stream_call():
        with client.stream(
            "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "답변1"}
        ) as resp:
            results["events"] = _read_ndjson_events(resp)

    thread = threading.Thread(target=stream_call)
    thread.start()
    assert stream_started_event.wait(timeout=5), "취소 전에 실제 LLM 스트림이 시작되지 않았습니다"

    cancel_resp = client.post(f"/ideation-conversation/{session_id}/cancel", json={})
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["cancelled"] is True
    assert cancel_resp.json()["session_locked"] is False  # 락이 실제로 풀렸다.

    thread.join(timeout=15)
    events = results.get("events", [])
    assert any(e.get("type") == "cancelled" for e in events)
    # 취소는 일반 오류가 아니므로 error 이벤트가 나가면 안 된다.
    assert not any(e.get("type") == "error" for e in events)

    # 취소 확인 응답을 받은 뒤에는 곧바로 새 요청을 보내도 409가 나지 않아야 한다(요청:
    # "취소 완료 전에 새 reply를 보내 세션 lock 409가 발생하지 않게"). 취소 시점의 phase는
    # 라운드 중간(expert_discussion)이라 일반 /reply가 아니라 "잠시만" 재개 경로
    # (target_speaker_id 지정)로만 이어갈 수 있다 — 정상적인 사용자 플로우 그대로다.
    fresh = _FakeStreamState(chunk_size=4)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda sid, m: fresh.build())
    with client.stream(
        "POST",
        f"/ideation-conversation/{session_id}/reply/stream",
        json={"message": "이어서 질문", "target_speaker_id": "planning_expert"},
    ) as follow_up:
        assert follow_up.status_code == 200
        follow_up_events = _read_ndjson_events(follow_up)
    assert not any(e.get("type") == "error" for e in follow_up_events)
    follow_up_state = next(e for e in follow_up_events if e.get("type") == "state")
    assert follow_up_state["state"]["phase"] != "failed"
    ordered_events = [
        "IDEATION_CANCEL_REQUESTED",
        "IDEATION_CANCEL_SIGNALLED",
        "IDEATION_STREAM_CLOSE_ATTEMPTED",
        "IDEATION_GRAPH_CANCELLED",
        "IDEATION_SESSION_UNLOCKED",
        "IDEATION_CANCEL_COMPLETED",
        "IDEATION_RESUME_STARTED",
    ]
    positions = [caplog.text.index(event) for event in ordered_events]
    assert positions == sorted(positions)


def _round_transition_discussion_payload(prompt: str) -> dict:
    """용준/Claude(2026-07-22, 요청: "잠시만" 취소 중 phase 오염 수정) — 실제 브라우저에서
    재현된 회귀를 그대로 재현하려면 facilitator가 "다음 라운드로 자동 진행"(continue_round)을
    결정해야 한다. 그러려면 같은 쟁점(mvp_scope)이 발언 캡(MAX_EXPERT_TURNS_PER_ISSUE=6)에
    도달할 때까지 해결되지 않고 반박이 이어져야 한다 — 기본 스텁(_discussion_payload_for)은
    개발 위원이 매번 즉시 issue_resolved=true를 반환해 그 조건을 만들 수 없으므로, 이
    테스트 전용으로 issue_resolved=False + 상대를 계속 지목하는 페이로드를 쓴다."""
    is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
    speaker = "planning_expert" if is_planning else "dev_expert"
    counterpart = "dev_expert" if is_planning else "planning_expert"
    payload = _discussion_payload_for(prompt)
    payload["issue_resolved"] = False
    payload["recommended_next_speaker"] = counterpart
    payload["needs_counterpart_response"] = True
    # 라운드가 여러 번 이어지며 planning_expert도 "response" 단계(직전 발언이 dev_expert)로
    # 여러 번 말하게 된다 — _validate_discussion_response는 그 단계에서 responding_to가
    # 비어 있으면 재시도를 유발하므로(기본 스텁은 dev_expert에만 채워 뒀다), 여기서는 두
    # 화자 모두 항상 채운다.
    payload["responding_to"] = "상대 전문가가 방금 말한 판단"
    payload["concern"] = "우려 지점입니다"
    return payload


class _RoundTransitionCancelStreamState:
    """라운드 1이 쟁점 발언 캡까지 채워져 facilitator가 continue_round를 결정한 "직후"
    (=라운드 2의 첫 기획 위원 발언 스트리밍 도중)에 실제 POST /cancel이 도착할 시간을 벌기
    위해, 정확히 cancel_at_call번째 stream_chat_completion 호출만 한 글자씩 느리게
    내보내고(그 직전까지는 한 번에 통째로 내보내 빠르게 지나간다) reached_target_event를
    그 호출이 시작되는 즉시 set()한다 — 테스트 스레드가 이 이벤트를 기다렸다가 실제
    /cancel HTTP 요청을 보낸다(기존 test_cancel_stops_active_stream_and_releases_lock_
    without_failing_phase와 같은 "진짜 취소 HTTP 요청" 패턴)."""

    def __init__(self, cancel_at_call: int, reached_target_event: threading.Event):
        self.cancel_at_call = cancel_at_call
        self.reached_target_event = reached_target_event
        self.call_counts = {"discussion": 0, "facilitator_summary": 0}

    def build(self):
        def stream_chat_completion(prompt: str):
            if "[의견 규칙]" in prompt:
                self.call_counts["discussion"] += 1
                payload = _round_transition_discussion_payload(prompt)
            elif "[진행자 정리 규칙]" in prompt:
                self.call_counts["facilitator_summary"] += 1
                payload = _facilitator_summary_payload_for(prompt)
            else:
                raise AssertionError(f"스트리밍 대상이 아닌 프롬프트가 왔습니다: {prompt[:100]}")
            call_index = self.call_counts["discussion"] + self.call_counts["facilitator_summary"]
            raw = json.dumps(payload, ensure_ascii=False)
            is_target = call_index == self.cancel_at_call
            if is_target:
                self.reached_target_event.set()
            chunk_size = 1 if is_target else len(raw)
            for i in range(0, len(raw), chunk_size):
                if is_target:
                    time.sleep(0.03)
                yield raw[i : i + chunk_size]

        def call_chat_completion(prompt: str) -> str:
            raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:100]}")

        return stream_chat_completion, call_chat_completion


def test_cancel_during_facilitator_round_transition_then_resume_via_interjection(client: TestClient, monkeypatch):
    """용준/Claude(2026-07-22) — 실제 브라우저에서 보고된 시나리오를 그대로 재현한다:
    "잠시만"이 진행자가 다음 라운드로 넘어가는 경계에서 눌리면, 이전에는 취소된 partial_state
    의 phase가 그래프 내부에서만 의미 있는 신호값("planning_question")으로 저장돼
    reply_to_interjection이 재개를 거부했다. 이제는 phase가 항상 "expert_discussion"으로
    정규화되어 재개가 성공하고, 지정 위원이 먼저 답한 뒤 상대 위원이 검토하며, 세션 락도
    정상 해제되어 후속 요청이 409 없이 처리돼야 한다."""
    session_id = _start_session(client)
    reached_target_event = threading.Event()
    # 라운드 1: 기획/개발이 같은 쟁점(mvp_scope)으로 6회 주고받아야 발언 캡에 도달해
    # facilitator가 continue_round를 결정한다(호출 1~6) + facilitator 정리(호출 7). 그 다음
    # (라운드 2의 첫 기획 위원 발언, 호출 8)이 시작되는 순간을 취소 타이밍으로 삼는다.
    fake = _RoundTransitionCancelStreamState(cancel_at_call=8, reached_target_event=reached_target_event)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda sid, m: fake.build())

    results = {}

    def stream_call():
        with client.stream(
            "POST", f"/ideation-conversation/{session_id}/reply/stream", json={"message": "MVP 범위를 다시 논의해주세요"}
        ) as resp:
            results["status"] = resp.status_code
            results["events"] = _read_ndjson_events(resp)

    thread = threading.Thread(target=stream_call)
    thread.start()
    assert reached_target_event.wait(timeout=10), "라운드 2 진입(8번째 호출)까지 도달하지 못했습니다"

    cancel_resp = client.post(f"/ideation-conversation/{session_id}/cancel", json={})
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["session_locked"] is False  # 취소 완료 전에 새 reply를 보내면 안 됨 — 락이 실제로 풀렸다.

    thread.join(timeout=15)
    assert results.get("status") == 200
    events = results.get("events", [])
    assert any(e.get("type") == "cancelled" for e in events)
    assert not any(e.get("type") == "error" for e in events)

    # 취소 직후 세션을 조회하면(스트림 종료로 워커의 finally가 이미 실행됐다) phase가
    # 절대 "planning_question"(내부 신호값)이나 "failed"가 아니라 재개 가능한 canonical
    # 값("expert_discussion")이어야 한다.
    get_resp = client.get(f"/ideation-conversation/{session_id}")
    assert get_resp.status_code == 200
    cancelled_state = get_resp.json()
    cancelled_phase = cancelled_state["phase"]
    assert cancelled_phase == "expert_discussion"
    assert cancelled_phase not in ("planning_question", "failed")
    messages_before_follow_up = len(cancelled_state["messages"])

    # 세션 락이 실제로 풀렸으므로 곧바로 다음 요청을 보내도 409가 나지 않아야 한다 — 그리고
    # 그 요청은 반드시 reply_to_interjection(target_speaker_id 지정) 경로로만 재개할 수
    # 있어야 한다(요청: "취소 직후 사용자가 지정한 위원이 먼저 답변하고 상대 위원이 검토하는
    # 기존 보장을 유지").
    fresh = _FakeStreamState(chunk_size=4)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda sid, m: fresh.build())
    with client.stream(
        "POST",
        f"/ideation-conversation/{session_id}/reply/stream",
        json={"message": "개발 위원 의견부터 다시 듣고 싶습니다", "target_speaker_id": "dev_expert"},
    ) as follow_up:
        assert follow_up.status_code == 200
        follow_up_events = _read_ndjson_events(follow_up)

    assert not any(e.get("type") == "error" for e in follow_up_events)
    follow_up_state = next(e for e in follow_up_events if e.get("type") == "state")["state"]
    assert follow_up_state["phase"] != "failed"
    # 취소 이전 세션 이력에도(무관하게) message_type="interjection"이 이미 있을 수 있으므로
    # (awaiting_user_decision에서 특정 질문 없이 자유 발언하면 같은 message_type을 쓴다 —
    # 별개의 기존 동작), 이번 reply_to_interjection 호출이 새로 추가한 메시지만 본다.
    new_messages = follow_up_state["messages"][messages_before_follow_up:]
    interjection = next(m for m in new_messages if m["message_type"] == "interjection")
    following = [
        m["speaker_id"] for m in new_messages[new_messages.index(interjection) + 1 :]
        if m["speaker_id"] in ("planning_expert", "dev_expert")
    ]
    assert following[0] == "dev_expert"
    assert "planning_expert" in following[1:]  # 상대 위원이 반드시 뒤이어 검토한다.


def test_cancel_on_session_with_no_active_request_is_idempotent(client: TestClient):
    """활성 요청이 없는 세션에 취소를 보내도(중복 취소 등) 에러 없이 안전하게 처리돼야
    한다."""
    session_id = _start_session(client)
    resp = client.post(f"/ideation-conversation/{session_id}/cancel", json={})
    assert resp.status_code == 200
    assert resp.json() == {"cancelled": True, "session_locked": False}
    # 두 번 연속 취소해도 여전히 안전하다(멱등).
    resp2 = client.post(f"/ideation-conversation/{session_id}/cancel", json={"request_id": "REQ-없음"})
    assert resp2.status_code == 200
    assert resp2.json()["cancelled"] is True


def test_cancel_unknown_session_returns_404(client: TestClient):
    resp = client.post("/ideation-conversation/NOT-A-REAL-SESSION/cancel", json={})
    assert resp.status_code == 404


def test_reply_stream_rejects_invalid_target_speaker_id(client: TestClient):
    session_id = _start_session(client)
    resp = client.post(
        f"/ideation-conversation/{session_id}/reply/stream",
        json={"message": "질문입니다", "target_speaker_id": "누군가"},
    )
    assert resp.status_code == 400


def test_reply_stream_with_target_speaker_id_routes_to_interjection(client: TestClient, monkeypatch):
    """target_speaker_id가 주어지면 reply_to_interjection 경로를 타고, 지정한 위원이 먼저
    응답한 뒤 상대 위원이 검토해야 한다(요청: "지정 위원이 먼저 답변, 다른 위원이 검토")."""
    fake = _FakeStreamState(chunk_size=4)
    monkeypatch.setattr(conv_route, "_build_streaming_backends", lambda session_id, model: fake.build())
    session_id = _start_session(client)

    with client.stream(
        "POST",
        f"/ideation-conversation/{session_id}/reply/stream",
        json={
            "message": "대학생 예비 창업자도 목표 사용자에 포함할 수 있나요?",
            "target_speaker_id": "planning_expert",
            "opinion_target_speaker_id": "dev_expert",
            "interrupted_speaker_id": "dev_expert",
            "active_issue_id": "target_user",
        },
    ) as resp:
        events = _read_ndjson_events(resp)

    state_events = [e for e in events if e.get("type") == "state"]
    assert state_events
    messages = state_events[-1]["state"]["messages"]
    interjection = next(m for m in messages if m["message_type"] == "interjection")
    assert interjection["structured"]["target_speaker_id"] == "planning_expert"
    assert interjection["structured"]["opinion_target_speaker_id"] == "dev_expert"
    assert interjection["structured"]["interrupted_speaker_id"] == "dev_expert"
    following = [
        m["speaker_id"] for m in messages[messages.index(interjection) + 1 :]
        if m["speaker_id"] in ("planning_expert", "dev_expert")
    ]
    assert following, "지정 위원 발언이 이어져야 한다"
    assert following[0] == "planning_expert"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
