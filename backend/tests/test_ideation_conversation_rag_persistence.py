# 작성자: 용준/Claude(2026-07-22)
# 목적: RAG 근거 유실 수정 2탄 회귀 테스트 — /ideation-conversation/start에서 결정된
#       use_rag/project_id가 /reply, /reply/stream에도 그대로 이어져 evidence_lookup이
#       다시 만들어지는지 검증한다.
#
#       실제 버그: ReplyRequest 스키마에 use_rag/project_id가 없고, 세션에도 저장하지
#       않아서 reply_ideation_conversation()/reply_to_interjection()이 evidence_lookup을
#       항상 기본값(None)으로 호출받았다 — /start에서 만든 후보 생성 단계까지만 RAG가
#       실제로 호출되고, 그 이후 모든 회의 턴(전문가 라운드테이블 discussion)은 RAG 검색
#       없이 진행됐다(로그의 elapsed_ms=0, chunk_ids=[] 전부가 이 경로였다).
#
#       실제 RoleAwareRetrievalService/Chroma는 쓰지 않는다 — conv_route._evidence_lookup_for
#       자체를 monkeypatch해 호출 인자(use_rag, project_id)와 호출 횟수만 기록하는 가짜로
#       바꾼다(요청: 외부 LLM/벡터DB에 의존하지 않는 테스트).
# import: fastapi.testclient, pytest; app.api.routes.ideation_conversation_preview 모듈.

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


def _discussion_payload(prompt: str) -> dict:
    is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
    is_dev = not is_planning
    speaker = "planning_expert" if is_planning else "dev_expert"
    return {
        "stance": "보완",
        "spoken_text": f"[{speaker}] 발화 판단",
        "judgment": f"[{speaker}] 판단",
        "reason": f"[{speaker}] 근거",
        "suggestion": f"[{speaker}] 제안",
        "interim_conclusion": f"[{speaker}] 현재 임시 결론",
        "responding_to": "상대 발언" if is_dev else None,
        "agreement": "동의 지점" if is_dev else "",
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


def _facilitator_payload(_prompt: str) -> dict:
    return {
        "agreements": [],
        "disagreements": [],
        "facilitator_summary": "두 전문가가 이번 라운드 의견을 정리했습니다.",
        "spoken_text": "두 위원이 이번 라운드 의견을 정리했습니다.",
        "needs_user_decision": False,
        "user_question": None,
    }


def _stub_llm_call(session_id: str, model: str):
    def llm_call(prompt: str) -> str:
        if "[의견 규칙]" in prompt:
            return json.dumps(_discussion_payload(prompt), ensure_ascii=False)
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(_facilitator_payload(prompt), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")

    return llm_call


class _RecordingEvidenceLookupFactory:
    """conv_route._evidence_lookup_for를 대체한다 — 실제 RAG 대신 호출 인자(use_rag,
    project_id)를 매번 기록하고, evidence_lookup 자체가 실제로 몇 번 호출됐는지도 센다."""

    def __init__(self):
        self.factory_calls: list[tuple[bool, str | None]] = []
        self.lookup_calls: list[tuple[str, str]] = []

    def __call__(self, use_rag: bool, project_id, *, session_id=None, selected_candidate_document_id=None):
        self.factory_calls.append((use_rag, project_id))
        if not use_rag:
            return None

        def lookup(persona_id: str, query: str):
            self.lookup_calls.append((persona_id, query))
            return [
                {
                    "chunk_id": "chk-1",
                    "document_id": "doc-1",
                    "document_name": "공고문.pdf",
                    "page": 1,
                    "text": "근거 원문",
                    "persona_id": persona_id,
                    "role_id": "planning" if persona_id == "planning_expert" else "technology",
                }
            ]

        return lookup


@pytest.fixture(autouse=True)
def _enable_preview_and_stub_llm(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_IDEATION_PREVIEW", True)
    monkeypatch.setattr(settings, "ENABLE_IDEATION_STREAMING", True)
    monkeypatch.setattr(conv_route, "_build_llm_call", _stub_llm_call)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(conv_route.router)
    return TestClient(app)


def test_reply_reuses_use_rag_and_project_id_from_start(client: TestClient, monkeypatch):
    """핵심 회귀 테스트: /start에서 use_rag=True, project_id="proj-1"로 시작하면, 뒤이은
    /reply도 같은 use_rag/project_id로 evidence_lookup을 다시 만들어야 한다 — 예전에는
    reply_ideation_conversation()이 evidence_lookup을 아예 받지 않아 이 두 번째 호출
    자체가 없었다(factory_calls에 한 번만 기록됐다)."""
    factory = _RecordingEvidenceLookupFactory()
    monkeypatch.setattr(conv_route, "_evidence_lookup_for", factory)

    start_resp = client.post(
        "/ideation-conversation/start",
        json={
            "competition_name": "데모 공모전",
            "user_idea": "소상공인이 손님 문의에 자동으로 답하는 챗봇",
            "use_rag": True,
            "project_id": "proj-1",
        },
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    # /start 시점에 evidence_lookup_for(True, "proj-1")가 호출됐고, 후보 없는 refinement
    # 모드라 /start 안에서 이미 1라운드 discussion(기획->개발)이 실제로 evidence_lookup을
    # 호출했어야 한다.
    assert (True, "proj-1") in factory.factory_calls
    calls_after_start = len(factory.factory_calls)
    lookups_after_start = len(factory.lookup_calls)
    assert lookups_after_start > 0, "/start 시점의 discussion 라운드가 evidence_lookup을 호출하지 않았습니다"

    reply_resp = client.post(
        f"/ideation-conversation/{session_id}/reply",
        json={"message": "타깃은 동네 카페 사장님입니다"},
    )
    assert reply_resp.status_code == 200

    # 버그 수정 전에는 /reply가 evidence_lookup_for를 아예 다시 호출하지 않았다(기본값
    # evidence_lookup=None으로 그래프가 진행됐다) — 수정 후에는 반드시 한 번 더 호출되고,
    # 그때도 같은 (True, "proj-1")이어야 한다.
    assert len(factory.factory_calls) > calls_after_start, "reply가 evidence_lookup을 다시 만들지 않았습니다"
    assert factory.factory_calls[-1] == (True, "proj-1")

    # evidence_lookup_for가 다시 "호출된 것"만으로는 부족하다 — 그 결과물(lookup 콜러블)이
    # 실제로 reply_ideation_conversation에 전달돼 discussion 노드에서 호출됐는지까지
    # 확인한다(버그 재현: 이 함수가 evidence_lookup 인자를 넘기지 않으면 이 값은 그대로다).
    assert len(factory.lookup_calls) > lookups_after_start, (
        "/reply 도중 discussion 노드가 evidence_lookup을 실제로 호출하지 않았습니다"
        "(evidence_lookup_for는 호출됐지만 그 결과가 reply_ideation_conversation에 전달되지 않았을 수 있습니다)"
    )
    called_personas = {persona_id for persona_id, _query in factory.lookup_calls[lookups_after_start:]}
    # 현재 회의 라우팅은 한 HTTP 요청에서 다음 위원 1명까지만 진행될 수 있다. 이 테스트의
    # 목적은 reply에서도 저장된 RAG 범위를 복원해 lookup을 실제 호출하는지 확인하는 것이므로
    # 해당 요청에서 반드시 두 위원이 모두 발언해야 한다고 제한하지 않는다.
    assert "planning_expert" in called_personas


def test_reply_stream_also_reuses_use_rag_and_project_id(client: TestClient, monkeypatch):
    """/reply/stream 경로(reply_to_interjection 포함)도 동일하게 evidence_lookup을
    다시 만들어야 한다 — 스트리밍 워커가 그래프를 호출하는 지점이 sync /reply와
    별도이므로 따로 검증한다."""
    factory = _RecordingEvidenceLookupFactory()
    monkeypatch.setattr(conv_route, "_evidence_lookup_for", factory)
    monkeypatch.setattr(
        conv_route,
        "_build_streaming_backends",
        lambda session_id, model: (
            lambda prompt: iter(
                [
                    json.dumps(_discussion_payload(prompt), ensure_ascii=False)
                    if "[의견 규칙]" in prompt
                    else json.dumps(_facilitator_payload(prompt), ensure_ascii=False)
                ]
            ),
            lambda prompt: (_ for _ in ()).throw(AssertionError("스트리밍 전용 경로에서 call_chat_completion 호출됨")),
        ),
    )

    start_resp = client.post(
        "/ideation-conversation/start",
        json={
            "competition_name": "데모 공모전",
            "user_idea": "소상공인이 손님 문의에 자동으로 답하는 챗봇",
            "use_rag": True,
            "project_id": "proj-stream-1",
        },
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]
    calls_after_start = len(factory.factory_calls)

    with client.stream(
        "POST",
        f"/ideation-conversation/{session_id}/reply/stream",
        json={"message": "타깃은 동네 카페 사장님입니다"},
    ) as resp:
        assert resp.status_code == 200
        for _ in resp.iter_lines():
            pass

    assert len(factory.factory_calls) > calls_after_start
    assert factory.factory_calls[-1] == (True, "proj-stream-1")
    assert factory.lookup_calls, "스트리밍 경로에서 evidence_lookup이 실제로 호출되지 않았습니다"


def test_reply_without_rag_does_not_configure_evidence_lookup(client: TestClient, monkeypatch):
    """use_rag=False로 시작한 세션은 /reply에서도 계속 RAG 없이 진행돼야 한다(정책 변경
    금지 — use_rag=False가 /reply 도중에 몰래 True로 바뀌면 안 된다)."""
    factory = _RecordingEvidenceLookupFactory()
    monkeypatch.setattr(conv_route, "_evidence_lookup_for", factory)

    start_resp = client.post(
        "/ideation-conversation/start",
        json={
            "competition_name": "데모 공모전",
            "user_idea": "소상공인이 손님 문의에 자동으로 답하는 챗봇",
        },
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    reply_resp = client.post(
        f"/ideation-conversation/{session_id}/reply",
        json={"message": "타깃은 동네 카페 사장님입니다"},
    )
    assert reply_resp.status_code == 200

    assert all(call == (False, None) for call in factory.factory_calls)
    assert not factory.lookup_calls
