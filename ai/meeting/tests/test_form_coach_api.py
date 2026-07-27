import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parents[3] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import app.api.routes.ideation_form_coach as route


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "[역할]" in prompt and "내부 planning_and_development 검토자" in prompt:
            return _Response(
                json.dumps(
                    {
                        "summary": "",
                        "planning": "이전 확정 내용을 기준으로 다음 기획 결정을 좁혀야 합니다.",
                        "development": "초기 구현 범위를 현실적으로 제한해야 합니다.",
                    },
                    ensure_ascii=False,
                )
            )
        has_answer = "민원 분류에 시간이 오래 걸립니다." in prompt
        return _Response(
            json.dumps(
                {
                    "internal_state": {
                        "phase": "target_and_context" if has_answer else "problem_definition",
                        "current_form_fields": ["form_field_1"],
                        "confirmed_summary": {},
                        "decision_reason": "문제를 먼저 좁혀야 합니다.",
                        "draft_patch": [],
                        "confirmed_state_update": (
                            {"problem": "민원 분류에 시간이 오래 걸린다."} if has_answer else {}
                        ),
                        "next_action": "ask_user",
                    },
                    "ui_message": {
                        "facilitator_text": "먼저 해결할 문제를 좁혀볼게요.",
                        "mentor_messages": [],
                        "question": "누가 이 문제를 가장 자주 겪나요?",
                        "choices": [],
                    },
                },
                ensure_ascii=False,
            )
        )


class _OpenAI:
    def __init__(self, **kwargs):
        self.chat = type("_Chat", (), {"completions": _Completions()})()


def test_form_coach_api_start_reply_finalize(monkeypatch):
    monkeypatch.setattr(route, "OpenAI", _OpenAI)
    monkeypatch.setattr(route.settings, "ENABLE_IDEATION_PREVIEW", True)
    monkeypatch.setattr(route.settings, "ENABLE_FORM_COACH_V2", True)
    monkeypatch.setattr(route, "trace_openai_client", lambda client: client)

    app = FastAPI()
    app.include_router(route.router)
    client = TestClient(app)
    start = client.post(
        "/ideation-form-coach/start",
        json={
            "competition_name": "테스트 공모전",
            "selected_idea": {"title": "민원 지원"},
            "application_form_items": [
                {"field_name": "현황 및 문제점", "description": "문제"},
                {"field_name": "사용 대상", "description": "대상"},
            ],
        },
    )

    assert start.status_code == 200
    state = start.json()
    assert state["conversation_flow"] == "form_coach_v2"
    session_id = state["session_id"]

    reply = client.post(
        f"/ideation-form-coach/{session_id}/reply",
        json={"message": "민원 분류에 시간이 오래 걸립니다."},
    )
    assert reply.status_code == 200
    replied = reply.json()
    assert replied["confirmed_state"]["problem"] == "민원 분류에 시간이 오래 걸린다."
    assert replied["application_form_draft"][0]["status"] == "draft"
    assert replied["application_form_draft"][0]["value"] == "민원 분류에 시간이 오래 걸린다."
    assert replied["current_phase"] == "target_and_context"

    edited = client.patch(
        f"/ideation-form-coach/{session_id}/draft/form_field_1",
        json={"value": "사용자가 직접 다듬은 문제 설명"},
    )
    assert edited.status_code == 200
    edited_row = edited.json()["application_form_draft"][0]
    assert edited_row["value"] == "사용자가 직접 다듬은 문제 설명"
    assert edited_row["source"] == "user"
    assert edited_row["editable"] is True
