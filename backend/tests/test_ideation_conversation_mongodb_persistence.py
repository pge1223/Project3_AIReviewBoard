import asyncio
import copy
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


class _FakeSessionRepository:
    def __init__(self):
        self.docs: dict[str, dict] = {}

    async def upsert(self, *, session_id, state, use_rag, project_id, user_email):
        previous = self.docs.get(session_id, {})
        self.docs[session_id] = {
            **previous,
            "session_id": session_id,
            "state": copy.deepcopy(state),
            "phase": state.get("phase"),
            "message_count": len(state.get("messages") or []),
            "use_rag": use_rag,
            "project_id": project_id,
            "user_email": user_email,
            "updated_at": len(self.docs) + 1,
        }

    async def find_by_session_id(self, session_id):
        doc = self.docs.get(session_id)
        return copy.deepcopy(doc) if doc else None

    async def find_latest_by_project(self, project_id, user_email):
        matches = [
            doc
            for doc in self.docs.values()
            if doc.get("project_id") == project_id and doc.get("user_email") == user_email
        ]
        return copy.deepcopy(max(matches, key=lambda doc: doc["updated_at"])) if matches else None


def _state(session_id: str, content: str = "사용자 의견") -> dict:
    return {
        "session_id": session_id,
        "phase": "expert_discussion",
        "round": 1,
        "max_rounds": 3,
        "notice_and_criteria": {"competition_name": "영속화 공모전"},
        "messages": [
            {
                "message_id": "MSG-user-1",
                "speaker_id": "user",
                "speaker_name": "사용자",
                "content": content,
            }
        ],
        "consensus": [],
        "unresolved_issues": [],
    }


@pytest.fixture
def persistence(monkeypatch):
    repository = _FakeSessionRepository()
    monkeypatch.setattr(conv_route, "_session_repo", repository)
    monkeypatch.setattr(settings, "ENABLE_IDEATION_PREVIEW", True)
    with conv_route._store._lock:
        conv_route._store._sessions.clear()
    yield repository
    with conv_route._store._lock:
        conv_route._store._sessions.clear()


def test_session_restores_full_state_and_rag_scope_after_memory_reset(persistence):
    state = _state("IDEA-CONV-persist1")
    conv_route._store.create(
        state,
        use_rag=True,
        project_id="project-1",
        user_email="guest@local",
    )
    record = conv_route._store.get_record(state["session_id"])

    assert asyncio.run(conv_route._persist_session_record(record)) is True
    with conv_route._store._lock:
        conv_route._store._sessions.clear()

    restored = asyncio.run(conv_route._restore_session_record(state["session_id"], "guest@local"))

    assert restored is not None
    assert restored.state["messages"][0]["content"] == "사용자 의견"
    assert restored.use_rag is True
    assert restored.project_id == "project-1"
    assert restored.user_email == "guest@local"


def test_other_user_cannot_restore_session(persistence):
    state = _state("IDEA-CONV-private")
    persistence.docs[state["session_id"]] = {
        "session_id": state["session_id"],
        "state": state,
        "use_rag": True,
        "project_id": "project-private",
        "user_email": "owner@example.com",
        "updated_at": 1,
    }

    restored = asyncio.run(
        conv_route._restore_session_record(state["session_id"], "other@example.com")
    )

    assert restored is None
    assert conv_route._store.get_record(state["session_id"]) is None


def test_latest_project_conversation_is_available_without_session_storage(persistence):
    old_state = _state("IDEA-CONV-old", "이전 의견")
    new_state = _state("IDEA-CONV-new", "최근 의견")
    asyncio.run(
        persistence.upsert(
            session_id=old_state["session_id"],
            state=old_state,
            use_rag=True,
            project_id="project-latest",
            user_email="guest@local",
        )
    )
    asyncio.run(
        persistence.upsert(
            session_id=new_state["session_id"],
            state=new_state,
            use_rag=True,
            project_id="project-latest",
            user_email="guest@local",
        )
    )

    app = FastAPI()
    app.include_router(conv_route.router)
    client = TestClient(app)
    response = client.get(
        "/ideation-conversation/project/project-latest/latest"
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "IDEA-CONV-new"
    assert response.json()["messages"][0]["content"] == "최근 의견"
