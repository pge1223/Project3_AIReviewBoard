from datetime import datetime, timezone
from typing import Optional


class IdeationConversationSessionModel:
    """대화형 아이디어 회의의 재개 가능한 전체 상태를 저장한다.

    실행 중 락과 취소 이벤트는 프로세스 자원이므로 MongoDB에 저장하지 않고, 그래프 상태와
    세션 범위(RAG/project/user)만 영속화한다.
    """

    collection_name = "ideation_conversation_sessions"

    def __init__(
        self,
        *,
        session_id: str,
        state: dict,
        use_rag: bool,
        project_id: Optional[str],
        user_email: str,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.session_id = session_id
        self.state = state
        self.use_rag = use_rag
        self.project_id = project_id
        self.user_email = user_email
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "state": self.state,
            "phase": self.state.get("phase"),
            "message_count": len(self.state.get("messages") or []),
            "use_rag": self.use_rag,
            "project_id": self.project_id,
            "user_email": self.user_email,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": "1.0",
        }
