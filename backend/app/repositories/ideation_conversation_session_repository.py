from datetime import datetime, timezone

from app.db.mongodb import get_db
from app.models.ideation_conversation_session import IdeationConversationSessionModel


class IdeationConversationSessionRepository:
    def get_collection(self):
        db = get_db()
        if db is None:
            raise RuntimeError("MongoDB가 연결되지 않았습니다.")
        return db[IdeationConversationSessionModel.collection_name]

    async def ensure_indexes(self) -> None:
        collection = self.get_collection()
        await collection.create_index("session_id", unique=True)
        await collection.create_index(
            [("project_id", 1), ("user_email", 1), ("updated_at", -1)]
        )

    async def upsert(
        self,
        *,
        session_id: str,
        state: dict,
        use_rag: bool,
        project_id: str | None,
        user_email: str,
    ) -> None:
        collection = self.get_collection()
        now = datetime.now(timezone.utc)
        await collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "state": state,
                    "phase": state.get("phase"),
                    "message_count": len(state.get("messages") or []),
                    "use_rag": use_rag,
                    "project_id": project_id,
                    "user_email": user_email,
                    "updated_at": now,
                    "schema_version": "1.0",
                },
                "$setOnInsert": {
                    "session_id": session_id,
                    "created_at": now,
                },
            },
            upsert=True,
        )

    async def find_by_session_id(self, session_id: str) -> dict | None:
        return await self.get_collection().find_one({"session_id": session_id})

    async def find_latest_by_project(self, project_id: str, user_email: str) -> dict | None:
        return await self.get_collection().find_one(
            {"project_id": project_id, "user_email": user_email},
            sort=[("updated_at", -1)],
        )

    async def delete_by_project_id(self, project_id: str) -> int:
        result = await self.get_collection().delete_many({"project_id": project_id})
        return result.deleted_count
