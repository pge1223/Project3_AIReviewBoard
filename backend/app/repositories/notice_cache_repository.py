from app.db.mongodb import get_db
from app.models.notice_cache import NoticeAnalysisCacheModel


class NoticeAnalysisCacheRepository:

    def get_collection(self):
        db = get_db()
        return db[NoticeAnalysisCacheModel.collection_name]

    async def ensure_indexes(self) -> None:
        collection = self.get_collection()
        await collection.create_index(
            [("cache_key", 1), ("analysis_kind", 1)],
            unique=True
        )

    # 가은/Claude(2026-07-23): analysis_kind를 cache_key와 함께 매치해야 한다 —
    # 미분류(document_type=None) 문서는 announcement/application_form 양쪽에서 다
    # 조회될 수 있어서, cache_key만으로는 어느 분석 결과인지 구분이 안 된다.
    async def find_by_cache_key(self, cache_key: str, analysis_kind: str) -> dict | None:
        collection = self.get_collection()
        doc = await collection.find_one({"cache_key": cache_key, "analysis_kind": analysis_kind})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    # upsert — 같은 (cache_key, analysis_kind)로 여러 프로젝트가 동시에 처음 분석을
    # 마칠 수 있어서(레이스), create 대신 update_one(upsert=True)로 마지막 결과가 이기게 한다.
    async def upsert(self, model: NoticeAnalysisCacheModel) -> None:
        collection = self.get_collection()
        await collection.update_one(
            {"cache_key": model.cache_key, "analysis_kind": model.analysis_kind},
            {"$set": model.to_dict()},
            upsert=True,
        )
