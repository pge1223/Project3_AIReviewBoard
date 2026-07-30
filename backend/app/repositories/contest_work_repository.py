from app.db.mongodb import get_db


# 가은/Claude(2026-07-21): kyh님이 크롤링(crawl_sotong_winners.py)해서 채운 소통혁신24
# 수상작 아카이브 조회 전용 — 이 앱이 직접 쓰는 컬렉션이 아니라 읽기 전용으로만 접근한다.
# category/source_org는 scripts/classify_contest_works.py가 나중에 채워 넣는 필드라
# 아직 분류 안 된 문서는 category가 없을 수 있다(그런 문서는 자동으로 매칭에서 빠진다).
class ContestWorkRepository:
    COLLECTION_NAME = "contest_works"

    def get_collection(self):
        db = get_db()
        return db[self.COLLECTION_NAME]

    # 실측 제보(가은, 2026-07-21): find_by_category가 그냥 category로만 걸러 앞에서부터
    # limit개를 가져오다 보니, 공모전 한 건이 부문별로 수상작을 여러 개 내면(예: 산림청
    # AI 활용 경진대회 하나가 "AI/데이터"에 8건) "유사사례 경향" 카드 4개가 전부 같은
    # 공모전으로 채워지는 문제가 있었다. contest_title 기준으로 최대 1건씩만 뽑아 서로
    # 다른 공모전이 섞여 나오게 한다.
    async def find_by_category(self, category: str, limit: int = 4) -> list[dict]:
        winners = await self._pick_diverse_by_contest(category, "winner", limit)
        remaining = limit - len(winners)
        candidates = []
        if remaining > 0:
            candidates = await self._pick_diverse_by_contest(category, "candidate", remaining)
        return winners + candidates

    async def find_rag_candidates(self, category: str, limit: int = 24) -> list[dict]:
        """사례 전용 벡터 컬렉션을 처음 채울 공개 사례 후보를 반환한다.

        출처 URL이 있고 제목 또는 OCR 본문이 있는 자료만 사용한다. 같은 공모전의
        항목만 과도하게 색인되지 않도록 contest_title 기준으로 분산한다.
        """
        collection = self.get_collection()
        query = {
            "category": category,
            "source_url": {"$nin": [None, ""]},
            "$or": [
                {"ocr_text": {"$nin": [None, ""]}},
                {"work_title": {"$nin": [None, ""]}},
                {"contest_title": {"$nin": [None, ""]}},
            ],
        }
        cursor = collection.find(query).sort("selection_status", -1)
        results: list[dict] = []
        seen_contests: set[str] = set()
        async for doc in cursor:
            contest_title = str(doc.get("contest_title") or "").strip()
            if contest_title and contest_title in seen_contests:
                continue
            if contest_title:
                seen_contests.add(contest_title)
            results.append(doc)
            if len(results) >= limit:
                break
        return results

    async def _pick_diverse_by_contest(self, category: str, selection_status: str, count: int) -> list[dict]:
        collection = self.get_collection()
        seen_titles = set()
        picked: list[dict] = []
        cursor = collection.find({"category": category, "selection_status": selection_status})
        async for doc in cursor:
            title = doc.get("contest_title")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            picked.append(doc)
            if len(picked) >= count:
                break
        return picked

    # 가은/Claude(2026-07-21): "이 공모전에서 어떤 아이디어가 수상했는지, 어떤 게 후보에
    # 그쳤는지" 상세 패널용 — 같은 contest_title을 가진 works 전부(수상작+후보작)를
    # 반환한다. 수상작을 먼저 보여준다.
    async def find_by_contest_title(self, contest_title: str) -> list[dict]:
        collection = self.get_collection()
        cursor = collection.find({"contest_title": contest_title}).sort("selection_status", -1)
        results = []
        async for doc in cursor:
            results.append(doc)
        return results
