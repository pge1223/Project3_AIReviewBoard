import asyncio
from types import SimpleNamespace

from ai.rag.similar_cases import SimilarCaseType
from app.api.routes import documents as documents_route


class _FakeSearchService:
    def __init__(self, results):
        self.results = results
        self.requests = []

    def search(self, request):
        self.requests.append(request)
        return SimpleNamespace(results=self.results)


def test_find_similar_works_prefers_kure_rag_results(monkeypatch):
    result = SimpleNamespace(
        title="디지털 민원 개선 사례",
        source_name="행정안전부",
        source_url="https://example.com/case",
        similarity_score=0.83,
        case_type=SimilarCaseType.AWARD_WINNER,
        matched_criteria=["혁신성"],
    )
    search_service = _FakeSearchService([result])
    monkeypatch.setattr(
        documents_route,
        "_get_similar_case_services",
        lambda: (SimpleNamespace(), SimpleNamespace(), search_service),
    )

    works = asyncio.run(
        documents_route._find_similar_works(
            "AI/데이터",
            document_summary="공공기관 민원 서비스를 AI로 개선",
            evaluation_criteria=["혁신성"],
            trace_id="project-1",
        )
    )

    assert len(works) == 1
    assert works[0].retrieval_method == "rag"
    assert works[0].similarity_score == 0.83
    assert works[0].source_url == "https://example.com/case"
    assert search_service.requests[0].document_summary == "공공기관 민원 서비스를 AI로 개선"


def test_find_similar_works_falls_back_to_category_when_rag_fails(monkeypatch):
    monkeypatch.setattr(
        documents_route,
        "_get_similar_case_services",
        lambda: (_ for _ in ()).throw(RuntimeError("chroma unavailable")),
    )

    async def find_by_category(category):
        assert category == "공공행정"
        return [
            {
                "work_title": "민원 안내 개선",
                "contest_title": "공공서비스 경진대회",
                "source_org": "행정안전부",
                "selection_status": "winner",
            }
        ]

    monkeypatch.setattr(documents_route.contest_work_repo, "find_by_category", find_by_category)

    works = asyncio.run(
        documents_route._find_similar_works(
            "공공행정",
            document_summary="민원 서비스 개선",
            evaluation_criteria=[],
            trace_id="project-2",
        )
    )

    assert len(works) == 1
    assert works[0].retrieval_method == "category_fallback"
    assert works[0].contest_title == "공공서비스 경진대회"
