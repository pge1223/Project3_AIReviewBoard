"""
Unit Tests for ai.rag.external_research.search_service (RAG-007)
=======================================================================
실제 chromadb PersistentClient(tmp_path) + FakeEvidenceEmbedder + mock provider만
사용한다. ai.meeting.graph, LangGraph, 실제 LLM/외부 네트워크를 호출하지 않는다.
"""

from datetime import date
from pathlib import Path

import pytest

from ai.rag.external_research.config import DEFAULT_COLLECTION_NAME, ExternalResearchConfig
from ai.rag.external_research.exceptions import ExternalProviderUnavailableError
from ai.rag.external_research.providers.base import ExternalEvidenceCandidate
from ai.rag.external_research.providers.dataset_provider import DatasetProvider
from ai.rag.external_research.providers.public_api_provider import PublicApiProvider
from ai.rag.external_research.query_builder import build_external_research_query
from ai.rag.external_research.repository import ExternalEvidenceRepository
from ai.rag.external_research.schemas import ExternalEvidenceDocument, ExternalEvidenceType, ExternalResearchRequest
from ai.rag.external_research.search_service import ExternalResearchService
from ai.rag.retrieval.chroma_store import create_persistent_client
from ai.rag.tests._external_research_fixtures import FakeEvidenceEmbedder

_DIM = 4
_UNIT_A = [1.0, 0.0, 0.0, 0.0]
_UNIT_B = [0.0, 1.0, 0.0, 0.0]


def _doc(document_id="DOC-001", chunk_id="CHUNK-001", domain="public_service", content="본문 내용입니다.", **overrides):
    base = dict(
        source_id="SRC-1",
        document_id=document_id,
        chunk_id=chunk_id,
        title="제목",
        evidence_type=ExternalEvidenceType.STATISTICS,
        publisher="통계청",
        source_url="https://example.org/data",
        domain=domain,
        evaluation_criteria=["시장성"],
        supported_roles=["marketing"],
        content=content,
        reference_date="2025-12-31",
    )
    base.update(overrides)
    return ExternalEvidenceDocument(**base)


def _request(**overrides) -> ExternalResearchRequest:
    base = dict(domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing", top_k=5)
    base.update(overrides)
    return ExternalResearchRequest(**base)


@pytest.fixture
def repository(tmp_path):
    client = create_persistent_client(path=str(tmp_path / "chroma_data"))
    return ExternalEvidenceRepository(
        client=client,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model="fake-external-research-embedder",
        embedding_dimension=_DIM,
        embedding_version="embedding_v1",
    )


class TestSearchEndToEnd:
    def test_top_k_applied(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A})
        for i in range(3):
            content = f"자료 {i}"
            embedder._overrides[content] = _UNIT_A
            repository.upsert_evidence_chunk(_doc(document_id=f"DOC-{i}", content=content), _UNIT_A)

        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)

        response = service.search(_request(top_k=2))
        assert len(response.results) == 2
        assert response.total_results == 2

    def test_min_score_applied(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={
            query_text: _UNIT_A, "close content": _UNIT_A, "far content": _UNIT_B,
        })
        repository.upsert_evidence_chunk(_doc(document_id="d-close", content="close content"), _UNIT_A)
        repository.upsert_evidence_chunk(_doc(document_id="d-far", content="far content"), _UNIT_B)

        dataset_provider = DatasetProvider(repository, embedder, config=ExternalResearchConfig(min_similarity_score=0.5))
        service = ExternalResearchService(dataset_provider, config=ExternalResearchConfig(min_similarity_score=0.5))

        response = service.search(_request())
        assert len(response.results) == 1
        assert response.results[0].document_id == "d-close"

    def test_domain_filter_applied(self, repository):
        query_text = build_external_research_query(
            domain="finance", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={
            query_text: _UNIT_A, "finance content": _UNIT_A, "public content": _UNIT_A,
        })
        repository.upsert_evidence_chunk(_doc(document_id="d-fin", domain="finance", content="finance content"), _UNIT_A)
        repository.upsert_evidence_chunk(_doc(document_id="d-pub", domain="public_service", content="public content"), _UNIT_A)

        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)

        response = service.search(_request(domain="finance"))
        assert len(response.results) == 1
        assert response.results[0].document_id == "d-fin"

    def test_role_score_reflected_in_final_ranking(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={
            query_text: _UNIT_A, "matches role": _UNIT_A, "wrong role": _UNIT_A,
        })
        repository.upsert_evidence_chunk(
            _doc(document_id="d-match", content="matches role", supported_roles=["marketing"]), _UNIT_A
        )
        repository.upsert_evidence_chunk(
            _doc(document_id="d-wrong", content="wrong role", supported_roles=["finance"]), _UNIT_A
        )

        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)

        response = service.search(_request(reviewer_role="marketing"))
        scores_by_doc = {r.document_id: r.role_score for r in response.results}
        assert scores_by_doc["d-match"] == 1.0
        assert scores_by_doc["d-wrong"] == 0.0
        assert response.results[0].document_id == "d-match"

    def test_criteria_score_reflected(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A, "content": _UNIT_A})
        repository.upsert_evidence_chunk(
            _doc(content="content", evaluation_criteria=["시장성"]), _UNIT_A
        )

        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)
        response = service.search(_request(evaluation_criteria=["시장성"]))

        assert response.results[0].criteria_score == 1.0
        assert "시장성" in response.results[0].matched_criteria

    def test_duplicate_chunks_deduplicated(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A, "content": _UNIT_A})
        repository.upsert_evidence_chunk(_doc(content="content"), _UNIT_A)
        # 같은 (document_id, chunk_id)로 재색인 -> upsert로 덮어써짐(중복 아님)
        repository.upsert_evidence_chunk(_doc(content="content"), _UNIT_A)

        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)
        response = service.search(_request())
        assert len(response.results) == 1

    def test_same_source_multiple_chunks_aggregated_to_one_result(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={
            query_text: _UNIT_A, "chunk one": _UNIT_A, "chunk two": _UNIT_A,
        })
        repository.upsert_evidence_chunk(_doc(chunk_id="c1", content="chunk one"), _UNIT_A)
        repository.upsert_evidence_chunk(_doc(chunk_id="c2", content="chunk two"), _UNIT_A)

        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)
        response = service.search(_request())

        # 같은 (source_id, document_id)의 두 청크가 하나의 결과로 집계돼야 한다.
        assert len(response.results) == 1

    def test_no_results_returns_empty_list_not_error(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A})
        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)

        response = service.search(_request())
        assert response.results == []
        assert response.total_results == 0
        assert any("찾지 못했습니다" in w for w in response.warnings)

    def test_no_fabricated_evidence_results_match_indexed_data(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A, "real content": _UNIT_A})
        indexed = _doc(content="real content")
        repository.upsert_evidence_chunk(indexed, _UNIT_A)

        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)
        response = service.search(_request())

        assert len(response.results) == 1
        result = response.results[0]
        assert result.source_url == indexed.source_url
        assert result.publisher == indexed.publisher
        assert result.quote == indexed.content

    def test_reference_only_always_true(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A, "content": _UNIT_A})
        repository.upsert_evidence_chunk(_doc(content="content"), _UNIT_A)

        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)
        response = service.search(_request())

        assert response.reference_only is True
        assert response.results[0].reference_only is True

    def test_results_include_publisher_source_and_date(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A, "content": _UNIT_A})
        repository.upsert_evidence_chunk(_doc(content="content"), _UNIT_A)

        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)
        response = service.search(_request())

        result = response.results[0]
        assert result.publisher == "통계청"
        assert result.source_url == "https://example.org/data"
        assert result.reference_date == "2025-12-31"
        assert result.date_status.value in {"current", "aging", "stale", "unknown"}


class TestPublicApiFailureIsolation:
    def test_public_api_receives_compact_live_query_not_labeled_rag_query(self, repository):
        embedder = FakeEvidenceEmbedder(dimension=_DIM)
        dataset_provider = DatasetProvider(repository, embedder)
        received_queries = []

        def fake_fetch(request, query_text):
            received_queries.append(query_text)
            return []

        config = ExternalResearchConfig(
            enable_dataset_search=False,
            enable_public_api_search=True,
        )
        service = ExternalResearchService(
            dataset_provider,
            public_api_provider=PublicApiProvider(fetch=fake_fetch, enabled=True),
            config=config,
        )

        service.search(
            _request(
                domain="competition",
                reviewer_role="technology",
                query_context="공공기관 AI 민원 서비스",
            )
        )

        assert received_queries == ["공공기관 AI 민원 서비스 최신 기술 동향 도입 사례"]
        assert "위원 역할:" not in received_queries[0]
        assert "평가 기준:" not in received_queries[0]

    def test_public_api_result_without_similarity_score_is_not_discarded(self, repository):
        embedder = FakeEvidenceEmbedder(dimension=_DIM)
        dataset_provider = DatasetProvider(repository, embedder)

        def fake_fetch(request, qt):
            return [{
                "source_id": "NAVER-1", "document_id": "NAVER-1", "chunk_id": "NAVER-1-01",
                "title": "실시간 뉴스", "evidence_type": "news", "publisher": "example.com",
                "source_url": "https://example.com/news/1", "domain": "competition",
                "content": "AI 공모전 관련 실시간 뉴스", "published_at": "2026-07-27",
                "semantic_score": None,
            }]

        public_api_provider = PublicApiProvider(fetch=fake_fetch, enabled=True)
        config = ExternalResearchConfig(
            enable_dataset_search=False,
            enable_public_api_search=True,
            min_similarity_score=0.45,
        )
        service = ExternalResearchService(
            dataset_provider,
            public_api_provider=public_api_provider,
            config=config,
        )

        response = service.search(_request(domain="competition"))
        assert len(response.results) == 1
        assert response.results[0].evidence_type == ExternalEvidenceType.NEWS
        assert response.results[0].semantic_score == 0.0

    def test_news_uses_separate_freshness_weight(self, repository):
        embedder = FakeEvidenceEmbedder(dimension=_DIM)
        dataset_provider = DatasetProvider(repository, embedder)

        def fake_fetch(request, qt):
            return [{
                "source_id": "NAVER-FRESH", "document_id": "NAVER-FRESH", "chunk_id": "NAVER-FRESH-01",
                "title": "오늘의 기술 뉴스", "evidence_type": "news", "publisher": "example.com",
                "source_url": "https://example.com/news/fresh", "domain": "competition",
                "content": "오늘 발표된 AI 기술 동향", "published_at": date.today().isoformat(),
                "semantic_score": None,
            }]

        config = ExternalResearchConfig(
            enable_dataset_search=False,
            enable_public_api_search=True,
            news_freshness_weight=0.30,
        )
        service = ExternalResearchService(
            dataset_provider,
            public_api_provider=PublicApiProvider(fetch=fake_fetch, enabled=True),
            config=config,
        )

        result = service.search(_request(domain="competition")).results[0]

        # semantic=0, role/criteria=0.5이고 뉴스 최신성=1.0이다. 비최신성 가중치
        # 0.55/0.20/0.15는 남은 0.70 안에서 같은 비율로 재조정된다.
        expected = 0.5 * (0.20 * 0.70 / 0.90) + 0.5 * (0.15 * 0.70 / 0.90) + 0.30
        assert result.final_score == pytest.approx(expected)
        assert result.freshness_score == 1.0

    def test_public_api_timeout_keeps_dataset_results(self, repository):
        import time

        from ai.rag.external_research.config import PublicApiProviderConfig

        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A, "content": _UNIT_A})
        repository.upsert_evidence_chunk(_doc(content="content"), _UNIT_A)
        dataset_provider = DatasetProvider(repository, embedder)

        def slow_fetch(request, qt):
            time.sleep(0.3)
            return []

        public_api_provider = PublicApiProvider(
            fetch=slow_fetch, enabled=True, config=PublicApiProviderConfig(timeout_seconds=0.02)
        )
        config = ExternalResearchConfig(enable_public_api_search=True)
        service = ExternalResearchService(dataset_provider, public_api_provider=public_api_provider, config=config)

        response = service.search(_request())
        assert len(response.results) == 1
        assert any("실시간 공공데이터 검색을 사용할 수 없어" in w for w in response.warnings)

    def test_public_api_unavailable_keeps_dataset_results(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A, "content": _UNIT_A})
        repository.upsert_evidence_chunk(_doc(content="content"), _UNIT_A)
        dataset_provider = DatasetProvider(repository, embedder)

        public_api_provider = PublicApiProvider(enabled=True)  # fetch 없음 -> ExternalProviderUnavailableError
        config = ExternalResearchConfig(enable_public_api_search=True)
        service = ExternalResearchService(dataset_provider, public_api_provider=public_api_provider, config=config)

        response = service.search(_request())
        assert len(response.results) == 1

    def test_public_api_disabled_by_default(self, repository):
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A, "content": _UNIT_A})
        repository.upsert_evidence_chunk(_doc(content="content"), _UNIT_A)
        dataset_provider = DatasetProvider(repository, embedder)

        service = ExternalResearchService(dataset_provider)  # config 기본값: enable_public_api_search=False
        response = service.search(_request())
        assert response.used_public_api_search is False

    def test_source_url_never_generated_by_service(self, repository):
        """public API 결과가 존재해도 source_url은 provider가 전달한 원본 값 그대로다."""
        query_text = build_external_research_query(
            domain="public_service", evaluation_criteria=["시장성"], reviewer_role="marketing"
        )
        embedder = FakeEvidenceEmbedder(dimension=_DIM, overrides={query_text: _UNIT_A})
        dataset_provider = DatasetProvider(repository, embedder)

        def fake_fetch(request, qt):
            return [{
                "source_id": "API-1", "document_id": "DOC-API-1", "chunk_id": "CHUNK-1",
                "title": "실시간 자료", "evidence_type": "market", "publisher": "산업통상자원부",
                "source_url": "https://real.gov.example/data", "domain": "public_service",
                "content": "실시간 시장 자료", "semantic_score": 0.9,
            }]

        public_api_provider = PublicApiProvider(fetch=fake_fetch, enabled=True)
        config = ExternalResearchConfig(enable_public_api_search=True, min_similarity_score=0.0)
        service = ExternalResearchService(dataset_provider, public_api_provider=public_api_provider, config=config)

        response = service.search(_request())
        assert len(response.results) == 1
        assert response.results[0].source_url == "https://real.gov.example/data"
        assert response.results[0].retrieval_source == "public_api"


class TestSourcelessResultsExcluded:
    def test_unverified_candidate_excluded_with_warning(self):
        unverified = ExternalEvidenceCandidate(
            source_id="s", document_id="d1", chunk_id="c1", title="t",
            evidence_type=ExternalEvidenceType.MARKET, publisher="", source_url="",
            domain="public_service", content="내용", semantic_score=0.9, verified_source=False,
            retrieval_source="dataset",
        )

        class _StubProvider:
            name = "dataset"

            def search(self, request, query_text):
                return [unverified]

        service = ExternalResearchService(_StubProvider())
        response = service.search(_request())
        assert response.results == []
        assert any("출처 검증에 실패" in w for w in response.warnings)


class TestIndependentExecution:
    def test_package_source_does_not_import_meeting_graph_or_langgraph(self):
        package_dir = Path(__file__).parent.parent / "external_research"
        forbidden_prefixes = ("import ai.meeting", "from ai.meeting", "import langgraph", "from langgraph")
        offenders = []
        for py_file in package_dir.rglob("*.py"):
            for line in py_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(forbidden_prefixes):
                    offenders.append((str(py_file.relative_to(package_dir)), stripped))
        assert offenders == []

    def test_package_source_does_not_import_evidence_sufficiency(self):
        """RAG-005(근거 충족도)와의 분리를 정적으로 재확인한다."""
        package_dir = Path(__file__).parent.parent / "external_research"
        offenders = []
        for py_file in package_dir.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if "evidence_sufficiency" in text:
                offenders.append(str(py_file.relative_to(package_dir)))
        assert offenders == []

    def test_service_constructible_without_public_api(self, repository):
        embedder = FakeEvidenceEmbedder(dimension=_DIM)
        dataset_provider = DatasetProvider(repository, embedder)
        service = ExternalResearchService(dataset_provider)
        assert service is not None

    def test_service_works_with_no_providers_at_all(self):
        service = ExternalResearchService()
        response = service.search(_request())
        assert response.results == []
        assert response.used_dataset_search is False
        assert response.used_public_api_search is False
