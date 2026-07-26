"""
Unit Tests for ai.rag.orchestration.ideation_external_evidence_service (RAG-007 wiring)
==============================================================================================
실제 chromadb PersistentClient(tmp_path) + FakeEvidenceEmbedder만 사용한다(KURE-v1/LangGraph/
실제 LLM/외부 네트워크 미사용). candidate_planning/candidate_feasibility 노드 통합은
ai/meeting/tests/test_ideation_discovery_graph.py 쪽에서 별도로 검증한다(ai/meeting은
ai.rag를 import하지 않아야 하므로 이 파일에서는 orchestration 계층만 테스트한다).
"""

from ai.rag.external_research.config import ExternalResearchConfig
from ai.rag.external_research.providers.dataset_provider import DatasetProvider
from ai.rag.external_research.repository import ExternalEvidenceRepository
from ai.rag.external_research.schemas import ExternalEvidenceDocument, ExternalEvidenceType
from ai.rag.external_research.search_service import ExternalResearchService
from ai.rag.orchestration.ideation_external_evidence_service import (
    resolve_external_reviewer_role,
    search_ideation_external_evidence,
    make_ideation_external_evidence_lookup,
)
from ai.rag.retrieval.chroma_store import create_persistent_client
from ai.rag.tests._external_research_fixtures import FakeEvidenceEmbedder

_DIM = 4


def _doc(document_id, chunk_id, *, content, reference_date=None, published_at=None, supported_roles=None, **overrides):
    base = dict(
        source_id="SRC-1",
        document_id=document_id,
        chunk_id=chunk_id,
        title="제목",
        evidence_type=ExternalEvidenceType.MARKET,
        publisher="통계청",
        source_url="https://example.org/data",
        domain="competition",
        evaluation_criteria=["시장성"],
        supported_roles=supported_roles or [],
        content=content,
        reference_date=reference_date,
        published_at=published_at,
    )
    base.update(overrides)
    return ExternalEvidenceDocument(**base)


def _build_service(tmp_path, *, min_score=-1.0):
    client = create_persistent_client(str(tmp_path / "chroma"))
    embedder = FakeEvidenceEmbedder(dimension=_DIM)
    config = ExternalResearchConfig(min_similarity_score=min_score)
    repo = ExternalEvidenceRepository(
        client=client,
        collection_name=config.collection_name,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.embedding_dimension,
        embedding_version="embedding_v1",
    )
    provider = DatasetProvider(repo, embedder, config=config)
    service = ExternalResearchService(provider, config=config)
    return service, repo, embedder


def _index(repo, embedder, docs):
    for doc in docs:
        repo.upsert_evidence_chunk(doc, embedder.embed_query(doc.content))


def test_role_mapping_planning_and_dev(tmp_path):
    assert resolve_external_reviewer_role("planning_expert") == "planning"
    assert resolve_external_reviewer_role("dev_expert") == "technology"
    # 등록되지 않은 persona_id는 그대로 통과한다(예외를 던지지 않음).
    assert resolve_external_reviewer_role("ideation_facilitator") == "ideation_facilitator"


def test_dataset_search_returns_at_least_one_result(tmp_path):
    service, repo, embedder = _build_service(tmp_path)
    _index(
        repo,
        embedder,
        [_doc("DOC-1", "CHUNK-1", content="스마트시티 시장 규모는 5조원입니다.", reference_date="2025-01-01")],
    )
    result = search_ideation_external_evidence("planning_expert", "스마트시티 공모전 시장 규모", service)
    assert len(result["external_evidence"]) >= 1
    assert result["used_dataset_search"] is True
    item = result["external_evidence"][0]
    assert item["publisher"] == "통계청"
    assert item["source_url"] == "https://example.org/data"
    assert item["reference_date"] == "2025-01-01"


def test_planning_and_dev_role_queries_map_to_different_reviewer_roles(tmp_path, monkeypatch):
    service, repo, embedder = _build_service(tmp_path)
    _index(repo, embedder, [_doc("DOC-1", "CHUNK-1", content="본문", reference_date="2025-01-01")])

    captured_roles = []
    original_search = service.search

    def spy(request):
        captured_roles.append(request.reviewer_role)
        return original_search(request)

    monkeypatch.setattr(service, "search", spy)

    search_ideation_external_evidence("planning_expert", "쿼리1", service)
    search_ideation_external_evidence("dev_expert", "쿼리2", service)

    assert captured_roles == ["planning", "technology"]


def test_missing_confirmed_date_is_dropped_from_display(tmp_path):
    """publisher/source_url은 있지만 reference_date/published_at이 모두 없는 자료는
    source_validator를 통과해도(검증은 날짜를 요구하지 않음) 화면 노출 전 단계에서 제외돼야
    한다(요청 9번)."""
    service, repo, embedder = _build_service(tmp_path)
    _index(
        repo,
        embedder,
        [
            _doc("DOC-DATED", "CHUNK-1", content="기준일이 있는 자료 본문.", reference_date="2025-06-01"),
            _doc("DOC-UNDATED", "CHUNK-2", content="기준일이 없는 자료 본문.", reference_date=None, published_at=None),
        ],
    )
    result = search_ideation_external_evidence("planning_expert", "자료", service, top_k=10)
    document_ids = {item["document_id"] for item in result["external_evidence"]}
    assert "DOC-DATED" in document_ids
    assert "DOC-UNDATED" not in document_ids


def test_no_results_does_not_raise(tmp_path):
    service, repo, embedder = _build_service(tmp_path)
    result = search_ideation_external_evidence("planning_expert", "아무 결과도 없는 질의", service)
    assert result["external_evidence"] == []
    assert result["used_dataset_search"] is True
    assert result["warnings"]  # "찾지 못했습니다" 계열 경고 문구가 남는다.


def test_search_failure_is_fail_closed(tmp_path):
    service, _repo, _embedder = _build_service(tmp_path)

    def boom(request):
        raise RuntimeError("search backend unavailable")

    service.search = boom  # type: ignore[assignment]
    result = search_ideation_external_evidence("planning_expert", "질의", service)
    assert result["external_evidence"] == []
    assert result["warnings"]


def test_lookup_caches_repeated_query_within_same_lookup_instance(tmp_path, monkeypatch):
    service, repo, embedder = _build_service(tmp_path)
    _index(repo, embedder, [_doc("DOC-1", "CHUNK-1", content="본문", reference_date="2025-01-01")])

    call_count = {"n": 0}
    original_search = service.search

    def counting_search(request):
        call_count["n"] += 1
        return original_search(request)

    monkeypatch.setattr(service, "search", counting_search)

    lookup = make_ideation_external_evidence_lookup(service, top_k=5)
    lookup("planning_expert", "같은 질의")
    lookup("planning_expert", "같은 질의")
    assert call_count["n"] == 1, "동일 질의가 캐시를 우회해 반복 검색됐습니다."

    lookup("planning_expert", "다른 질의")
    assert call_count["n"] == 2


def test_lookup_signature_accepts_runtime_scope_kwarg_without_error(tmp_path):
    """ai/meeting/graph/ideation_conv_discovery.py의 _call_external_evidence_lookup은
    external_evidence_lookup(persona_id, query)만 호출하지만, ai/meeting/graph/ideation_nodes.py::
    call_evidence_lookup처럼 runtime_scope 키워드를 넘기는 다른 호출부가 생기더라도 안전해야
    한다 — lookup이 **kwargs를 받아 흡수한다."""
    service, repo, embedder = _build_service(tmp_path)
    lookup = make_ideation_external_evidence_lookup(service, top_k=3)
    result = lookup("planning_expert", "질의", runtime_scope={"session_id": "S1"})
    assert isinstance(result, dict)
    assert "external_evidence" in result
