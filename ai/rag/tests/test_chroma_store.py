"""
Unit Tests for ai.rag.retrieval.chroma_store (실제 chromadb PersistentClient + tmp_path 사용,
KURE-v1 모델은 로딩하지 않음)
"""

import chromadb
import pytest

from ai.rag.domain import IndexingContext
from ai.rag.domain.config import DEFAULT_COLLECTION_NAME
from ai.rag.domain.schemas import CollectionConfigMismatchError, InvalidTopKError
from ai.rag.embedding.schemas import EmbeddedChunk, EmbeddingResult
from ai.rag.retrieval.chroma_store import ChromaVectorStore, build_record_id, create_persistent_client

_DIM = 4
_MODEL = "fake-model"
_VERSION = "embedding_v1"
_COLLECTION = DEFAULT_COLLECTION_NAME


def _vec(seed: float) -> list[float]:
    return [seed, seed + 0.1, seed + 0.2, seed + 0.3]


def _make_embedded_chunk(chunk_id: str, document_id: str, project_id: str, seed: float, **metadata_overrides) -> EmbeddedChunk:
    metadata = {
        "project_id": project_id,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "chunk_index": 0,
        "source_type": "url_webpage",
        "source_url": "https://example.com",
        "source_page_url": None,
        "source_filename": None,
        "file_type": None,
        "location_type": "web_section",
        "location_number": None,
        "section_title": "섹션 제목",
        "content_kind": "body",
        "source_block_ids": ["b1", "b2"],
        "source_block_orders": [0, 1],
        "chunking_version": "chunking_v2",
        "embedding_model": _MODEL,
        "embedding_version": _VERSION,
        "indexable": True,
        "document_title": "문서 제목",
    }
    metadata.update(metadata_overrides)
    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        embedding=_vec(seed),
        embedding_dimension=_DIM,
        embedding_text=f"임베딩용 텍스트 {chunk_id}",
        content=f"원문 내용 {chunk_id}",
        metadata=metadata,
    )


def _make_embedding_result(project_id: str, document_id: str, chunks: list[EmbeddedChunk], **overrides) -> EmbeddingResult:
    defaults = dict(
        project_id=project_id,
        document_id=document_id,
        embedded_chunks=chunks,
        embedding_count=len(chunks),
        skipped_chunk_ids=[],
        failed_chunk_ids=[],
        warnings=[],
        model_name=_MODEL,
        embedding_dimension=_DIM,
        normalized=True,
        embedding_version=_VERSION,
    )
    defaults.update(overrides)
    return EmbeddingResult(**defaults)


@pytest.fixture
def store(tmp_path):
    client = create_persistent_client(path=str(tmp_path / "chroma_data"))
    return ChromaVectorStore(
        client=client,
        collection_name=_COLLECTION,
        embedding_model=_MODEL,
        embedding_dimension=_DIM,
        embedding_version=_VERSION,
    )


class TestUpsert:
    def test_all_records_have_project_and_document_id(self, store):
        chunks = [_make_embedded_chunk("c1", "doc-1", "p1", 0.1)]
        embedding_result = _make_embedding_result("p1", "doc-1", chunks)
        context = IndexingContext(project_id="p1", document_id="doc-1")
        store.upsert_embedding_result(embedding_result, context)

        record_id = build_record_id("p1", "c1")
        got = store._collection.get(ids=[record_id])
        assert got["metadatas"][0]["project_id"] == "p1"
        assert got["metadatas"][0]["document_id"] == "doc-1"

    def test_chunk_content_stored_as_document(self, store):
        chunks = [_make_embedded_chunk("c1", "doc-1", "p1", 0.1)]
        embedding_result = _make_embedding_result("p1", "doc-1", chunks)
        store.upsert_embedding_result(embedding_result, IndexingContext(project_id="p1", document_id="doc-1"))

        got = store._collection.get(ids=[build_record_id("p1", "c1")])
        assert got["documents"][0] == "원문 내용 c1"
        assert got["documents"][0] != chunks[0].embedding_text

    def test_source_metadata_round_trips(self, store):
        chunks = [_make_embedded_chunk("c1", "doc-1", "p1", 0.1)]
        embedding_result = _make_embedding_result("p1", "doc-1", chunks)
        store.upsert_embedding_result(embedding_result, IndexingContext(project_id="p1", document_id="doc-1"))

        results = store.search(query_embedding=_vec(0.1), project_id="p1", top_k=1)
        assert results[0].metadata["section_title"] == "섹션 제목"
        assert results[0].metadata["source_block_ids"] == ["b1", "b2"]
        assert results[0].metadata["source_block_orders"] == [0, 1]
        assert results[0].metadata["document_title"] == "문서 제목"

    def test_section_page_document_id_chunk_id_round_trip(self, store):
        """section_title/location_number(page)/document_id/chunk_id/document_role이
        Chroma 저장·조회를 거쳐도 그대로 유지되어야 한다."""
        chunks = [
            _make_embedded_chunk(
                "c1", "doc-1", "p1", 0.1,
                section_title="1) 개요", location_number=3, document_role="submission",
            )
        ]
        embedding_result = _make_embedding_result("p1", "doc-1", chunks)
        store.upsert_embedding_result(embedding_result, IndexingContext(project_id="p1", document_id="doc-1"))

        results = store.search(query_embedding=_vec(0.1), project_id="p1", top_k=1)
        assert results[0].document_id == "doc-1"
        assert results[0].chunk_id == "c1"
        assert results[0].metadata["section_title"] == "1) 개요"
        assert results[0].metadata["location_number"] == 3
        assert results[0].metadata["document_role"] == "submission"

    def test_section_title_none_is_dropped_not_stored_as_null_string(self, store):
        """section을 확실히 판단하지 못해 None인 청크는 Chroma에 None 키가 저장되지 않고
        (chroma가 None을 조용히 버림), 조회 시에도 section_title 키가 없거나 None이어야 한다."""
        chunks = [_make_embedded_chunk("c1", "doc-1", "p1", 0.1, section_title=None)]
        embedding_result = _make_embedding_result("p1", "doc-1", chunks)
        store.upsert_embedding_result(embedding_result, IndexingContext(project_id="p1", document_id="doc-1"))

        results = store.search(query_embedding=_vec(0.1), project_id="p1", top_k=1)
        assert results[0].metadata.get("section_title") is None

    def test_reindexing_same_chunk_no_duplicate(self, store):
        chunks = [_make_embedded_chunk("c1", "doc-1", "p1", 0.1)]
        embedding_result = _make_embedding_result("p1", "doc-1", chunks)
        context = IndexingContext(project_id="p1", document_id="doc-1")
        store.upsert_embedding_result(embedding_result, context)
        result2 = store.upsert_embedding_result(embedding_result, context)
        assert result2.stored_record_count == 1

    def test_stale_records_removed_after_rechunking(self, store):
        context = IndexingContext(project_id="p1", document_id="doc-1")
        first = _make_embedding_result("p1", "doc-1", [
            _make_embedded_chunk("c1", "doc-1", "p1", 0.1),
            _make_embedded_chunk("c2", "doc-1", "p1", 0.2),
        ])
        store.upsert_embedding_result(first, context)

        second = _make_embedding_result("p1", "doc-1", [_make_embedded_chunk("c3", "doc-1", "p1", 0.3)])
        result = store.upsert_embedding_result(second, context)

        assert result.deleted_stale_count == 2
        assert result.stored_record_count == 1

    def test_empty_embedding_result_status_empty(self, store):
        embedding_result = _make_embedding_result("p1", "doc-1", [], embedding_count=0)
        result = store.upsert_embedding_result(embedding_result, IndexingContext(project_id="p1", document_id="doc-1"))
        assert result.status.value == "empty"
        assert result.stored_record_count == 0

    def test_different_projects_same_chunk_id_no_collision(self, store):
        context1 = IndexingContext(project_id="p1", document_id="doc-1")
        context2 = IndexingContext(project_id="p2", document_id="doc-1")
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-1", [_make_embedded_chunk("chk_same", "doc-1", "p1", 0.1)]), context1)
        store.upsert_embedding_result(_make_embedding_result("p2", "doc-1", [_make_embedded_chunk("chk_same", "doc-1", "p2", 0.9)]), context2)

        p1_results = store.search(query_embedding=_vec(0.1), project_id="p1", top_k=5)
        p2_results = store.search(query_embedding=_vec(0.9), project_id="p2", top_k=5)
        assert len(p1_results) == 1
        assert len(p2_results) == 1
        assert p1_results[0].record_id != p2_results[0].record_id


class TestDeleteAndSearch:
    def test_delete_document(self, store):
        context = IndexingContext(project_id="p1", document_id="doc-1")
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-1", [
            _make_embedded_chunk("c1", "doc-1", "p1", 0.1),
            _make_embedded_chunk("c2", "doc-1", "p1", 0.2),
        ]), context)
        deleted = store.delete_document("p1", "doc-1")
        assert deleted == 2
        assert store.search(query_embedding=_vec(0.1), project_id="p1", top_k=5) == []

    def test_search_excludes_other_projects(self, store):
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-1", [_make_embedded_chunk("c1", "doc-1", "p1", 0.1)]), IndexingContext(project_id="p1", document_id="doc-1"))
        store.upsert_embedding_result(_make_embedding_result("p2", "doc-1", [_make_embedded_chunk("c2", "doc-1", "p2", 0.1)]), IndexingContext(project_id="p2", document_id="doc-1"))

        results = store.search(query_embedding=_vec(0.1), project_id="p1", top_k=10)
        assert all(r.metadata["project_id"] == "p1" for r in results)
        assert len(results) == 1

    def test_search_document_id_filter(self, store):
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-1", [_make_embedded_chunk("c1", "doc-1", "p1", 0.1)]), IndexingContext(project_id="p1", document_id="doc-1"))
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-2", [_make_embedded_chunk("c2", "doc-2", "p1", 0.1)]), IndexingContext(project_id="p1", document_id="doc-2"))

        results = store.search(query_embedding=_vec(0.1), project_id="p1", document_id="doc-1", top_k=10)
        assert len(results) == 1
        assert results[0].document_id == "doc-1"

    def test_top_k_applied(self, store):
        chunks = [_make_embedded_chunk(f"c{i}", "doc-1", "p1", i * 0.1) for i in range(5)]
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-1", chunks), IndexingContext(project_id="p1", document_id="doc-1"))

        results = store.search(query_embedding=_vec(0.0), project_id="p1", top_k=2)
        assert len(results) == 2

    def test_top_k_below_one_raises(self, store):
        with pytest.raises(InvalidTopKError):
            store.search(query_embedding=_vec(0.1), project_id="p1", top_k=0)


class TestDeleteProject:
    def test_deletes_all_documents_in_project(self, store):
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-1", [
            _make_embedded_chunk("c1", "doc-1", "p1", 0.1),
            _make_embedded_chunk("c2", "doc-1", "p1", 0.2),
        ]), IndexingContext(project_id="p1", document_id="doc-1"))
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-2", [
            _make_embedded_chunk("c3", "doc-2", "p1", 0.3),
        ]), IndexingContext(project_id="p1", document_id="doc-2"))

        deleted = store.delete_project("p1")
        assert deleted == 3

    def test_other_projects_untouched(self, store):
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-1", [
            _make_embedded_chunk("c1", "doc-1", "p1", 0.1),
        ]), IndexingContext(project_id="p1", document_id="doc-1"))
        store.upsert_embedding_result(_make_embedding_result("p2", "doc-1", [
            _make_embedded_chunk("c2", "doc-1", "p2", 0.1),
        ]), IndexingContext(project_id="p2", document_id="doc-1"))

        store.delete_project("p1")

        p2_results = store.search(query_embedding=_vec(0.1), project_id="p2", top_k=5)
        assert len(p2_results) == 1
        assert p2_results[0].metadata["project_id"] == "p2"

    def test_nonexistent_project_returns_zero(self, store):
        assert store.delete_project("no-such-project") == 0

    def test_search_empty_after_delete(self, store):
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-1", [
            _make_embedded_chunk("c1", "doc-1", "p1", 0.1),
            _make_embedded_chunk("c2", "doc-1", "p1", 0.2),
        ]), IndexingContext(project_id="p1", document_id="doc-1"))

        store.delete_project("p1")

        assert store.search(query_embedding=_vec(0.1), project_id="p1", top_k=10) == []

    def test_get_returns_no_ids_after_delete(self, store):
        store.upsert_embedding_result(_make_embedding_result("p1", "doc-1", [
            _make_embedded_chunk("c1", "doc-1", "p1", 0.1),
        ]), IndexingContext(project_id="p1", document_id="doc-1"))

        store.delete_project("p1")

        assert store._list_record_ids("p1") == []


class TestCollectionConfigValidation:
    def test_model_mismatch_detected(self, tmp_path):
        client = create_persistent_client(path=str(tmp_path / "chroma_data"))
        ChromaVectorStore(client=client, collection_name=_COLLECTION, embedding_model="model-a", embedding_dimension=_DIM, embedding_version=_VERSION)
        with pytest.raises(CollectionConfigMismatchError):
            ChromaVectorStore(client=client, collection_name=_COLLECTION, embedding_model="model-b", embedding_dimension=_DIM, embedding_version=_VERSION)

    def test_dimension_mismatch_detected(self, tmp_path):
        client = create_persistent_client(path=str(tmp_path / "chroma_data"))
        ChromaVectorStore(client=client, collection_name=_COLLECTION, embedding_model=_MODEL, embedding_dimension=4, embedding_version=_VERSION)
        with pytest.raises(CollectionConfigMismatchError):
            ChromaVectorStore(client=client, collection_name=_COLLECTION, embedding_model=_MODEL, embedding_dimension=99, embedding_version=_VERSION)


class TestPersistence:
    def test_persistent_client_reopen_retains_data(self, tmp_path):
        db_path = str(tmp_path / "chroma_data")
        client1 = create_persistent_client(path=db_path)
        store1 = ChromaVectorStore(client=client1, collection_name=_COLLECTION, embedding_model=_MODEL, embedding_dimension=_DIM, embedding_version=_VERSION)
        store1.upsert_embedding_result(
            _make_embedding_result("p1", "doc-1", [_make_embedded_chunk("c1", "doc-1", "p1", 0.1)]),
            IndexingContext(project_id="p1", document_id="doc-1"),
        )
        del client1, store1

        client2 = create_persistent_client(path=db_path)
        store2 = ChromaVectorStore(client=client2, collection_name=_COLLECTION, embedding_model=_MODEL, embedding_dimension=_DIM, embedding_version=_VERSION)
        results = store2.search(query_embedding=_vec(0.1), project_id="p1", top_k=5)
        assert len(results) == 1
