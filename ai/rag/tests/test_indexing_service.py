"""
Unit Tests for ai.rag.retrieval.service.RAGIndexingService
(fake_kure_embedder + 실제 chromadb PersistentClient, KURE-v1 모델은 로딩하지 않음)
"""

import threading

import pytest

from ai.rag.chunking.chunker import chunk_document
from ai.rag.chunking.schemas import (
    Chunk,
    ChunkingConfig,
    ChunkingResult,
    ChunkLocationType,
    ChunkSourceContext,
    ContentKind,
    SourceType,
)
from ai.rag.domain import IndexingContext
from ai.rag.domain.config import DEFAULT_COLLECTION_NAME
from ai.rag.loaders.schemas import WebBlockType, WebContentBlock
from ai.rag.preprocessing.schemas import CleanedWebContent
from ai.rag.retrieval.chroma_store import ChromaVectorStore, create_persistent_client
from ai.rag.retrieval.exceptions import EmbeddingStageError, RAGIndexingError, VectorStoreStageError
from ai.rag.retrieval.service import RAGIndexingService

_COLLECTION = DEFAULT_COLLECTION_NAME


def _make_chunking_result() -> ChunkingResult:
    chunks = [
        Chunk(
            chunk_id="chk_1",
            document_id="doc-1",
            content="접수기간은 2026년 3월 1일부터 3월 31일까지입니다.",
            chunk_index=0,
            content_kind=ContentKind.BODY,
            source_type=SourceType.URL_WEBPAGE,
            location_type=ChunkLocationType.WEB_SECTION,
            location_number=None,
            section_title="접수 안내",
            char_count=30,
            indexable=True,
        ),
        Chunk(
            chunk_id="chk_2",
            document_id="doc-1",
            content="목차\n1. 개요\n2. 접수방법",
            chunk_index=1,
            content_kind=ContentKind.TOC,
            source_type=SourceType.URL_WEBPAGE,
            location_type=ChunkLocationType.WEB_SECTION,
            location_number=None,
            section_title="목차",
            char_count=20,
            indexable=False,
        ),
    ]
    return ChunkingResult(document_id="doc-1", chunks=chunks, chunk_count=len(chunks), config=ChunkingConfig())


def _make_service(fake_kure_embedder, tmp_path) -> RAGIndexingService:
    client = create_persistent_client(path=str(tmp_path / "chroma_data"))
    store = ChromaVectorStore(
        client=client,
        collection_name=_COLLECTION,
        embedding_model=fake_kure_embedder.model_name,
        embedding_dimension=fake_kure_embedder.embedding_dimension,
        embedding_version="embedding_v1",
    )
    return RAGIndexingService(fake_kure_embedder, store)


class TestIndexChunkingResult:
    def test_full_flow_index_and_search(self, fake_kure_embedder, tmp_path):
        service = _make_service(fake_kure_embedder, tmp_path)
        context = IndexingContext(project_id="p1", document_id="doc-1", document_title="공모전 안내")

        indexing_result = service.index_chunking_result(_make_chunking_result(), context)
        assert indexing_result.embedded_count == 1  # TOC 청크(chk_2)는 indexable=False라 제외
        assert indexing_result.skipped_count == 1
        assert indexing_result.stored_record_count == 1

        results = service.search("접수기간이 언제인가요?", project_id="p1", top_k=3)
        assert len(results) == 1
        assert results[0].chunk_id == "chk_1"
        assert "접수기간" in results[0].content

    def test_chunk_count_and_stored_count_consistent(self, fake_kure_embedder, tmp_path):
        service = _make_service(fake_kure_embedder, tmp_path)
        context = IndexingContext(project_id="p1", document_id="doc-1")
        summary = service.index_chunking_result_with_summary(_make_chunking_result(), context)

        assert summary.chunk_count == 2
        assert summary.indexable_chunk_count == 1
        assert summary.stored_count == summary.embedding_count == 1

    def test_summary_excludes_vectors_and_embedding_text(self, fake_kure_embedder, tmp_path):
        service = _make_service(fake_kure_embedder, tmp_path)
        summary = service.index_chunking_result_with_summary(
            _make_chunking_result(), IndexingContext(project_id="p1", document_id="doc-1")
        )
        dumped = summary.model_dump()
        assert "embedding" not in dumped
        assert "embedding_text" not in dumped
        assert "embedded_chunks" not in dumped

    def test_reindex_after_removed_chunk_deletes_stale(self, fake_kure_embedder, tmp_path):
        service = _make_service(fake_kure_embedder, tmp_path)
        context = IndexingContext(project_id="p1", document_id="doc-1")
        service.index_chunking_result(_make_chunking_result(), context)

        smaller_result = ChunkingResult(
            document_id="doc-1",
            chunks=[],
            chunk_count=0,
            config=ChunkingConfig(),
        )
        result2 = service.index_chunking_result(smaller_result, context)
        assert result2.deleted_stale_count == 1
        assert result2.stored_record_count == 0


class TestDeleteProject:
    def test_deletes_project_and_search_becomes_empty(self, fake_kure_embedder, tmp_path):
        """PRJ-004: RAGIndexingService.delete_project()를 실제 ChromaVectorStore로 통합 실행."""
        service = _make_service(fake_kure_embedder, tmp_path)
        context = IndexingContext(project_id="p1", document_id="doc-1")
        service.index_chunking_result(_make_chunking_result(), context)

        deleted = service.delete_project("p1")

        assert deleted == 1  # chk_2는 indexable=False라 애초에 저장되지 않음(chk_1만 색인됨)
        assert service.search("접수기간이 언제인가요?", project_id="p1", top_k=3) == []

    def test_delegates_to_vector_store_and_returns_its_value(self, fake_kure_embedder):
        """vector_store.delete_project()에 project_id를 그대로 전달하고 반환값을 그대로 돌려주는지
        (실제 Chroma 없이) fake vector_store로 위임 자체만 검증한다."""

        class _FakeVectorStore:
            def __init__(self):
                self.calls: list[str] = []

            def delete_project(self, project_id: str) -> int:
                self.calls.append(project_id)
                return 42

        fake_store = _FakeVectorStore()
        service = RAGIndexingService(fake_kure_embedder, fake_store)

        result = service.delete_project("PRJ-004")

        assert fake_store.calls == ["PRJ-004"]
        assert result == 42


def _make_webpage_chunking_result(document_id: str = "doc-web-1") -> tuple[ChunkingResult, ChunkSourceContext]:
    """2026-07-18, fetch-url 색인 hang 조사(용준/Claude): 실제 리포트된 repro URL
    (sotong.go.kr 에필로그 페이지, cleaned_text_length=1577)과 비슷한 짧은 한국어
    웹페이지 콘텐츠로 chunk_document()의 URL_WEBPAGE 경로 전체(정제->청킹->임베딩->저장)가
    정상 동작함을 회귀 테스트로 고정한다."""
    blocks = [
        WebContentBlock(order=0, block_type=WebBlockType.HEADING, content="2023 공공분야 챗봇 AI 활용 가이드라인 공모전"),
        WebContentBlock(
            order=1, block_type=WebBlockType.PARAGRAPH,
            content="공공분야에서 챗봇 AI를 활용하고 있는 사례를 발굴하기 위해 실시하는 "
                    "'2023 공공분야 챗봇 AI 활용 가이드라인 공모전' 수상작을 발표합니다.",
        ),
        WebContentBlock(
            order=2, block_type=WebBlockType.LIST,
            content="- [포스터] 2023 공공분야 챗봇AI 활용 가이드라인 공모전.jpg(199KB)\n"
                    "- [기획서] 2023 공공분야 챗봇AI 활용 가이드라인 공모전.hwpx(52KB)",
        ),
    ]
    text = "\n\n".join(b.content for b in blocks)
    cleaned = CleanedWebContent(
        source_url="https://example.go.kr/notice/1",
        original_block_count=len(blocks),
        cleaned_block_count=len(blocks),
        cleaned_blocks=blocks,
        removed_blocks=[],
        original_text_length=len(text),
        cleaned_text_length=len(text),
        retention_ratio=1.0,
    )
    chunk_context = ChunkSourceContext(
        document_id=document_id,
        source_type=SourceType.URL_WEBPAGE,
        source_url="https://example.go.kr/notice/1",
        document_title="2023 공공분야 챗봇 AI 활용 가이드라인 공모전",
    )
    return chunk_document(cleaned, chunk_context), chunk_context


class TestShortKoreanWebpageIndexing:
    """실제 hang 재현 시도(repro_hang.py/repro_real_url.py, 스크래치패드)로 chunk->embed->store
    파이프라인 자체는 정상 완료됨을 확인했다 — 그 경로를 기본 테스트 스위트에 회귀 테스트로 남긴다."""

    def test_webpage_source_indexes_successfully(self, fake_kure_embedder, tmp_path):
        service = _make_service(fake_kure_embedder, tmp_path)
        chunking_result, chunk_context = _make_webpage_chunking_result()
        context = IndexingContext(
            project_id="p-web", document_id=chunking_result.document_id, document_title=chunk_context.document_title
        )

        result = service.index_chunking_result(chunking_result, context)

        assert result.status.value in ("success", "partial")
        assert result.stored_record_count > 0

        results = service.search("공모전 수상작 발표", project_id="p-web", top_k=5)
        assert len(results) > 0
        assert all(r.metadata["source_type"] == "url_webpage" for r in results)


class TestStageFailureIsolation:
    """DoD: 임베딩 단계 실패와 Chroma 저장 단계 실패를 호출부가 구분할 수 있어야 한다
    (backend/documents.py의 except 블록이 어떤 단계에서 멈췄는지 로그로 알 수 있도록)."""

    def test_embedding_failure_raises_embedding_stage_error(self, fake_kure_embedder, tmp_path, monkeypatch):
        service = _make_service(fake_kure_embedder, tmp_path)
        context = IndexingContext(project_id="p1", document_id="doc-1")

        def _boom(*args, **kwargs):
            raise RuntimeError("모델 encode 실패 시뮬레이션")

        monkeypatch.setattr(fake_kure_embedder, "embed_chunking_result", _boom)

        with pytest.raises(EmbeddingStageError) as exc_info:
            service.index_chunking_result(_make_chunking_result(), context)

        assert isinstance(exc_info.value, RAGIndexingError)
        assert exc_info.value.stage == "embed"
        assert exc_info.value.document_id == "doc-1"
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_chroma_failure_raises_vector_store_stage_error(self, fake_kure_embedder, tmp_path, monkeypatch):
        service = _make_service(fake_kure_embedder, tmp_path)
        context = IndexingContext(project_id="p1", document_id="doc-1")

        def _boom(*args, **kwargs):
            raise RuntimeError("Chroma upsert 실패 시뮬레이션")

        monkeypatch.setattr(service._vector_store, "upsert_embedding_result", _boom)

        with pytest.raises(VectorStoreStageError) as exc_info:
            service.index_chunking_result(_make_chunking_result(), context)

        assert isinstance(exc_info.value, RAGIndexingError)
        assert exc_info.value.stage == "chroma_upsert"
        assert exc_info.value.document_id == "doc-1"
        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestConcurrentIndexing:
    """동시에 여러 문서를 색인해도(같은 RAGIndexingService/ChromaVectorStore 인스턴스를
    여러 스레드가 공유) 서로의 결과를 침범하지 않고 각자 정상적으로 저장되어야 한다."""

    def test_concurrent_index_calls_do_not_corrupt_each_other(self, fake_kure_embedder, tmp_path):
        service = _make_service(fake_kure_embedder, tmp_path)
        n_threads = 8
        errors: list[Exception] = []

        def _index(i: int):
            try:
                context = IndexingContext(project_id="p1", document_id=f"doc-{i}")
                chunks = [
                    Chunk(
                        chunk_id=f"chk-{i}",
                        document_id=f"doc-{i}",
                        content=f"문서 {i}의 본문 내용입니다.",
                        chunk_index=0,
                        content_kind=ContentKind.BODY,
                        source_type=SourceType.URL_WEBPAGE,
                        location_type=ChunkLocationType.WEB_SECTION,
                        location_number=None,
                        section_title=None,
                        char_count=10,
                        indexable=True,
                    )
                ]
                result = ChunkingResult(document_id=f"doc-{i}", chunks=chunks, chunk_count=1, config=ChunkingConfig())
                service.index_chunking_result(result, context)
            except Exception as exc:  # pragma: no cover - 실패 시 assert에서 드러남
                errors.append(exc)

        threads = [threading.Thread(target=_index, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not any(t.is_alive() for t in threads), "일부 스레드가 30초 내에 끝나지 않았습니다 (hang 의심)"
        assert errors == []

        for i in range(n_threads):
            # fake 임베딩은 의미 기반이 아니라 정확한 문자열 해시 기반이라 document_id로
            # 필터링해서 조회한다 — 여기서 검증하려는 건 검색 랭킹이 아니라 "동시에 색인해도
            # 각 문서의 청크가 다른 문서에 덮어써지거나 유실되지 않는지"다.
            results = service.search(f"문서 {i}의 본문", project_id="p1", document_id=f"doc-{i}", top_k=1)
            assert len(results) == 1
            assert results[0].document_id == f"doc-{i}"
