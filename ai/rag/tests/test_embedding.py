"""
Unit Tests for ai.rag.embedding (실제 KURE-v1 모델은 로딩하지 않고 fake_kure_embedder 사용)
"""

import logging

import numpy as np
import pytest

from ai.rag.chunking.schemas import (
    Chunk,
    ChunkingConfig,
    ChunkingResult,
    ChunkLocationType,
    ContentKind,
    SourceType,
)
from ai.rag.domain import IndexingContext
from ai.rag.embedding import EmptyQueryError, NonFiniteEmbeddingError, build_embedding_text
from ai.rag.embedding.kure_embedder import KUREEmbedder
from ai.rag.embedding.schemas import EmbeddingConfig
from ai.rag.tests.embedding_fixtures import FakeSentenceTransformer, FAKE_EMBEDDING_DIMENSION


def _make_chunk(
    chunk_id: str,
    document_id: str = "doc-1",
    content: str = "본문 내용",
    chunk_index: int = 0,
    indexable: bool = True,
    content_kind: ContentKind = ContentKind.BODY,
    section_title: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        chunk_index=chunk_index,
        content_kind=content_kind,
        source_type=SourceType.URL_WEBPAGE,
        location_type=ChunkLocationType.WEB_SECTION,
        location_number=None,
        section_title=section_title,
        char_count=len(content),
        indexable=indexable,
    )


def _make_chunking_result(chunks: list[Chunk], document_id: str = "doc-1") -> ChunkingResult:
    return ChunkingResult(
        document_id=document_id,
        chunks=chunks,
        chunk_count=len(chunks),
        config=ChunkingConfig(),
    )


# --- text_builder ---

class TestBuildEmbeddingText:
    def test_combines_title_section_content(self):
        text = build_embedding_text("본문", document_title="공모전 안내", section_title="접수 방법")
        assert text == "문서 제목: 공모전 안내\n섹션: 접수 방법\n\n본문"

    def test_none_values_excluded(self):
        text = build_embedding_text("본문", document_title=None, section_title=None)
        assert text == "본문"

    def test_blank_values_excluded(self):
        text = build_embedding_text("본문", document_title="   ", section_title="")
        assert text == "본문"

    def test_duplicate_title_and_section_not_repeated(self):
        text = build_embedding_text("본문", document_title="같은제목", section_title="같은제목")
        assert text.count("같은제목") == 1

    def test_title_equal_to_content_is_dropped(self):
        text = build_embedding_text("동일텍스트", document_title="동일텍스트", section_title=None)
        assert text == "동일텍스트"


# --- KUREEmbedder.embed_chunking_result ---

class TestEmbedChunkingResult:
    def test_indexable_true_only(self, fake_kure_embedder):
        chunks = [
            _make_chunk("c1", indexable=True),
            _make_chunk("c2", indexable=False, content_kind=ContentKind.TOC, chunk_index=1),
        ]
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), IndexingContext(project_id="p1", document_id="doc-1"))
        assert result.embedding_count == 1
        assert [c.chunk_id for c in result.embedded_chunks] == ["c1"]

    def test_indexable_false_recorded_as_skipped(self, fake_kure_embedder):
        chunks = [
            _make_chunk("c1", indexable=True),
            _make_chunk("c2", indexable=False, chunk_index=1),
        ]
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), IndexingContext(project_id="p1", document_id="doc-1"))
        assert result.skipped_chunk_ids == ["c2"]

    def test_embedding_text_uses_section_title_and_content(self, fake_kure_embedder):
        chunks = [_make_chunk("c1", content="표 데이터", section_title="세부 규격")]
        result = fake_kure_embedder.embed_chunking_result(
            _make_chunking_result(chunks), IndexingContext(project_id="p1", document_id="doc-1")
        )
        assert "세부 규격" in result.embedded_chunks[0].embedding_text
        assert "표 데이터" in result.embedded_chunks[0].embedding_text

    def test_embedding_text_uses_document_title(self, fake_kure_embedder):
        chunks = [_make_chunk("c1", content="본문")]
        context = IndexingContext(project_id="p1", document_id="doc-1", document_title="2026 공모전")
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), context)
        assert "2026 공모전" in result.embedded_chunks[0].embedding_text

    def test_duplicate_title_not_repeated_in_embedding_text(self, fake_kure_embedder):
        chunks = [_make_chunk("c1", content="본문", section_title="같은제목")]
        context = IndexingContext(project_id="p1", document_id="doc-1", document_title="같은제목")
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), context)
        assert result.embedded_chunks[0].embedding_text.count("같은제목") == 1

    def test_preserves_input_order(self, fake_kure_embedder):
        chunks = [_make_chunk(f"c{i}", content=f"내용{i}", chunk_index=i) for i in range(5)]
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), IndexingContext(project_id="p1", document_id="doc-1"))
        assert [c.chunk_id for c in result.embedded_chunks] == [f"c{i}" for i in range(5)]

    def test_empty_chunking_result(self, fake_kure_embedder):
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result([]), IndexingContext(project_id="p1", document_id="doc-1"))
        assert result.embedding_count == 0
        assert result.embedded_chunks == []
        assert result.warnings

    def test_document_role_propagated_from_indexing_context_to_metadata(self, fake_kure_embedder):
        """IndexingContext.document_role이 KUREEmbedder를 거쳐 EmbeddedChunk.metadata까지
        실제로 전달되는지 확인한다(test_chroma_store.py는 EmbeddedChunk.metadata에 직접 값을
        넣어 Chroma 왕복만 검증하므로, IndexingContext -> KUREEmbedder 구간은 여기서 검증)."""
        chunks = [_make_chunk("c1", content="본문")]
        context = IndexingContext(project_id="p1", document_id="doc-1", document_role="target")
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), context)
        assert result.embedded_chunks[0].metadata["document_role"] == "target"

    def test_document_role_none_by_default_and_dropped_from_chroma_metadata(self, fake_kure_embedder):
        """document_role을 넘기지 않은 기존 호출은 metadata에 None으로 채워지고(하위 호환
        유지), sanitize_metadata_for_chroma()가 None 값 키를 제거하므로 실제 Chroma 저장
        직전 단계에서는 document_role 키 자체가 빠져야 한다."""
        from ai.rag.retrieval.metadata import sanitize_metadata_for_chroma

        chunks = [_make_chunk("c1", content="본문")]
        context = IndexingContext(project_id="p1", document_id="doc-1")
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), context)

        assert result.embedded_chunks[0].metadata["document_role"] is None
        sanitized = sanitize_metadata_for_chroma(result.embedded_chunks[0].metadata)
        assert "document_role" not in sanitized

    def test_document_type_propagated_from_indexing_context_to_metadata(self, fake_kure_embedder):
        chunks = [_make_chunk("c1", "분석 대상")]
        context = IndexingContext(
            project_id="p1",
            document_id="doc-1",
            document_role="criteria",
            document_type="announcement",
        )
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), context)
        assert result.embedded_chunks[0].metadata["document_type"] == "announcement"

    def test_zero_indexable_chunks(self, fake_kure_embedder):
        chunks = [_make_chunk("c1", indexable=False)]
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), IndexingContext(project_id="p1", document_id="doc-1"))
        assert result.embedding_count == 0
        assert result.skipped_chunk_ids == ["c1"]
        assert result.warnings

    def test_empty_content_chunk_marked_failed(self, fake_kure_embedder):
        chunks = [_make_chunk("c1", content="   ")]
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), IndexingContext(project_id="p1", document_id="doc-1"))
        assert result.embedding_count == 0
        assert result.failed_chunk_ids == ["c1"]

    def test_document_id_mismatch_raises(self, fake_kure_embedder):
        chunks = [_make_chunk("c1")]
        with pytest.raises(ValueError):
            fake_kure_embedder.embed_chunking_result(
                _make_chunking_result(chunks, document_id="doc-1"),
                IndexingContext(project_id="p1", document_id="doc-OTHER"),
            )

    def test_all_vectors_same_dimension(self, fake_kure_embedder):
        chunks = [_make_chunk(f"c{i}", content=f"내용{i}") for i in range(3)]
        result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), IndexingContext(project_id="p1", document_id="doc-1"))
        dims = {c.embedding_dimension for c in result.embedded_chunks}
        assert dims == {FAKE_EMBEDDING_DIMENSION}
        assert result.embedding_dimension == FAKE_EMBEDDING_DIMENSION

    def test_nan_embedding_raises(self, monkeypatch, fake_kure_embedder):
        def _bad_encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
            return np.full((len(texts), FAKE_EMBEDDING_DIMENSION), np.nan, dtype=np.float32)

        monkeypatch.setattr(FakeSentenceTransformer, "encode", _bad_encode)
        chunks = [_make_chunk("c1")]
        with pytest.raises(NonFiniteEmbeddingError):
            fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), IndexingContext(project_id="p1", document_id="doc-1"))

    def test_original_chunking_result_not_mutated(self, fake_kure_embedder):
        chunks = [_make_chunk("c1", content="본문")]
        chunking_result = _make_chunking_result(chunks)
        snapshot = chunking_result.model_copy(deep=True)
        fake_kure_embedder.embed_chunking_result(chunking_result, IndexingContext(project_id="p1", document_id="doc-1"))
        assert chunking_result == snapshot

    def test_encode_log_contains_batch_and_lock_metrics(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "ai.rag.embedding.kure_embedder.SentenceTransformer",
            FakeSentenceTransformer,
        )
        embedder = KUREEmbedder(EmbeddingConfig(model_name="fake-model", batch_size=2))
        chunks = [_make_chunk(f"c{i}", content=f"내용{i}", chunk_index=i) for i in range(5)]

        with caplog.at_level(logging.INFO, logger="ai.rag.embedding.kure_embedder"):
            embedder.embed_chunking_result(
                _make_chunking_result(chunks),
                IndexingContext(project_id="p1", document_id="doc-1"),
            )

        done_log = next(record.message for record in caplog.records if "rag.embed.encode_done" in record.message)
        assert "encode_ms=" in done_log
        assert "lock_wait_ms=" in done_log
        assert "batch_size=2" in done_log
        assert "batch_count=3" in done_log


# --- KUREEmbedder.embed_query ---

class TestEmbedQuery:
    def test_empty_query_rejected(self, fake_kure_embedder):
        with pytest.raises(EmptyQueryError):
            fake_kure_embedder.embed_query("")

    def test_blank_query_rejected(self, fake_kure_embedder):
        with pytest.raises(EmptyQueryError):
            fake_kure_embedder.embed_query("   ")

    def test_query_embedding_dimension_matches_documents(self, fake_kure_embedder):
        chunks = [_make_chunk("c1")]
        doc_result = fake_kure_embedder.embed_chunking_result(_make_chunking_result(chunks), IndexingContext(project_id="p1", document_id="doc-1"))
        query_vector = fake_kure_embedder.embed_query("질문입니다")
        assert len(query_vector) == doc_result.embedded_chunks[0].embedding_dimension

    def test_query_embedding_is_finite(self, fake_kure_embedder):
        vector = fake_kure_embedder.embed_query("질문입니다")
        assert all(np.isfinite(v) for v in vector)


# --- EmbeddingConfig ---

class TestEmbeddingConfig:
    def test_batch_size_must_be_positive(self):
        with pytest.raises(ValueError):
            EmbeddingConfig(batch_size=0)

    def test_cpu_threads_must_be_positive_when_configured(self):
        with pytest.raises(ValueError):
            EmbeddingConfig(cpu_threads=0)

    def test_cpu_threads_can_use_runtime_default(self):
        assert EmbeddingConfig().cpu_threads is None

    def test_configured_cpu_threads_are_applied_before_model_load(self, monkeypatch):
        import torch

        configured: list[int] = []
        monkeypatch.setattr(torch, "set_num_threads", configured.append)
        monkeypatch.setattr(torch, "get_num_threads", lambda: 3)
        monkeypatch.setattr(
            "ai.rag.embedding.kure_embedder.SentenceTransformer",
            FakeSentenceTransformer,
        )

        KUREEmbedder(EmbeddingConfig(model_name="fake-model", cpu_threads=3))

        assert configured == [3]
