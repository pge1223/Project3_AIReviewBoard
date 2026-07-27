"""
KURE-v1 실제 모델 통합 테스트
================================
기본 pytest 실행에서는 수집되지 않도록 모듈 최상단에서 RUN_KURE_INTEGRATION 환경변수를
확인해 전체를 skip한다 (전체 회귀 테스트에서 모델 다운로드가 자동으로 시작되지 않게 함).

실행 (PowerShell, review-board conda env):
    $env:RUN_KURE_INTEGRATION="1"
    python -m pytest ai/rag/tests/test_kure_embedding_integration.py -v -m embedding_integration

첫 실행은 huggingface_hub가 nlpai-lab/KURE-v1을 다운로드하므로 몇 분 걸릴 수 있다
(기본 캐시 경로: ~/.cache/huggingface/hub/models--nlpai-lab--KURE-v1).
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.embedding_integration

if os.environ.get("RUN_KURE_INTEGRATION") != "1":
    pytest.skip("RUN_KURE_INTEGRATION=1일 때만 실행 (실제 KURE-v1 모델 다운로드/로딩)", allow_module_level=True)

from ai.rag.chunking.schemas import Chunk, ChunkingConfig, ChunkingResult, ChunkLocationType, ContentKind, SourceType
from ai.rag.domain import IndexingContext
from ai.rag.domain.config import DEFAULT_COLLECTION_NAME
from ai.rag.embedding.kure_embedder import KUREEmbedder
from ai.rag.embedding.schemas import EmbeddingConfig
from ai.rag.retrieval.chroma_store import ChromaVectorStore, create_persistent_client
from ai.rag.retrieval.service import RAGIndexingService


@pytest.fixture(scope="module")
def real_embedder() -> KUREEmbedder:
    return KUREEmbedder(EmbeddingConfig(device="cpu"))


class TestRealKUREEmbedding:
    def test_model_loads(self, real_embedder):
        assert real_embedder.model_name == "nlpai-lab/KURE-v1"
        assert real_embedder.embedding_dimension > 0

    def test_embed_two_korean_sentences(self, real_embedder):
        chunks = [
            Chunk(
                chunk_id=f"chk_{i}",
                document_id="doc-1",
                content=text,
                chunk_index=i,
                content_kind=ContentKind.BODY,
                source_type=SourceType.URL_WEBPAGE,
                location_type=ChunkLocationType.WEB_SECTION,
                location_number=None,
                section_title=None,
                char_count=len(text),
                indexable=True,
            )
            for i, text in enumerate(["안녕하세요, 첫 번째 문장입니다.", "이것은 두 번째 문장입니다."])
        ]
        chunking_result = ChunkingResult(document_id="doc-1", chunks=chunks, chunk_count=2, config=ChunkingConfig())
        result = real_embedder.embed_chunking_result(chunking_result, IndexingContext(project_id="p1", document_id="doc-1"))

        assert result.embedding_count == 2
        dims = {c.embedding_dimension for c in result.embedded_chunks}
        assert len(dims) == 1
        assert dims.pop() > 0
        for c in result.embedded_chunks:
            assert all(np.isfinite(v) for v in c.embedding)

    def test_normalize_embeddings_norm_is_one(self, real_embedder):
        vector = real_embedder.embed_query("정규화 확인용 질의")
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-2

    def test_query_and_document_embedding_same_dimension(self, real_embedder):
        query_vector = real_embedder.embed_query("접수기간은 언제인가요?")
        assert len(query_vector) == real_embedder.embedding_dimension


class TestRealKUREChromaSearch:
    def test_store_and_search_korean_query(self, real_embedder, tmp_path):
        client = create_persistent_client(path=str(tmp_path / "chroma_data"))
        store = ChromaVectorStore(
            client=client,
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_model=real_embedder.model_name,
            embedding_dimension=real_embedder.embedding_dimension,
            embedding_version="embedding_v1",
        )
        service = RAGIndexingService(real_embedder, store)

        chunks = [
            Chunk(
                chunk_id="chk_period", document_id="doc-1",
                content="접수기간은 2026년 3월 1일부터 3월 31일까지입니다.",
                chunk_index=0, content_kind=ContentKind.BODY, source_type=SourceType.URL_WEBPAGE,
                location_type=ChunkLocationType.WEB_SECTION, location_number=None,
                section_title="접수 안내", char_count=30, indexable=True,
            ),
            Chunk(
                chunk_id="chk_문의", document_id="doc-1",
                content="문의처 전화번호는 02-1234-5678입니다.",
                chunk_index=1, content_kind=ContentKind.BODY, source_type=SourceType.URL_WEBPAGE,
                location_type=ChunkLocationType.WEB_SECTION, location_number=None,
                section_title="문의처", char_count=20, indexable=True,
            ),
        ]
        chunking_result = ChunkingResult(document_id="doc-1", chunks=chunks, chunk_count=2, config=ChunkingConfig())

        indexing_result = service.index_chunking_result(chunking_result, IndexingContext(project_id="p1", document_id="doc-1"))
        assert indexing_result.stored_record_count == 2

        results = service.search("접수기간은 언제인가요?", project_id="p1", top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "chk_period"
        assert results[0].metadata.get("project_id") == "p1"
        assert results[0].content

        other_project_results = service.search("접수기간은 언제인가요?", project_id="p-other", top_k=5)
        assert other_project_results == []
