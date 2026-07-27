"""
RAG Indexing Service (Integration Layer)
===========================================
백엔드가 KUREEmbedder/ChromaVectorStore를 직접 조립하지 않아도 되도록 묶어주는
단일 진입점. LangGraph/FastAPI/MongoDB와 무관하게 단독 호출 가능하다.
"""

import logging
import threading
import time
from typing import Optional

from ai.rag.chunking.schemas import ChunkingResult
from ai.rag.domain.schemas import IndexingContext
from ai.rag.embedding.kure_embedder import KUREEmbedder
from ai.rag.retrieval.chroma_store import ChromaVectorStore
from ai.rag.retrieval.diagnostics import StageWatchdog
from ai.rag.retrieval.exceptions import EmbeddingStageError, VectorStoreStageError
from ai.rag.retrieval.schemas import IndexingResult, IndexingSummary, SearchResult

logger = logging.getLogger(__name__)

# 2026-07-18, fetch-url 색인 hang 조사(용준/Claude): 이 임계값을 넘기면 StageWatchdog가
# 그 시점의 전체 스레드 스택을 로그에 남긴다(요청 자체를 취소/타임아웃시키지는 않음).
_STAGE_STUCK_THRESHOLD_SECONDS = 30.0


class RAGIndexingService:
    def __init__(self, embedder: KUREEmbedder, vector_store: ChromaVectorStore):
        self._embedder = embedder
        self._vector_store = vector_store
        # 한 서비스가 공유하는 PersistentClient에 여러 background thread가 동시에
        # upsert/list/delete 트랜잭션을 수행하지 않도록 문서 단위 색인을 직렬화한다.
        # 파싱·HWP 변환은 이 서비스 호출 전에 끝나므로 해당 작업들은 계속 병렬로 돈다.
        self._index_lock = threading.Lock()

    @property
    def embedder(self) -> KUREEmbedder:
        """documents.py의 싱글턴 KUREEmbedder를 다른 모듈(예: meetings.py)이 재사용할 수
        있도록 노출한다 — KUREEmbedder(SentenceTransformer 로딩 비용 큼)를 프로세스당
        한 번만 로딩하기 위함(2026-07-18, 중복 로딩 제거 조사 참고)."""
        return self._embedder

    @property
    def vector_store(self) -> ChromaVectorStore:
        """documents.py의 싱글턴 ChromaVectorStore(및 그 내부 chromadb client)를 다른
        모듈이 재사용할 수 있도록 노출한다 — 같은 CHROMA_PERSIST_DIR을 가리키는 별도의
        chromadb.PersistentClient를 프로세스 안에 여러 개 만들지 않기 위함."""
        return self._vector_store

    def index_chunking_result(
        self,
        chunking_result: ChunkingResult,
        context: IndexingContext,
    ) -> IndexingResult:
        wait_started = time.monotonic()
        with self._index_lock:
            waited_ms = (time.monotonic() - wait_started) * 1000
            if waited_ms >= 1:
                logger.info(
                    "rag.indexing.lock_acquired document_id=%s waited_ms=%.0f",
                    context.document_id,
                    waited_ms,
                )
            return self._index_chunking_result_locked(chunking_result, context)

    def _index_chunking_result_locked(
        self,
        chunking_result: ChunkingResult,
        context: IndexingContext,
    ) -> IndexingResult:
        """ChunkingResult -> 임베딩 -> Chroma upsert -> stale record 삭제 -> IndexingResult.

        각 단계(embed/chroma_upsert) 시작·종료를 elapsed_ms와 함께 로그로 남기고,
        30초 넘게 멈추면 StageWatchdog가 스레드 스택을 덤프한다. 단계별 실패는
        EmbeddingStageError/VectorStoreStageError로 감싸 다시 던져(raise ... from exc)
        호출부가 어느 단계에서 실패했는지 구분할 수 있게 한다."""
        document_id = context.document_id
        project_id = context.project_id
        logger.info(
            "rag.indexing.start document_id=%s project_id=%s chunk_count=%d",
            document_id, project_id, chunking_result.chunk_count,
        )

        t0 = time.monotonic()
        try:
            with StageWatchdog("embed", document_id, _STAGE_STUCK_THRESHOLD_SECONDS):
                embedding_result = self._embedder.embed_chunking_result(chunking_result, context)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.exception(
                "rag.indexing.embed_failed document_id=%s elapsed_ms=%.0f", document_id, elapsed_ms
            )
            raise EmbeddingStageError("embed", document_id, str(exc)) from exc
        embed_elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "rag.indexing.embed_done document_id=%s elapsed_ms=%.0f embedded_count=%d failed_count=%d",
            document_id, embed_elapsed_ms, embedding_result.embedding_count, len(embedding_result.failed_chunk_ids),
        )

        t1 = time.monotonic()
        try:
            with StageWatchdog("chroma_upsert", document_id, _STAGE_STUCK_THRESHOLD_SECONDS):
                result = self._vector_store.upsert_embedding_result(embedding_result, context)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t1) * 1000
            logger.exception(
                "rag.indexing.store_failed document_id=%s elapsed_ms=%.0f", document_id, elapsed_ms
            )
            raise VectorStoreStageError("chroma_upsert", document_id, str(exc)) from exc
        store_elapsed_ms = (time.monotonic() - t1) * 1000
        logger.info(
            "rag.indexing.store_done document_id=%s elapsed_ms=%.0f status=%s stored_count=%d",
            document_id, store_elapsed_ms, result.status, result.stored_record_count,
        )
        return result

    def index_chunking_result_with_summary(
        self,
        chunking_result: ChunkingResult,
        context: IndexingContext,
    ) -> IndexingSummary:
        """index_chunking_result()와 동일하게 색인하되, 벡터/embedding_text를 뺀
        프런트엔드 전달용 요약(IndexingSummary)을 반환한다."""
        indexable_chunk_count = sum(1 for c in chunking_result.chunks if c.indexable)
        indexing_result = self.index_chunking_result(chunking_result, context)
        return IndexingSummary(
            project_id=indexing_result.project_id,
            document_id=indexing_result.document_id,
            status=indexing_result.status,
            chunk_count=chunking_result.chunk_count,
            indexable_chunk_count=indexable_chunk_count,
            embedding_count=indexing_result.embedded_count,
            stored_count=indexing_result.stored_record_count,
            skipped_count=indexing_result.skipped_count,
            failed_count=indexing_result.failed_count,
            embedding_model=self._embedder.model_name,
            embedding_dimension=self._embedder.embedding_dimension,
            collection_name=indexing_result.collection_name,
            warnings=indexing_result.warnings,
        )

    def search(
        self,
        query: str,
        project_id: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """질문 문자열 -> KURE-v1 query 임베딩 -> project_id 필터 검색 -> SearchResult."""
        query_embedding = self._embedder.embed_query(query)
        return self._vector_store.search(
            query_embedding=query_embedding,
            project_id=project_id,
            document_id=document_id,
            top_k=top_k,
        )

    def delete_project(self, project_id: str) -> int:
        """project_id 기준으로 프로젝트 전체 벡터 청크를 삭제한다. ChromaVectorStore.delete_project()를
        그대로 호출하고 삭제된 건수를 그대로 반환한다(PRJ-004 프로젝트 삭제 연동, backend는
        run_in_threadpool로 감싸 호출한다 — 이 메서드 자체는 동기다)."""
        return self._vector_store.delete_project(project_id)
