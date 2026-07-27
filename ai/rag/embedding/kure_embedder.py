"""
KURE-v1 Embedding Service
===========================
ChunkingResult(indexable=True 청크)를 KURE-v1로 임베딩한다. LangGraph/FastAPI/MongoDB와
무관하게 단독 실행 가능하며, 모델은 인스턴스 생성 시 한 번만 로딩되어 재사용된다.
"""

import logging
import math
import threading
import time
from typing import Optional

from sentence_transformers import SentenceTransformer

from ai.rag.chunking.schemas import Chunk, ChunkingResult
from ai.rag.domain.schemas import IndexingContext
from ai.rag.embedding.schemas import EmbeddingConfig, EmbeddedChunk, EmbeddingResult
from ai.rag.embedding.text_builder import build_embedding_text

logger = logging.getLogger(__name__)


class EmptyQueryError(ValueError):
    """embed_query()에 빈 문자열(또는 공백만 있는 문자열)을 전달했을 때 발생"""


class NonFiniteEmbeddingError(RuntimeError):
    """임베딩 벡터에 NaN 또는 Infinity가 포함되어 있을 때 발생"""


class KUREEmbedder:
    """KURE-v1 임베딩 서비스. 요청마다 모델을 다시 로딩하지 않고 재사용한다."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self._config = config or EmbeddingConfig()
        self._model = SentenceTransformer(
            self._config.model_name,
            device=self._resolve_device(self._config.device),
            cache_folder=self._config.model_cache_dir,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._dimension = self._model.get_embedding_dimension()
        # SentenceTransformer/PyTorch의 동일 모델 인스턴스를 여러 FastAPI background
        # thread가 동시에 encode하면 tokenizer/model 내부 상태가 충돌할 수 있다.
        # 파일 여러 개와 URL을 한 번에 등록하는 화면은 실제로 이 경로를 만든다.
        self._encode_lock = threading.RLock()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def embed_query(self, query: str) -> list[float]:
        """질의 문자열 1개를 임베딩한다. 빈 문자열(공백만 포함)은 거부한다."""
        if not query or not query.strip():
            raise EmptyQueryError("query는 빈 문자열일 수 없습니다")

        with self._encode_lock:
            vector = self._model.encode(
                [query],
                batch_size=1,
                normalize_embeddings=self._config.normalize_embeddings,
                show_progress_bar=False,
            )[0]
        self._validate_finite(vector)
        return vector.tolist()

    def embed_chunking_result(
        self,
        chunking_result: ChunkingResult,
        context: IndexingContext,
    ) -> EmbeddingResult:
        """
        indexable=True 청크만 선택해 배치 임베딩한다. chunking_result는 절대 mutate하지 않는다.
        """
        context.ensure_matches(chunking_result)

        warnings: list[str] = []
        skipped_chunk_ids = [c.chunk_id for c in chunking_result.chunks if not c.indexable]

        candidate_chunks = [c for c in chunking_result.chunks if c.indexable]

        failed_chunk_ids: list[str] = []
        embed_texts: list[str] = []
        valid_chunks: list[Chunk] = []
        for chunk in candidate_chunks:
            if not chunk.content or not chunk.content.strip():
                failed_chunk_ids.append(chunk.chunk_id)
                warnings.append(f"chunk_id={chunk.chunk_id}: content가 비어 있어 임베딩에서 제외했습니다.")
                continue
            embed_texts.append(build_embedding_text(chunk.content, context.document_title, chunk.section_title))
            valid_chunks.append(chunk)

        if not valid_chunks:
            if not chunking_result.chunks:
                warnings.append("ChunkingResult에 청크가 없습니다.")
            else:
                warnings.append("색인 대상 청크가 없습니다 (indexable=True이며 내용이 있는 청크 0개).")
            return EmbeddingResult(
                project_id=context.project_id,
                document_id=context.document_id,
                embedded_chunks=[],
                embedding_count=0,
                skipped_chunk_ids=skipped_chunk_ids,
                failed_chunk_ids=failed_chunk_ids,
                warnings=warnings,
                model_name=self.model_name,
                embedding_dimension=self._dimension,
                normalized=self._config.normalize_embeddings,
                embedding_version=self._config.embedding_version,
            )

        total_chars = sum(len(t) for t in embed_texts)
        max_chars = max((len(t) for t in embed_texts), default=0)
        logger.info(
            "rag.embed.encode_start document_id=%s text_count=%d total_chars=%d max_chars=%d batch_size=%d",
            context.document_id, len(embed_texts), total_chars, max_chars, self._config.batch_size,
        )
        t0 = time.monotonic()
        with self._encode_lock:
            vectors = self._model.encode(
                embed_texts,
                batch_size=self._config.batch_size,
                normalize_embeddings=self._config.normalize_embeddings,
                show_progress_bar=self._config.show_progress,
            )
        logger.info(
            "rag.embed.encode_done document_id=%s elapsed_ms=%.0f",
            context.document_id, (time.monotonic() - t0) * 1000,
        )
        self._validate_finite(vectors)

        embedded_chunks: list[EmbeddedChunk] = []
        for chunk, embed_text, vector in zip(valid_chunks, embed_texts, vectors):
            vector_list = vector.tolist()
            if len(vector_list) != self._dimension:
                raise RuntimeError(
                    f"chunk_id={chunk.chunk_id}: 벡터 차원({len(vector_list)})이 모델 차원({self._dimension})과 다릅니다"
                )
            embedded_chunks.append(EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                embedding=vector_list,
                embedding_dimension=len(vector_list),
                embedding_text=embed_text,
                content=chunk.content,
                metadata=_build_raw_metadata(
                    chunk=chunk,
                    context=context,
                    embedding_model=self.model_name,
                    embedding_version=self._config.embedding_version,
                ),
            ))

        return EmbeddingResult(
            project_id=context.project_id,
            document_id=context.document_id,
            embedded_chunks=embedded_chunks,
            embedding_count=len(embedded_chunks),
            skipped_chunk_ids=skipped_chunk_ids,
            failed_chunk_ids=failed_chunk_ids,
            warnings=warnings,
            model_name=self.model_name,
            embedding_dimension=self._dimension,
            normalized=self._config.normalize_embeddings,
            embedding_version=self._config.embedding_version,
        )

    @staticmethod
    def _validate_finite(vectors) -> None:
        flat = vectors.reshape(-1) if vectors.ndim > 1 else vectors
        for value in flat:
            if math.isnan(value) or math.isinf(value):
                raise NonFiniteEmbeddingError("임베딩 벡터에 NaN 또는 Infinity가 포함되어 있습니다")


def _build_raw_metadata(
    *,
    chunk: Chunk,
    context: IndexingContext,
    embedding_model: str,
    embedding_version: str,
) -> dict:
    """Chroma 저장 전 단계의 원시 메타데이터. 타입 정제(enum→value 등)는 ai.rag.retrieval.metadata에서 한다."""
    return {
        "project_id": context.project_id,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "source_type": chunk.source_type,
        "source_url": chunk.source_url,
        "source_page_url": chunk.source_page_url,
        "source_filename": chunk.source_filename,
        "file_type": chunk.file_type,
        "location_type": chunk.location_type,
        "location_number": chunk.location_number,
        "section_title": chunk.section_title,
        "content_kind": chunk.content_kind,
        "source_block_ids": chunk.source_block_ids,
        "source_block_orders": chunk.source_block_orders,
        "chunking_version": chunk.chunking_version,
        "embedding_model": embedding_model,
        "embedding_version": embedding_version,
        "indexable": chunk.indexable,
        "document_title": context.document_title,
        "document_role": context.document_role,
        # 용준/Claude(2026-07-22, 요청: 선택된 아이디어/사용자 답변을 target evidence로 색인)
        # — IndexingContext.extra_metadata를 그대로 펼쳐 넣는다. 일반 파일/URL 업로드는 이
        # 필드를 쓰지 않으므로(항상 None) 기존 색인 경로는 전혀 영향받지 않는다. 여기서
        # 개별 ideation 전용 필드(ideation_source_type/session_id/candidate_id 등)를 하나씩
        # IndexingContext에 추가하지 않고 범용 통로 하나로 열어두는 이유: 이 값들은 오직
        # 아이디어 회의 target evidence 색인 호출자(ai/rag/orchestration/
        # ideation_target_indexing_service.py)만 채우고, 다른 모든 호출자(문서 업로드 등)는
        # 그대로 두어도 되는 부가 정보이기 때문이다.
        **(context.extra_metadata or {}),
    }
