"""
Pydantic Schemas for Embedding
================================
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ai.rag.embedding.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_DEVICE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NORMALIZE_EMBEDDINGS,
    DEFAULT_SHOW_PROGRESS,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_TRUST_REMOTE_CODE,
    EMBEDDING_VERSION,
)


class EmbeddingConfig(BaseModel):
    """KUREEmbedder 생성 시 넘기는 설정. 모델은 인스턴스 생성 시 1회만 로딩된다."""

    model_name: str = DEFAULT_MODEL_NAME
    device: Literal["cpu", "cuda", "auto"] = DEFAULT_DEVICE
    batch_size: int = DEFAULT_BATCH_SIZE
    cpu_threads: Optional[int] = Field(
        default=None,
        description="CPU 추론에 사용할 PyTorch intra-op 스레드 수. None이면 PyTorch 기본값 사용",
    )
    normalize_embeddings: bool = DEFAULT_NORMALIZE_EMBEDDINGS
    show_progress: bool = DEFAULT_SHOW_PROGRESS
    model_cache_dir: Optional[str] = DEFAULT_MODEL_CACHE_DIR
    trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE
    embedding_version: str = EMBEDDING_VERSION

    @field_validator("batch_size")
    @classmethod
    def _batch_size_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size는 0보다 커야 합니다")
        return v

    @field_validator("cpu_threads")
    @classmethod
    def _cpu_threads_must_be_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("cpu_threads는 0보다 커야 합니다")
        return v


class EmbeddedChunk(BaseModel):
    """청크 1개의 임베딩 결과. Chroma 저장 여부와 무관하게 임베딩 단계의 산출물이다."""

    chunk_id: str
    document_id: str
    embedding: list[float]
    embedding_dimension: int
    embedding_text: str = Field(..., description="실제로 모델에 입력된 텍스트 (document_title + section_title + content 결합)")
    content: str = Field(..., description="청크 원문. Chroma document로는 이 값이 저장됨 (embedding_text 아님)")
    metadata: dict = Field(default_factory=dict, description="Chroma 저장용 원시 메타데이터 (아직 타입 정제 전)")


class EmbeddingResult(BaseModel):
    """KUREEmbedder.embed_chunking_result()의 반환 스키마"""

    project_id: str
    document_id: str
    embedded_chunks: list[EmbeddedChunk] = Field(default_factory=list)
    embedding_count: int
    skipped_chunk_ids: list[str] = Field(default_factory=list, description="indexable=False라서 건너뛴 chunk_id 목록")
    failed_chunk_ids: list[str] = Field(default_factory=list, description="content가 비어 있는 등 임베딩 실패한 chunk_id 목록")
    warnings: list[str] = Field(default_factory=list)
    model_name: str
    embedding_dimension: int
    normalized: bool
    embedding_version: str
