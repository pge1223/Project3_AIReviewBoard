"""
External Evidence Chroma Repository (RAG-007)
====================================================
외부 공개 통계·시장·정책 자료 전용 Chroma 컬렉션에 접근하는 저장소. 사용자 프로젝트
문서 컬렉션(ai.rag.retrieval.chroma_store.ChromaVectorStore, project_documents_kure_v1)
및 RAG-006 사례 컬렉션(ai.rag.similar_cases.repository.SimilarCaseRepository,
similar_success_cases)과 완전히 분리된 별도 컬렉션(기본 external_market_policy_evidence)을
쓴다. client(chromadb.ClientAPI)는 생성자 주입이며 특정 client 구현에 결합되지 않는다.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import chromadb

from ai.rag.external_research.exceptions import ExternalCollectionUnavailableError
from ai.rag.external_research.schemas import ExternalEvidenceDocument, ExternalEvidenceType
from ai.rag.retrieval.metadata import restore_metadata, sanitize_metadata_for_chroma

logger = logging.getLogger(__name__)

_DISTANCE_METRIC = "cosine"
_SCHEMA_VERSION = "external_research_v1"

# sanitize_metadata_for_chroma()는 str/int/float/bool이 아닌 값(list 포함)을 모두 JSON
# 문자열로 직렬화한다. ai.rag.retrieval.metadata.restore_metadata()는
# source_block_ids/source_block_orders만 복원하므로, 이 저장소가 쓰는 list 필드는
# 직접 복원한다(ai.rag.similar_cases.repository와 동일 패턴).
_JSON_LIST_FIELDS: frozenset[str] = frozenset({"evaluation_criteria", "supported_roles"})


def build_evidence_record_id(source_id: str, document_id: str, chunk_id: str) -> str:
    """외부자료 컬렉션의 Chroma record ID. 별도 컬렉션에 저장되므로 project_id/case_id
    네임스페이스와 충돌하지 않는다."""
    return f"{source_id}::{document_id}::{chunk_id}"


def _restore_evidence_metadata(raw_metadata: Optional[dict]) -> dict:
    restored = restore_metadata(raw_metadata or {})
    for key in _JSON_LIST_FIELDS:
        value = restored.get(key)
        if isinstance(value, str):
            try:
                restored[key] = json.loads(value)
            except json.JSONDecodeError:
                restored[key] = []
    return restored


@dataclass(frozen=True)
class ExternalEvidenceHit:
    """Chroma 검색 결과 1건(청크 단위, 아직 랭킹/집계되지 않음)."""

    record_id: str
    source_id: str
    document_id: str
    chunk_id: str
    content: str
    distance: Optional[float]
    score: Optional[float]
    metadata: dict = field(default_factory=dict)


class ExternalEvidenceRepository:
    """외부자료 전용 Chroma 컬렉션에 대한 upsert/search."""

    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        embedding_version: str,
    ):
        self._client = client
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._embedding_version = embedding_version
        self._collection = self._get_or_create_collection()

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def _get_or_create_collection(self):
        try:
            collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={
                    "hnsw:space": _DISTANCE_METRIC,
                    "embedding_model": self._embedding_model,
                    "embedding_dimension": self._embedding_dimension,
                    "embedding_version": self._embedding_version,
                    "schema_version": _SCHEMA_VERSION,
                },
            )
        except Exception as exc:  # pragma: no cover - chromadb 자체 연결 실패 등 방어적 처리
            raise ExternalCollectionUnavailableError(
                f"외부자료 전용 컬렉션 '{self._collection_name}'을 열 수 없습니다: {exc}"
            ) from exc

        existing = collection.metadata or {}
        for key, current in (
            ("embedding_model", self._embedding_model),
            ("embedding_dimension", self._embedding_dimension),
            ("embedding_version", self._embedding_version),
        ):
            existing_value = existing.get(key)
            if existing_value is not None and existing_value != current:
                raise ExternalCollectionUnavailableError(
                    f"컬렉션 '{self._collection_name}'의 기존 {key}('{existing_value}')가 "
                    f"현재 설정('{current}')과 달라 같은 컬렉션에 섞어 저장할 수 없습니다."
                )
        return collection

    def upsert_evidence_chunk(self, document: ExternalEvidenceDocument, embedding: list[float]) -> str:
        """외부자료 청크 1건을 upsert한다. 동일 (source_id, document_id, chunk_id)는 같은
        record를 덮어써 중복 색인되지 않는다."""
        record_id = build_evidence_record_id(document.source_id, document.document_id, document.chunk_id)
        raw_metadata: dict[str, Any] = {
            "source_id": document.source_id,
            "document_id": document.document_id,
            "chunk_id": document.chunk_id,
            "title": document.title,
            "evidence_type": document.evidence_type,
            "publisher": document.publisher,
            "source_url": document.source_url,
            "domain": document.domain,
            "evaluation_criteria": document.evaluation_criteria,
            "supported_roles": document.supported_roles,
            "reference_date": document.reference_date,
            "published_at": document.published_at,
            "retrieved_at": document.retrieved_at,
            "region": document.region,
            "period": document.period,
            "metric_name": document.metric_name,
            "metric_value": document.metric_value,
            "metric_unit": document.metric_unit,
            "page": document.page,
            "section": document.section,
            # 용준/Claude(2026-07-30, 요청: RAG-007 색인 청크 품질 정제) — 신규 1급
            # 필드도 metadata로 함께 저장한다(ExternalEvidenceDocument.metadata dict가
            # 아니라 스키마 필드에서 직접 읽어야 색인 스크립트가 값을 빠뜨려도 여기서
            # None으로라도 항상 채워진다).
            "file_hash": document.file_hash,
            "source_content_hash": document.source_content_hash,
            "chunk_content_hash": document.chunk_content_hash,
            "normalized_content_hash": document.normalized_content_hash,
            "page_start": document.page_start,
            "page_end": document.page_end,
            "content_level": document.content_level,
            "url_verified": document.url_verified,
            "direct_file_url": document.direct_file_url,
            "allow_grounded_claim": document.allow_grounded_claim,
        }
        for key, value in document.metadata.items():
            raw_metadata.setdefault(key, value)

        self._collection.upsert(
            ids=[record_id],
            embeddings=[embedding],
            documents=[document.content],
            metadatas=[sanitize_metadata_for_chroma(raw_metadata)],
        )
        return record_id

    def search(
        self,
        query_embedding: list[float],
        *,
        domain: Optional[str] = None,
        evidence_types: Optional[Sequence[ExternalEvidenceType]] = None,
        top_k: int,
    ) -> list[ExternalEvidenceHit]:
        """청크 단위 검색 결과를 반환한다(랭킹/집계는 search_service의 책임).
        domain/evidence_types가 주어지면 metadata 필터로 적용한다."""
        where = self._build_where(domain, evidence_types)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw["ids"][0] if raw.get("ids") else []
        documents = raw["documents"][0] if raw.get("documents") else [None] * len(ids)
        metadatas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(ids)
        distances = raw["distances"][0] if raw.get("distances") else [None] * len(ids)

        hits: list[ExternalEvidenceHit] = []
        for record_id, content, raw_metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = _restore_evidence_metadata(raw_metadata)
            score = self._distance_to_score(distance)
            hits.append(
                ExternalEvidenceHit(
                    record_id=record_id,
                    source_id=metadata.get("source_id", ""),
                    document_id=metadata.get("document_id", ""),
                    chunk_id=metadata.get("chunk_id", record_id),
                    content=content or "",
                    distance=distance,
                    score=score,
                    metadata=metadata,
                )
            )
        return hits

    @staticmethod
    def _build_where(domain: Optional[str], evidence_types: Optional[Sequence[ExternalEvidenceType]]) -> Optional[dict]:
        clauses: list[dict] = []
        if domain:
            clauses.append({"domain": domain})
        if evidence_types:
            clauses.append({"evidence_type": {"$in": [t.value for t in evidence_types]}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _distance_to_score(distance: Optional[float]) -> Optional[float]:
        """hnsw:space='cosine' 컬렉션에서 Chroma distance = 1 - cosine_similarity
        (ai.rag.retrieval.chroma_store.ChromaVectorStore._distance_to_score와 동일 정의)."""
        if distance is None:
            return None
        return 1.0 - distance

    def count(self) -> int:
        return self._collection.count()


__all__ = ["ExternalEvidenceRepository", "ExternalEvidenceHit", "build_evidence_record_id"]
