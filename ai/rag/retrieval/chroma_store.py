"""
Chroma Vector Store
=====================
EmbeddingResult를 Chroma 컬렉션에 저장/삭제/검색한다. project_id 필터는 모든 검색에서
강제되며, chunk_id는 project_id와 결합한 별도의 Chroma record ID로 변환해 프로젝트 간
충돌을 막는다.

client는 생성자에서 주입받는다 — 로컬 개발은 chromadb.PersistentClient(path=...),
NCP 배포는 chromadb.HttpClient(host=..., port=...)를 그대로 넘기면 되고, 이 클래스는
어느 쪽이든 chromadb.ClientAPI 인터페이스로만 사용한다 (특정 client 생성 방식에 결합되지 않음).
"""

import logging
import time
from typing import Optional

import chromadb

from ai.rag.domain.schemas import CollectionConfigMismatchError, IndexingContext, InvalidTopKError
from ai.rag.embedding.schemas import EmbeddingResult
from ai.rag.retrieval.config import DISTANCE_METRIC, RETRIEVAL_SCHEMA_VERSION, DEFAULT_TOP_K
from ai.rag.retrieval.metadata import sanitize_metadata_for_chroma, restore_metadata
from ai.rag.retrieval.schemas import IndexingResult, IndexingStatus, SearchResult

logger = logging.getLogger(__name__)


def build_record_id(project_id: str, chunk_id: str) -> str:
    """Chroma record ID = project_id::chunk_id. chunk_id 자체엔 project_id가 포함되지 않으므로
    (ai.rag.chunking._generate_chunk_id 참고), 서로 다른 프로젝트의 동일 chunk_id 충돌을 막는다."""
    return f"{project_id}::{chunk_id}"


class ChromaVectorStore:
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
    def client(self) -> chromadb.ClientAPI:
        """이 ChromaVectorStore가 쓰는 chromadb client를 노출한다 — 같은
        CHROMA_PERSIST_DIR을 가리키는 별도의 chromadb.PersistentClient를 프로세스 안에
        중복 생성하지 않고 재사용하기 위함(2026-07-18, documents.py/meetings.py 중복
        PersistentClient+KUREEmbedder 조사 참고)."""
        return self._client

    @property
    def collection_name(self) -> str:
        """Return the active collection name for diagnostics and recovery checks."""
        return self._collection_name

    def _get_or_create_collection(self):
        collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={
                "hnsw:space": DISTANCE_METRIC,
                "embedding_model": self._embedding_model,
                "embedding_dimension": self._embedding_dimension,
                "embedding_version": self._embedding_version,
                "distance_metric": DISTANCE_METRIC,
                "schema_version": RETRIEVAL_SCHEMA_VERSION,
            },
        )
        # get_or_create_collection은 이미 존재하는 컬렉션이면 metadata 인자를 무시하고
        # 기존 메타데이터를 그대로 유지한다 (실제 chromadb 1.5.9로 확인) — 즉 여기서 값이
        # 다르더라도 조용히 덮어써지지 않으므로, 우리가 직접 비교해서 명시적으로 막아야 한다.
        existing = collection.metadata or {}
        for key, current in (
            ("embedding_model", self._embedding_model),
            ("embedding_dimension", self._embedding_dimension),
            ("embedding_version", self._embedding_version),
        ):
            existing_value = existing.get(key)
            if existing_value is not None and existing_value != current:
                raise CollectionConfigMismatchError(
                    f"컬렉션 '{self._collection_name}'의 기존 {key}('{existing_value}')가 "
                    f"현재 설정('{current}')과 달라 같은 컬렉션에 섞어 저장할 수 없습니다."
                )
        return collection

    def upsert_embedding_result(self, embedding_result: EmbeddingResult, context: IndexingContext) -> IndexingResult:
        """
        임베딩이 끝난 뒤에만 upsert하고, 그 다음에만 stale record(이전엔 있었지만 새
        청킹 결과엔 없는 chunk)를 지운다 — 임베딩 실패로 기존 정상 색인이 사라지는 일을 막기 위함.
        Chroma에는 여러 컬렉션에 걸친 트랜잭션이 없으므로, upsert 성공 후 삭제 사이에
        프로세스가 죽으면 stale record가 남을 수 있다 (재색인 시 다시 정리됨).
        """
        if context.project_id != embedding_result.project_id or context.document_id != embedding_result.document_id:
            raise ValueError("IndexingContext와 EmbeddingResult의 project_id/document_id가 일치하지 않습니다")
        if context.collection_name != self._collection_name:
            raise ValueError(
                f"IndexingContext.collection_name('{context.collection_name}')이 이 ChromaVectorStore의 "
                f"컬렉션('{self._collection_name}')과 다릅니다"
            )

        warnings = list(embedding_result.warnings)
        document_id = embedding_result.document_id

        t0 = time.monotonic()
        previous_ids = self._list_record_ids(embedding_result.project_id, embedding_result.document_id)
        logger.info(
            "rag.chroma.list_previous_ids_done document_id=%s elapsed_ms=%.0f count=%d",
            document_id, (time.monotonic() - t0) * 1000, len(previous_ids),
        )

        new_record_ids: list[str] = []
        if embedding_result.embedded_chunks:
            ids, embeddings, documents, metadatas = [], [], [], []
            for chunk in embedding_result.embedded_chunks:
                record_id = build_record_id(embedding_result.project_id, chunk.chunk_id)
                ids.append(record_id)
                embeddings.append(chunk.embedding)
                documents.append(chunk.content)
                metadatas.append(sanitize_metadata_for_chroma(chunk.metadata))
            t1 = time.monotonic()
            self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            logger.info(
                "rag.chroma.upsert_done document_id=%s elapsed_ms=%.0f count=%d",
                document_id, (time.monotonic() - t1) * 1000, len(ids),
            )
            new_record_ids = ids

        stale_ids = [rid for rid in previous_ids if rid not in set(new_record_ids)]
        if stale_ids:
            t2 = time.monotonic()
            self._collection.delete(ids=stale_ids)
            logger.info(
                "rag.chroma.delete_stale_done document_id=%s elapsed_ms=%.0f count=%d",
                document_id, (time.monotonic() - t2) * 1000, len(stale_ids),
            )

        t3 = time.monotonic()
        stored_count = len(self._list_record_ids(embedding_result.project_id, embedding_result.document_id))
        logger.info(
            "rag.chroma.list_stored_ids_done document_id=%s elapsed_ms=%.0f count=%d",
            document_id, (time.monotonic() - t3) * 1000, stored_count,
        )

        embedded_count = embedding_result.embedding_count
        failed_count = len(embedding_result.failed_chunk_ids)
        skipped_count = len(embedding_result.skipped_chunk_ids)

        if embedded_count == 0:
            status = IndexingStatus.FAILED if failed_count > 0 else IndexingStatus.EMPTY
            if status == IndexingStatus.EMPTY:
                warnings.append("저장 대상 청크가 0개입니다 (indexable=True인 유효 청크 없음).")
        elif failed_count > 0:
            status = IndexingStatus.PARTIAL
        else:
            status = IndexingStatus.SUCCESS

        return IndexingResult(
            project_id=embedding_result.project_id,
            document_id=embedding_result.document_id,
            collection_name=self._collection_name,
            embedded_count=embedded_count,
            upserted_count=len(new_record_ids),
            deleted_stale_count=len(stale_ids),
            skipped_count=skipped_count,
            failed_count=failed_count,
            stored_record_count=stored_count,
            warnings=warnings,
            status=status,
        )

    def delete_document(self, project_id: str, document_id: str) -> int:
        """project_id+document_id 기준으로 문서 전체를 삭제하고 삭제된 건수를 반환한다."""
        ids = self._list_record_ids(project_id, document_id)
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def delete_project(self, project_id: str) -> int:
        """project_id 기준으로 프로젝트에 속한 모든 문서·청크를 삭제하고 삭제된 건수를
        반환한다. document_id를 넘기지 않아 project_id만 일치하면 전부 대상이 된다
        (delete_document와 동일하게, 삭제 전 대상 ID를 조회해 건수를 계산한 뒤 삭제한다).
        대상이 없으면 예외 없이 0을 반환한다."""
        ids = self._list_record_ids(project_id, None)
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def search(
        self,
        query_embedding: list[float],
        project_id: str,
        document_id: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise InvalidTopKError("top_k는 1 이상이어야 합니다")

        where = self._build_where(project_id, document_id)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        ids = raw["ids"][0] if raw["ids"] else []
        documents = raw["documents"][0] if raw.get("documents") else [None] * len(ids)
        metadatas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(ids)
        distances = raw["distances"][0] if raw.get("distances") else [None] * len(ids)

        for record_id, content, raw_metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = restore_metadata(raw_metadata or {})
            score = self._distance_to_score(distance)
            results.append(SearchResult(
                record_id=record_id,
                chunk_id=metadata.get("chunk_id", record_id),
                document_id=metadata.get("document_id", ""),
                content=content or "",
                distance=distance,
                score=score,
                metadata=metadata,
            ))
        return results

    def get_by_chunk_id(self, project_id: str, chunk_id: str) -> Optional[str]:
        """project_id+chunk_id로 저장된 청크 원문을 정확히 하나 가져온다(유사도 검색이
        아니라 ID 직접 조회). 재인/Claude(2026-07-21): "AI 피드백" 워크벤치가 위원이
        인용한 청크(evidence.chunk_id)의 원문을 그대로(요약/재인용 없이) 하이라이트로
        쓰기 위해 추가 - search()처럼 쿼리 임베딩이 필요 없어 AI 호출 없이 조회만 한다.
        없으면 None."""
        record_id = build_record_id(project_id, chunk_id)
        result = self._collection.get(ids=[record_id], include=["documents"])
        docs = result.get("documents") or []
        return docs[0] if docs else None

    @staticmethod
    def _distance_to_score(distance: Optional[float]) -> Optional[float]:
        """
        컬렉션을 hnsw:space='cosine'으로 생성했을 때 Chroma가 반환하는 distance는
        (1 - cosine_similarity)로 정의된다 (Chroma 공식 문서 기준). 이 정의가 성립하는
        경우에만 score = 1.0 - distance로 변환한다. distance가 없으면 score도 None.
        """
        if distance is None:
            return None
        return 1.0 - distance

    @staticmethod
    def _build_where(project_id: str, document_id: Optional[str]) -> dict:
        if document_id is None:
            return {"project_id": project_id}
        return {"$and": [{"project_id": project_id}, {"document_id": document_id}]}

    def _list_record_ids(self, project_id: str, document_id: Optional[str] = None) -> list[str]:
        where = self._build_where(project_id, document_id)
        result = self._collection.get(where=where, include=[])
        return list(result.get("ids", []))


def create_persistent_client(path: str) -> chromadb.ClientAPI:
    """로컬 개발용 편의 함수. NCP 배포 시에는 chromadb.HttpClient(host=..., port=...)를
    직접 만들어 ChromaVectorStore(client=...)에 넘기면 된다 (이 클래스는 client 구현체에
    의존하지 않음)."""
    return chromadb.PersistentClient(path=path)
