"""Rebuild a damaged external-evidence HNSW segment from Chroma's SQLite metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ai.rag.embedding.kure_embedder import KUREEmbedder

DEFAULT_CHROMA_PATH = REPOSITORY_ROOT / "chroma_db"
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data" / "rag007" / "public_ai_verified_sources_20260729.json"
SOURCE_COLLECTION = "external_market_policy_evidence"
TARGET_COLLECTION = "external_market_policy_evidence_rebuilt_logbacked_20260730"
BACKUP_COLLECTION = "external_market_policy_evidence_damaged_sync1000_20260730"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _metadata_value(row: sqlite3.Row) -> Any:
    if row["string_value"] is not None:
        return row["string_value"]
    if row["int_value"] is not None:
        return row["int_value"]
    if row["float_value"] is not None:
        return row["float_value"]
    if row["bool_value"] is not None:
        return bool(row["bool_value"])
    return None


def _extract_records(chroma_path: Path) -> list[dict[str, Any]]:
    database_path = (chroma_path / "chroma.sqlite3").resolve()
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        collection = connection.execute(
            "SELECT id FROM collections WHERE name = ?",
            (SOURCE_COLLECTION,),
        ).fetchone()
        if collection is None:
            raise RuntimeError(f"collection not found: {SOURCE_COLLECTION}")
        metadata_segment = connection.execute(
            """
            SELECT id FROM segments
            WHERE collection = ? AND scope = 'METADATA'
            """,
            (collection["id"],),
        ).fetchone()
        if metadata_segment is None:
            raise RuntimeError("metadata segment not found")

        grouped: dict[int, dict[str, Any]] = defaultdict(dict)
        record_ids: dict[int, str] = {}
        rows = connection.execute(
            """
            SELECT e.id, e.embedding_id, m.key, m.string_value, m.int_value,
                   m.float_value, m.bool_value
            FROM embeddings AS e
            LEFT JOIN embedding_metadata AS m ON m.id = e.id
            WHERE e.segment_id = ?
            ORDER BY e.id
            """,
            (metadata_segment["id"],),
        )
        for row in rows:
            record_ids[row["id"]] = row["embedding_id"]
            if row["key"] is not None:
                grouped[row["id"]][row["key"]] = _metadata_value(row)
    finally:
        connection.close()

    records: list[dict[str, Any]] = []
    for internal_id, values in grouped.items():
        document = str(values.pop("chroma:document", "") or "").strip()
        if not document:
            continue
        records.append(
            {
                "id": record_ids[internal_id],
                "document": document,
                "metadata": {key: value for key, value in values.items() if value is not None},
            }
        )
    return records


def _append_verified_fallback(records: list[dict[str, Any]], manifest_path: Path) -> None:
    existing_ids = {record["id"] for record in records}
    sources = json.loads(manifest_path.read_text(encoding="utf-8"))
    for source in sources:
        content = str(source.get("fallback_content") or "").strip()
        if not content:
            continue
        digest = hashlib.sha256(
            (source["source_url"] + "#verified-web-excerpt").encode("utf-8")
        ).hexdigest()[:16]
        document_id = f"{source['source_id']}:{digest}"
        chunk_id = f"{document_id}:excerpt:0"
        record_id = f"{source['source_id']}::{document_id}::{chunk_id}"
        if record_id in existing_ids:
            continue
        records.append(
            {
                "id": record_id,
                "document": content,
                "metadata": {
                    "source_id": source["source_id"],
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "title": source["title"],
                    "evidence_type": source["evidence_type"],
                    "publisher": source["publisher"],
                    "source_url": source["source_url"],
                    "domain": "competition",
                    "evaluation_criteria": json.dumps(
                        source["evaluation_criteria"], ensure_ascii=False
                    ),
                    "supported_roles": json.dumps(
                        source["supported_roles"], ensure_ascii=False
                    ),
                    "reference_date": source["published_at"],
                    "published_at": source["published_at"],
                    "retrieved_at": "2026-07-30",
                    "region": "대한민국",
                    "section": "공식 상세 페이지 소개",
                    "source_kind": "official_page_summary",
                    "source_type": "official_page_summary",
                    "summary_only": True,
                    "allow_grounded_claim": False,
                    "collection_error": "ConnectTimeout",
                },
            }
        )


def main() -> int:
    args = parse_args()
    chroma_path = args.chroma_path.resolve()
    records = _extract_records(chroma_path)
    _append_verified_fallback(records, args.manifest.resolve())
    if not records:
        raise RuntimeError("rebuild records are empty")
    print(f"recovery_records={len(records)}", flush=True)

    embedder = KUREEmbedder()
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(TARGET_COLLECTION)
    except Exception:
        pass
    target = client.create_collection(
        TARGET_COLLECTION,
        configuration={
            "hnsw": {
                "space": "cosine",
                # Chroma 1.5.9 on this Windows workspace failed while persisting
                # exactly the default 1,000-record threshold. Keep this local
                # development collection log-backed until it moves to the
                # server-backed Chroma deployment.
                "sync_threshold": 100_000,
                "batch_size": 100,
            }
        },
        metadata={
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.embedding_dimension,
            "embedding_version": "embedding_v1",
            "schema_version": "external_research_v1",
        },
    )

    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        texts = [record["document"] for record in batch]
        vectors = embedder._model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=embedder._config.normalize_embeddings,
            show_progress_bar=False,
        ).tolist()
        target.upsert(
            ids=[record["id"] for record in batch],
            documents=texts,
            metadatas=[record["metadata"] for record in batch],
            embeddings=vectors,
        )
        print(f"rebuilt={min(start + len(batch), len(records))}/{len(records)}", flush=True)

    rebuilt_count = target.count()
    if rebuilt_count != len(records):
        raise RuntimeError(f"count mismatch: expected={len(records)} actual={rebuilt_count}")

    source = client.get_collection(SOURCE_COLLECTION)
    try:
        client.delete_collection(BACKUP_COLLECTION)
    except Exception:
        pass
    source.modify(name=BACKUP_COLLECTION)
    target.modify(name=SOURCE_COLLECTION)

    active = client.get_collection(SOURCE_COLLECTION)
    print(
        json.dumps(
            {
                "collection": SOURCE_COLLECTION,
                "count": active.count(),
                "damaged_backup_collection": BACKUP_COLLECTION,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
