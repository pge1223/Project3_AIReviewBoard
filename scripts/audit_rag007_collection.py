"""Audit RAG-007 records for duplication, missing metadata, and likely boilerplate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import chromadb

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.app.config import settings

# 용준/Claude(2026-07-30, 요청: RAG-007 색인 청크 품질 정제) — 반복 머리글/바닥글
# 판정 로직을 ai/rag/preprocessing/boilerplate_detection.py로 추출해 색인 스크립트
# (scripts/index_rag007_verified_sources.py)와 동일한 알고리즘을 재사용한다. 감사와
# 색인이 서로 다른 로직을 쓰면 "감사에서는 잡히는데 색인에서는 안 걸러지는"
# 불일치가 생기기 때문이다.
from ai.rag.preprocessing import extract_boundary_lines, find_repeated_boundary_texts

DEFAULT_MANIFEST = REPOSITORY_ROOT / "data" / "rag007" / "public_ai_verified_sources_20260729.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reports" / "rag007_e2e_20260730" / "index_quality.json"
PRODUCTION_COLLECTION_NAME = "external_market_policy_evidence"


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _text_hash(value: str) -> str:
    return hashlib.sha256(_normalized_text(value).encode("utf-8")).hexdigest()


def _duplicate_group_count(groups: dict[str, set[str]]) -> tuple[int, int]:
    duplicates = [values for values in groups.values() if len(values) > 1]
    return len(duplicates), sum(len(values) for values in duplicates)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--collection",
        default=PRODUCTION_COLLECTION_NAME,
        help="감사할 Chroma 컬렉션 이름 (staging 컬렉션과 비교하려면 이 값을 바꿔서 두 번 실행).",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    new_source_ids = {item["source_id"] for item in manifest}

    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collection = client.get_collection(args.collection)
    raw = collection.get(include=["documents", "metadatas"])
    rows = [
        {"record_id": record_id, "document": document or "", "metadata": metadata or {}}
        for record_id, document, metadata in zip(raw["ids"], raw["documents"], raw["metadatas"])
    ]

    source_ids = {row["metadata"].get("source_id") for row in rows if row["metadata"].get("source_id")}
    existing_source_ids = source_ids - new_source_ids

    document_sources: dict[str, set[str]] = defaultdict(set)
    file_hash_documents: dict[str, set[str]] = defaultdict(set)
    content_hash_records: dict[str, set[str]] = defaultdict(set)
    content_hash_sources: dict[str, set[str]] = defaultdict(set)
    source_url_sources: dict[str, set[str]] = defaultdict(set)
    source_kind_counts: Counter[str] = Counter()

    empty_count = 0
    short_count = 0
    missing_source_url = 0
    missing_page = 0
    missing_section = 0
    missing_page_and_section = 0
    missing_file_hash = 0

    row_boundaries: dict[str, set[str]] = {}
    boundary_items: list[tuple[str, str]] = []

    for row in rows:
        metadata = row["metadata"]
        record_id = row["record_id"]
        document_id = str(metadata.get("document_id") or "")
        source_id = str(metadata.get("source_id") or "")
        source_url = str(metadata.get("source_url") or "").strip()
        content = _normalized_text(row["document"])

        if document_id:
            document_sources[document_id].add(source_id)
        file_hash = str(metadata.get("file_hash") or metadata.get("sha256") or "").strip()
        if file_hash:
            file_hash_documents[file_hash].add(document_id)
        else:
            missing_file_hash += 1
        if content:
            digest = _text_hash(content)
            content_hash_records[digest].add(record_id)
            content_hash_sources[digest].add(source_id)
        else:
            empty_count += 1
        if content and len(content) < 100:
            short_count += 1
        if not source_url:
            missing_source_url += 1
        elif source_id:
            source_url_sources[source_url].add(source_id)

        page_missing = metadata.get("page") in (None, "")
        section_missing = not str(metadata.get("section") or "").strip()
        missing_page += int(page_missing)
        missing_section += int(section_missing)
        missing_page_and_section += int(page_missing and section_missing)
        source_kind_counts[str(metadata.get("source_kind") or "unknown")] += 1

        boundaries = extract_boundary_lines(row["document"] or "")
        row_boundaries[record_id] = boundaries
        boundary_items.append((document_id, row["document"] or ""))

    repeated_boundary_lines = find_repeated_boundary_texts(boundary_items, min_group_count=3)
    suspected_boilerplate_rows = [
        row["record_id"]
        for row in rows
        if row_boundaries.get(row["record_id"], set()) & repeated_boundary_lines
    ]

    duplicate_document_ids = {
        key: sorted(values) for key, values in document_sources.items() if len(values) > 1
    }
    duplicate_file_hashes = {
        key: sorted(values) for key, values in file_hash_documents.items() if len(values) > 1
    }
    duplicate_chunk_groups = {
        key: sorted(values) for key, values in content_hash_records.items() if len(values) > 1
    }

    cross_content_hashes = [
        digest
        for digest, sources in content_hash_sources.items()
        if sources & existing_source_ids and sources & new_source_ids
    ]
    cross_source_urls = [
        url
        for url, sources in source_url_sources.items()
        if sources & existing_source_ids and sources & new_source_ids
    ]
    cross_document_ids = [
        document_id
        for document_id, sources in document_sources.items()
        if sources & existing_source_ids and sources & new_source_ids
    ]

    duplicate_chunk_group_count, duplicate_chunk_record_count = _duplicate_group_count(content_hash_records)
    report: dict[str, Any] = {
        "collection": args.collection,
        "record_count": len(rows),
        "source_count": len(source_ids),
        "new_manifest_source_count": len(new_source_ids),
        "existing_source_count": len(existing_source_ids),
        "duplicate_document_id_across_sources_count": len(duplicate_document_ids),
        "duplicate_document_id_across_sources_samples": dict(list(duplicate_document_ids.items())[:10]),
        "duplicate_file_hash_count": len(duplicate_file_hashes),
        "duplicate_file_hash_record_count": sum(len(values) for values in duplicate_file_hashes.values()),
        "missing_file_hash_count": missing_file_hash,
        "duplicate_chunk_group_count": duplicate_chunk_group_count,
        "duplicate_chunk_record_count": duplicate_chunk_record_count,
        "duplicate_chunk_samples": dict(list(duplicate_chunk_groups.items())[:10]),
        "empty_chunk_count": empty_count,
        "under_100_chars_chunk_count": short_count,
        "suspected_repeated_header_footer_chunk_count": len(suspected_boilerplate_rows),
        "suspected_repeated_header_footer_samples": suspected_boilerplate_rows[:20],
        "missing_source_url_chunk_count": missing_source_url,
        "missing_page_chunk_count": missing_page,
        "missing_section_chunk_count": missing_section,
        "missing_page_and_section_chunk_count": missing_page_and_section,
        "existing_vs_new_overlap": {
            "document_id_count": len(cross_document_ids),
            "source_url_count": len(cross_source_urls),
            "exact_chunk_hash_count": len(cross_content_hashes),
            "document_ids": cross_document_ids[:20],
            "source_urls": cross_source_urls[:20],
        },
        "source_kind_counts": dict(source_kind_counts),
        "summary_only_records": [
            {
                "record_id": row["record_id"],
                "source_id": row["metadata"].get("source_id"),
                "source_type": row["metadata"].get("source_type"),
                "summary_only": row["metadata"].get("summary_only"),
                "allow_grounded_claim": row["metadata"].get("allow_grounded_claim"),
            }
            for row in rows
            if row["metadata"].get("summary_only")
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
