"""Index the curated official public-AI seed dataset into the RAG-007 collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ai.rag.embedding.kure_embedder import KUREEmbedder
from ai.rag.external_research import (
    ExternalEvidenceDocument,
    ExternalEvidenceIndexingService,
    ExternalEvidenceRepository,
    ExternalResearchConfig,
)

DEFAULT_DATASET = REPOSITORY_ROOT / "data" / "rag007" / "public_ai_official_seed_20260727.json"
DEFAULT_CHROMA_PATH = REPOSITORY_ROOT / "chroma_db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    chroma_path = args.chroma_path.resolve()

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    documents = [ExternalEvidenceDocument.model_validate(item) for item in payload]

    config = ExternalResearchConfig()
    embedder = KUREEmbedder()
    client = chromadb.PersistentClient(path=str(chroma_path))
    repository = ExternalEvidenceRepository(
        client=client,
        collection_name=config.collection_name,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.embedding_dimension,
        embedding_version="embedding_v1",
    )
    service = ExternalEvidenceIndexingService(repository, embedder)
    summary = service.index_evidence(documents, trace_id="official-public-ai-seed-20260727")

    print(summary.model_dump_json(indent=2))
    print(f"collection_count={repository._collection.count()}")
    return 0 if summary.skipped_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
