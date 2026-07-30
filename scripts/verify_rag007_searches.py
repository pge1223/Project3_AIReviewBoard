"""Run the four requested RAG-007 searches against a Chroma collection.

용준/Claude(2026-07-30, 요청: RAG-007 색인 청크 품질 정제) — before/after 비교를
위해 --collection/--output을 인자화했다. 기본값은 기존과 동일(운영 컬렉션, 기존
출력 경로)이라 인자 없이 실행하면 하위 호환이다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.rag.embedding import KUREEmbedder
from ai.rag.external_research import (
    DatasetProvider,
    ExternalEvidenceRepository,
    ExternalResearchConfig,
    ExternalResearchRequest,
    ExternalResearchService,
)
from backend.app.config import settings

DEFAULT_OUTPUT = ROOT / "reports" / "rag007_e2e_20260730" / "search_top5.json"
PRODUCTION_COLLECTION_NAME = "external_market_policy_evidence"
QUERIES = [
    ("planning", "공공기관 AI 서비스의 개인정보 보호 위험"),
    ("planning", "고령자와 장애인의 디지털 접근성 문제"),
    ("technology", "공공부문 AI 도입 시 필요한 안전성과 윤리 기준"),
    ("planning", "공공 AI 서비스의 실증 및 확산 사례"),
]


def parse_args() -> argparse.Namespace:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=PRODUCTION_COLLECTION_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExternalResearchConfig(
        collection_name=args.collection,
        enable_public_api_search=False,
        min_similarity_score=0.0,
        default_top_k=5,
    )
    embedder = KUREEmbedder()
    repository = ExternalEvidenceRepository(
        client=chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR),
        collection_name=config.collection_name,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.embedding_dimension,
        embedding_version="embedding_v1",
    )
    service = ExternalResearchService(
        DatasetProvider(repository, embedder, config=config),
        config=config,
        embedder=embedder,
    )
    output = []
    for index, (role, query) in enumerate(QUERIES, start=1):
        response = service.search(
            ExternalResearchRequest(
                domain="competition",
                evaluation_criteria=[query],
                reviewer_role=role,
                query_context=query,
                top_k=5,
                min_score=0.0,
                trace_id=f"rag007-e2e-search-{index}",
            )
        )
        output.append(
            {
                "query": query,
                "reviewer_role": role,
                "results": [
                    {
                        "rank": rank,
                        "title": item.title,
                        "publisher": item.publisher,
                        "source_type": item.source_type or item.evidence_type.value,
                        "score": round(item.final_score, 6),
                        "semantic_score": round(item.semantic_score, 6),
                        "page": item.page,
                        "section": item.section,
                        "page_start": item.page_start,
                        "page_end": item.page_end,
                        "content_level": item.content_level,
                        "source_url": item.source_url,
                        "quote": item.quote[:500],
                        "chunk_id": item.chunk_id,
                        "summary_only": item.summary_only,
                        "allow_grounded_claim": item.allow_grounded_claim,
                    }
                    for rank, item in enumerate(response.results, start=1)
                ],
                "warnings": response.warnings,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
