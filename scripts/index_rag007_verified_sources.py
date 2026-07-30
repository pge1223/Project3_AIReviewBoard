"""Collect verified official URLs and index their parsed chunks into RAG-007.

용준/Claude(2026-07-30, 요청: RAG-007 색인 청크 품질 정제) — 기존 스크립트(원문 수집 →
HTML 정제 → 청킹 → 그대로 upsert)에 다음 단계를 추가했다:
  1. PDF 첨부파일은 청킹 전에 같은 문서 내 3페이지 이상 반복되는 머리글/바닥글을 제거한다.
  2. 청킹 후, 짧은 청크(<100자)는 인접 청크와 병합하거나(가능하면) 법령 조문/통계
     수치/정의문처럼 짧아도 의미 있는 패턴만 보존하고 그 외는 제거한다.
  3. 이번 실행에서 수집한 배치 전체를 대상으로, 3개 이상 서로 다른 document_id에
     반복 등장하는 문구(사이트 공통 메뉴 등)를 제거한다.
  4. 문서 단위(file_hash/source_content_hash) + 청크 단위(chunk_content_hash/
     normalized_content_hash) 해시를 계산하고, 배치 내 근사 중복 청크를 제거한다.
  5. page_start/page_end, content_level, url_verified, direct_file_url,
     allow_grounded_claim(page/section 둘 다 없으면 False) metadata를 채운다.

안전장치: 기존 운영 컬렉션(external_market_policy_evidence)에 실수로 덮어쓰지 않도록
기본 컬렉션명을 staging 이름으로 바꿨다. 운영 컬렉션 이름을 명시적으로 넘기려면
--i-know-this-is-production 플래그를 함께 줘야 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import chromadb

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ai.rag.chunking import ChunkSourceContext, SourceType, chunk_document
from ai.rag.embedding.kure_embedder import KUREEmbedder
from ai.rag.external_research import (
    ExternalEvidenceDocument,
    ExternalEvidenceIndexingService,
    ExternalEvidenceRepository,
    ExternalEvidenceType,
    ExternalResearchConfig,
)
from ai.rag.loaders import load_from_url
from ai.rag.preprocessing import (
    clean_page_content,
    compute_chunk_content_hash,
    compute_normalized_content_hash,
    compute_source_content_hash,
    deduplicate_chunks,
    find_repeated_boundary_texts,
    is_boilerplate_text,
    merge_or_filter_short_chunks,
)

DEFAULT_MANIFEST = REPOSITORY_ROOT / "data" / "rag007" / "public_ai_verified_sources_20260729.json"
DEFAULT_CHROMA_PATH = REPOSITORY_ROOT / "chroma_db"
PRODUCTION_COLLECTION_NAME = "external_market_policy_evidence"
DEFAULT_STAGING_COLLECTION_NAME = "external_market_policy_evidence_clean_v1"
# 문서 간(사이트 메뉴 등) 반복 판정 최소 document_id 수 — 감사 스크립트
# (scripts/audit_rag007_collection.py)와 동일 임계값을 쓴다.
_CROSS_DOCUMENT_MIN_GROUP_COUNT = 3
# 문서 내(PDF 머리글/바닥글) 반복 판정 최소 페이지 수.
_CROSS_PAGE_MIN_GROUP_COUNT = 3


def parse_args() -> argparse.Namespace:
    # Windows 콘솔(cp949 등)에서 --help 출력 중 유니코드 문자(em dash 등)가 있으면
    # UnicodeEncodeError로 죽는 문제를 막는다(다른 RAG-007 스크립트와 동일 조치).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--source-id", action="append", default=[], help="Index only the selected source ID")
    parser.add_argument(
        "--collection",
        default=DEFAULT_STAGING_COLLECTION_NAME,
        help=(
            "색인 대상 Chroma 컬렉션 이름. 기본값은 staging 컬렉션이다. 운영 컬렉션"
            f"('{PRODUCTION_COLLECTION_NAME}')을 지정하려면 --i-know-this-is-production도 함께 줘야 한다."
        ),
    )
    parser.add_argument(
        "--i-know-this-is-production",
        action="store_true",
        help="--collection이 운영 컬렉션 이름과 같을 때 실수 방지를 위해 명시적으로 필요한 확인 플래그.",
    )
    args = parser.parse_args()
    if args.collection == PRODUCTION_COLLECTION_NAME and not args.i_know_this_is_production:
        parser.error(
            f"--collection={PRODUCTION_COLLECTION_NAME}(운영 컬렉션)로 실행하려면 "
            "--i-know-this-is-production 플래그가 필요합니다. staging에서 먼저 검증하세요."
        )
    return args


def _document_id(source_id: str, discriminator: str) -> str:
    digest = hashlib.sha256(discriminator.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}:{digest}"


def _strip_pdf_repeated_boilerplate(extraction: Any) -> tuple[Any, int]:
    """같은 문서(extraction) 안에서 3페이지 이상 반복되는 머리글/바닥글 블록을
    제거한 사본을 반환한다. 원본 extraction은 수정하지 않는다(project 관례:
    ai/rag/preprocessing/html_cleaner.py도 입력을 mutate하지 않는다)."""
    page_texts: dict[str, list[str]] = {}
    for block in extraction.blocks:
        if block.location_number is None or not block.content:
            continue
        page_texts.setdefault(str(block.location_number), []).append(block.content)
    if len(page_texts) < _CROSS_PAGE_MIN_GROUP_COUNT:
        return extraction, 0

    page_items = [(page, "\n".join(texts)) for page, texts in page_texts.items()]
    repeated_lines = find_repeated_boundary_texts(page_items, min_group_count=_CROSS_PAGE_MIN_GROUP_COUNT)
    if not repeated_lines:
        return extraction, 0

    kept_blocks = [block for block in extraction.blocks if not is_boilerplate_text(block.content, repeated_lines)]
    removed_count = len(extraction.blocks) - len(kept_blocks)
    if removed_count == 0:
        return extraction, 0
    return extraction.model_copy(update={"blocks": kept_blocks, "block_count": len(kept_blocks)}), removed_count


def _to_chunk_records(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    """원문을 수집해 (아직 ExternalEvidenceDocument로 변환하지 않은) plain dict 청크
    레코드 목록을 반환한다 — 배치 전체를 모은 뒤에야 문서 간 반복 탐지/중복 제거를
    할 수 있으므로, 이 단계에서는 dict로만 다룬다."""
    evidence_type = ExternalEvidenceType(source["evidence_type"])
    retrieved_at = date.today().isoformat()
    records: list[dict[str, Any]] = []
    stats = {"pdf_boilerplate_blocks_removed": 0}

    try:
        result = load_from_url(source["source_url"])
    except Exception as exc:
        fallback_content = str(source.get("fallback_content") or "").strip()
        if not fallback_content:
            raise
        document_id = _document_id(source["source_id"], source["source_url"] + "#verified-web-excerpt")
        records.append(
            {
                "source_id": source["source_id"],
                "document_id": document_id,
                "chunk_id": f"{document_id}:excerpt:0",
                "title": source["title"],
                "evidence_type": evidence_type,
                "publisher": source["publisher"],
                "source_url": source["source_url"],
                "domain": "competition",
                "evaluation_criteria": source["evaluation_criteria"],
                "supported_roles": source["supported_roles"],
                "content": fallback_content,
                "reference_date": source["published_at"],
                "published_at": source["published_at"],
                "retrieved_at": retrieved_at,
                "region": "대한민국",
                "page": None,
                "section": "공식 상세 페이지 소개",
                "content_level": "official_page_summary",
                "url_verified": False,
                "metadata": {
                    "source_kind": "official_page_summary",
                    "source_type": "official_page_summary",
                    "summary_only": True,
                    "allow_grounded_claim": False,
                    "collection_error": type(exc).__name__,
                },
            }
        )
        return records, [f"원문 서버 연결 실패로 검증된 공식 웹페이지 발췌를 사용했습니다: {type(exc).__name__}"], stats

    warnings = list(result.warnings)

    def append_chunks(chunking_result, *, direct_file_url: str | None, file_name: str | None, content_level: str) -> None:
        for chunk in chunking_result.chunks:
            if not chunk.indexable or not chunk.content.strip():
                continue
            records.append(
                {
                    "source_id": source["source_id"],
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "title": source["title"],
                    "evidence_type": evidence_type,
                    "publisher": source["publisher"],
                    "source_url": source["source_url"],
                    "domain": "competition",
                    "evaluation_criteria": source["evaluation_criteria"],
                    "supported_roles": source["supported_roles"],
                    "content": chunk.content,
                    "reference_date": source["published_at"],
                    "published_at": source["published_at"],
                    "retrieved_at": retrieved_at,
                    "region": "대한민국",
                    "page": chunk.location_number,
                    "section": chunk.section_title,
                    "content_level": content_level,
                    "url_verified": True,
                    "metadata": {
                        "source_kind": "verified_official_original",
                        "direct_file_url": direct_file_url,
                        "file_name": file_name,
                        "content_kind": chunk.content_kind.value,
                        "chunking_version": chunk.chunking_version,
                    },
                }
            )

    if result.page_content and result.page_content.text.strip():
        document_id = _document_id(source["source_id"], source["source_url"] + "#web")
        cleaned = clean_page_content(result.page_content)
        web_chunks = chunk_document(
            cleaned,
            ChunkSourceContext(
                document_id=document_id,
                source_type=SourceType.URL_WEBPAGE,
                source_url=source["source_url"],
                document_title=source["title"],
            ),
        )
        append_chunks(web_chunks, direct_file_url=None, file_name=None, content_level="web_page")
        warnings.extend(web_chunks.warnings)

    for attachment in result.attachments:
        document_id = _document_id(source["source_id"], attachment.attachment_url)
        extraction = attachment.extraction
        if extraction.file_type.value == "pdf":
            extraction, removed = _strip_pdf_repeated_boilerplate(extraction)
            stats["pdf_boilerplate_blocks_removed"] += removed
        attachment_chunks = chunk_document(
            extraction,
            ChunkSourceContext(
                document_id=document_id,
                source_type=SourceType.URL_ATTACHMENT,
                source_url=attachment.attachment_url,
                source_page_url=source["source_url"],
                source_filename=attachment.file_name,
                document_title=source["title"],
                file_type=attachment.extraction.file_type.value,
            ),
        )
        append_chunks(
            attachment_chunks,
            direct_file_url=attachment.attachment_url,
            file_name=attachment.file_name,
            content_level="pdf_attachment" if extraction.file_type.value == "pdf" else "attachment",
        )
        warnings.extend(attachment.extraction.warnings)
        warnings.extend(attachment_chunks.warnings)

    warnings.extend(f"{item.file_name}: {item.message}" for item in result.failed_attachments)
    warnings.extend(f"{item.file_name}: {item.reason}" for item in result.unsupported_attachments)
    return records, warnings, stats


def _finalize_records(records: list[dict[str, Any]]) -> tuple[list[ExternalEvidenceDocument], dict[str, Any]]:
    """배치 전체 레코드에 짧은 청크 정리 → 문서 간 반복 제거 → 해시/중복 제거 →
    metadata 보완을 순서대로 적용하고, 최종 ExternalEvidenceDocument 목록을 만든다."""
    report: dict[str, Any] = {}

    short_result = merge_or_filter_short_chunks(records)
    report["short_chunk_merged_count"] = short_result.merged_count
    report["short_chunk_preserved_count"] = len(short_result.preserved_short)
    report["short_chunk_dropped_count"] = len(short_result.dropped)
    records = short_result.kept

    # 문서 간 반복(사이트 공통 메뉴 등) — 청크 텍스트 자체를 group_key=document_id로
    # 비교한다(요청 3번: "동일 도메인 페이지 간 공통 boilerplate 탐지").
    cross_document_items = [(str(r["document_id"]), r["content"]) for r in records]
    repeated_lines = find_repeated_boundary_texts(cross_document_items, min_group_count=_CROSS_DOCUMENT_MIN_GROUP_COUNT)
    before_boilerplate = len(records)
    records = [r for r in records if not is_boilerplate_text(r["content"], repeated_lines)]
    report["cross_document_boilerplate_removed_count"] = before_boilerplate - len(records)

    # 문서 단위 해시 — 같은 document_id의 모든(정제 후) 청크 원문을 이어붙여 계산한다.
    document_raw_text: dict[str, list[str]] = {}
    for record in records:
        document_raw_text.setdefault(record["document_id"], []).append(record["content"])
    document_hashes = {
        document_id: compute_source_content_hash("\n".join(texts))
        for document_id, texts in document_raw_text.items()
    }

    for record in records:
        record["chunk_content_hash"] = compute_chunk_content_hash(record["content"])
        record["normalized_content_hash"] = compute_normalized_content_hash(record["content"])
        record["file_hash"] = document_hashes[record["document_id"]]
        record["source_content_hash"] = document_hashes[record["document_id"]]

    kept, dropped_duplicates = deduplicate_chunks(records, key="normalized_content_hash")
    report["duplicate_chunks_removed_count"] = len(dropped_duplicates)

    documents: list[ExternalEvidenceDocument] = []
    for record in kept:
        page_start = record.get("page_start", record.get("page"))
        page_end = record.get("page_end", record.get("page"))
        page_missing_and_section_missing = record.get("page") is None and not (record.get("section") or "").strip()
        metadata = dict(record.get("metadata") or {})
        allow_grounded_claim = not page_missing_and_section_missing and bool(
            metadata.get("allow_grounded_claim", True)
        )
        documents.append(
            ExternalEvidenceDocument(
                source_id=record["source_id"],
                document_id=record["document_id"],
                chunk_id=record["chunk_id"],
                title=record["title"],
                evidence_type=record["evidence_type"],
                publisher=record["publisher"],
                source_url=record["source_url"],
                domain=record["domain"],
                evaluation_criteria=record["evaluation_criteria"],
                supported_roles=record["supported_roles"],
                content=record["content"],
                reference_date=record["reference_date"],
                published_at=record["published_at"],
                retrieved_at=record["retrieved_at"],
                region=record["region"],
                page=record.get("page"),
                section=record.get("section"),
                file_hash=record.get("file_hash"),
                source_content_hash=record.get("source_content_hash"),
                chunk_content_hash=record.get("chunk_content_hash"),
                normalized_content_hash=record.get("normalized_content_hash"),
                page_start=page_start,
                page_end=page_end,
                content_level=record.get("content_level"),
                url_verified=record.get("url_verified"),
                direct_file_url=metadata.get("direct_file_url"),
                allow_grounded_claim=allow_grounded_claim,
                metadata=metadata,
            )
        )

    report["final_document_count"] = len(documents)
    return documents, report


def main() -> int:
    args = parse_args()
    sources = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    selected_ids = set(args.source_id)
    if selected_ids:
        sources = [source for source in sources if source["source_id"] in selected_ids]

    embedder = KUREEmbedder()
    config = ExternalResearchConfig(collection_name=args.collection)
    client = chromadb.PersistentClient(path=str(args.chroma_path.resolve()))
    repository = ExternalEvidenceRepository(
        client=client,
        collection_name=config.collection_name,
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.embedding_dimension,
        embedding_version="embedding_v1",
    )
    indexing_service = ExternalEvidenceIndexingService(repository, embedder)

    all_records: list[dict[str, Any]] = []
    collection_warnings: list[str] = []
    failed_sources: list[dict[str, str]] = []
    pdf_boilerplate_removed_total = 0

    for index, source in enumerate(sources, start=1):
        source_id = source["source_id"]
        print(f"[{index}/{len(sources)}] collecting {source_id}", flush=True)
        try:
            records, warnings, stats = _to_chunk_records(source)
            if not records:
                raise RuntimeError("색인 가능한 원문 청크가 없습니다.")
            all_records.extend(records)
            collection_warnings.extend(warnings)
            pdf_boilerplate_removed_total += stats["pdf_boilerplate_blocks_removed"]
            print(f"[{index}/{len(sources)}] collected source_id={source_id} raw_chunk_count={len(records)}", flush=True)
        except Exception as exc:
            failed_sources.append({"source_id": source_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(sources)}] failed source_id={source_id} error={exc}", flush=True)

    documents, cleanup_report = _finalize_records(all_records)
    cleanup_report["pdf_boilerplate_blocks_removed_total"] = pdf_boilerplate_removed_total

    indexed_total = 0
    if documents:
        summary = indexing_service.index_evidence(documents, trace_id="verified-sources-batch")
        indexed_total = summary.indexed_count
        collection_warnings.extend(summary.warnings)

    report = {
        "source_count": len(sources),
        "raw_chunk_count": len(all_records),
        "cleanup": cleanup_report,
        "indexed_chunk_count": indexed_total,
        "failed_sources": failed_sources,
        "collection_name": config.collection_name,
        "collection_count": repository.count(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failed_sources else 1


if __name__ == "__main__":
    raise SystemExit(main())
