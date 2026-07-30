# 작성자: 용준/Claude(2026-07-30, 요청: RAG-007 색인 청크 품질 정제)
# 목적: RAG-007(external_market_policy_evidence) 색인 파이프라인에 문서/청크 단위
#       content hash를 부여해 (1) 같은 원문이 재실행 시 새 record로 중복 색인되지
#       않게 하고, (2) 배치 내 근사 중복 청크(같은 내용이 서로 다른 chunk_id로 쪼개진
#       경우)를 감지·병합할 수 있게 한다.
#
#       ai/rag/preprocessing/html_cleaner.py::_dedup_exact와 같은 "정규화 →
#       set 기반 첫 항목 유지" 알고리즘을 청크 리스트 단위로 재사용 가능하게 만든
#       것이다(그 함수 자체는 페이지 내부 WebContentBlock 전용이라 그대로 호출하지
#       않는다).
from __future__ import annotations

import hashlib
import re
from typing import Any


def compute_source_content_hash(raw_text: str) -> str:
    """문서(웹페이지/첨부파일) 단위 원문 해시. 정규화하지 않는다 — 원문이 한 글자라도
    달라지면 다른 해시가 나와야, 같은 원문을 다시 색인했을 때만 "이미 있음"으로
    판정할 수 있다(file_hash/source_content_hash에 사용)."""
    return hashlib.sha256((raw_text or "").encode("utf-8")).hexdigest()


def normalize_for_content_hash(text: str) -> str:
    """연속 공백 통합 + 줄바꿈 정규화. 문장부호·숫자는 그대로 둔다(요청 6번: 법령
    조문·통계 수치가 다른 청크는 서로 다른 것으로 유지해야 하므로 숫자를 지우면 안
    된다)."""
    return re.sub(r"\s+", " ", text or "").strip()


def compute_normalized_content_hash(text: str) -> str:
    """정규화된 텍스트의 해시 — 공백/줄바꿈 차이만 있는 사실상 동일 청크를 같은
    해시로 묶기 위함(근사 중복 판정 키)."""
    return hashlib.sha256(normalize_for_content_hash(text).encode("utf-8")).hexdigest()


def compute_chunk_content_hash(text: str) -> str:
    """청크 원문 그대로의 해시(정규화 없음) — 완전 동일 문자열 탐지용, 감사
    스크립트(scripts/audit_rag007_collection.py)의 _text_hash와 별개로 색인 시점에
    미리 저장해두는 값이다."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def deduplicate_chunks(
    chunks: list[dict[str, Any]],
    *,
    key: str = "normalized_content_hash",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """chunks(각 항목은 최소 key로 지정한 해시 필드와 "page"/"section"을 가진 dict)를
    순서대로 훑어, 같은 해시를 가진 항목 중 첫 번째만 유지하고 나머지는 제거 목록으로
    분리한다(요청 6번: "중복이면 하나만 유지하되 원래 page/section 범위는 metadata로
    보존"). 유지된 항목의 dict를 직접 mutate하지 않고 새 dict를 반환한다 — 제거된
    항목들의 (document_id, page, section) 범위를 유지 항목의
    metadata["merged_source_ranges"]에 쌓아 어떤 위치의 중복이 합쳐졌는지 추적할 수
    있게 한다.

    반환값: (kept, dropped) — dropped 항목에는 어떤 kept 항목으로 흡수됐는지
    "duplicate_of_chunk_id"를 덧붙여 반환한다(로그/보고용, 실제 색인에는 쓰이지
    않음)."""
    seen: dict[str, dict[str, Any]] = {}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for chunk in chunks:
        digest = chunk.get(key)
        if not digest:
            kept.append(chunk)
            continue
        existing = seen.get(digest)
        if existing is None:
            merged_chunk = dict(chunk)
            existing_metadata = dict(merged_chunk.get("metadata") or {})
            merged_chunk["metadata"] = existing_metadata
            seen[digest] = merged_chunk
            kept.append(merged_chunk)
            continue

        existing_metadata = existing.setdefault("metadata", {})
        merged_ranges = existing_metadata.setdefault("merged_source_ranges", [])
        merged_ranges.append(
            {
                "document_id": chunk.get("document_id"),
                "chunk_id": chunk.get("chunk_id"),
                "page": chunk.get("page"),
                "section": chunk.get("section"),
            }
        )
        dropped.append({**chunk, "duplicate_of_chunk_id": existing.get("chunk_id")})

    return kept, dropped


__all__ = [
    "compute_source_content_hash",
    "normalize_for_content_hash",
    "compute_normalized_content_hash",
    "compute_chunk_content_hash",
    "deduplicate_chunks",
]
