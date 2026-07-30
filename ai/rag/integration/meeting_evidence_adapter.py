"""
Meeting Evidence Adapter
============================
RAG-003(RoleAwareRetrievalService)/RAG-004(EvidenceLinkingService)의 결과를
ai/meeting/graph가 소비하는 형식으로 옮기는 순수 변환 함수 모음.

여기서는 evidence_id를 만들지 않는다 — ID 발급은 ai/meeting/graph/evidence.py의
EvidencePool 책임이다(회의 파이프라인 담당자가 (document_id, chunk_id)로 역조회).
새 검색/임베딩/LLM 호출도 하지 않는다 — 이미 받은 RAG-003/004 결과만 옮긴다.
"""

import logging
from typing import Optional

from ai.rag.evidence_linking.linker import resolve_score
from ai.rag.evidence_linking.metadata import (
    extract_document_title,
    extract_page_number,
    extract_section_title,
)
from ai.rag.evidence_linking.schemas import LinkedEvaluation
from ai.rag.integration.schemas import (
    MeetingLinkedEvidenceRef,
    MeetingRetrievedEvidence,
    PersonaRoleSearchResponse,
)
from ai.rag.role_retrieval.schemas import RoleSearchResponse

logger = logging.getLogger(__name__)


def _safe_int(value: object) -> Optional[int]:
    """location_number처럼 정수여야 하는 metadata 값을 안전하게 변환한다.
    변환할 수 없으면(예: 임의 문자열) 조용히 None을 반환한다 — 잘못된 값으로
    변환 전체를 중단시키지 않는다(기존 evidence_linking.metadata.extract_page_number와
    동일한 정책)."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_location_type(metadata: Optional[dict]) -> Optional[str]:
    if not metadata:
        return None
    value = metadata.get("location_type")
    if value is None:
        return None
    return str(value)


def to_retrieved_evidence(
    role_response: RoleSearchResponse,
    *,
    persona_id: str,
    role_id: Optional[str] = None,
) -> list[MeetingRetrievedEvidence]:
    """RoleSearchResponse.results를 회의 run_meeting(retrieved_evidence=...)에 넘길
    flat list 항목으로 변환한다.

    role_id를 명시하지 않으면 role_response.role_id(실제 검색에 쓰인 값, semantic-only
    fallback이면 None)를 그대로 쓴다. 같은 persona 내에서 (document_id, chunk_id)가
    중복되면 먼저 등장한 결과만 남긴다(검색 결과 순서 유지).
    """
    effective_role_id = role_id if role_id is not None else role_response.role_id

    items: list[MeetingRetrievedEvidence] = []
    seen_keys: set[tuple[str, str]] = set()
    duplicate_count = 0

    for result in role_response.results:
        key = (result.document_id, result.chunk_id)
        if key in seen_keys:
            duplicate_count += 1
            continue
        seen_keys.add(key)

        metadata = result.metadata or {}
        location_number = _safe_int(metadata.get("location_number"))
        score = resolve_score(result)

        items.append(
            MeetingRetrievedEvidence(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                persona_id=persona_id,
                role_id=effective_role_id,
                document_name=extract_document_title(metadata),
                section=extract_section_title(metadata),
                page=extract_page_number(metadata),
                location_number=location_number,
                location_type=_extract_location_type(metadata),
                text=result.content,
                semantic_score=result.semantic_score,
                role_score=result.role_score,
                final_score=result.final_score,
                score=score,
                document_role=metadata.get("document_role"),
                document_type=metadata.get("document_type"),
                ideation_source_type=metadata.get("ideation_source_type"),
                session_id=metadata.get("session_id"),
                origin_type=metadata.get("source_type"),
                file_name=metadata.get("source_filename"),
                source_url=metadata.get("source_url"),
            )
        )

    logger.info(
        "[RAG_MEETING_ADAPTER_COMPLETE] persona_id=%s role_id=%s input_count=%d "
        "output_count=%d duplicate_count=%d",
        persona_id,
        effective_role_id,
        len(role_response.results),
        len(items),
        duplicate_count,
    )
    return items


def build_meeting_retrieved_evidence(
    responses: list[PersonaRoleSearchResponse],
) -> list[MeetingRetrievedEvidence]:
    """여러 persona의 RoleSearchResponse를 하나의 flat list로 합친다.

    persona별 중복 제거는 to_retrieved_evidence()가 각자 처리하므로, 여기서는
    단순 이어붙이기만 한다 — 서로 다른 persona가 같은 (document_id, chunk_id)를
    검색했더라도 각 persona 몫은 그대로 유지된다.
    """
    combined: list[MeetingRetrievedEvidence] = []
    for entry in responses:
        combined.extend(
            to_retrieved_evidence(
                entry.response,
                persona_id=entry.persona_id,
                role_id=entry.role_id,
            )
        )
    return combined


def to_linked_evidence_refs(linked_evaluation: LinkedEvaluation) -> list[MeetingLinkedEvidenceRef]:
    """RAG-004 LinkedEvaluation.evidence[]를 (document_id, chunk_id)가 보존된
    plain dict 목록으로 옮긴다. evidence_id는 만들지 않는다 — 회의 쪽이
    (document_id, chunk_id)로 EvidencePool이 이미 발급한 evidence_id를 역조회한다.
    has_evidence=False(evidence=[])면 빈 리스트를 그대로 반환한다."""
    return [
        MeetingLinkedEvidenceRef(
            document_id=source.document_id,
            chunk_id=source.chunk_id,
            quote=source.quote,
            document_name=source.document_title,
            section=source.section_title,
            page=source.page_number,
            semantic_score=source.semantic_score,
            role_score=source.role_score,
            final_score=source.final_score,
        )
        for source in linked_evaluation.evidence
    ]
