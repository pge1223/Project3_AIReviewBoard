# 작성자: 용준/Claude(2026-07-29, target/criteria/외부근거/전문가판단 분리)
# 목적: 대화형 아이디어 발전 회의(ideation) 위원 발언에 RAG-006(SimilarCaseSearchService,
#       ai/rag/similar_cases — 유사 공공서비스 사례)을 외부 근거 후보로 연결한다.
#
#       ideation_external_evidence_service.py(RAG-007 — 외부 통계·시장·정책·법령)와 동일한
#       fail-closed/캐시 패턴을 그대로 재사용한다. similar_success_cases 컬렉션은 이 작업
#       시점에 시딩된 문서가 없다(조사 결과 확인) — SimilarCaseSearchService.search()가
#       빈 컬렉션에서도 예외 없이 정상 빈 응답을 반환하므로, 이 래퍼는 그 응답을 그대로
#       감싸기만 하고 별도 방어 로직(가짜 사례 생성 등)을 추가하지 않는다. 나중에 컬렉션이
#       채워지면 같은 인터페이스로 자동 연결된다.
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ai.rag.orchestration.ideation_evidence_service import resolve_ideation_role_id
from ai.rag.similar_cases.schemas import SimilarCaseSearchRequest
from ai.rag.similar_cases.search_service import SimilarCaseSearchService

logger = logging.getLogger(__name__)

# ai/meeting/graph/ideation_conv_nodes.py에 주입할 수 있는 Callable(persona_id, query_text)
# -> dict 계약. ideation_external_evidence_service.ExternalEvidenceLookup과 동일한 형태다.
SimilarCaseLookup = Callable[..., dict]

# RAG-007 래퍼(ideation_external_evidence_service.py)와 동일하게 고정값을 쓴다 — ideation
# 대화형 회의는 아직 구조화된 domain 필드를 따로 갖지 않는다.
DEFAULT_IDEATION_SIMILAR_CASE_DOMAIN = "competition"
DEFAULT_TOP_K = 3


def _result_to_dict(result: Any) -> dict:
    """SimilarCaseResult -> plain dict. 실제 schemas.py 필드만 사용한다(organization에
    대응하는 필드가 없어 source_name으로, published_at에 대응하는 필드가 없어 None으로
    둔다 — 지어내지 않는다)."""
    first_evidence = result.evidence[0] if getattr(result, "evidence", None) else None
    short_summary = None
    if result.common_points:
        short_summary = result.common_points[0]
    elif result.similarity_reasons:
        short_summary = result.similarity_reasons[0]
    return {
        "case_id": result.case_id,
        "document_id": first_evidence.document_id if first_evidence else result.case_id,
        "chunk_id": first_evidence.chunk_id if first_evidence else result.case_id,
        "title": result.title,
        "document_title": result.title,
        "case_type": result.case_type.value if hasattr(result.case_type, "value") else result.case_type,
        "domain": result.domain,
        "organization": result.source_name,
        "source_url": result.source_url,
        "published_at": None,
        "page_or_section": first_evidence.section if first_evidence else None,
        "matched_criteria": result.matched_criteria,
        "similarity_reasons": result.similarity_reasons,
        "common_points": result.common_points,
        "different_points": result.different_points,
        "current_document_gaps": result.current_document_gaps,
        "short_summary": short_summary,
        "quote": first_evidence.quote if first_evidence else None,
        "final_score": result.similarity_score,
        # RAG-006 결과는 항상 참고 자료다(SimilarCaseResult.reference_only=True) — LLM이
        # 확정적 평가 근거로 오해하지 않도록 그대로 재노출한다.
        "reference_only": True,
    }


def search_ideation_similar_cases(
    persona_id: str,
    query_text: str,
    service: SimilarCaseSearchService,
    *,
    domain: str = DEFAULT_IDEATION_SIMILAR_CASE_DOMAIN,
    top_k: int = DEFAULT_TOP_K,
    cache: Optional[dict[tuple[str, str], dict]] = None,
) -> dict:
    """전문가 1명의 이번 턴 유사 공공서비스 사례를 검색한다.

    반환값은 {"similar_cases": list[dict], "warnings": list[str]}다. 검색이 실패하거나
    결과가 없어도(컬렉션이 비어 있어도) 예외를 던지지 않는다 — 회의는 기존 흐름 그대로
    진행돼야 한다(search_ideation_external_evidence와 동일 정책)."""
    cache_key = (persona_id, query_text)
    if cache is not None and cache_key in cache:
        logger.info(
            "[IDEATION_SIMILAR_CASE_CACHE_HIT] persona_id=%s query_len=%d", persona_id, len(query_text)
        )
        return cache[cache_key]

    reviewer_role = resolve_ideation_role_id(persona_id) or persona_id
    request = SimilarCaseSearchRequest(
        document_summary=(query_text[:2000] if query_text else "") or reviewer_role,
        domain=domain,
        evaluation_criteria=["공모전 평가"],
        top_k=top_k,
        trace_id=f"ideation-similar-case-{persona_id}",
    )

    try:
        response = service.search(request)
    except Exception:
        # SimilarCaseSearchService.search()는 빈 컬렉션에서도 예외를 던지지 않고 정상 빈
        # 응답을 반환한다(확인됨) — 여기서 잡는 예외는 그 외의 예상치 못한 오류다.
        logger.exception(
            "[IDEATION_SIMILAR_CASE_SEARCH_UNEXPECTED_ERROR] persona_id=%s reviewer_role=%s",
            persona_id,
            reviewer_role,
        )
        result = {
            "similar_cases": [],
            "warnings": ["유사 사례 검색 중 오류가 발생해 이번 턴은 참고자료 없이 진행합니다."],
        }
        if cache is not None:
            cache[cache_key] = result
        return result

    result = {
        "similar_cases": [_result_to_dict(r) for r in response.results],
        "warnings": list(response.warnings),
    }
    if cache is not None:
        cache[cache_key] = result

    logger.info(
        "[IDEATION_SIMILAR_CASE_SEARCH_COMPLETE] persona_id=%s reviewer_role=%s result_count=%d",
        persona_id,
        reviewer_role,
        len(response.results),
    )
    return result


def make_ideation_similar_case_lookup(
    service: SimilarCaseSearchService,
    *,
    domain: str = DEFAULT_IDEATION_SIMILAR_CASE_DOMAIN,
    top_k: int = DEFAULT_TOP_K,
) -> Callable[[str, str], dict]:
    """ai/meeting/graph/ideation_conv_nodes.py::make_conv_discussion_node(similar_case_lookup=...)에
    주입할 수 있는 Callable(persona_id, query_text) -> dict를 만든다.
    make_ideation_external_evidence_lookup과 동일한 요청-수명 캐시 패턴을 쓴다."""
    cache: dict[tuple[str, str], dict] = {}

    def lookup(persona_id: str, query_text: str, **_ignored: Any) -> dict:
        return search_ideation_similar_cases(
            persona_id, query_text, service, domain=domain, top_k=top_k, cache=cache
        )

    return lookup


__all__ = [
    "SimilarCaseLookup",
    "DEFAULT_IDEATION_SIMILAR_CASE_DOMAIN",
    "DEFAULT_TOP_K",
    "search_ideation_similar_cases",
    "make_ideation_similar_case_lookup",
]
