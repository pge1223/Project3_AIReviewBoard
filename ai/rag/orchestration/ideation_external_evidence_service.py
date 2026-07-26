# 작성자: 용준/Claude(2026-07-27, RAG-007 연결)
# 목적: 대화형 아이디어 발전 회의(ideation)의 discovery(아이디어 발굴) 모드 후보 생성
#       노드(candidate_planning/candidate_feasibility)에 RAG-007(ExternalResearchService,
#       ai/rag/external_research)을 연결한다.
#
#       ideation_evidence_service.py(RAG-006/RAG-003 — 프로젝트 문서·유사 사례 근거)와는
#       완전히 분리된 별도 콜백이다. RAG-006은 "현재 문서/공고문/유사 사례" 근거를 찾고,
#       RAG-007은 "외부 통계·시장·정책·법령" 참고자료를 찾는다 — 책임이 다르므로 같은
#       evidence_lookup 채널에 섞지 않는다(ai/rag/external_research/README.md 2절 표 그대로).
#
#       persona_id -> reviewer_role 매핑은 ideation_evidence_service.py의
#       resolve_ideation_role_id()가 이미 정의한 값(planning_expert -> "planning",
#       dev_expert -> "technology")을 그대로 재사용한다 — 새 매핑을 만들지 않는다.
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ai.rag.external_research.exceptions import ExternalResearchError
from ai.rag.external_research.schemas import ExternalResearchRequest
from ai.rag.external_research.search_service import ExternalResearchService
from ai.rag.orchestration.ideation_evidence_service import resolve_ideation_role_id

logger = logging.getLogger(__name__)

# ai/meeting/graph/ideation_nodes.py::EvidenceLookup과 동일한 계약(Callable[[persona_id,
# query], list[dict]], 선택적으로 runtime_scope 키워드 인자) — candidate_planning/
# candidate_feasibility 노드는 evidence_lookup과 완전히 별개인 external_evidence_lookup
# 콜백으로 이 타입의 값을 받는다.
ExternalEvidenceLookup = Callable[..., list[dict]]

# ideation 대화형 회의는 아직 "공모전/정부지원/스타트업" 같은 구조화된 domain 필드를
# notice_and_criteria에 담지 않는다(경쟁공모전 전용 화면이라 사실상 고정) — 기존
# analyze_project()의 doc_type 어휘("competition"/"government_support"/"startup",
# app/api/routes/documents.py DOC_TYPE_OPTIONS)와 맞춰 고정값을 쓴다. domain 필터는
# DatasetProvider.domain_filter_fallback_to_all=True(기본값)라 실제 색인 데이터의 domain이
# 다르더라도 검색 자체가 막히지 않는다(전체 컬렉션으로 자동 폴백).
DEFAULT_IDEATION_EXTERNAL_DOMAIN = "competition"

DEFAULT_TOP_K = 3


def resolve_external_reviewer_role(persona_id: str) -> str:
    """ExternalResearchRequest.reviewer_role에 넘길 값. resolve_ideation_role_id가 아는
    persona_id(planning_expert/dev_expert)는 RAG-003 role_id로 변환하고(요청 5번 매핑),
    모르는 persona_id는 그대로 둔다(reviewer_role은 non-blank이기만 하면 되고, query_builder/
    ranking은 등록되지 않은 역할이 들어와도 예외를 던지지 않는다 — 검색어를 지어내지 않고
    확장 검색어 없이 진행할 뿐이다)."""
    return resolve_ideation_role_id(persona_id) or persona_id


def _has_confirmed_source(result: Any) -> bool:
    """publisher/source_url/(reference_date 또는 published_at)이 모두 확인된 자료만 화면에
    노출한다(요청 9번). ExternalResearchService가 이미 publisher/source_url 없는 후보는
    제거하지만(source_validator, verified_source 게이트) 날짜 필드는 필수로 강제하지
    않으므로(reference_date/published_at 둘 다 없어도 verified_source=True일 수 있음),
    여기서 표시 전용 게이트를 한 번 더 건다 — 검색/랭킹 결과 자체를 바꾸지 않고, 다음 단계
    (프롬프트/화면)에 넘길지 여부만 결정한다."""
    return bool(
        (result.publisher or "").strip()
        and (result.source_url or "").strip()
        and ((result.reference_date or "").strip() or (result.published_at or "").strip())
    )


def _result_to_dict(result: Any) -> dict:
    return {
        "source_id": result.source_id,
        "document_id": result.document_id,
        "chunk_id": result.chunk_id,
        "title": result.title,
        "evidence_type": result.evidence_type.value if hasattr(result.evidence_type, "value") else result.evidence_type,
        "publisher": result.publisher,
        "source_url": result.source_url,
        "domain": result.domain,
        "matched_criteria": result.matched_criteria,
        "quote": result.quote,
        "reference_date": result.reference_date,
        "published_at": result.published_at,
        "region": result.region,
        "period": result.period,
        "metric_name": result.metric_name,
        "metric_value": result.metric_value,
        "metric_unit": result.metric_unit,
        "final_score": result.final_score,
        "retrieval_source": result.retrieval_source,
        # RAG-007 결과는 항상 참고 자료다(ExternalEvidenceResult.reference_only=True) —
        # LLM이 이 값을 확정적 평가 근거로 오해하지 않도록 프롬프트/화면에도 그대로 넘긴다.
        "reference_only": True,
    }


def search_ideation_external_evidence(
    persona_id: str,
    query_text: str,
    service: ExternalResearchService,
    *,
    domain: str = DEFAULT_IDEATION_EXTERNAL_DOMAIN,
    top_k: int = DEFAULT_TOP_K,
    cache: Optional[dict[tuple[str, str], dict]] = None,
) -> dict:
    """전문가 1명의 이번 턴 외부 통계·시장·정책 참고자료를 검색한다.

    반환값은 {"external_evidence": list[dict], "used_dataset_search": bool,
    "used_public_api_search": bool, "warnings": list[str]}다 — candidate_planning/
    candidate_feasibility 노드가 이 dict를 그대로 state에 실어 API 응답에 노출한다(요청
    11번). 검색이 실패하거나 결과가 없어도 예외를 던지지 않는다(요청 10번 — 회의는 기존
    흐름 그대로 진행돼야 한다) — 빈 external_evidence와 warnings 문구만 반환한다."""
    cache_key = (persona_id, query_text)
    if cache is not None and cache_key in cache:
        logger.info(
            "[IDEATION_EXTERNAL_EVIDENCE_CACHE_HIT] persona_id=%s query_len=%d", persona_id, len(query_text)
        )
        return cache[cache_key]

    reviewer_role = resolve_external_reviewer_role(persona_id)
    # evaluation_criteria는 빈 리스트를 허용하지 않는다(schemas.py) — ideation 대화형 회의는
    # 아직 구조화된 평가 기준 목록을 따로 들고 있지 않으므로(notice_and_criteria는 공모전명 +
    # 공고문 원문 텍스트뿐), 검색어 자체(query_text, 공모전명·공고문·문제 상황·후보 아이디어
    # 내용을 이미 포함)를 query_context로 넘기고 evaluation_criteria는 최소 마커 하나만 채운다
    # — 실제 평가 기준을 지어내지 않는다(criteria_score는 최종 점수 가중치 0.15의 보조
    # 신호일 뿐이라, 이 마커가 정확히 일치하지 않아도 검색 자체가 막히지 않는다).
    request = ExternalResearchRequest(
        domain=domain,
        evaluation_criteria=["공모전 평가"],
        reviewer_role=reviewer_role,
        query_context=query_text[:2000] if query_text else None,
        top_k=top_k,
        trace_id=f"ideation-external-{persona_id}",
    )

    try:
        response = service.search(request)
    except ExternalResearchError:
        logger.exception(
            "[IDEATION_EXTERNAL_EVIDENCE_SEARCH_FAILED] persona_id=%s reviewer_role=%s", persona_id, reviewer_role
        )
        result = {
            "external_evidence": [],
            "used_dataset_search": False,
            "used_public_api_search": False,
            "warnings": ["외부 통계·정책 자료 검색 중 오류가 발생해 이번 턴은 참고자료 없이 진행합니다."],
        }
        if cache is not None:
            cache[cache_key] = result
        return result
    except Exception:
        # 예상치 못한 오류도 회의 자체를 막지 않는다(요청 10번, ai/rag/orchestration의
        # 기존 fail-closed 정책과 동일 — search_ideation_evidence 참고).
        logger.exception(
            "[IDEATION_EXTERNAL_EVIDENCE_SEARCH_UNEXPECTED_ERROR] persona_id=%s reviewer_role=%s",
            persona_id,
            reviewer_role,
        )
        result = {
            "external_evidence": [],
            "used_dataset_search": False,
            "used_public_api_search": False,
            "warnings": ["외부 통계·정책 자료 검색 중 오류가 발생해 이번 턴은 참고자료 없이 진행합니다."],
        }
        if cache is not None:
            cache[cache_key] = result
        return result

    display_ready = [r for r in response.results if _has_confirmed_source(r)]
    dropped = len(response.results) - len(display_ready)
    warnings = list(response.warnings)
    if dropped:
        warnings.append(f"발행기관·출처 URL·기준일이 모두 확인되지 않은 외부자료 {dropped}건을 화면에서 제외했습니다.")

    result = {
        "external_evidence": [_result_to_dict(r) for r in display_ready],
        "used_dataset_search": response.used_dataset_search,
        "used_public_api_search": response.used_public_api_search,
        "warnings": warnings,
    }
    if cache is not None:
        cache[cache_key] = result

    logger.info(
        "[IDEATION_EXTERNAL_EVIDENCE_SEARCH_COMPLETE] persona_id=%s reviewer_role=%s "
        "result_count=%d dropped_unconfirmed=%d used_dataset_search=%s",
        persona_id,
        reviewer_role,
        len(display_ready),
        dropped,
        response.used_dataset_search,
    )
    return result


def make_ideation_external_evidence_lookup(
    service: ExternalResearchService,
    *,
    domain: str = DEFAULT_IDEATION_EXTERNAL_DOMAIN,
    top_k: int = DEFAULT_TOP_K,
) -> Callable[[str, str], dict]:
    """ai/meeting/graph/ideation_conv_discovery.py의 candidate_planning/candidate_feasibility
    노드에 주입할 수 있는 Callable(persona_id, query_text) -> dict를 만든다.

    반환하는 dict 형태는 search_ideation_external_evidence()와 동일하다(evidence_lookup처럼
    list[dict]만 반환하지 않는 이유: used_dataset_search/warnings 같은 응답 단위 메타데이터도
    노드가 state에 그대로 실어 API로 노출해야 하기 때문 — 요청 11번).

    cache(요청 12번, "세션 내 동일 질의가 반복되지 않도록 캐시")는 이 lookup 콜러블 하나의
    수명 동안만 유지된다 — ideation_evidence_service.py의 evidence_lookup과 동일한 제약으로,
    evidence_lookup 자체가 그래프 state에 직렬화될 수 없는 콜러블이라 매 HTTP 요청마다 backend가
    새로 만든다(ideation_conversation_preview.py 참고). 그래서 "세션 내"가 아니라 "이번 요청
    내"가 실질적인 캐시 범위다 — candidate_planning과 candidate_feasibility가 같은 요청 안에서
    정지 없이 연달아 실행되므로(ideation_conv_discovery.py), 두 노드가 우연히 완전히 같은
    질의문을 만들면(예: 재추천 없이 같은 공모전 설명만 반복) 이 캐시가 중복 검색을 막는다."""
    cache: dict[tuple[str, str], dict] = {}

    def lookup(persona_id: str, query_text: str, **_ignored: Any) -> dict:
        return search_ideation_external_evidence(
            persona_id, query_text, service, domain=domain, top_k=top_k, cache=cache
        )

    return lookup


__all__ = [
    "ExternalEvidenceLookup",
    "DEFAULT_IDEATION_EXTERNAL_DOMAIN",
    "DEFAULT_TOP_K",
    "resolve_external_reviewer_role",
    "search_ideation_external_evidence",
    "make_ideation_external_evidence_lookup",
]
