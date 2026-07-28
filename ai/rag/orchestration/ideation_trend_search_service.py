# 작성자: pge/Claude(2026-07-27, 주제 브레인스토밍 — 네이버 트렌드 검색 연동)
# 목적: 대화형 아이디어 발전 회의(ideation)의 discovery(아이디어 발굴) 모드 후보 생성
#       노드(candidate_planning)에 실시간 이슈 검색(ai/rag/trend_search, 네이버 검색
#       API)을 연결한다.
#
#       ideation_external_evidence_service.py(RAG-007 — 사전 색인된 외부 통계·시장·
#       정책 참고자료)와는 완전히 분리된 별도 콜백이다. RAG-007은 "이미 색인된 신뢰
#       자료"를 찾고, 이 모듈은 "지금 실시간으로 떠오르는 뉴스·이슈"를 찾는다 —
#       책임이 다르므로 같은 evidence_lookup 채널에 섞지 않는다.
from __future__ import annotations

import logging
from typing import Any, Callable

from ai.rag.trend_search.exceptions import TrendSearchError
from ai.rag.trend_search.schemas import InputType, TrendEvidenceItem, TrendSearchRequest
from ai.rag.trend_search.service import TrendSearchService

logger = logging.getLogger(__name__)

# candidate_planning 노드에 주입할 콜백 계약: Callable(query_text) -> dict.
# ideation_external_evidence_service.ExternalEvidenceLookup(persona_id, query)과 달리
# persona_id를 받지 않는다 — 트렌드 검색은 기획/개발 위원 역할과 무관하게 세션당
# 한 번만 실행되기 때문이다.
TrendSearchLookup = Callable[[str], dict]

DEFAULT_TOP_K = 5


def _result_to_dict(item: TrendEvidenceItem) -> dict:
    return {
        "title": item.title,
        "snippet": item.snippet,
        "source_url": item.source_url,
        "publisher": item.publisher,
        "published_at": item.published_at,
        "retrieved_at": item.retrieved_at,
        # 트렌드 근거는 항상 참고 자료다(TrendEvidenceItem.reference_only=True) —
        # LLM이 확정적 평가 근거로 오해하지 않도록 프롬프트/화면에도 그대로 넘긴다.
        "reference_only": True,
    }


def search_ideation_trend_evidence(
    query_text: str,
    service: TrendSearchService,
    *,
    input_type: InputType = "issue",
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """이번 세션의 트렌드 근거(실시간 이슈)를 검색한다.

    반환값은 {"trend_evidence": list[dict], "used_naver_search": bool,
    "warnings": list[str]}다 — candidate_planning 노드가 이 dict를 그대로 state에
    실어 API 응답에 노출한다(RAG-007 external_evidence와 동일한 형태). 검색이
    실패하거나 결과가 없어도 예외를 던지지 않는다(회의는 기존 흐름 그대로
    진행돼야 한다) — 빈 trend_evidence와 warnings 문구만 반환한다."""
    try:
        request = TrendSearchRequest(
            query_context=(query_text or "")[:2000], input_type=input_type, top_k=top_k
        )
    except TrendSearchError:
        logger.exception("[IDEATION_TREND_EVIDENCE_REQUEST_INVALID] query_len=%d", len(query_text or ""))
        return {
            "trend_evidence": [],
            "used_naver_search": False,
            "warnings": ["트렌드 검색 질의를 만들지 못해 참고자료 없이 진행합니다."],
        }

    try:
        response = service.search(request)
    except TrendSearchError:
        logger.exception("[IDEATION_TREND_EVIDENCE_SEARCH_FAILED]")
        return {
            "trend_evidence": [],
            "used_naver_search": False,
            "warnings": ["실시간 이슈 검색 중 오류가 발생해 참고자료 없이 진행합니다."],
        }
    except Exception:
        # 예상치 못한 오류도 회의 자체를 막지 않는다 — ai/rag/orchestration의 기존
        # fail-closed 정책과 동일(search_ideation_external_evidence 참고).
        logger.exception("[IDEATION_TREND_EVIDENCE_SEARCH_UNEXPECTED_ERROR]")
        return {
            "trend_evidence": [],
            "used_naver_search": False,
            "warnings": ["실시간 이슈 검색 중 오류가 발생해 참고자료 없이 진행합니다."],
        }

    result = {
        "trend_evidence": [_result_to_dict(item) for item in response.items],
        "used_naver_search": response.used_naver_search,
        "warnings": response.warnings,
    }
    logger.info(
        "[IDEATION_TREND_EVIDENCE_SEARCH_COMPLETE] result_count=%d used_naver_search=%s",
        len(result["trend_evidence"]),
        result["used_naver_search"],
    )
    return result


def make_ideation_trend_search_lookup(
    service: TrendSearchService,
    *,
    input_type: InputType = "issue",
    top_k: int = DEFAULT_TOP_K,
    max_queries_per_session: int = 1,
) -> TrendSearchLookup:
    """ai/meeting/graph/ideation_conv_discovery.py의 candidate_planning 노드에
    주입할 수 있는 Callable(query_text) -> dict를 만든다.

    트렌드 검색은 세션 시작 시 딱 한 번만 실행돼야 한다(요청: "모든 위원 발언마다
    실행하면 안 됨 — 느리고 비용도 커짐"). 이 lookup 콜러블 하나의 수명(=이번 HTTP
    요청, ideation_conversation_preview.py가 매 요청마다 새로 만든다) 동안
    max_queries_per_session회를 넘는 호출은 실제 검색 없이 빈 결과를 반환한다 —
    candidate_planning은 discovery 노드 진입당 이 콜백을 1회만 부르므로, 이 캡은
    같은 요청 안에서 우연히 여러 번 불려도(예: 재추천 재시도) 검색이 반복되지
    않게 막는 안전장치다."""
    call_count = {"n": 0}

    def lookup(query_text: str, **_ignored: Any) -> dict:
        if call_count["n"] >= max_queries_per_session:
            logger.info(
                "[IDEATION_TREND_EVIDENCE_CAP_REACHED] max_queries_per_session=%d", max_queries_per_session
            )
            return {"trend_evidence": [], "used_naver_search": False, "warnings": []}
        call_count["n"] += 1
        return search_ideation_trend_evidence(query_text, service, input_type=input_type, top_k=top_k)

    return lookup


__all__ = [
    "TrendSearchLookup",
    "DEFAULT_TOP_K",
    "search_ideation_trend_evidence",
    "make_ideation_trend_search_lookup",
]
