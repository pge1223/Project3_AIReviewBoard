"""
External Research Search Service (RAG-007)
=================================================
도메인·평가 기준·위원 역할로 사전 수집 데이터셋(및 선택적으로 공공데이터 API)을
검색하고, semantic/role/criteria/freshness 점수를 조합한 최종 점수로 정렬해 자료
단위 Top-K를 반환한다. LangGraph나 ai.meeting.graph에 의존하지 않으며 단독으로
생성/호출할 수 있다.

외부자료는 항상 참고 자료다 — 이 서비스는 RAG-005(근거 충족도)나 확정적 평가
점수를 직접 만들지 않는다(ExternalEvidenceResult.reference_only=True로 명시,
RAG-005 코드는 이 패키지 어디에서도 import하지 않는다).
"""

import dataclasses
import logging
import math
import time
from typing import Optional

from ai.rag.external_research.config import ExternalResearchConfig, FreshnessConfig
from ai.rag.external_research.exceptions import (
    ExternalEvidenceSearchError,
    ExternalResearchError,
)
from ai.rag.external_research.freshness import compute_freshness
from ai.rag.external_research.indexing_service import EmbedderLike
from ai.rag.external_research.providers.base import ExternalEvidenceCandidate, ExternalResearchProvider
from ai.rag.external_research.providers.dataset_provider import DatasetProvider
from ai.rag.external_research.query_builder import build_external_research_query
from ai.rag.external_research.ranking import compute_criteria_score, compute_final_score, compute_role_score
from ai.rag.external_research.schemas import (
    ExternalEvidenceResult,
    ExternalResearchRequest,
    ExternalResearchResponse,
)

logger = logging.getLogger(__name__)


def _is_finite(value: Optional[float]) -> bool:
    return value is not None and not math.isnan(value) and not math.isinf(value)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


_LIVE_QUERY_SUFFIX = {
    "planning": "최근 뉴스 정책 동향",
    "marketing": "최근 뉴스 시장 동향",
    "technology": "최신 기술 동향 도입 사례",
}

# 용준/Claude(2026-07-28, RAG-007 NAVER 0건 버그 수정): NAVER 뉴스 검색은 질의어의
# 모든 단어를 AND 조건으로 취급한다 — 실측(NAVER API HUB, 같은 문맥으로 단어 수만
# 늘려가며 확인) 결과 문맥 단어가 8개를 넘어가면 total=0으로 무너진다(2단어=84102건,
# 6단어=444건, 10단어=0건). 기존 코드는 문자 수(300자, 한글 기준 60~100단어)로만
# 잘랐기 때문에 실제 회의에서 쓰이는 문장형 검색 문맥은 항상 이 임계값을 넘어 실전에서
# 늘 0건이었다(문제 발견 경위: 실제 세션 로그에서 result_count=0이 반복 확인됨). 접미사
# 단어까지 포함해 총 단어 수가 임계값 아래로 유지되도록 문맥은 단어 수 기준으로 자른다.
_LIVE_QUERY_CONTEXT_WORD_LIMIT = 5


def _build_live_api_query(request: ExternalResearchRequest, fallback_query: str) -> str:
    """검색 API에는 RAG용 라벨 문장이 아니라 짧은 검색어를 전달한다."""
    context = " ".join((request.query_context or "").split())
    if not context:
        context = " ".join(fallback_query.split())
    # NAVER 검색에서 공고문 전체가 그대로 질의가 되지 않도록 검색 문맥을 제한한다
    # (문자 수가 아니라 단어 수로 자른다 — 위 주석 참고).
    context_words = context.split(" ")[:_LIVE_QUERY_CONTEXT_WORD_LIMIT]
    suffix = _LIVE_QUERY_SUFFIX.get(request.reviewer_role, "최근 동향")
    return f"{' '.join(context_words)} {suffix}".strip()


class ExternalResearchService:
    """RAG-003/RAG-006과 완전히 분리된 외부 시장·정책 자료 검색 서비스.
    provider(들)를 생성자로 주입받으며, 새 Chroma client나 임베딩 모델, 외부 LLM을
    내부에서 만들지 않는다."""

    def __init__(
        self,
        dataset_provider: Optional[ExternalResearchProvider] = None,
        *,
        public_api_provider: Optional[ExternalResearchProvider] = None,
        config: Optional[ExternalResearchConfig] = None,
        freshness_config: Optional[FreshnessConfig] = None,
        # 용준/Claude(2026-07-28, 요청: 관련 없는 뉴스가 그대로 노출되는 문제 수정) —
        # PublicApiProvider(NAVER 등)는 자체 유사도 점수를 주지 않아(semantic_score=None,
        # public_api_provider.py 참고) 출처(publisher/URL/날짜)만 확인되면 검색 문맥과
        # 전혀 무관한 기사도 그대로 통과했다(실측: "공공기관 AI 혁신 챌린지" 문맥에서
        # 무관한 금융 감사 기사가 노출됨). embedder가 주어지면 KURE로 문맥-기사 유사도를
        # 계산해 semantic_score를 채운다 — DatasetProvider와 같은 embedder를 재사용해야
        # 하므로(새 SentenceTransformer 로딩 금지) 호출자가 주입한다.
        embedder: Optional[EmbedderLike] = None,
    ):
        self._dataset_provider = dataset_provider
        self._public_api_provider = public_api_provider
        self._config = config or ExternalResearchConfig()
        self._freshness_config = freshness_config or FreshnessConfig()
        self._embedder = embedder

    def search(self, request: ExternalResearchRequest) -> ExternalResearchResponse:
        start = time.monotonic()
        top_k = min(request.top_k, self._config.max_top_k)
        min_score = request.min_score if request.min_score is not None else self._config.min_similarity_score

        query_text = build_external_research_query(
            domain=request.domain,
            evaluation_criteria=request.evaluation_criteria,
            reviewer_role=request.reviewer_role,
            query_context=request.query_context,
            region=request.region,
            evidence_types=request.evidence_types,
        )

        logger.info(
            "[EXTERNAL_RESEARCH_SEARCH_START] trace_id=%s domain=%s reviewer_role=%s "
            "criteria_count=%d top_k=%d",
            request.trace_id,
            request.domain,
            request.reviewer_role,
            len(request.evaluation_criteria),
            top_k,
        )

        warnings: list[str] = []
        candidates: list[ExternalEvidenceCandidate] = []
        used_dataset_search = False
        used_public_api_search = False

        if self._config.enable_dataset_search and self._dataset_provider is not None:
            used_dataset_search = True
            candidates.extend(self._call_dataset_provider(request, query_text, warnings))

        if self._config.enable_public_api_search and self._public_api_provider is not None:
            used_public_api_search = True
            candidates.extend(
                self._call_public_api_provider(
                    request,
                    _build_live_api_query(request, query_text),
                    warnings,
                    relevance_query=query_text,
                )
            )

        valid_candidates, rejected_count = self._filter_candidates(candidates, min_score)
        if rejected_count:
            warnings.append(f"출처 검증에 실패한 외부자료 {rejected_count}건을 결과에서 제외했습니다.")

        deduped = self._dedupe(valid_candidates)
        best_per_document = self._aggregate_by_document(deduped)

        results = [self._build_result(request, candidate) for candidate in best_per_document.values()]
        results.sort(key=lambda r: r.final_score, reverse=True)
        results = results[:top_k]

        if not results:
            logger.info(
                "[EXTERNAL_RESEARCH_SEARCH_EMPTY] trace_id=%s domain=%s reviewer_role=%s",
                request.trace_id,
                request.domain,
                request.reviewer_role,
            )
            warnings.append("현재 조건에 맞는 외부 시장·정책 자료를 찾지 못했습니다.")

        response = ExternalResearchResponse(
            results=results,
            total_results=len(results),
            query_text=query_text,
            reviewer_role=request.reviewer_role,
            used_dataset_search=used_dataset_search,
            used_public_api_search=used_public_api_search,
            trace_id=request.trace_id,
            warnings=warnings,
        )

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "[EXTERNAL_RESEARCH_SEARCH_COMPLETE] trace_id=%s domain=%s reviewer_role=%s "
            "candidate_count=%d result_count=%d duration_ms=%d",
            request.trace_id,
            request.domain,
            request.reviewer_role,
            len(candidates),
            len(results),
            duration_ms,
        )
        return response

    def _call_dataset_provider(
        self, request: ExternalResearchRequest, query_text: str, warnings: list[str]
    ) -> list[ExternalEvidenceCandidate]:
        provider = self._dataset_provider
        logger.info(
            "[EXTERNAL_PROVIDER_CALL_START] trace_id=%s provider_name=%s", request.trace_id, provider.name
        )
        try:
            found = provider.search(request, query_text)
        except Exception as exc:
            logger.warning(
                "[EXTERNAL_PROVIDER_CALL_FAILED] trace_id=%s provider_name=%s error_code=%s",
                request.trace_id,
                provider.name,
                type(exc).__name__,
            )
            # 데이터셋 검색은 1순위 검색 경로이므로, 실패를 조용히 삼키지 않고
            # 명확한 예외로 알린다(공공 API 실패와 달리 fallback할 다른 사전 데이터가 없음).
            raise ExternalEvidenceSearchError(
                f"외부자료 데이터셋 검색 중 오류가 발생했습니다: {type(exc).__name__}"
            ) from exc

        if isinstance(provider, DatasetProvider) and provider.last_search_used_domain_fallback:
            warnings.append(
                f"요청한 도메인('{request.domain}')과 정확히 일치하는 자료가 없어 "
                "전체 외부자료에서 검색했습니다."
            )

        logger.info(
            "[EXTERNAL_PROVIDER_CALL_COMPLETE] trace_id=%s provider_name=%s candidate_count=%d",
            request.trace_id,
            provider.name,
            len(found),
        )
        return found

    def _call_public_api_provider(
        self,
        request: ExternalResearchRequest,
        query_text: str,
        warnings: list[str],
        *,
        relevance_query: str,
    ) -> list[ExternalEvidenceCandidate]:
        provider = self._public_api_provider
        logger.info(
            "[EXTERNAL_PROVIDER_CALL_START] trace_id=%s provider_name=%s", request.trace_id, provider.name
        )
        try:
            found = provider.search(request, query_text)
        except ExternalResearchError as exc:
            # 실시간 API 실패(timeout/미가용 등)는 사전 수집 데이터 검색 결과를 무효화하지
            # 않는다 — warning만 남기고 계속 진행한다(섹션 25).
            logger.warning(
                "[EXTERNAL_PROVIDER_CALL_FAILED] trace_id=%s provider_name=%s error_code=%s",
                request.trace_id,
                provider.name,
                type(exc).__name__,
            )
            warnings.append("실시간 공공데이터 검색을 사용할 수 없어 사전 수집 자료로만 결과를 구성했습니다.")
            return []
        except Exception as exc:
            logger.warning(
                "[EXTERNAL_PROVIDER_CALL_FAILED] trace_id=%s provider_name=%s error_code=%s",
                request.trace_id,
                provider.name,
                type(exc).__name__,
            )
            warnings.append("실시간 공공데이터 검색을 사용할 수 없어 사전 수집 자료로만 결과를 구성했습니다.")
            return []

        logger.info(
            "[EXTERNAL_PROVIDER_CALL_COMPLETE] trace_id=%s provider_name=%s candidate_count=%d",
            request.trace_id,
            provider.name,
            len(found),
        )
        return self._score_public_api_relevance(found, relevance_query, request.trace_id)

    def _score_public_api_relevance(
        self,
        candidates: list[ExternalEvidenceCandidate],
        relevance_query: str,
        trace_id: Optional[str],
    ) -> list[ExternalEvidenceCandidate]:
        """PublicApiProvider 후보는 semantic_score=None으로 들어온다 — embedder가 있으면
        검색 문맥(relevance_query, 라벨 포함 원본 질의) 대비 기사(title+content) 유사도를
        계산해 semantic_score를 채운다. 이후 _filter_candidates()가 기존
        min_similarity_score 임계값으로 데이터셋 결과와 동일하게 걸러낸다 — 새 필터
        경로를 따로 만들지 않는다. embedder가 없으면(주입 안 됨) 기존 동작 그대로
        candidates를 그대로 반환한다(하위 호환)."""
        if self._embedder is None or not candidates:
            return candidates
        try:
            query_vector = self._embedder.embed_query(relevance_query)
        except Exception:
            logger.warning(
                "[EXTERNAL_RELEVANCE_SCORING_FAILED] trace_id=%s reason=query_embedding_failed", trace_id
            )
            return candidates

        scored: list[ExternalEvidenceCandidate] = []
        for candidate in candidates:
            if candidate.semantic_score is not None:
                scored.append(candidate)
                continue
            text = f"{candidate.title} {candidate.content}".strip()
            if not text:
                scored.append(dataclasses.replace(candidate, semantic_score=0.0))
                continue
            try:
                doc_vector = self._embedder.embed_query(text)
            except Exception:
                logger.warning(
                    "[EXTERNAL_RELEVANCE_SCORING_FAILED] trace_id=%s source_id=%s reason=doc_embedding_failed",
                    trace_id,
                    candidate.source_id,
                )
                scored.append(candidate)
                continue
            score = _cosine_similarity(query_vector, doc_vector)
            scored.append(dataclasses.replace(candidate, semantic_score=score))
        return scored

    @staticmethod
    def _filter_candidates(
        candidates: list[ExternalEvidenceCandidate], min_score: float
    ) -> tuple[list[ExternalEvidenceCandidate], int]:
        valid: list[ExternalEvidenceCandidate] = []
        rejected = 0
        for candidate in candidates:
            if not candidate.verified_source:
                rejected += 1
                logger.info(
                    "[EXTERNAL_SOURCE_REJECTED] source_id=%s document_id=%s chunk_id=%s",
                    candidate.source_id,
                    candidate.document_id,
                    candidate.chunk_id,
                )
                continue
            # Chroma 검색 결과만 유사도 임계값을 적용한다. NAVER 뉴스처럼 API가
            # 유사도 점수를 제공하지 않는 결과(None)를 0점으로 만들어 전부 버리지는
            # 않는다. 최종 랭킹에서는 기존 계약대로 semantic_score=0.0을 사용한다.
            if candidate.semantic_score is not None:
                if not _is_finite(candidate.semantic_score):
                    continue
                if candidate.semantic_score < min_score:
                    continue
            valid.append(candidate)
        return valid, rejected

    @staticmethod
    def _dedupe(candidates: list[ExternalEvidenceCandidate]) -> list[ExternalEvidenceCandidate]:
        seen: set[tuple[str, str]] = set()
        deduped: list[ExternalEvidenceCandidate] = []
        for candidate in candidates:
            key = (candidate.document_id, candidate.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _aggregate_by_document(
        candidates: list[ExternalEvidenceCandidate],
    ) -> dict[tuple[str, str], ExternalEvidenceCandidate]:
        """동일 (source_id, document_id)의 여러 청크 중 semantic_score가 가장 높은 청크
        하나만 대표로 남긴다 — 같은 자료가 결과에 여러 번 반복되지 않게 한다."""
        best: dict[tuple[str, str], ExternalEvidenceCandidate] = {}
        for candidate in candidates:
            score = candidate.semantic_score if candidate.semantic_score is not None else 0.0
            key = (candidate.source_id, candidate.document_id)
            existing = best.get(key)
            existing_score = (existing.semantic_score or 0.0) if existing else -math.inf
            if existing is None or existing_score < score:
                best[key] = candidate
        return best

    def _build_result(
        self, request: ExternalResearchRequest, candidate: ExternalEvidenceCandidate
    ) -> ExternalEvidenceResult:
        semantic_score = candidate.semantic_score if candidate.semantic_score is not None else 0.0
        role_score = compute_role_score(candidate.supported_roles, request.reviewer_role)
        criteria_score = compute_criteria_score(candidate.evaluation_criteria, request.evaluation_criteria)
        date_status, freshness_score = compute_freshness(
            evidence_type=candidate.evidence_type,
            reference_date=candidate.reference_date,
            published_at=candidate.published_at,
            config=self._freshness_config,
        )
        semantic_weight = self._config.semantic_weight
        role_weight = self._config.role_weight
        criteria_weight = self._config.criteria_weight
        freshness_weight = self._config.freshness_weight
        if candidate.evidence_type.value == "news":
            # 뉴스만 최신성 비중을 높인다. 나머지 신호의 상대 비율은 유지하면서
            # 총 가중치를 1로 맞춰 다른 자료 유형과 점수 크기를 비교할 수 있게 한다.
            freshness_weight = self._config.news_freshness_weight
            non_fresh_total = semantic_weight + role_weight + criteria_weight
            if non_fresh_total > 0:
                scale = (1.0 - freshness_weight) / non_fresh_total
                semantic_weight *= scale
                role_weight *= scale
                criteria_weight *= scale

        final_score = compute_final_score(
            semantic_score=semantic_score,
            role_score=role_score,
            criteria_score=criteria_score,
            freshness_score=freshness_score,
            semantic_weight=semantic_weight,
            role_weight=role_weight,
            criteria_weight=criteria_weight,
            freshness_weight=freshness_weight,
        )

        candidate_criteria_norm = {c.strip().lower() for c in candidate.evaluation_criteria}
        matched_criteria = [
            criterion for criterion in request.evaluation_criteria
            if criterion.strip().lower() in candidate_criteria_norm
        ]

        return ExternalEvidenceResult(
            source_id=candidate.source_id,
            document_id=candidate.document_id,
            chunk_id=candidate.chunk_id,
            title=candidate.title,
            evidence_type=candidate.evidence_type,
            publisher=candidate.publisher,
            source_url=candidate.source_url,
            domain=candidate.domain,
            supported_roles=candidate.supported_roles,
            matched_criteria=matched_criteria,
            quote=candidate.content,
            reference_date=candidate.reference_date,
            published_at=candidate.published_at,
            retrieved_at=candidate.retrieved_at,
            date_status=date_status,
            region=candidate.region,
            period=candidate.period,
            metric_name=candidate.metric_name,
            metric_value=candidate.metric_value,
            metric_unit=candidate.metric_unit,
            page=candidate.page,
            section=candidate.section,
            semantic_score=semantic_score,
            role_score=role_score,
            criteria_score=criteria_score,
            freshness_score=freshness_score,
            final_score=final_score,
            retrieval_source=candidate.retrieval_source,
        )


__all__ = ["ExternalResearchService"]
