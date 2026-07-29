"""
Ideation Evidence Service
=============================
용준/Claude(2026-07-20). "아이디어 발전 회의(ideation)" 전문가(planning_expert/dev_expert)에게
넘길 근거를 검색한다.

기존 MeetingEvidenceOrchestrationService(meeting_evidence_service.py)는 (persona_id,
criterion_id) 단위로 동작하는데, ideation 모드에는 rubric criterion 개념이 없다 — 채점 기준별로
배정되는 게 아니라 공모전 공고·평가기준과 사용자 아이디어를 놓고 대화할 뿐이다. 그래서 이 모듈은
criterion 단위 오케스트레이션(사전 근거충족도 판정, RAG-004 사후 링크)을 그대로 가져다 쓰지 않고,
더 가벼운 함수 하나로 persona_id -> 고정 role_id 매핑만 하고 RoleAwareRetrievalService.
search_by_role()을 직접 호출한다.

role_id는 새로 만들지 않고 기존 RAG-003 role 레지스트리에 이미 있는 값을 재사용한다
(ai/rag/orchestration/role_mapping.py의 competition/government_support 매핑에서 이미 확인된 값:
planning_expert -> "planning"(문서 구조·기획 관점), dev_expert -> "technology"(기술 구성·구현
가능성)가 기존 위원들에도 쓰이고 있다). ai/meeting을 import하지 않는다(회의 ↔ RAG 분리 유지,
기존 meeting_evidence_service.py와 같은 원칙).
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from ai.rag.domain.document_types import normalize_document_type, preferred_document_types
from ai.rag.evidence_linking.relevance import calculate_relevance_score, extract_keywords
from ai.rag.integration.meeting_evidence_adapter import build_meeting_retrieved_evidence
from ai.rag.integration.schemas import PersonaRoleSearchResponse
from ai.rag.role_retrieval.service import RoleAwareRetrievalService

logger = logging.getLogger(__name__)

EvidenceLookup = Callable[[str, str], list[dict]]

# persona_id -> RAG-003 role_id. ideation은 committee 화이트리스트가 없는 자유 모드라
# role_mapping.py의 strict 정책(매핑 없으면 예외)을 그대로 따르지 않고, 매핑에 없는
# persona_id는 조용히 None(semantic-only 검색)으로 처리한다 — 진행자(ideation_facilitator)처럼
# 애초에 근거 검색이 필요 없는 역할도 있기 때문이다.
_PERSONA_ROLE_MAPPING: dict[str, str] = {
    "planning_expert": "planning",
    "dev_expert": "technology",
}


def resolve_ideation_role_id(persona_id: str) -> Optional[str]:
    """ideation 전문가 persona_id에 대응하는 RAG-003 role_id. 매핑에 없으면 None."""
    return _PERSONA_ROLE_MAPPING.get(persona_id)


# 용준/Claude(2026-07-22, 요청: 역할별 검색 데이터 구성) — 기존 metadata를 확인한 결과
# document_role은 backend/app/models/document.py 기준 "criteria"(공고문·평가기준)와
# "target"(평가 대상 문서/기획서) 두 값만 실제로 쓰인다("domain"/"similar_case"는 이
# 색인 파이프라인에 존재하지 않는 값이라 임의로 가정하지 않는다 — 요청 사항 그대로).
# planning_expert는 공고문·평가기준(criteria)을 우선 참고하고, dev_expert는 사용자가 이미
# 밝힌 아이디어/자료(target)를 우선 참고하되 criteria의 실현 가능성 관련 항목도 일부
# 참고한다 — top_k=5 기준 쿼터.
_DOCUMENT_ROLE_QUOTAS: dict[str, dict[str, int]] = {
    "planning_expert": {"criteria": 3, "target": 2},
    "dev_expert": {"target": 3, "criteria": 2},
}
# 쿼터 계산을 위해 검색해 둘 후보 풀 크기 배수 — top_k보다 넉넉히 검색해야 role별로
# 나눠 담을 후보가 부족하지 않다(정확한 candidate_k는 RoleAwareRetrievalService 자체
# 기본값을 그대로 따르되, 여기서는 role 필터링을 위해 한 번 더 넉넉한 top_k를 요청한다).
_CANDIDATE_POOL_MULTIPLIER = 4

# 용준/Claude(2026-07-29, 요청: target/criteria/외부근거/전문가판단 분리) — RAG-007
# (ai/rag/external_research, ExternalEvidenceResult.evidence_type)과 RAG-006
# (ai/rag/similar_cases)을 이 서비스가 구성하는 근거 풀에 실제로 합치기 위한 태깅 규칙.
# document_role(target/criteria)과는 별개의 축이다 — project 문서는 document_role을 그대로
# source_type으로 쓰고, 외부 자료는 evidence_type/출처 종류에 따라 source_type을 매긴다.
# 매핑에 없는 evidence_type은 "external_other"로 두고 news보다 우선순위를 높이지 않는다
# (요청: 뉴스가 공식 문서/통계를 앞지르면 안 됨 — 미분류를 news보다 낮은 자리에 둔다).
_EXTERNAL_SOURCE_TYPE_BY_EVIDENCE_TYPE: dict[str, str] = {
    "statistics": "official_statistics",
    "public_data": "official_statistics",
    "law": "official_report",
    "policy": "official_report",
    "guideline": "official_report",
    "research_report": "official_report",
    "market": "official_report",
    "press_release": "press_release",
    "news": "news",
}
_SIMILAR_CASE_SOURCE_TYPE = "similar_case"
# 요청 우선순위: 공모전 공식문서(criteria, 이 함수 밖 role-quota에서 이미 최우선 처리) >
# 정부·공공기관 공식 보고서·통계 > 유사 공공서비스 사례 > 공식 보도자료 > 일반 뉴스 >
# 미분류.
_EXTERNAL_SOURCE_TYPE_PRIORITY: dict[str, int] = {
    "official_statistics": 0,
    "official_report": 1,
    "similar_case": 2,
    "press_release": 3,
    "news": 4,
    "external_other": 5,
    "official_page_summary": 6,
}
_DEFAULT_MAX_EXTERNAL_EVIDENCE = 2


def classify_external_source_type(item: dict) -> str:
    """RAG-006(ideation_similar_case_service._result_to_dict가 붙이는 "_source":
    "similar_case" 마커) 결과와 RAG-007(evidence_type 필드) 결과를 하나의 source_type
    축으로 정규화한다. 두 값 다 없으면(예: project target/criteria 항목이 실수로 들어온
    경우) "external_other"로 보수적으로 분류한다 — 임의로 criteria/target을 사칭하지
    않는다."""
    if item.get("_source") == "similar_case":
        return _SIMILAR_CASE_SOURCE_TYPE
    if item.get("summary_only") or item.get("source_type") == "official_page_summary":
        return "official_page_summary"
    evidence_type = item.get("evidence_type")
    return _EXTERNAL_SOURCE_TYPE_BY_EVIDENCE_TYPE.get(evidence_type, "external_other")


def compose_ideation_evidence_pool(
    project_items: list[dict],
    external_items: list[dict],
    similar_case_items: list[dict],
    *,
    max_external_evidence: int = _DEFAULT_MAX_EXTERNAL_EVIDENCE,
) -> list[dict]:
    """search_ideation_evidence()가 이미 구성한 project_items(document_role=target/criteria,
    role-quota 적용 완료)의 개수·순서는 전혀 건드리지 않고, RAG-007/RAG-006 후보를 "추가
    슬롯"으로만 얹는다 — 기존 document_fact_query·role-quota 경로 회귀를 방지한다.

    external_items/similar_case_items는 우선순위(_EXTERNAL_SOURCE_TYPE_PRIORITY) +
    final_score(없으면 score) 내림차순으로 정렬해 상위 max_external_evidence개만 남긴다.
    관련 있는 후보가 그보다 적으면 있는 만큼만 반환한다 — 무관한 자료로 자리를 채우지
    않는다(요청: quota를 억지로 채우지 않음). 이 함수는 I/O를 하지 않는 순수 병합
    함수이며, ai.rag는 ai.meeting을 import하지 않는 기존 경계를 유지한다 — 실제 검색은
    호출자가 이미 마친 결과를 받는다.

    각 반환 항목에는 source_type이 태깅된다: project_items는 document_role 그대로,
    external/similar_case 항목은 classify_external_source_type() 결과. 반환 리스트는
    MeetingRetrievedEvidence TypedDict를 엄격히 만족하지 않을 수 있다(외부 항목은
    organization/published_at/source_url 등 추가 필드를 갖는 plain dict) — retrieved/
    turn_evidence는 원래도 TypedDict로 런타임 강제되지 않는 plain dict 리스트이므로
    하위 호환이다. ref 부여는 이 함수의 책임이 아니다(호출자의 기존 "E{n}" 로직 재사용)."""
    tagged_project: list[dict] = []
    for item in project_items:
        tagged = dict(item)
        tagged.setdefault("source_type", item.get("document_role"))
        tagged_project.append(tagged)

    tagged_external: list[dict] = []
    for item in external_items:
        tagged = dict(item)
        tagged["source_type"] = classify_external_source_type(tagged)
        tagged.setdefault("document_role", None)
        tagged_external.append(tagged)
    for item in similar_case_items:
        tagged = dict(item)
        tagged["_source"] = "similar_case"
        tagged["source_type"] = _SIMILAR_CASE_SOURCE_TYPE
        tagged.setdefault("document_role", None)
        tagged_external.append(tagged)

    tagged_external.sort(
        key=lambda item: (
            _EXTERNAL_SOURCE_TYPE_PRIORITY.get(item.get("source_type"), 9),
            -(item.get("final_score") or item.get("score") or 0.0),
        )
    )
    selected_external = tagged_external[: max(0, max_external_evidence)]

    return tagged_project + selected_external

# 긴 아이디어/역할별 query는 target 검색에는 유리하지만, 공모전 평가표의 짧은 세부 문항은
# 의미가 희석돼 Top-5 밖으로 밀릴 수 있다. 현재 쟁점 제목을 topic_query에서 꺼내 한 번 더
# 짧게 검색한 뒤 criteria 후보만 합친다. LLM 호출은 없고 기존 KURE/Chroma만 한 번 더 쓴다.
_ISSUE_FOCUSED_QUERY_TERMS: dict[str, str] = {
    "문제 정의": "도시 문제의 설정 사용자 피해 위험 발생 원인 현재 한계",
    "목표 사용자": "목표 사용자 수혜자 시민 참여 사용자 요구",
    "핵심 가치": "시민 편익 삶의 질 사회적 가치",
    "공모전 적합성": "평가 기준 공모 목적 AI 스마트시티 적합성",
    "차별성과 고객 가치": "혁신성 기존 방식 대비 차별성 개선 효과 고객 가치",
    "MVP 범위": "실현 가능성 기술 완성도 경제성 핵심 기능 범위",
    "데이터 확보 방안": "데이터 활용 수집 품질 보안",
    "AI 활용 방식": "AI 기술 활용 도시 문제 해결 운영 혁신",
    "확장 로드맵": "확장성 확산 가능성 지속 가능성 추진 전략",
}

_PRE_TARGET_PHASES = frozenset(
    {
        "problem_discovery",
        "awaiting_problem_focus_selection",
        "problem_focus_selection",
        "idea_divergence",
        "idea_conflict_and_merge",
        "awaiting_conflict_resolution",
        "conflict_resolution",
        "candidate_generation",
        "awaiting_candidate_selection",
    }
)
_CURRENT_ISSUE_PATTERN = re.compile(r"(?:^|\|)\s*현재 쟁점:\s*([^|]+)")

# 용준/Claude(2026-07-29, 요청: B05 회귀 — "최종 확정한 방향이 뭔가요" 질문에서 아직 검토
# 중인 후보 목록 chunk("현재 활성 해결 방향(...)으로 검증을 진행합니다")가 실제 확정된
# 아이디어보다 앞서 top_k 안에 들어, 위원이 후보를 최종안으로 오인하는 문제가 실측됐다
# (Ragas Context Precision=0.0, Faithfulness=0.0). 항목을 삭제하지 않고 순서만 바꾼다 —
# quota가 남으면 후보 목록도 여전히 포함될 수 있다.
_FINAL_DIRECTION_QUERY_RE = re.compile(r"최종|확정된|확정한|확정 여부|선택한 아이디어|선택된 아이디어")
_CONFIRMATION_TEXT_RE = re.compile(r"최종 확정합니다|이 방향으로 확정")
_PROVISIONAL_CANDIDATE_TEXT_RE = re.compile(r"현재 활성 해결 방향")


def _is_final_direction_query(topic_query: str) -> bool:
    return bool(_FINAL_DIRECTION_QUERY_RE.search(topic_query or ""))


def _prioritize_final_direction(items: list[dict], topic_query: str) -> list[dict]:
    """"최종/확정" 질문일 때만 순서를 조정한다(그 외 질문은 그대로 반환). 우선순위:
    0) ideation_source_type=="ideation_candidate"(선택된 아이디어 문서 자체) 1) "이 방향으로
    최종 확정합니다" 류 확정 발언 2) 그 외 3) "현재 활성 해결 방향(...)으로 검증을
    진행합니다" 류 검토 중 후보 목록(맨 뒤로 미룸). sorted()는 안정 정렬이라 같은 순위 내
    원래 상대 순서는 유지된다."""
    if not _is_final_direction_query(topic_query):
        return items

    def _rank(item: dict) -> int:
        text = item.get("text") or ""
        if item.get("ideation_source_type") == "ideation_candidate":
            return 0
        if _CONFIRMATION_TEXT_RE.search(text):
            return 1
        if _PROVISIONAL_CANDIDATE_TEXT_RE.search(text):
            return 3
        return 2

    return sorted(items, key=_rank)


def _build_issue_focused_query(topic_query: str) -> str | None:
    """topic_query의 구조화된 '현재 쟁점'을 짧은 criteria 검색어로 변환한다."""
    match = _CURRENT_ISSUE_PATTERN.search(topic_query or "")
    if not match:
        return None
    issue_title = match.group(1).strip()
    terms = _ISSUE_FOCUSED_QUERY_TERMS.get(issue_title)
    return f"{issue_title} {terms}" if terms else None


def _search_issue_focused_criteria(
    role_retrieval_service: RoleAwareRetrievalService,
    *,
    persona_id: str,
    topic_query: str,
    project_id: str,
    top_k: int,
) -> list[dict]:
    """현재 쟁점 전용 semantic 검색 결과 중 criteria 문서만 반환한다."""
    focused_query = _build_issue_focused_query(topic_query)
    if not focused_query:
        return []
    try:
        response = role_retrieval_service.search_by_role(
            query=focused_query,
            project_id=project_id,
            role_id=None,
            top_k=top_k,
        )
    except Exception:
        logger.exception(
            "[IDEATION_ISSUE_FOCUSED_CRITERIA_SEARCH_FAILED] persona_id=%s project_id=%s",
            persona_id,
            project_id,
        )
        return []
    items = build_meeting_retrieved_evidence(
        [PersonaRoleSearchResponse(persona_id=persona_id, response=response, role_id=None)]
    )
    return [dict(item) for item in items if item.get("document_role") == "criteria"]


def _search_plain_query_candidates(
    role_retrieval_service: RoleAwareRetrievalService,
    *,
    persona_id: str,
    topic_query: str,
    project_id: str,
    top_k: int,
) -> list[dict]:
    """Search the original query without role expansion.

    Exact administrative questions (schedule, email, eligibility, score) can
    lose their key terms when the planning/technology role instruction is
    prepended.  This second, small candidate search preserves those terms; the
    normal role-aware result still participates in final composition.
    """
    try:
        response = role_retrieval_service.search_by_role(
            query=topic_query,
            project_id=project_id,
            role_id=None,
            top_k=top_k,
        )
    except Exception:
        logger.exception(
            "[IDEATION_PLAIN_QUERY_SEARCH_FAILED] persona_id=%s project_id=%s",
            persona_id,
            project_id,
        )
        return []
    return [
        dict(item)
        for item in build_meeting_retrieved_evidence(
            [PersonaRoleSearchResponse(persona_id=persona_id, response=response, role_id=None)]
        )
    ]


def _rank_by_document_type(items: list[dict], topic_query: str) -> list[dict]:
    """Annotate legacy chunks and prefer the document type implied by query."""
    preferred = preferred_document_types(topic_query)
    preference_rank = {document_type: index for index, document_type in enumerate(preferred)}
    annotated: list[dict] = []
    for position, item in enumerate(items):
        typed_item = dict(item)
        typed_item["document_type"] = normalize_document_type(
            typed_item.get("document_type"),
            document_name=typed_item.get("document_name"),
            text=typed_item.get("text"),
        )
        typed_item["_retrieval_position"] = position
        annotated.append(typed_item)

    if preferred:
        annotated.sort(
            key=lambda item: (
                preference_rank.get(item["document_type"], len(preferred)),
                -(item.get("final_score") or item.get("score") or 0.0),
                item["_retrieval_position"],
            )
        )
    for item in annotated:
        item.pop("_retrieval_position", None)
    return annotated


def _scope_target_evidence(
    items: list[dict], *, session_id: Optional[str], selected_candidate_document_id: Optional[str]
) -> list[dict]:
    """용준/Claude(2026-07-22, 요청: 세션 범위 검색 + 후보 변경 시 이전 candidate target 제외).

    Chroma where 절은 project_id(+document_id)만 지원하므로(ai/rag/retrieval/chroma_store.py::
    _build_where), session/후보 범위 필터링은 여기서 검색 결과를 후처리한다 — 넓게 검색한 뒤
    metadata로 안전하게 걸러내는 방식(요청 5번의 두 대안 중 하나).

    - ideation_source_type이 없는 항목(일반 project criteria/target 문서)은 항상 통과시킨다.
    - ideation_source_type="ideation_candidate"(선택된 후보 target)는 document_id가 현재
      세션의 "현재 선택된" 후보 document_id(selected_candidate_document_id)와 일치할 때만
      통과한다 — 사용자가 후보를 다시 선택/결합하면 이전 후보의 target은 회의 이력으로
      Chroma에 남아있어도 더 이상 근거로 쓰이지 않는다(요청 17-5번).
    - ideation_source_type="user_session_answer"(사용자 답변 target)는 session_id가 현재
      세션과 일치할 때만 통과한다 — 다른 회의 세션의 사용자 답변이 섞이지 않는다(요청 5번)."""
    scoped: list[dict] = []
    for item in items:
        ideation_source_type = item.get("ideation_source_type")
        if ideation_source_type is None:
            scoped.append(item)
        elif ideation_source_type == "ideation_candidate":
            if selected_candidate_document_id and item.get("document_id") == selected_candidate_document_id:
                scoped.append(item)
        elif ideation_source_type == "user_session_answer":
            if session_id and item.get("session_id") == session_id:
                scoped.append(item)
        else:
            scoped.append(item)
    return scoped


def _diversify_by_document(candidates: list[dict], quota: int) -> tuple[list[dict], list[dict]]:
    """용준/Claude(2026-07-30, 요청 7번 — "하나의 URL 본문 청크가 top-k를 전부 차지하지 않게
    하세요") — candidates(final_score 내림차순 정렬 유지)에서 quota 자리를 채울 때, 같은
    document_id(공고 URL 본문 하나 또는 첨부파일 하나)의 청크가 먼저 나온 순서 그대로 quota를
    모두 차지하지 않도록 서로 다른 document_id를 한 번씩 우선 채운다. 서로 다른 문서 수가
    quota보다 적으면(다양화할 후보 자체가 부족) 남은 자리는 점수 순으로 그대로 채운다 —
    존재하지 않는 문서를 억지로 만들어 채우지 않는다.

    반환값: (선택된 항목, 다양화로 밀려난 나머지 — 기존 leftover 보충 경로에 합류시킨다)."""
    selected: list[dict] = []
    seen_documents: set[str] = set()
    leftover: list[dict] = []
    for item in candidates:
        if len(selected) >= quota:
            leftover.append(item)
            continue
        document_id = item.get("document_id", "")
        if document_id and document_id in seen_documents:
            leftover.append(item)
            continue
        selected.append(item)
        if document_id:
            seen_documents.add(document_id)
    if len(selected) < quota and leftover:
        remaining_needed = quota - len(selected)
        selected.extend(leftover[:remaining_needed])
        leftover = leftover[remaining_needed:]
    return selected, leftover


def _compose_by_document_role(
    items: list[dict], *, persona_id: str, top_k: int, phase: Optional[str] = None
) -> tuple[list[dict], list[str]]:
    """검색된 후보(items, final_score 내림차순 정렬 상태 유지)를 persona별
    _DOCUMENT_ROLE_QUOTAS에 맞춰 재구성한다. 원하는 role의 후보가 전혀 없으면
    missing_document_roles에 기록한다(그 role을 무관한 다른 문서로 억지로 채우지 않는다
    — 부족한 만큼만 다른 role/미분류 후보로 보충한다).

    반환값: (구성된 top_k개 이하의 리스트, missing_document_roles)."""
    # Before a candidate exists, target evidence is not merely missing: it is
    # conceptually inapplicable.  Do not reserve quota or emit a false warning.
    quotas = (
        {"criteria": top_k}
        if phase in _PRE_TARGET_PHASES and persona_id in _DOCUMENT_ROLE_QUOTAS
        else _DOCUMENT_ROLE_QUOTAS.get(persona_id)
    )
    if not quotas:
        return items[:top_k], []

    buckets: dict[str, list[dict]] = {role: [] for role in quotas}
    unclassified: list[dict] = []
    for item in items:
        role = item.get("document_role")
        if role in buckets:
            buckets[role].append(item)
        else:
            unclassified.append(item)

    missing_document_roles = [role for role, candidates in buckets.items() if not candidates]

    composed: list[dict] = []
    used_ids: set[tuple[str, str]] = set()
    leftover: list[dict] = []
    for role, quota in quotas.items():
        # criteria는 공고 URL 본문/URL 첨부파일/직접 업로드가 한 project에 섞여 색인되므로
        # document_id 다양성을 적용한다. target은 보통 선택된 아이디어 문서 하나뿐이라
        # 다양화할 대상이 없어 기존 점수 순 그대로 둔다.
        if role == "criteria":
            taken, remainder = _diversify_by_document(buckets[role], quota)
        else:
            taken, remainder = buckets[role][:quota], buckets[role][quota:]
        leftover.extend(remainder)
        for item in taken:
            key = (item.get("document_id", ""), item.get("chunk_id", ""))
            if key not in used_ids:
                used_ids.add(key)
                composed.append(item)

    # 쿼터를 채우지 못한 role이 있으면(예: target 후보가 2개뿐이라 3개 쿼터를 못 채움)
    # 다른 role의 남은 후보나 미분류 후보로 top_k까지 채운다 — "관련 없는 공고문으로 전부
    # 채우지 마세요"는 missing_document_roles를 아예 숨기지 말라는 뜻이지, 검색 결과 자체를
    # 강제로 버리라는 뜻은 아니므로 이미 검색된(관련성 있다고 판단된) 후보로만 보충한다.
    fill_candidates = sorted(
        leftover + unclassified, key=lambda item: item.get("final_score") or item.get("score") or 0.0, reverse=True
    )
    for item in fill_candidates:
        if len(composed) >= top_k:
            break
        key = (item.get("document_id", ""), item.get("chunk_id", ""))
        if key in used_ids:
            continue
        used_ids.add(key)
        composed.append(item)

    return composed[:top_k], missing_document_roles


def _search_candidate_target_direct(
    role_retrieval_service: RoleAwareRetrievalService,
    *,
    persona_id: str,
    role_id: Optional[str],
    topic_query: str,
    project_id: str,
    document_id: str,
    top_k: int,
) -> list[dict]:
    """용준/Claude(2026-07-23, 요청: stale closure 수정 + target starvation 보강) — 선택된
    후보의 target document_id를 이미 알고 있으므로, project-wide semantic top-N 순위에
    끼어들었는지에 의존하지 않고 그 document_id만 직접 검색한다(RAG-003
    RoleAwareRetrievalService.search_by_role(document_id=...)가 이미 RAGIndexingService.
    search()에 document_id 필터를 그대로 전달한다 — 별도 Chroma 쿼리를 새로 만들지 않는다).

    criteria 청크가 top-N 후보 풀을 모두 차지해 target이 project-wide 검색 결과에서 아예
    빠지더라도(요청 진단의 핵심 원인), 이 직접 검색은 그 top-N 경쟁과 무관하게 항상 그
    document_id의 청크를 찾는다. 검색 실패/결과 없음은 fail-closed로 빈 리스트를 반환한다
    (다른 검색 실패와 동일한 정책 — target을 가짜로 채우지 않는다). 반환된 항목의
    document_id가 요청한 값과 다르면 방어적으로 제외한다(다른 문서가 섞여 들어오는 것을
    원천 차단)."""
    try:
        response = role_retrieval_service.search_by_role(
            query=topic_query,
            project_id=project_id,
            role_id=role_id,
            document_id=document_id,
            top_k=top_k,
        )
    except Exception:
        logger.exception(
            "[IDEATION_CANDIDATE_TARGET_DIRECT_SEARCH_FAILED] persona_id=%s project_id=%s document_id=%s",
            persona_id,
            project_id,
            document_id,
        )
        return []

    items = build_meeting_retrieved_evidence(
        [PersonaRoleSearchResponse(persona_id=persona_id, response=response, role_id=role_id)]
    )
    return [dict(item) for item in items if item.get("document_id") == document_id]


def _raw_question_component(topic_query: str) -> str:
    """topic_query는 ai/meeting의 _topic_query()가 항상 "idea_summary(사용자 질문/아이디어
    원문) | 현재 쟁점: ... | 검토 관점: ..." 순서로 조립한다(순서가 계약이다 — _topic_query
    docstring 참고). 그 첫 " | " 이전 구간만 뽑으면 쟁점/역할 관점 텍스트 없이 사용자 질문
    원문만 남는다."""
    if not topic_query:
        return topic_query
    return topic_query.split(" | ", 1)[0].strip()


def _reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = 60) -> list[dict]:
    """용준/Claude(2026-07-29, 요청: RAG 품질 개선 3단계 — dual-query 방식) — 여러 순위
    목록을 병합한다(Reciprocal Rank Fusion). 특정 리스트에만 있는 항목도 버려지지 않고,
    여러 리스트에 공통으로 높은 순위인 항목이 위로 올라온다. 항목 동일성은
    (document_id, chunk_id)로 판단하고, 같은 키가 여러 리스트에 있으면 먼저 등장한 dict의
    메타데이터를 그대로 쓴다(내용은 어느 리스트에서 와도 같은 chunk)."""
    scores: dict[tuple[str, str], float] = {}
    first_seen: dict[tuple[str, str], dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = (item.get("document_id", ""), item.get("chunk_id", ""))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, item)
    ordered_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [first_seen[key] for key in ordered_keys]


# 용준/Claude(2026-07-29, 요청: 생성 품질 개선 3단계 — document_fact_query 전용 검색).
# ai/meeting/graph/ideation_conv_nodes.py::_QUERY_TYPE_KEYWORDS["document_fact_query"]와
# 같은 키워드 세트를 그대로 옮겼다(값 자체를 공유 모듈로 빼지 않는 이유는 위
# _FINAL_DIRECTION_QUERY_RE와 같다 — ai/meeting과 ai/rag는 프로덕션 코드에서 서로 import하지
# 않는다). A04 전용 키워드나 chunk_id는 여기 없다 — 일반화된 세트 그대로다.
_DOCUMENT_FACT_QUERY_MIN_SCORE = 3
_DOCUMENT_FACT_QUERY_KEYWORDS: dict[str, int] = {
    "결격": 3, "결격 사유": 4,
    "배점": 4, "배점 기준": 4, "신청 자격": 4, "참가 자격": 4, "참여 자격": 4,
    "제출 서류": 4, "제출서류": 4, "제출해야": 2, "일정": 2, "마감": 3,
    "공고문": 2, "몇 점": 3, "기한": 2, "요건": 2, "어떻게 되나요": 1,
    "어떤 내용": 1, "무엇을 작성": 2,
    # 용준/Claude(2026-07-29) — "~에 어떤 내용을 작성해야 하나요" 류 신청서식/보고서 작성
    # 항목 질문 일반화(요청 §3 "작성할 내용을 묻는 경우" 범주). A04 하나만을 위한 키워드가
    # 아니라 신청서식·수행보고서 작성 항목을 묻는 질문 전반에 적용된다.
    "작성해야": 3, "작성 방법": 3, "작성 요령": 3,
    # 용준/Claude(2026-07-29, 요청: B02 회귀 — 수치·규모형 공식 문서 질문 일반화). "몇 개
    # 과제가 선정되나요" 같은 질문은 B02 전용이 아니라 이 범주 전체(몇 개/몇 건/몇 명,
    # 선정·모집·지원 규모, 금액/횟수)에 적용된다 — ai/meeting/graph/ideation_conv_nodes.py::
    # _QUERY_TYPE_KEYWORDS["document_fact_query"]와 동일 가중치.
    "몇 개": 4, "몇 건": 4, "몇 명": 4, "선정 규모": 4, "선정 개수": 4,
    "선정 수": 4, "모집 수": 4, "모집 규모": 4, "지원 규모": 4, "지원 금액": 4,
    "금액": 3, "횟수": 3,
}
# document_fact_query 재정렬 보조 가산점(요청: 하드코딩 chunk_id 없이 일반 규칙화).
_WRITING_INTENT_RE = re.compile(r"작성|기재")
_WRITING_MARKER_RE = re.compile(r"작성\s*요령|작성\s*항목|기재")
_SCALE_INTENT_RE = re.compile(r"선정|배점|규모|개수|몇\s*개|몇\s*점")
_DIGIT_RE = re.compile(r"\d")
_ELIGIBILITY_INTENT_RE = re.compile(r"자격|결격|제외")
_ELIGIBILITY_MARKER_RE = re.compile(r"자격|결격|제외|참가\s*대상|참여\s*기관")
# planner의 MIN_ISSUE_RELEVANCE_SCORE(ideation_evidence_planner.py)와 동일한 임계값을
# 재사용한다 — "이 근거가 이 질문에 실제로 답하는가"를 관대하지 않게 보는 목적이 같다.
_DOCUMENT_FACT_SUFFICIENCY_THRESHOLD = 0.15
_DOCUMENT_FACT_CANDIDATE_TOP_K = 15
_DOCUMENT_FACT_CANDIDATE_K = 45


def _is_document_fact_query(topic_query: str) -> bool:
    """LLM 미사용, 결정론적. 임계값 미만이면 False → 기존 role-quota 경로 그대로."""
    haystack = _raw_question_component(topic_query) or topic_query or ""
    score = sum(weight for keyword, weight in _DOCUMENT_FACT_QUERY_KEYWORDS.items() if keyword in haystack)
    return score >= _DOCUMENT_FACT_QUERY_MIN_SCORE


def _search_document_fact_candidates(
    role_retrieval_service: RoleAwareRetrievalService,
    *,
    persona_id: str,
    topic_query: str,
    project_id: str,
) -> list[dict]:
    """역할 쿼터를 타지 않는 넓은 후보 풀 검색 — Query A(질문 원문) + Query B(topic_query
    전체, 쟁점 텍스트 포함)를 role_id=None으로 각각 top_k=15(candidate_k=45)까지 뽑고
    RRF로 합친다. role_id=None이면 role_score가 항상 0이라(reranker.py::compute_role_score)
    사실상 semantic + 중복제거 + 섹션 다양성만 적용된 순수 후보 풀이다."""

    def _search(query: str) -> list[dict]:
        if not query:
            return []
        try:
            response = role_retrieval_service.search_by_role(
                query=query,
                project_id=project_id,
                role_id=None,
                top_k=_DOCUMENT_FACT_CANDIDATE_TOP_K,
                candidate_k=_DOCUMENT_FACT_CANDIDATE_K,
            )
        except Exception:
            logger.exception(
                "[IDEATION_DOCUMENT_FACT_SEARCH_FAILED] persona_id=%s project_id=%s query=%s",
                persona_id,
                project_id,
                query[:80],
            )
            return []
        return [
            dict(item)
            for item in build_meeting_retrieved_evidence(
                [PersonaRoleSearchResponse(persona_id=persona_id, response=response, role_id=None)]
            )
        ]

    query_a_items = _search(_raw_question_component(topic_query))
    query_b_items = _search(topic_query)
    return _reciprocal_rank_fusion(query_a_items, query_b_items)


def _document_fact_auxiliary_boost(item: dict, topic_query: str) -> float:
    """요청 §3 규칙 — semantic 점수 하나로만 최종 순위를 정하지 않도록, 질문 의도별로
    관련 마커가 있는 청크에 소폭 가산한다(0~0.3, calculate_relevance_score와 같은 0~1
    스케일에 얹을 수 있도록 작게 유지)."""
    text = (item.get("text") or "") + " " + (item.get("section") or "")
    boost = 0.0
    if _WRITING_INTENT_RE.search(topic_query) and _WRITING_MARKER_RE.search(text):
        boost += 0.15
    if _SCALE_INTENT_RE.search(topic_query) and _DIGIT_RE.search(text):
        boost += 0.1
    if _ELIGIBILITY_INTENT_RE.search(topic_query) and _ELIGIBILITY_MARKER_RE.search(text):
        boost += 0.15
    return min(boost, 0.3)


def _section_title_keyword_boost(item: dict, topic_query: str) -> float:
    """요청 §3 "질문 핵심어와 section 제목의 일치". 같은 신청서식 문서 안의 형제 섹션들은
    "< 작성 요령 >" 같은 보일러플레이트 문구를 전부 공유해 calculate_relevance_score의
    content_overlap만으로는 서로 잘 구분되지 않는다(실측: A04에서 이 보일러플레이트 공유
    때문에 정답 섹션이 형제 섹션들보다 낮은 점수를 받았다) — section 제목에 질문 핵심어가
    그대로 등장하는지를 별도의 강한 신호로 추가한다."""
    section = item.get("section") or ""
    if not section:
        return 0.0
    keywords = extract_keywords(_raw_question_component(topic_query))
    hits = sum(1 for keyword in keywords if len(keyword) >= 2 and keyword[:2] in section)
    return min(hits * 0.25, 0.5)


def _rerank_document_fact_candidates(items: list[dict], topic_query: str, top_k: int) -> list[dict]:
    """semantic_score와 calculate_relevance_score(재사용, 새 lexical overlap 구현 안 함)를
    가중 결합하고, section 제목 일치·의도별 보조 가산점을 더해 재정렬한다. RRF 순위나
    semantic 점수 단독으로 최종 순위를 정하지 않는다(요청 그대로).

    용준/Claude(2026-07-29) — 애초에 이 전용 경로를 만든 이유가 "semantic 유사도만으로는
    같은 문서 안의 형제 섹션(각자 다른 '작성 요령'을 담은 항목)을 구분 못 한다"였다(A04
    실측: 형제 섹션들이 semantic 점수가 서로 비슷하고, 게다가 "< 작성 요령 >" 보일러플레이트
    문구를 전부 공유해 calculate_relevance_score의 content_overlap조차 형제 섹션을 잘
    구분하지 못했다). 그래서 relevance 비중을 semantic과 동등 이상으로 두고,
    section 제목 일치를 별도의 강한 신호로 더한다."""
    scored: list[tuple[float, dict]] = []
    for item in items:
        semantic = item.get("semantic_score")
        semantic = semantic if isinstance(semantic, (int, float)) else (item.get("score") or 0.0)
        relevance = calculate_relevance_score(
            topic_query,
            item.get("text") or "",
            item.get("section"),
            item.get("document_name"),
        )
        combined = (
            0.35 * semantic
            + 0.35 * relevance
            + _section_title_keyword_boost(item, topic_query)
            + _document_fact_auxiliary_boost(item, topic_query)
        )
        scored.append((combined, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def _assess_document_fact_sufficiency(items: list[dict], topic_query: str) -> bool:
    """최상위 항목이 질문과 실제로 관련 있는지(개수/점수 임계값이 아니라 질의-내용 일치)를
    본다 — evidence_sufficiency 패키지는 개수/점수 임계값 게이트라 이 목적에는 안 맞아
    재사용하지 않는다(계획에서 확인)."""
    if not items:
        return False
    top = items[0]
    score = calculate_relevance_score(topic_query, top.get("text") or "", top.get("section"), top.get("document_name"))
    return score >= _DOCUMENT_FACT_SUFFICIENCY_THRESHOLD


def _rewrite_document_fact_query(topic_query: str) -> str:
    """1회만 재검색할 축약 질의 — 핵심 키워드만 남긴다(extract_keywords 재사용, 새 축약
    로직 안 만듦)."""
    keywords = extract_keywords(_raw_question_component(topic_query))
    return " ".join(sorted(keywords)) if keywords else topic_query


def _search_document_fact_evidence(
    role_retrieval_service: RoleAwareRetrievalService,
    *,
    persona_id: str,
    topic_query: str,
    project_id: str,
    top_k: int,
) -> list[dict]:
    fused = _search_document_fact_candidates(
        role_retrieval_service, persona_id=persona_id, topic_query=topic_query, project_id=project_id
    )
    reranked = _rerank_document_fact_candidates(fused, topic_query, top_k)
    if _assess_document_fact_sufficiency(reranked, topic_query):
        return reranked

    rewritten_query = _rewrite_document_fact_query(topic_query)
    if rewritten_query == topic_query:
        return reranked
    logger.info(
        "[IDEATION_DOCUMENT_FACT_INSUFFICIENT_RETRY] persona_id=%s project_id=%s rewritten_query=%s",
        persona_id,
        project_id,
        rewritten_query[:80],
    )
    fused_retry = _search_document_fact_candidates(
        role_retrieval_service, persona_id=persona_id, topic_query=rewritten_query, project_id=project_id
    )
    reranked_retry = _rerank_document_fact_candidates(fused_retry, topic_query, top_k)
    # 재검색 결과가 원래보다 못하면(빈 결과 등) 원래 결과를 유지한다 — 재시도가 있던 근거를
    # 지우지 않는다.
    return reranked_retry if reranked_retry else reranked


def search_ideation_evidence(
    persona_id: str,
    topic_query: str,
    project_id: str,
    role_retrieval_service: RoleAwareRetrievalService,
    top_k: int = 5,
    *,
    session_id: Optional[str] = None,
    selected_candidate_document_id: Optional[str] = None,
    phase: Optional[str] = None,
) -> list[dict]:
    """전문가 1명의 이번 턴 근거를 검색해 회의 그래프가 바로 쓸 수 있는 plain dict 목록으로
    반환한다. 검색 결과가 없거나 검색 자체가 실패하면 빈 리스트를 반환한다(fail-closed) —
    근거 없음은 ideation_common.txt의 근거 사용 규칙("근거 부족"으로 표시하고 사용자에게
    필요한 정보를 요청)이 프롬프트 레벨에서 처리하도록 위임한다.

    용준/Claude(2026-07-22, 요청: 역할별 검색 데이터 구성) — 기존에는 planning_expert/
    dev_expert가 같은 semantic 검색 결과 풀을 그대로 top_k개 받아, 실제로는 둘 다 대부분
    같은(가장 점수가 높은) 공고문 청크만 받는 문제가 있었다. top_k보다 넉넉한 후보 풀을
    검색한 뒤 _scope_target_evidence(세션/후보 범위 필터) -> _compose_by_document_role(역할별
    쿼터) 순서로 적용한다 — 세션 범위를 먼저 걸러야 다른 세션의 사용자 답변이 쿼터 자리를
    차지하지 않는다.

    용준/Claude(2026-07-23, 요청: target starvation 보강) — project-wide 검색만으로는
    criteria 청크가 top-N 후보 풀을 모두 차지해 target이 아예 검색되지 않는 문제가 실측
    확인됐다(실 사이트 로그: target_count=0). selected_candidate_document_id가 있으면
    project-wide 검색과 별도로 그 document_id를 직접 검색해(_search_candidate_target_direct)
    병합한다 — 직접 검색 결과를 우선 순위에 두되, project-wide 검색이 이미 찾은 target/
    criteria 결과를 대체하거나 가짜로 채우지 않는다(실제 Chroma 검색 결과만 병합).

    용준/Claude(2026-07-29, 요청: RAG 품질 개선 3단계 — dual-query 방식) — topic_query에
    쟁점/역할 관점 텍스트가 붙으면(정상적인 아이디어 논의 검색어 구성) 짧고 구체적인 사용자
    질문이 임베딩 유사도에서 희석되는 문제가 실측됐다(예: "결격 사유가 뭔가요"가 "기술
    구조와 구현 가능성..." 같은 훨씬 긴 역할 설명에 묻힘). 특정 키워드가 있을 때만 예외
    처리하는 대신(그 목록에 없는 표현은 여전히 희석됨 — 실측: "몇 개 과제가 선정되나요"),
    매 호출마다 항상 두 검색을 병행한다 — Query A(topic_query의 " | " 앞부분=사용자 질문
    원문만, role_id=None plain 검색)와 Query B(topic_query 전체, 역할 확장 검색). 두 결과를
    (document_id, chunk_id) 기준 중복 제거 후 Reciprocal Rank Fusion으로 합친다."""
    # 용준/Claude(2026-07-29, 요청: 생성 품질 개선 3단계) — document_fact_query는 "위원 역할
    # 적합성"이 아니라 "질문과 문서의 직접 관련성"이 기준이므로 _DOCUMENT_ROLE_QUOTAS 등
    # 아래 role-quota 파이프라인을 전혀 타지 않고 별도 경로로 처리한다. 판정이 False면
    # (session_state_query/expert_analysis_query/일반 discussion) 이 함수의 나머지는 100%
    # 기존 그대로 실행된다.
    if _is_document_fact_query(topic_query):
        composed = _search_document_fact_evidence(
            role_retrieval_service,
            persona_id=persona_id,
            topic_query=topic_query,
            project_id=project_id,
            top_k=top_k,
        )
        logger.info(
            "[IDEATION_DOCUMENT_FACT_SEARCH] persona_id=%s project_id=%s result_count=%d",
            persona_id,
            project_id,
            len(composed),
        )
        return composed

    role_id = resolve_ideation_role_id(persona_id)
    candidate_k = top_k * _CANDIDATE_POOL_MULTIPLIER if persona_id in _DOCUMENT_ROLE_QUOTAS else None
    try:
        role_response = role_retrieval_service.search_by_role(
            query=topic_query,
            project_id=project_id,
            role_id=role_id,
            top_k=candidate_k or top_k,
        )
        query_b_items = [
            dict(item)
            for item in build_meeting_retrieved_evidence(
                [PersonaRoleSearchResponse(persona_id=persona_id, response=role_response, role_id=role_id)]
            )
        ]
    except Exception:
        logger.exception(
            "[IDEATION_EVIDENCE_SEARCH_FAILED] persona_id=%s role_id=%s project_id=%s",
            persona_id,
            role_id,
            project_id,
        )
        query_b_items = []

    query_a_items = _search_plain_query_candidates(
        role_retrieval_service,
        persona_id=persona_id,
        topic_query=_raw_question_component(topic_query),
        project_id=project_id,
        top_k=candidate_k or top_k,
    )

    fused_items = _reciprocal_rank_fusion(query_a_items, query_b_items)

    raw_target_count = sum(1 for item in fused_items if item.get("document_role") == "target")
    scoped_items = _scope_target_evidence(
        fused_items, session_id=session_id, selected_candidate_document_id=selected_candidate_document_id
    )
    scoped_target_count = sum(1 for item in scoped_items if item.get("document_role") == "target")

    issue_focused_criteria_items = _search_issue_focused_criteria(
        role_retrieval_service,
        persona_id=persona_id,
        topic_query=topic_query,
        project_id=project_id,
        top_k=top_k,
    )

    candidate_direct_items: list[dict] = []
    if selected_candidate_document_id:
        candidate_direct_items = _search_candidate_target_direct(
            role_retrieval_service,
            persona_id=persona_id,
            role_id=role_id,
            topic_query=topic_query,
            project_id=project_id,
            document_id=selected_candidate_document_id,
            top_k=top_k,
        )

    priority_items = candidate_direct_items + issue_focused_criteria_items
    priority_keys = {(item.get("document_id", ""), item.get("chunk_id", "")) for item in priority_items}
    merged_items = priority_items + [
        item for item in scoped_items if (item.get("document_id", ""), item.get("chunk_id", "")) not in priority_keys
    ]

    merged_items = _rank_by_document_type(merged_items, topic_query)
    merged_items = _prioritize_final_direction(merged_items, topic_query)
    composed, missing_document_roles = _compose_by_document_role(
        merged_items,
        persona_id=persona_id,
        top_k=top_k,
        phase=phase,
    )
    final_target_count = sum(1 for item in composed if item.get("document_role") == "target")

    logger.info(
        "[IDEATION_EVIDENCE_SEARCH_DEBUG] persona_id=%s project_id=%s session_id=%s "
        "selected_candidate_document_id=%s phase=%s raw_target_count=%d scoped_target_count=%d "
        "candidate_target_direct_search_count=%d issue_focused_criteria_count=%d "
        "final_target_count=%d missing_document_roles=%s",
        persona_id,
        project_id,
        session_id,
        selected_candidate_document_id,
        phase,
        raw_target_count,
        scoped_target_count,
        len(candidate_direct_items),
        len(issue_focused_criteria_items),
        final_target_count,
        missing_document_roles,
    )
    if missing_document_roles:
        logger.warning(
            "[IDEATION_EVIDENCE_MISSING_DOCUMENT_ROLES] persona_id=%s project_id=%s missing_document_roles=%s "
            "candidate_count=%d — 해당 role 문서가 없어 다른 관련 후보로만 보충했습니다.",
            persona_id,
            project_id,
            missing_document_roles,
            len(merged_items),
        )
    return composed


def make_ideation_evidence_lookup(
    project_id: str,
    role_retrieval_service: RoleAwareRetrievalService,
    top_k: int = 5,
    *,
    session_id: Optional[str] = None,
    selected_candidate_document_id: Optional[str] = None,
) -> EvidenceLookup:
    """ai/meeting/graph/ideation_nodes.py::make_ideation_expert_node(evidence_lookup=...)에
    그대로 넘길 수 있는 Callable(persona_id, topic_query) -> list[dict]를 만든다.

    session_id/selected_candidate_document_id는 이 lookup이 만들어지는 시점(매 /reply 호출마다
    backend가 새로 만든다 — evidence_lookup은 그래프 state에 직렬화될 수 없는 콜러블이라 매
    요청마다 다시 조립해야 하는 기존 정책, ideation_conversation_preview.py 참고)의 기본값
    (closure_snapshot)이다.

    용준/Claude(2026-07-23, 요청: stale closure 수정) — 후보 선택과 첫 전문가 검색이 같은
    /reply 안에서 이어지면, 위 closure 값은 이 lookup이 만들어질 때(요청 시작 시점, 아직
    후보가 선택되기 전)의 값으로 고정된 채 남는다 — 그 사이 그래프가 candidate_selection
    노드를 실행해 state["selected_idea_document_id"]를 갱신해도 이 closure는 갱신되지 않는다
    (실측 확인된 버그: target upsert는 성공하지만 같은 요청의 다음 검색이 여전히
    selected_candidate_document_id=None으로 진행되어 _scope_target_evidence가 방금 색인한
    target을 제거함). 그래서 이 lookup은 매 호출마다 선택적 키워드 인자 runtime_scope를
    받는다 — 값이 있으면(ai/meeting/graph 노드가 evidence_lookup을 호출하는 바로 그 순간의
    최신 graph state에서 읽은 값) closure 스냅샷을 덮어쓴다. runtime_scope가 없으면(배치형
    ideation_nodes.py처럼 후보 개념이 없는 호출자) 기존과 동일하게 closure 값만 쓴다 —
    완전히 하위 호환이다."""

    def lookup(persona_id: str, topic_query: str, *, runtime_scope: Optional[dict] = None) -> list[dict]:
        effective_session_id = session_id
        effective_selected_candidate_document_id = selected_candidate_document_id
        effective_phase = None
        scope_source = "closure_snapshot"
        if runtime_scope:
            if "session_id" in runtime_scope:
                effective_session_id = runtime_scope["session_id"]
            if "selected_candidate_document_id" in runtime_scope:
                effective_selected_candidate_document_id = runtime_scope["selected_candidate_document_id"]
            if "phase" in runtime_scope:
                effective_phase = runtime_scope["phase"]
            scope_source = "runtime_graph_state"
        logger.info(
            "[IDEATION_EVIDENCE_LOOKUP_SCOPE] persona_id=%s session_id=%s "
            "selected_candidate_document_id=%s phase=%s selected_candidate_document_id_source=%s",
            persona_id,
            effective_session_id,
            effective_selected_candidate_document_id,
            effective_phase,
            scope_source,
        )
        return search_ideation_evidence(
            persona_id,
            topic_query,
            project_id,
            role_retrieval_service,
            top_k=top_k,
            session_id=effective_session_id,
            selected_candidate_document_id=effective_selected_candidate_document_id,
            phase=effective_phase,
        )

    return lookup


__all__ = [
    "EvidenceLookup",
    "resolve_ideation_role_id",
    "search_ideation_evidence",
    "make_ideation_evidence_lookup",
    "classify_external_source_type",
    "compose_ideation_evidence_pool",
]
