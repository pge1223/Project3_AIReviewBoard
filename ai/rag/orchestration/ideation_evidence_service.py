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
_CURRENT_ISSUE_PATTERN = re.compile(r"(?:^|\|)\s*현재 쟁점:\s*([^|]+)")


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


def _compose_by_document_role(
    items: list[dict], *, persona_id: str, top_k: int
) -> tuple[list[dict], list[str]]:
    """검색된 후보(items, final_score 내림차순 정렬 상태 유지)를 persona별
    _DOCUMENT_ROLE_QUOTAS에 맞춰 재구성한다. 원하는 role의 후보가 전혀 없으면
    missing_document_roles에 기록한다(그 role을 무관한 다른 문서로 억지로 채우지 않는다
    — 부족한 만큼만 다른 role/미분류 후보로 보충한다).

    반환값: (구성된 top_k개 이하의 리스트, missing_document_roles)."""
    quotas = _DOCUMENT_ROLE_QUOTAS.get(persona_id)
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
        taken = buckets[role][:quota]
        leftover.extend(buckets[role][quota:])
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


def search_ideation_evidence(
    persona_id: str,
    topic_query: str,
    project_id: str,
    role_retrieval_service: RoleAwareRetrievalService,
    top_k: int = 5,
    *,
    session_id: Optional[str] = None,
    selected_candidate_document_id: Optional[str] = None,
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
    criteria 결과를 대체하거나 가짜로 채우지 않는다(실제 Chroma 검색 결과만 병합)."""
    role_id = resolve_ideation_role_id(persona_id)
    candidate_k = top_k * _CANDIDATE_POOL_MULTIPLIER if persona_id in _DOCUMENT_ROLE_QUOTAS else None
    try:
        role_response = role_retrieval_service.search_by_role(
            query=topic_query,
            project_id=project_id,
            role_id=role_id,
            top_k=candidate_k or top_k,
        )
        plain_items = [
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
        plain_items = []

    raw_target_count = sum(1 for item in plain_items if item.get("document_role") == "target")
    scoped_items = _scope_target_evidence(
        plain_items, session_id=session_id, selected_candidate_document_id=selected_candidate_document_id
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

    composed, missing_document_roles = _compose_by_document_role(merged_items, persona_id=persona_id, top_k=top_k)
    final_target_count = sum(1 for item in composed if item.get("document_role") == "target")

    logger.info(
        "[IDEATION_EVIDENCE_SEARCH_DEBUG] persona_id=%s project_id=%s session_id=%s "
        "selected_candidate_document_id=%s raw_target_count=%d scoped_target_count=%d "
        "candidate_target_direct_search_count=%d issue_focused_criteria_count=%d "
        "final_target_count=%d missing_document_roles=%s",
        persona_id,
        project_id,
        session_id,
        selected_candidate_document_id,
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
        scope_source = "closure_snapshot"
        if runtime_scope:
            if "session_id" in runtime_scope:
                effective_session_id = runtime_scope["session_id"]
            if "selected_candidate_document_id" in runtime_scope:
                effective_selected_candidate_document_id = runtime_scope["selected_candidate_document_id"]
            scope_source = "runtime_graph_state"
        logger.info(
            "[IDEATION_EVIDENCE_LOOKUP_SCOPE] persona_id=%s session_id=%s "
            "selected_candidate_document_id=%s selected_candidate_document_id_source=%s",
            persona_id,
            effective_session_id,
            effective_selected_candidate_document_id,
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
        )

    return lookup


__all__ = [
    "EvidenceLookup",
    "resolve_ideation_role_id",
    "search_ideation_evidence",
    "make_ideation_evidence_lookup",
]
