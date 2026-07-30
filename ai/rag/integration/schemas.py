"""
RAG -> 회의 파이프라인 출력 계약
====================================
RAG-003(RoleAwareRetrievalService)/RAG-004(EvidenceLinkingService)의 결과를
ai/meeting/graph가 소비하는 plain dict 형태로 옮기기 위한 타입 계약.

여기서 정의하는 TypedDict는 런타임 검증을 하지 않는다 — 실제 값은 항상
plain dict로 반환되며(EvidencePool 등 회의 쪽 코드가 `.get()`으로 읽는 관례를
따름), TypedDict는 정적 타입 힌트로만 쓰인다.
"""

from dataclasses import dataclass
from typing import Optional, TypedDict

from ai.rag.role_retrieval.schemas import RoleSearchResponse

# 용준/Claude(2026-07-23, Phase 1 "Shadow Deterministic Evidence Planner"): 회의 발언을
# 생성하기 전에 규칙 기반으로 evidence를 미리 선택·검증하는 shadow planner(ai/rag/
# orchestration/ideation_evidence_planner.py)의 출력 계약. Phase 1에서는 이 결과가 prompt/
# claims/grounding/routing에 전혀 쓰이지 않고 trace 로그로만 기록된다 — 다른 TypedDict와
# 마찬가지로 런타임 검증은 하지 않으며, 실제 값은 plain dict로 오간다(ai/meeting/graph는 이
# 모듈을 import하지 않는다 — backend가 콜러블만 주입한다).


class EvidencePlanIssue(TypedDict):
    """planner가 참조한 "지금 검토 중인 쟁점" — ai/meeting/graph/ideation_conv_nodes.py::
    resolve_effective_issue()가 만드는 것과 정확히 같은 issue_id/title, 그리고 그 턴의 실제
    retrieval query를 담는다(요청: retrieval에 실제 사용된 issue와 planner issue가 반드시
    동일해야 한다)."""

    issue_id: str
    title: str
    query: str


class PlannedEvidence(TypedDict):
    """selected_evidence 1건 — retrieved_evidence 중 실제로 선택된 항목의 근거·인용 정보."""

    ref: str
    chunk_id: str
    document_id: str
    document_role: str
    claim_type: str
    quote: str
    quote_start: int
    quote_end: int
    retrieval_score: Optional[float]
    issue_relevance_score: float
    selection_reason_code: str
    reused_in_same_issue: bool


class EvidencePlanValidation(TypedDict):
    valid: bool
    errors: list[str]


class EvidencePlan(TypedDict):
    plan_id: str
    policy_version: str
    persona_id: str
    issue: EvidencePlanIssue
    eligible_evidence_count: int
    grounded_claim_required: bool
    expert_judgment_required: bool
    selected_evidence: list[PlannedEvidence]
    empty_plan_reason: Optional[str]
    validation: EvidencePlanValidation


class MeetingRetrievedEvidence(TypedDict):
    """run_meeting(retrieved_evidence=...)에 그대로 넘길 수 있는 근거 1건.

    chunk_id/document_name/page/section/text/score는 기존 EvidencePool이 읽는
    필드라 이름과 의미를 그대로 유지한다(ai/meeting/graph/evidence.py 참고)."""

    chunk_id: str
    document_id: str
    persona_id: str
    role_id: Optional[str]

    document_name: Optional[str]
    section: Optional[str]
    page: Optional[int]

    location_number: Optional[int]
    location_type: Optional[str]

    text: str

    semantic_score: Optional[float]
    role_score: Optional[float]
    final_score: Optional[float]
    score: float

    # 용준/Claude(2026-07-22, 요청: 역할별 검색 데이터 구성) — 색인 시점의 IndexingContext.
    # document_role("criteria"=공고문·평가기준, "target"=평가 대상 문서/기획서)을 그대로
    # 노출한다. 색인 시 document_role을 지정하지 않은 청크는 None이다(하위 호환 — 기존
    # 색인 데이터에는 이 값이 없을 수 있다). ai/rag/orchestration/ideation_evidence_service.py가
    # 이 값으로 역할별 top_k 구성(criteria/target 쿼터)을 계산한다.
    document_role: Optional[str]
    document_type: Optional[str]

    # 용준/Claude(2026-07-22, 요청: 선택된 아이디어/사용자 답변을 target evidence로 색인) —
    # ai/rag/orchestration/ideation_target_indexing_service.py가 IndexingContext.extra_metadata로
    # 실어 보낸 값이 청크 metadata에 그대로 남아 있으면 그대로 노출한다. 일반 문서 업로드로
    # 만들어진 청크는 이 값이 없으므로(색인 시 extra_metadata를 쓰지 않음) 둘 다 None이다 —
    # ideation_evidence_service.py의 세션/후보 범위 필터링(_scope_target_evidence)이
    # ideation_source_type이 None이면 "일반 프로젝트 문서"로 간주해 항상 통과시킨다.
    ideation_source_type: Optional[str]
    session_id: Optional[str]

    # 용준/Claude(2026-07-30, 요청: criteria 근거의 출처 구분 — 공고 URL 본문/URL 첨부파일/
    # 직접 업로드) — Chunk.source_type(ai/rag/chunking/schemas.py::SourceType:
    # file_upload/url_attachment/url_webpage/ideation_generated)을 그대로 노출한다. 이 필드는
    # ideation_evidence_service.py::compose_ideation_evidence_pool이 별도로 매기는
    # "source_type"(target/criteria/official_statistics 등 화면 4버킷 분류용)과 이름이
    # 겹치므로 혼동을 막기 위해 origin_type으로 새로 노출한다 — 기존 document_role/
    # source_type 필드는 그대로 두고 순수 추가만 한다. 일반 파일 업로드로 만들어진 청크는
    # source_filename이, URL 수집 청크는 source_url이 채워진다(chunker.py가 이미 채우는 값,
    # 여기서는 노출만 한다).
    origin_type: Optional[str]
    file_name: Optional[str]
    source_url: Optional[str]


class MeetingLinkedEvidenceRef(TypedDict):
    """RAG-004 LinkedEvaluation.evidence[] 1건을 (document_id, chunk_id) 키가
    보존된 plain dict로 옮긴 것. evidence_id는 만들지 않는다 — 회의 쪽이
    (document_id, chunk_id)로 자신이 이미 발급한 evidence_id를 역조회한다."""

    document_id: str
    chunk_id: str
    quote: str

    document_name: Optional[str]
    section: Optional[str]
    page: Optional[int]

    semantic_score: Optional[float]
    role_score: Optional[float]
    final_score: Optional[float]


@dataclass(frozen=True)
class PersonaRoleSearchResponse:
    """persona_id/role_id 정보가 없는 RoleSearchResponse 하나를 build_meeting_retrieved_evidence()에
    넘기기 위한 입력 래퍼. RoleSearchResponse.role_id는 검색에 실제 사용된 role_id를 담지만,
    role_id=None(semantic-only fallback)으로 검색한 persona도 있을 수 있어 persona_id는 항상
    호출자가 명시적으로 넘긴다."""

    persona_id: str
    response: RoleSearchResponse
    role_id: Optional[str] = None
