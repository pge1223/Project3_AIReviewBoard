# 작성자: 용준/Claude(2026-07-22)
# 목적: RAG 품질 평가 데이터셋(JSONL 입력)과 실행 결과(리포트) Pydantic 스키마.
#       기존 ai/rag/evaluation/schemas.py(EvaluationCase, 레거시 배치 위원회 domain/
#       persona_id 체계)와는 별개다 — ideation persona_id("planning_expert"/"dev_expert")는
#       그 체계의 role_mapping 화이트리스트에 없으므로 여기서 새 스키마를 쓴다.
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

PersonaId = Literal["planning_expert", "dev_expert"]
ClaimVerdictLiteral = Literal["supported", "partially_supported", "unsupported", "contradicted", "non_factual"]


class RagEvalFilters(BaseModel):
    """실제 검색 경로(RoleAwareRetrievalService.search_by_role)에 존재하는 필터만 담는다 —
    요청 원본 스키마의 category/source_org는 이 검색 경로에 존재하지 않아(계획 문서 1번
    분석) project_id/role_id로 대체했다. role_id를 명시하지 않으면
    ai.rag.orchestration.ideation_evidence_service.resolve_ideation_role_id(persona_id)가
    정하는 기본값을 그대로 쓴다(운영 코드와 동일한 매핑)."""

    project_id: str
    role_id: Optional[str] = None
    phase: Optional[str] = None

    @field_validator("project_id")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("filters.project_id는 빈 문자열일 수 없습니다")
        return v


class RagEvalSessionState(BaseModel):
    """용준/Claude(2026-07-29, 요청: 평가 정합성 — 정답 누출 분리). 실제 세션이라면
    ai/meeting의 그래프 state(예: state["selected_idea_document_id"])에서 나왔을 정보를
    담는다. 채점용 정답 필드(gold_document_ids/expected_source_chunk_ids/reference_answer)와
    이 필드는 반드시 분리한다 — 검색 입력으로는 이 session_state만 쓰고, 채점 필드는
    검색에 절대 넘기지 않는다(RAG-004 원칙과 동일: 채점 정보와 시스템 입력을 섞지 않는다).
    이 값들이 채점 필드와 우연히 같은 문서를 가리킬 수는 있다(실제로 그 세션이 그 문서를
    확정했으니까) — 하지만 "정답을 알아서 넣은 것"이 아니라 "그 세션의 실제 상태를
    독립적으로 기록한 것"이라는 점에서 근본적으로 다르다."""

    selected_idea_document_id: Optional[str] = None
    selected_idea_title: Optional[str] = None
    idea_locked: bool = False
    current_phase: Optional[str] = None


class RagEvalCase(BaseModel):
    """평가 케이스 1건(JSONL 한 줄). human_verified=False가 기본값이다 — Claude가 만든
    gold_document_ids 초안은 사람이 확인하기 전까지 정식 점수에 들어가지 않는다(요청 2번)."""

    id: str
    query: str
    persona_id: PersonaId
    filters: RagEvalFilters
    gold_document_ids: list[str] = Field(default_factory=list)
    expected_evidence_topics: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    expect_no_evidence: bool = False
    human_verified: bool = False
    notes: Optional[str] = None
    reference_answer: Optional[str] = None
    """사람이 작성한 정답 발언(선택). 별도 Ragas 평가 환경(ai/rag/evaluation/ragas_standalone/)의
    Context Precision/Context Recall처럼 reference가 있어야 계산되는 지표에만 쓰인다. 없으면
    해당 지표는 "측정 불가"로 표시된다 — 값을 억지로 만들어 채우지 않는다. 채점 전용 —
    검색 입력으로 쓰지 않는다."""
    expected_source_chunk_ids: list[str] = Field(default_factory=list)
    """reference_answer를 실제로 뒷받침하는 원문 chunk_id(들)(선택). gold_document_ids보다
    더 정밀한 근거 추적용 — reference_answer가 어느 chunk에서 나왔는지 사람이 다시 검증할 때
    쓴다. 채점 전용 — 검색 입력으로 쓰지 않는다."""
    session_state: Optional[RagEvalSessionState] = None
    """검색 입력으로 쓰는 유일한 상태 정보(선택) — 실제 그래프 state를 흉내낸다. 채점 필드가
    아니다. generation_eval.py/export_ragas_dataset.py는 evidence_lookup에 이 값만 전달한다."""

    @field_validator("id", "query")
    @classmethod
    def _not_blank(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name}은(는) 빈 문자열일 수 없습니다")
        return v

    @model_validator(mode="after")
    def _gold_docs_consistent_with_expect_no_evidence(self) -> "RagEvalCase":
        if self.expect_no_evidence and self.gold_document_ids:
            raise ValueError(
                f"case {self.id!r}: expect_no_evidence=true인데 gold_document_ids가 비어있지 않습니다"
            )
        if not self.expect_no_evidence and not self.gold_document_ids:
            raise ValueError(
                f"case {self.id!r}: expect_no_evidence=false이면 gold_document_ids가 최소 1개 필요합니다"
            )
        deduped = list(dict.fromkeys(self.gold_document_ids))
        if len(deduped) != len(self.gold_document_ids):
            raise ValueError(f"case {self.id!r}: gold_document_ids에 중복이 있습니다")
        return self


class RagEvalDataset(BaseModel):
    dataset_name: str
    version: str
    cases: list[RagEvalCase]

    @model_validator(mode="after")
    def _unique_ids(self) -> "RagEvalDataset":
        if not self.cases:
            raise ValueError("cases는 최소 1개 이상이어야 합니다")
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"중복된 case id입니다: {case.id!r}")
            seen.add(case.id)
        return self


# --------------------------------------------------------------------------
# Retrieval 결과
# --------------------------------------------------------------------------


class RetrievedDocumentHit(BaseModel):
    document_id: str
    chunk_id: str
    rank: int
    score: float
    document_name: Optional[str] = None


class RetrievalCaseResult(BaseModel):
    case_id: str
    query: str
    persona_id: PersonaId
    project_id: str
    role_id: Optional[str] = None
    gold_document_ids: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedDocumentHit] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list, description="중복 제거된, 점수순 document_id")
    recall_at_k: float = 0.0
    hit_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    expect_no_evidence: bool = False
    empty_result: bool = False
    human_verified: bool = False
    retrieval_time_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class RetrievalAggregate(BaseModel):
    k: int
    case_count: int = 0
    human_verified_case_count: int = 0
    recall_at_k_macro: Optional[float] = None
    hit_at_k_macro: Optional[float] = None
    mrr_macro: Optional[float] = None
    reference_recall_at_k_macro: Optional[float] = None
    reference_hit_at_k_macro: Optional[float] = None
    reference_mrr_macro: Optional[float] = None
    no_evidence_case_count: int = 0
    no_evidence_accuracy: Optional[float] = None
    retrieval_failure_rate: Optional[float] = None
    avg_retrieval_time_ms: float = 0.0


# --------------------------------------------------------------------------
# Faithfulness / Hallucination 결과
# --------------------------------------------------------------------------


class ClaimVerdict(BaseModel):
    claim: str
    verdict: ClaimVerdictLiteral
    supporting_document_ids: list[str] = Field(default_factory=list)
    supporting_evidence_excerpt: Optional[str] = None
    reason: str = ""
    confidence: float = 0.0


class PersonaFitResult(BaseModel):
    persona_id: PersonaId
    message_id: str
    score: int  # 0~4
    normalized_score: float  # score/4
    evidence_document_ids: list[str] = Field(default_factory=list)
    role_aligned_points: list[str] = Field(default_factory=list)
    role_mismatch_points: list[str] = Field(default_factory=list)
    rationale: str = ""


class ExpertJudgmentResult(BaseModel):
    """용준/Claude(2026-07-29, 요청: 생성 품질 개선 — expert_judgment G-Eval). expert_judgment
    claim 1건에 대한 역할 적합성/논리성/실행 가능성/사실 충돌 여부 판정."""

    persona_id: PersonaId
    message_id: str
    claim_id: Optional[str] = None
    role_fit: int  # 0~2
    logic: int  # 0~2
    actionability: int  # 0~2
    normalized_score: float  # (role_fit+logic+actionability)/6
    conflicts_with_grounded_fact: bool = False
    conflict_detail: str = ""
    rationale: str = ""


class MessageEvalResult(BaseModel):
    """발언(ConvMessage) 1건에 대한 Faithfulness + Persona Evidence Fit 판정."""

    case_id: str
    message_id: str
    persona_id: PersonaId
    round: int
    content_preview: str
    claims: list[ClaimVerdict] = Field(default_factory=list)
    faithfulness_score: Optional[float] = None  # None = not_applicable(분모 0)
    hallucination_rate: Optional[float] = None
    unsupported_count: int = 0
    contradicted_count: int = 0
    non_factual_count: int = 0
    forbidden_claim_hits: list[str] = Field(default_factory=list)
    persona_fit: Optional[PersonaFitResult] = None
    judge_error: Optional[str] = None
    # 용준/Claude(2026-07-29, 요청: 생성 품질 개선 — 발언 유형 분리). 기존 필드는 그대로 두고
    # 평가 시점에 ai/rag/evaluation/rag_quality/utterance_type.py로 파생한 값을 선택적으로
    # 채운다 — 새 스키마가 아니라 기존 결과에 라벨 하나만 덧붙이는 것이다.
    utterance_type: Optional[str] = None  # "facilitator_message" | "final_answer" | "discussion"
    grounded_claim_count: Optional[int] = None
    expert_judgment_claim_count: Optional[int] = None
    grounding_unsupported_claim_count: Optional[int] = None
    linked_evidence_refs: list[str] = Field(default_factory=list)
    expert_judgments: list[ExpertJudgmentResult] = Field(default_factory=list)


class GenerationCaseResult(BaseModel):
    case_id: str
    query: str
    messages: list[MessageEvalResult] = Field(default_factory=list)
    generation_error: Optional[str] = None
    generation_time_ms: float = 0.0


class GenerationAggregate(BaseModel):
    case_count: int = 0
    message_count: int = 0
    faithfulness_macro: Optional[float] = None
    hallucination_rate_macro: Optional[float] = None
    persona_evidence_fit_macro: Optional[float] = None  # 0~1
    persona_evidence_fit_percent: Optional[float] = None
    generation_failure_rate: Optional[float] = None
    eval_failure_rate: Optional[float] = None
    avg_generation_time_ms: float = 0.0
    severe_hallucination_examples: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# 최종 리포트
# --------------------------------------------------------------------------


class EvalSettingsSnapshot(BaseModel):
    dataset_name: str
    dataset_version: str
    dataset_path: str
    mode: Literal["retrieval", "generation", "all"]
    top_k: int
    generation_model: Optional[str] = None
    eval_model: Optional[str] = None
    eval_prompt_version: str
    chroma_collection: Optional[str] = None
    human_verified_only: bool
    cache_enabled: bool


class EvalReport(BaseModel):
    run_id: str
    executed_at: datetime
    settings: EvalSettingsSnapshot
    retrieval_results: list[RetrievalCaseResult] = Field(default_factory=list)
    retrieval_aggregate: Optional[RetrievalAggregate] = None
    generation_results: list[GenerationCaseResult] = Field(default_factory=list)
    generation_aggregate: Optional[GenerationAggregate] = None
    estimated_cost_usd: Optional[float] = None
    notes: Optional[str] = None
