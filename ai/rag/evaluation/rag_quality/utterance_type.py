# 작성자: 용준/Claude(2026-07-29, 요청: 생성 품질 개선 — 발언·주장 유형 분리)
# 목적: grounded_claim/expert_judgment/facilitator_message/final_answer 유형을 새 필드로
#       저장하지 않고, ai/meeting이 이미 만들어 내는 값(speaker_id, structured.query_type,
#       claim_type) 조합으로 "평가 시점에만" 파생한다. 그래프 상태·프롬프트 스키마는 건드리지
#       않는다(요청: "중복 스키마를 새로 만들지 말고 기존 필드와 호환되게 구현").
#
#       매핑 근거:
#       - facilitator_message: speaker_id == "ideation_facilitator" (ConvMessage.speaker_id,
#         ideation_conv_state.py). 회의 진행 안내/요약 발화는 항상 이 speaker가 만든다.
#       - final_answer: structured.query_type in {"document_fact_query", "session_state_query"}
#         (ai/meeting/graph/ideation_conv_nodes.py::classify_query_type — 2단계에서 신규 도입).
#         용준/Claude(2026-07-29, 버그 수정): 이전 버전은 message_type=="answer"로 판정했는데,
#         실측 결과 message_type="answer"는 **사용자 메시지에만** 붙고(ideation_conv_run.py::
#         _new_user_message 기본값) 위원이 생성한 발언에는 절대 붙지 않는다 — 그래서
#         final_answer 버킷이 항상 0건이었다. structured.query_type은 위원 발언에 직접 붙는
#         값이라 이 문제가 없다.
#       - grounded_claim: claim_type in {"document_fact", "user_provided_fact"} — 둘 다
#         evidence_refs로 검증 가능한 사실 주장이라는 점에서 요청한 grounded_claim과 동일하다
#         (ai/rag/evidence_linking/claim_grounding.py의 ClaimType).
#       - expert_judgment: claim_type == "expert_judgment" — 요청한 expert_judgment와 이름·
#         의미가 그대로 같다.
from __future__ import annotations

from typing import Literal

MessageUtteranceType = Literal["facilitator_message", "final_answer", "discussion"]
ClaimUtteranceType = Literal["grounded_claim", "expert_judgment"]

_FACILITATOR_SPEAKER_ID = "ideation_facilitator"
_GROUNDED_CLAIM_TYPES = ("document_fact", "user_provided_fact")
_FINAL_ANSWER_QUERY_TYPES = ("document_fact_query", "session_state_query")


def classify_message(message: dict) -> MessageUtteranceType:
    """ConvMessage(dict) 1건을 facilitator_message/final_answer/discussion 중 하나로 분류한다.
    discussion은 expert_analysis_query로 분류됐거나 분류 자체가 안 된(None) 일반 토론
    발화를 담는 잔여 버킷이다 — 요청한 4유형 밖의 발화를 억지로 final_answer/expert_judgment로
    밀어넣지 않기 위해 별도로 둔다."""
    if message.get("speaker_id") == _FACILITATOR_SPEAKER_ID:
        return "facilitator_message"
    structured = message.get("structured") or {}
    if isinstance(structured, dict) and structured.get("query_type") in _FINAL_ANSWER_QUERY_TYPES:
        return "final_answer"
    return "discussion"


def classify_claim(claim: dict) -> ClaimUtteranceType:
    """claim(dict, ConvMessage.claims의 원소) 1건을 grounded_claim/expert_judgment로 분류한다."""
    if claim.get("claim_type") in _GROUNDED_CLAIM_TYPES:
        return "grounded_claim"
    return "expert_judgment"


def bucket_claims(claims: list[dict]) -> dict[ClaimUtteranceType, list[dict]]:
    buckets: dict[ClaimUtteranceType, list[dict]] = {"grounded_claim": [], "expert_judgment": []}
    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        buckets[classify_claim(claim)].append(claim)
    return buckets
