"""
Claim-Level Evidence Grounding (Ideation Conversation)
===========================================================
아이디어 회의(ideation-conversation)에서 전문가 발언의 각 "주장(claim)"이 실제로 검색된
근거 청크와 연결되는지 검증한다. 검색된 청크를 프롬프트에 삽입했다는 사실(injected)만으로
"근거를 활용했다"고 판단하지 않고, 주장 단위로 다음을 검사한다:

1. evidence_refs가 실제로 이번 턴에 검색·주입된 청크(retrieved_evidence)에 존재하는가
2. claim_type이 document_fact인데 evidence_refs가 비어 있지 않은가
3. claim 텍스트와 인용된 청크의 내용/섹션/문서명이 실제로 관련 있는가
   (ai.rag.evidence_linking.relevance의 규칙 기반 키워드 겹침 로직을 그대로 재사용 —
   LLM을 다시 호출하지 않는다)

검증을 통과한 근거만 linked_evidence_refs에 남는다. retrieved_evidence 전체를 그대로
linked_evidence로 취급하지 않는다(RAG-004 EvidenceLinkingService와 동일한 원칙이지만,
그 서비스는 "평가 의견 1건" 단위라 committee/batch 흐름 전용이다 — 이 모듈은 "주장 여러 개를
가진 발언 1건" 단위로 같은 관련성 판정 로직을 재사용한다).
"""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from ai.rag.evidence_linking.config import EvidenceLinkingConfig
from ai.rag.evidence_linking.relevance import extract_keywords, is_relevant_candidate

ClaimType = Literal["document_fact", "expert_judgment", "user_provided_fact"]
# "expert_judgment_only"(용준/Claude(2026-07-22, 요청: partially_grounded 오판정 수정) 추가) —
# linked_evidence_count==0인데(실제 청크와 연결된 주장이 하나도 없는데) 문서 근거가 있는 것처럼
# 오인될 수 있는 "partially_grounded"/"grounded"로 판정되던 버그를 막기 위한 전용 상태다.
# 허용된 주장이 전부 expert_judgment/user_provided_fact이고(document_fact가 없거나 전부
# unsupported가 아니라 애초에 없고) 실제 연결된 근거가 0건일 때만 이 상태가 된다.
EvidenceStatus = Literal[
    "grounded", "partially_grounded", "expert_judgment_only", "ungrounded", "no_evidence_available"
]

_VALID_CLAIM_TYPES: frozenset[str] = frozenset({"document_fact", "expert_judgment", "user_provided_fact"})
_NUMBER_RE = re.compile(r"\d+(?:[,.]\d+)*(?:\s*(?:%|퍼센트|원|만원|억원|명|개|건|회|년|개월|일|시간|분|초))?")
_CRITERIA_SCOPE_TERMS = (
    "평가", "심사", "기준", "항목", "배점", "가점", "자격", "요건", "공모", "선정",
)
_ASSERTED_CAPABILITY_TERMS = (
    "해결", "개선", "감소", "증가", "향상", "예측", "탐지", "분석", "자동", "제공",
    "확보", "보장", "가능", "효과", "절감", "최적화", "구현", "운영",
)
_ALIGNMENT_GENERIC_TERMS = frozenset(
    {"측면", "검토", "필요", "중요", "사항", "부분", "관점", "대상", "현재"}
)


def _normalized_numbers(text: str) -> set[str]:
    return {re.sub(r"[\s,]", "", match) for match in _NUMBER_RE.findall(text or "")}


def _keyword_stems(text: str) -> set[str]:
    generic_stems = {term if len(term) <= 2 else term[:2] for term in _ALIGNMENT_GENERIC_TERMS}
    return {
        token if len(token) <= 2 else token[:2]
        for token in extract_keywords(text)
        if (token if len(token) <= 2 else token[:2]) not in generic_stems
    }


def _claim_evidence_alignment_failure(claim: Claim, evidence_item: dict) -> str | None:
    """단순 주제 유사도를 넘어, 인용문이 주장 범위를 직접 뒷받침하는지 검사한다.

    임베딩/키워드가 비슷해도 평가 질문을 제품 성능의 증명으로 쓰거나, 문서에 없는 수치와
    세부 기능을 덧붙이면 근거–주장 정합성이 깨진다. 이 함수는 그런 명확한 범위 확대만
    결정적으로 차단하고, 애매한 의미 추론은 사람이 검수하는 오프라인 평가에 남긴다.
    """
    claim_text = claim["text"]
    evidence_text = str(evidence_item.get("text") or evidence_item.get("quote") or "")
    document_role = evidence_item.get("document_role")
    claim_type = claim["claim_type"]

    if claim_type == "expert_judgment":
        return "expert_judgment_cannot_be_document_grounded"
    if document_role == "criteria" and claim_type == "user_provided_fact":
        return "claim_type_document_role_mismatch"
    if document_role == "target" and claim_type == "document_fact":
        return "claim_type_document_role_mismatch"

    # 공모전명·문서명에 포함된 연도(예: "WSCE 2026 Awards")는 quote 본문이 아니라
    # source/document_name에만 남을 수 있다. 이를 제품 수치 주장으로 오인하면 실제
    # 평가항목 quote가 있어도 evidence_missing_numeric_detail로 탈락한다. 수치 정합성
    # 검사에 출처 메타데이터를 함께 사용하되, 아래 핵심어 coverage에는 quote 본문만 써서
    # 파일명 하나로 사실 주장이 통과하는 일은 막는다.
    numeric_evidence_text = " ".join(
        str(value)
        for value in (
            evidence_text,
            evidence_item.get("document_name"),
            evidence_item.get("source"),
            evidence_item.get("section"),
        )
        if value
    )
    missing_numbers = _normalized_numbers(claim_text) - _normalized_numbers(numeric_evidence_text)
    if missing_numbers:
        return "evidence_missing_numeric_detail"

    # "AI 기술 활용 여부를 평가하는가?" 같은 criteria 문장은 AI가 실제로 문제를 해결하거나
    # 성능을 낸다는 사실을 증명하지 않는다. 문서의 평가/요건 자체를 설명하는 주장만 허용한다.
    if (
        document_role == "criteria"
        and claim_type == "document_fact"
        and claim_text.strip() not in evidence_text
        and any(term in claim_text for term in _ASSERTED_CAPABILITY_TERMS)
        and not any(term in claim_text for term in _CRITERIA_SCOPE_TERMS)
    ):
        return "criteria_scope_overreach"

    # 사실형 주장은 핵심어 절반 이상이 인용문 본문에 있어야 한다. 기존 relevance의
    # "한 키워드만 겹쳐도 통과" 규칙은 검색 후보에는 적절하지만 claim 증명에는 너무 느슨하다.
    if claim_type in ("document_fact", "user_provided_fact"):
        claim_stems = _keyword_stems(claim_text)
        evidence_stems = _keyword_stems(evidence_text)
        if len(claim_stems) >= 2:
            covered = len(claim_stems & evidence_stems) / len(claim_stems)
            if covered < 0.5:
                return "insufficient_claim_coverage"

    return None


class Claim(TypedDict, total=False):
    claim_id: str
    text: str
    claim_type: str
    evidence_refs: list[str]


class UnsupportedClaim(TypedDict):
    claim_id: str
    text: str
    claim_type: str
    reason: str


class ClaimEvidenceLink(TypedDict):
    """claim 1건이 실제로 연결된 근거의 ref/chunk_id 쌍(용준/Claude(2026-07-23, 요청:
    IDEATION_EVIDENCE_LINKED 로그 매핑 수정)). claim["evidence_refs"]는 LLM이 인용한 ref
    ("E1")를 그대로 담고, linked_evidence_refs(전역)는 실제 chunk_id만 담아 서로 다른
    값 공간이라 `ref in linked_evidence_refs` 비교가 항상 실패했다 — 이 필드가 claim 단위로
    "어떤 ref가 어떤 chunk_id로 연결됐는지"를 명시적으로 짝지어 로그·디버깅에 쓸 수 있게 한다."""

    claim_id: str
    evidence_refs: list[str]
    chunk_ids: list[str]


class ClaimGroundingResult(TypedDict):
    claims: list[Claim]
    linked_evidence_refs: list[str]
    unsupported_claims: list[UnsupportedClaim]
    # 용준/Claude(2026-07-23, 요청: IDEATION_EVIDENCE_LINKED 로그 매핑 수정) — claim 단위로
    # 실제 연결된 (ref, chunk_id) 쌍을 노출하는 신규 선택 필드. 관련성 검증을 통과해 실제로
    # 연결된 claim만 항목을 갖는다(연결 실패 claim은 포함되지 않는다).
    claim_evidence_links: list[ClaimEvidenceLink]
    # supported_claim_count는 기존 API 호환을 위해 의미를 바꾸지 않는다("검증을 통과해
    # 발언에 남을 수 있는 claim 수" — document_fact/실제 연결 + expert_judgment/
    # user_provided_fact의 근거 없는 허용까지 전부 포함하는 기존 합산값). 문서 근거로
    # 실제 검증됐다는 뜻으로 오해되지 않도록, 아래 신규 필드들이 그 구성 요소를 분리해
    # 노출한다(용준/Claude(2026-07-22, 요청: claim 통계 의미 분리)).
    supported_claim_count: int
    unsupported_claim_count: int
    # 형식·정책상 허용된 전체 주장 수(=supported_claim_count와 같은 값 — 이름만 명확하게
    # 신규로 노출한다. document_fact가 실제로 검증된 것과 전문가 판단이 그냥 허용된 것을
    # 합친 값이므로, 이 값 자체를 "문서로 검증됨"으로 해석하면 안 된다).
    accepted_claim_count: int
    # 실제 검색된 청크와 연결·관련성 검증까지 통과한 주장 수(문서 근거로 진짜 뒷받침된
    # claim만 센다 — claim_type과 무관하게 claim_linked가 비어있지 않은 경우).
    grounded_claim_count: int
    # 문서 근거 없이(claim_type="expert_judgment", evidence_refs 없음) 허용된 주장 수.
    expert_judgment_count: int
    # 검증을 통과한 고유 청크 수(len(linked_evidence_refs)) — evidence_status 판정과
    # 로그에서 반복적으로 "linked_evidence_count"라는 이름으로 쓰이므로 별도 필드로 노출한다.
    linked_evidence_count: int
    missing_information: list[str]
    evidence_status: EvidenceStatus
    prompt_guard: str
    allow_definitive_judgment: bool


_GUARD_BY_STATUS: dict[EvidenceStatus, str] = {
    "grounded": (
        "검증된 근거 범위 안에서만 확정적으로 말하세요. "
        "근거에 없는 사실이나 수치를 임의로 추가하지 마세요."
    ),
    "partially_grounded": (
        "일부 주장만 근거로 검증되었습니다. 검증된 주장만 문서 사실처럼 말하고, "
        "검증되지 않은 주장은 전문가 판단이나 추가 확인이 필요한 사항으로 표현하세요. "
        "확정적인 합격/탈락 판단이나 단정적 수치를 만들지 마세요."
    ),
    "ungrounded": (
        "이번 발언의 핵심 주장이 검색 근거로 뒷받침되지 않았습니다. "
        "문서에서 확인된 사실처럼 말하지 말고, 확인이 필요한 정보로 표현하거나 "
        "사용자에게 되물으세요. 확정적 판단·수치를 만들지 마세요."
    ),
    "no_evidence_available": (
        "이번 턴에는 검토할 문서 근거 자체가 없습니다. 문서 사실을 단정하지 말고 "
        "전문가 판단으로만 말하거나 추가로 필요한 정보를 명시하세요."
    ),
    "expert_judgment_only": (
        "이번 발언은 검색된 문서와 연결된 사실이 하나도 없고 전문가 판단(또는 사용자가 "
        "이미 밝힌 내용)만 있습니다. 문서로 확인된 것처럼 말하지 말고, 전문가 판단임을 "
        "분명히 밝히거나 사용자에게 필요한 정보를 구체적으로 되물으세요. 확정적 판단·수치를 "
        "만들지 마세요."
    ),
}


def _normalize_claim_type(raw_type: Any) -> str:
    if raw_type in _VALID_CLAIM_TYPES:
        return raw_type
    return "expert_judgment"


def _normalize_claims(raw_claims: Any) -> list[Claim]:
    if not isinstance(raw_claims, list):
        return []
    normalized: list[Claim] = []
    for idx, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        refs = item.get("evidence_refs")
        ref_list = [r for r in refs if isinstance(r, str) and r.strip()] if isinstance(refs, list) else []
        claim_id = item.get("claim_id")
        normalized.append(
            Claim(
                claim_id=claim_id if isinstance(claim_id, str) and claim_id.strip() else f"claim_{idx + 1}",
                text=text.strip(),
                claim_type=_normalize_claim_type(item.get("claim_type")),
                evidence_refs=ref_list,
            )
        )
    return normalized


def _evidence_chunk_id(item: dict) -> str | None:
    chunk_id = item.get("chunk_id")
    return chunk_id if isinstance(chunk_id, str) and chunk_id else None


def _evidence_lookup_key(item: dict) -> str | None:
    """LLM이 evidence_refs에 실제로 인용해야 하는 키를 찾는다. "ref"(용준/Claude(2026-07-23,
    요청: RAG 근거 실제 활용 강화)가 call_evidence_lookup에서 부여하는 짧은 순번 참조,
    예: "E1")가 있으면 그것을 우선한다 — chunk_id(해시 20자 안팎)를 LLM이 그대로 베껴 써야
    하는 부담을 없애기 위함이다. ref가 없는 항목(구버전 fixture, "ref" 없이 chunk_id만 있는
    테스트)은 기존과 동일하게 chunk_id로 조회한다 — 완전히 하위 호환이다."""
    ref = item.get("ref")
    if isinstance(ref, str) and ref:
        return ref
    return _evidence_chunk_id(item)


def ground_claims(
    raw_claims: Any,
    retrieved_evidence: list[dict],
    *,
    role_keywords: list[str] | None = None,
    config: EvidenceLinkingConfig | None = None,
) -> ClaimGroundingResult:
    """LLM이 반환한 claims를 이번 턴에 실제로 검색된 retrieved_evidence와 대조해 검증한다.

    retrieved_evidence는 evidence_lookup()이 반환한 MeetingRetrievedEvidence dict 리스트
    (chunk_id/document_id/document_name/section/page/text/semantic_score/role_score/
    final_score)다 — LLM 프롬프트에 그대로 주입된 것과 동일한 값을 넘겨야 "존재하지 않는
    chunk_id" 판정이 정확하다.

    claims[].evidence_refs는 각 항목의 "ref"(ai/meeting/graph/ideation_nodes.py::
    call_evidence_lookup이 부여하는 짧은 순번 참조, 예: "E1" — 용준/Claude(2026-07-23, 요청:
    RAG 근거 실제 활용 강화)로 조회를 시도하고, ref가 없는 항목(구버전 fixture)은 chunk_id로
    대신 조회한다(_evidence_lookup_key). 어느 쪽으로 조회했든 검증을 통과한 근거는 항상 실제
    chunk_id로 정규화되어 linked_evidence_refs에 담긴다.
    """
    cfg = config or EvidenceLinkingConfig()
    claims = _normalize_claims(raw_claims)
    evidence_by_id = {
        key: item
        for item in retrieved_evidence
        if isinstance(item, dict) and (key := _evidence_lookup_key(item))
    }

    linked_evidence_refs: list[str] = []
    claim_evidence_links: list[ClaimEvidenceLink] = []
    unsupported_claims: list[UnsupportedClaim] = []
    missing_information: list[str] = []
    supported_count = 0
    grounded_count = 0
    expert_judgment_count = 0

    for claim in claims:
        claim_type = claim["claim_type"]
        refs = claim.get("evidence_refs") or []

        if claim_type == "expert_judgment" and not refs:
            supported_count += 1
            expert_judgment_count += 1
            continue

        if not refs:
            if claim_type == "document_fact":
                unsupported_claims.append(
                    UnsupportedClaim(
                        claim_id=claim["claim_id"],
                        text=claim["text"],
                        claim_type=claim_type,
                        reason="document_fact_missing_evidence",
                    )
                )
                missing_information.append(claim["text"])
            else:
                # user_provided_fact without a reference is assumed to come from the
                # user's own input/document, not from RAG search — not held to the
                # same chunk-citation standard.
                supported_count += 1
            continue

        # claim_linked_refs/claim_linked_chunk_ids는 인덱스가 서로 대응하는 병렬 리스트다 —
        # claim_linked_refs[i]가 claim_linked_chunk_ids[i]로 연결됐다(LLM이 실제로 인용한
        # ref와 그 ref가 가리키는 실제 chunk_id의 짝, 용준/Claude(2026-07-23, 요청:
        # IDEATION_EVIDENCE_LINKED 로그 매핑 수정)).
        claim_linked_refs: list[str] = []
        claim_linked_chunk_ids: list[str] = []
        claim_reason: str | None = None
        for ref in refs:
            evidence_item = evidence_by_id.get(ref)
            if evidence_item is None:
                claim_reason = "unknown_chunk_id"
                continue
            alignment_failure = _claim_evidence_alignment_failure(claim, evidence_item)
            if alignment_failure:
                claim_reason = claim_reason or alignment_failure
                continue
            relevant = is_relevant_candidate(
                claim["text"],
                # MeetingRetrievedEvidence(운영)는 "text", 일부 옛 fixture/호출부는 "quote"를
                # 쓴다 — 둘 다 청크 본문을 뜻하므로 둘 다 받는다.
                evidence_item.get("text") or evidence_item.get("quote") or "",
                section_title=evidence_item.get("section"),
                document_title=evidence_item.get("document_name"),
                role_keywords=role_keywords,
                config=cfg,
            )
            if relevant:
                # linked_evidence_refs는 항상 실제 chunk_id를 담는다(frontend가
                # message.evidence[].chunk_id와 대조하는 기존 계약, IdeationConversationScreen.
                # jsx 참고) — LLM이 인용한 값이 짧은 ref("E1")든 구버전 chunk_id든, 여기서는
                # evidence_item의 진짜 chunk_id로 정규화해 담는다.
                chunk_id = _evidence_chunk_id(evidence_item) or ref
                if chunk_id not in claim_linked_chunk_ids:
                    claim_linked_refs.append(ref)
                    claim_linked_chunk_ids.append(chunk_id)
            else:
                claim_reason = claim_reason or "evidence_not_relevant"

        if claim_linked_chunk_ids:
            supported_count += 1
            grounded_count += 1
            for chunk_id in claim_linked_chunk_ids:
                if chunk_id not in linked_evidence_refs:
                    linked_evidence_refs.append(chunk_id)
            claim_evidence_links.append(
                ClaimEvidenceLink(
                    claim_id=claim["claim_id"],
                    evidence_refs=claim_linked_refs,
                    chunk_ids=claim_linked_chunk_ids,
                )
            )
        else:
            unsupported_claims.append(
                UnsupportedClaim(
                    claim_id=claim["claim_id"],
                    text=claim["text"],
                    claim_type=claim_type,
                    reason=claim_reason or "unknown_chunk_id",
                )
            )
            if claim_type == "document_fact":
                missing_information.append(claim["text"])

    supported_claim_count = supported_count
    unsupported_claim_count = len(unsupported_claims)
    accepted_claim_count = supported_claim_count
    grounded_claim_count = grounded_count
    linked_evidence_count = len(linked_evidence_refs)

    # 용준/Claude(2026-07-22, 요청: linked_evidence_count=0인데 partially_grounded/grounded가
    # 나오면 안 됨) — 판정은 항상 linked_evidence_count(실제 청크와 연결된 주장이 있는지)를
    # 최우선 기준으로 삼는다. "허용된 주장이 있다(supported_claim_count>0)"는 사실만으로는
    # 절대 grounded/partially_grounded로 승격시키지 않는다.
    if not claims:
        evidence_status: EvidenceStatus = "no_evidence_available"
    elif not retrieved_evidence:
        # 이번 턴에 검색된 근거 자체가 없었다 — claims가 존재해도(예: user_provided_fact,
        # expert_judgment) 판정할 문서 근거 풀이 없었다는 뜻이라 no_evidence_available이다.
        evidence_status = "no_evidence_available"
    elif linked_evidence_count > 0 and unsupported_claim_count == 0 and grounded_claim_count == accepted_claim_count:
        # 허용된 주장 전부가 실제로 문서 근거에 연결됐을 때만 grounded다 — 문서 근거 없는
        # expert_judgment/user_provided_fact가 하나라도 섞여 있으면(grounded_claim_count <
        # accepted_claim_count) partially_grounded로 내려간다.
        evidence_status = "grounded"
    elif linked_evidence_count > 0:
        # 실제로 연결된 근거가 있지만(linked_evidence_count>0) 일부 주장은 검증에 실패했거나
        # (unsupported_claim_count>0) 문서 근거 없는 전문가 판단이 섞여 있다.
        evidence_status = "partially_grounded"
    elif unsupported_claim_count > 0:
        # document_fact가 있는데 근거가 없거나 검증에 실패했다(존재하지 않는/무관한 chunk_id 등).
        evidence_status = "ungrounded"
    elif accepted_claim_count > 0:
        # linked_evidence_count==0이고 unsupported도 없다 — 허용된 주장이 전부
        # expert_judgment/user_provided_fact뿐이라는 뜻이다. 문서로 검증됐다고 표시하면
        # 안 되므로 partially_grounded/grounded로 승격시키지 않고 전용 상태로 분리한다.
        evidence_status = "expert_judgment_only"
    else:
        evidence_status = "ungrounded"

    return ClaimGroundingResult(
        claims=claims,
        linked_evidence_refs=linked_evidence_refs,
        claim_evidence_links=claim_evidence_links,
        unsupported_claims=unsupported_claims,
        supported_claim_count=supported_claim_count,
        unsupported_claim_count=unsupported_claim_count,
        accepted_claim_count=accepted_claim_count,
        grounded_claim_count=grounded_claim_count,
        expert_judgment_count=expert_judgment_count,
        linked_evidence_count=linked_evidence_count,
        missing_information=missing_information,
        evidence_status=evidence_status,
        prompt_guard=_GUARD_BY_STATUS[evidence_status],
        allow_definitive_judgment=evidence_status == "grounded",
    )


def has_hard_grounding_failure(result: ClaimGroundingResult) -> bool:
    """재생성을 유발할 만한 실패인지 판단한다 — document_fact 주장이 하나라도 있는데 전부
    unsupported면(존재하지 않는 chunk_id 인용, 근거 없음, 무관한 청크 인용 포함) 재생성
    대상이다. expert_judgment만 있거나 전부 검증을 통과했으면 재생성하지 않는다."""
    document_fact_claims = [c for c in result["claims"] if c["claim_type"] == "document_fact"]
    if not document_fact_claims:
        return False
    unsupported_ids = {c["claim_id"] for c in result["unsupported_claims"]}
    return all(c["claim_id"] in unsupported_ids for c in document_fact_claims)


__all__ = [
    "Claim",
    "ClaimEvidenceLink",
    "UnsupportedClaim",
    "ClaimGroundingResult",
    "ClaimType",
    "EvidenceStatus",
    "ground_claims",
    "has_hard_grounding_failure",
]
