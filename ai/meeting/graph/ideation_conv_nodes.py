# 작성자: 용준/Claude(2026-07-20)
# 목적: 대화형 아이디어 발전 회의(ideation-conversation) LangGraph 노드.
#       배치형 ideation_nodes.py와 노드 "내용"은 비슷하지만(페르소나 프롬프트 조립, LLM
#       호출, JSON 정규화), 각 노드가 "질문 하나를 던지고 즉시 멈춘다"는 점이 근본적으로
#       다르다 — 배치형은 위원 4명이 한 그래프 실행 안에서 순서대로 다 말하고 나서야
#       facilitator가 사용자 질문 여부를 판단하지만, 이 모듈의 노드들은 그 자체가 "정지
#       지점"이라서 각 노드가 끝나면 그래프는 항상 END로 간다(정지는 그래프 조립부
#       ideation_conv_build.py가 아니라 여기 각 노드가 반환하는 phase가 결정한다).
# import: prompts.build_ideation_conv_*(형제 패키지), 같은 패키지의 ideation_conv_state/llm,
#         _safe_call_json은 배치형 노드(ideation_nodes.py)의 것을 그대로 재사용한다(LLM 호출
#         실패 시 재시도 1회 후 폴백하는 정책을 새로 만들지 않고 통일하기 위함).

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from prompts import (
    build_ideation_conv_canvas_update_prompt,
    build_ideation_conv_discussion_facilitator_prompt,
    build_ideation_conv_discussion_prompt,
    build_ideation_conv_expert_delegation_facilitator_prompt,
    build_ideation_conv_expert_delegation_prompt,
    build_ideation_conv_expert_delegation_review_prompt,
    build_ideation_conv_question_prompt,
    build_ideation_conv_sufficiency_prompt,
    build_ideation_conv_synthesis_prompt,
    get_persona_card,
)

from .ideation_conv_state import (
    TOPIC_PRIORITY,
    ConvMessage,
    DiscussionRoundRecord,
    IdeationCancelled,
    IdeationConvState,
    remaining_topics_for,
)
from .ideation_nodes import EvidenceLookup, _safe_call_json, call_evidence_lookup
from .ideation_trace import sanitize_preview, trace_event
from .llm import LLMCall, parse_json_response

logger = logging.getLogger(__name__)

# 용준/Claude(2026-07-22, 요청: RAG 근거 실제 활용 강화) — EvidenceLookup(ai/meeting/graph/
# ideation_nodes.py)과 같은 이유로, claim-evidence 연결 검증도 ai/meeting이 ai.rag를 직접
# import하지 않는다(ai/rag/tests/test_meeting_evidence_service.py::TestScopeBoundary가 이
# 경계를 강제한다 — ai/meeting/graph는 ai.rag를 몰라야 하고, 실제 검색/근거 판정 구현은
# 항상 호출부(backend)가 주입한다). ClaimGroundingFn은 claims(LLM이 반환한 리스트)와
# retrieved_evidence(이번 턴 검색 결과)를 받아 ai.rag.evidence_linking.claim_grounding.
# ground_claims()와 같은 모양의 dict(claims/linked_evidence_refs/unsupported_claims/
# supported_claim_count/unsupported_claim_count/missing_information/evidence_status/
# prompt_guard/allow_definitive_judgment)를 반환해야 한다 — 이 파일은 그 결과의 "모양"만
# 알고 실제 관련성 판정 로직은 모른다. persona_id를 함께 넘겨 역할별 관련성 키워드(기획/개발)
# 적용은 주입된 함수(호출부)가 알아서 하게 한다.
ClaimGroundingFn = Callable[[str, Any, list[dict]], dict]

# 용준/Claude(2026-07-23, Phase 1 "Shadow Deterministic Evidence Planner") — evidence_lookup/
# ground_claims와 같은 경계 원칙: 실제 규칙 기반 evidence 선택 구현(ai/rag/orchestration/
# ideation_evidence_planner.py::build_evidence_plan)은 backend가 주입하고, 이 파일은 그
# 결과가 plain dict(EvidencePlan 모양)라는 것만 안다. persona_id/effective_issue(issue_id/
# title/query)/retrieved_evidence(이번 턴 검색 결과)/runtime_scope(session_id/
# selected_candidate_document_id)/shadow_history(같은 speaker/issue의 이전 shadow 선택
# 이력)를 keyword-only로 받는다 — 키워드 전용으로 둔 이유는 이 콜러블이 신규 도입이라
# 위치 인자 순서에 의존할 legacy 호출자가 없기 때문이다. Phase 1에서는 이 결과가 prompt/
# claims/grounding/routing 어디에도 쓰이지 않고 trace 로그로만 기록된다.
EvidencePlanningFn = Callable[..., dict]

# 같은 speaker/issue 조합("persona_id:issue_id")별로 보관하는 shadow 선택 이력 최대 개수 —
# 무제한 누적을 막는다(요청: 최소 정보만 유지).
_SHADOW_HISTORY_KEEP = 20

_EMPTY_GROUNDING: dict = {
    "claims": [],
    "linked_evidence_refs": [],
    # 용준/Claude(2026-07-23, 요청: IDEATION_EVIDENCE_LINKED 로그 매핑 수정) — ai.rag.
    # evidence_linking.claim_grounding.ClaimGroundingResult와 모양을 맞춘다.
    "claim_evidence_links": [],
    "unsupported_claims": [],
    "supported_claim_count": 0,
    "unsupported_claim_count": 0,
    "accepted_claim_count": 0,
    "grounded_claim_count": 0,
    "expert_judgment_count": 0,
    "linked_evidence_count": 0,
    "missing_information": [],
    "evidence_status": "no_evidence_available",
    "prompt_guard": "",
    "allow_definitive_judgment": False,
}


def _has_hard_grounding_failure(grounding: dict) -> bool:
    """document_fact 주장이 있는데 전부 unsupported면 재생성 대상이다 — ai.rag.
    evidence_linking.claim_grounding.has_hard_grounding_failure와 동일한 판정을 dict
    형태(주입된 함수의 반환값)에 대해 그대로 수행한다(로직 자체는 ai.rag 쪽에 있고, 여기서는
    그 결과 dict의 구조만 본다)."""
    document_fact_claims = [c for c in grounding["claims"] if c.get("claim_type") == "document_fact"]
    if not document_fact_claims:
        return False
    unsupported_ids = {c["claim_id"] for c in grounding["unsupported_claims"]}
    return all(c.get("claim_id") in unsupported_ids for c in document_fact_claims)

_CANVAS_TEXT_FIELDS = ("problem", "target_user", "core_value", "solution", "differentiation", "contest_fit")
_VALID_CANVAS_FEASIBILITY = {"high", "medium", "low", ""}
_MAX_CANVAS_RISKS = 4

_VALID_DISCUSSION_STANCES = {"동의", "조건부_동의", "반박", "보완", "대안_제시"}
_EXPERT_ROLE_LABELS = {
    "planning_expert": "기획 위원",
    "dev_expert": "개발 위원",
    "ideation_facilitator": "진행자",
    "user": "사용자",
}
_SELF_REFERENCE_PATTERNS = {
    "planning_expert": re.compile(r"기획\s*(?:전문가|위원|측|자)(?:가|이|은|는|의|에서|에게|대로)?"),
    "dev_expert": re.compile(r"개발\s*(?:전문가|위원|측|자)(?:가|이|은|는|의|에서|에게|대로)?"),
}
_EMPTY_MEETING_PHRASES = (
    "좋은 의견입니다",
    "중요성에 동의",
    "중요한 부분입니다",
    "구체적인 계획이 필요",
    "구체적인 고민이 필요",
    "추가적인 논의가 필요",
    "추가 논의가 필요",
)
# 용준/Claude(2026-07-21): 질문 노드가 반환하는 expected_answer_type의 허용값. sufficiency
# 판정이 "답변 충분성"(방금 질문에 답했는가)과 "아이디어 완성도"(전체적으로 충분히
# 구체적인가)를 혼동하지 않도록 질문의 성격을 알려주는 보조 정보다 — preference/selection은
# 하나를 명확히 고르기만 해도 충분하다고 판정해야 한다(ideation_conv_sufficiency.txt 참고).
_VALID_EXPECTED_ANSWER_TYPES = {"preference", "selection", "definition", "constraint", "evidence", "specification"}

# 용준/Claude(2026-07-21): 사용자 메시지 3분류 — "용어가 무슨 뜻인가요?" 같은 설명 요청을
# 답변 불충분으로 오판해 같은 질문을 반복하지 않도록, sufficiency 판정을 이진(충분/불충분)이
# 아니라 3분류로 바꾼다.
#   - answer: 질문에 실제로 답했다 -> 기존과 동일하게 다음 단계로 진행.
#   - clarification_request: 용어 설명/예시/선택지를 요청했다 -> 재질문 카운터를 늘리지 않고
#     설명 + 선택지를 먼저 준 뒤 같은 핵심 질문을 더 쉽게 다시 던진다.
#   - insufficient_answer: 회피/모순/미선택 등 실제로 불충분한 답이다 -> 기존 재질문 로직
#     그대로(카운터 증가, 상한 도달 시 강제 진행).
#   - expert_delegation(용준/Claude(2026-07-21), 요청: "모르겠다" UX 개선): 사용자가 질문에
#     답하는 대신 전문가 판단에 명시적으로 맡겼다("잘 모르겠어", "전문가가 정해주세요" 등) ->
#     같은 질문을 반복하지 않고, 담당 전문가가 자신의 평가 범위 안에서 합리적인 방향을
#     "임시 가정"으로 제안한 뒤 다음 단계로 진행한다(ideation_conv_run.py::_delegate_to_expert).
_VALID_ANSWER_TYPES = {"answer", "clarification_request", "insufficient_answer", "expert_delegation"}

# 용준/Claude(2026-07-21, 요청: "모르겠다" UX 개선): 명시적인 위임 표현은 작은 모델이 잘못
# 분류하지 않도록 LLM 판정(judge_answer_sufficiency) 앞에서 결정적 규칙으로 먼저 감지한다
# (요청 사항 그대로) — 매칭되면 sufficiency LLM 호출 자체를 건너뛴다. "MVP가 무슨 뜻인지
# 모르겠어요"처럼 용어 설명을 요청하는 clarification_request와 "모르겠"이 겹치므로,
# _TERM_QUESTION_MARKERS(용어 질문 표지)가 있으면 절대 위임으로 보지 않는다 — 그 경우는
# 기존 LLM 판정이 clarification_request로 분류하도록 그대로 둔다.
_TERM_QUESTION_MARKERS = (
    "무슨 뜻",
    "무슨 의미",
    "뜻인가요",
    "뜻이에요",
    "뜻이 뭐",
    "뭔가요",
    "의미인가요",
    "무엇인가요",
    "뭔지",
    "뭐에요",
    "뭐예요",
)

_EXPERT_DELEGATION_PATTERNS = (
    re.compile(r"^(음+[.,]?\s*|그냥\s*|아+\s*)*(잘\s*)?모르겠(어|어요|습니다|네요|어서)[.!]*$"),
    re.compile(r"생각해\s*본\s*적\s*(이\s*)?없"),
    re.compile(r"감(이|이가)?\s*안\s*(와|옵니다|와요)"),
    re.compile(r"전문가.{0,6}(정해|추천|판단)"),
    re.compile(r"알아서\s*(제안|정해|골라)"),
    re.compile(r"추천해\s*(주세요|줘|주실래요|주시면)"),
    re.compile(r"구체적으로\s*(설명하기|말하기|설명을?)\s*(어려워요|못\s*하겠)"),
    # "어떤 방향이 좋은지 모르겠어요" / "어떤 기술이 좋은지 모르겠어요"처럼 "어떤 (무엇)이
    # 좋을지/좋은지 모르겠다" 형태를 폭넓게 잡는다(요청 예시 그대로 — 기획/개발 질문 모두
    # 이 형태로 위임할 수 있다).
    re.compile(r"어떤\s*[가-힣]{0,12}이?\s*좋(을지|은지)\s*모르겠"),
)

# 짧고 명확한 위임 발화만 결정적으로 처리한다 — 사용자가 위임 표현과 함께 실제 내용을
# 덧붙인 긴 문장("생각해 본 적 없어요, 그런데 아마도 ~일 것 같아요")까지 무조건 위임으로
# 단정하면 실제 답변을 놓칠 위험이 있으므로, 이런 혼합 문장은 길이 제한을 넘겨 LLM
# 판정(judge_answer_sufficiency)으로 넘긴다.
_EXPERT_DELEGATION_MAX_CHARS = 30


def is_expert_delegation_request(text: str) -> bool:
    """사용자가 pending_question에 답하는 대신 전문가의 판단에 맡기겠다는 의도를 명시적으로
    밝혔는지 결정적 규칙으로 감지한다."""
    normalized = (text or "").strip()
    if not normalized or len(normalized) > _EXPERT_DELEGATION_MAX_CHARS:
        return False
    if any(marker in normalized for marker in _TERM_QUESTION_MARKERS):
        return False
    return any(pattern.search(normalized) for pattern in _EXPERT_DELEGATION_PATTERNS)

# 용준/Claude(2026-07-21, 질문 품질 개선): 실제 사용자 테스트에서 전문가 의견이 너무
# 길고 반복적이라는 문제가 확인됐다(요청 8번) — 구조화 응답 검증 단계에서 분량을 강제한다.
# 문자열을 임의로 잘라내지 않는다("의미가 잘릴 수 있으므로 단순 문자열 강제 절단은 피하세요")
# — 초과하면 _safe_call_structured_json의 기존 재시도(최대 1회) 정책을 그대로 타고, 재시도
# 후에도 초과하면 다른 구조화 검증 실패와 동일하게 phase="failed"로 처리한다(이 코드베이스
# 전체가 구조화 응답 검증 실패에 일관되게 적용하는 정책 — 후보 생성 검증 등과 동일).
_MAX_JUDGMENT_CHARS = 200
_MAX_REASON_CHARS = 400
_MAX_SUGGESTION_CHARS = 300
_MAX_CONFIRMED_ITEMS = 3
_MAX_UNCONFIRMED_ITEMS = 3
# 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — interim_conclusion(임시 결론)은
# 매 발언마다 필수다(요청 4번: "현재 임시 결론"). 다른 짧은 필드와 같은 상한을 둔다.
_MAX_INTERIM_CONCLUSION_CHARS = 200
# 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — 화면에 실제로
# 보이는 spoken_text의 분량 상한. 위원 발언/질문/위임 제안·검토는 1~3문장을 기준으로 300자,
# 진행자(라운드 정리·위임 최종 권고)는 1~2문장을 기준으로 더 짧은 200자를 둔다.
_MAX_SPOKEN_TEXT_CHARS = 300
_MAX_FACILITATOR_SPOKEN_TEXT_CHARS = 200

_STANCE_TO_MESSAGE_TYPE = {
    "동의": "agreement",
    "조건부_동의": "agreement",
    "반박": "disagreement",
}
_RECENT_MESSAGES_LIMIT = 8

# awaiting_planning_answer/awaiting_developer_answer 중 사용자가 방금 답한 질문을 던진
# 전문가가 누구인지 phase만으로 판별한다(ideation_conv_run.py의 answer_sufficiency 게이트에서
# 사용 — 요청 3번 재질문 조건).
PHASE_TO_PENDING_PERSONA = {
    "awaiting_planning_answer": "planning_expert",
    "awaiting_developer_answer": "dev_expert",
}


def _blank(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def _safe_call_structured_json(
    llm_call: LLMCall,
    prompt: str,
    validate: Callable[[dict], str | None],
    node_name: str,
    retry_note_for: Callable[[str], str] | None = None,
) -> tuple[dict | None, bool, int]:
    """JSON 파싱 + 스키마(필수 필드 비어있지 않음) 검증을 함께, 최대 2회(최초 1회 + 재시도
    1회) 시도한다 — 요청 7번 "구조화 응답이 유효하지 않으면 최대 1회 재시도" +
    "필수 문자열이 없거나 공백이면 성공으로 처리하지 않음". validate(raw)는 문제가 있으면
    이유 문자열을, 없으면 None을 반환해야 한다.

    반환값은 (raw_또는_None, 성공여부, 실제 시도 횟수) — 세 번째 값은 호출부가
    llm_calls_used를 정확히 누적하기 위한 것이다(재시도가 실제로 LLM을 한 번 더 호출하기
    때문). 실패하면 warning 로그를 남기되, 이전에 발생했던 "빈 카드" 문제를 재현하지
    않도록 실패 사유(필드명 수준)만 남기고 프롬프트 원문·LLM 원응답·사용자 입력은 로그에
    포함하지 않는다."""
    last_reason = "unknown"
    for attempt in range(1, 3):
        attempt_prompt = prompt
        if attempt > 1 and retry_note_for is not None:
            attempt_prompt += retry_note_for(last_reason)
        try:
            raw = parse_json_response(llm_call(attempt_prompt))
        except (ValueError, KeyError, TypeError):
            last_reason = "json_parse_failed"
            trace_event(
                "IDEATION_STRUCTURED_RESPONSE_VALIDATION_FAILED",
                node=node_name,
                attempt=attempt,
                reason=last_reason,
                will_retry=attempt < 2,
            )
            discard = getattr(llm_call, "discard_streamed_prompt", None)
            if callable(discard):
                discard(attempt_prompt, last_reason, attempt < 2)
            continue
        problem = validate(raw)
        if problem is None:
            return raw, True, attempt
        last_reason = problem
        trace_event(
            "IDEATION_STRUCTURED_RESPONSE_VALIDATION_FAILED",
            node=node_name,
            attempt=attempt,
            reason=last_reason,
            will_retry=attempt < 2,
        )
        discard = getattr(llm_call, "discard_streamed_prompt", None)
        if callable(discard):
            discard(attempt_prompt, last_reason, attempt < 2)
    logger.warning("[%s] 구조화 응답 검증 실패 reason=%s", node_name, last_reason)
    return None, False, 2


def _safe_discussion_fallback(
    *,
    persona_id: str,
    state: IdeationConvState,
    discussion_stage: str,
    responding_to_message_id: str | None,
    responding_to_content: str | None,
) -> dict:
    """두 번의 구조화 응답 검증이 모두 실패했을 때 저장할 서버 생성 발언.

    문서 내용을 추측하지 않는 expert_judgment만 만들고 기존 discussion 후처리와 라우터를
    그대로 통과시킨다. 따라서 일시적인 LLM 형식 오류가 회의 전체를 failed로 만들지 않는다.
    """
    issue = resolve_effective_issue(state, persona_id)
    issue_id = issue["issue_id"]
    issue_title = issue["title"]
    counterpart = _DISCUSSION_COUNTERPART.get(persona_id, "ideation_facilitator")
    role_guidance = {
        "planning_expert": {
            "problem": "대상 사용자가 겪는 상황·원인·영향을 한 문장씩 구분해 문제 정의를 구체화하겠습니다.",
            "target_user": "핵심 사용자를 하나로 좁히고 사용 상황과 가장 큰 불편을 우선 검증하겠습니다.",
            "core_value": "사용 전후의 변화를 측정할 수 있는 핵심 가치와 지표를 먼저 정하겠습니다.",
            "differentiation": "기존 방식과 비교해 사용자 경험이 달라지는 지점을 하나의 차별점으로 좁히겠습니다.",
            "mvp": "사용자 가치 검증에 꼭 필요한 기능만 남겨 초기 MVP 범위를 축소하겠습니다.",
        },
        "dev_expert": {
            "problem": "구현으로 넘어가기 전에 문제를 관측할 수 있는 데이터와 측정 지표부터 정의하겠습니다.",
            "data": "공공 데이터와 자체 수집 데이터를 구분하고 정확도·갱신 주기·수집 비용을 먼저 검증하겠습니다.",
            "ai_role": "AI가 담당할 판단과 규칙 기반으로 처리할 기능을 분리해 기술 위험을 줄이겠습니다.",
            "differentiation": "차별 기능에 필요한 데이터와 구현 난도를 확인해 실제 개발 가능한 범위로 좁히겠습니다.",
            "mvp": "핵심 데이터 흐름 하나를 기준으로 수집·분석·알림까지의 최소 기능을 먼저 검증하겠습니다.",
        },
    }.get(persona_id, {})
    guidance = role_guidance.get(
        issue_id,
        (
            "사용자 가치와 검증 기준을 먼저 정해 다음 논의를 구체화하겠습니다."
            if persona_id == "planning_expert"
            else "필요 데이터와 구현 위험을 구분해 검증 가능한 최소 범위부터 제안하겠습니다."
        ),
    )
    spoken_text = (
        f"{issue_title}에 관한 문서 사실을 추가로 단정하지 않고 전문가 판단으로 진행하겠습니다. "
        f"{guidance}"
    )
    responding_to = (
        "앞선 의견의 세부 근거를 추가로 확인해야 합니다."
        if discussion_stage == "response" and responding_to_content
        else None
    )
    return {
        "stance": "보완",
        "judgment": guidance,
        "reason": "문서에서 확인된 사실과 전문가의 제안을 분리하면서도 회의를 중단하지 않기 위한 판단입니다.",
        "suggestion": guidance,
        "interim_conclusion": f"{issue_title}은 위 검증 방향을 임시 기준으로 삼아 다음 논의를 이어갑니다.",
        "spoken_text": spoken_text,
        "responding_to": responding_to,
        "agreement": "추가 검토가 필요하다는 점은 수용합니다." if responding_to else None,
        "concern": "현재 자료만으로 구체적인 사실을 단정할 수 없습니다." if responding_to else None,
        "revision": None,
        "confirmed": [],
        "unconfirmed": [],
        "referenced_message_ids": [responding_to_message_id] if responding_to_message_id else [],
        "claims": [
            {
                "claim_id": "safe_fallback_judgment",
                "text": guidance,
                "claim_type": "expert_judgment",
                "evidence_refs": [],
            }
        ],
        "next_action": None,
        "active_issue_id": issue_id,
        "active_issue_title": issue_title,
        # 구조화 응답이 두 번 모두 실패했다는 것은 이 턴에서 검증 가능한 새 논점을
        # 얻지 못했다는 뜻이다. 이를 새 정보/대안으로 저장하면 다음 턴이 서버가 만든
        # 상투적인 문구를 다시 반박하며 같은 쟁점을 이어간다.
        "new_information": [],
        "proposal": None,
        "changed_position": False,
        "needs_counterpart_response": False,
        "recommended_next_speaker": "ideation_facilitator",
        "issue_resolved": False,
        "needs_user_input": False,
        "user_question": None,
        "safe_fallback": True,
        "safe_fallback_reason": "structured_response_validation_failed_twice",
    }


def _safe_fallback_spoken_text(grounding: dict) -> str:
    """검증에 최종 실패했을 때 화면에 노출할 안전한 발언(요청 7번). 근거 없는 사실을 확정
    표현으로 내보내는 대신, 확인이 필요한 항목을 있는 그대로 말한다."""
    missing = [m for m in grounding["missing_information"] if m][:2]
    if missing:
        detail = " · ".join(missing)
        text = f"현재 제공된 자료에서는 {detail} 부분을 문서로 확인하기 어렵습니다. 추가 확인이 필요합니다."
    else:
        text = "현재 제공된 자료에서는 이 주장을 문서 근거로 확인하기 어렵습니다. 추가 확인이 필요합니다."
    return text[:_MAX_SPOKEN_TEXT_CHARS]


def _evidence_unavailable_discussion_response(
    *,
    persona_id: str,
    state: IdeationConvState,
    discussion_stage: str,
    responding_to_message_id: str | None,
    responding_to_content: str | None,
) -> dict:
    """Planner가 현재 쟁점에 적격 근거가 없다고 확정한 턴을 일반 전문가 의견으로 만들지 않는다."""
    raw = _safe_discussion_fallback(
        persona_id=persona_id,
        state=state,
        discussion_stage=discussion_stage,
        responding_to_message_id=responding_to_message_id,
        responding_to_content=responding_to_content,
    )
    issue_title = raw.get("active_issue_title") or "현재 쟁점"
    notice = (
        f"{issue_title}에 직접 연결할 수 있는 문서 근거를 찾지 못해 이번 전문가 판단은 "
        "생성하지 않고 다음 쟁점 판단으로 넘기겠습니다."
    )
    return {
        **raw,
        "judgment": notice,
        "reason": "Evidence Planner가 현재 쟁점에 적격한 근거를 선택하지 못했습니다.",
        "suggestion": "진행자가 근거 부족을 기록하고 다음 쟁점으로 이동합니다.",
        "interim_conclusion": notice,
        "spoken_text": notice,
        "claims": [],
        "new_information": [],
        "proposal": None,
        "needs_counterpart_response": False,
        "recommended_next_speaker": "ideation_facilitator",
        "safe_fallback_reason": "evidence_first_no_eligible_evidence",
        "evidence_first_skipped": True,
    }


def _evidence_anchor_response(raw: dict, retrieved: list[dict]) -> dict | None:
    """LLM 응답이 근거를 하나도 연결하지 못했을 때 선택 근거 자체로 안전한 발언을 만든다.

    근거 quote를 그대로 claim text로 사용하므로 새로운 문서 사실을 만들지 않는다. target
    근거를 criteria보다 우선해 아이디어 자체에 대한 논의가 평가표 일반론보다 앞서게 한다.
    """
    candidates = [
        item
        for item in retrieved
        if isinstance(item, dict)
        and isinstance(item.get("ref"), str)
        and item.get("ref")
        and isinstance(item.get("quote") or item.get("text"), str)
        and (item.get("quote") or item.get("text")).strip()
    ]
    if not candidates:
        return None
    evidence = sorted(
        candidates,
        key=lambda item: (0 if item.get("document_role") == "target" else 1),
    )[0]
    quote = str(evidence.get("quote") or evidence.get("text")).strip()
    ref = evidence["ref"]
    claim_type = evidence.get("claim_type")
    if claim_type not in ("document_fact", "user_provided_fact"):
        claim_type = "user_provided_fact" if evidence.get("document_role") == "target" else "document_fact"
    issue_title = raw.get("active_issue_title") or "현재 쟁점"
    judgment = f"{issue_title}에서는 이 근거가 요구하거나 설명하는 내용을 구체화해야 합니다."
    spoken_text = f'근거 자료에는 “{quote}”라고 제시되어 있습니다. {judgment}'
    return {
        **raw,
        "judgment": judgment,
        "reason": quote,
        "suggestion": judgment,
        "interim_conclusion": judgment,
        "spoken_text": spoken_text[:_MAX_SPOKEN_TEXT_CHARS],
        "confirmed": [quote],
        "unconfirmed": [],
        "claims": [
            {
                "claim_id": "evidence_anchor_fact",
                "text": quote,
                "claim_type": claim_type,
                "evidence_refs": [ref],
            },
            {
                "claim_id": "evidence_anchor_judgment",
                "text": judgment,
                "claim_type": "expert_judgment",
                "evidence_refs": [],
            },
        ],
        "new_information": [quote],
        "proposal": None,
        "changed_position": False,
        "needs_counterpart_response": False,
        "recommended_next_speaker": "ideation_facilitator",
        "needs_user_input": False,
        "user_question": None,
        "evidence_first_fallback": True,
    }


def _ground_and_finalize_claims(
    *,
    persona_id: str,
    raw: dict,
    retrieved: list[dict],
    prompt: str,
    llm_call: LLMCall,
    validate: Callable[[dict], str | None],
    used: int,
    ground_claims_fn: ClaimGroundingFn | None,
    require_linked_evidence: bool = False,
) -> tuple[dict, dict, int]:
    """구조화 검증(_safe_call_structured_json)을 통과한 raw 응답의 claims를 실제 검색 근거와
    대조 검증한다(요청: 최종 발언의 주장과 실제 청크가 연결되고 관련성 검증을 통과해야만
    RAG 활용 성공으로 판단). document_fact 주장이 전부 근거 연결에 실패하면 실패 사유를
    프롬프트에 덧붙여 딱 한 번만 재생성한다 — 무한 재시도는 하지 않는다(기존
    _safe_call_structured_json의 재시도 1회 정책과는 별개 축이다: 그쪽은 "필수 필드
    누락"을, 이쪽은 "구조는 유효했지만 근거 연결에 실패"를 다룬다). 재시도 후에도 실패하면
    spoken_text를 안전한 표현으로 교체한 뒤 그대로 진행한다(발언 자체를 폐기하지 않는다 —
    요청: 회의가 중단되지 않아야 한다).

    ground_claims_fn이 없으면(use_rag 미사용 세션 등) 검증을 건너뛰고 빈 grounding 결과를
    돌려준다 — retrieved가 없을 때 evidence_lookup을 건너뛰는 것과 같은 원칙이다."""
    if ground_claims_fn is None:
        return raw, dict(_EMPTY_GROUNDING), used

    started = time.perf_counter()
    trace_event(
        "IDEATION_CLAIM_GROUNDING_START",
        speaker=persona_id,
        claim_count=len(raw.get("claims") or []),
        retrieved_evidence_count=len(retrieved),
        injected_evidence_count=len(retrieved),
    )
    grounding = ground_claims_fn(persona_id, raw.get("claims"), retrieved)

    if _has_hard_grounding_failure(grounding):
        reasons = sorted({c["reason"] for c in grounding["unsupported_claims"]})
        trace_event("IDEATION_GROUNDING_RETRY", speaker=persona_id, retry_count=1, reasons=reasons)
        # 용준/Claude(2026-07-23, 요청: grounding 재시도 프롬프트의 ref 계약 오류 수정) —
        # 이 안내문은 예전에 evidence_refs가 chunk_id(해시)를 직접 담던 시절 문구가 그대로
        # 남아 있었다. call_evidence_lookup이 각 근거에 짧은 순번 참조("ref": "E1")를 부여한
        # 뒤부터는 evidence_refs에 chunk_id를 쓰라고 재시도 안내가 말하면 프롬프트 본문의
        # [근거 인용 규칙](ref만 사용)과 재시도 안내가 서로 다른 지시를 하게 된다 — 재시도
        # 안내도 반드시 ref 계약으로 통일한다. ref -> 실제 chunk_id 변환은 항상 서버의
        # ground_claims가 담당하고 LLM은 절대 chunk_id를 직접 다루지 않는다.
        retry_note = (
            "\n\n[근거 연결 재시도 안내] 방금 응답의 claims 중 document_fact로 표시한 주장이 "
            "retrieved_evidence의 실제 ref와 연결되지 않았습니다(사유: "
            + ", ".join(reasons)
            + "). 존재하지 않는 ref를 인용했거나, 인용한 근거가 그 주장과 무관하거나, "
            "evidence_refs가 비어 있었습니다. evidence_refs에는 retrieved_evidence 각 항목의 "
            "\"ref\" 필드 값(예: \"E1\", \"E2\")만 그대로 쓰고, chunk_id(긴 해시 문자열)는 "
            "직접 쓰지 마세요 — ref와 실제 chunk_id 사이의 변환은 서버가 처리합니다. "
            "retrieved_evidence에 실제로 있는 ref만 evidence_refs에 넣고, 문서에서 확인할 수 "
            "없는 내용은 claim_type을 expert_judgment로 바꾸거나 claims에서 제외하세요."
        )
        # 용준/Claude(2026-07-23, 요청: 스트리밍 UX 버그 수정 — grounding retry에서 이전
        # 초안과 재시도 초안이 동시에 화면에 남는 문제): 이 재시도는 _safe_call_structured_json
        # 의 재시도 축과 달리 프롬프트 문자열이 달라지므로(prompt + retry_note), llm_call
        # 내부의 "동일 prompt 재호출" 감지로는 이전 스트림 말풍선이 자동으로 지워지지
        # 않는다. 재시도 호출 전에 명시적으로 이전 스트림을 discard해야, 프런트가 두 초안을
        # 동시에 들고 있지 않고 "검토 중" 표시로 전환한 뒤 재시도 말풍선으로 교체한다.
        discard = getattr(llm_call, "discard_streamed_prompt", None)
        if callable(discard):
            discard(prompt, "grounding_retry", True)
        try:
            retry_raw = parse_json_response(llm_call(prompt + retry_note))
        except (ValueError, KeyError, TypeError):
            retry_raw = None
        if retry_raw is not None and validate(retry_raw) is None:
            used += 1
            raw = retry_raw
            grounding = ground_claims_fn(persona_id, raw.get("claims"), retrieved)

    if _has_hard_grounding_failure(grounding):
        raw = dict(raw)
        raw["spoken_text"] = _safe_fallback_spoken_text(grounding)

    if require_linked_evidence and retrieved and grounding["linked_evidence_count"] == 0:
        anchored_raw = _evidence_anchor_response(raw, retrieved)
        if anchored_raw is not None:
            anchored_grounding = ground_claims_fn(
                persona_id,
                anchored_raw.get("claims"),
                retrieved,
            )
            if anchored_grounding["linked_evidence_count"] > 0:
                raw = anchored_raw
                grounding = anchored_grounding
                trace_event(
                    "IDEATION_EVIDENCE_FIRST_FALLBACK",
                    speaker=persona_id,
                    selected_ref=anchored_raw["claims"][0]["evidence_refs"][0],
                    linked_evidence_count=grounding["linked_evidence_count"],
                    reason="generated_response_had_no_linked_evidence",
                )

    trace_event(
        "IDEATION_CLAIM_GROUNDING_RESULT",
        speaker=persona_id,
        claim_count=len(grounding["claims"]),
        supported_claim_count=grounding["supported_claim_count"],
        unsupported_claim_count=grounding["unsupported_claim_count"],
        # 용준/Claude(2026-07-22, 요청: claim 통계 의미 분리) — supported_claim_count(기존
        # 필드, 호환 유지)와 별도로 "실제 문서 근거로 검증됨(grounded)"과 "근거 없이 허용된
        # 전문가 판단(expert_judgment)"을 분리해서 남긴다. linked_evidence_count=0인데
        # supported/accepted_claim_count>0인 것만 보고 "근거를 썼다"고 오인하지 않도록 한다.
        accepted_claim_count=grounding["accepted_claim_count"],
        grounded_claim_count=grounding["grounded_claim_count"],
        expert_judgment_count=grounding["expert_judgment_count"],
        linked_evidence_count=grounding["linked_evidence_count"],
        evidence_status=grounding["evidence_status"],
        missing_information=grounding["missing_information"],
        unsupported_reasons=[c["reason"] for c in grounding["unsupported_claims"]],
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    # 용준/Claude(2026-07-23, 요청: IDEATION_EVIDENCE_LINKED 로그 매핑 수정) — claim이
    # evidence_refs에 담는 값은 LLM이 인용한 ref("E1")이고, grounding["linked_evidence_refs"]는
    # 항상 실제 chunk_id다. 두 값 공간이 다르므로 "ref in linked_evidence_refs" 직접 비교는
    # (ref와 chunk_id가 우연히 같은 레거시 fixture를 빼면) 사실상 항상 실패했다 — 이제
    # ground_claims가 claim 단위로 짝지어 반환하는 claim_evidence_links(ref, chunk_id 쌍)를
    # 그대로 로그에 옮긴다. 구버전 ground_claims_fn(이 필드가 없는 테스트 더블 등)이 주입된
    # 경우를 위해 없으면 빈 리스트로 안전하게 처리한다.
    for link in grounding.get("claim_evidence_links") or []:
        trace_event(
            "IDEATION_EVIDENCE_LINKED",
            speaker=persona_id,
            claim_id=link["claim_id"],
            evidence_refs=link["evidence_refs"],
            chunk_ids=link["chunk_ids"],
        )
    return raw, grounding, used


def _make_validate_question_response(
    resolved_topics: list[str], roadmap_allowed: bool, require_combine_structure: bool = False
) -> Callable[[dict], str | None]:
    """질문 노드 응답 검증기를 만든다(요청: 질문 주제 구조화). resolved_topics/roadmap_allowed는
    이번 호출 시점의 state에서 계산된 값을 클로저로 고정한다 — 이 두 값 자체는 검증 중에
    바뀌지 않는다(같은 호출 안에서의 재시도 1회 동안은 동일한 상태를 기준으로 판정해야
    일관적이다).

    질문 노드가 반환한 question_topic이 다음 중 하나라도 해당하면 무효(재시도 유발)다 —
    "구조화 응답 실패"와 동일하게 취급해 phase="failed"로 이어질 수 있는 것은 다른 구조화
    검증과 같은 정책이다.
      - 허용되지 않은 값이거나 아예 없다(요청 10번: question_topic 누락/허용되지 않은 값 처리).
      - 이미 resolved_topics에 있는 주제다(요청 9번: 이미 해결된 주제를 다시 묻지 않는다 —
        프롬프트 규칙만으로는 보장이 안 되므로 코드로 한 번 더 막는다).
      - "roadmap"인데 roadmap_allowed=False다(요청: 선행 주제 미확인 시 roadmap 질문 금지 —
        이 검증이 그 강제의 최종 방어선이다. 1차 방어선은 애초에 프롬프트에 remaining_topics
        에서 roadmap을 아예 제외해 넘기는 것이다).

    require_combine_structure=True(용준/Claude(2026-07-21, 후보 결합 컨텍스트 보존)면 이번
    호출이 "후보 결합 직후 첫 전문가 메시지"다(요청 6번) — user_selection_summary/proposal
    필드도 비어있지 않아야 한다(_compose_question_content가 이 값들로 [사용자 선택 반영]/
    [제안] 섹션을 만든다)."""
    resolved_set = set(resolved_topics)

    def _validate(raw: dict) -> str | None:
        topic = raw.get("question_topic")
        if topic not in TOPIC_PRIORITY:
            return "invalid_or_missing_question_topic"
        if topic in resolved_set:
            return "question_topic_already_resolved"
        if topic == "roadmap" and not roadmap_allowed:
            return "roadmap_prerequisites_not_met"
        if _blank(raw.get("judgment")) or _blank(raw.get("question")):
            return "missing_or_empty_field:judgment_or_question"
        if _blank(raw.get("spoken_text")):
            return "missing_or_empty_field:spoken_text"
        if len(raw.get("spoken_text", "")) > _MAX_SPOKEN_TEXT_CHARS:
            return "spoken_text_too_long"
        if require_combine_structure:
            if _blank(raw.get("user_selection_summary")):
                return "missing_or_empty_field:user_selection_summary"
            if _blank(raw.get("proposal")):
                return "missing_or_empty_field:proposal"
        return None

    return _validate


def validate_spoken_text_speaker_reference(
    current_speaker_id: str,
    responding_to_speaker_id: str | None,
    spoken_text: str,
) -> str | None:
    """화자가 자기 역할을 상대방처럼 부르는 명백한 오류를 결정적으로 거부한다."""
    pattern = _SELF_REFERENCE_PATTERNS.get(current_speaker_id)
    if pattern and pattern.search(spoken_text or ""):
        trace_event(
            "IDEATION_SPEAKER_REFERENCE_WARNING",
            level=logging.WARNING,
            speaker=current_speaker_id,
            target=responding_to_speaker_id,
            reason="self_role_reference",
            text=sanitize_preview(spoken_text, limit=200),
        )
        return "spoken_text_self_role_reference"
    for mentioned_speaker_id, mentioned_pattern in _SELF_REFERENCE_PATTERNS.items():
        if mentioned_speaker_id != responding_to_speaker_id and mentioned_pattern.search(spoken_text or ""):
            trace_event(
                "IDEATION_SPEAKER_REFERENCE_WARNING",
                level=logging.WARNING,
                speaker=current_speaker_id,
                target=responding_to_speaker_id,
                mentioned=mentioned_speaker_id,
                reason="role_reference_target_mismatch",
                text=sanitize_preview(spoken_text, limit=200),
            )
            return "spoken_text_role_reference_target_mismatch"
    if any(phrase in (spoken_text or "") for phrase in _EMPTY_MEETING_PHRASES):
        return "spoken_text_report_like_or_empty_phrase"
    return None


_RESTATEMENT_STOPWORDS = frozenset(
    {
        "것",
        "점",
        "현재",
        "매우",
        "단순",
        "실제",
        "다만",
        "하지만",
        "그리고",
        "따라서",
        "아니라",
        "어떻게",
        "통해",
        "기반",
        "위해",
        "대한",
        "관련",
        "문제",
        "중요",
        "중요한",
        "필요",
        "필요한",
        "필수",
        "핵심",
        "방안",
        "방법",
        "전략",
        "체계",
        "마련",
        "고려",
        "결정",
        "과정",
        "이유",
        "설계",
        "개선",
        "시스템",
        "확대",
        "기회",
        "보장",
        "보장할",
        "효과적",
        "있습니다",
        "합니다",
        "됩니다",
    }
)
_RESTATEMENT_SUFFIX_RE = re.compile(
    r"(입니다|합니다|됩니다|있습니다|없습니다|해야합니다|해야|하며|하고|하는|한|된|"
    r"에서|에게|으로|부터|까지|처럼|보다|과|와|을|를|은|는|이|가|에|도|만)$"
)
_RESTATEMENT_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("피드백", "의견수렴", "의견"), "피드백"),
    (("마케팅", "프로모션", "홍보", "유도"), "참여유도"),
    (("정확성", "정확도", "신뢰성", "신뢰도", "품질"), "품질"),
    (("수집", "확보"), "확보"),
    (("연계", "연동", "통합"), "통합"),
    (("시민", "주민"), "시민"),
    (("참여",), "참여"),
    (("정책",), "정책"),
    (("반영",), "반영"),
    (("소통",), "소통"),
    (("경로", "채널"), "소통경로"),
)


def _restatement_terms(text: str | None) -> set[str]:
    """표현만 바꾼 회의 발언을 비교하기 위한 결정론적 핵심어 집합."""
    normalized = (text or "").lower().replace("의견 수렴", "의견수렴")
    terms: set[str] = set()
    for raw_token in re.findall(r"[가-힣A-Za-z0-9]+", normalized):
        token = raw_token
        previous = None
        while token != previous:
            previous = token
            token = _RESTATEMENT_SUFFIX_RE.sub("", token)
        if (
            len(token) < 2
            or token in _RESTATEMENT_STOPWORDS
            or any(
                token.startswith(prefix)
                for prefix in ("중요", "필요", "구체", "명확", "효과적", "실제로", "필수")
            )
        ):
            continue
        canonical = token
        for aliases, replacement in _RESTATEMENT_ALIASES:
            if any(alias in token for alias in aliases):
                canonical = replacement
                break
        if canonical not in _RESTATEMENT_STOPWORDS:
            terms.add(canonical)
    return terms


def _restatement_similarity(spoken_text: str, previous_content: str | None) -> float:
    """두 발언의 핵심어 유사도. 짧은 쪽의 재진술도 잡도록 Jaccard와 포함률을 함께 본다."""
    current = _restatement_terms(spoken_text)
    previous = _restatement_terms(previous_content)
    if len(current) < 3 or len(previous) < 3:
        return 0.0
    shared = len(current & previous)
    if shared < 3:
        return 0.0
    union = len(current | previous)
    jaccard = shared / union if union else 0.0
    containment = shared / min(len(current), len(previous))
    return max(jaccard, containment * 0.8)


def _looks_like_restatement(spoken_text: str, responding_to_content: str | None) -> bool:
    """핵심 주장이 같은 발언을 감지한다. LLM의 new_information 자기 신고는 사용하지 않는다."""
    return _restatement_similarity(spoken_text, responding_to_content) >= 0.45


def _looks_like_near_verbatim_restatement(spoken_text: str, responding_to_content: str | None) -> bool:
    """응답 검증용의 엄격한 반복 판정.

    상대 발언을 인용한 뒤 수정안을 덧붙이는 정상 응답은 핵심어 포함률이 높다. 따라서 검증
    단계에서는 예전 계약처럼 원문 토큰 Jaccard가 0.82 이상인 사실상 복사 발언만 거부한다.
    의미 반복의 조기 라우팅은 별도의 다중 메시지 검사에서 담당한다.
    """
    if not responding_to_content:
        return False
    tokenize = lambda text: {
        token
        for token in re.findall(r"[가-힣A-Za-z0-9]+", (text or "").lower())
        if len(token) >= 2
    }
    current = tokenize(spoken_text)
    previous = tokenize(responding_to_content)
    if len(current) < 5 or len(previous) < 5:
        return False
    return len(current & previous) / len(current | previous) >= 0.82


def _recent_issue_restatement_matches(
    messages: list[dict],
    *,
    issue_id: str | None,
    spoken_text: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """현재 발언과 의미가 겹치는 최근 동일 쟁점 전문가 발언을 최대 ``limit``개 찾는다."""
    if not issue_id:
        return []
    candidates: list[dict] = []
    for message in reversed(messages):
        structured = message.get("structured") or {}
        if structured.get("active_issue_id") != issue_id:
            continue
        if message.get("speaker_id") not in ("planning_expert", "dev_expert"):
            continue
        candidates.append(message)
        if len(candidates) >= limit:
            break

    matches: list[dict[str, Any]] = []
    for message in candidates:
        score = _restatement_similarity(spoken_text, message.get("content"))
        if score >= 0.45:
            matches.append(
                {
                    "message_id": message.get("message_id"),
                    "speaker_id": message.get("speaker_id"),
                    "similarity": round(score, 3),
                }
            )
    return matches


def _issue_evidence_exhausted(
    messages: list[dict],
    *,
    issue_id: str | None,
    current_speaker_id: str,
    current_linked_chunk_ids: list[str],
) -> bool:
    """두 전문가가 검토를 마쳤고 현재 근거가 모두 재사용이면 쟁점 근거가 소진된 것으로 본다.

    문장 유사도만으로는 한국어 어미와 표현 변경 때문에 같은 결론을 놓칠 수 있다. 반면 active
    Planner가 같은 쟁점에 같은 chunk만 다시 주입했다는 사실은 결정론적으로 확인 가능하다.
    첫 화자만 말한 상태에서는 상대 관점을 보장하기 위해 닫지 않고, 현재 턴까지 기획/개발
    양쪽이 모두 발언한 경우에만 적용한다.
    """
    if not issue_id or not current_linked_chunk_ids:
        return False

    prior_chunk_ids: set[str] = set()
    speakers = {current_speaker_id}
    for message in messages:
        structured = message.get("structured") or {}
        if structured.get("active_issue_id") != issue_id:
            continue
        speaker_id = message.get("speaker_id")
        if speaker_id not in ("planning_expert", "dev_expert"):
            continue
        speakers.add(speaker_id)
        prior_chunk_ids.update(
            chunk_id
            for chunk_id in (message.get("linked_evidence_refs") or [])
            if isinstance(chunk_id, str) and chunk_id
        )

    return (
        speakers == {"planning_expert", "dev_expert"}
        and bool(prior_chunk_ids)
        and set(current_linked_chunk_ids).issubset(prior_chunk_ids)
    )


_USER_QUESTION_STOPWORDS = {
    "관련", "관해", "대해", "대한", "문제", "문제를", "어떻게", "무엇", "뭐가",
    "해결", "해결해야지", "해결해야하지", "궁금", "궁금해", "알려줘", "설명해줘",
    "가능", "가능한가요", "되나요", "할까요", "해야하나요", "해야하지",
}

_ISSUE_FOCUS_MARKERS: dict[str, tuple[str, ...]] = {
    "problem": ("문제", "불편", "위험", "피해", "원인", "영향", "비효율", "오염", "어려", "부족"),
    "target_user": ("사용자", "이용자", "고객", "시민", "주민", "대상", "상황"),
    "core_value": ("가치", "효과", "개선", "절감", "편의", "안전", "혜택"),
    "contest_fit": ("공모", "평가", "심사", "기준", "주제", "적합"),
    "differentiation": ("차별", "기존", "대비", "독창", "혁신", "경쟁"),
    "mvp": ("MVP", "최소", "우선", "범위", "초기", "핵심 기능"),
    "data": ("데이터", "수집", "확보", "품질", "센서", "연동"),
    "ai_role": ("AI", "모델", "알고리즘", "예측", "분석", "자동화"),
    "roadmap": ("확장", "단계", "로드맵", "향후", "도입", "고도화"),
}

_ISSUE_DRIFT_MARKERS: dict[str, tuple[str, ...]] = {
    # 문제 정의를 한 번 언급한 뒤 곧바로 구현/MVP/확장성으로 넘어가는 실측 실패를 차단한다.
    "problem": (
        "MVP",
        "구현",
        "데이터 확보",
        "데이터 수집",
        "센서 연동",
        "API",
        "운영 비용",
        "개인정보",
        "보안",
        "확장성",
        "적용성",
        "혁신성",
        "차별성",
        "KPI",
        "로드맵",
    ),
}

_EVALUATIVE_CONCLUSION_MARKERS = (
    "부족",
    "미흡",
    "구체적이지 않",
    "불명확",
    "우려",
    "약합니다",
    "타당하지",
    "문제가 있",
    "필요",
    "해야",
    "타당",
    "적절",
    "생각",
    "가치",
    "효과",
    "권고",
    "제안",
)
_FACTUAL_ATTRIBUTION_MARKERS = (
    "공고문에 명시",
    "문서에 명시",
    "자료에 명시",
    "공고문에 따르면",
    "문서에 따르면",
    "자료에 따르면",
)


def _has_evaluative_conclusion(spoken_text: str) -> bool:
    """문서 사실의 단순 전달이 아니라 전문가의 평가·권고가 포함됐는지 판정한다."""
    return (
        any(marker in spoken_text for marker in _EVALUATIVE_CONCLUSION_MARKERS)
        and not any(marker in spoken_text for marker in _FACTUAL_ATTRIBUTION_MARKERS)
    )


def _repair_evaluative_expert_judgment_claim(
    raw: dict,
    evidence_claim_types_by_ref: dict[str, str],
) -> None:
    """평가기준 인용과 그 기준을 적용한 전문가 판단을 한 claim으로 뭉친 응답을 안전하게 분리한다.

    평가기준 자체는 document_fact지만, 아이디어가 부족하거나 보완이 필요하다는 결론은
    expert_judgment다. 이 분류는 결정론적으로 확정할 수 있으므로 동일 프롬프트를 다시 호출해
    비용과 실패율을 늘리는 대신 별도 claim을 추가한다. 원래 document_fact claim과 ref는
    변경하지 않으며 새로운 문서 사실도 만들지 않는다.
    """
    claims = [claim for claim in (raw.get("claims") or []) if isinstance(claim, dict)]
    has_evidence_fact_claim = any(
        claim.get("claim_type") in ("document_fact", "user_provided_fact")
        for claim in claims
    )
    has_expert_judgment_claim = any(claim.get("claim_type") == "expert_judgment" for claim in claims)
    spoken_text = raw.get("spoken_text") or ""
    if (
        not evidence_claim_types_by_ref
        or not has_evidence_fact_claim
        or has_expert_judgment_claim
        or not _has_evaluative_conclusion(spoken_text)
    ):
        return

    judgment_text = (raw.get("judgment") or spoken_text).strip()
    if not judgment_text:
        return
    used_ids = {
        claim.get("claim_id")
        for claim in claims
        if isinstance(claim.get("claim_id"), str)
    }
    suffix = 1
    claim_id = "claim_expert_judgment"
    while claim_id in used_ids:
        suffix += 1
        claim_id = f"claim_expert_judgment_{suffix}"
    claims.append(
        {
            "claim_id": claim_id,
            "text": judgment_text,
            "claim_type": "expert_judgment",
            "evidence_refs": [],
        }
    )
    raw["claims"] = claims
    trace_event(
        "IDEATION_DISCUSSION_CLAIM_REPAIRED",
        reason="spoken_evaluation_missing_expert_judgment_claim",
        added_claim_id=claim_id,
        claim_type="expert_judgment",
    )


def _discussion_retry_note(reason: str) -> str:
    guidance = {
        "spoken_text_issue_mismatch": (
            "spoken_text의 모든 문장을 현재 active_issue에 직접 맞추고 다른 쟁점으로 이동하지 마세요."
        ),
        "spoken_text_issue_drift": (
            "현재 쟁점의 판단만 말하고 MVP·구현·비용·확장성 등 다음 쟁점의 내용은 제외하세요."
        ),
        "active_issue_id_mismatch": "active_issue_id와 active_issue_title을 입력으로 받은 현재 쟁점과 정확히 일치시키세요.",
        "spoken_text_does_not_answer_user_question": "spoken_text의 첫 문장부터 사용자의 질문에 직접 답하세요.",
        "claim_type_evidence_role_mismatch": "각 claim_type을 인용한 evidence ref의 claim_type과 일치시키세요.",
        "claim_mixes_document_roles": "criteria 근거와 target 근거를 서로 다른 claim으로 분리하세요.",
        "document_fact_missing_evidence": "document_fact에는 실제 retrieved_evidence의 ref를 넣으세요.",
        "user_provided_fact_missing_target_evidence": "user_provided_fact에는 실제 target 근거 ref를 넣으세요.",
        "expert_judgment_must_not_cite_evidence": "expert_judgment의 evidence_refs는 빈 배열로 두세요.",
    }.get(reason, "검증 실패 사유를 수정하되 기존 JSON 스키마와 현재 쟁점을 그대로 유지하세요.")
    return (
        "\n\n[구조화 응답 재시도]\n"
        f"이전 응답이 검증에 실패했습니다. 실패 코드: {reason}\n"
        f"수정 지침: {guidance}\n"
        "JSON 객체 하나만 다시 반환하고, 이전의 잘못된 문장을 그대로 반복하지 마세요."
    )


_USER_DECISION_MARKERS = (
    "예산",
    "비용 상한",
    "대상 지역",
    "대상 기관",
    "핵심 사용자",
    "우선순위",
    "선택",
    "선호",
    "일정",
    "마감",
    "보유 데이터",
    "보유 센서",
    "내부 시스템",
    "mvp",
    "mvp 범위",
    "운영 주체",
)


def _is_actionable_user_decision_question(question: str | None) -> bool:
    """사용자만 결정할 수 있는 제품 제약·선택 질문인지 보수적으로 판정한다."""
    normalized = (question or "").strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _USER_DECISION_MARKERS)


def _user_question_focus_terms(text: str) -> set[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", (text or "").lower())
    terms: set[str] = set()
    for token in tokens:
        normalized = re.sub(
            r"(인가요|한가요|할까요|해야하나요|해야하지|입니다|습니까|에서|으로|에게|부터|까지|"
            r"과|와|을|를|은|는|이|가|에|도|만|면)$",
            "",
            token,
        )
        if len(normalized) >= 2 and normalized not in _USER_QUESTION_STOPWORDS:
            terms.add(normalized)
    return terms


def _answers_user_question(spoken_text: str, user_question: str) -> bool:
    focus_terms = _user_question_focus_terms(user_question)
    if not focus_terms:
        return True
    normalized_answer = (spoken_text or "").lower()
    return any(term in normalized_answer for term in focus_terms)


def _spoken_text_matches_issue(spoken_text: str, issue_id: str, issue_title: str | None = None) -> bool:
    """구조화 issue_id만 맞춰 쓰고 실제 발언은 다른 주제로 이탈하는 경우를 막는다."""
    normalized = (spoken_text or "").lower()
    markers = _ISSUE_FOCUS_MARKERS.get(issue_id)
    if markers:
        return any(marker.lower() in normalized for marker in markers)
    title_terms = _user_question_focus_terms(issue_title or "")
    return not title_terms or any(term in normalized for term in title_terms)


def _spoken_text_issue_validation_reason(
    spoken_text: str,
    issue_id: str,
    issue_title: str | None = None,
) -> str | None:
    if not _spoken_text_matches_issue(spoken_text, issue_id, issue_title):
        return "spoken_text_issue_mismatch"
    drift_markers = _ISSUE_DRIFT_MARKERS.get(issue_id, ())
    if not drift_markers:
        return None
    normalized = (spoken_text or "").lower()
    has_drift = any(marker.lower() in normalized for marker in drift_markers)
    focus_count = sum(
        1
        for marker in _ISSUE_FOCUS_MARKERS.get(issue_id, ())
        if marker.lower() in normalized
    )
    # 문제 marker 하나만 형식적으로 넣고 본문 대부분이 구현 계획인 경우에만 재시도한다.
    if has_drift and focus_count < 2:
        return "spoken_text_issue_drift"
    return None


def _validate_discussion_response(
    raw: dict,
    discussion_stage: str = "initial_position",
    current_speaker_id: str | None = None,
    responding_to_speaker_id: str | None = None,
    responding_to_content: str | None = None,
    require_user_question_focus: bool = False,
    expected_issue_id: str | None = None,
    expected_issue_title: str | None = None,
    require_issue_content_focus: bool = True,
    evidence_claim_types_by_ref: dict[str, str] | None = None,
) -> str | None:
    """용준/Claude(2026-07-21, 요청: 위원 간 실제 회의로 개편): discussion_stage가
    "review"(상대 의견 검토) 또는 "revision"(검토 반영 수정)이면 responding_to가 비어
    있으면 재시도를 유발한다 — "좋은 의견입니다" 류의 빈 상호참조를 막기 위한 최종
    방어선이다(1차 방어선은 프롬프트의 [상호참조 규칙]). agreement/concern은 최소 하나는
    실제 내용이 있어야 한다(둘 다 비어 있으면 상대 의견에 실질적으로 반응하지 않은 것)."""
    if _blank(raw.get("judgment")) or _blank(raw.get("reason")):
        return "missing_or_empty_field:judgment_or_reason"
    if len(raw.get("judgment", "")) > _MAX_JUDGMENT_CHARS:
        return "judgment_too_long"
    if len(raw.get("reason", "")) > _MAX_REASON_CHARS:
        return "reason_too_long"
    if len(raw.get("suggestion") or "") > _MAX_SUGGESTION_CHARS:
        return "suggestion_too_long"
    if _blank(raw.get("spoken_text")):
        return "missing_or_empty_field:spoken_text"
    if len(raw.get("spoken_text", "")) > _MAX_SPOKEN_TEXT_CHARS:
        return "spoken_text_too_long"
    if current_speaker_id:
        speaker_problem = validate_spoken_text_speaker_reference(
            current_speaker_id, responding_to_speaker_id, raw.get("spoken_text", "")
        )
        if speaker_problem:
            return speaker_problem
    if (
        require_user_question_focus
        and responding_to_speaker_id == "user"
        and responding_to_content
        and not _answers_user_question(raw.get("spoken_text", ""), responding_to_content)
    ):
        return "spoken_text_does_not_answer_user_question"
    generated_issue_id = (raw.get("active_issue_id") or "").strip()
    if expected_issue_id and generated_issue_id != expected_issue_id:
        return "active_issue_id_mismatch"
    if expected_issue_id and require_issue_content_focus:
        issue_problem = _spoken_text_issue_validation_reason(
            raw.get("spoken_text", ""), expected_issue_id, expected_issue_title
        )
        if issue_problem:
            return issue_problem
    if evidence_claim_types_by_ref is not None:
        has_target_evidence = "user_provided_fact" in evidence_claim_types_by_ref.values()
        claims = [claim for claim in (raw.get("claims") or []) if isinstance(claim, dict)]
        has_evidence_fact_claim = any(
            claim.get("claim_type") in ("document_fact", "user_provided_fact")
            for claim in claims
        )
        has_expert_judgment_claim = any(claim.get("claim_type") == "expert_judgment" for claim in claims)
        spoken_text = raw.get("spoken_text", "")
        if (
            has_evidence_fact_claim
            and not has_expert_judgment_claim
            and _has_evaluative_conclusion(spoken_text)
        ):
            # 평가표의 질문/기준(document_fact)은 "무엇을 평가하는지"만 증명한다. 그 근거만
            # 인용한 채 아이디어가 부족/미흡하다고 말하면 평가 기준을 판정 결과로 확대한
            # 것이므로, 해당 결론을 별도 expert_judgment claim으로 명시하게 재시도한다.
            return "spoken_evaluation_missing_expert_judgment_claim"
        for claim in claims:
            claim_type = claim.get("claim_type")
            refs = claim.get("evidence_refs")
            refs = refs if isinstance(refs, list) else []
            known_types = {
                evidence_claim_types_by_ref[ref]
                for ref in refs
                if isinstance(ref, str) and ref in evidence_claim_types_by_ref
            }
            if len(known_types) > 1:
                return "claim_mixes_document_roles"
            if known_types and claim_type not in known_types:
                return "claim_type_evidence_role_mismatch"
            if claim_type == "document_fact" and not refs:
                return "document_fact_missing_evidence"
            if claim_type == "user_provided_fact" and has_target_evidence and not refs:
                return "user_provided_fact_missing_target_evidence"
            if claim_type == "expert_judgment" and refs:
                return "expert_judgment_must_not_cite_evidence"
    if discussion_stage == "response" and _looks_like_near_verbatim_restatement(
        raw.get("spoken_text", ""), responding_to_content
    ):
        return "spoken_text_restates_responding_message"
    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 매 발언은 "현재 임시 결론"을
    # 반드시 담아야 한다(요청 4번). discussion_stage와 무관하게 항상 검증한다.
    if _blank(raw.get("interim_conclusion")):
        return "missing_or_empty_field:interim_conclusion"
    if len(raw.get("interim_conclusion", "")) > _MAX_INTERIM_CONCLUSION_CHARS:
        return "interim_conclusion_too_long"
    confirmed = raw.get("confirmed")
    if isinstance(confirmed, list) and len(confirmed) > _MAX_CONFIRMED_ITEMS:
        return "confirmed_too_many_items"
    unconfirmed = raw.get("unconfirmed")
    if isinstance(unconfirmed, list) and len(unconfirmed) > _MAX_UNCONFIRMED_ITEMS:
        return "unconfirmed_too_many_items"
    if discussion_stage == "response":
        if _blank(raw.get("responding_to")):
            return "missing_or_empty_field:responding_to"
        agreement = (raw.get("agreement") or "").strip()
        concern = (raw.get("concern") or "").strip()
        if not agreement and not concern:
            return "missing_or_empty_field:agreement_or_concern"
    # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — 반복 발언 방지의 코드 강제
    # 지점(요청 17번): new_information이 비어 있으면 재시도를 유발한다.
    if _blank(raw.get("active_issue_id")):
        return "missing_or_empty_field:active_issue_id"
    new_information = raw.get("new_information")
    if not isinstance(new_information, list) or not any(
        isinstance(v, str) and v.strip() for v in new_information
    ):
        return "missing_or_empty_field:new_information"
    if responding_to_content and all(
        isinstance(value, str) and value.strip() and value.strip() in responding_to_content
        for value in new_information
        if isinstance(value, str) and value.strip()
    ):
        return "new_information_only_repeats_responding_message"
    if raw.get("needs_user_input") and _blank(raw.get("user_question")):
        return "missing_or_empty_field:user_question"
    return None


def _validate_facilitator_response(raw: dict) -> str | None:
    if _blank(raw.get("facilitator_summary")):
        return "missing_or_empty_field:facilitator_summary"
    if bool(raw.get("needs_user_decision")) and _blank(raw.get("user_question")):
        return "missing_or_empty_field:user_question"
    if _blank(raw.get("spoken_text")):
        return "missing_or_empty_field:spoken_text"
    if len(raw.get("spoken_text", "")) > _MAX_FACILITATOR_SPOKEN_TEXT_CHARS:
        return "spoken_text_too_long"
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_message_id() -> str:
    return f"MSG-{uuid.uuid4().hex[:10]}"


def _speaker_fields(persona_id: str) -> tuple[str, str]:
    """speaker_name/role은 LLM 출력을 신뢰하지 않고 persona_cards.json에서 가져온다 —
    배치형 _normalize_turn()이 speaker_id를 항상 호출부 값으로 덮어쓰는 것과 같은 이유
    (LLM이 이름/역할을 지어내 화면 표시가 흔들리는 것을 막기 위함)."""
    card = get_persona_card(persona_id)
    return card.get("display_name", persona_id), card.get("role", "")


def _referenced_ids(raw_ids: Any, known_message_ids: set[str]) -> list[str]:
    """LLM이 반환한 referenced_message_ids 중 실제로 존재하는 message_id만 남긴다 —
    존재하지 않는 id를 그대로 프론트에 흘려보내면(오인용 또는 프롬프트 인젝션 시도) 프론트가
    참조 링크를 못 찾아 깨진다."""
    if not isinstance(raw_ids, list):
        return []
    return [mid for mid in raw_ids if isinstance(mid, str) and mid in known_message_ids]


def _build_message(
    *,
    persona_id: str,
    round_number: int,
    message_type: str,
    content: str,
    referenced_message_ids: list[str],
    evidence: Any,
    structured: dict | None = None,
    message_id: str | None = None,
    grounding: dict | None = None,
) -> ConvMessage:
    """structured는 선택 필드다(요청 9번 — 프런트 "상세 보기" 토글용, 기존 content 문자열은
    그대로 유지한 채 순수 추가). opinion/agreement/disagreement 메시지만 값을 채우고
    (make_conv_discussion_node 참고), 질문/답변/설명/요약 메시지는 None으로 둔다 — 기존
    content 렌더링만으로도 완전한 정보이기 때문이다.

    grounding은 ground_claims()의 결과다(요청: RAG 근거 실제 활용 강화) — evidence(위)는
    "이번 턴에 주입된 검색 결과 전체"라는 기존 의미를 그대로 유지하고, grounding이 주어지면
    그중 실제로 주장과 연결·검증된 부분만 별도 필드로 추가한다. grounding이 없는 메시지
    타입(진행자 정리 등 claims 개념이 없는 메시지)은 빈 값으로 채운다."""
    speaker_name, role = _speaker_fields(persona_id)
    return ConvMessage(
        message_id=message_id or _new_message_id(),
        speaker_id=persona_id,
        speaker_name=speaker_name,
        role=role,
        round=round_number,
        message_type=message_type,  # type: ignore[typeddict-item]
        content=content,
        referenced_message_ids=referenced_message_ids,
        evidence=evidence or [],
        created_at=_now_iso(),
        structured=structured,
        claims=list(grounding["claims"]) if grounding else [],
        linked_evidence_refs=list(grounding["linked_evidence_refs"]) if grounding else [],
        supported_claim_count=grounding["supported_claim_count"] if grounding else 0,
        unsupported_claim_count=grounding["unsupported_claim_count"] if grounding else 0,
        # 용준/Claude(2026-07-22, 요청: claim 통계 의미 분리) — 순수 추가 필드. 구버전
        # 클라이언트는 무시하면 그대로 동작한다.
        accepted_claim_count=grounding["accepted_claim_count"] if grounding else 0,
        grounded_claim_count=grounding["grounded_claim_count"] if grounding else 0,
        expert_judgment_count=grounding["expert_judgment_count"] if grounding else 0,
        missing_information=list(grounding["missing_information"]) if grounding else [],
        evidence_status=grounding["evidence_status"] if grounding else None,
        sufficiency=(
            "sufficient"
            if grounding and grounding["evidence_status"] == "grounded"
            else "partial"
            if grounding and grounding["evidence_status"] in ("partially_grounded", "expert_judgment_only")
            else "insufficient"
            if grounding
            else None
        ),
    )


def _as_string_list(items: Any) -> list[str]:
    """LLM이 배열 대신 문자열·null·객체 등 다른 타입을 반환해도(구조화 출력이 100% 강제되지
    않는 JSON 모드의 한계) 안전하게 문자열 리스트로 정규화한다. 리스트가 아니면 통째로
    버린다 — 예를 들어 confirmed가 문자열 하나로 오면 그 문자열의 각 글자를 항목으로 쪼개는
    사고(예: "for item in raw_string")를 방지한다."""
    if not isinstance(items, list):
        return []
    return [str(v) for v in items if v is not None]


def _bullets(items: Any) -> str:
    values = _as_string_list(items)
    if not values:
        return "- (없음)"
    return "\n".join(f"- {v}" for v in values)


def _compose_question_content(spoken_text: str) -> str:
    """용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환):
    화면에 실제로 보이는 content는 LLM이 만든 spoken_text 그대로다 — 예전에는 이 함수가
    judgment/question(및 후보 결합 시 user_selection_summary/proposal)을 [현재 판단]/
    [핵심 질문] 같은 보고서 헤더로 이어붙였지만, 그 구조는 채팅에서 보고서처럼 보인다는
    문제가 있었다. judgment/question/user_selection_summary/proposal은 여전히
    make_conv_question_node()가 message["structured"]에 그대로 저장한다 — 내부 상태와
    프롬프트 컨텍스트용으로는 계속 쓰이고, 화면에만 노출되지 않을 뿐이다."""
    return spoken_text.strip()


def _compose_discussion_content(spoken_text: str) -> str:
    """용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환): 화면에
    실제로 보이는 content는 LLM이 만든 spoken_text 그대로다 — 예전에는 이 함수가
    judgment/reason/suggestion/responding_to/agreement/concern/revision/interim_conclusion/
    confirmed/unconfirmed를 역할별 헤더([기획 관점]/[기술 검토]/... discussion_headers_for가
    정하던 라벨)로 이어붙였지만, 그 구조는 채팅에서 보고서처럼 보인다는 문제가 있었다. 이
    필드들은 여전히 message["structured"]에 그대로 저장된다(make_conv_discussion_node 참고)
    — 내부 상태·다음 판단 근거로는 계속 쓰이고 화면에만 노출되지 않을 뿐이다."""
    return spoken_text.strip()


def _discussion_round_snapshot_text(
    persona_id: str,
    judgment: str,
    reason: str,
    suggestion: str,
    confirmed: Any,
    unconfirmed: Any,
    interim_conclusion: str = "",
    responding_to: str | None = None,
    agreement: str | None = None,
    concern: str | None = None,
    revision: str | None = None,
) -> str:
    """discussion_rounds(라운드별 발언 스냅샷, API에는 노출되지만 현재 프런트 채팅 UI는 쓰지
    않는 내부 회의록 아카이브)에 저장할 상세 텍스트를 조립한다. 채팅 말풍선(content/
    spoken_text)과는 별개 관심사라 역할별 헤더 라벨(discussion_headers_for, 요청: 보고서형
    메시지 개편으로 제거됨) 없이 고정된 일반 라벨을 쓴다 — 이 텍스트는 라이브 스트리밍 대상이
    아니므로 DISCUSSION_STREAM_FIELDS와 일치할 필요가 없다."""
    parts = [f"[판단]\n{judgment}", f"[근거]\n{reason}", f"[제안]\n{suggestion}"]
    if responding_to:
        parts.append(f"[상대 의견 검토]\n{responding_to}")
    if agreement:
        parts.append(f"[동의]\n{agreement}")
    if concern:
        parts.append(f"[우려/제약]\n{concern}")
    if revision:
        parts.append(f"[수정 내용]\n{revision}")
    parts.append(f"[임시 결론]\n{interim_conclusion}")
    parts.append(f"[확정 사항]\n{_bullets(confirmed)}")
    parts.append(f"[미확정 사항]\n{_bullets(unconfirmed)}")
    return "\n\n".join(parts)


# 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — 사용자에게
# 실제로 보여줄 텍스트는 이제 각 스키마의 spoken_text 필드 하나뿐이다(예전에는 judgment/
# question/reason/suggestion 등 여러 필드를 순서대로 헤더와 함께 이어붙였지만, 그 결과가
# 채팅에서 보고서처럼 보인다는 문제가 있었다). backend/app/api/routes/
# ideation_conversation_streaming.py의 스트리밍 llm_call이 이 목록을 그대로 써서 OpenAI
# 델타 안에서 spoken_text 값만 실시간으로 흘려보낸다 — 헤더가 없으므로 모든 목록의 두 번째
# 값은 항상 None이다.
QUESTION_STREAM_FIELDS: tuple[tuple[str, str | None], ...] = (("spoken_text", None),)
DISCUSSION_STREAM_FIELDS: tuple[tuple[str, str | None], ...] = (("spoken_text", None),)
FACILITATOR_SUMMARY_STREAM_FIELDS: tuple[tuple[str, str | None], ...] = (("spoken_text", None),)
EXPERT_DELEGATION_STREAM_FIELDS: tuple[tuple[str, str | None], ...] = (("spoken_text", None),)
# make_expert_delegation_message가 spoken_text 뒤에 항상 고정으로 덧붙이는 문구 — 스트리밍
# 종료 직후 화면에도 동일하게 붙여야 canonical 메시지와 스트리밍 미리보기가 일치한다.
EXPERT_DELEGATION_TRAILER = "\n\n사용자가 나중에 다른 방향을 제시하면 이 가정은 언제든 수정할 수 있습니다."

# 용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장) — 담당 전문가의
# 임시 제안을 반대 역할 전문가가 검토하는 턴의 스트리밍 필드.
DELEGATION_REVIEW_STREAM_FIELDS: tuple[tuple[str, str | None], ...] = (("spoken_text", None),)
# 위임 흐름 전용 진행자 최종 권고안 — 사용자에게 되물을 필드가 스키마에 아예 없으므로
# (요청: "다시 사용자에게 같은 질문을 넘기면 안 됩니다") facilitator_summary와 달리 두 번째
# 스트림 필드가 없다.
DELEGATION_FACILITATOR_STREAM_FIELDS: tuple[tuple[str, str | None], ...] = (("spoken_text", None),)


# 용준/Claude(2026-07-23, 요청: RAG 근거 실제 활용 강화 — query 품질 개선) — 이전에는
# user_idea dict의 모든 값(제목/문제/대상 사용자/해결 방식/주요 기능/기술 접근/MVP 등)을
# 그대로 이어붙여 검색어로 썼다. 후보 필드가 많을수록 검색어가 길고 광범위해져, 지금 실제로
# 논의 중인 쟁점(active_issue, 예: "차별성")과 무관한 청크까지 상위권에 섞여 들어왔다 —
# 그 결과 검색 자체(raw/scoped/final target_count>0, criteria 검색됨)는 "성공"으로 보여도,
# 검색된 근거가 이번 쟁점과 실제로 관련이 없어 claim_grounding의 관련성 검사(ai.rag.
# evidence_linking.relevance.is_relevant_candidate)를 통과하지 못하거나, LLM 스스로
# "이번 쟁점과 무관하면 document_fact claim을 만들지 않는다"는 프롬프트 규칙(ideation_conv_
# discussion.txt [근거 인용 규칙])을 지켜 아예 인용을 시도하지 않는 결과로 이어졌다
# (linked_evidence_count=0, evidence_status="expert_judgment_only"). 이제 검색어를 (1) 아이디어
# 핵심 요약(제목/문제/해결 방식만, 필드 전부가 아니라), (2) 지금 실제로 다루는 쟁점,
# (3) 역할별 검토 관점, 세 부분으로 명시적으로 구분해 조합한다 — 이는 이번 발언의 실제
# 관심사에 검색을 집중시키기 위함이지, 단순히 키워드를 더 많이 넣는 것이 아니다.
_ROLE_QUERY_FOCUS: dict[str, str] = {
    "planning_expert": "사용자 가치와 문제 적합성, 공모전 심사 기준(혁신성·확장성 등), 기존 서비스 대비 차별성, 사업성·운영 구조",
    "dev_expert": "기술 구조와 구현 가능성, 데이터 수집·품질, 외부 시스템/API/센서 연동, 보안·성능·운영 위험, MVP 기술 범위",
}

# 용준/Claude(2026-07-23, 요청: 역할별 target query 필드 개선) — planning은 사용자 가치/차별화
# 관점의 필드가, dev는 기술 구현 관점의 필드가 검색에 더 적합하다. 이전에는 두 역할이 항상
# 같은 3개 필드(title/problem/solution)만 봤다 — planning에는 충분했지만 dev 검색에는 기술
# 정보(요구 데이터·기술 접근·MVP 범위·리스크)가 빠져 있었다. idea dict에 없는 필드는 그냥
# 건너뛴다(모든 후보가 이 필드를 다 채우지는 않는다).
_PLANNING_SUMMARY_FIELDS = ("title", "problem", "target_user", "solution", "differentiation", "core_value")
_DEV_SUMMARY_FIELDS = ("title", "solution", "main_features", "required_data", "technical_approach", "mvp_scope", "risks")
_IDEA_SUMMARY_FIELDS_BY_ROLE: dict[str, tuple[str, ...]] = {
    "planning_expert": _PLANNING_SUMMARY_FIELDS,
    "dev_expert": _DEV_SUMMARY_FIELDS,
}
_DEFAULT_IDEA_SUMMARY_FIELDS = ("title", "problem", "solution")
# 리스트 필드(main_features, risks 등)는 핵심 항목만 반영한다 — 후보에 기능이 10개 있어도
# 전부 이어붙이면 검색어가 다시 광범위해진다.
_MAX_SUMMARY_LIST_ITEMS = 3
# 검색어 전체(아이디어 요약 부분)가 지나치게 길어지지 않도록 문자 수 상한을 둔다.
_MAX_IDEA_SUMMARY_CHARS = 300

_ROLE_DEFAULT_RETRIEVAL_TOPIC: dict[str, str] = {
    "planning_expert": "차별성과 고객 가치",
    "dev_expert": "기술 구현 가능성",
}
# TOPIC_PRIORITY(질문 주제 slug) -> 검색어에 쓸 사람이 읽는 제목. 질문 턴(make_conv_question_
# node)의 question_topic과 같은 값 집합을 그대로 재사용한다 — 새 분류 체계를 만들지 않는다.
_TOPIC_ID_TITLES: dict[str, str] = {
    "problem": "문제 정의",
    "target_user": "목표 사용자",
    "core_value": "핵심 가치",
    "contest_fit": "공모전 적합성",
    "differentiation": "차별성과 고객 가치",
    "mvp": "MVP 범위",
    "data": "데이터 확보 방안",
    "ai_role": "AI 활용 방식",
    "roadmap": "확장 로드맵",
}


def _stringify_idea_field(value: object) -> str:
    """idea dict 값 하나를 검색어에 쓸 짧은 문자열로 정규화한다. 리스트는 앞의 핵심 항목
    몇 개만(_MAX_SUMMARY_LIST_ITEMS), 그 외 타입은 str()로 변환한다."""
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        return ", ".join(items[:_MAX_SUMMARY_LIST_ITEMS])
    if value is None:
        return ""
    return str(value).strip()


def _idea_core_summary(idea: object, persona_id: str | None = None) -> str:
    """user_idea/selected_idea dict에서 역할별 핵심 필드만 뽑아 짧게 요약한다(요청: 역할별
    target query 필드 개선 — planning/dev가 서로 다른 필드 집합을 본다). persona_id가
    없거나 매핑에 없으면 title/problem/solution만 쓰는 기존 기본값을 그대로 따른다. 값이
    빈 필드는 건너뛰고, 같은 문자열이 이미 포함돼 있으면 중복으로 넣지 않는다. 이 필드들이
    하나도 없는 구버전 idea(예: {"description": ...}만 있는 refinement 세션)는 있는 값
    전체를 이어붙인다. 결과가 너무 길면(_MAX_IDEA_SUMMARY_CHARS) 잘라낸다 — 의미 손실보다
    검색어 희석을 막는 것이 우선이다."""
    if isinstance(idea, dict):
        fields = _IDEA_SUMMARY_FIELDS_BY_ROLE.get(persona_id or "", _DEFAULT_IDEA_SUMMARY_FIELDS)
        seen: set[str] = set()
        core: list[str] = []
        for field in fields:
            text = _stringify_idea_field(idea.get(field))
            if text and text not in seen:
                seen.add(text)
                core.append(text)
        if core:
            summary = " / ".join(core)
        else:
            fallback: list[str] = []
            fallback_seen: set[str] = set()
            for value in idea.values():
                text = _stringify_idea_field(value)
                if text and text not in fallback_seen:
                    fallback_seen.add(text)
                    fallback.append(text)
            summary = " ".join(fallback)
    else:
        summary = str(idea or "")
    if len(summary) > _MAX_IDEA_SUMMARY_CHARS:
        summary = summary[:_MAX_IDEA_SUMMARY_CHARS].rstrip()
    return summary


def _active_issue_title(state: IdeationConvState) -> str | None:
    """지금 활성 쟁점(active_issue_id)의 사람이 읽는 제목을 open_issues/resolved_issues에서
    찾는다. 아직 쟁점 레코드가 만들어지지 않은 순간(예: 이번 발언에서 새로 여는 중)에는
    active_issue_id 자체(slug)를 대신 쓴다 — 검색어가 비게 두지 않기 위함이다."""
    issue_id = state.get("active_issue_id")
    if not issue_id:
        return None
    for issue in list(state.get("open_issues") or []) + list(state.get("resolved_issues") or []):
        if issue.get("issue_id") == issue_id:
            return issue.get("title") or issue_id
    return issue_id


def _slugify_issue_title(title: str) -> str:
    """issue title을 결정적 slug로 정규화한다(요청: Phase 1 shadow planner의 issue_id는
    불안정한 Python hash()를 쓰지 않는다) — 텍스트 자체가 같으면 항상 같은 slug가 나온다."""
    normalized = re.sub(r"[^0-9A-Za-z가-힣]+", "_", (title or "").strip()).strip("_")
    return normalized.lower() or "unknown_issue"


def _active_user_interjection(state: IdeationConvState) -> ConvMessage | None:
    """지정 위원 응답/상대 검토가 끝나기 전까지 가장 최근 사용자 개입을 반환한다."""
    interjection_in_progress = bool(
        state.get("interjection_target_speaker_id")
        or state.get("required_counterpart_speaker_id")
    )
    if not interjection_in_progress:
        return None
    for message in reversed(state.get("messages") or []):
        if message.get("speaker_id") == "user" and message.get("message_type") == "interjection":
            content = (message.get("content") or "").strip()
            if content:
                return message
    return None


def resolve_effective_issue(state: IdeationConvState, persona_id: str | None = None) -> dict[str, str]:
    """용준/Claude(2026-07-23, Phase 1 "Shadow Deterministic Evidence Planner"): 이번 턴
    retrieval이 실제로 초점을 맞추는 쟁점을 issue_id/title 구조로 반환한다.
    resolve_retrieval_issue()와 정확히 같은 우선순위를 따르며(아래에서 그 함수가 이 함수의
    title만 재사용하도록 리팩터링했다 — 요청: "_topic_query()가 사용한 issue title/query와
    Planner의 issue가 반드시 동일") 반환하는 title 문자열은 항상 같다."""
    user_interjection = _active_user_interjection(state)
    if user_interjection is not None:
        question = user_interjection["content"].strip()
        issue_id = state.get("active_issue_id") or _slugify_issue_title(question)
        return {
            "issue_id": issue_id,
            "title": question,
            "query": question,
            "source": "user_interjection",
        }

    issue_id = state.get("active_issue_id")
    if issue_id:
        title = _active_issue_title(state) or issue_id
        return {"issue_id": issue_id, "title": title, "source": "active_issue_id"}

    if state.get("unresolved_issues"):
        title = " ".join(state["unresolved_issues"])
        return {"issue_id": _slugify_issue_title(title), "title": title, "source": "unresolved_issues"}

    resolved_topic_ids = list(state.get("resolved_topics") or [])
    resolved_issue_titles = {
        issue.get("title") for issue in (state.get("resolved_issues") or []) if issue.get("title")
    }
    for topic_id in remaining_topics_for(resolved_topic_ids):
        title = _TOPIC_ID_TITLES.get(topic_id, topic_id)
        if title not in resolved_issue_titles:
            return {"issue_id": topic_id, "title": title, "source": "topic_priority"}

    title = _ROLE_DEFAULT_RETRIEVAL_TOPIC.get(persona_id or "", "핵심 검토 사항")
    return {"issue_id": _slugify_issue_title(title), "title": title, "source": "role_default"}


def resolve_retrieval_issue(state: IdeationConvState, persona_id: str | None = None) -> str:
    """이번 턴 검색이 초점을 맞출 "지금 검토 중인 쟁점"의 제목을 다음 우선순위로 정한다(요청:
    첫 전문가 턴에도 실제 검토 쟁점 반영 — active_issue_id는 discussion 노드가 첫 발언을 마친
    "뒤"에야 열리므로, planning의 첫 발언 시점에는 항상 None이다. 이전에는 그 경우 idea 요약과
    role focus만 남아 검색어가 여전히 광범위했다):
    1) active_issue_id가 있으면 그 제목(_active_issue_title).
    2) 없으면 진행자가 이미 남긴 미해결 쟁점(unresolved_issues, discussion_facilitator가
       "unconfirmed"에서 채운다 — 있으면 TOPIC_PRIORITY보다 더 구체적인 최신 신호다), 그것도
       없으면 아직 해결되지 않은 TOPIC_PRIORITY 주제 중 우선순위가 가장 높은 것(question_topic
       체계를 그대로 재사용 — 이미 resolved_topics/resolved_issues에 있는 주제·제목은
       건너뛴다).
    3) 그것도 없으면(모든 주제가 이미 해결됐거나 두 목록이 모두 비어 있으면) 역할별 기본
       검토 주제로 폴백한다 — 검색어가 아이디어 요약뿐인 채로 비어 있는 상태를 만들지
       않는다.

    실제 title 계산은 resolve_effective_issue()에 위임한다(요청: retrieval과 Phase 1 shadow
    planner가 반드시 같은 title/query를 봐야 한다)."""
    return resolve_effective_issue(state, persona_id)["title"]


def _topic_query(state: IdeationConvState, persona_id: str | None = None) -> str:
    """이번 턴의 근거 검색어를 조립한다. persona_id를 넘기면 역할별 아이디어 요약 필드와
    검토 관점이 함께 반영돼 planning_expert/dev_expert가 서로 다른 검색어를 받는다(요청:
    역할별 검색 결과 차별화) — persona_id가 없으면(진행자 등 role_id가 없는 호출자) 이전과
    동일하게 기본 필드 요약 + 이슈만 반환한다."""
    parts: list[str] = []
    effective_issue = resolve_effective_issue(state, persona_id)
    if effective_issue.get("source") == "user_interjection":
        parts.append(f"사용자 직접 질문: {effective_issue['query']}")

    idea_summary = _idea_core_summary(state["user_idea"], persona_id)
    if idea_summary:
        parts.append(idea_summary)

    issue_title = effective_issue["title"]
    if issue_title:
        parts.append(f"현재 쟁점: {issue_title}")

    role_focus = _ROLE_QUERY_FOCUS.get(persona_id or "")
    if role_focus:
        parts.append(f"검토 관점: {role_focus}")

    if not parts:
        return idea_summary
    return " | ".join(parts)


def _runtime_scope_for(state: IdeationConvState) -> dict[str, Any]:
    """용준/Claude(2026-07-23, 요청: stale closure 수정) — evidence_lookup을 실제로 호출하는
    이 순간의 최신 state에서 session_id/selected_idea_document_id를 다시 읽는다. backend가
    만드는 evidence_lookup closure(ideation_conversation_preview.py::_evidence_lookup_for)는
    /reply 요청이 시작될 때(previous_state, candidate_selection 실행 전)의 값을 캡처하므로,
    같은 요청 안에서 후보가 방금 선택/변경돼도 그 closure 값은 갱신되지 않는다 — 실측
    확인된 버그(target upsert는 성공하지만 같은 요청의 다음 검색이 여전히
    selected_candidate_document_id=None으로 진행됨)의 원인이다. 노드가 검색을 호출하는
    이 지점에서는 candidate_selection 노드가 이미 state["selected_idea_document_id"]를
    갱신한 뒤이므로(그래프는 노드를 순차 실행한다), 여기서 다시 읽으면 항상 최신 값이다."""
    return {
        "session_id": state.get("session_id"),
        "selected_candidate_document_id": state.get("selected_idea_document_id"),
    }


def _last_user_answer(messages: list[ConvMessage]) -> dict | None:
    for msg in reversed(messages):
        if msg.get("message_type") == "answer":
            return msg
    return None


def conversation_context_for(state: IdeationConvState) -> dict[str, Any]:
    """질문/의견/재질문 판정 프롬프트가 공통으로 쓰는 대화 맥락 요약. ideation_conv_run.py의
    answer_sufficiency 게이트도 이 함수를 그대로 재사용한다(맥락 조립 로직 중복 방지)."""
    messages = state["messages"]
    return {
        "round": state["round"],
        "recent_messages": messages[-_RECENT_MESSAGES_LIMIT:],
        "last_user_answer": _last_user_answer(messages),
        "consensus_so_far": state["consensus"],
        "unresolved_issues": state["unresolved_issues"],
    }


_DISCUSSION_CONTEXT_MESSAGE_KEYS = (
    "message_id",
    "speaker_id",
    "speaker_name",
    "role",
    "message_type",
    "content",
    "responding_to",
    "referenced_message_ids",
    "created_at",
)


def _isolate_discussion_evidence_context(context: dict[str, Any]) -> dict[str, Any]:
    """Active evidence 턴의 현재 ref와 과거 메시지의 ref/chunk namespace를 격리한다.

    ConvMessage 전체에는 UI/세션 복원을 위한 evidence, claims, linked_evidence_refs와
    structured grounding 필드가 들어 있다. 이를 다음 discussion prompt에 그대로 직렬화하면
    LLM이 현재 턴 근거 대신 과거 턴의 E번호를 재사용할 수 있다. 대화 의미를 유지하는 필드만
    allow-list로 복사하며 원본 state/message는 수정하지 않는다.
    """

    def public_message(message: Any) -> Any:
        if not isinstance(message, dict):
            return message
        return {
            key: message[key]
            for key in _DISCUSSION_CONTEXT_MESSAGE_KEYS
            if key in message and message[key] is not None
        }

    isolated = dict(context)
    isolated["recent_messages"] = [public_message(item) for item in context.get("recent_messages") or []]
    last_user_answer = context.get("last_user_answer")
    isolated["last_user_answer"] = public_message(last_user_answer) if last_user_answer is not None else None
    return isolated


def _selection_context_for(state: IdeationConvState) -> dict[str, Any]:
    """용준/Claude(2026-07-21, 후보 결합 컨텍스트 보존): discovery 모드에서 후보를 선택/
    결합/추천한 직후부터 refinement 질문 프롬프트가 참조할 구조화된 선택 컨텍스트를
    만든다(요청 9번 — conversation_context의 최근 메시지에 우연히 포함되기를 기대하지
    않고 별도로 명시 전달). refinement로 시작했거나 아직 후보를 선택하지 않은 세션은
    selection_intent가 None이므로 빈 dict를 반환한다 — 템플릿은 빈 dict를 "선택 컨텍스트
    없음"으로 처리한다."""
    selection_intent = state.get("selection_intent")
    if not selection_intent:
        return {}
    return {
        "selection_intent": selection_intent,
        "user_selection_message": state.get("user_selection_message"),
        "source_candidates": state.get("source_candidates") or [],
        "merge_analysis": state.get("merge_analysis"),
        "selected_idea": state.get("selected_idea"),
    }


def _is_first_question_after_combine(state: IdeationConvState) -> bool:
    """이번 질문 노드 호출이 "후보 결합 직후 첫 전문가 메시지"인지 판별한다(요청 6번).
    candidate_selection 노드가 결합을 확정하면 항상 speaker_id="ideation_facilitator",
    message_type="summary"인 요약 메시지를 마지막에 추가한 뒤(ideation_conv_discovery.py::
    _resolve_selection) 같은 그래프 호출 안에서 곧바로 planning_question 노드로 이어지므로
    (ideation_conv_build.py::_route_after_candidate_selection), 이 시점의 "마지막 메시지"가
    그 요약 메시지인지만 보면 정확히 이 순간만 골라낼 수 있다 — 이후 질문(2번째 라운드 등)은
    마지막 메시지가 다른 종류이므로 False가 된다."""
    if state.get("selection_intent") != "combine":
        return False
    messages = state["messages"]
    if not messages:
        return False
    last = messages[-1]
    return last.get("speaker_id") == "ideation_facilitator" and last.get("message_type") == "summary"


def judge_answer_sufficiency(
    llm_call: LLMCall,
    persona_id: str,
    pending_question: str,
    user_answer: str,
    retry_count: int,
    conversation_context: dict[str, Any],
    expected_answer_type: str | None = None,
    user_idea: Any = None,
    idea_candidates: Any = None,
) -> dict[str, Any]:
    """사용자가 pending_question에 답한 직후, 이 메시지가 답변인지/설명 요청인지/불충분한
    답변인지 3분류로 판정한다(요청 3번 + 용어 설명 요청 오판 방지). ideation_conv_run.py::
    reply_ideation_conversation가 apply_user_answer로 phase를 넘기기 전에 호출한다 —
    그래프/노드 구조(질문·의견 노드, 엣지)는 건드리지 않고 별도의 짧은 판정 호출 하나만
    앞에 끼워 넣는 방식이라, 기존 라운드 진행 로직과 완전히 분리돼 있다.

    expected_answer_type(질문 노드가 만든 이번 질문의 기대 답변 유형 — 예: "preference",
    "selection")을 함께 넘긴다. 이 판정은 "방금 pending_question이 요구한 것에 답했는가"만
    보는 것이지 "아이디어 전체가 충분히 구체적인가"를 보는 게 아니다(요청: 답변 충분성과
    아이디어 완성도 분리) — expected_answer_type이 preference/selection이면 하나를 명확히
    선택한 것만으로 충분하다고 판정하도록 프롬프트가 요구 수준을 낮춘다. None이면(질문
    노드가 값을 만들지 못했거나 구버전 응답) 기존의 일반 기준으로만 판정한다.

    user_idea/idea_candidates는 clarification_request일 때 "현재 후보와 대화 맥락에 맞는"
    선택지를 만들기 위한 근거 자료다(요청 사항 예시처럼 "두 후보를 고려하면 생활비 절감과
    안전 향상이 주요 선택지입니다"처럼 실제 아이디어에 근거한 예시를 만들려면 필요하다).

    반환값의 answer_type이 세 값 중 하나다("answer"/"clarification_request"/
    "insufficient_answer"). 하위 호환을 위해 is_sufficient(answer_type=="answer")도 함께
    돌려준다 — LLM이 구버전 스키마(is_sufficient 불리언만)를 반환해도 answer_type이 없으면
    그 값으로 answer/insufficient_answer를 유추한다(clarification_request는 유추할 수 없으므로
    이 경우 절대 선택되지 않는다 — 구버전 스키마는 애초에 그 개념을 모른다).

    판정 호출이 실패(파싱 실패 등)하면 안전하게 "충분함"으로 fail-open한다 — 이 판정은
    부가적인 품질 게이트일 뿐 회의의 핵심 콘텐츠 생성이 아니므로, 판정 자체의 인프라
    오류로 사용자가 답변을 진행하지 못하게 막는 것은 과도하다(질문/의견/종합 노드처럼
    phase="failed"로 보내지 않는다)."""
    prompt = build_ideation_conv_sufficiency_prompt(
        persona_id,
        pending_question,
        user_answer,
        retry_count,
        conversation_context,
        expected_answer_type,
        user_idea,
        idea_candidates,
    )
    raw, ok = _safe_call_json(llm_call, prompt)
    if not ok or raw is None:
        return {
            "answer_type": "answer",
            "is_sufficient": True,
            "reason": "판정 응답 파싱 실패로 자동 통과",
            "follow_up_question": None,
            "clarification_response": None,
        }

    answer_type = raw.get("answer_type")
    if answer_type not in _VALID_ANSWER_TYPES:
        # 하위 호환 — 구버전 스키마(is_sufficient 불리언만)로 응답한 경우 유추한다.
        answer_type = "answer" if bool(raw.get("is_sufficient", True)) else "insufficient_answer"

    return {
        "answer_type": answer_type,
        "is_sufficient": answer_type == "answer",
        "reason": raw.get("reason", ""),
        "follow_up_question": raw.get("follow_up_question") or None,
        "clarification_response": raw.get("clarification_response") or None,
    }


def make_follow_up_message(*, persona_id: str, round_number: int, reason: str, follow_up_question: str) -> ConvMessage:
    """answer_sufficiency 판정이 불충분으로 나왔을 때 사용자에게 다시 보여줄 재질문 메시지를
    만든다. _build_message를 그대로 재사용해 message_id/speaker_name/role 등 필드 규칙을
    질문/의견 메시지와 통일한다.

    용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환): 예전에는
    reason/follow_up_question을 [재질문]/[핵심 질문] 헤더로 나눠 보여줬지만, 이 두 문장은
    이미 자연스럽게 이어지는 문장이라(judge_answer_sufficiency의 reason은 "왜 다시 묻는지",
    follow_up_question은 "무엇을 묻는지") 헤더 없이 한 문단으로 이어붙인다."""
    content = f"{reason}\n\n{follow_up_question}".strip()
    return _build_message(
        persona_id=persona_id,
        round_number=round_number,
        message_type="question",
        content=content,
        referenced_message_ids=[],
        evidence=[],
    )


def make_clarification_message(*, persona_id: str, round_number: int, clarification_response: str) -> ConvMessage:
    """사용자가 용어 설명/예시/선택지를 요청했을 때(answer_type="clarification_request")
    보여줄 메시지를 만든다. clarification_response는 이미 "설명 + 3~5개 선택지 + 다시 던지는
    핵심 질문"을 한 덩어리로 담고 있다(ideation_conv_sufficiency.txt::[명확화 응답 작성
    규칙] 참고) — 이 함수는 그 값을 메시지 스키마로 감싸기만 한다.

    용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환): [설명] 라벨을
    더 이상 붙이지 않는다 — clarification_response 자체가 이미 완결된 자연스러운 응답이다."""
    content = clarification_response.strip()
    return _build_message(
        persona_id=persona_id,
        round_number=round_number,
        message_type="question",
        content=content,
        referenced_message_ids=[],
        evidence=[],
    )


def _validate_expert_delegation_response(raw: dict, stage: str = "initial") -> str | None:
    if _blank(raw.get("proposal")) or _blank(raw.get("reason")) or _blank(raw.get("assumption")):
        return "missing_or_empty_field:proposal_or_reason_or_assumption"
    if _blank(raw.get("spoken_text")):
        return "missing_or_empty_field:spoken_text"
    if len(raw.get("spoken_text", "")) > _MAX_SPOKEN_TEXT_CHARS:
        return "spoken_text_too_long"
    if stage == "revision":
        if _blank(raw.get("responding_to")) or _blank(raw.get("revision")):
            return "missing_or_empty_field:responding_to_or_revision"
    return None


def generate_expert_delegation_proposal(
    llm_call: LLMCall,
    persona_id: str,
    pending_question: str,
    notice_and_criteria: Any,
    user_idea: Any,
    retrieved_evidence: Any,
    conversation_context: dict[str, Any],
    stage: str = "initial",
    counterpart_review: Any = None,
) -> tuple[dict | None, bool, int]:
    """용준/Claude(2026-07-21, 요청: "모르겠다" UX 개선): answer_type="expert_delegation"일
    때(사용자가 pending_question에 답하는 대신 전문가 판단에 맡겼을 때) 담당 페르소나가
    자신의 평가 범위 안에서 합리적인 방향을 "임시 가정"으로 제안하게 만든다. 질문/의견
    노드와 같은 구조화 검증·재시도 정책(_safe_call_structured_json, 최대 1회 재시도)을
    그대로 재사용한다 — 실패 시(재시도 후에도 무효) 호출부가 다른 콘텐츠 생성 노드와
    동일하게 phase="failed"로 처리한다.

    stage/counterpart_review(용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간
    상호 검토로 확장): stage="revision"이면 상대 전문가의 검토(counterpart_review)를
    반영해 원래 제안을 수정하거나 유지한다 — responding_to/revision이 비어 있으면 재시도를
    유발한다(빈 상호참조 방지, discussion 노드와 동일한 정책)."""
    prompt = build_ideation_conv_expert_delegation_prompt(
        persona_id,
        notice_and_criteria,
        user_idea,
        retrieved_evidence,
        conversation_context,
        pending_question,
        stage=stage,
        counterpart_review=counterpart_review,
    )
    validate = lambda raw, _stage=stage: _validate_expert_delegation_response(raw, _stage)  # noqa: E731
    return _safe_call_structured_json(llm_call, prompt, validate, f"expert_delegation__{persona_id}")


def _validate_expert_delegation_review_response(raw: dict) -> str | None:
    if _blank(raw.get("judgment")) or _blank(raw.get("reason")) or _blank(raw.get("responding_to")):
        return "missing_or_empty_field:judgment_or_reason_or_responding_to"
    if _blank(raw.get("recommendation")):
        return "missing_or_empty_field:recommendation"
    if _blank(raw.get("spoken_text")):
        return "missing_or_empty_field:spoken_text"
    if len(raw.get("spoken_text", "")) > _MAX_SPOKEN_TEXT_CHARS:
        return "spoken_text_too_long"
    agreement = (raw.get("agreement") or "").strip()
    concern = (raw.get("concern") or "").strip()
    if not agreement and not concern:
        return "missing_or_empty_field:agreement_or_concern"
    return None


def generate_expert_delegation_review(
    llm_call: LLMCall,
    counterpart_persona_id: str,
    pending_question: str,
    notice_and_criteria: Any,
    user_idea: Any,
    retrieved_evidence: Any,
    conversation_context: dict[str, Any],
    proposal_under_review: Any,
) -> tuple[dict | None, bool, int]:
    """용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장): 담당
    전문가의 임시 제안을 반대 역할 전문가(counterpart_persona_id)가 자신의 평가 범위에서
    검토하게 만든다. proposal_under_review는 generate_expert_delegation_proposal()의 raw
    반환값(dict)을 그대로 넘긴다."""
    prompt = build_ideation_conv_expert_delegation_review_prompt(
        counterpart_persona_id,
        notice_and_criteria,
        user_idea,
        retrieved_evidence,
        conversation_context,
        pending_question,
        proposal_under_review,
    )
    return _safe_call_structured_json(
        llm_call, prompt, _validate_expert_delegation_review_response, f"expert_delegation_review__{counterpart_persona_id}"
    )


def _validate_expert_delegation_facilitator_response(raw: dict) -> str | None:
    if _blank(raw.get("final_recommendation")):
        return "missing_or_empty_field:final_recommendation"
    if _blank(raw.get("spoken_text")):
        return "missing_or_empty_field:spoken_text"
    if len(raw.get("spoken_text", "")) > _MAX_FACILITATOR_SPOKEN_TEXT_CHARS:
        return "spoken_text_too_long"
    return None


def generate_expert_delegation_facilitator_recommendation(
    llm_call: LLMCall,
    notice_and_criteria: Any,
    pending_question: str,
    proposal: Any,
    review: Any,
    revision: Any,
) -> tuple[dict | None, bool, int]:
    """용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장): 제안 ->
    검토 -> (있다면) 수정까지 끝난 뒤 진행자가 최종 권고안 하나로 정리하게 만든다. 출력
    스키마에 사용자 재질문 필드가 없어 구조적으로 같은 질문을 반복할 수 없다."""
    prompt = build_ideation_conv_expert_delegation_facilitator_prompt(
        notice_and_criteria, pending_question, proposal, review, revision
    )
    return _safe_call_structured_json(
        llm_call, prompt, _validate_expert_delegation_facilitator_response, "expert_delegation_facilitator"
    )


def _compose_expert_delegation_content(spoken_text: str) -> str:
    """용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환): 화면에
    실제로 보이는 content는 spoken_text에 고정 안내 문구(이 제안이 임시 가정이라는 사실,
    요청 10번)를 이어붙인 것이다 — 예전에는 proposal/reason/assumption을 [○○ 제안]/[제안
    이유]/[임시 가정] 헤더로 이어붙였지만, 그 구조는 채팅에서 보고서처럼 보인다는 문제가
    있었다. 안내 문구는 LLM이 빠뜨려도 항상 보장되도록 여기서 고정으로 덧붙인다(요청
    10번 그대로 유지) — 스트리밍 시 EXPERT_DELEGATION_TRAILER가 spoken_text 델타가 닫힌
    직후 같은 문구를 흘려보낸다."""
    return f"{spoken_text.strip()}\n\n사용자가 나중에 다른 방향을 제시하면 이 가정은 언제든 수정할 수 있습니다."


def make_expert_delegation_message(
    *,
    persona_id: str,
    round_number: int,
    spoken_text: str,
    proposal: str,
    reason: str,
    assumption: str,
    referenced_message_ids: Any,
    evidence: Any,
    known_message_ids: set[str],
    responding_to: str | None = None,
    revision: str | None = None,
) -> ConvMessage:
    content = _compose_expert_delegation_content(spoken_text)
    return _build_message(
        persona_id=persona_id,
        round_number=round_number,
        message_type="opinion",
        content=content,
        referenced_message_ids=_referenced_ids(referenced_message_ids, known_message_ids),
        evidence=evidence,
        structured={
            "proposal": proposal,
            "reason": reason,
            "assumption": assumption,
            "responding_to": responding_to,
            "revision": revision,
        },
    )


def _compose_expert_delegation_review_content(spoken_text: str) -> str:
    """용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환): 화면에
    실제로 보이는 content는 spoken_text 그대로다 — 예전에는 judgment/reason/responding_to/
    agreement/concern/recommendation을 [검토]/[근거]/[제안 검토]/... 헤더로 이어붙였지만,
    그 필드들은 여전히 structured에 저장되어 내부 상태로 쓰인다."""
    return spoken_text.strip()


def make_expert_delegation_review_message(
    *,
    persona_id: str,
    round_number: int,
    raw: dict,
    known_message_ids: set[str],
    evidence: Any = None,
) -> ConvMessage:
    stance = raw.get("stance")
    if stance not in _VALID_DISCUSSION_STANCES:
        stance = "보완"
    message_type = _STANCE_TO_MESSAGE_TYPE.get(stance, "opinion")
    judgment = raw.get("judgment", "")
    reason = raw.get("reason", "")
    responding_to = raw.get("responding_to", "")
    agreement = raw.get("agreement") or ""
    concern = raw.get("concern") or ""
    recommendation = raw.get("recommendation", "")
    content = _compose_expert_delegation_review_content(raw.get("spoken_text", ""))
    return _build_message(
        persona_id=persona_id,
        round_number=round_number,
        message_type=message_type,
        content=content,
        referenced_message_ids=_referenced_ids(raw.get("referenced_message_ids"), known_message_ids),
        # 용준/Claude(2026-07-22, RAG 근거 유실 수정): raw.get("evidence") 대신 호출부가
        # 실제 RAG 검색으로 얻어 프롬프트에 주입한 근거(evidence 인자)를 저장한다.
        evidence=evidence,
        structured={
            "judgment": judgment,
            "reason": reason,
            "responding_to": responding_to,
            "agreement": agreement,
            "concern": concern,
            "recommendation": recommendation,
            "stance": stance,
        },
    )


def make_expert_delegation_facilitator_message(*, round_number: int, raw: dict) -> ConvMessage:
    """요청: "다시 사용자에게 같은 질문을 넘기면 안 됩니다" — 이 메시지는 질문 필드를 아예
    갖지 않는 스키마(generate_expert_delegation_facilitator_recommendation)의 결과만 담으므로
    구조적으로 재질문이 불가능하다.

    용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환): 화면에 실제로
    보이는 content는 spoken_text(1~2문장) 그대로다 — final_recommendation(결정 어조의 상세
    권고)과 considerations([참고 사항] 목록)는 더 이상 content에 붙지 않고 structured에만
    남는다."""
    final_recommendation = raw.get("final_recommendation", "")
    considerations = _as_string_list(raw.get("considerations"))
    content = raw.get("spoken_text", "").strip()
    return _build_message(
        persona_id="ideation_facilitator",
        round_number=round_number,
        message_type="summary",
        content=content,
        referenced_message_ids=[],
        evidence=[],
        structured={
            "agreements": _as_string_list(raw.get("agreements")),
            "considerations": considerations,
            "final_recommendation": final_recommendation,
        },
    )


def make_conv_question_node(
    persona_id: str,
    awaiting_phase: str,
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
    ground_claims: ClaimGroundingFn | None = None,
) -> Callable[[IdeationConvState], dict]:
    """기획 전문가(planning_expert)/개발 전문가(dev_expert)의 "질문 턴" 노드를 만든다.
    질문 하나를 만들고 나면 반드시 awaiting_phase로 멈춘다 — 이 노드 자신은 절대
    다음 전문가로 이어가지 않는다(요청 4번: 기획 질문 직후, 개발 질문 직후 각각 정지)."""

    def node(state: IdeationConvState) -> dict:
        query = _topic_query(state, persona_id)
        runtime_scope = _runtime_scope_for(state)
        trace_event(
            "IDEATION_EVIDENCE_LOOKUP_SCOPE",
            session_id=runtime_scope["session_id"],
            speaker=persona_id,
            selected_candidate_document_id=runtime_scope["selected_candidate_document_id"],
            selected_candidate_document_id_source="runtime_graph_state",
        )
        retrieved = call_evidence_lookup(evidence_lookup, persona_id, query, runtime_scope=runtime_scope)
        context = conversation_context_for(state)
        # 용준/Claude(2026-07-21, 질문 주제 구조화): resolved_topics는 state["resolved_topics"]
        # 그대로 읽는다(요청: 답변이 "answer"로 판정되어 다음 단계로 넘어갈 때만 추가되므로,
        # 후보 데이터에 problem/target_user 등이 있어도 여기 없으면 아직 미확정이다 —
        # discovery 후보는 전문가 초안일 뿐 사용자 확정이 아니라는 요청 4번 원칙이 바로
        # 이 지점에서 지켜진다: 후보 내용은 프롬프트의 user_idea/idea_candidates 컨텍스트로만
        # 전달되고, resolved_topics 판단에는 전혀 영향을 주지 않는다).
        resolved_topics = list(state.get("resolved_topics") or [])
        remaining_topics = remaining_topics_for(resolved_topics)
        roadmap_allowed = "roadmap" in remaining_topics
        selection_context = _selection_context_for(state)
        require_combine_structure = _is_first_question_after_combine(state)
        prompt = build_ideation_conv_question_prompt(
            persona_id,
            state["notice_and_criteria"],
            state["user_idea"],
            retrieved,
            context,
            resolved_topics=resolved_topics,
            remaining_topics=remaining_topics,
            roadmap_allowed=roadmap_allowed,
            selection_context=selection_context,
            require_combine_structure=require_combine_structure,
        )
        validate = _make_validate_question_response(resolved_topics, roadmap_allowed, require_combine_structure)
        raw, ok, attempts = _safe_call_structured_json(llm_call, prompt, validate, f"question__{persona_id}")
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"phase": "failed", "failed_node": f"question__{persona_id}", "llm_calls_used": used}

        raw, grounding, used = _ground_and_finalize_claims(
            persona_id=persona_id,
            raw=raw,
            retrieved=retrieved,
            prompt=prompt,
            llm_call=llm_call,
            validate=validate,
            used=used,
            ground_claims_fn=ground_claims,
        )

        judgment = raw.get("judgment", "")
        question = raw.get("question", "")
        question_topic = raw.get("question_topic")  # validate()가 이미 TOPIC_PRIORITY 소속을 보장한다.
        known_ids = {m["message_id"] for m in state["messages"]}
        message = _build_message(
            persona_id=persona_id,
            round_number=state["round"],
            message_type="question",
            content=_compose_question_content(raw.get("spoken_text", "")),
            referenced_message_ids=_referenced_ids(raw.get("referenced_message_ids"), known_ids),
            # 용준/Claude(2026-07-22, RAG 근거 유실 수정): 예전에는 raw.get("evidence")로 LLM이
            # JSON 응답에 자발적으로 되돌려준 evidence만 저장했다 — 이 필드는 검증되지 않는
            # 선택 필드라 대부분의 응답이 비워 보냈고, 그 결과 retrieved(위에서 RAG 검색으로
            # 실제로 찾아 프롬프트에 주입한 근거)가 있어도 메시지·로그(IDEATION_TURN_END의
            # evidence_count)에는 항상 0으로 남았다. 프롬프트에 실제로 삽입한 근거(retrieved)를
            # 그대로 메시지 evidence로 저장해야 "삽입된 근거 수"가 사실과 일치한다.
            evidence=retrieved,
            grounding=grounding,
            # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) —
            # judgment/question/user_selection_summary/proposal은 더 이상 content에 헤더로
            # 노출되지 않으므로, 내부 상태·다음 턴 프롬프트 컨텍스트로 계속 쓸 수 있도록
            # structured에 옮겨 담는다(순수 추가 — content가 대체됐을 뿐 정보는 유지된다).
            structured={
                "judgment": judgment,
                "question": question,
                "question_topic": question_topic,
                "user_selection_summary": raw.get("user_selection_summary"),
                "proposal": raw.get("proposal"),
            },
        )
        # expected_answer_type은 선택 필드다 — LLM이 만들지 않거나(구버전 프롬프트 응답 등)
        # 허용값 밖의 값을 반환해도 질문 생성 자체를 실패시키지 않고 None으로 저장한다
        # (sufficiency 판정이 기존의 일반 기준으로 대체 판정한다).
        expected_answer_type = raw.get("expected_answer_type")
        if expected_answer_type not in _VALID_EXPECTED_ANSWER_TYPES:
            expected_answer_type = None
        return {
            "messages": [message],
            "phase": awaiting_phase,
            # 재질문 판정(judge_answer_sufficiency)이 참조할 "질문 그 자체"만 저장한다
            # (judgment 설명까지 합친 표시용 content 전체가 아니라).
            "pending_question": question or message["content"],
            "pending_expected_answer_type": expected_answer_type,
            "pending_question_topic": question_topic,
            "llm_calls_used": used,
        }

    return node


# 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): 고정 그래프 시퀀스(기획 1회 → 개발
# 1회 → [조건부 수정] → 진행자 정리)를 쟁점 단위 동적 라우팅으로 교체하면서 REVISION_TRIGGER_
# STANCES(수정 턴 실행 여부를 stance로 게이팅하던 옛 메커니즘)는 더 이상 그래프 라우팅에
# 쓰이지 않는다 — 다음 발언자는 이제 항상 recommended_next_speaker(코드가 검증)로 정해진다.
# 이름은 하위 호환을 위해 유지한다(다른 모듈이 import할 수 있음 — grep으로 미사용 확인 필요).
REVISION_TRIGGER_STANCES = {"반박", "조건부_동의", "대안_제시"}

_DISCUSSION_COUNTERPART = {"planning_expert": "dev_expert", "dev_expert": "planning_expert"}
_VALID_NEXT_SPEAKERS = {"planning_expert", "dev_expert", "ideation_facilitator", "user"}

# 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — 발언 수/종료 조건 캡. 쟁점 하나당
# 최소/최대 발언 수와 라운드(=API 호출 1회 안에서 이어지는 전문가 발언 구간) 전체의 최소/최대
# 발언 수를 분리한다 — 라운드 캡은 "언제 진행자가 개입해도 되는가"(consensus_reached 판단에
# 최소한의 논의는 있었는지), 쟁점 캡은 "같은 쟁점에서 무한히 반론이 오가는 것"을 막는다.
MIN_EXPERT_TURNS_PER_ISSUE = 2
MAX_EXPERT_TURNS_PER_ISSUE = 6
MIN_EXPERT_TURNS_PER_ROUND = 4
MAX_EXPERT_TURNS_PER_ROUND = 8


def _most_recent_message_by(messages: list[ConvMessage], speaker_id: str) -> ConvMessage | None:
    for msg in reversed(messages):
        if msg.get("speaker_id") == speaker_id:
            return msg
    return None


def _discussion_stage_for(state: IdeationConvState, persona_id: str) -> str:
    """용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): 이번 발언이 "직전 발언에
    반응하는 것"인지를 결정한다. previous_speaker(라운드 내 마지막 "전문가" 발언자) 대신
    messages의 마지막 메시지가 실제로 상대 전문가인지를 직접 본다 — 그래야 사용자가
    "잠시만"으로 끼어들거나(interjection) 진행자가 방금 라운드를 열었을 때도, 다음 발언이
    "상대 전문가에게 반응"이 아니라 "방금 들어온 메시지(사용자/진행자)에 반응"으로 정확히
    분류된다(요청 6번 — 사용자 개입 메시지가 다음 전문가 발언의 responding_to로 잡혀야
    한다)."""
    messages = state["messages"]
    last = messages[-1] if messages else None
    counterpart = _DISCUSSION_COUNTERPART.get(persona_id)
    if last is not None and last.get("speaker_id") == counterpart:
        return "response"
    return "initial_position"


def _responding_to_for(state: IdeationConvState, persona_id: str, discussion_stage: str) -> ConvMessage | None:
    """이번 발언이 실제로 무엇에 반응하는지(responding_to_message_id/speaker_id)를 코드가
    결정적으로 찾는다 — LLM이 존재하지 않는 id를 지어낼 위험을 피하기 위해 절대 LLM에게
    맡기지 않는다.

    "response"면 상대 페르소나가 가장 최근에 남긴 메시지를 찾는다. "initial_position"이면
    직전 메시지가 사용자 개입일 때만 그 사용자 메시지를 대상으로 삼고, 진행자 안건/정리는
    대화 컨텍스트로만 참고한다."""
    messages = state["messages"]
    if discussion_stage == "response":
        counterpart = _DISCUSSION_COUNTERPART.get(persona_id)
        target = _most_recent_message_by(messages, counterpart) if counterpart else None
        if target is not None:
            return target
    # 최초 입장에서는 방금 들어온 사용자 개입에만 직접 응답한다. 진행자의 안건/정리는
    # 대화 컨텍스트로 참고할 뿐 전문가 상호검토의 responding_to 대상으로 저장하지 않는다.
    if messages:
        latest = messages[-1]
        if latest.get("speaker_id") == "user":
            return latest
    return None


_ISSUE_ID_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _fallback_issue_id(state: IdeationConvState) -> str:
    """LLM이 active_issue_id를 비워서 돌려주는 경우(구조화 검증이 이미 재시도를 유발하므로
    정상 흐름에서는 거의 도달하지 않는다)에 대비한 최후 방어값 — 이미 열린 쟁점이 있으면 그걸
    그대로 쓰고, 없으면 라운드 번호 기반 slug를 만든다."""
    open_issues = state.get("open_issues") or []
    if open_issues:
        return open_issues[-1]["issue_id"]
    return f"issue_r{state.get('round', 1)}"


# 용준/Claude(2026-07-23, 요청: "동일 쟁점 표현 변경 반복 루프" 수정) — 진행자/전문가가 같은
# 쟁점을 문장만 바꿔 새 issue_id로 재등록하는 문제를 결정론적 코드로 막는다. 임베딩·LLM 호출
# 없이 (1) 표현을 정규화하고 (2) TOPIC_PRIORITY 공식 평가축 키워드로 canonical family를
# 판정하고 (3) active/open/resolved(강제 종료 포함) 쟁점과 비교해 중복이면 등록을 억제하며
# 필요하면 아직 다루지 않은 다음 공식 평가축으로 로테이션한다.
_ISSUE_NORMALIZE_STOPWORDS = frozenset(
    {
        "필요하다", "필요합니다", "필요한", "필요", "구체적으로", "구체적인", "구체적", "방안", "방법",
        "검토", "추가적으로", "추가로", "추가", "가능성", "확보", "확인", "관련", "대한", "위한", "합니다",
        "해야", "해야합니다", "입니다", "됩니다", "되어야", "있습니다", "있다", "한다", "것", "의", "및",
    }
)
_ISSUE_PARTICLE_RE = re.compile(r"(을|를|이|가|은|는|의|에서|에게|으로|로|와|과|도|만|까지|부터)$")
_ISSUE_VERB_SUFFIX_RE = re.compile(r"(합니다|됩니다|습니다|입니다|한다|된다|해요|해야|하다|되다)$")
_ISSUE_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_issue_text(text: str | None) -> str:
    """쟁점 문구를 결정론적으로 정규화한다 — 공백·문장부호를 지우고, 흔한 조사·서술어
    종결 어미를 어절 끝에서 떼어내고, "필요하다/구체적/방안/방법/검토/추가" 같은 일반
    표현의 영향을 줄인 뒤, 어절을 정렬해 어순만 다른 동일 표현도 같은 결과가 되게 한다
    (bag-of-words 비교). 임베딩·외부 API를 쓰지 않는 순수 문자열 함수다."""
    if not text:
        return ""
    lowered = _ISSUE_PUNCT_RE.sub(" ", text.strip().lower())
    tokens: list[str] = []
    for raw_token in lowered.split():
        token = _ISSUE_PARTICLE_RE.sub("", raw_token)
        token = _ISSUE_VERB_SUFFIX_RE.sub("", token)
        if not token or token in _ISSUE_NORMALIZE_STOPWORDS:
            continue
        tokens.append(token)
    return " ".join(sorted(tokens))


# TOPIC_PRIORITY(공식 평가축) 각 슬러그로 자유 문장 쟁점 제목을 매핑하기 위한 키워드.
# question_topic과 동일한 분류 체계를 재사용한다(요청: "새로운 임의 분류 체계를 중복
# 생성하지 마세요") — 여기서 새 축을 만들지 않고, 이미 있는 TOPIC_PRIORITY만 재사용한다.
#
# 용준/Claude(2026-07-23, 요청: "canonical family 다중 키워드 충돌 수정") — 각 키워드에
# 가중치(int)를 붙인다. "문제"("니즈" 등 포괄적인 단일 단어)는 낮은 가중치를, "실시간
# 데이터"/"데이터 수집"/"api"/"센서"처럼 구체적인 표현은 높은 가중치를 준다 — 첫 매칭
# 우선(첫 family 즉시 반환) 대신, family별 총점을 비교해 가장 구체적으로 걸린 family를
# 고른다(예: "문제 해결을 위한 실시간 데이터 수집 방안"은 problem이 아니라 data여야 한다).
_ISSUE_FAMILY_KEYWORDS: dict[str, tuple[tuple[str, int], ...]] = {
    "problem": (("문제", 1), ("니즈", 2), ("페인포인트", 3), ("과제 정의", 3)),
    "target_user": (
        ("타겟", 2),
        ("대상 사용자", 3),
        ("사용자층", 3),
        ("고객층", 3),
        ("목표 사용자", 3),
    ),
    "core_value": (("핵심 가치", 3), ("가치 제안", 3), ("핵심가치", 3)),
    "contest_fit": (
        ("공모전", 2),
        ("선정 기준", 3),
        ("심사 기준", 3),
        ("적합성", 2),
        ("평가 기준", 3),
    ),
    "differentiation": (("차별", 2), ("경쟁사", 2), ("경쟁 우위", 3)),
    "mvp": (("mvp", 2), ("구축 범위", 3), ("우선순위 기능", 3), ("핵심 기능", 3)),
    "data": (
        ("데이터", 2),
        ("api", 3),
        ("소스", 1),
        ("수집", 1),
        ("실시간", 2),
        ("제공자", 2),
        ("연계", 1),
        ("센서", 3),
        ("실시간 데이터", 4),
        ("데이터 수집", 4),
        ("데이터 확보", 4),
    ),
    "ai_role": (("ai 역할", 3), ("인공지능 역할", 3), ("모델 역할", 3), ("ai 활용", 3)),
    "roadmap": (("로드맵", 3), ("확장 계획", 3), ("향후 계획", 3), ("확장 기능", 2)),
}

# TOPIC_PRIORITY 슬러그 -> 사람이 읽는 한국어 제목(로테이션 시 다음 공식 쟁점의 표시용).
_TOPIC_TITLE_KO: dict[str, str] = {
    "problem": "문제 정의",
    "target_user": "목표 사용자",
    "core_value": "핵심 가치",
    "contest_fit": "공모전 적합성",
    "differentiation": "차별성",
    "mvp": "MVP 범위",
    "data": "데이터 확보 방안",
    "ai_role": "AI 활용 방식",
    "roadmap": "로드맵",
}


def resolve_canonical_issue_family(text: str | None, issue_id: str | None = None) -> str:
    """쟁점 제목(또는 missing_information)을 canonical family로 매핑한다.

    용준/Claude(2026-07-23, 요청: "canonical family 다중 키워드 충돌 수정"):
    1) issue_id가 공식 슬러그 형태("topic_<family>")면 텍스트 추정 없이 그 family를 그대로
       쓴다(요청: "공식 issue_id가 명시돼 있으면 텍스트 추정보다 우선").
    2) 아니면 family별로 걸린 키워드의 가중치 합(점수)을 계산해 가장 높은 family를 고른다
       — 첫 매칭 즉시 반환하던 이전 방식은 "문제"처럼 포괄적인 단일 단어가 "실시간 데이터
       수집"처럼 더 구체적인 표현보다 먼저 걸려 잘못 분류되는 문제가 있었다.
    3) 점수가 동점이면 TOPIC_PRIORITY 순서로 첫 번째를 쓴다(기존 우선순위 규칙 유지).
    4) 어떤 family도 걸리지 않으면(예: "리스크", "기술 구현 가능성"처럼 공식 9축에 없는
       실제로 다른 주제) 정규화된 텍스트 자체를 family로 써서 서로 다른 쟁점끼리 잘못
       합쳐지지 않게 한다."""
    if issue_id and issue_id.startswith("topic_"):
        official_family = issue_id[len("topic_"):]
        if official_family in TOPIC_PRIORITY:
            return official_family

    haystack = (text or "").lower()
    scores: dict[str, int] = {}
    for family in TOPIC_PRIORITY:
        weighted_keywords = _ISSUE_FAMILY_KEYWORDS.get(family, ())
        score = sum(weight for keyword, weight in weighted_keywords if keyword in haystack)
        if score > 0:
            scores[family] = score

    if scores:
        best_score = max(scores.values())
        for family in TOPIC_PRIORITY:
            if scores.get(family) == best_score:
                return family

    normalized = normalize_issue_text(text)
    return f"custom:{normalized}" if normalized else "custom:unknown"


def is_duplicate_issue(family_a: str | None, family_b: str | None) -> bool:
    """두 canonical family가 같은 쟁점을 가리키는지 판정하는 순수 함수."""
    return bool(family_a) and bool(family_b) and family_a == family_b


def _select_next_issue_family(
    *,
    excluded_family: str | None,
    open_issues: list[dict],
    resolved_issues: list[dict],
    resolved_topics: list[str] | None,
) -> str | None:
    """발언 상한으로 닫힌(또는 중복 판정된) family 다음에 다룰 공식 평가축을 고른다.
    우선순위: (1) 이미 열려 있는 다른 open_issues의 family, (2) TOPIC_PRIORITY에서 아직
    다루지 않은 다음 공식 쟁점. 둘 다 없으면 None(더 다룰 공식 쟁점이 없다는 뜻 — 호출부가
    이 경우 회의를 정리한다)."""
    for issue in open_issues:
        family = issue.get("family") or resolve_canonical_issue_family(issue.get("title"))
        if family and family != excluded_family:
            return family

    covered = {excluded_family} if excluded_family else set()
    for issue in resolved_issues:
        family = issue.get("family") or resolve_canonical_issue_family(issue.get("title"))
        if family:
            covered.add(family)
    covered.update(resolved_topics or [])

    for topic in TOPIC_PRIORITY:
        if topic not in covered:
            return topic
    return None


def resolve_issue_duplicate(
    *,
    candidate_issue_id: str,
    candidate_issue_title: str,
    current_active_issue_id: str | None,
    open_issues: list[dict],
    resolved_issues: list[dict],
    resolved_topics: list[str] | None = None,
) -> dict[str, Any]:
    """이번 턴이 제안한 쟁점(candidate_issue_id/title)이 세션 안에서 의미상 이미 다뤄진
    쟁점인지 결정적으로 판정한다 — LLM이 문장을 바꿔 같은 쟁점을 새 id로 반환해도, 여기서
    최종적으로 어떤 issue_id/title을 이번 턴에 쓸지 코드가 확정한다.

    반환 dict:
      - issue_id/issue_title: 이번 턴이 실제로 써야 할 값(None이면 로테이션할 다음 공식
        쟁점도 없다는 뜻 — 호출부가 활성 쟁점 없이 이번 턴을 처리해야 한다).
      - duplicate: 이번 제안이 기존 쟁점과 중복 판정됐는지.
      - duplicate_of_issue_id/duplicate_source("active"/"open"/"resolved"/"parked"): 무엇과
        중복인지.
      - rotated: 중복이라 다음 공식 평가축으로 이동했는지(True면 issue_id/title이 후보가
        아니라 로테이션된 다음 공식 쟁점이다).
    """
    candidate_family = resolve_canonical_issue_family(
        candidate_issue_title or candidate_issue_id, issue_id=candidate_issue_id
    )

    def _accepted() -> dict[str, Any]:
        return {
            "issue_id": candidate_issue_id,
            "issue_title": candidate_issue_title,
            "canonical_family": candidate_family,
            "duplicate": False,
            "duplicate_of_issue_id": None,
            "duplicate_source": None,
            "rotated": False,
        }

    if current_active_issue_id is not None and candidate_issue_id == current_active_issue_id:
        return _accepted()

    for issue in open_issues:
        if issue.get("issue_id") == candidate_issue_id:
            return {
                "issue_id": candidate_issue_id,
                "issue_title": issue.get("title") or candidate_issue_title,
                "canonical_family": issue.get("family") or candidate_family,
                "duplicate": False,
                "duplicate_of_issue_id": None,
                "duplicate_source": None,
                "rotated": False,
            }

    for issue in open_issues:
        issue_family = issue.get("family") or resolve_canonical_issue_family(issue.get("title"))
        if is_duplicate_issue(candidate_family, issue_family):
            source = "active" if issue.get("issue_id") == current_active_issue_id else "open"
            return {
                "issue_id": issue["issue_id"],
                "issue_title": issue.get("title") or candidate_issue_title,
                "canonical_family": issue_family,
                "duplicate": True,
                "duplicate_of_issue_id": issue["issue_id"],
                "duplicate_source": source,
                "rotated": False,
            }

    matched_closed: tuple[str, str] | None = None  # (duplicate_of_issue_id, source)
    for issue in resolved_issues:
        issue_family = issue.get("family") or resolve_canonical_issue_family(issue.get("title"))
        if is_duplicate_issue(candidate_family, issue_family):
            source = "parked" if issue.get("resolution_kind") == "parked_expert_judgment" else "resolved"
            matched_closed = (issue.get("issue_id"), source)
            break

    if matched_closed is None:
        return _accepted()

    duplicate_of_issue_id, source = matched_closed
    next_family = _select_next_issue_family(
        excluded_family=candidate_family,
        open_issues=open_issues,
        resolved_issues=resolved_issues,
        resolved_topics=resolved_topics,
    )
    if next_family is not None:
        return {
            "issue_id": f"topic_{next_family}",
            "issue_title": _TOPIC_TITLE_KO.get(next_family, next_family),
            "canonical_family": next_family,
            "duplicate": True,
            "duplicate_of_issue_id": duplicate_of_issue_id,
            "duplicate_source": source,
            "rotated": True,
        }
    return {
        "issue_id": None,
        "issue_title": None,
        "canonical_family": candidate_family,
        "duplicate": True,
        "duplicate_of_issue_id": duplicate_of_issue_id,
        "duplicate_source": source,
        "rotated": False,
    }


def _update_issue_records(
    *,
    open_issues: list[dict],
    resolved_issues: list[dict],
    persona_id: str,
    issue_id: str,
    issue_title: str,
    position_text: str,
    resolved: bool,
    resolution_text: str,
    family: str | None = None,
    closed_reason: str | None = None,
    resolution_kind: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): open_issues/resolved_issues를
    코드가 결정적으로 갱신한다 — LLM은 issue_resolved bool만 판단하고, 레코드 생성·이동은
    항상 여기서 수행한다(라우팅이 LLM 추천을 그대로 신뢰하지 않는 것과 같은 원칙).

    용준/Claude(2026-07-23, 요청: 동일 쟁점 표현 변경 반복 루프 수정) — family(canonical
    family)를 레코드에 함께 저장해 다음 턴의 resolve_issue_duplicate가 재계산 없이 바로
    비교할 수 있게 한다. resolved=True일 때만 closed_reason/resolution_kind를 채운다(요청:
    "강제 종료와 합의 완료 구분") — 기본값은 정상 합의 해결이다."""
    open_issues = list(open_issues)
    position_key = "planning_position" if persona_id == "planning_expert" else "development_position"
    resolved_family = family or resolve_canonical_issue_family(issue_title or issue_id)

    idx = next((i for i, issue in enumerate(open_issues) if issue["issue_id"] == issue_id), None)
    if idx is None:
        record = {
            "issue_id": issue_id,
            "title": issue_title or issue_id,
            "status": "open",
            "planning_position": None,
            "development_position": None,
            "resolution": None,
            "turns": 0,
            "family": resolved_family,
            "closed_reason": None,
            "resolution_kind": None,
        }
        open_issues.append(record)
        idx = len(open_issues) - 1

    record = dict(open_issues[idx])
    record[position_key] = position_text
    record["turns"] = record.get("turns", 0) + 1
    if issue_title:
        record["title"] = issue_title
    record["family"] = record.get("family") or resolved_family

    if resolved:
        record["status"] = "resolved"
        record["resolution"] = resolution_text
        record["closed_reason"] = closed_reason or "consensus_reached"
        record["resolution_kind"] = resolution_kind or "agreed_resolution"
        resolved_issues = list(resolved_issues) + [record]
        open_issues = [issue for i, issue in enumerate(open_issues) if i != idx]
    else:
        open_issues[idx] = record

    return open_issues, resolved_issues


def _route_next_expert_turn(state: IdeationConvState) -> str:
    """용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): "기획 1회 → 개발 1회 → 진행자
    정리"로 고정됐던 그래프 엣지를 대체하는 조건부 라우터. 쟁점·반론·발언 캡에 따라 다음에
    실행할 노드("planning_expert"/"dev_expert"/"facilitator"/"failed")를 계산한다.
    ideation_conv_build.py가 planning_expert_discussion/dev_expert_discussion 두 노드 모두에서
    이 함수로 조건부 엣지를 건다(양방향 — 기획이 개발을 부르고, 개발도 기획을 다시 부를 수
    있다). LLM이 추천한 recommended_next_speaker를 그대로 신뢰하지 않고, 캡·상태와 함께
    검증한 뒤에만 그 노드로 보낸다."""
    def routed(selected: str, reason: str, recommended: str | None = None) -> str:
        last_message = state["messages"][-1] if state.get("messages") else None
        last_structured = (last_message.get("structured") or {}) if last_message else {}
        active_issue_id = state.get("active_issue_id")
        issue_turn_count = next(
            (
                issue.get("turns", 0)
                for issue in (state.get("open_issues") or [])
                if issue.get("issue_id") == active_issue_id
            ),
            0,
        )
        trace_event(
            "IDEATION_ROUTE_DECISION",
            session_id=state.get("session_id"),
            current=last_message.get("speaker_id") if last_message else None,
            recommended=recommended or last_structured.get("recommended_next_speaker"),
            selected=selected,
            validated_next=selected,
            issue=active_issue_id,
            resolved=last_structured.get("issue_resolved"),
            needs_counterpart=last_structured.get("needs_counterpart_response"),
            required_counterpart=state.get("required_counterpart_speaker_id"),
            counterpart_review_completed=state.get("counterpart_review_completed"),
            turn_count=state.get("expert_turn_count", 0),
            issue_turn_count=issue_turn_count,
            stop_reason=reason if selected == "facilitator" else None,
            reason=reason,
            route_reason=reason,
        )
        return selected

    if state.get("phase") == "failed":
        return routed("failed", "phase_failed")

    messages = state["messages"]
    last = messages[-1] if messages else None
    if last is None or last.get("speaker_id") not in ("planning_expert", "dev_expert"):
        # 방어적 기본값 — 정상 흐름에서는 항상 방금 전문가 발언 직후에만 이 라우터가 불린다.
        return routed("facilitator", "missing_expert_message")
    structured = last.get("structured") or {}

    turn_count = state.get("expert_turn_count", 0)
    hit_round_cap = turn_count >= MAX_EXPERT_TURNS_PER_ROUND

    active_issue_id = state.get("active_issue_id")
    hit_issue_cap = False
    if active_issue_id:
        for issue in state.get("open_issues") or []:
            if issue["issue_id"] == active_issue_id and issue.get("turns", 0) >= MAX_EXPERT_TURNS_PER_ISSUE:
                hit_issue_cap = True
                break

    # 용준/Claude(2026-07-22, 요청: 지정 위원 질문 후 상대 검토 코드 강제) — required_counterpart_
    # speaker_id가 남아있고 아직 검토가 끝나지 않았다면(counterpart_review_completed=False),
    # issue_resolved/needs_user_input/recommended_next_speaker 등 다른 어떤 신호보다 이 검토를
    # 우선한다 — "지정 위원이 issue_resolved=true를 반환해도 상대 검토 전에는 최종 확정 금지"
    # (요청 7번)와 "counterpart_review_completed=false인 동안 facilitator 이동 금지"(요청 6번)의
    # 실제 강제 지점이다. 발언·LLM 호출 상한(요청 10번)만은 예외로 여전히 절대 우선한다 — 상한에
    # 도달했는데도 검토를 강제하면 무한 루프 방지 장치 자체가 무력화되기 때문이다.
    required_counterpart = state.get("required_counterpart_speaker_id")
    review_pending = bool(required_counterpart) and not state.get("counterpart_review_completed", True)
    if review_pending:
        if hit_round_cap or hit_issue_cap:
            return routed("facilitator", "hard_turn_cap")
        return routed(required_counterpart, "required_counterpart_review")

    if hit_round_cap:
        return routed("facilitator", "max_turns_reached")
    if structured.get("needs_user_input"):
        return routed("facilitator", "user_input_required")
    if hit_issue_cap:
        return routed("facilitator", "max_issue_turns_reached")

    # 첫 전문가가 상대 검토가 필요 없다고 판단하더라도 한 라운드에 두 전문가는 최소 한 번씩
    # 발언해야 한다. LLM의 단일 판단으로 planning → facilitator가 반복되어 개발 관점이
    # 완전히 사라지는 것을 라우터에서 결정적으로 차단한다.
    current_round = state.get("round")
    round_speakers = {
        message.get("speaker_id")
        for message in messages
        if message.get("round") == current_round
        and message.get("speaker_id") in ("planning_expert", "dev_expert")
    }
    counterpart = _DISCUSSION_COUNTERPART.get(last["speaker_id"])
    if (
        turn_count < 2
        and counterpart
        and counterpart not in round_speakers
    ):
        return routed(counterpart, "round_counterpart_required")

    if turn_count >= MIN_EXPERT_TURNS_PER_ROUND and not (state.get("open_issues") or []):
        # 열린 쟁점이 하나도 없다 — 지금까지 다룬 쟁점이 전부 해결됐다는 뜻이므로 정리한다.
        return routed("facilitator", "consensus_reached")

    recommended = structured.get("recommended_next_speaker")
    if recommended not in _VALID_NEXT_SPEAKERS:
        recommended = _DISCUSSION_COUNTERPART.get(last["speaker_id"], "ideation_facilitator")
    if recommended in ("ideation_facilitator", "user"):
        return routed("facilitator", "recommended_facilitator_or_user", recommended)
    if recommended == last["speaker_id"]:
        # 용준/Claude(2026-07-22, 실측: 실제 OpenAI 호출로 라이브 검증 중 같은 화자가 스스로를
        # 다음 발언자로 반복 추천해 3턴 연속 독백하는 사례를 확인했다) — 화자가 스스로를
        # 다음 발언자로 추천하는 것은 코드가 그대로 신뢰하지 않는다(요청: "의미 없는 독백은
        # 제한"). needs_counterpart_response 값과 무관하게 항상 상대에게 넘긴다 — 상대가
        # 정말 더 할 말이 없다고 판단하면 상대 자신의 다음 발언에서 진행자에게 넘기게 된다.
        recommended = _DISCUSSION_COUNTERPART.get(last["speaker_id"], "ideation_facilitator")
        if recommended == "ideation_facilitator":
            return routed("facilitator", "self_recommendation_without_counterpart")
        return routed(recommended, "self_recommendation_redirected", structured.get("recommended_next_speaker"))
    return routed(recommended, "validated_recommendation", recommended)


def _trace_shadow_plan_created(
    *,
    session_id: str | None,
    persona_id: str,
    plan: dict,
    retrieved_evidence_count: int,
    query: str,
    elapsed_ms: float,
) -> None:
    """용준/Claude(2026-07-23, Phase 1 "Shadow Deterministic Evidence Planner"): planner가
    만든 plan을 trace 로그 하나로 남긴다(요청: CREATED/EMPTY/INVALID 이벤트 구분). plan
    내용은 prompt/claims/grounding/routing에 전혀 영향을 주지 않는다 — 여기서 로그로만
    소비된다."""
    validation = plan.get("validation") or {"valid": True, "errors": []}
    issue = plan.get("issue") or {}
    fields = dict(
        session_id=session_id,
        speaker=persona_id,
        plan_id=plan.get("plan_id"),
        policy_version=plan.get("policy_version"),
        effective_issue_id=issue.get("issue_id"),
        effective_issue_title=issue.get("title"),
        retrieval_query_preview=sanitize_preview(query, limit=160),
        retrieved_evidence_count=retrieved_evidence_count,
        eligible_evidence_count=plan.get("eligible_evidence_count"),
        selected_evidence_count=len(plan.get("selected_evidence") or []),
        empty_plan_reason=plan.get("empty_plan_reason"),
        validation_valid=validation.get("valid"),
        validation_errors=validation.get("errors"),
        elapsed_ms=elapsed_ms,
        selected_evidence=[
            {
                "ref": item.get("ref"),
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "document_role": item.get("document_role"),
                "claim_type": item.get("claim_type"),
                "quote_preview": sanitize_preview(item.get("quote", ""), limit=160),
                "field_label": item.get("field_label"),
                "quote_start": item.get("quote_start"),
                "quote_end": item.get("quote_end"),
                "retrieval_score": item.get("retrieval_score"),
                "issue_relevance_score": item.get("issue_relevance_score"),
                "selection_reason_code": item.get("selection_reason_code"),
                "reused_in_same_issue": item.get("reused_in_same_issue"),
            }
            for item in plan.get("selected_evidence") or []
        ],
    )
    if not validation.get("valid", True):
        trace_event("IDEATION_EVIDENCE_PLAN_SHADOW_INVALID", **fields)
    elif plan.get("empty_plan_reason"):
        trace_event("IDEATION_EVIDENCE_PLAN_SHADOW_EMPTY", **fields)
    else:
        trace_event("IDEATION_EVIDENCE_PLAN_SHADOW_CREATED", **fields)


def _run_shadow_evidence_planner(
    *,
    evidence_planner: "EvidencePlanningFn | None",
    persona_id: str,
    session_id: str | None,
    effective_issue: dict[str, str],
    query: str,
    retrieved: list[dict],
    runtime_scope: dict[str, Any],
    shadow_history_map: dict[str, list[dict]],
) -> tuple[dict | None, dict[str, list[dict]], float | None]:
    """shadow planner를 호출하고(있으면), plan/갱신된 shadow_history_map/실행 소요시간(ms)을
    반환한다. 예외가 나도 회의 발언 생성 자체는 절대 막지 않는다(요청: "planner 예외: 기존
    발언 생성을 실패시키지 않고 shadow failure 로그 후 기존 흐름 진행") — 실패하면
    (None, 기존 맵, None)을 그대로 돌려준다.

    용준/Claude(2026-07-23, Phase 2 "Active Evidence Injection"): 이 함수는 shadow 로그
    전용이 아니라 planner의 유일한 호출 지점이다 — active 모드(evidence_planner.active=True)가
    함께 켜져 있어도 여기서 만든 plan을 그대로 재사용한다(요청: "active와 shadow가 동시에
    켜져도 Planner를 중복 실행하지 말 것", "한 번 생성한 plan을 shadow 기록과 active
    injection에 함께 사용할 것"). 반환된 elapsed_ms는 shadow 로그와 active/fallback 로그가
    함께 쓴다."""
    if evidence_planner is None:
        return None, shadow_history_map, None

    shadow_key = f"{persona_id}:{effective_issue['issue_id']}"
    shadow_history_items = shadow_history_map.get(shadow_key, [])
    plan_started = time.perf_counter()
    try:
        plan = evidence_planner(
            persona_id=persona_id,
            effective_issue={**effective_issue, "query": query},
            retrieved_evidence=retrieved,
            runtime_scope=runtime_scope,
            shadow_history=shadow_history_items,
        )
    except Exception:
        logger.exception(
            "[IDEATION_EVIDENCE_PLAN_SHADOW_FAILED] session_id=%s speaker=%s", session_id, persona_id
        )
        trace_event(
            "IDEATION_EVIDENCE_PLAN_SHADOW_FAILED",
            level=logging.WARNING,
            session_id=session_id,
            speaker=persona_id,
            effective_issue_id=effective_issue["issue_id"],
            effective_issue_title=effective_issue["title"],
        )
        return None, shadow_history_map, None

    elapsed_ms = round((time.perf_counter() - plan_started) * 1000, 1)
    _trace_shadow_plan_created(
        session_id=session_id,
        persona_id=persona_id,
        plan=plan,
        retrieved_evidence_count=len(retrieved),
        query=query,
        elapsed_ms=elapsed_ms,
    )

    selected_chunk_ids = [item["chunk_id"] for item in plan.get("selected_evidence") or [] if item.get("chunk_id")]
    if not selected_chunk_ids:
        return plan, shadow_history_map, elapsed_ms

    updated_items = shadow_history_items + [
        {"speaker": persona_id, "effective_issue_id": effective_issue["issue_id"], "chunk_id": chunk_id}
        for chunk_id in selected_chunk_ids
    ]
    new_shadow_history_map = dict(shadow_history_map)
    new_shadow_history_map[shadow_key] = updated_items[-_SHADOW_HISTORY_KEEP:]
    return plan, new_shadow_history_map, elapsed_ms


def _build_active_planned_evidence(plan: dict, retrieved: list[dict]) -> list[dict]:
    """용준/Claude(2026-07-23, Phase 2 "Active Evidence Injection") — plan["selected_evidence"]를
    prompt/grounding에 그대로 주입할 근거 항목으로 변환한다. 원본 청크 전체 text가 아니라
    planner가 원문 substring 검증까지 마친 quote만 담는다(요청: "Planner가 추출한 검증된
    quote를 사용"). ref/chunk_id/document_id/document_role/claim_type/quote/source(=문서명)/
    page/effective_issue_id/effective_issue_title을 함께 전달한다(요청 필드 그대로) —
    document_name/section은 claim_grounding.is_relevant_candidate의 관련성 판정 품질을
    legacy(retrieved 전체 주입) 경로와 최대한 동등하게 유지하기 위한 내부 보조 필드다(LLM
    프롬프트 지시문이 요구하는 필드 목록에는 없지만, 그 값 자체가 이번 턴에 선택된 근거의
    사실 정보라 노출해도 "선택되지 않은 evidence 노출 금지" 요구와 충돌하지 않는다)."""
    issue = plan.get("issue") or {}
    by_ref = {item.get("ref"): item for item in retrieved if isinstance(item, dict) and item.get("ref")}
    items: list[dict] = []
    for selected in plan.get("selected_evidence") or []:
        source = by_ref.get(selected.get("ref")) or {}
        items.append(
            {
                "ref": selected.get("ref"),
                "chunk_id": selected.get("chunk_id"),
                "document_id": selected.get("document_id"),
                "document_role": selected.get("document_role"),
                "claim_type": selected.get("claim_type"),
                "quote": selected.get("quote"),
                "field_label": selected.get("field_label"),
                "source": source.get("document_name"),
                "page": source.get("page"),
                "effective_issue_id": issue.get("issue_id"),
                "effective_issue_title": issue.get("title"),
                "document_name": source.get("document_name"),
                "section": source.get("section"),
            }
        )
    return items


def _resolve_discussion_evidence(
    *,
    evidence_planner: "EvidencePlanningFn | None",
    shadow_plan: dict | None,
    retrieved: list[dict],
) -> tuple[list[dict], str, str | None]:
    """용준/Claude(2026-07-23, Phase 2 "Active Evidence Injection") — evidence_planner의
    active 속성(백엔드가 ENABLE_IDEATION_EVIDENCE_PLANNER_DISCUSSION일 때만 세팅,
    _evidence_planner_for 참고)이 꺼져 있으면 무조건 기존 legacy 경로(retrieved 전체)를
    그대로 돌려준다 — Phase 1 shadow 전용일 때와 완전히 동일하게 동작한다.

    active가 켜져 있으면 이번 턴 prompt/grounding에 실제로 쓸 evidence와 모드를 결정한다:
      - "fallback": planner가 기술적으로 실패했다(예외로 shadow_plan이 None이거나, plan이
        validation을 통과하지 못했다) — retrieved 전체로 되돌아간다(요청 A. "Planner 기술
        실패" 정책). 부적합한 plan을 그대로 못 믿는 것이지, "근거가 없다"는 판단 자체가
        틀렸다는 뜻이 아니므로 기존 경로가 안전하다.
      - "valid_empty": plan validation은 통과했지만 이번 쟁점에 맞는 근거가 하나도 없다고
        planner가 정상적으로 판단했다(selected_evidence=[]) — retrieved 전체로 fallback하지
        않는다(요청 B. "부적합한 근거를 다시 주입하면 Planner eligibility 정책이
        무효화되기 때문"). Evidence-first 활성 경로에서는 호출부가 일반 전문가 생성을
        생략하고 진행자에게 넘긴다.
      - "active": plan이 유효하고 selected_evidence가 있다 — 그 근거만 prompt/grounding에
        노출한다(선택되지 않은 evidence는 노출하지 않는다).
    """
    if evidence_planner is None or not getattr(evidence_planner, "active", False):
        return retrieved, "inactive", None
    if shadow_plan is None:
        return retrieved, "fallback", "planner_exception"
    validation = shadow_plan.get("validation") or {"valid": True, "errors": []}
    if not validation.get("valid", True):
        return retrieved, "fallback", "plan_validation_failed"
    selected = shadow_plan.get("selected_evidence") or []
    if not selected:
        return [], "valid_empty", None
    return _build_active_planned_evidence(shadow_plan, retrieved), "active", None


def _plan_supplemental_evidence(
    *,
    evidence_planner: "EvidencePlanningFn | None",
    persona_id: str,
    effective_issue: dict[str, str],
    supplemental_query: str,
    retrieved: list[dict],
    runtime_scope: dict[str, Any],
) -> tuple[list[dict], dict | None]:
    """보완 검색 결과도 최초 검색과 동일한 active Planner 계약을 통과시킨다."""
    if not retrieved:
        return [], None
    if evidence_planner is None or not getattr(evidence_planner, "active", False):
        # Planner active 플래그가 꺼진 레거시 세션은 기존 보완 검색 동작을 보존한다.
        return list(retrieved), None
    try:
        plan = evidence_planner(
            persona_id=persona_id,
            effective_issue={
                **effective_issue,
                "query": supplemental_query,
                "planner_stage": "supplemental_retrieval",
            },
            retrieved_evidence=retrieved,
            runtime_scope=runtime_scope,
            shadow_history=[],
        )
    except Exception:
        logger.exception(
            "[IDEATION_SUPPLEMENTAL_EVIDENCE_PLAN_FAILED] speaker=%s issue=%s",
            persona_id,
            effective_issue.get("issue_id"),
        )
        return [], None
    validation = plan.get("validation") or {"valid": True, "errors": []}
    if not validation.get("valid", True) or not (plan.get("selected_evidence") or []):
        return [], plan
    return _build_active_planned_evidence(plan, retrieved), plan


def _build_evidence_plan_notice(mode: str, plan: dict | None) -> str:
    """용준/Claude(2026-07-23, Phase 2 "Active Evidence Injection") — active/valid_empty
    모드일 때만 prompt에 붙일 안내문을 만든다(요청: "이번 발언이 다룰 effective_issue_id/
    title", "해당 쟁점 범위 안에서 판단할 것", "다른 쟁점으로 임의 전환하지 말 것"을 명확히
    전달). routing이나 state의 active_issue_id를 강제로 바꾸지 않는다 — prompt 지시문으로만
    범위를 좁힌다."""
    if mode not in ("active", "valid_empty") or not plan:
        return ""
    issue = plan.get("issue") or {}
    issue_id = issue.get("issue_id") or ""
    issue_title = issue.get("title") or issue_id
    if mode == "active":
        allowed_refs = [
            item.get("ref")
            for item in plan.get("selected_evidence") or []
            if isinstance(item, dict) and isinstance(item.get("ref"), str) and item.get("ref")
        ]
        allowed_refs_text = ", ".join(allowed_refs) if allowed_refs else "없음"
        return (
            f'이번 발언은 아래 [검색 근거]에 나열된 항목만 문서 근거로 인용할 수 있습니다 — '
            f"이 목록에 없는 근거를 지어내거나 가정하지 마세요. 이번 발언이 다뤄야 할 쟁점은 "
            f'effective_issue_id="{issue_id}"(제목: "{issue_title}")입니다. 이 쟁점 범위 안에서만 '
            f"판단하고, 특별한 이유 없이 다른 쟁점으로 임의 전환하지 마세요. "
            f"현재 턴에서 허용된 evidence_refs는 [{allowed_refs_text}]뿐입니다. "
            f"evidence_refs에는 이번 [검색 근거] 목록에 실제로 표시된 ref만 사용할 수 있습니다. "
            f"최소 한 개의 document_fact 또는 user_provided_fact claim이 위 ref 중 하나를 "
            f"인용해야 하며, 전문가 판단은 그 근거를 적용한 별도 expert_judgment claim으로 "
            f"구분하세요. "
            f"대화 맥락의 과거 발언에서 보았던 E번호나 chunk_id는 현재 턴의 근거가 아니므로 "
            f"절대 재사용하지 마세요."
        )
    return (
        f'이번 쟁점(effective_issue_id="{issue_id}", 제목: "{issue_title}")에는 인용할 문서 근거가 '
        f"없습니다. 문서 사실(document_fact)을 새로 만들지 말고 전문가 판단(expert_judgment)으로만 "
        f"판단하세요. 이 쟁점 범위 안에서만 판단하고 다른 쟁점으로 임의 전환하지 마세요."
    )


def _build_user_interjection_notice(state: IdeationConvState) -> str:
    message = _active_user_interjection(state)
    if message is None:
        return ""
    question = message["content"].strip()
    structured = message.get("structured") or {}
    opinion_target = structured.get("opinion_target_speaker_id")
    interrupted_speaker = structured.get("interrupted_speaker_id")
    target_label = {
        "planning_expert": "기획 위원",
        "dev_expert": "개발 위원",
        "both": "기획 위원과 개발 위원 모두",
    }.get(opinion_target)
    target_notice = (
        f"사용자는 {target_label}의 의견을 대상으로 말하고 있습니다. "
        if target_label
        else ""
    )
    interrupted_targeted = (
        interrupted_speaker == opinion_target
        or opinion_target == "both"
        and interrupted_speaker in ("planning_expert", "dev_expert")
    )
    interrupted_notice = (
        "대상 위원의 직전 발언은 도중에 중단되어 완성 발언으로 저장되지 않았으므로, "
        "중단된 문장을 추측하지 말고 사용자의 질문에 적힌 내용만 기준으로 답하세요. "
        if interrupted_targeted
        else ""
    )
    return (
        "### 사용자 직접 질문 최우선 규칙\n"
        f'사용자가 방금 질문했습니다: "{question}"\n'
        f"{target_notice}{interrupted_notice}"
        "이번 발언의 첫 문장부터 이 질문에 직접 답하세요. 기존 쟁점의 일반론을 반복하거나 "
        "질문과 무관한 MVP·비용·기능 논의로 돌아가지 마세요. 문서 근거가 부족하면 그 사실을 "
        "밝힌 뒤 전문가 판단으로 실행 가능한 답을 제시하세요."
    )


def _trace_active_evidence_plan(
    *,
    session_id: str | None,
    persona_id: str,
    mode: str,
    fallback_reason: str | None,
    plan: dict | None,
    retrieved_evidence_count: int,
    injected_evidence_count: int,
    elapsed_ms: float | None,
) -> None:
    """용준/Claude(2026-07-23, Phase 2 "Active Evidence Injection") — prompt 조립 직전(아직
    LLM 응답/grounding 이전)에 이번 턴의 evidence 결정을 한 이벤트로 남긴다. claim/grounding
    관련 필드는 이 시점에 존재하지 않으므로 _trace_evidence_plan_compliance(응답 이후)가
    별도로 남긴다."""
    validation = (plan or {}).get("validation") or {"valid": True, "errors": []}
    issue = (plan or {}).get("issue") or {}
    fields = dict(
        session_id=session_id,
        speaker=persona_id,
        plan_id=(plan or {}).get("plan_id"),
        effective_issue_id=issue.get("issue_id"),
        effective_issue_title=issue.get("title"),
        retrieved_evidence_count=retrieved_evidence_count,
        eligible_evidence_count=(plan or {}).get("eligible_evidence_count"),
        selected_evidence_count=len((plan or {}).get("selected_evidence") or []),
        injected_planned_evidence_count=injected_evidence_count,
        selected_refs=[item.get("ref") for item in (plan or {}).get("selected_evidence") or []],
        fallback_reason=fallback_reason,
        validation_valid=validation.get("valid"),
        validation_errors=validation.get("errors"),
        elapsed_ms=elapsed_ms,
    )
    if mode == "active":
        trace_event("IDEATION_EVIDENCE_PLAN_ACTIVE", **fields)
    elif mode == "valid_empty":
        trace_event("IDEATION_EVIDENCE_PLAN_VALID_EMPTY", **fields)
    elif mode == "fallback":
        trace_event("IDEATION_EVIDENCE_PLAN_FALLBACK", level=logging.WARNING, **fields)


def _trace_evidence_plan_compliance(
    *,
    session_id: str | None,
    persona_id: str,
    mode: str,
    plan: dict,
    raw: dict,
    grounding: dict,
    generated_issue_id: str | None,
    generated_issue_title: str | None,
) -> None:
    """용준/Claude(2026-07-23, Phase 2 "Active Evidence Injection", 요청: "쟁점 정합성" —
    Phase 1에서 issue mismatch 10% 관측) — 발언 생성·grounding이 끝난 뒤 계획된 쟁점과 실제
    생성된 쟁점을 비교하고, 선택된 근거가 실제로 claims/grounding에 쓰였는지 로그로 남긴다.
    이 로그는 기록용이다 — mismatch만으로 재생성하거나 라우팅을 바꾸지 않는다(기존
    _safe_call_structured_json/_ground_and_finalize_claims의 재시도 정책과 별개 축)."""
    issue = plan.get("issue") or {}
    selected_refs = [item.get("ref") for item in plan.get("selected_evidence") or [] if item.get("ref")]
    claim_evidence_refs = sorted(
        {
            ref
            for claim in (raw.get("claims") or [])
            if isinstance(claim, dict)
            for ref in (claim.get("evidence_refs") or [])
            if isinstance(ref, str)
        }
    )
    trace_event(
        "IDEATION_EVIDENCE_PLAN_COMPLIANCE",
        session_id=session_id,
        speaker=persona_id,
        plan_id=plan.get("plan_id"),
        mode=mode,
        effective_issue_id=issue.get("issue_id"),
        effective_issue_title=issue.get("title"),
        generated_issue_id=generated_issue_id,
        generated_issue_title=generated_issue_title,
        issue_match=(generated_issue_id is None or generated_issue_id == issue.get("issue_id")),
        selected_refs=selected_refs,
        claim_evidence_refs=claim_evidence_refs,
        used_selected_evidence=bool(set(selected_refs) & set(claim_evidence_refs)),
        claim_count=len(grounding["claims"]),
        grounded_claim_count=grounding["grounded_claim_count"],
        linked_evidence_count=grounding["linked_evidence_count"],
        linked_chunk_ids=list(grounding["linked_evidence_refs"]),
    )


# 용준/Claude(2026-07-23, 요청: "사용자 정보 수집형 회의"에서 "근거 기반 자율 토론형
# 회의"로 개편) — 사용자가 실제로 결정해야 하는 주제(예산 상한/대상 지역·기관/핵심 사용자
# 우선순위/MVP 범위/보유 데이터·센서·내부 시스템 여부/일정·운영 제약)와, 문서 검색·전문가
# 판단으로 해결 가능한 일반 기술/구현 질문을 결정적 키워드로 구분한다. LLM이
# needs_user_input=true를 반환했다는 사실만으로는 사용자에게 라우팅하지 않는다 — 이
# 키워드에 걸리는 경우에만 "진짜 사용자 결정"으로 취급한다(resolve_user_input_gate 참고).
_USER_DECISION_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "budget": ("예산", "비용 상한", "예산 상한", "가용 예산", "투자 규모"),
    "target_scope": ("대상 지역", "대상 기관", "타겟 지역", "서비스 지역", "적용 지역", "대상 기관 범위"),
    "user_priority": ("핵심 사용자", "우선순위", "우선 대상", "타겟 사용자 선정"),
    "mvp_scope": ("mvp 범위", "초기 범위", "1차 범위", "mvp 우선순위", "출시 범위"),
    "data_availability": ("보유 데이터", "내부 시스템", "보유 센서", "자체 데이터", "연동 가능한 시스템"),
    "schedule_constraint": ("마감 일정", "운영 제약", "필수 일정", "고정 일정", "납기"),
}

# 주제별 결정 질문 템플릿. 코드가 결정적으로 조립한다(요청: "포괄적인 자유 입력 질문
# 금지" — 왜 지금 결정이 필요한지 + 2~3개의 구체적 선택지(장단점 포함) + 응답이 없을 때
# 적용할 권장 기본값). LLM을 다시 호출하지 않는다.
_DECISION_TEMPLATES: dict[str, dict[str, Any]] = {
    "budget": {
        "why": "예산 상한에 따라 추천할 구현 범위가 달라집니다",
        "options": [
            {"label": "저비용안", "detail": "공공 데이터·오픈소스 중심, 초기 투자 최소화"},
            {"label": "고비용안", "detail": "자체 인프라·유료 서비스 포함, 정확도·확장성 우선"},
        ],
        "default": "저비용안",
    },
    "target_scope": {
        "why": "대상 지역·기관 범위에 따라 필요한 데이터와 협력 대상이 달라집니다",
        "options": [
            {"label": "소규모 시범 지역", "detail": "1개 지역·기관에서 먼저 검증"},
            {"label": "광역 적용", "detail": "여러 지역·기관을 동시에 포괄"},
        ],
        "default": "소규모 시범 지역",
    },
    "user_priority": {
        "why": "우선 대응할 핵심 사용자층에 따라 기능 우선순위가 달라집니다",
        "options": [
            {"label": "일반 시민 우선", "detail": "접근성과 사용 편의 중심"},
            {"label": "전문 운영 인력 우선", "detail": "정확도와 관리 기능 중심"},
        ],
        "default": "일반 시민 우선",
    },
    "mvp_scope": {
        "why": "초기 MVP에 포함할 범위에 따라 개발 기간과 검증 항목이 달라집니다",
        "options": [
            {"label": "핵심 기능만", "detail": "가장 중요한 기능 1~2개만 우선 검증"},
            {"label": "확장 기능 포함", "detail": "부가 기능까지 포함해 완성도 우선"},
        ],
        "default": "핵심 기능만",
    },
    "data_availability": {
        "why": "보유 데이터·센서·내부 시스템 여부에 따라 구현 방식이 달라집니다",
        "options": [
            {"label": "공공 데이터만 활용", "detail": "별도 구축 없이 공개 데이터로 시작"},
            {"label": "자체 데이터·센서 연동", "detail": "보유·구축 예정인 시스템을 연동"},
        ],
        "default": "공공 데이터만 활용",
    },
    "schedule_constraint": {
        "why": "반드시 지켜야 하는 일정·운영 제약에 따라 우선순위가 달라집니다",
        "options": [
            {"label": "일정 준수 우선", "detail": "범위를 줄여서라도 일정 내 완료"},
            {"label": "완성도 우선", "detail": "일정보다 완성도·검증을 우선"},
        ],
        "default": "일정 준수 우선",
    },
    "product_direction_choice": {
        "why": "문서와 전문가 판단만으로는 두 방향 중 하나를 확정할 수 없습니다",
        "options": [
            {"label": "제안 A", "detail": "기획 위원이 제시한 방향"},
            {"label": "제안 B", "detail": "개발 위원이 제시한 방향"},
        ],
        "default": "제안 A",
    },
}


def extract_distinct_alternatives(
    *, messages: list[dict], active_issue_id: str | None, max_alternatives: int = 3
) -> list[dict[str, str]]:
    """세션 메시지에서 이번 쟁점(active_issue_id)에 대해 planning_expert/dev_expert가 이미
    말한 proposal(없으면 recommendation/interim_conclusion) 텍스트를 모아, 표현만 바뀐
    같은 제안은 normalize_issue_text로 하나로 합치고 실제로 서로 다른 실행 대안만 남긴다.
    새 LLM 호출이나 embedding 없이 기존 structured 필드만 재사용한다(요청: "실제 발언
    내용을 사용하고 제안 A/B로 만들지 않는다")."""
    if not active_issue_id:
        return []
    distinct: list[dict[str, str]] = []
    seen_normalized: set[str] = set()
    for message in messages:
        structured = message.get("structured") or {}
        if structured.get("active_issue_id") != active_issue_id:
            continue
        speaker = message.get("speaker_id")
        if speaker not in ("planning_expert", "dev_expert"):
            continue
        text = (
            structured.get("proposal")
            or structured.get("recommendation")
            or structured.get("interim_conclusion")
            or ""
        ).strip()
        if not text:
            continue
        normalized = normalize_issue_text(text)
        if not normalized or normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        distinct.append({"speaker": speaker, "text": text, "message_id": message.get("message_id")})
    return distinct[:max_alternatives]


def classify_user_decision_topic(
    *,
    missing_information: list[str],
    issue_title: str | None,
    near_issue_cap: bool,
    distinct_alternatives: list[dict] | None = None,
) -> str | None:
    """실제 사용자 결정이 필요한 주제인지 결정적 키워드로 분류한다. 일반 기술 접근 방법·
    공개 데이터 활용 가능성·평가 기준 해석처럼 요청에서 명시한 "사용자 질문 사유가 될 수
    없는" 주제는 여기 걸리지 않는다 — None이면 전문가 판단(또는 보완 검색)으로 해결해야
    한다는 뜻이다.

    용준/Claude(2026-07-23, 요청: "실제 전문가 대안 기반 사용자 선택 게이트") —
    near_issue_cap만으로는 더 이상 product_direction_choice를 만들지 않는다. distinct_
    alternatives(extract_distinct_alternatives 결과)가 실제로 서로 다른 실행 대안을 2개
    이상 담고 있을 때만 사용자에게 "선택"을 묻는다 — 대안이 없거나 하나뿐이면(단순 정보
    부족) 전문가 판단으로 계속 진행한다."""
    haystack = " ".join([*missing_information, issue_title or ""])
    for topic, keywords in _USER_DECISION_TOPIC_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return topic
    if near_issue_cap and distinct_alternatives is not None and len(distinct_alternatives) >= 2:
        return "product_direction_choice"
    return None


def _decision_fingerprint(issue_id: str | None, topic: str, missing_information: list[str]) -> str:
    """동일하거나 의미상 유사한 missing_information으로 반복 질문하지 않기 위한 지문
    (요청: "세션 내 질문 fingerprint 또는 reason code를 기록")."""
    key = f"{issue_id or ''}:{topic}:{'|'.join(sorted(missing_information))}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


_ALTERNATIVE_LABEL_TRAILING_RE = re.compile(
    r"(을|를|이|가|은|는|의|에서|에게|으로|로|와|과|도|만|까지|부터|"
    r"합니다|됩니다|습니다|입니다|한다|된다|해요|해야|하다|되다)+$"
)


def _short_label_from_alternative_text(text: str, max_len: int = 22) -> str:
    """용준/Claude(2026-07-23, 요청: "decision_options 라벨은 화자 이름이 아니라 실제 내용에서
    추출") — "기획 위원 제안"/"개발 위원 제안" 대신 전문가 발언 원문에서 결정론적으로 짧은
    선택지 제목을 뽑는다(새 LLM 호출 없음). 첫 문장(또는 첫 쉼표 이전)만 쓰고 조사/서술어
    종결 어미를 끝에서 떼어낸 뒤 길이를 제한한다 — 의미 요약이 아니라 표시용 축약이므로
    완벽한 제목을 보장하지 않으며, detail(원문 전체)이 항상 함께 표시되어 이를 보완한다."""
    first_clause = re.split(r"[.!?\n]", text.strip())[0]
    first_clause = first_clause.split(",")[0].strip()
    trimmed = _ALTERNATIVE_LABEL_TRAILING_RE.sub("", first_clause).strip()
    trimmed = trimmed or first_clause
    if len(trimmed) > max_len:
        trimmed = trimmed[:max_len].rstrip() + "…"
    return trimmed or "제안"


def _compose_decision_question(
    *,
    topic: str,
    issue_title: str | None,
    missing_information: list[str],
    distinct_alternatives: list[dict] | None = None,
) -> tuple[str, list[dict[str, str]], str]:
    """[왜 지금 결정이 필요한지 + 2~3개 선택지(장단점 포함) + 응답 없을 때 기본값] 형식으로
    질문을 조립한다(요청: 포괄적인 자유 입력 질문 금지). LLM을 다시 호출하지 않는다.

    topic이 "product_direction_choice"이고 distinct_alternatives(실제 전문가 발언에서 뽑은
    서로 다른 대안)가 있으면, 범용 "제안 A/제안 B" 템플릿 대신 그 실제 발언 내용으로 선택지를
    조립한다(요청: "단순히 제안 A, 제안 B라고 만들지 말고 실제 발언 내용을 사용")."""
    if topic == "product_direction_choice" and distinct_alternatives:
        options = [
            {
                "label": f"{idx + 1}. {_short_label_from_alternative_text(alt['text'])}",
                "detail": alt["text"],
            }
            for idx, alt in enumerate(distinct_alternatives)
        ]
        default_label = options[0]["label"]
        option_lines = "\n".join(f"{opt['label']} — {opt['detail']}" for opt in options)
        context_detail = issue_title or "이 쟁점"
        question = (
            f"{context_detail}에 대해 전문가들이 서로 다른 실행 대안을 제시해 문서와 전문가 판단만으로는 "
            f"하나를 확정할 수 없습니다. 다음 중 하나를 선택해 주세요.\n"
            f"{option_lines}\n"
            f"응답이 없으면 기본값({default_label} — 가장 먼저 제시된 전문가 제안)으로 진행합니다."
        )
        return question, options, default_label

    template = _DECISION_TEMPLATES.get(topic, _DECISION_TEMPLATES["product_direction_choice"])
    context_detail = missing_information[0] if missing_information else (issue_title or "이 쟁점")
    options: list[dict[str, str]] = template["options"]
    default_label = template["default"]
    option_lines = "\n".join(f"{idx + 1}. {opt['label']} — {opt['detail']}" for idx, opt in enumerate(options))
    question = (
        f"{context_detail}과 관련해 {template['why']}. 다음 중 하나를 선택해 주세요.\n"
        f"{option_lines}\n"
        f"응답이 없으면 기본값({default_label})으로 진행합니다."
    )
    return question, options, default_label


def evaluate_user_decision_requirement(
    *,
    missing_information: list[str],
    issue_title: str | None,
    issue_turn_count: int,
    max_issue_turns: int = MAX_EXPERT_TURNS_PER_ISSUE,
    distinct_alternatives: list[dict] | None = None,
) -> dict[str, Any]:
    """사용자 결정이 실제로 필요한지(예산/대상 지역·기관/우선순위/MVP 범위/보유 데이터/일정
    제약/상충하는 대안 중 선택 등)를 결정적으로 판정한다. resolve_user_input_gate가 이
    결과에 보완 검색·중복 질문 억제 로직을 더한다."""
    near_issue_cap = issue_turn_count >= max(max_issue_turns - 1, 1)
    topic = classify_user_decision_topic(
        missing_information=missing_information,
        issue_title=issue_title,
        near_issue_cap=near_issue_cap,
        distinct_alternatives=distinct_alternatives,
    )
    return {
        "user_decision_required": topic is not None,
        "decision_topic": topic,
        "blocking_reason_code": f"user_decision_required:{topic}" if topic else None,
        "near_issue_cap": near_issue_cap,
    }


def resolve_user_input_gate(
    *,
    missing_information: list[str],
    issue_id: str | None,
    issue_title: str | None,
    issue_turn_count: int,
    supplemental_attempted_issue_ids: list[str],
    asked_decision_fingerprints: list[str],
    distinct_alternatives: list[dict] | None = None,
) -> dict[str, Any]:
    """LLM이 needs_user_input을 반환했는지와 무관하게, (a) 먼저 보완 검색을 시도해야
    하는지, (b) 전문가 판단으로 자율 진행해도 되는지, (c) 정말 사용자 결정이 필요한지를
    결정적으로 정한다(요청: "결정론적 사용자 질문 게이트"). resolution_mode는
    continue_with_evidence/continue_with_expert_judgment/supplemental_retrieval/
    require_user_decision 중 하나다. 이 함수 자체는 상태를 바꾸지 않는 순수 함수다 —
    호출부(make_conv_discussion_node)가 supplemental_retrieval을 실제로 수행한 뒤 필요하면
    다시 호출한다."""
    decision = evaluate_user_decision_requirement(
        missing_information=missing_information,
        issue_title=issue_title,
        issue_turn_count=issue_turn_count,
        distinct_alternatives=distinct_alternatives,
    )
    topic = decision["decision_topic"]

    if topic is not None:
        fingerprint = _decision_fingerprint(issue_id, topic, missing_information)
        if fingerprint in asked_decision_fingerprints:
            return {
                "resolution_mode": "continue_with_expert_judgment",
                "decision_topic": topic,
                "blocking_reason_code": "duplicate_question_suppressed",
                "user_question_suppressed_reason": "same_topic_already_asked_this_session",
                "fingerprint": fingerprint,
                "near_issue_cap": decision["near_issue_cap"],
            }
        question, options, default_label = _compose_decision_question(
            topic=topic,
            issue_title=issue_title,
            missing_information=missing_information,
            distinct_alternatives=distinct_alternatives,
        )
        return {
            "resolution_mode": "require_user_decision",
            "decision_topic": topic,
            "blocking_reason_code": decision["blocking_reason_code"],
            "decision_question": question,
            "decision_options": options,
            "decision_default": default_label,
            "fingerprint": fingerprint,
            "near_issue_cap": decision["near_issue_cap"],
        }

    if issue_id and issue_id not in supplemental_attempted_issue_ids:
        return {
            "resolution_mode": "supplemental_retrieval",
            "decision_topic": None,
            "blocking_reason_code": "supplemental_retrieval_pending",
            "near_issue_cap": decision["near_issue_cap"],
        }

    return {
        "resolution_mode": "continue_with_expert_judgment",
        "decision_topic": None,
        "blocking_reason_code": "expert_judgment_fallback",
        "near_issue_cap": decision["near_issue_cap"],
    }


def make_conv_discussion_node(
    persona_id: str,
    llm_call: LLMCall,
    evidence_lookup: EvidenceLookup | None = None,
    ground_claims: ClaimGroundingFn | None = None,
    evidence_planner: "EvidencePlanningFn | None" = None,
) -> Callable[[IdeationConvState], dict]:
    """용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): 기획/개발 전문가의 "발언 턴"
    노드. 예전에는 speaks_second/discussion_stage를 빌드 시점에 고정해 "기획 1회 → 개발
    1회 → [조건부] 수정 1회"만 가능했지만, 이제 discussion_stage는 매 실행마다 state로
    계산하고(_discussion_stage_for), 다음 발언자·라운드 종료는 이 노드가 아니라 그래프
    라우터(_route_next_expert_turn)와 discussion_facilitator가 맡는다 — 이 노드는 발언
    하나를 만들고 open_issues/expert_turn_count/previous_speaker만 갱신한 뒤 그대로
    반환한다(phase를 절대 바꾸지 않는다)."""

    def node(state: IdeationConvState) -> dict:
        turn_started = time.perf_counter()
        # 용준/Claude(2026-07-22, 요청: 지정 위원 질문 후 상대 검토 코드 강제) — 이번 실행이
        # (a) "잠시만" 재개로 지정된 위원의 첫 응답인지, (b) 그 응답을 검토해야 할 반대편
        # 위원의 검토 발언인지를 state만 보고 결정적으로 판별한다(LLM 판단에 맡기지 않는다).
        # forced_next_speaker는 _route_entry가 이 노드를 강제 진입시켰을 때만 남아있고
        # (실행 즉시 리셋되므로), interjection_response_message_id가 아직 비어 있으면
        # "아직 첫 응답을 기록하지 않은 지정 위원 실행"이라는 뜻이다.
        is_interjection_first_response = (
            state.get("forced_next_speaker") == persona_id
            and state.get("interjection_target_speaker_id") is not None
            and not state.get("interjection_response_message_id")
        )
        is_required_counterpart_review = (
            persona_id == state.get("required_counterpart_speaker_id")
            and not state.get("counterpart_review_completed", True)
        )
        discussion_stage = _discussion_stage_for(state, persona_id)
        responding_to_target = _responding_to_for(state, persona_id, discussion_stage)
        responding_to_message_id = responding_to_target["message_id"] if responding_to_target else None
        responding_to_speaker_id = responding_to_target["speaker_id"] if responding_to_target else None
        if responding_to_speaker_id == persona_id:
            trace_event(
                "IDEATION_SPEAKER_REFERENCE_WARNING",
                level=logging.WARNING,
                session_id=state.get("session_id"),
                speaker=persona_id,
                target=responding_to_speaker_id,
                message=responding_to_message_id,
                reason="self_responding_target",
            )
            return {"phase": "failed", "failed_node": f"discussion__{persona_id}__self_target"}
        query = _topic_query(state, persona_id)
        evidence_started = time.perf_counter()
        runtime_scope = _runtime_scope_for(state)
        trace_event(
            "IDEATION_EVIDENCE_LOOKUP_SCOPE",
            session_id=runtime_scope["session_id"],
            speaker=persona_id,
            selected_candidate_document_id=runtime_scope["selected_candidate_document_id"],
            selected_candidate_document_id_source="runtime_graph_state",
        )
        retrieved = call_evidence_lookup(evidence_lookup, persona_id, query, runtime_scope=runtime_scope)
        # 용준/Claude(2026-07-22, RAG 근거 유실 수정): evidence_lookup 자체가 None이면(use_rag
        # 미사용 세션) "검색을 아예 안 했다"는 뜻이고, evidence_lookup은 있지만 결과가 빈
        # 배열이면 "검색은 했지만 0건"이다 — 둘 다 result_count=0으로 보이지만 원인이 다르므로
        # fallback_reason으로 구분한다(요청: 검색 실패와 0건 구분).
        fallback_reason = None
        if evidence_lookup is None:
            fallback_reason = "rag_not_configured"
        elif not retrieved:
            fallback_reason = "no_search_results"
        trace_event(
            "IDEATION_EVIDENCE_LOOKUP",
            session_id=state.get("session_id"),
            speaker=persona_id,
            role="planning" if persona_id == "planning_expert" else "technology",
            project_id=getattr(evidence_lookup, "trace_project_id", None),
            top_k=getattr(evidence_lookup, "trace_top_k", None),
            issue=state.get("active_issue_id"),
            query=sanitize_preview(query, limit=160),
            retrieved_evidence_count=len(retrieved),
            fallback_reason=fallback_reason,
            chunk_ids=[item.get("chunk_id") for item in retrieved if isinstance(item, dict) and item.get("chunk_id")],
            elapsed_ms=round((time.perf_counter() - evidence_started) * 1000, 1),
        )
        # 용준/Claude(2026-07-23, Phase 1 "Shadow Deterministic Evidence Planner") — 기존
        # retrieval 직후, prompt 조립 전에 shadow planner를 실행한다(요청 순서 그대로). plan
        # 결과는 아래 prompt 조립/LLM 호출/grounding 어디에도 전달하지 않고, trace 로그
        # (_trace_shadow_plan_created)와 shadow_history_map 갱신에만 쓰인다. evidence_planner가
        # None이면(플래그 꺼짐 등) 기존과 완전히 동일하게 동작한다.
        effective_issue = resolve_effective_issue(state, persona_id)
        shadow_plan, shadow_history_map, plan_elapsed_ms = _run_shadow_evidence_planner(
            evidence_planner=evidence_planner,
            persona_id=persona_id,
            session_id=state.get("session_id"),
            effective_issue=effective_issue,
            query=query,
            retrieved=retrieved,
            runtime_scope=runtime_scope,
            shadow_history_map=state.get("evidence_plan_shadow_history") or {},
        )
        # 용준/Claude(2026-07-23, Phase 2 "Active Evidence Injection") — 위에서 만든 plan을
        # 다시 실행하지 않고 그대로 재사용해 이번 턴 prompt/grounding에 쓸 evidence를
        # 결정한다. evidence_planner.active가 꺼져 있으면(기본값, 또는 shadow 전용) 항상
        # evidence_mode="inactive"이고 turn_evidence는 retrieved 그대로다 — 기존 동작과 100%
        # 동일하다.
        turn_evidence, evidence_mode, evidence_fallback_reason = _resolve_discussion_evidence(
            evidence_planner=evidence_planner,
            shadow_plan=shadow_plan,
            retrieved=retrieved,
        )
        if evidence_mode != "inactive":
            _trace_active_evidence_plan(
                session_id=state.get("session_id"),
                persona_id=persona_id,
                mode=evidence_mode,
                fallback_reason=evidence_fallback_reason,
                plan=shadow_plan,
                retrieved_evidence_count=len(retrieved),
                injected_evidence_count=len(turn_evidence),
                elapsed_ms=plan_elapsed_ms,
            )
        notice_parts = [
            _build_user_interjection_notice(state),
            _build_evidence_plan_notice(evidence_mode, shadow_plan),
        ]
        evidence_plan_notice = "\n\n".join(part for part in notice_parts if part)
        context = conversation_context_for(state)
        if evidence_mode in ("active", "valid_empty"):
            context = _isolate_discussion_evidence_context(context)
        speaker_name, _ = _speaker_fields(persona_id)
        responding_name = _EXPERT_ROLE_LABELS.get(responding_to_speaker_id or "", responding_to_speaker_id)
        message_id = _new_message_id()
        trace_event(
            "IDEATION_TURN_START",
            session_id=state.get("session_id"),
            phase=state.get("phase"),
            round=state.get("round"),
            issue=state.get("active_issue_id"),
            speaker=persona_id,
            speaker_name=speaker_name,
            message=message_id,
            target=responding_to_speaker_id,
            target_name=responding_name,
            responding_to_message=responding_to_message_id,
            responding_to_preview=sanitize_preview(
                responding_to_target.get("content") if responding_to_target else "", limit=200
            ),
            # retrieved_evidence_count: RAG 검색으로 찾아 이번 프롬프트에 주입할 근거 수
            # (아래 build_ideation_conv_discussion_prompt 호출에 retrieved 그대로 전달).
            retrieved_evidence_count=len(retrieved),
            fallback_reason=fallback_reason,
        )
        issue_control_active = evidence_mode in ("active", "valid_empty")
        expected_issue_id = (
            effective_issue.get("issue_id") or state.get("active_issue_id")
            if issue_control_active
            else None
        )
        prompt = build_ideation_conv_discussion_prompt(
            persona_id,
            state["notice_and_criteria"],
            state["user_idea"],
            turn_evidence,
            context,
            speaks_second=(discussion_stage == "response"),
            discussion_stage=discussion_stage,
            active_issue_id=expected_issue_id or state.get("active_issue_id"),
            open_issues=state.get("open_issues") or [],
            resolved_issues=state.get("resolved_issues") or [],
            current_speaker={"speaker_id": persona_id, "role_name": speaker_name},
            responding_to_message=(
                {
                    "speaker_id": responding_to_speaker_id,
                    "role_name": responding_name,
                    "message_id": responding_to_message_id,
                    "spoken_text": responding_to_target.get("content", ""),
                }
                if responding_to_target
                else {}
            ),
            evidence_plan_notice=evidence_plan_notice,
            # 가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입) — 순수 추가 인자.
            application_form_items=state.get("application_form_items") or None,
        )
        evidence_claim_types_by_ref = {
            item["ref"]: item["claim_type"]
            for item in turn_evidence
            if isinstance(item, dict)
            and isinstance(item.get("ref"), str)
            and item.get("claim_type") in ("document_fact", "user_provided_fact")
        }
        def validate(raw: dict, _stage: str = discussion_stage) -> str | None:
            _repair_evaluative_expert_judgment_claim(raw, evidence_claim_types_by_ref)
            return _validate_discussion_response(
                raw,
                _stage,
                current_speaker_id=persona_id,
                responding_to_speaker_id=responding_to_speaker_id,
                responding_to_content=(responding_to_target.get("content") if responding_to_target else None),
                require_user_question_focus=is_interjection_first_response,
                expected_issue_id=expected_issue_id,
                expected_issue_title=effective_issue.get("title"),
                require_issue_content_focus=effective_issue.get("source") != "user_interjection",
                evidence_claim_types_by_ref=evidence_claim_types_by_ref,
            )

        if evidence_mode == "valid_empty" and ground_claims is not None:
            # Evidence-first 계약: Planner가 적격 근거 없음(valid empty)을 확정한 상태에서
            # LLM의 자유로운 전문가 의견을 생성하지 않는다. 제어 메시지만 남겨 진행자가
            # 다음 쟁점으로 이동하게 하며, 불필요한 API 호출도 사용하지 않는다.
            raw = _evidence_unavailable_discussion_response(
                persona_id=persona_id,
                state=state,
                discussion_stage=discussion_stage,
                responding_to_message_id=responding_to_message_id,
                responding_to_content=(responding_to_target.get("content") if responding_to_target else None),
            )
            ok = True
            attempts = 0
            trace_event(
                "IDEATION_EVIDENCE_FIRST_TURN_SKIPPED",
                session_id=state.get("session_id"),
                speaker=persona_id,
                issue=effective_issue.get("issue_id"),
                reason=(shadow_plan or {}).get("empty_plan_reason", "no_eligible_evidence"),
                llm_call_skipped=True,
            )
        else:
            raw, ok, attempts = _safe_call_structured_json(
                llm_call,
                prompt,
                validate,
                f"discussion__{persona_id}",
                retry_note_for=_discussion_retry_note,
            )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            raw = _safe_discussion_fallback(
                persona_id=persona_id,
                state=state,
                discussion_stage=discussion_stage,
                responding_to_message_id=responding_to_message_id,
                responding_to_content=(responding_to_target.get("content") if responding_to_target else None),
            )
            trace_event(
                "IDEATION_STRUCTURED_RESPONSE_SAFE_FALLBACK",
                session_id=state.get("session_id"),
                node=f"discussion__{persona_id}",
                attempts=attempts,
                speaker=persona_id,
                issue=raw["active_issue_id"],
                next_speaker=raw["recommended_next_speaker"],
                claim_type="expert_judgment",
            )

        raw, grounding, used = _ground_and_finalize_claims(
            persona_id=persona_id,
            raw=raw,
            retrieved=turn_evidence,
            prompt=prompt,
            llm_call=llm_call,
            validate=validate,
            used=used,
            ground_claims_fn=ground_claims,
            require_linked_evidence=(evidence_mode == "active"),
        )

        stance = raw.get("stance")
        if stance not in _VALID_DISCUSSION_STANCES:
            stance = "보완"
        message_type = _STANCE_TO_MESSAGE_TYPE.get(stance, "opinion")

        known_ids = {m["message_id"] for m in state["messages"]}
        judgment = raw.get("judgment", "")
        reason = raw.get("reason", "")
        suggestion = raw.get("suggestion", "")
        interim_conclusion = raw.get("interim_conclusion", "")
        responding_to = raw.get("responding_to") or None
        agreement = raw.get("agreement") or None
        concern = raw.get("concern") or None
        revision_text = raw.get("revision") or None
        confirmed_items = _as_string_list(raw.get("confirmed"))
        unconfirmed_items = _as_string_list(raw.get("unconfirmed"))

        # 상호참조는 LLM 호출 전에 코드가 결정하고 프롬프트/검증/저장에 같은 값을 쓴다.
        referenced_ids_raw = list(raw.get("referenced_message_ids") or [])
        if responding_to_message_id and responding_to_message_id not in referenced_ids_raw:
            referenced_ids_raw.append(responding_to_message_id)

        # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — 다음 발언자/쟁점 판단
        # 필드. recommended_next_speaker는 여기서 정규화만 하고, 실제 라우팅 검증은
        # _route_next_expert_turn(그래프 라우터)이 한다(LLM 추천을 그대로 신뢰하지 않는다).
        generated_issue_id_raw = (raw.get("active_issue_id") or "").strip() or None
        generated_issue_title_raw = (raw.get("active_issue_title") or "").strip() or None
        if shadow_plan is not None and generated_issue_id_raw is not None:
            # 용준/Claude(2026-07-23, Phase 1 "Shadow Deterministic Evidence Planner", 요청:
            # 첫 턴 issue mismatch rate 측정) — retrieval이 실제로 겨냥한 쟁점(effective_issue)과
            # LLM이 자체적으로 반환한 쟁점(generated_issue_id_raw)이 다르면 로그만 남긴다.
            # mismatch가 발생해도 발언을 재생성하거나 라우팅을 바꾸지 않는다(Phase 1 범위 밖).
            if generated_issue_id_raw != shadow_plan["issue"]["issue_id"]:
                trace_event(
                    "IDEATION_EVIDENCE_PLAN_SHADOW_ISSUE_MISMATCH",
                    session_id=state.get("session_id"),
                    speaker=persona_id,
                    plan_id=shadow_plan.get("plan_id"),
                    effective_issue_id=shadow_plan["issue"]["issue_id"],
                    effective_issue_title=shadow_plan["issue"]["title"],
                    generated_issue_id=generated_issue_id_raw,
                    generated_issue_title=generated_issue_title_raw,
                    issue_match=False,
                )
        # 용준/Claude(2026-07-23, Phase 2 "Active Evidence Injection", 요청: "쟁점 정합성"
        # compliance 로그) — evidence_mode가 "inactive"가 아닐 때만(discussion 플래그가 켜져
        # planner를 실제로 참고한 턴에만) 남긴다. mismatch/미사용이 확인돼도 여기서 발언을
        # 재생성하거나 라우팅을 바꾸지 않는다 — 순수 기록이다.
        if evidence_mode != "inactive" and shadow_plan is not None:
            _trace_evidence_plan_compliance(
                session_id=state.get("session_id"),
                persona_id=persona_id,
                mode=evidence_mode,
                plan=shadow_plan,
                raw=raw,
                grounding=grounding,
                generated_issue_id=generated_issue_id_raw,
                generated_issue_title=generated_issue_title_raw,
            )
        candidate_issue_id = generated_issue_id_raw or _fallback_issue_id(state)
        candidate_issue_title = generated_issue_title_raw or candidate_issue_id
        issue_dedup = resolve_issue_duplicate(
            candidate_issue_id=candidate_issue_id,
            candidate_issue_title=candidate_issue_title,
            current_active_issue_id=state.get("active_issue_id"),
            open_issues=state.get("open_issues") or [],
            resolved_issues=state.get("resolved_issues") or [],
            resolved_topics=state.get("resolved_topics") or [],
        )
        if issue_dedup["duplicate"]:
            trace_event(
                "IDEATION_ISSUE_DUPLICATE_SUPPRESSED",
                session_id=state.get("session_id"),
                candidate_issue_title=candidate_issue_title,
                canonical_family=issue_dedup["canonical_family"],
                duplicate_of_issue_id=issue_dedup["duplicate_of_issue_id"],
                duplicate_source=issue_dedup["duplicate_source"],
                reason_code="semantic_duplicate_issue",
            )
            if issue_dedup["rotated"]:
                trace_event(
                    "IDEATION_ISSUE_ROTATED",
                    session_id=state.get("session_id"),
                    previous_issue_id=issue_dedup["duplicate_of_issue_id"],
                    previous_issue_family=resolve_canonical_issue_family(candidate_issue_title),
                    next_issue_id=issue_dedup["issue_id"],
                    next_issue_family=issue_dedup["canonical_family"],
                    rotation_reason=f"duplicate_source:{issue_dedup['duplicate_source']}",
                    skipped_duplicate_count=1,
                )
        active_issue_id = issue_dedup["issue_id"]
        active_issue_title = issue_dedup["issue_title"]
        active_issue_family = issue_dedup["canonical_family"]
        new_information = _as_string_list(raw.get("new_information"))
        proposal = raw.get("proposal") or None
        changed_position = bool(raw.get("changed_position"))
        needs_counterpart_response = raw.get("needs_counterpart_response")
        needs_counterpart_response = True if needs_counterpart_response is None else bool(needs_counterpart_response)
        recommended_next_speaker = raw.get("recommended_next_speaker")
        if recommended_next_speaker not in _VALID_NEXT_SPEAKERS:
            recommended_next_speaker = _DISCUSSION_COUNTERPART.get(persona_id, "ideation_facilitator")
        issue_resolved = bool(raw.get("issue_resolved"))
        # 용준/Claude(2026-07-22, 요청 7번: "첫 답변자가 issue_resolved=true를 반환해도 상대
        # 검토 전에는 최종 resolved로 확정 금지") — 지정 위원의 첫 응답 자신이 이 쟁점을
        # 해결됐다고 판단해도, 코드가 강제로 open 상태를 유지한다(active_issue_id도 아래에서
        # 계속 살아있게 된다) — 상대가 검토한 뒤 자신도 issue_resolved=true를 반환해야만 실제로
        # resolved_issues로 옮겨간다. 이 override는 message 생성 이전에 적용되므로
        # structured["issue_resolved"]도 함께 False로 기록된다(화면/로그가 "아직 미확정"이라는
        # 실제 상태와 다른 값을 보여주지 않도록 하기 위함 — raw 원본 값을 숨기는 것이 목적이
        # 아니라, "이 시점에 회의가 실제로 이 쟁점을 닫았는가"를 정확히 반영하려는 것이다).
        if is_interjection_first_response and issue_resolved:
            issue_resolved = False
        llm_requested_user_input = bool(raw.get("needs_user_input"))
        llm_user_question = (raw.get("user_question") or None) if llm_requested_user_input else None

        # 용준/Claude(2026-07-22, 요청: 반복되는 근거 없는 의견을 사용자 질문으로 전환) —
        # evidence_status="ungrounded"(document_fact 인용 실패) 외에도 linked_evidence_count=0인
        # 턴이나 expert_judgment_only 상태, 같은 missing_information이 별다른 진전 없이
        # "쟁점이 바뀌지 않은 채" 반복되면(2회 연속) 전문가 둘이서 끝없이 같은 판단을
        # 되풀이하는 대신 반복을 감지한다(요청 4번).
        #
        # ground_claims_fn이 없으면(use_rag=False 등 RAG 자체를 쓰지 않는 세션) grounding은
        # 항상 _EMPTY_GROUNDING(linked_evidence_count=0)이다 — 이건 "근거 연결이 반복 실패"가
        # 아니라 "애초에 RAG를 검증하지 않기로 한 세션"이므로, 이 반복 감지 자체를 건너뛴다
        # (그렇지 않으면 RAG 미사용 세션마다 두 번째 발언에서 곧바로 사용자에게 되묻게 된다).
        #
        # new_information(발언 스키마 필수 필드)의 어휘 반복 여부는 로그로만 남기고(같은
        # 화자가 라운드마다 정형화된 문구를 쓰는 정상적인 경우까지 오탐하기 쉬워 라우팅
        # 트리거로 쓰지 않는다) 사용자 전환 조건에서는 제외한다.
        ground_claims_configured = ground_claims is not None
        issue_changed = active_issue_id != state.get("active_issue_id")
        missing_info_normalized = sorted({m.strip() for m in grounding["missing_information"] if m and m.strip()})
        new_information_text = " ".join(new_information)
        spoken_text = raw.get("spoken_text", "")
        restatement_matches = _recent_issue_restatement_matches(
            state.get("messages") or [],
            issue_id=active_issue_id,
            spoken_text=spoken_text,
        )
        # 한 발언과만 겹치는 것은 정상적인 반론·응답일 수 있다. 최근 동일 쟁점 발언 두 개
        # 이상과 핵심어가 겹치거나, 구조화 응답 실패 후 서버 fallback이 만들어졌을 때만
        # "실질적 새 정보 없음"으로 확정한다.
        # 세 번째 발언까지 생성한 뒤 반복을 잡으면 사용자는 이미 같은 스트리밍 초안을 본다.
        # 핵심어 3개 이상·유사도 0.45를 통과한 직전 동일 쟁점 발언이 하나라도 있고 실제 입장
        # 변경이 없다면 두 번째 발언에서 즉시 진행자에게 넘긴다.
        semantic_restatement = bool(restatement_matches) and not changed_position
        safe_fallback_triggered = bool(raw.get("safe_fallback"))
        evidence_exhausted = (
            evidence_mode == "active"
            and _issue_evidence_exhausted(
                state.get("messages") or [],
                issue_id=active_issue_id,
                current_speaker_id=persona_id,
                current_linked_chunk_ids=grounding["linked_evidence_refs"],
            )
        )

        if not ground_claims_configured:
            consecutive_zero_linked_turns = 0
            consecutive_expert_judgment_only_turns = 0
            consecutive_repeated_missing_information_turns = 0
            consecutive_no_new_information_turns = 0
        elif issue_changed:
            zero_linked_turn = grounding["linked_evidence_count"] == 0
            consecutive_zero_linked_turns = 1 if zero_linked_turn else 0
            expert_judgment_only_turn = grounding["evidence_status"] == "expert_judgment_only"
            consecutive_expert_judgment_only_turns = 1 if expert_judgment_only_turn else 0
            consecutive_repeated_missing_information_turns = 0
            consecutive_no_new_information_turns = 0
        else:
            prev_zero_linked = state.get("consecutive_zero_linked_turns", 0)
            prev_expert_judgment_only = state.get("consecutive_expert_judgment_only_turns", 0)
            prev_missing_streak = state.get("consecutive_repeated_missing_information_turns", 0)
            prev_no_new_info_streak = state.get("consecutive_no_new_information_turns", 0)
            prev_missing_information = state.get("last_missing_information", [])
            prev_new_information_text = state.get("last_new_information_text", "")

            zero_linked_turn = grounding["linked_evidence_count"] == 0
            consecutive_zero_linked_turns = prev_zero_linked + 1 if zero_linked_turn else 0
            expert_judgment_only_turn = grounding["evidence_status"] == "expert_judgment_only"
            consecutive_expert_judgment_only_turns = prev_expert_judgment_only + 1 if expert_judgment_only_turn else 0
            missing_information_repeated = (
                bool(missing_info_normalized) and missing_info_normalized == prev_missing_information
            )
            consecutive_repeated_missing_information_turns = prev_missing_streak + 1 if missing_information_repeated else 0
            no_new_information_turn = (
                semantic_restatement
                or safe_fallback_triggered
                or (
                    bool(prev_new_information_text)
                    and _looks_like_restatement(new_information_text, prev_new_information_text)
                )
            )
            consecutive_no_new_information_turns = prev_no_new_info_streak + 1 if no_new_information_turn else 0

        _REPETITION_TURN_THRESHOLD = 2
        # 근거를 인용하지 않은 expert_judgment는 정상적인 전문가 역할이며 사용자 질문 사유가
        # 아니다. 실제로 같은 누락 정보나 같은 새 정보가 반복될 때만 반복으로 취급한다.
        repetition_triggered = ground_claims_configured and (
            consecutive_repeated_missing_information_turns >= _REPETITION_TURN_THRESHOLD
            or consecutive_no_new_information_turns >= _REPETITION_TURN_THRESHOLD
            or semantic_restatement
            or safe_fallback_triggered
            or evidence_exhausted
        )
        if repetition_triggered:
            repetition_reason = (
                "structured_response_safe_fallback"
                if safe_fallback_triggered
                else "evidence_exhausted"
                if evidence_exhausted
                else "semantic_restatement"
            )
            trace_event(
                "IDEATION_INTRA_ISSUE_REPETITION_DETECTED",
                session_id=state.get("session_id"),
                speaker=persona_id,
                issue=active_issue_id,
                reason=repetition_reason,
                matched_messages=restatement_matches,
                linked_chunk_ids=grounding["linked_evidence_refs"],
                evidence_exhausted=evidence_exhausted,
                consecutive_no_new_information_turns=consecutive_no_new_information_turns,
            )

        # 용준/Claude(2026-07-23, 요청: "사용자 정보 수집형 회의"에서 "근거 기반 자율
        # 토론형 회의"로 개편) — LLM이 needs_user_input=true를 반환했든, 문서 근거 연결이
        # 실패했든(ungrounded), 같은 판단이 반복됐든, 실제로 사용자에게 물어도 되는지는
        # 항상 결정론적 게이트(resolve_user_input_gate)가 최종 결정한다(요청: "LLM이
        # needs_user_input=true라고 출력했다는 이유만으로 사용자에게 라우팅하지 마세요").
        # _is_actionable_user_decision_question(LLM이 고른 질문 문구 자체의 신호)과
        # classify_user_decision_topic(missing_information/쟁점 제목 신호)을 함께 봐서 실제
        # 사용자 결정 주제인지 판정하고, 아니면 먼저 보완 검색을 1회 시도한 뒤에도 안 되면
        # 전문가 판단으로 자율 진행한다.
        insufficiency_detected = (
            llm_requested_user_input or grounding["evidence_status"] == "ungrounded" or repetition_triggered
        )
        needs_user_input = False
        user_question: str | None = None
        resolution_mode = "continue_with_evidence"
        blocking_reason_code: str | None = None
        decision_topic: str | None = None
        supplemental_retrieval_attempted_this_turn = False
        supplemental_evidence_count = 0
        autonomous_assumption_note: str | None = None
        decision_options: list[dict[str, str]] = []
        gate: dict[str, Any] = {}

        supplemental_attempted_issue_ids = list(state.get("supplemental_retrieval_issue_ids") or [])
        asked_decision_fingerprints = list(state.get("asked_decision_fingerprints") or [])

        if insufficiency_detected:
            issue_turn_count = next(
                (
                    issue.get("turns", 0)
                    for issue in (state.get("open_issues") or [])
                    if issue.get("issue_id") == active_issue_id
                ),
                0,
            )
            gate_missing_information = list(missing_info_normalized)
            if (
                llm_requested_user_input
                and _is_actionable_user_decision_question(llm_user_question)
                and llm_user_question
                and llm_user_question not in gate_missing_information
            ):
                gate_missing_information = gate_missing_information + [llm_user_question]

            # 용준/Claude(2026-07-23, 요청: "실제 전문가 대안 기반 사용자 선택 게이트") —
            # 이번 쟁점에서 이미 나온 기획/개발 위원의 실행 대안(proposal 등)에 이번 턴 자신의
            # proposal도 더해 서로 다른 대안이 실제로 몇 개인지 계산한다. 이 값이 있어야
            # classify_user_decision_topic이 near_issue_cap만으로 product_direction_choice를
            # 만들지 않고 "진짜 대안이 2개 이상인지"를 검사할 수 있다. 새 LLM 호출은 없다.
            distinct_alternatives = extract_distinct_alternatives(
                messages=state["messages"], active_issue_id=active_issue_id
            )
            current_turn_alt_text = (proposal or interim_conclusion or "").strip()
            if current_turn_alt_text:
                normalized_current_alt = normalize_issue_text(current_turn_alt_text)
                if normalized_current_alt and normalized_current_alt not in {
                    normalize_issue_text(alt["text"]) for alt in distinct_alternatives
                }:
                    distinct_alternatives = distinct_alternatives + [
                        {"speaker": persona_id, "text": current_turn_alt_text, "message_id": message_id}
                    ]

            gate = resolve_user_input_gate(
                missing_information=gate_missing_information,
                issue_id=active_issue_id,
                issue_title=active_issue_title,
                issue_turn_count=issue_turn_count,
                supplemental_attempted_issue_ids=supplemental_attempted_issue_ids,
                asked_decision_fingerprints=asked_decision_fingerprints,
                distinct_alternatives=distinct_alternatives,
            )
            resolution_mode = gate["resolution_mode"]
            blocking_reason_code = gate["blocking_reason_code"]
            decision_topic = gate.get("decision_topic")

            if evidence_exhausted and resolution_mode == "supplemental_retrieval":
                # 같은 쟁점에서 두 전문가가 이미 동일 근거 세트를 모두 사용했다. 빈 누락
                # 정보로 "문제 정의" 같은 범용 검색을 한 번 더 실행해도 새 판단 근거가
                # 생기지 않으므로 진행자에게 넘겨 다음 쟁점으로 이동한다.
                resolution_mode = "continue_with_expert_judgment"
                blocking_reason_code = "evidence_set_exhausted"
                decision_topic = None

            if resolution_mode == "supplemental_retrieval" and active_issue_id:
                # 문서에서 추가로 확인할 가치가 있는 구체적인 누락 정보가 있으면, 현재
                # 세션의 target/criteria 문서를 대상으로 보완 검색을 최대 1회 수행한다(요청
                # 3번). 이 쟁점에는 다시 시도하지 않도록 즉시 기록한다(쟁점당 최대 1회).
                supplemental_retrieval_attempted_this_turn = True
                supplemental_attempted_issue_ids = supplemental_attempted_issue_ids + [active_issue_id]
                supplemental_query = " · ".join(missing_info_normalized[:3]) or (active_issue_title or query)
                supplemental_raw_evidence = call_evidence_lookup(
                    evidence_lookup, persona_id, supplemental_query, runtime_scope=runtime_scope
                )
                existing_chunk_ids = {
                    item.get("chunk_id") for item in turn_evidence if isinstance(item, dict) and item.get("chunk_id")
                }
                raw_new_evidence_items = [
                    item
                    for item in supplemental_raw_evidence
                    if isinstance(item, dict) and item.get("chunk_id") not in existing_chunk_ids
                ]
                planned_supplemental_evidence, supplemental_plan = _plan_supplemental_evidence(
                    evidence_planner=evidence_planner,
                    persona_id=persona_id,
                    effective_issue=effective_issue,
                    supplemental_query=supplemental_query,
                    retrieved=raw_new_evidence_items,
                    runtime_scope=runtime_scope,
                )
                # 최초 turn_evidence의 ref namespace는 그대로 보존하고, 보완 근거에만 새 ref를
                # 부여한다. 기존 claim의 E번호가 보완 검색 때문에 다른 chunk를 가리키는 일을
                # 차단한다.
                new_evidence_items = [
                    {**item, "ref": f"E{len(turn_evidence) + index + 1}"}
                    for index, item in enumerate(planned_supplemental_evidence)
                ]
                supplemental_evidence_count = len(new_evidence_items)
                grounding_improved = False
                if new_evidence_items and ground_claims is not None:
                    # 새로 찾은 근거가 이번 발언의 미검증 document_fact 주장을 실제로
                    # 뒷받침하는지는 항상 claim_grounding(주입된 ground_claims)이 판정한다
                    # (요청: "보완 검색으로 찾은 내용은 기존과 동일하게 Planner 선별과 claim
                    # grounding을 통과해야 합니다") — 여기서는 아직 인용되지 않은 새 근거를
                    # 미검증 document_fact 주장에 시도 삼아 붙여줄 뿐이고, 관련성 검증을
                    # 통과하지 못하면 그대로 unsupported로 남는다. 문서에서 찾을 수 없으면
                    # document_fact를 만들지 않는다 — claims 자체를 조작하지 않고 grounding
                    # 결과만 재판정한다.
                    merged_evidence = list(turn_evidence) + new_evidence_items
                    new_refs = [item["ref"] for item in new_evidence_items]
                    unsupported_claim_ids = {c["claim_id"] for c in grounding["unsupported_claims"]}
                    augmented_claims = []
                    for claim in raw.get("claims") or []:
                        if not isinstance(claim, dict):
                            continue
                        claim_copy = dict(claim)
                        # claim_id가 없으면 unsupported_claim_ids와 매칭될 수 없다(방어적 —
                        # 실제로는 항상 claim_grounding._normalize_claims가 채운 claim_id를
                        # 그대로 들고 있다). evidence_refs가 비어 있던 document_fact뿐 아니라,
                        # 기존 ref가 있었지만 검증에 실패했던 document_fact도 새 ref를
                        # 추가로 시도해 본다(기존 ref는 남겨 둔다 — 다른 ref가 맞을 수도
                        # 있으므로 지우지 않는다).
                        if claim_copy.get("claim_type") == "document_fact" and (
                            not claim_copy.get("evidence_refs")
                            or claim_copy.get("claim_id") in unsupported_claim_ids
                        ):
                            existing_refs = claim_copy.get("evidence_refs") or []
                            claim_copy["evidence_refs"] = list(dict.fromkeys([*existing_refs, *new_refs]))
                        augmented_claims.append(claim_copy)
                    previous_linked_count = grounding["linked_evidence_count"]
                    previous_grounded_count = grounding["grounded_claim_count"]
                    retry_grounding = ground_claims(persona_id, augmented_claims, merged_evidence)
                    grounding_improved = (
                        retry_grounding["linked_evidence_count"] > previous_linked_count
                        and retry_grounding["grounded_claim_count"] > previous_grounded_count
                    )
                    if grounding_improved:
                        raw = {**raw, "claims": augmented_claims}
                        grounding = retry_grounding
                        turn_evidence = merged_evidence
                        retrieved = retrieved + new_evidence_items
                        missing_info_normalized = sorted(
                            {m.strip() for m in grounding["missing_information"] if m and m.strip()}
                        )
                trace_event(
                    "IDEATION_SUPPLEMENTAL_RETRIEVAL",
                    session_id=state.get("session_id"),
                    speaker=persona_id,
                    issue=active_issue_id,
                    missing_information=missing_info_normalized,
                    query=sanitize_preview(supplemental_query, limit=160),
                    raw_new_evidence_count=len(raw_new_evidence_items),
                    new_evidence_count=supplemental_evidence_count,
                    planner_selected_evidence_count=supplemental_evidence_count,
                    planner_empty_reason=(supplemental_plan or {}).get("empty_plan_reason"),
                    grounding_improved=grounding_improved,
                    evidence_status_after=grounding["evidence_status"],
                )
                if not grounding_improved:
                    gate = resolve_user_input_gate(
                        missing_information=missing_info_normalized,
                        issue_id=active_issue_id,
                        issue_title=active_issue_title,
                        issue_turn_count=issue_turn_count,
                        supplemental_attempted_issue_ids=supplemental_attempted_issue_ids,
                        asked_decision_fingerprints=asked_decision_fingerprints,
                        distinct_alternatives=distinct_alternatives,
                    )
                    resolution_mode = gate["resolution_mode"]
                    blocking_reason_code = gate["blocking_reason_code"]
                    decision_topic = gate.get("decision_topic")
                else:
                    resolution_mode = "continue_with_evidence"
                    blocking_reason_code = None
                    decision_topic = None

            if resolution_mode == "require_user_decision":
                needs_user_input = True
                user_question = gate["decision_question"]
                decision_options = gate.get("decision_options", [])
                recommended_next_speaker = "user"
                needs_counterpart_response = False
                asked_decision_fingerprints = asked_decision_fingerprints + [gate["fingerprint"]]
                trace_event(
                    "IDEATION_USER_DECISION_REQUIRED",
                    session_id=state.get("session_id"),
                    speaker=persona_id,
                    issue=active_issue_id,
                    missing_information=missing_info_normalized,
                    decision_topic=decision_topic,
                    decision_reason=blocking_reason_code,
                    blocking_reason_code=blocking_reason_code,
                    decision_question=sanitize_preview(user_question, limit=300),
                    decision_options=[opt.get("label") for opt in decision_options],
                    option_count=len(decision_options),
                    option_labels=[opt.get("label") for opt in decision_options],
                    source_message_ids=[alt.get("message_id") for alt in distinct_alternatives if alt.get("message_id")],
                    default_option=gate.get("decision_default"),
                    alternatives_are_distinct=(decision_topic != "product_direction_choice" or len(distinct_alternatives) >= 2),
                    next_speaker="user",
                    user_decision_required=True,
                )
            else:
                needs_user_input = False
                user_question = None
                if repetition_triggered:
                    # 같은 판단이 반복돼 더 진전이 없다 — 진행자가 정리하도록 넘긴다(기존
                    # 반복 감지 동작과 동일).
                    recommended_next_speaker = "ideation_facilitator"
                elif recommended_next_speaker == "user":
                    recommended_next_speaker = _DISCUSSION_COUNTERPART.get(persona_id, "ideation_facilitator")
                needs_counterpart_response = recommended_next_speaker != "ideation_facilitator"
                if blocking_reason_code == "duplicate_question_suppressed":
                    trace_event(
                        "IDEATION_USER_QUESTION_SUPPRESSED",
                        session_id=state.get("session_id"),
                        speaker=persona_id,
                        issue=active_issue_id,
                        reason=gate.get("user_question_suppressed_reason", "duplicate_question_suppressed"),
                        missing_information=missing_info_normalized,
                    )
                elif gate.get("near_issue_cap") and len(distinct_alternatives) < 2:
                    # 용준/Claude(2026-07-23, 요청: "대안이 부족해 질문하지 않은 경우") —
                    # 발언 상한에 근접했지만 실제로 서로 다른 실행 대안이 2개 미만이면
                    # product_direction_choice를 강제로 만들지 않는다(요청: "near_issue_cap
                    # 이어도 대안이 없으면 만들지 않음"). 왜 묻지 않았는지도 로그로 남긴다.
                    trace_event(
                        "IDEATION_USER_DECISION_SKIPPED",
                        session_id=state.get("session_id"),
                        speaker=persona_id,
                        issue_id=active_issue_id,
                        canonical_family=active_issue_family,
                        reason_code="no_distinct_alternatives",
                        candidate_option_count=len(distinct_alternatives),
                    )
                    autonomous_assumption_note = (
                        f"{persona_id}: '{active_issue_title}'은 서로 다른 실행 대안이 충분히 확인되지 않아 "
                        f"전문가 판단(가정)으로 진행합니다 — {judgment or interim_conclusion}"
                    )
                    trace_event(
                        "IDEATION_AUTONOMOUS_RESOLUTION",
                        session_id=state.get("session_id"),
                        speaker=persona_id,
                        issue=active_issue_id,
                        missing_information=missing_info_normalized,
                        resolution_mode=resolution_mode,
                        blocking_reason_code=blocking_reason_code,
                        supplemental_retrieval_attempted=supplemental_retrieval_attempted_this_turn,
                        supplemental_evidence_count=supplemental_evidence_count,
                        next_speaker=recommended_next_speaker,
                        user_decision_required=False,
                    )
                else:
                    autonomous_assumption_note = (
                        f"{persona_id}: '{active_issue_title}'은 문서 근거만으로 확정할 수 없어 "
                        f"전문가 판단(가정)으로 진행합니다 — {judgment or interim_conclusion}"
                    )
                    trace_event(
                        "IDEATION_AUTONOMOUS_RESOLUTION",
                        session_id=state.get("session_id"),
                        speaker=persona_id,
                        issue=active_issue_id,
                        missing_information=missing_info_normalized,
                        resolution_mode=resolution_mode,
                        blocking_reason_code=blocking_reason_code,
                        supplemental_retrieval_attempted=supplemental_retrieval_attempted_this_turn,
                        supplemental_evidence_count=supplemental_evidence_count,
                        next_speaker=recommended_next_speaker,
                        user_decision_required=False,
                    )
        next_action = "await_user_input" if needs_user_input else "continue_discussion"

        content = _compose_discussion_content(raw.get("spoken_text", ""))
        message = _build_message(
            persona_id=persona_id,
            round_number=state["round"],
            message_type=message_type,
            content=content,
            referenced_message_ids=_referenced_ids(referenced_ids_raw, known_ids),
            # 용준/Claude(2026-07-22, RAG 근거 유실 수정): make_conv_question_node와 동일한
            # 이유로 raw.get("evidence") 대신 retrieved(이번 턴 RAG 검색 결과, 프롬프트에
            # 이미 주입됨)를 그대로 저장한다.
            evidence=retrieved,
            grounding=grounding,
            # 용준/Claude(2026-07-21, 프런트 상세보기): content(위)는 하위 호환을 위해 그대로
            # 유지하고, 필드별 구조를 선택 정보로 추가한다 — 프런트가 판단/제안만 기본 노출하고
            # 근거/확정/미확정을 "상세 보기"로 접을 수 있게 한다(요청 9번). content를 대체하는
            # 것이 아니라 순수 추가다.
            structured={
                "judgment": judgment,
                "reason": reason,
                "suggestion": suggestion,
                "interim_conclusion": interim_conclusion,
                "responding_to": responding_to,
                "agreement": agreement,
                "concern": concern,
                "revision": revision_text,
                "confirmed": confirmed_items,
                "unconfirmed": unconfirmed_items,
                "responding_to_message_id": responding_to_message_id,
                "responding_to_speaker_id": responding_to_speaker_id,
                # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — 다음 발언자/쟁점
                # 판단 근거. _route_next_expert_turn과 "잠시만" 재개(reply_to_interjection)가
                # 그대로 참조한다.
                "active_issue_id": active_issue_id,
                "active_issue_title": active_issue_title,
                "new_information": new_information,
                "proposal": proposal,
                "changed_position": changed_position,
                "needs_counterpart_response": needs_counterpart_response,
                "recommended_next_speaker": recommended_next_speaker,
                "issue_resolved": issue_resolved,
                "needs_user_input": needs_user_input,
                "user_question": user_question,
                # 용준/Claude(2026-07-23, 요청: 결정론적 사용자 질문 게이트 진단 필드) —
                # resolve_user_input_gate가 실제로 무엇을 판단했는지 구조화 상태에 남긴다
                # (요청: "다음 진단 필드를 구조화 상태나 로그에 남기세요").
                "user_decision_required": needs_user_input,
                "resolution_mode": resolution_mode,
                "blocking_reason_code": blocking_reason_code,
                "decision_topic": decision_topic,
                "decision_options": decision_options,
                "assumptions": [autonomous_assumption_note] if autonomous_assumption_note else [],
                "supplemental_retrieval_attempted": supplemental_retrieval_attempted_this_turn,
                "supplemental_evidence_count": supplemental_evidence_count,
                # 용준/Claude(2026-07-22, 요청: 반복 방지 결과를 로그·다음 라우팅이 참조할 수
                # 있게 명시적으로 남긴다) — needs_user_input과 항상 일치하는 파생값이지만,
                # "왜 지금 사용자에게 넘기는지"를 별도 필드로 노출해 로그에서 바로 읽을 수 있다.
                "next_action": next_action,
                # 용준/Claude(2026-07-22, 요청: 진행자가 검증된 주장만 사실로 요약하도록
                # 제한) — discussion_facilitator 프롬프트는 planning_position/
                # development_review를 이 dict 그대로 JSON 직렬화해 받는다(prompt_loader.py
                # build_ideation_conv_discussion_facilitator_prompt). evidence_status/
                # unsupported_claims를 함께 넘겨 진행자가 "문서로 확인된 내용"과 "근거가
                # 부족한 내용"을 구분해서 정리하도록 한다 — 검증 안 된 주장을 새 사실처럼
                # 정리하지 못하게 막는 1차 방어선(2차는 아래 프롬프트 지시문).
                "evidence_status": grounding["evidence_status"],
                "linked_evidence_refs": grounding["linked_evidence_refs"],
                "unsupported_claims": [c["text"] for c in grounding["unsupported_claims"]],
                "missing_information": grounding["missing_information"],
                # 서버가 만든 구조화 실패 복구 메시지인지 감사·재현 시 구분한다.
                "safe_fallback": bool(raw.get("safe_fallback")),
                "safe_fallback_reason": raw.get("safe_fallback_reason"),
                "evidence_first_skipped": bool(raw.get("evidence_first_skipped")),
                "evidence_first_fallback": bool(raw.get("evidence_first_fallback")),
                "repetition_detected": repetition_triggered,
                "evidence_exhausted": evidence_exhausted,
                "repetition_matches": restatement_matches,
            },
            message_id=message_id,
        )
        trace_event(
            "IDEATION_TURN_END",
            session_id=state.get("session_id"),
            message=message["message_id"],
            speaker=persona_id,
            target=responding_to_speaker_id,
            issue=active_issue_id,
            stance=stance,
            resolved=issue_resolved,
            needs_counterpart=needs_counterpart_response,
            recommended_next=recommended_next_speaker,
            new_information_count=len(new_information),
            text_length=len(content),
            text=sanitize_preview(content),
            # injected_evidence_count: 실제로 이번 prompt에 담겨 나간 근거 수. evidence_mode가
            # "inactive"(기존 legacy 경로)면 위 IDEATION_TURN_START의 retrieved_evidence_count와
            # 항상 같다 — turn_evidence가 retrieved 그대로이기 때문이다. Phase 2 active 모드
            # (evidence_mode="active"/"valid_empty")에서는 이보다 작을 수 있다(요청: 선택되지
            # 않은 evidence는 prompt에 넣지 않는다) — message["evidence"]는 검색·감사 기록
            # 용도로 항상 retrieved 전체를 유지하므로(기존 계약 그대로) 여기서는 그 값 대신
            # turn_evidence 길이를 쓴다.
            injected_evidence_count=len(turn_evidence),
            # 용준/Claude(2026-07-22, 요청: RAG 근거 실제 활용 강화) — 성공 판단 기준은
            # injected_evidence_count가 아니라 linked_evidence_count다(요청 18번).
            linked_evidence_count=grounding["linked_evidence_count"],
            supported_claim_count=grounding["supported_claim_count"],
            unsupported_claim_count=grounding["unsupported_claim_count"],
            accepted_claim_count=grounding["accepted_claim_count"],
            grounded_claim_count=grounding["grounded_claim_count"],
            expert_judgment_count=grounding["expert_judgment_count"],
            evidence_status=grounding["evidence_status"],
            missing_information=grounding["missing_information"],
            consecutive_zero_linked_turns=consecutive_zero_linked_turns,
            consecutive_expert_judgment_only_turns=consecutive_expert_judgment_only_turns,
            consecutive_repeated_missing_information_turns=consecutive_repeated_missing_information_turns,
            consecutive_no_new_information_turns=consecutive_no_new_information_turns,
            next_action=next_action,
            elapsed_ms=round((time.perf_counter() - turn_started) * 1000, 1),
        )

        new_consensus = list(state["consensus"])
        for item in _as_string_list(raw.get("confirmed")):
            if item not in new_consensus:
                new_consensus.append(item)

        previous_open_ids = {issue.get("issue_id") for issue in (state.get("open_issues") or [])}
        previous_resolved_ids = {issue.get("issue_id") for issue in (state.get("resolved_issues") or [])}
        if active_issue_id is not None:
            # 용준/Claude(2026-07-23, 요청: 동일 쟁점 표현 변경 반복 루프 수정) — 이번 턴이
            # resolve_issue_duplicate에서 중복으로 판정됐지만 로테이션할 다음 공식 쟁점도
            # 없었다면(active_issue_id=None) 새 레코드를 만들지 않는다 — 이미 다뤄진 쟁점을
            # 다시 open_issues에 등록하지 않기 위함이다(요청: "신규 쟁점을 다시 등록하지
            # 않음").
            open_issues, resolved_issues = _update_issue_records(
                open_issues=state.get("open_issues") or [],
                resolved_issues=state.get("resolved_issues") or [],
                persona_id=persona_id,
                issue_id=active_issue_id,
                issue_title=active_issue_title,
                position_text=proposal or interim_conclusion or judgment,
                resolved=issue_resolved,
                resolution_text=proposal or interim_conclusion or judgment,
                family=active_issue_family,
            )
        else:
            open_issues = state.get("open_issues") or []
            resolved_issues = state.get("resolved_issues") or []
        if active_issue_id is None:
            trace_event(
                "IDEATION_ISSUE_UPDATE_SKIPPED",
                session_id=state.get("session_id"),
                updated_by=persona_id,
                reason="duplicate_issue_no_rotation_target",
                remaining_open_issue_count=len(open_issues),
            )
        elif active_issue_id not in previous_open_ids and not issue_resolved:
            trace_event(
                "IDEATION_ISSUE_OPENED",
                session_id=state.get("session_id"),
                issue=active_issue_id,
                title=active_issue_title,
                updated_by=persona_id,
                previous_status="missing",
                new_status="open",
                remaining_open_issue_count=len(open_issues),
            )
        elif issue_resolved and active_issue_id not in previous_resolved_ids:
            trace_event(
                "IDEATION_ISSUE_RESOLVED",
                session_id=state.get("session_id"),
                issue=active_issue_id,
                title=active_issue_title,
                updated_by=persona_id,
                previous_status="open",
                new_status="resolved",
                resolution=sanitize_preview(proposal or interim_conclusion or judgment, limit=200),
                remaining_open_issue_count=len(open_issues),
            )
        else:
            trace_event(
                "IDEATION_ISSUE_UPDATED",
                session_id=state.get("session_id"),
                issue=active_issue_id,
                title=active_issue_title,
                updated_by=persona_id,
                previous_status="open",
                new_status="open",
                remaining_open_issue_count=len(open_issues),
            )

        unconfirmed = _as_string_list(raw.get("unconfirmed"))
        # "unconfirmed" 키 자체가 없으면(구버전 응답 등) 기존 unresolved_issues를 그대로
        # 둔다 — 키가 있는데 배열이 아니면(타입 오류) 안전하게 빈 배열로 정규화한다.
        resolved_unresolved_issues = unconfirmed if "unconfirmed" in raw else state["unresolved_issues"]
        if autonomous_assumption_note and autonomous_assumption_note not in resolved_unresolved_issues:
            # 용준/Claude(2026-07-23, 요청: expert_judgment 처리 시 가정·한계를 명확히 남김) —
            # 사용자에게 묻지 않고 전문가 판단으로 진행했다는 사실과 그 가정을 회의록에
            # 남긴다(요청 6번: "전문가 판단에는 가정, 한계, 추가 검증 사항을 명확히 남깁니다").
            resolved_unresolved_issues = resolved_unresolved_issues + [autonomous_assumption_note]
        update: dict[str, Any] = {
            "messages": [message],
            "consensus": new_consensus,
            "unresolved_issues": resolved_unresolved_issues,
            "llm_calls_used": used,
            "open_issues": open_issues,
            "resolved_issues": resolved_issues,
            "active_issue_id": None if issue_resolved else active_issue_id,
            "previous_speaker": persona_id,
            "expert_turn_count": state.get("expert_turn_count", 0) + 1,
            # "잠시만" 재개(reply_to_interjection)가 강제 지정한 다음 발언자는 실행 즉시
            # 소비된다 — 어느 전문가 노드가 실행되든(강제 지정 대상이든 아니든) 매 턴마다
            # None으로 리셋해 다음 라운드에 잔류하지 않게 한다.
            "forced_next_speaker": None,
            # 용준/Claude(2026-07-22, 요청: "잠시만" 취소 중 phase 오염 수정) — discussion_
            # facilitator/candidate_selection이 남긴 next_route("continue_round"/
            # "to_refinement")는 이 노드(그 라우팅의 목적지)가 실행되는 순간 소비 완료다 —
            # forced_next_speaker와 같은 이유로 다음 라운드/다음 요청에 잔류하지 않게 리셋한다.
            "next_route": None,
            # 용준/Claude(2026-07-22, 요청: 반복되는 근거 없는 의견을 사용자 질문으로 전환) —
            # 다음 턴이 이어서 참조할 반복 감지 카운터. apply_user_answer가 사용자 답변 시점에
            # 0/빈 값으로 리셋한다(ideation_conv_state.py 참고).
            "consecutive_zero_linked_turns": consecutive_zero_linked_turns,
            "consecutive_expert_judgment_only_turns": consecutive_expert_judgment_only_turns,
            "last_missing_information": missing_info_normalized,
            "consecutive_repeated_missing_information_turns": consecutive_repeated_missing_information_turns,
            "last_new_information_text": new_information_text,
            "consecutive_no_new_information_turns": consecutive_no_new_information_turns,
            # 용준/Claude(2026-07-23, Phase 1 "Shadow Deterministic Evidence Planner") — 다음
            # 턴이 이어서 참조할 shadow 선택 이력(세션 범위 state, API 응답에는 노출하지 않는다).
            # evidence_planner가 주입되지 않았으면(기본값) 이 필드는 항상 기존 값 그대로다.
            "evidence_plan_shadow_history": shadow_history_map,
            # 용준/Claude(2026-07-23, 요청: 근거 기반 자율 토론형 회의로 개편) — 쟁점당 보완
            # 검색 최대 1회, 같은 결정 질문 반복 금지를 세션 전체에서 추적한다.
            "supplemental_retrieval_issue_ids": supplemental_attempted_issue_ids,
            "asked_decision_fingerprints": asked_decision_fingerprints,
        }
        if is_interjection_first_response:
            # 용준/Claude(2026-07-22, 요청: 지정 위원 질문 후 상대 검토 코드 강제) — "검토
            # 대상"을 이 메시지로 확정한다. required_counterpart_speaker_id/
            # counterpart_review_completed는 reply_to_interjection이 이미 설정해 둔 값
            # 그대로 유지한다(여기서는 아직 검토가 끝나지 않았으므로 바꾸지 않는다).
            update["interjection_response_message_id"] = message["message_id"]
        if is_required_counterpart_review:
            # 상대 위원이 실제로 검토를 마쳤다 — 이 인터젝션에 대한 강제 라우팅을 여기서
            # 종료한다(요청 9번: "상대 검토가 끝난 뒤에만 facilitator 또는 다음 쟁점으로
            # 이동"). 네 필드를 한 세트로 리셋해 다음 인터젝션과 섞이지 않게 한다.
            update["counterpart_review_completed"] = True
            update["interjection_target_speaker_id"] = None
            update["interjection_response_message_id"] = None
            update["required_counterpart_speaker_id"] = None
        return update

    return node


def _stop_reason_for(state: IdeationConvState) -> str:
    """용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): _route_next_expert_turn이
    "facilitator"로 라우팅한 이유를 다시 계산한다(라우터와 같은 우선순위) — 진행자 프롬프트와
    stop_reason 저장에 쓴다. 라우팅 자체를 바꾸지 않는 순수 조회 함수다."""
    messages = state["messages"]
    last = messages[-1] if messages else None
    structured = (last.get("structured") or {}) if last else {}
    if structured.get("needs_user_input"):
        return "user_input_required"
    if structured.get("repetition_detected"):
        return "semantic_repetition_detected"
    if state.get("expert_turn_count", 0) >= MAX_EXPERT_TURNS_PER_ROUND:
        return "max_turns_reached"
    active_issue_id = state.get("active_issue_id")
    if active_issue_id:
        for issue in state.get("open_issues") or []:
            if issue["issue_id"] == active_issue_id and issue.get("turns", 0) >= MAX_EXPERT_TURNS_PER_ISSUE:
                return "max_turns_reached"
    if not (state.get("open_issues") or []):
        return "consensus_reached"
    return "no_new_information"


def make_discussion_facilitator_node(llm_call: LLMCall) -> Callable[[IdeationConvState], dict]:
    """용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): 예전에는 기획/개발이 각각
    정확히 1회씩 말한 "매 라운드 끝"에 무조건 실행됐지만, 이제 _route_next_expert_turn이
    쟁점·반론·발언 캡에 따라 실제로 라운드가 끝났다고 판단했을 때만 실행된다(전문가가
    needs_user_input로 신호를 주거나, 열린 쟁점이 모두 해결되거나, 발언 캡에 도달했을
    때). 다음 라운드로 자동 진행할지(continue_round) 사용자 결정을 기다릴지
    (await_user_decision)는 이제 전문가 노드가 아니라 이 노드가 직접 결정한다 — 그 판단은
    항상 round/max_rounds 상한(무한 루프 방지)과 stop_reason을 코드가 먼저 검증한 뒤에만
    LLM이 정리 문장을 만든다."""

    def node(state: IdeationConvState) -> dict:
        turn_started = time.perf_counter()
        stop_reason = _stop_reason_for(state)
        round_number = state["round"]
        max_rounds = state["max_rounds"]
        open_issues = state.get("open_issues") or []
        resolved_issues = state.get("resolved_issues") or []
        parked_issue_id: str | None = None

        # 용준/Claude(2026-07-23, 요청: max_issue_turns_reached 이후 다음 쟁점으로 이동) —
        # stop_reason="max_turns_reached"는 "라운드 전체 발언 캡"과 "이 쟁점만의 발언 캡" 두
        # 원인을 모두 나타내는 기존 값이라(state["stop_reason"] 계약은 유지해야 하므로 바꾸지
        # 않는다), 여기서는 그중 "이 쟁점 자체가 막혔다"만 별도로 다시 확인한다. 라운드 전체
        # 캡이 원인이면(expert_turn_count가 이미 라운드 캡에 도달) 다른 쟁점도 아직 발언 기회가
        # 없었을 수 있으므로 강제 종료하지 않는다 — 이 쟁점만 개별적으로 캡에 도달했을 때만
        # "발언 상한 도달로 강제 종료"하고 다음 라운드가 다른 쟁점을 다루도록 active_issue_id를
        # 비운다(요청: "특히 max_issue_turns_reached 이후 다음 쟁점으로 이동하는 로직을
        # 추가하되 기존 테스트를 보존").
        active_issue_id = state.get("active_issue_id")
        round_cap_hit = state.get("expert_turn_count", 0) >= MAX_EXPERT_TURNS_PER_ROUND
        if stop_reason == "semantic_repetition_detected" and active_issue_id:
            parked_issue_id = active_issue_id
        elif stop_reason == "max_turns_reached" and not round_cap_hit and active_issue_id:
            for issue in open_issues:
                if issue["issue_id"] == active_issue_id and issue.get("turns", 0) >= MAX_EXPERT_TURNS_PER_ISSUE:
                    parked_issue_id = active_issue_id
                    break

        parked_family: str | None = None
        next_issue_family: str | None = None
        if parked_issue_id:
            parked_record = next(issue for issue in open_issues if issue["issue_id"] == parked_issue_id)
            parked_family = parked_record.get("family") or resolve_canonical_issue_family(parked_record.get("title"))
            closed_reason = (
                "semantic_repetition_detected"
                if stop_reason == "semantic_repetition_detected"
                else "max_issue_turns_reached"
            )
            resolution_text = (
                "새로운 정보 없이 같은 판단이 반복되어 잠정 결론으로 보류하고 다음 쟁점으로 넘어갑니다."
                if closed_reason == "semantic_repetition_detected"
                else "발언 상한 도달로 강제 종료 — 전문가 판단으로 잠정 결론을 채택하고 다음 쟁점으로 넘어갑니다."
            )
            parked_record = {
                **parked_record,
                "status": "resolved",
                "resolution": resolution_text,
                "family": parked_family,
                "closed_reason": closed_reason,
                "resolution_kind": "parked_expert_judgment",
            }
            open_issues = [issue for issue in open_issues if issue["issue_id"] != parked_issue_id]
            resolved_issues = resolved_issues + [parked_record]
            trace_event(
                "IDEATION_ISSUE_RESOLVED",
                session_id=state.get("session_id"),
                issue=parked_issue_id,
                title=parked_record.get("title"),
                updated_by="ideation_facilitator",
                previous_status="open",
                new_status="resolved",
                resolution=(
                    "semantic_repetition_forced_close"
                    if closed_reason == "semantic_repetition_detected"
                    else "max_issue_turns_reached_forced_close"
                ),
                closed_reason=closed_reason,
                resolution_kind="parked_expert_judgment",
                remaining_open_issue_count=len(open_issues),
            )
            # 용준/Claude(2026-07-23, 요청: "동일 쟁점 표현 변경 반복 루프" 수정 — 발언 상한
            # 이후 로테이션) — 우선순위: (1) 이미 열려 있는 다른 open_issues의 family,
            # (2) TOPIC_PRIORITY에서 아직 다루지 않은 다음 공식 쟁점. 다음 쟁점의 family는
            # 반드시 parked_family와 달라야 한다(_select_next_issue_family가 excluded_family로
            # 강제한다).
            next_issue_family = _select_next_issue_family(
                excluded_family=parked_family,
                open_issues=open_issues,
                resolved_issues=resolved_issues,
                resolved_topics=state.get("resolved_topics") or [],
            )
            trace_event(
                "IDEATION_ISSUE_ROTATED",
                session_id=state.get("session_id"),
                previous_issue_id=parked_issue_id,
                previous_issue_family=parked_family,
                next_issue_id=(f"topic_{next_issue_family}" if next_issue_family and not open_issues else None),
                next_issue_family=next_issue_family,
                rotation_reason=closed_reason,
                skipped_duplicate_count=0,
            )

        # 용준/Claude(2026-07-23, 요청: "로테이션 결과를 실제 다음 턴에 강제 반영") —
        # next_issue_family는 로그로만 남기면 다음 전문가 턴의 검색어/프롬프트/spoken_text가
        # 여전히 방금 종료된 쟁점을 가리킬 수 있다. 여기서 다음 공식 쟁점을 active_issue_id로
        # 확정한다(이미 open_issues에 그 family가 있으면 그 issue_id를 그대로 쓰고, 없으면
        # 공식 레코드를 미리 만든다) — resolve_effective_issue/resolve_retrieval_issue가
        # active_issue_id를 최우선으로 보므로, 다음 턴은 새 LLM 호출 없이 처음부터 새 family를
        # 쓰게 된다. 다음 전문가가 그래도 이전 active_issue_id를 반환하면 기존
        # _validate_discussion_response의 active_issue_id_mismatch 검증이 재시도를 유발해
        # 신규 쟁점을 덮어쓰지 않는다.
        next_active_issue_id: str | None = None
        next_active_issue_title: str | None = None
        if parked_issue_id and next_issue_family is not None:
            existing_next_issue = next(
                (issue for issue in open_issues if issue.get("family") == next_issue_family),
                None,
            )
            if existing_next_issue is not None:
                next_active_issue_id = existing_next_issue["issue_id"]
                next_active_issue_title = existing_next_issue.get("title") or next_active_issue_id
            else:
                next_active_issue_id = f"topic_{next_issue_family}"
                next_active_issue_title = _TOPIC_TITLE_KO.get(next_issue_family, next_issue_family)
                open_issues = open_issues + [
                    {
                        "issue_id": next_active_issue_id,
                        "title": next_active_issue_title,
                        "status": "open",
                        "planning_position": None,
                        "development_position": None,
                        "resolution": None,
                        "turns": 0,
                        "family": next_issue_family,
                        "closed_reason": None,
                        "resolution_kind": None,
                    }
                ]

        if stop_reason == "user_input_required":
            decided_next_action = "await_user_decision"
        elif round_number >= max_rounds:
            decided_next_action = "await_user_decision"
        elif stop_reason == "max_turns_reached" and open_issues:
            decided_next_action = "continue_round"
        elif parked_issue_id and not open_issues and next_issue_family is None:
            # 용준/Claude(2026-07-23, 요청: 전체 회의 종료 보장) — 방금 쟁점을 강제 종료했고
            # 다른 열린 쟁점도 없고 아직 다루지 않은 공식 평가축도 더 없다면, 예전에는 이
            # 조합이 어느 분기에도 걸리지 않아 "else: continue_round"로 빠지면서 experts가
            # 매 라운드 새 쟁점을 지어내는 무한 루프의 원인이 됐다 — 이제 회의를 정리한다.
            decided_next_action = "await_user_decision"
        elif stop_reason == "consensus_reached" and not open_issues:
            decided_next_action = "await_user_decision"
        else:
            decided_next_action = "continue_round"

        planning_msg = _most_recent_message_by(state["messages"], "planning_expert")
        dev_msg = _most_recent_message_by(state["messages"], "dev_expert")
        planning_position = planning_msg.get("structured") if planning_msg else None
        development_review = dev_msg.get("structured") if dev_msg else None
        # 용준/Claude(2026-07-23, 요청: 결정론적 사용자 질문 게이트) — resolve_user_input_gate가
        # 이미 "정말 사용자 결정이 필요하다"고 판단해 잘 구성된 질문(선택지+기본값 포함)을
        # 만들어 뒀다면, 진행자가 자기 말로 다시 지어내 형식을 무너뜨리지 않도록 그대로
        # 이어받는다 — 게이트의 판단이 최종적이다.
        last_message = state["messages"][-1] if state.get("messages") else None
        last_structured = (last_message.get("structured") or {}) if last_message else {}
        gated_decision_required = bool(last_structured.get("user_decision_required"))
        gated_decision_question = last_structured.get("user_question") if gated_decision_required else None
        message_id = _new_message_id()
        trace_event(
            "IDEATION_TURN_START",
            session_id=state.get("session_id"),
            message=message_id,
            phase=state.get("phase"),
            round=round_number,
            issue=state.get("active_issue_id"),
            speaker="ideation_facilitator",
            speaker_name="진행자",
            evidence_count=0,
        )

        prompt = build_ideation_conv_discussion_facilitator_prompt(
            state["notice_and_criteria"],
            planning_position,
            development_review,
            None,
            state["consensus"],
            state["unresolved_issues"],
            decided_next_action,
            round_number,
            max_rounds,
            open_issues=open_issues,
            resolved_issues=resolved_issues,
            stop_reason=stop_reason,
            next_issue_hint=next_active_issue_title,
        )
        raw, ok, attempts = _safe_call_structured_json(
            llm_call, prompt, _validate_facilitator_response, "discussion_facilitator"
        )
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"phase": "failed", "failed_node": "discussion_facilitator", "llm_calls_used": used}

        summary_text = raw.get("facilitator_summary", "")
        agreements = _as_string_list(raw.get("agreements"))
        disagreements = _as_string_list(raw.get("disagreements"))
        needs_user_decision = bool(raw.get("needs_user_decision"))
        user_question = raw.get("user_question") if needs_user_decision else None
        facilitator_question_suppressed = False
        if needs_user_decision and not _is_actionable_user_decision_question(user_question):
            trace_event(
                "IDEATION_USER_QUESTION_SUPPRESSED",
                session_id=state.get("session_id"),
                speaker="ideation_facilitator",
                issue=state.get("active_issue_id"),
                reason="facilitator_question_not_actionable",
                question=sanitize_preview(user_question or ""),
            )
            facilitator_question_suppressed = True
            needs_user_decision = False
            user_question = None

        if gated_decision_required and gated_decision_question:
            # 결정론적 게이트가 이미 "진짜 사용자 결정"이라고 판정했다 — 진행자 LLM의 자체
            # 판단(needs_user_decision/user_question)보다 이 값이 우선한다.
            needs_user_decision = True
            user_question = gated_decision_question
        elif stop_reason == "user_input_required" and not gated_decision_required:
            # 게이트가 사용자 결정이 필요 없다고(전문가 판단/보완 검색으로 자율 진행) 이미
            # 판정했다면, 진행자가 스스로 다시 사용자에게 묻지 않는다 — 게이트 판단이
            # 최종적이다.
            needs_user_decision = False
            user_question = None

        if decided_next_action == "await_user_decision" and not (needs_user_decision and user_question):
            # 실제 결정 질문이 없으면 사용자 대기 상태를 만들지 않는다. 남은 쟁점이 있으면
            # 다음 라운드로 이동하고, 모두 소진됐으면 synthesis로 정상 종료한다.
            needs_user_decision = False
            user_question = None
            if open_issues and round_number <= max_rounds:
                decided_next_action = "continue_round"
                trace_event(
                    "IDEATION_USER_DECISION_SKIPPED",
                    session_id=state.get("session_id"),
                    speaker="ideation_facilitator",
                    reason="no_actionable_question_with_remaining_issue",
                    next_issue_id=next_active_issue_id
                    or (open_issues[0].get("issue_id") if open_issues else None),
                    next_issue_title=next_active_issue_title
                    or (open_issues[0].get("title") if open_issues else None),
                    remaining_open_issue_count=len(open_issues),
                )
            else:
                decided_next_action = "complete_discussion"
                trace_event(
                    "IDEATION_DISCUSSION_COMPLETED",
                    session_id=state.get("session_id"),
                    speaker="ideation_facilitator",
                    reason=(
                        "round_limit_grace_consumed"
                        if open_issues
                        else "no_actionable_question_and_no_remaining_issue"
                    ),
                    remaining_open_issue_count=len(open_issues),
                    resolved_issue_count=len(resolved_issues),
                )

        # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — 채팅에
        # 실제로 보이는 content는 spoken_text(1~2문장의 자연스러운 정리, needs_user_decision=
        # true면 질문 자체를 자연스럽게 포함) 그대로다.
        content = raw.get("spoken_text", "").strip()
        if not needs_user_decision and decided_next_action == "continue_round" and content.endswith("?"):
            content = summary_text or "현재 논의를 정리하고 다음 세부 쟁점으로 이어가겠습니다."
        if needs_user_decision and gated_decision_required and gated_decision_question and user_question not in content:
            # 결정론적 게이트가 만든 선택지+기본값 형식의 질문을 화면에 보이는 문장에도 그대로
            # 반영한다 — 진행자가 자기 말로 요약하면서 선택지 구조가 사라지는 것을 막는다.
            content = f"{content}\n\n{user_question}" if content else user_question
        elif needs_user_decision and user_question and user_question not in content:
            content = f"{content}\n\n{user_question}" if content else user_question

        message = _build_message(
            persona_id="ideation_facilitator",
            round_number=round_number,
            message_type="summary",
            content=content,
            referenced_message_ids=[],
            evidence=[],
            structured={
                "facilitator_summary": summary_text,
                "agreements": agreements,
                "disagreements": disagreements,
                "needs_user_decision": needs_user_decision,
                "user_question": user_question,
                "stop_reason": stop_reason,
            },
            message_id=message_id,
        )
        trace_event(
            "IDEATION_TURN_END",
            session_id=state.get("session_id"),
            message=message["message_id"],
            speaker="ideation_facilitator",
            phase=state.get("phase"),
            round=round_number,
            stop_reason=stop_reason,
            next_action=decided_next_action,
            text_length=len(content),
            text=sanitize_preview(content),
            elapsed_ms=round((time.perf_counter() - turn_started) * 1000, 1),
        )

        new_consensus = list(state["consensus"])
        for item in agreements:
            if item not in new_consensus:
                new_consensus.append(item)

        def _snapshot(persona_id: str, raw_or_none: dict | None) -> str:
            if not raw_or_none:
                return ""
            return _discussion_round_snapshot_text(
                persona_id,
                raw_or_none.get("judgment", ""),
                raw_or_none.get("reason", ""),
                raw_or_none.get("suggestion", ""),
                raw_or_none.get("confirmed"),
                raw_or_none.get("unconfirmed"),
                raw_or_none.get("interim_conclusion", ""),
                raw_or_none.get("responding_to"),
                raw_or_none.get("agreement"),
                raw_or_none.get("concern"),
                raw_or_none.get("revision"),
            )

        record = DiscussionRoundRecord(
            round=round_number,
            planning_position=_snapshot("planning_expert", planning_position),
            development_review=_snapshot("dev_expert", development_review),
            revised_proposal=None,
            facilitator_summary=summary_text,
            needs_user_decision=needs_user_decision,
        )

        update: dict[str, Any] = {
            "messages": [message],
            "consensus": new_consensus,
            "discussion_rounds": [record],
            "llm_calls_used": used,
            "stop_reason": stop_reason,
        }
        if parked_issue_id:
            # 발언 상한으로 강제 종료한 쟁점은 다음 라운드에 다시 등장하지 않는다 —
            # open_issues/resolved_issues와 active_issue_id를 함께 갱신해야 다음 라운드가
            # 다른(아직 열려 있는) 쟁점을 다룬다. active_issue_id는 None이 아니라 위에서
            # 확정한 next_active_issue_id(공식 다음 쟁점, 더 다룰 쟁점이 없으면 None)로
            # 설정한다 — None으로 비우기만 하면 다음 전문가가 이전 쟁점을 그대로 이어갈 수
            # 있다.
            update["open_issues"] = open_issues
            update["resolved_issues"] = resolved_issues
            update["active_issue_id"] = next_active_issue_id
        if decided_next_action == "continue_round":
            # 용준/Claude(2026-07-22, 요청: "잠시만" 취소 중 phase 오염 수정) — 다음 라운드로
            # 자동 진행한다(같은 그래프 호출 안에서, 사용자 입력 없이). phase는 항상 이 시점의
            # 실제 canonical 상태("expert_discussion", 이미 REPLYABLE/진입 가능한 값)로
            # 유지하고, "다음 라운드 전문가 노드로 곧장 이어간다"는 그래프 내부 라우팅
            # 신호는 next_route에 별도로 담는다(ideation_conv_build.py::_route_after_facilitator
            # 참고) — phase 자체를 그래프 밖에서는 의미 없는 내부 신호값으로 잠깐 바꾸던
            # 이전 방식은, 바로 이 시점(다음 노드 실행 중)에 취소되면 그 "잠깐"의 phase가
            # 그대로 세션에 저장돼 재개를 막는 문제가 있었다. 열린 쟁점(있다면)은 다음
            # 라운드에도 그대로 이어간다 — expert_turn_count만 리셋해 새 라운드의 발언 캡을
            # 다시 확보한다.
            update["phase"] = "expert_discussion"
            update["next_route"] = "continue_round"
            update["round"] = round_number + 1
            update["expert_turn_count"] = 0
            update["pending_question"] = None
            update["pending_question_topic"] = None
        elif needs_user_decision and user_question:
            update["phase"] = "awaiting_user_decision"
            update["next_route"] = None
            update["pending_question"] = user_question
            update["pending_question_topic"] = "facilitator_decision"
        elif decided_next_action == "complete_discussion":
            update["phase"] = "discussion_complete"
            update["next_route"] = None
            update["pending_question"] = None
            update["pending_question_topic"] = None
        else:
            # 불변조건: awaiting_user_decision은 실행 가능한 질문과 항상 함께 존재한다.
            update["phase"] = "failed"
            update["failed_node"] = "discussion_facilitator_routing"
            update["next_route"] = None
            update["pending_question"] = None
            update["pending_question_topic"] = None
        return update

    return node


def _validate_canvas_response(raw: dict) -> str | None:
    """캔버스 응답의 키와 타입을 검증한다. 논의 전 빈 문자열은 허용한다."""
    for key in _CANVAS_TEXT_FIELDS:
        if not isinstance(raw.get(key), str):
            return f"missing_or_not_string:{key}"
    feasibility = raw.get("feasibility")
    if not isinstance(feasibility, str) or feasibility.strip() not in _VALID_CANVAS_FEASIBILITY:
        return "invalid_feasibility"
    if not isinstance(raw.get("risks"), list):
        return "risks_not_a_list"
    return None


def make_canvas_update_node(llm_call: LLMCall) -> Callable[[IdeationConvState], dict]:
    """라운드 종료 뒤 캔버스를 갱신한다. 실패해도 회의 상태는 실패로 바꾸지 않는다."""

    def node(state: IdeationConvState) -> dict:
        prompt = build_ideation_conv_canvas_update_prompt(
            state.get("idea_canvas"),
            state.get("selected_idea"),
            state.get("initial_idea"),
            state.get("discussion_planning_position"),
            state.get("discussion_development_review"),
            state.get("discussion_revised_proposal"),
            state["consensus"],
            state["unresolved_issues"],
            state["notice_and_criteria"],
        )
        try:
            raw, ok, attempts = _safe_call_structured_json(
                llm_call, prompt, _validate_canvas_response, "canvas_update"
            )
        except IdeationCancelled:
            raise
        except Exception:
            # 캔버스는 보조 표시 계층이다. 공급자 오류나 테스트용 LLM의 미지원 응답 때문에
            # 이미 완료된 전문가 회의까지 실패시키지 않는다. 취소 신호만은 반드시 전파한다.
            logger.warning("아이디어 캔버스 갱신 호출에 실패해 직전 값을 유지합니다", exc_info=True)
            return {"llm_calls_used": state.get("llm_calls_used", 0) + 1}
        used = state.get("llm_calls_used", 0) + attempts
        if not ok:
            return {"llm_calls_used": used}

        canvas: dict[str, Any] = {key: raw[key].strip() for key in _CANVAS_TEXT_FIELDS}
        canvas["feasibility"] = raw["feasibility"].strip()
        canvas["risks"] = _as_string_list(raw.get("risks"))[:_MAX_CANVAS_RISKS]
        return {"idea_canvas": canvas, "llm_calls_used": used}

    return node


def make_conv_synthesis_node(llm_call: LLMCall) -> Callable[[IdeationConvState], dict]:
    """사용자가 확정 버튼을 눌렀을 때만 실행되는 최종 종합 노드(요청 9~10번 — 오케스트레이션
    레벨에서 phase="finalizing"으로만 진입 가능하게 막아 두었으므로, 이 노드 자체가 또
    안전장치를 하나 더 두는 것은 아니다 — ideation_conv_run.py::finalize_ideation_conversation()
    참고)."""

    def node(state: IdeationConvState) -> dict:
        discovery_history = None
        if state.get("ideation_mode") == "discovery":
            # 요청 8번 — discovery 세션이면 최초 생성 후보/선택된 후보/선택 이유를 최종
            # 결과에 포함한다. state["ideation_mode"]를 여기서 다시 판단하지 않고(이미
            # initial_conv_state가 결정한 값을) 그대로 읽기만 한다.
            discovery_history = {
                "original_candidates": state.get("original_idea_candidates", []),
                "selected_idea": state.get("selected_idea"),
                "selection_reason": state.get("selection_reason"),
            }
        prompt = build_ideation_conv_synthesis_prompt(
            state["notice_and_criteria"],
            state["user_idea"],
            state["messages"],
            state["consensus"],
            state["unresolved_issues"],
            discovery_history=discovery_history,
        )
        raw, ok = _safe_call_json(llm_call, prompt)
        used = state.get("llm_calls_used", 0) + 1
        if not ok:
            return {"phase": "failed", "failed_node": "conv_synthesis", "llm_calls_used": used}
        return {"idea_proposal": raw, "phase": "finalized", "llm_calls_used": used}

    return node
