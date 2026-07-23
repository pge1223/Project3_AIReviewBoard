# 작성자: 용준/Claude(2026-07-20)
# 목적: "아이디어 발전 회의(ideation)"의 대화형(턴마다 사용자 응답을 기다리는) 버전 State.
#       기존 IdeationState(ideation_state.py)는 한 번의 그래프 실행 안에서
#       planning_expert -> dev_expert -> planning_expert_revise -> facilitator가 전부
#       돌고 나서야 사용자 질문 여부를 판단하므로, "기획 질문 직후 정지 -> 사용자 답변 ->
#       개발 질문 직후 정지 -> 사용자 답변 -> 두 전문가 의견 보완"이라는 요구를 그대로
#       담을 수 없다(질문 하나당 정지 지점이 필요). 이 State는 그 정지 지점들을 phase로
#       명시적으로 표현한다. 기존 IdeationState/그래프는 전혀 수정하지 않는다.
# import: 표준 라이브러리 operator/typing만 사용(외부 의존성 없음).

from __future__ import annotations

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

# 대화 진행 단계. "실패"는 기존 IdeationStage와 통일해 한국어 대신 영문 slug를 쓴다 —
# 이 phase는 프론트가 직접 분기 렌더링에 쓰는 값이라(요구된 8개 상태 그대로) 계약을
# 영문으로 고정해 프론트/백엔드 문자열 매칭 실수를 줄인다.
#
# 용준/Claude(2026-07-21): discovery(아이디어 발굴) 모드용 phase 3개를 추가한다 —
# candidate_generation(기획 후보 생성 -> 개발 실현가능성 검토, 정지 없이 연속 실행),
# awaiting_candidate_selection(후보 제시 후 사용자 선택 대기, 정지 지점),
# candidate_selection(사용자의 선택/결합/재추천/전문가추천 요청 처리). refinement 전용
# phase(planning_question 등)는 값 하나도 바꾸지 않는다 — discovery는 이 phase들을 거쳐
# 최종적으로 정확히 refinement의 "planning_question" phase로 합류한다(요청 4번).
ConvPhase = Literal[
    "candidate_generation",
    "awaiting_candidate_selection",
    "candidate_selection",
    "planning_question",
    "awaiting_planning_answer",
    "developer_question",
    "awaiting_developer_answer",
    "expert_discussion",
    "awaiting_user_decision",
    "discussion_complete",
    "finalized",
    "failed",
    # 내부 전이용 값 — API/프론트에 노출되는 공개 상태에는 포함되지 않는다.
    # request_finalize()가 잠깐 이 값으로 바꿔 그래프 진입 라우팅(_route_entry)이
    # synthesis 노드로 가게 만들 뿐, synthesis 노드가 끝나면 항상 "finalized" 또는
    # "failed"로 바뀌므로 이 값이 API 응답에 그대로 나가는 일은 없다.
    "finalizing",
]

# 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환): "interjection"은 사용자가 진행자의
# 직접 질문(pending_question)에 답한 게 아니라, 라운드 사이에 자발적으로 끼어든 발언이다 —
# reply_ideation_conversation이 previous_state.get("pending_question") 존재 여부로 "answer"와
# "interjection"을 구분한다(요청 6번: "user_interjection으로 기록").
MessageType = Literal["question", "answer", "interjection", "opinion", "agreement", "disagreement", "summary"]

_TERMINAL_ENTRY_PHASES = {
    "candidate_generation",
    "candidate_selection",
    "planning_question",
    "developer_question",
    "expert_discussion",
}

IdeationMode = Literal["refinement", "discovery"]

# 용준/Claude(2026-07-21): 질문 생성 노드가 이번 질문에서 어떤 종류의 답을 기대하는지
# 표시하는 값. sufficiency 판정이 "답변 충분성"과 "아이디어 완성도"를 혼동하지 않도록
# 돕는다 — preference/selection(선호·선택·방향성)은 하나를 명확히 고르기만 해도
# 충분하고, definition/constraint/evidence/specification은 상대적으로 더 구체적인
# 내용을 요구한다(ideation_conv_sufficiency.txt 참고). 질문 노드가 이 값을 만들지
# 못하거나(구버전 응답 등) 유효하지 않은 값을 반환하면 None으로 저장하고, sufficiency
# 프롬프트는 그 경우 기존의 일반 기준으로 판정한다(하위 호환 — 이 값은 있으면 정확도를
# 높이는 보조 정보이지, 없다고 판정 자체가 막히지 않는다).
ExpectedAnswerType = Literal["preference", "selection", "definition", "constraint", "evidence", "specification"]

# 용준/Claude(2026-07-21, 질문 주제 구조화): 실제 사용자 테스트에서 "문제·목표 사용자·핵심
# 가치·공모전 적합성이 정리되지 않았는데 로드맵부터 질문"하거나 "한 질문에서 여러 쟁점을
# 동시에 묻는" 문제가 확인됐다. 이를 막기 위해 질문 하나가 다루는 주제를 명시적인 값
# (question_topic)으로 구조화하고, 그 우선순위를 코드가 강제한다 — 이 순서는 "문제 정의가
# 안 됐는데 확장 로드맵부터 묻는" 실패를 원천적으로 막기 위한 것이다(요청 목표 1~9번 순서
# 그대로). 모든 주제를 기계적으로 다 물어야 하는 것은 아니다 — 이미 resolved_topics에
# 있으면 건너뛴다(아래 remaining_topics_for 참고).
QuestionTopic = Literal[
    "problem", "target_user", "core_value", "contest_fit", "differentiation", "mvp", "data", "ai_role", "roadmap"
]

TOPIC_PRIORITY: tuple[str, ...] = (
    "problem",
    "target_user",
    "core_value",
    "contest_fit",
    "differentiation",
    "mvp",
    "data",
    "ai_role",
    "roadmap",
)

# roadmap(확장 기능/도입 순서)은 이 5개 주제가 모두 resolved_topics에 있어야만 질문할 수
# 있다(요청 2번 — "확장 기능과 로드맵을 너무 일찍 질문"하는 문제의 직접적인 원인 제거).
ROADMAP_PREREQUISITE_TOPICS: frozenset[str] = frozenset({"problem", "target_user", "core_value", "contest_fit", "mvp"})


def remaining_topics_for(resolved_topics: list[str] | None) -> list[str]:
    """아직 확인되지 않은 주제를 우선순위 순서로 반환한다. roadmap의 선행 주제
    (ROADMAP_PREREQUISITE_TOPICS)가 모두 resolved_topics에 없으면 roadmap 자체를 목록에서
    제외한다 — 질문 노드가 애초에 roadmap을 고를 수 없는 후보 목록만 보게 하는 것이,
    "질문 규칙으로만 금지"하는 것보다 더 확실한 강제 방법이다. resolved_topics가 None이면
    (구버전 state) 빈 리스트로 취급한다(하위 호환)."""
    resolved_set = set(resolved_topics or [])
    remaining = [topic for topic in TOPIC_PRIORITY if topic not in resolved_set]
    if "roadmap" in remaining and not ROADMAP_PREREQUISITE_TOPICS.issubset(resolved_set):
        remaining = [topic for topic in remaining if topic != "roadmap"]
    return remaining

# 용준/Claude(2026-07-21): ideation_mode는 세션이 "최초 진입할 때" discovery였는지
# refinement였는지만 기록한다(initial_conv_state가 딱 한 번 결정하고 이후 절대 바뀌지
# 않음) — 그런데 discovery 세션도 후보 선택 후에는 refinement와 동일한 질문/의견 흐름을
# 탄다. 프론트가 ideation_mode만 보고 배지를 표시하면 후보 선택 후에도 계속 "아이디어
# 발굴 모드"로 잘못 표시된다. active_stage는 그 문제를 풀기 위해 "현재 진행 단계"를
# 별도로 노출한다 — phase(그래프 내부 상태 기계 값, 세분화돼 있고 일부는 API에 절대
# 노출되지 않는 전이용 값)와 달리, active_stage는 프론트 배지 전용의 넓은 4단계
# 요약이다.
ActiveStage = Literal["candidate_discovery", "candidate_selection", "refinement", "finalized"]

# phase -> active_stage 매핑. candidate_generation/awaiting_candidate_selection은 아직
# 후보를 고르지 않은 단계라 "candidate_discovery"(아이디어 발굴 모드), candidate_selection은
# 사용자의 선택/결합/재추천 요청을 처리하는 중(그래프 내부 전이만으로 존재하고 API 응답에
# phase 자체로는 절대 노출되지 않지만, active_stage 매핑은 완전성을 위해 모든 phase를
# 다룬다), planning_question부터 awaiting_user_decision까지는 후보 선택이 끝나고 아이디어를
# 다듬는 "refinement"(아이디어 발전 모드), finalized/finalizing은 "finalized"다. failed는
# 별도로 처리한다(고정 4단계에 없음 — 아래 active_stage_for 참고).
_PHASE_TO_ACTIVE_STAGE: dict[str, ActiveStage] = {
    "candidate_generation": "candidate_discovery",
    "awaiting_candidate_selection": "candidate_discovery",
    "candidate_selection": "candidate_selection",
    "planning_question": "refinement",
    "awaiting_planning_answer": "refinement",
    "developer_question": "refinement",
    "awaiting_developer_answer": "refinement",
    "expert_discussion": "refinement",
    "awaiting_user_decision": "refinement",
    "finalized": "finalized",
    "finalizing": "finalized",
}


def active_stage_for(phase: str) -> ActiveStage | Literal["failed"]:
    """phase(세분화된 그래프 상태)를 프론트 배지용 넓은 진행 단계로 축약한다. refinement
    모드로 시작한 세션은 처음부터 "refinement"이고, discovery 모드로 시작한 세션은 후보
    선택 전까지 "candidate_discovery"였다가 선택 확정 순간부터 "refinement"로 바뀐다 —
    ideation_mode(최초 진입 모드, 절대 안 바뀜)와 달리 이 값은 세션 도중 바뀌는 것이
    핵심이다."""
    if phase == "failed":
        return "failed"
    return _PHASE_TO_ACTIVE_STAGE.get(phase, "refinement")


class IssueRecord(TypedDict):
    """용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편): expert_discussion이 다루는
    쟁점 1개. round 번호가 아니라 쟁점 단위로 회의를 관리하기 위한 최소 단위 — LLM은
    active_issue_id/issue_resolved bool만 판단하고, 이 레코드의 생성·이동(open→resolved)은
    항상 코드가 결정적으로 수행한다(라우팅이 LLM 추천을 맹신하지 않는 것과 같은 원칙)."""

    issue_id: str
    title: str
    status: Literal["open", "resolved"]
    planning_position: str | None
    development_position: str | None
    resolution: str | None
    turns: int
    # 용준/Claude(2026-07-23, 요청: 동일 쟁점 표현 변경 반복 루프 수정) — 이 쟁점이 속한
    # 결정론적 canonical family(TOPIC_PRIORITY 슬러그 또는 "custom:<정규화 텍스트>").
    # resolve_canonical_issue_family가 채우며, 표현만 바뀐 재등록을 이 값으로 판별한다.
    # 구버전 저장 레코드에는 없을 수 있으므로 읽는 쪽은 항상 `.get("family")`로 접근한다.
    family: str
    # 강제 종료(발언 상한 도달)인지 실제 합의 해결인지 구분한다 — status="resolved"가 된
    # 이유를 코드·로그·후속 라우팅이 구분할 수 있게 한다(요청: "강제 종료와 합의 완료
    # 구분"). status가 "open"인 동안은 둘 다 None이다.
    closed_reason: str | None  # None | "consensus_reached" | "max_issue_turns_reached" | "semantic_repetition_detected"
    resolution_kind: str | None  # None | "agreed_resolution" | "parked_expert_judgment"


class DiscussionRoundRecord(TypedDict):
    """용준/Claude(2026-07-21, 요청: 위원 간 실제 회의로 개편): expert_discussion 라운드
    1회가 만든 발언들의 텍스트 스냅샷. messages(원본 발언 전체)와 별도로 이 요약을 두는
    이유는, 다음 단계(synthesis 등)가 "이번 라운드에 정확히 무슨 입장 변화가 있었는지"를
    messages 전체를 다시 훑지 않고 바로 참조할 수 있게 하기 위함이다 — content는 messages와
    중복 저장되지만(참조가 아니라 텍스트 스냅샷), 그래야 이후 다른 세션 필드처럼 dict로
    바로 직렬화해 API 응답/프롬프트에 넘기기 쉽다."""

    round: int
    planning_position: str
    development_review: str
    revised_proposal: str | None
    facilitator_summary: str
    needs_user_decision: bool


class ConvMessage(TypedDict):
    message_id: str
    speaker_id: str
    speaker_name: str
    role: str
    round: int
    message_type: MessageType
    content: str
    referenced_message_ids: list[str]
    evidence: list[dict]
    created_at: str
    # 용준/Claude(2026-07-21, 전문가 의견 UX 개선; 2026-07-22, 요청: 보고서형 메시지 →
    # 자연스러운 회의 발화 전환으로 범위 확장): judgment/reason/suggestion/agreement/
    # concern/proposal/interim_conclusion/confirmed/unconfirmed/responding_to_message_id/
    # responding_to_speaker_id 등 내부 판단·상태 필드를 담는다. content는 이제 LLM이 만든
    # spoken_text(사용자에게 보이는 자연스러운 발화 문장) 그대로이고, structured는 그
    # spoken_text를 만들기 위한 재료이자 다음 턴 프롬프트·요약 카드가 참조하는 내부 상태다
    # — content가 없어지는 게 아니라 값이 spoken_text로 바뀌었을 뿐이므로 structured를 모르는
    # 기존 클라이언트는 영향받지 않는다. 답변(answer/interjection) 메시지만 항상 None이다.
    structured: dict | None
    # 용준/Claude(2026-07-22, 요청: RAG 근거 실제 활용 강화) — evidence(위)는 그대로
    # "이번 턴 프롬프트에 주입된 검색 결과 전체"라는 기존 의미를 유지한다(하위 호환).
    # 아래 필드들은 그중 실제로 주장(claim)과 연결·검증된 부분만 별도로 담는 신규 선택
    # 필드다 — claims/grounding 관련 필드가 없는 메시지 타입(질문 응답, 진행자 정리 등
    # 일부)은 빈 리스트/None/0으로 채운다(TypedDict는 런타임에 강제되지 않으므로 이 필드를
    # 모르는 기존 코드는 영향받지 않는다).
    claims: list[dict]
    linked_evidence_refs: list[str]
    supported_claim_count: int
    unsupported_claim_count: int
    # 용준/Claude(2026-07-22, 요청: claim 통계 의미 분리) — supported_claim_count(기존 필드,
    # 의미 변경 없이 유지)와 별도로 "실제 문서 근거로 검증됨"과 "근거 없이 허용된 전문가
    # 판단"을 분리한 신규 선택 필드. claims/grounding이 없는 메시지 타입은 0으로 채운다.
    accepted_claim_count: int
    grounded_claim_count: int
    expert_judgment_count: int
    missing_information: list[str]
    evidence_status: str | None
    sufficiency: str | None


class IdeationConvState(TypedDict):
    """대화형 회의 세션 1개가 그래프 호출 사이(=HTTP 요청 사이)에 들고 다니는 상태.

    messages는 시간순으로 이어붙이는 리스트다(리듀서 operator.add) — 기존
    IdeationState.turns와 같은 이유(순서가 실제 대화 순서를 그대로 반영해야 함)다.
    phase가 이 State의 핵심이다: 그래프는 매 호출마다 phase를 보고 어느 노드부터
    시작할지 결정하고(ideation_conv_build.py::_route_entry), 실행한 노드는 다음에
    무엇을 해야 하는지를 나타내는 새 phase를 반환한다. "awaiting_*"과
    "awaiting_user_decision"은 그래프가 아니라 API 레이어가 사용자 입력을 받을 때까지
    멈춰 있는 지점이다(그래프 자신은 이 phase들로는 절대 진입하지 않고, 오직
    이 phase로 "끝난다").
    """

    session_id: str
    notice_and_criteria: dict
    user_idea: dict
    round: int
    max_rounds: int
    messages: Annotated[list[ConvMessage], operator.add]
    phase: ConvPhase
    pending_question: str | None
    # pending_question과 함께 세팅/리셋된다(질문 노드가 생성할 때 채우고,
    # apply_user_answer가 다음 단계로 넘어갈 때 None으로 되돌린다) — pending_question이
    # 가리키는 "지금 이 질문"이 어떤 종류의 답을 기대하는지에 대한 보조 정보다.
    pending_expected_answer_type: str | None
    # 용준/Claude(2026-07-21, 질문 주제 구조화): pending_question이 다루는 주제
    # (TOPIC_PRIORITY 중 하나). pending_question/pending_expected_answer_type과 함께
    # 세팅/리셋된다. 구버전 저장 state에는 이 키가 없을 수 있으므로 읽는 쪽은 항상
    # `.get("pending_question_topic")`로 접근한다(하위 호환).
    pending_question_topic: str | None
    # 사용자가 "answer"로 판정된 답을 해서 실제로 다음 단계로 진행한 주제만 담는다 —
    # clarification_request/insufficient_answer/재질문 진행 중/구조화 응답 실패는 이
    # 리스트에 추가되지 않는다(ideation_conv_run.py::_apply_answer_sufficiency_gate 참고).
    # 구버전 저장 state에는 이 키가 없을 수 있으므로 읽는 쪽은 항상
    # `.get("resolved_topics", [])`로 접근한다(하위 호환).
    resolved_topics: list[str]
    consensus: list[str]
    unresolved_issues: list[str]
    idea_proposal: dict | None
    idea_canvas: dict | None
    # 가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입): 공모전 신청양식에서 추출한
    # 항목 목록([{field_name, description, char_limit}], 양식에 있는 만큼 전부 — 개수
    # 상한 없음). 세션 시작 시 한 번 채워지고 이후 절대 바뀌지 않는다(discussion_rounds처럼
    # 매 라운드 갱신되는 값이 아니다). make_conv_discussion_node가 매 발언 프롬프트에
    # "참고 자료"로만 주입한다(질문 주제·순서는 여전히 코드가 결정 — 이 값은 같은 주제를
    # 다룰 때 표현만 다듬는 데 쓰인다, ideation_conv_discussion.txt의 [신청양식 참고 규칙]
    # 참고). 없으면 빈 리스트(양식 미등록) — 구버전 저장 state에는 이 키가 없을 수 있으므로
    # 읽는 쪽은 항상 `.get("application_form_items", [])`로 접근한다(하위 호환).
    application_form_items: list[dict]
    failed_node: str | None
    llm_calls_used: int
    # 용준/Claude(2026-07-20): 같은 쟁점(pending_question)으로 재질문한 횟수. 사용자가
    # 질문에 답할 때마다 answer_sufficiency 판정을 거치는데, 무한 재질문을 막기 위해
    # 이 값이 retry_cap(ideation_conv_run.py::_MAX_ANSWER_RETRY)에 도달하면 판정 결과와
    # 무관하게 다음 단계로 강제 진행한다. 재질문이 아니라 다음 단계로 넘어갈 때마다 0으로
    # 리셋된다(쟁점이 바뀌었으므로).
    answer_retry_count: int

    # 용준/Claude(2026-07-21): discovery(아이디어 발굴) 모드 전용 필드. refinement 세션에서는
    # ideation_mode="refinement" 외에는 전부 초기값(빈 값)에서 바뀌지 않는다 — 요청 2번
    # "모드 판단을 여러 노드에서 반복하지 말고 시작 시 결정한 ideation_mode를 그래프 전체에서
    # 사용" — initial_conv_state()가 세션 시작 시 딱 한 번 결정해서 저장하고, 이후 모든 노드는
    # 이 필드를 읽기만 한다(다시 계산하지 않는다).
    ideation_mode: IdeationMode
    initial_idea: str | None
    contest_analysis: dict | None
    # 현재 유효한 후보 목록 — "다시 추천" 시 이 리스트가 교체된다.
    idea_candidates: list[dict]
    # 최초로 생성된 후보 목록 — 재추천으로 idea_candidates가 바뀌어도 이 값은 보존된다
    # (요청 8번 "discovery 모드의 최종 결과에는 최초 생성 후보... 이력을 포함").
    original_idea_candidates: list[dict]
    selected_idea: dict | None
    # 용준/Claude(2026-07-22, 요청: 선택된 아이디어를 target 문서로 생성) — candidate_selection
    # 노드가 선택/결합된 아이디어를 target evidence로 색인한 뒤 그 document_id를 저장한다
    # (ai/meeting/graph/ideation_conv_discovery.py::_resolve_selection). 색인이 주입되지
    # 않았거나(use_rag=False 등) 실패했으면 None이다 — 이 경우 RAG 검색은 이 후보의 target
    # 근거를 아직 찾지 못한 것으로만 취급한다(회의가 막히지 않는다). 사용자가 후보를 다시
    # 선택/결합하면 이 값이 새 document_id로 교체되어, 이전 후보의 target은 더 이상 현재
    # 근거로 검색되지 않는다(ai/rag/orchestration/ideation_evidence_service.py::
    # _scope_target_evidence 참고 — 이전 후보 chunk 자체는 회의 이력으로 Chroma에 남는다).
    # 구버전 저장 state에는 이 키가 없을 수 있으므로 읽는 쪽은 항상 `.get(...)`로 접근한다.
    selected_idea_document_id: str | None
    selection_reason: str | None
    # "다시 추천" 요청 횟수 — ideation_conv_discovery.py::MAX_CANDIDATE_REGENERATIONS에
    # 도달하면 더 이상 LLM을 호출해 후보를 재생성하지 않는다(요청: 무한 반복/LLM 호출 제한
    # 우회 방지).
    candidate_regeneration_count: int

    # 용준/Claude(2026-07-21, 후보 결합 컨텍스트 보존): 사용자가 "1번과 2번 결합"처럼
    # 후보를 선택/결합/추천한 직후, 그 요청이 refinement(질문/의견) 단계로 넘어가면서
    # 사라지지 않도록 별도로 보존하는 필드들 — conversation_context의 최근 메시지에 우연히
    # 남아있는 것에 기대지 않고, 질문 프롬프트가 구조화된 형태로 명시적으로 참조할 수 있게
    # 한다(ideation_conv_nodes.py::_selection_context_for 참고). selected_idea가 확정되지
    # 않는 경우(결합 적합도 low로 재질문하는 중)에도 이 필드들은 채워질 수 있다 —
    # selected_idea만 아직 None/이전 값일 뿐이다.
    selection_intent: str | None
    # 사용자가 후보 선택/결합을 요청한 원문 메시지 그대로.
    user_selection_message: str | None
    # 선택/결합 대상이 된 원본 후보(들)의 전체 필드(title/problem/target_user/core_value/
    # main_features 등) — 결합으로 새로 만들어진 selected_idea와 달리 이 값들은 병합 전
    # 원본 그대로다.
    source_candidates: list[dict]
    # candidate_selection 노드가 "combine" 해석 시 함께 만드는 결합 분석 결과(공통 문제/
    # 공통 가치/결합 적합도/주 기능/보조 기능/충돌 지점/미확정 사항). combine이 아니면 None.
    merge_analysis: dict | None

    # 용준/Claude(2026-07-21, 요청: 위원 간 실제 회의로 개편): expert_discussion phase가
    # 실행될 때마다(라운드마다) 1건씩 쌓인다(리듀서 operator.add — messages와 같은 원칙).
    # 구버전 저장 state에는 이 키가 없을 수 있으므로 읽는 쪽은 항상
    # `.get("discussion_rounds", [])`로 접근한다(하위 호환).
    discussion_rounds: Annotated[list[DiscussionRoundRecord], operator.add]

    # 용준/Claude(2026-07-21, 요청: 위원 간 실제 회의로 개편): 이번 라운드의 discussion 서브
    # 그래프(planning_expert_discussion -> dev_expert_discussion -> [선택적 revision] ->
    # discussion_facilitator)가 노드 사이에서 주고받는 임시 값들. Annotated(operator.add)가
    # 아니므로 매 라운드 노드가 반환하면 그대로 덮어써진다(messages처럼 누적하지 않는다) —
    # discussion_facilitator가 이번 라운드 값만 읽으면 되기 때문이다. 구버전 저장 state에는
    # 이 키들이 없을 수 있으므로 읽는 쪽은 항상 `.get(...)`로 접근한다(하위 호환).
    discussion_planning_position: dict | None
    discussion_development_review: dict | None
    discussion_revised_proposal: dict | None
    # dev_expert_discussion(review 단계)가 정한 다음 행동("continue_round"/
    # "await_user_decision") — 이 값 자체는 discussion_facilitator가 절대 바꾸지 않는다
    # (요청: 기존에 검증된 라운드 진행/max_rounds 강제 로직을 그대로 재사용).
    discussion_next_action: str | None
    # dev_expert_discussion(review 단계)가 고른 stance — planning_expert_revision을 실행할지
    # 결정하는 조건부 엣지(ideation_conv_build.py::_route_after_review)가 참조한다.
    discussion_review_stance: str | None

    # 용준/Claude(2026-07-22, 요청: 동적 전문가 회의로 개편) — round 번호가 아니라 쟁점
    # 단위로 회의를 관리한다. 구버전 저장 state에는 이 키들이 없을 수 있으므로 읽는 쪽은
    # 항상 `.get(...)`로 접근한다(하위 호환).
    open_issues: list[IssueRecord]
    resolved_issues: list[IssueRecord]
    active_issue_id: str | None
    # 직전 발언자(persona_id 또는 "ideation_facilitator") — 같은 화자의 의미 없는 연속 발언을
    # 판단하는 라우터(_route_next_expert_turn)가 참조한다.
    previous_speaker: str | None
    # 이번 라운드(=이번 API 호출 동안 그래프가 한 번에 처리하는 구간) 안에서 실행된 전문가
    # 발언 수 — _MAX_EXPERT_TURNS_PER_ROUND/_MIN_EXPERT_TURNS_PER_ROUND 캡 판단에 쓴다.
    # discussion_facilitator가 라운드를 마무리할 때 0으로 리셋된다.
    expert_turn_count: int
    # 라운드/토론이 왜 끝났는지 기록한다: consensus_reached/user_input_required/
    # no_new_information/max_turns_reached/user_finalized/interrupted_by_user.
    stop_reason: str | None
    # "잠시만" 재개(reply_to_interjection)가 다음 그래프 진입을 특정 전문가로 강제 지정할 때만
    # 채운다 — 해당 노드가 실행되자마자 None으로 리셋되어 다음 라운드에 잔류하지 않는다.
    forced_next_speaker: str | None

    # 용준/Claude(2026-07-22, 요청: 지정 위원 질문 후 상대 검토 코드 강제) — reply_to_interjection이
    # 사용자가 지정한 대상(target_speaker_id 원본값 — "planning_expert"/"dev_expert"/"both")을
    # 그대로 기록한다. 이 네 필드는 서로 세트로 채워지고(reply_to_interjection이 한 번에
    # 설정) counterpart_review_completed=True가 되는 순간 다시 함께 리셋된다(다음 인터젝션과
    # 섞이지 않도록). 구버전 저장 state에는 이 키들이 없을 수 있으므로 읽는 쪽은 항상
    # `.get(...)`로 접근한다(하위 호환 — 없으면 "보류 중인 상대 검토 없음"으로 취급).
    interjection_target_speaker_id: str | None
    # 지정 위원이 인터젝션에 처음 답한 메시지의 message_id — make_conv_discussion_node가
    # 그 위원의 발언을 만든 직후 채운다(요청: 어느 발언이 "검토 대상"인지 코드가 결정적으로
    # 추적). 상대 검토가 끝나면 required_counterpart_speaker_id 등과 함께 None으로 리셋된다.
    interjection_response_message_id: str | None
    # 반드시 한 번 더 발언해야 하는 반대편 위원("planning_expert"/"dev_expert") —
    # reply_to_interjection이 지정 위원의 반대편으로 설정한다. _route_next_expert_turn이
    # 이 값이 남아있는 한(counterpart_review_completed=False) 다른 어떤 라우팅 신호
    # (issue_resolved/needs_user_input/발언 캡 이외)보다 우선해 이 위원에게 발언을 넘긴다.
    required_counterpart_speaker_id: str | None
    # required_counterpart_speaker_id가 실제로 발언을 완료했는지 여부. False인 동안은
    # facilitator로 이동할 수 없다(요청 6번) — reply_to_interjection이 False로 설정하고,
    # 그 위원의 discussion 노드 실행이 끝나면 True로 바뀌며 위 세 필드도 함께 리셋된다.
    counterpart_review_completed: bool

    # 용준/Claude(2026-07-22, 요청: "잠시만" 취소 중 phase 오염 수정) — 그래프 내부에서만
    # 의미가 있는 "다음 라우팅 목적지" 신호. discussion_facilitator가 continue_round를
    # 결정했을 때(_route_after_facilitator)와 candidate_selection이 결합/선택을 확정했을 때
    # (_route_after_candidate_selection)만 값을 채운다 — 이전에는 이 두 곳이 phase 자체를
    # "planning_question"으로 잠깐 바꿔 그래프 내부 라우팅에만 쓰고 곧바로 다음 노드가
    # 실행되길 기대했지만, 취소가 바로 그 다음 노드 실행 중(스트리밍 llm_call)에 일어나면
    # graph.stream()이 이미 그 "잠깐의" phase를 스냅샷으로 내보낸 뒤였다 — 그 스냅샷이
    # IdeationCancelled.partial_state로 세션에 그대로 저장되면서 canonical phase가 그래프
    # 밖에서는 의미 없는 내부 신호값으로 오염됐다(reply_to_interjection이 이 값을 유효한
    # 재개 지점으로 인식하지 못해 거부). 이제 phase는 항상 그 시점의 실제 canonical 상태
    # ("expert_discussion")로 유지하고, 라우팅 목적지만 이 필드로 분리해서 넘긴다 — 목적지
    # 노드(planning_expert_discussion)가 실행되자마자 None으로 리셋되므로(forced_next_speaker와
    # 동일한 패턴) 다음 라운드/다음 요청에 잔류하지 않는다. 구버전 저장 state에는 이 키가
    # 없을 수 있으므로 읽는 쪽은 항상 `.get("next_route")`로 접근한다(하위 호환).
    next_route: str | None

    # 용준/Claude(2026-07-22, 요청: 반복되는 근거 없는 의견을 사용자 질문으로 전환) — 같은
    # 쟁점(active_issue_id)이 바뀌지 않는 한 이어서 누적되고, 이슈가 바뀌거나 사용자가 실제로
    # 답변하면(apply_user_answer) 0/빈 값으로 리셋된다. make_conv_discussion_node가 매 발언
    # 직후 갱신하고, 임계값(2회 연속)에 도달하면 needs_user_input=True로 강제 전환한다(
    # _route_next_expert_turn은 기존 needs_user_input 라우팅을 그대로 재사용한다 — 새 라우팅
    # 분기를 추가하지 않는다). 구버전 저장 state에는 이 키들이 없을 수 있으므로 읽는 쪽은
    # 항상 `.get(...)`로 접근한다(하위 호환).
    consecutive_zero_linked_turns: int
    # evidence_status="expert_judgment_only"(문서 근거 없이 전문가 판단만 있는 턴)가 연속된 횟수.
    consecutive_expert_judgment_only_turns: int
    # 직전 턴의 missing_information(claim_grounding 결과, 정규화·정렬된 텍스트 목록) — 다음
    # 턴이 같은 값을 반복하는지 비교하는 데 쓴다.
    last_missing_information: list[str]
    # missing_information이 직전 턴과 완전히 동일하게(비어있지 않은 채) 반복된 연속 횟수.
    consecutive_repeated_missing_information_turns: int
    # 직전 턴의 new_information(발언 스키마 필수 필드)을 이어붙인 텍스트 — 다음 턴이 같은
    # 내용을 어휘만 바꿔 반복하는지(_looks_like_restatement와 동일한 유사도 판정) 비교한다.
    last_new_information_text: str
    # new_information이 직전 턴과 의미상 거의 동일하게 반복된 연속 횟수.
    consecutive_no_new_information_turns: int

    # 용준/Claude(2026-07-23, Phase 1 "Shadow Deterministic Evidence Planner") — 같은
    # speaker/issue("persona_id:issue_id" 키)별 이전 shadow planner 선택 이력({"speaker",
    # "effective_issue_id", "chunk_id"}만 담는 최소 정보). API 응답(_serialize_state)에는
    # 노출하지 않는다 — 순수 내부 진단용 상태다. evidence_planner가 주입되지 않으면(기본,
    # ENABLE_IDEATION_EVIDENCE_PLANNER_SHADOW=False) 항상 빈 dict로 유지된다. 구버전 저장
    # state에는 이 키가 없을 수 있으므로 읽는 쪽은 항상 `.get("evidence_plan_shadow_history",
    # {})`로 접근한다(하위 호환).
    evidence_plan_shadow_history: dict[str, list[dict]]

    # 용준/Claude(2026-07-23, 요청: 근거 기반 자율 토론형 회의로 개편) — 쟁점(issue_id)별로
    # 보완 RAG 검색을 이미 시도했는지 기록한다("쟁점당 최대 1회"). 세션 전체에 걸쳐
    # 누적되고(라운드/이슈 변경으로 리셋되지 않는다), 같은 이슈가 다시 열려도 이미 시도한
    # 검색을 반복하지 않는다. 구버전 저장 state에는 이 키가 없을 수 있으므로 읽는 쪽은
    # 항상 `.get("supplemental_retrieval_issue_ids", [])`로 접근한다(하위 호환).
    supplemental_retrieval_issue_ids: list[str]
    # 이미 사용자에게 물은 결정 질문의 지문(issue_id+주제+missing_information 기반 해시) —
    # 같은 쟁점에서 같은(또는 의미상 유사한) 사유로 반복 질문하지 않도록 막는다(요청:
    # "세션 내 질문 fingerprint 또는 reason code를 기록"). 구버전 저장 state에는 이 키가
    # 없을 수 있으므로 읽는 쪽은 항상 `.get("asked_decision_fingerprints", [])`로 접근한다.
    asked_decision_fingerprints: list[str]


def _extract_initial_idea_text(user_idea: dict | str | None) -> str:
    """user_idea에서 trim된 초기 아이디어 텍스트를 뽑아낸다. dict({"description": ...})와
    plain str을 모두 받아들인다 — 호출부(ideation_conv_run.py::start_ideation_conversation)의
    기존 시그니처(user_idea: dict)를 그대로 유지하면서, 이 함수 안에서만 "trim 결과가
    비어 있는지"로 모드를 결정하기 위함이다(요청 2번: 서버가 trim 결과 기준으로 자동 결정)."""
    if isinstance(user_idea, dict):
        return str(user_idea.get("description") or "").strip()
    if isinstance(user_idea, str):
        return user_idea.strip()
    return ""


def build_roundtable_opening_message(idea_text: str, round_number: int = 1) -> ConvMessage:
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환): 라운드테이블 진입 직전
    진행자의 안건 제시 메시지를 만든다. LLM을 부르지 않는다 — 사용자가 이미 입력한 텍스트를
    그대로 인용해 안건으로 재진술할 뿐이라 사실 왜곡 위험이 없고, LLM 호출 상한을 소비하지
    않는다. speaker_name/role은 페르소나 카드 조회 없이 고정값을 쓴다
    (ideation_conv_run.py::_new_facilitator_message와 동일한 기존 관례)."""
    idea = (idea_text or "").strip() or "제출하신 아이디어"
    content = f"오늘은 '{idea}'에 대한 문제와 구현 범위를 논의하겠습니다."
    return ConvMessage(
        message_id=f"MSG-{uuid.uuid4().hex[:10]}",
        speaker_id="ideation_facilitator",
        speaker_name="회의 진행자",
        role="진행자",
        round=round_number,
        message_type="summary",
        content=content,
        referenced_message_ids=[],
        evidence=[],
        created_at=datetime.now(timezone.utc).isoformat(),
        structured=None,
    )


def initial_conv_state(
    session_id: str,
    notice_and_criteria: dict,
    user_idea: dict,
    max_rounds: int = 3,
    application_form_items: list[dict] | None = None,
) -> IdeationConvState:
    """준비 상태. user_idea(trim 결과)가 있으면 refinement로 시작한다 — 용준/Claude(2026-07-21,
    요청: 전문가 라운드테이블 전환) 진행자의 안건 제시 메시지(LLM 호출 없음, 위
    build_roundtable_opening_message 참고)를 messages에 먼저 넣고, phase는 더 이상
    "planning_question"(1:1 인터뷰 진입점)이 아니라 "expert_discussion"(라운드테이블
    진입점)이다 — 기획/개발 위원이 서로를 상대로 먼저 토론하고, 사용자에게 직접 질문하는
    것은 진행자만 한다. 비어 있으면 discovery로 시작해 후보 생성 단계(candidate_generation)
    부터 진행한다(요청 1~2번, 변경 없음). ideation_mode는 여기서 딱 한 번 결정되어 이후
    그래프 전체가 이 값을 그대로 읽는다."""
    initial_idea = _extract_initial_idea_text(user_idea)
    mode: IdeationMode = "refinement" if initial_idea else "discovery"
    opening_messages = [build_roundtable_opening_message(initial_idea, round_number=1)] if mode == "refinement" else []
    return IdeationConvState(
        session_id=session_id,
        notice_and_criteria=notice_and_criteria,
        user_idea={"description": initial_idea} if initial_idea else {},
        round=1,
        max_rounds=max_rounds,
        messages=opening_messages,
        phase="expert_discussion" if mode == "refinement" else "candidate_generation",
        pending_question=None,
        pending_expected_answer_type=None,
        pending_question_topic=None,
        resolved_topics=[],
        consensus=[],
        unresolved_issues=[],
        idea_proposal=None,
        idea_canvas=None,
        application_form_items=application_form_items or [],
        failed_node=None,
        llm_calls_used=0,
        answer_retry_count=0,
        ideation_mode=mode,
        initial_idea=initial_idea or None,
        contest_analysis=None,
        idea_candidates=[],
        original_idea_candidates=[],
        selected_idea=None,
        selected_idea_document_id=None,
        selection_reason=None,
        candidate_regeneration_count=0,
        selection_intent=None,
        user_selection_message=None,
        source_candidates=[],
        merge_analysis=None,
        discussion_rounds=[],
        discussion_planning_position=None,
        discussion_development_review=None,
        discussion_revised_proposal=None,
        discussion_next_action=None,
        discussion_review_stance=None,
        open_issues=[],
        resolved_issues=[],
        active_issue_id=None,
        previous_speaker=None,
        expert_turn_count=0,
        stop_reason=None,
        forced_next_speaker=None,
        interjection_target_speaker_id=None,
        interjection_response_message_id=None,
        required_counterpart_speaker_id=None,
        counterpart_review_completed=True,
        next_route=None,
        consecutive_zero_linked_turns=0,
        consecutive_expert_judgment_only_turns=0,
        last_missing_information=[],
        consecutive_repeated_missing_information_turns=0,
        last_new_information_text="",
        consecutive_no_new_information_turns=0,
        evidence_plan_shadow_history={},
        supplemental_retrieval_issue_ids=[],
        asked_decision_fingerprints=[],
    )


def apply_user_answer(previous_state: IdeationConvState, answer_message: ConvMessage) -> IdeationConvState:
    """awaiting_planning_answer 또는 awaiting_developer_answer 상태에 사용자 답변
    메시지를 추가하고, 다음에 실행할 노드를 가리키는 phase로 전환한다.

    다음 phase 결정: 이 함수를 부르기 전 상태(phase)만으로 결정되며 LLM 판단을
    거치지 않는다 — "사용자가 답하지 않은 내용을 임의로 확정하지 않는다"는 요구와
    별개로, 애초에 다음에 어느 전문가 차례인지는 사용자 판단이 개입할 여지가 없는
    고정 순서(기획 질문 -> 개발 질문 -> 두 전문가 보완)이기 때문이다.
    """
    prev_phase = previous_state["phase"]
    next_phase: ConvPhase
    if prev_phase == "awaiting_candidate_selection":
        # 용준/Claude(2026-07-21): discovery 모드 — 사용자가 후보 선택/결합/재추천/전문가
        # 추천 중 하나로 답했다. 실제 해석(번호 선택인지, 결합인지, 재추천인지)은 이
        # 함수가 하지 않는다 — candidate_selection 노드가 담당한다(요청: 단순 선택은
        # 코드로 결정적으로, 자연어 결합/수정은 LLM으로).
        next_phase = "candidate_selection"
    elif prev_phase == "awaiting_planning_answer":
        next_phase = "developer_question"
    elif prev_phase == "awaiting_developer_answer":
        next_phase = "expert_discussion"
    elif prev_phase in {"awaiting_user_decision", "discussion_complete"}:
        # 요청 8번 "필요한 경우 추가 질문 라운드" — 시스템이 스스로 판단해 다음 라운드로
        # 넘어가는 경우(next_action="continue_round")와 별개로, 사용자가 확정 버튼을
        # 누르지 않고 자유롭게 한 마디 더 남기면 그 발언도 두 전문가의 보완 의견 대상이
        # 된다. round는 새로 늘리지 않는다 — 새 질문 사이클이 시작된 게 아니라 같은
        # 라운드의 대화가 이어지는 것이기 때문이다.
        next_phase = "expert_discussion"
    else:
        raise ValueError(f"사용자 답변을 받을 수 없는 phase입니다: {prev_phase!r}")

    return IdeationConvState(
        **{
            **previous_state,
            "messages": previous_state["messages"] + [answer_message],
            "phase": next_phase,
            "pending_question": None,
            "pending_expected_answer_type": None,
            "pending_question_topic": None,
            # 다음 단계로 실제로 넘어가는 시점이므로 재질문 카운터를 리셋한다(새 쟁점 시작).
            "answer_retry_count": 0,
            # 용준/Claude(2026-07-22, 요청: 반복 감지 카운터는 사용자가 실제로 새 정보를
            # 제공하면 리셋) — 사용자가 방금 근거 부족/반복 질문에 답했으므로, 다음 전문가
            # 발언은 이 새 답변을 근거로 다시 시작해야 한다(이전 반복 이력이 그대로 남아
            # 곧바로 다시 사용자에게 되묻는 것을 막는다).
            "consecutive_zero_linked_turns": 0,
            "consecutive_expert_judgment_only_turns": 0,
            "last_missing_information": [],
            "consecutive_repeated_missing_information_turns": 0,
            "last_new_information_text": "",
            "consecutive_no_new_information_turns": 0,
        }
    )


def request_finalize(previous_state: IdeationConvState) -> IdeationConvState:
    """사용자가 '주제 확정하고 초안 받기'를 눌렀을 때만 호출된다(요구 9~10번 —
    전문가/진행자가 임의로 최종 확정하지 않는다). phase="awaiting_user_decision"이 아니면
    호출부(API)가 이 함수를 부르기 전에 이미 막아야 한다."""
    if previous_state["phase"] not in {"awaiting_user_decision", "discussion_complete"}:
        raise ValueError(
            "awaiting_user_decision 또는 discussion_complete 상태에서만 최종 확정할 수 "
            f"있습니다(현재: {previous_state['phase']!r})."
        )
    return IdeationConvState(**{**previous_state, "phase": "finalizing"})


class IdeationCancelled(Exception):
    """용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소): 사용자가 "잠시만"으로 진행 중인
    요청을 취소했을 때, 스트리밍 llm_call이 던지는 전용 예외. 일반 LLM 오류(RuntimeError 등)와
    달리 _safe_call_structured_json/_safe_call_json이 재시도하지 않고 그대로 상위(그래프
    실행)까지 전파해야 한다 — 재시도하면 이미 끊긴 OpenAI 스트림에 다시 과금 요청을 보내는
    낭비가 생기고, phase="failed"로 만들면 "취소는 일반 오류가 아니다"라는 요구를 어기게
    된다."""

    def __init__(self, session_id: str, request_id: str | None = None):
        super().__init__(f"[{session_id}] 사용자가 요청(request_id={request_id})을 취소했습니다.")
        self.session_id = session_id
        self.request_id = request_id
        # ideation_conv_run.py::_drive_graph가 취소 시점까지 완료된 마지막 그래프 스냅샷을
        # 실어 보낸다 — 완료된 발언이 하나도 없으면(첫 노드 실행 중 취소) None 그대로 둔다.
        self.partial_state: "IdeationConvState | None" = None


def is_graph_entry_phase(phase: str) -> bool:
    """그래프가 이 phase로 새로 진입해 노드를 실행해도 되는지 여부.
    awaiting_*/finalized/failed/awaiting_user_decision은 API가 그래프를 다시 부르지
    않고 사용자 입력을 기다려야 하는 지점이다."""
    return phase in _TERMINAL_ENTRY_PHASES or phase == "finalizing"
