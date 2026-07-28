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
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from .application_form_draft import initialize_application_form_draft

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
    # 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) —
    # discovery 모드 전용 "확정 이전" 단계 8개. candidate_generation(기존, 이제
    # idea_conflict_and_merge 이후 "압축된 provisional 후보 생성"으로 재정의)보다
    # 앞에 온다. refinement 모드(초기 아이디어를 이미 입력한 세션)는 이 phase들을
    # 전혀 거치지 않는다(initial_conv_state가 mode별로 진입 phase 자체를 분기).
    "problem_discovery",
    "awaiting_problem_focus_selection",
    "problem_focus_selection",
    "idea_divergence",
    "idea_conflict_and_merge",
    "awaiting_conflict_resolution",
    "conflict_resolution",
    # 용준/Claude(2026-07-27): awaiting_candidate_selection 이후 사용자의 응답을 처리하는
    # discovery 전용 진입점 — 기존 "candidate_selection"(즉시 확정) 대신 이 값을 쓴다.
    # 기존 candidate_selection phase/노드 자체는 코드에서 지우지 않았다(직접 호출하는
    # 기존 단위 테스트가 계속 통과해야 한다) — 다만 이 그래프의 정상 흐름에서는 더 이상
    # apply_user_answer가 그 값으로 전이시키지 않는다.
    "provisional_selection",
    "idea_validation",
    "awaiting_concept_confirmation",
    "concept_confirmation",
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
    # 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — API가
    # apply_user_answer를 거쳐 새로 진입시킬 수 있는 discovery 전용 엔트리 4개.
    # idea_divergence/idea_validation은 각각 problem_focus_selection/candidate_selection
    # 안에서 같은 그래프 호출 중에 자동으로 이어지는 노드일 뿐, API가 phase 값으로 직접
    # 재진입시키는 지점이 아니므로 여기 포함하지 않는다(_ENTRY_NODES 주석 참고).
    "problem_discovery",
    "problem_focus_selection",
    "conflict_resolution",
    "provisional_selection",
    "concept_confirmation",
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
# 가치가 정리되지 않았는데 로드맵부터 질문"하거나 "한 질문에서 여러 쟁점을
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

# 공모전 적합성(contest_fit)은 후보 생성·최종 캔버스에는 계속 보존하지만, 전문가 회의의
# 자동 의제로는 열지 않는다. 공고문의 결과 발표·접수 일정 같은 행정 문장을 심사 기준으로
# 오인해 억지로 인용하는 실사용 실패가 확인됐기 때문이다. TOPIC_PRIORITY는 과거 세션의
# issue_id 해석과 canonical family 판정을 위해 그대로 두고, 자동 질문·로테이션만 이 목록을
# 사용한다.
DISCUSSION_TOPIC_PRIORITY: tuple[str, ...] = tuple(
    topic for topic in TOPIC_PRIORITY if topic != "contest_fit"
)

# roadmap(확장 기능/도입 순서)은 문제·사용자·가치·MVP가 모두 정리된 뒤에만 질문한다.
# contest_fit은 자동 회의 의제에서 빠졌으므로 선행 조건에서도 제외한다.
ROADMAP_PREREQUISITE_TOPICS: frozenset[str] = frozenset({"problem", "target_user", "core_value", "mvp"})


def remaining_topics_for(resolved_topics: list[str] | None) -> list[str]:
    """아직 확인되지 않은 주제를 우선순위 순서로 반환한다. roadmap의 선행 주제
    (ROADMAP_PREREQUISITE_TOPICS)가 모두 resolved_topics에 없으면 roadmap 자체를 목록에서
    제외한다 — 질문 노드가 애초에 roadmap을 고를 수 없는 후보 목록만 보게 하는 것이,
    "질문 규칙으로만 금지"하는 것보다 더 확실한 강제 방법이다. resolved_topics가 None이면
    (구버전 state) 빈 리스트로 취급한다(하위 호환)."""
    resolved_set = set(resolved_topics or [])
    remaining = [topic for topic in DISCUSSION_TOPIC_PRIORITY if topic not in resolved_set]
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
    "problem_discovery": "candidate_discovery",
    "awaiting_problem_focus_selection": "candidate_discovery",
    "problem_focus_selection": "candidate_discovery",
    "idea_divergence": "candidate_discovery",
    "idea_conflict_and_merge": "candidate_discovery",
    "awaiting_conflict_resolution": "candidate_discovery",
    "conflict_resolution": "candidate_discovery",
    "candidate_generation": "candidate_discovery",
    "awaiting_candidate_selection": "candidate_discovery",
    "candidate_selection": "candidate_selection",
    "provisional_selection": "candidate_selection",
    # 용준/Claude(2026-07-27): 검증·확정 대기는 "이미 후보를 고른 뒤" 단계라 refinement에
    # 더 가깝지만, 아직 idea_locked=False라 refinement(전문가 라운드테이블)로 넘기면
    # 프론트 배지가 "확정된 아이디어를 다듬는 중"으로 오인시킨다 — candidate_selection과
    # 같은 버킷(선택/확정 처리 중)으로 묶는다.
    "idea_validation": "candidate_selection",
    "awaiting_concept_confirmation": "candidate_selection",
    "concept_confirmation": "candidate_selection",
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


# 용준/Claude(2026-07-27, 요청: "저장된 구버전(2026-07-27 문제 발견 단계 도입 이전) discovery
# 세션이 브라우저 세션 재개로 계속 다시 뜬다 — 자동으로 폐기해 달라") — 개편 이전 discovery
# 세션은 problem_discovery/problem_focus_selection/problem_definition/idea_divergence/
# idea_conflict_and_merge를 전혀 거치지 않고 candidate_generation부터 바로 시작했다. 반면
# 개편 이후 세션은 candidate_generation(now candidate_planning)에 도달하기 전에 반드시
# problem_definition 노드를 거치므로 problem_definition이 항상 채워져 있다 — 이 차이가
# "이 세션이 신규 파이프라인을 거쳤는지"를 저장된 state만 보고 결정론적으로 구분할 수 있는
# 유일한 신호다(phase나 round 값만으로는 재추천을 여러 번 거친 신규 세션과 구분할 수 없다).
# provisional_selection이 만드는 phase 값(idea_validation 이후 단계들)은 이미
# problem_definition을 반드시 거쳐야만 도달하므로 대상에 포함하지 않는다 — candidate_generation/
# awaiting_candidate_selection 두 phase만 "구버전에서 바로 시작했을 수 있는" 지점이다.
_LEGACY_PRE_PROBLEM_STAGE_ENTRY_PHASES = frozenset({"candidate_generation", "awaiting_candidate_selection"})


def is_legacy_pre_problem_stage_discovery_session(state: dict) -> bool:
    """저장된 discovery 세션이 문제 발견 단계 개편(2026-07-27) 이전에 candidate_generation
    으로 곧바로 시작한 구버전 세션인지 판정한다. True면 호출부(백엔드 세션 조회 API)가 이
    세션을 재개 대상에서 제외해 프론트가 새 problem_discovery 세션을 시작하게 해야 한다.
    refinement 세션(ideation_mode != "discovery")과, 이미 problem_definition을 거친 신규
    discovery 세션은 항상 False다."""
    if state.get("ideation_mode") != "discovery":
        return False
    if state.get("phase") not in _LEGACY_PRE_PROBLEM_STAGE_ENTRY_PHASES:
        return False
    return not state.get("problem_definition")


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


# 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — discovery
# 모드가 candidate_generation 이전에 거치는 "문제 발견 -> 문제 정의 -> 아이디어 발산 ->
# 반론·결합 -> (기존 후보 압축/선택) -> 검증 -> 확정" 단계의 자료 구조. 기존 필드와 의미가
# 겹치는 것(예: 후보의 differentiation/contest_fit/target_user)은 새로 만들지 않고
# idea_candidates/selected_idea 등 기존 구조를 그대로 재사용한다 — 아래는 그 기존 구조로는
# 표현할 수 없는, "아직 후보가 되기 전" 단계의 신규 개념만 담는다.


class ProblemArea(TypedDict):
    """problem_discovery 노드가 만드는 문제 영역 카드. 완성된 서비스 이름/기능을 포함하지
    않는다(요청: "완성된 서비스 이름이나 기능을 제시하지 않는다") — problem_definition 이후
    에야 구체화된다."""

    area_id: str
    title: str
    summary: str
    who_is_affected: str


class ProblemDefinition(TypedDict):
    """problem_focus_selection 이후 problem_definition 노드가 채우는 구조화된 문제 정의.
    problem_defined/target_user_defined/existing_limitations_defined(요청 5번 boolean들)는
    이 dict의 필드 존재 여부로 코드가 결정론적으로 계산한다(아래 problem_defined() 등 참고) —
    LLM이 boolean을 직접 반환하지 않는다."""

    problem: str
    target_user: str
    user_context: str
    root_cause: str
    existing_solution: str
    existing_limitations: str


class SolutionDirection(TypedDict):
    """idea_divergence 노드가 만드는 해결 방향 1개. 후보(candidate)와 달리 제목/기능을
    확정하지 않는다 — "해결 원리 자체가 다른" 방향을 나타내는 최소 정보만 담는다(요청:
    "단순히 기능만 다른 것이 아니라 해결 접근 방식 자체가 달라야 한다"). idea_conflict_and_merge가
    status를 "active"/"dropped"/"merged"로 갱신한다(폐기/결합 기록은 idea_evolution에
    남긴다 — 이 필드는 "지금 시점의 유효 상태"만 나타낸다)."""

    direction_id: str
    title: str
    core_principle: str  # 해결 원리(기능이 아니라 접근 방식) 1~2문장
    mechanism: str
    target_user_fit: str
    status: Literal["active", "dropped", "merged", "superseded", "revised"]
    origin_direction_ids: list[str]  # 결합으로 생성됐다면 원본 direction_id들, 아니면 빈 리스트
    parent_direction_ids: NotRequired[list[str]]
    strengths: NotRequired[list[str]]
    open_assumptions: NotRequired[list[str]]


class IdeaEvolutionRecord(TypedDict):
    """요청: "회의 중 아이디어가 어떻게 변했는지 사용자가 확인할 수 있도록 변화 기록을
    저장". idea_conflict_and_merge/idea_validation/concept_confirmation이 append한다
    (operator.add 리듀서 — messages/discussion_rounds와 같은 원칙). critique_count/
    merge_or_revision_count(요청 5번)는 이 리스트에서 action_type별 개수를 세어 코드가
    계산한다 — LLM이 카운트를 직접 반환하지 않는다(아래 critique_count()/merge_or_revision_count()
    참고)."""

    record_id: str
    stage: str  # ConvPhase 중 이 기록이 발생한 단계(예: "idea_conflict_and_merge")
    # 용준/Claude(2026-07-27, 요청: "problem focus 변화 이력 추가") — problem_focus_selected/
    # problem_focus_merged는 problem_focus_selection 노드가 select_problem_focus/
    # combine_problem_focus 액션이 성공한 시점에 남긴다. 기존 7종은 그대로 유지한다.
    action_type: Literal[
        "divergence",
        "critique",
        "merge",
        "revision",
        "drop",
        "candidate_generation",
        "validation",
        "confirmation",
        "problem_focus_selected",
        "problem_focus_merged",
    ]
    title: str
    content: str
    changed_by: str  # persona_id 또는 "user"
    # 이 발언/변경이 가리키는 id(들) — solution_direction 단계에서는 solution_direction_id,
    # problem_focus_selected/merged에서는 area_id를 담는다(필드 이름은 기존 그대로 재사용 —
    # "이 기록이 가리키는 대상 id 목록"이라는 의미는 동일하다).
    target_direction_ids: list[str]
    before: NotRequired[Any]
    after: NotRequired[Any]
    result_direction_id: NotRequired[str]
    created_at: str


class ValidationResult(TypedDict):
    """idea_validation 노드가 채우는 검증 결과. planning_validation_completed/
    technical_validation_completed(요청 5·7번)는 이 dict의 해당 섹션이 채워졌는지로
    코드가 판단한다 — LLM이 "완료 여부" boolean을 직접 반환하지 않는다."""

    planning: dict | None  # {value_worth_solving, target_user_clarity, differentiation, usage_motivation, contest_alignment}
    technical: dict | None  # {data_availability, feasibility, ai_necessity, privacy_or_security_risks, prototype_feasibility}


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
    # 진행자 v02가 매 턴 draft_patch로 갱신하는 신청 양식 초안. 원본 양식 항목은 보존하고
    # 별도 상태로 관리하므로 구 프롬프트로 롤백해도 application_form_items 계약은 바뀌지 않는다.
    application_form_draft: list[dict]
    # 가은/Claude(2026-07-27, 요청: "보완이 필요한 정보 섹션") — generate_application_form_draft()가
    # 신청서 초안을 쓰면서 입력(idea_proposal/existing_draft)에 없어 본문에 넣지 못한 항목을
    # 여기에 남긴다. 구버전 저장 state에는 이 키가 없을 수 있으므로 읽는 쪽은 항상
    # `.get("application_form_supplement_notes", [])`로 접근한다(하위 호환).
    application_form_supplement_notes: list[str]
    failed_node: str | None
    llm_calls_used: int
    # 용준/Claude(2026-07-20): 같은 쟁점(pending_question)으로 재질문한 횟수. 사용자가
    # 질문에 답할 때마다 answer_sufficiency 판정을 거치는데, 무한 재질문을 막기 위해
    # 이 값이 retry_cap(ideation_conv_run.py::_MAX_ANSWER_RETRY)에 도달하면 판정 결과와
    # 무관하게 다음 단계로 강제 진행한다. 재질문이 아니라 다음 단계로 넘어갈 때마다 0으로
    # 리셋된다(쟁점이 바뀌었으므로).
    answer_retry_count: int

    # 용준/Claude(2026-07-27, 요청: "진행자 질문이 무한 반복된다" 버그 수정) — 진행자가
    # awaiting_user_decision에서 던진 질문(pending_question)에 사용자가 방금 답한 직후,
    # apply_user_answer가 그 질문 텍스트를 여기 보존한다. discussion_facilitator가 그 다음
    # 턴에 또 사용자 결정 질문을 만들면 이 값과 의미가 같은지(_looks_like_restatement)
    # 비교해 "이미 답변받은 질문을 다시 묻는지" 판단하는 데 쓴다. 사용자가 답하지 않은
    # phase 전환에서는 항상 None으로 되돌아간다(관련 없는 질문과 잘못 비교하지 않도록).
    last_answered_facilitator_question: str | None
    # 위 last_answered_facilitator_question과 의미가 같은 질문을 진행자가 연속으로 다시
    # 만든 횟수. answer_retry_count(전문가 질문 쪽 재질문 상한)와 대칭되는 안전장치 —
    # discussion_facilitator가 이 값이 상한(_MAX_FACILITATOR_DECISION_REPEAT)에 도달하면
    # 판정과 무관하게 방금 받은 사용자 답을 그대로 받아들이고 다음 단계로 강제 진행한다.
    # 새 질문(의미가 다름)을 만들면 0으로 리셋된다.
    facilitator_decision_repeat_count: int

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

    # 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — discovery
    # 모드가 candidate_generation(이제 "압축된 provisional 후보 생성"으로 재정의)보다 먼저
    # 거치는 문제 발견/정의/발산/반론·결합 단계 전용 필드. refinement 모드에서는 전부
    # 초기값에서 바뀌지 않는다(ideation_mode="discovery"일 때만 채워진다). 구버전 저장
    # state에는 이 키들이 없을 수 있으므로 읽는 쪽은 항상 `.get(...)`로 접근한다.
    problem_areas: list[ProblemArea]
    # 사용자가 선택한 문제 영역 원본(최대 2개) — "결합해서 탐색"이면 둘 다 담긴다. 이
    # 시점에는 아직 problem_definition이 없다(요청 3번: "문제 영역 선택"과 "문제 정의"는
    # 상태상 구분되어야 한다 — problem_focus는 선택 그 자체, problem_definition은 그
    # 선택을 구체화한 결과).
    problem_focus: list[ProblemArea]
    # "다른 문제 제안 요청" 횟수 — MAX_PROBLEM_REGENERATIONS(ideation_conv_problem.py)
    # 도달 시 더 이상 LLM을 호출해 문제 영역을 재생성하지 않는다(candidate_regeneration_count와
    # 동일한 원칙).
    problem_regeneration_count: int
    # problem_definition 노드가 채우는 구조화된 문제 정의. problem_defined()/
    # target_user_defined()/existing_limitations_defined()가 이 필드의 부분 필드 존재
    # 여부로 코드가 결정론적으로 계산한다(요청 5번) — None이면 아직 문제 정의가 안 된 것.
    problem_definition: ProblemDefinition | None
    # idea_divergence가 만들고 idea_conflict_and_merge가 갱신하는 해결 방향 목록.
    # solution_direction_count(요청 5·6번 전환 조건)는 status="active"인 항목 수를 코드가
    # 직접 센다 — LLM이 개수를 직접 보고하지 않는다.
    solution_directions: list[SolutionDirection]
    # 요청: "회의 중 아이디어가 어떻게 변했는지 사용자가 확인할 수 있도록 변화 기록을
    # 저장" — divergence/critique/merge/revision/drop/validation/confirmation을 시간순으로
    # 누적한다(operator.add — messages와 같은 원칙).
    idea_evolution: Annotated[list[IdeaEvolutionRecord], operator.add]
    # idea_conflict_and_merge가 실행된 라운드 수(사용자 개입 없이 자동 반복되는 라운드 —
    # MAX_CONFLICT_ROUNDS 상한 판단에 쓴다. discussion_rounds/round와는 별개 카운터).
    conflict_round_count: int
    # 용준/Claude(2026-07-28, 요청: "위원들이 결합하는 방식으로" 카드 선택 단계 제거) —
    # provisional_from_merge가 카드 선택 없이 바로 idea_validation으로 넘어가면서,
    # "검증 실패 -> idea_conflict_and_merge 재실행 -> 조건 재충족 -> 검증 -> 실패 -> ..."
    # 순환을 끊어줄 자연스러운 정지점(예전에는 candidate_feasibility가 매번 멈춰줬다)이
    # 사라졌다. 이 카운터가 그 정지점을 대신한다 — make_technical_validation_node가
    # needs_revision일 때마다 늘리고, MAX_VALIDATION_REVISE_ROUNDS(ideation_conv_state.py)에
    # 도달하면 더 이상 자동으로 idea_conflict_and_merge로 되돌아가지 않고
    # awaiting_concept_confirmation에서 사용자가 직접 재검토/확정을 판단하게 한다.
    # provisional_from_merge/provisional_selection이 새 provisional_idea를 채택할 때마다
    # 0으로 리셋된다(요청: 이전 후보의 재시도 횟수가 새 후보에 넘어오면 안 된다).
    validation_revise_count: int
    # candidate_selection이 골라낸 "검증 대상 잠정 후보" — concept_confirmation에서
    # 사용자가 최종 확정하기 전까지는 이 필드만 채워지고 selected_idea는 그대로 None이다
    # (요청 3번: "잠정 후보 선택"과 "최종 아이디어 확정"은 상태상 구분되어야 한다).
    provisional_idea: dict | None
    # idea_validation이 채우는 기획/개발 관점 검증 결과. planning_validation_completed/
    # technical_validation_completed(요청 5·7번)는 이 dict의 해당 섹션이 채워졌는지로
    # 코드가 판단한다.
    validation_result: ValidationResult | None
    # 사용자가 concept_confirmation에서 "확정"으로 답했는지 — request_finalize와 달리
    # 이 값은 오직 API 레이어가 사용자의 명시적 확정 응답을 파싱했을 때만 코드가 True로
    # 세팅한다(LLM이 직접 True를 반환하지 않는다, 요청 5번).
    user_confirmed: bool
    # 요청 2·3번 — concept_confirmation에서 사용자가 최종 확정한 순간에만 True가 된다.
    # discovery 모드의 provisional_idea/candidate_selection 단계에서는 계속 False다.
    # refinement 모드(초기 아이디어를 이미 입력한 세션)는 이 값이 처음부터 False이고,
    # request_finalize() 호출 시점(회의 결과 정리·종료를 요청하는 기존 지점)에 True가 된다
    # — concept_confirmation(discovery 전용, "방향 확정")과 finalizing/finalized(공통,
    # "회의 결과 정리·종료")는 서로 다른 개념이므로 역할이 겹치지 않는다.
    idea_locked: bool

    # 용준/Claude(2026-07-27, 후속 요청 4번: "2차 프론트에서는 action code를 함께 보낼
    # 예정 — 백엔드는 action code를 우선 사용하고 자연어 키워드 판정은 하위 호환용
    # 폴백으로 유지") — API 레이어(ideation_conv_run.py::reply_ideation_conversation)가
    # 이번 한 번의 그래프 호출 동안만 채우는 임시(transient) 필드다. 세션에 영구
    # 저장되는 값이 아니라 "이번 사용자 응답을 이렇게 해석하라"는 1회성 지시이므로,
    # apply_user_answer가 전이시킨 phase의 노드가 이 값을 소비한 뒤에는 다음 요청에
    # 잔류하지 않도록 각 노드가 스스로 None으로 되돌린다(다른 transient 필드인
    # forced_next_speaker/next_route와 같은 패턴). {"code": str, "payload": dict} 형태 —
    # code가 그 phase에서 기대하는 값과 다르면 무시하고 기존 자연어 파싱으로 폴백한다.
    pending_user_action: dict | None

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
    # 재인/Claude(2026-07-23, 아바타 페이싱 연동): continue_ideation_expert_turn이 다음 그래프
    # 진입을 특정 전문가/진행자로 강제 지정할 때 채운다 — 해당 노드가 실행되자마자 None으로
    # 리셋되어 다음 라운드에 잔류하지 않는다.
    forced_next_speaker: str | None

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

    # 용준/Claude(2026-07-27, RAG-007 연결) — candidate_planning/candidate_feasibility가
    # 검색한 외부 통계·시장·정책 참고자료(ai/rag/orchestration/ideation_external_evidence_service.py
    # 참고). RAG-006 evidence_lookup 결과(ConvMessage.evidence, 프로젝트 문서 근거)와는 완전히
    # 분리된 필드다 — 두 후보 노드가 각자 검색한 결과를 (source_id, document_id, chunk_id)
    # 기준으로 중복 없이 누적한다(discussion_rounds처럼 operator.add 리듀서를 쓰지 않고 노드가
    # 직접 병합해 반환한다 — 두 노드가 같은 요청 안에서 연속 실행되므로 하나의 정확한 값만
    # 필요하다). use_rag=False거나 검색 결과가 없으면 빈 리스트다. 구버전 저장 state에는 이
    # 키가 없을 수 있으므로 읽는 쪽은 항상 `.get("external_evidence", [])`로 접근한다.
    external_evidence: list[dict]
    # external_evidence 검색의 응답 단위 메타데이터 — used_dataset_search/used_public_api_search
    # (bool)와 warnings(list[str], 예: 출처 미확인으로 제외된 건수, 도메인 폴백 여부). 구버전
    # 저장 state에는 이 키가 없을 수 있으므로 읽는 쪽은 항상 `.get("external_evidence_meta", {})`
    # 로 접근한다.
    external_evidence_meta: dict


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
    것은 진행자만 한다.

    용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — 비어 있으면
    discovery로 시작하되, 더 이상 후보 생성 단계(candidate_generation)부터 바로 시작하지
    않는다. candidate_generation은 이제 "문제 발견(problem_discovery) -> 문제 정의 ->
    아이디어 발산(idea_divergence) -> 반론·결합(idea_conflict_and_merge)"을 거친 뒤 그
    결과를 압축해 provisional 후보를 만드는 단계로 재정의됐다 — 이 함수가 그 진입점을
    "problem_discovery"로 바꾼 것 외에 discovery 모드의 다른 동작(1차 개편 범위: refinement는
    영향 없음)은 바뀌지 않는다. ideation_mode는 여기서 딱 한 번 결정되어 이후 그래프
    전체가 이 값을 그대로 읽는다."""
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
        phase="expert_discussion" if mode == "refinement" else "problem_discovery",
        pending_question=None,
        pending_expected_answer_type=None,
        pending_question_topic=None,
        resolved_topics=[],
        consensus=[],
        unresolved_issues=[],
        idea_proposal=None,
        idea_canvas=None,
        application_form_items=application_form_items or [],
        application_form_draft=initialize_application_form_draft(application_form_items),
        application_form_supplement_notes=[],
        failed_node=None,
        llm_calls_used=0,
        answer_retry_count=0,
        last_answered_facilitator_question=None,
        facilitator_decision_repeat_count=0,
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
        problem_areas=[],
        problem_focus=[],
        problem_regeneration_count=0,
        problem_definition=None,
        solution_directions=[],
        idea_evolution=[],
        conflict_round_count=0,
        validation_revise_count=0,
        provisional_idea=None,
        validation_result=None,
        user_confirmed=False,
        idea_locked=False,
        pending_user_action=None,
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
        external_evidence=[],
        external_evidence_meta={},
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
    if prev_phase == "awaiting_problem_focus_selection":
        # 용준/Claude(2026-07-27): discovery 모드 — 사용자가 문제 영역 선택/결합/다른 문제
        # 요청/직접 입력 중 하나로 답했다. 실제 해석은 problem_focus_selection 노드가
        # 담당한다(candidate_selection과 동일한 원칙 — 단순 선택은 코드로, 나머지는 필요시
        # LLM으로).
        next_phase = "problem_focus_selection"
    elif prev_phase == "awaiting_conflict_resolution":
        # idea_conflict_and_merge가 라운드 상한(MAX_CONFLICT_ROUNDS)에 도달했는데도 최소
        # 조건(요청 6번: 해결 방향 3개↑/반론 1회↑/수정·결합 1회↑)을 못 채웠을 때만 여기서
        # 멈춘다 — 사용자의 결합/방향추가/방향폐기/검증진행 요청을 conflict_resolution
        # 노드가 해석한다.
        next_phase = "conflict_resolution"
    elif prev_phase == "awaiting_concept_confirmation":
        # idea_validation 이후 진행자가 "이 방향으로 확정할지, 더 수정할지" 물은 지점 —
        # 사용자의 확정/재수정 요청을 concept_confirmation 노드가 해석한다. 이 판단은
        # LLM이 아니라 이 노드(코드)가 결정적으로 내린다(요청 5번: user_confirmed/
        # idea_locked는 LLM이 직접 세팅하지 않는다).
        next_phase = "concept_confirmation"
    elif prev_phase == "awaiting_candidate_selection":
        # 용준/Claude(2026-07-27, 요청: "선택 즉시 확정" 구조 개편) — 이 시점에 확정되는
        # 것은 더 이상 selected_idea가 아니라 provisional_idea다(요청 3번: "잠정 후보
        # 선택"과 "최종 아이디어 확정"은 상태상 구분되어야 한다). 실제 해석(번호 선택인지,
        # 결합인지, 재추천인지)은 이 함수가 하지 않는다 — provisional_selection 노드가
        # 기존 candidate_selection 노드를 그대로 감싸 재사용하며 결과만 provisional_idea로
        # 재해석한다(ideation_conv_problem.py::make_provisional_selection_node 참고,
        # 기존 candidate_selection 노드/프롬프트 자체는 전혀 수정하지 않았다).
        next_phase = "provisional_selection"
    elif prev_phase == "awaiting_planning_answer":
        next_phase = "developer_question"
    elif prev_phase == "awaiting_developer_answer":
        next_phase = "expert_discussion"
    elif prev_phase == "discussion_complete":
        # 요청 8번 "필요한 경우 추가 질문 라운드" — 시스템이 스스로 판단해 다음 라운드로
        # 넘어가는 경우(next_action="continue_round")와 별개로, 사용자가 확정 버튼을
        # 누르지 않고 자유롭게 한 마디 더 남기면 그 발언도 두 전문가의 보완 의견 대상이
        # 된다. round는 새로 늘리지 않는다 — 새 질문 사이클이 시작된 게 아니라 같은
        # 라운드의 대화가 이어지는 것이기 때문이다.
        next_phase = "expert_discussion"
    elif prev_phase == "awaiting_user_decision":
        # 2026-07-26 라운드테이블 재설계(진행자 주도 사이클): 이 phase는 진행자가 방금
        # 선택지/질문을 던지고 사용자 결정을 기다리던 지점이다. 사용자가 답했다는 것은
        # "이번 사이클의 사용자 참여"가 끝났다는 뜻이므로, 전문가를 다시 거치지 않고
        # 곧장 진행자에게 돌려준다(전문가는 진행자가 질문을 던지기 전에만 개입한다는
        # 목표 루프). phase 자체는 그래프 진입점 표(_ENTRY_NODES)와 맞추기 위해 여전히
        # "expert_discussion"으로 두고, forced_next_speaker로 실제 목적지만 바꾼다 —
        # ideation_conv_build.py::_route_entry의 _FORCED_SPEAKER_TO_NODE["facilitator"]가
        # discussion_facilitator로 직접 진입시킨다.
        next_phase = "expert_discussion"
    else:
        raise ValueError(f"사용자 답변을 받을 수 없는 phase입니다: {prev_phase!r}")

    forced_next_speaker = "facilitator" if prev_phase == "awaiting_user_decision" else None
    # 용준/Claude(2026-07-27, 진행자 질문 무한 반복 버그 수정): 사용자가 방금 답한 질문이
    # 진행자가 던진 것(awaiting_user_decision)이었을 때만 그 질문 텍스트를 보존한다 —
    # discussion_facilitator가 다음 턴에 "이미 답변받은 질문을 다시 묻는지" 비교할 대상이다.
    # 다른 phase 전환(전문가 질문 답변 등)에서는 관련 없는 이전 값이 남지 않도록 None으로
    # 되돌린다.
    last_answered_facilitator_question = (
        previous_state.get("pending_question") if prev_phase == "awaiting_user_decision" else None
    )

    return IdeationConvState(
        **{
            **previous_state,
            "messages": previous_state["messages"] + [answer_message],
            "phase": next_phase,
            "forced_next_speaker": forced_next_speaker,
            "pending_question": None,
            "pending_expected_answer_type": None,
            "pending_question_topic": None,
            "last_answered_facilitator_question": last_answered_facilitator_question,
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
    호출부(API)가 이 함수를 부르기 전에 이미 막아야 한다.

    용준/Claude(2026-07-27, 요청: concept_confirmation과 finalizing/finalized의 역할
    분리) — idea_locked=True를 여기서도 세팅한다(이미 True인 discovery 세션에는 아무
    영향이 없다). refinement 모드는 concept_confirmation을 거치지 않으므로, "회의 결과를
    문서로 굳혀도 되는 시점"이 이 함수 호출 시점 하나뿐이다 — 이 시점 이전까지는
    idea_locked=False가 유지되어(요청 1번의 refinement 공통 가드) 신청서/사업계획서 작성
    언급을 차단하는 근거가 된다."""
    if previous_state["phase"] not in {"awaiting_user_decision", "discussion_complete"}:
        raise ValueError(
            "awaiting_user_decision 또는 discussion_complete 상태에서만 최종 확정할 수 "
            f"있습니다(현재: {previous_state['phase']!r})."
        )
    return IdeationConvState(**{**previous_state, "phase": "finalizing", "idea_locked": True})


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


# ============================================================================
# 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) —
# 단계 전환 조건을 위한 결정론적 파생 함수/상수. 요청 5번("LLM이 boolean이나 count를
# 임의로 직접 결정하도록 만들지 마세요")에 따라, 아래 값은 전부 state에 이미 저장된
# 구조화 데이터(problem_definition/solution_directions/idea_evolution/validation_result)
# 로부터 코드가 계산한다 — 별도의 "완료 여부" 필드를 LLM 응답에서 그대로 받아 저장하지
# 않는다.
# ============================================================================

MIN_SOLUTION_DIRECTIONS = 3
# 용준/Claude(2026-07-28, 요청: 기본 discovery 회의를 1라운드로 단축) — 기존 2회에서 1회로
# 낮춘다. 사용자가 명시적으로 결합/방향추가를 요청하면(conflict_resolution 노드의
# _request_another_conflict_round, ideation_conv_problem.py) conflict_round_count를 1
# 감소시켜 라운드를 한 번 더 여는 기존 메커니즘을 그대로 재사용하므로("사용자가 요청한
# 경우에만 1회 추가"), 이 상수 하나만 낮춰도 "기본 1라운드 + 필요 시 사용자 요청으로 +1"
# 요구사항이 그대로 성립한다 — 그 메커니즘 자체는 손대지 않았다.
MAX_CONFLICT_ROUNDS = 1
MAX_PROBLEM_REGENERATIONS = 2
# 용준/Claude(2026-07-28, 요청: 카드 선택 단계 제거로 사라진 자연 정지점 대체) —
# provisional_from_merge가 카드 선택 없이 바로 idea_validation으로 이어지면서, "검증
# 실패 -> idea_conflict_and_merge 재실행 -> 조건 재충족 -> 검증 -> 실패 -> ..." 순환이
# 사용자 개입 없이 한 번의 그래프 호출 안에서 무한 반복될 수 있게 됐다(이전에는
# candidate_feasibility가 재실행마다 항상 멈춰줬다). validation_revise_count가 이 상수에
# 도달하면 make_technical_validation_node는 더 이상 자동으로 idea_conflict_and_merge로
# 되돌리지 않고 awaiting_concept_confirmation에서 사용자가 직접 재검토/확정을 선택하게
# 한다.
MAX_VALIDATION_REVISE_ROUNDS = 1


def _non_blank(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def problem_defined(state: IdeationConvState) -> bool:
    """problem_definition.problem이 채워졌는지 — LLM이 별도 boolean을 반환하지 않고,
    이 필드 자체가 채워졌는지로만 판단한다."""
    definition = state.get("problem_definition")
    return isinstance(definition, dict) and _non_blank(definition.get("problem"))


def target_user_defined(state: IdeationConvState) -> bool:
    definition = state.get("problem_definition")
    return isinstance(definition, dict) and _non_blank(definition.get("target_user"))


def existing_limitations_defined(state: IdeationConvState) -> bool:
    definition = state.get("problem_definition")
    return isinstance(definition, dict) and _non_blank(definition.get("existing_limitations"))


def solution_direction_count(state: IdeationConvState) -> int:
    """status="active"인 해결 방향 수만 센다 — 폐기(dropped)되거나 다른 방향에 결합되어
    사라진(merged) 방향은 "서로 다른 해결 방향"으로 더 이상 유효하지 않다."""
    directions = state.get("solution_directions") or []
    return sum(1 for d in directions if isinstance(d, dict) and d.get("status") == "active")


def critique_count(state: IdeationConvState) -> int:
    """idea_evolution에 실제로 기록된 action_type="critique" 건수 — LLM이 "반론했다"고
    주장하는 것이 아니라, idea_conflict_and_merge 노드가 반론으로 분류해 실제로 append한
    기록만 센다."""
    evolution = state.get("idea_evolution") or []
    return sum(1 for r in evolution if isinstance(r, dict) and r.get("action_type") == "critique")


def merge_or_revision_count(state: IdeationConvState) -> int:
    evolution = state.get("idea_evolution") or []
    return sum(1 for r in evolution if isinstance(r, dict) and r.get("action_type") in ("merge", "revision"))


def planning_validation_completed(state: IdeationConvState) -> bool:
    result = state.get("validation_result")
    return isinstance(result, dict) and isinstance(result.get("planning"), dict) and bool(result.get("planning"))


def technical_validation_completed(state: IdeationConvState) -> bool:
    result = state.get("validation_result")
    return isinstance(result, dict) and isinstance(result.get("technical"), dict) and bool(result.get("technical"))


def meets_conflict_and_merge_min_conditions(state: IdeationConvState) -> bool:
    """요청 6번의 idea_conflict_and_merge 최소 조건: 해결 방향 3개 이상 / 반론 1회 이상 /
    수정·결합 1회 이상. 세 조건 모두 만족해야 candidate_generation(후보 압축)으로 자동
    진행할 수 있다 — 라운드 상한(MAX_CONFLICT_ROUNDS)에 도달했는데 이 조건을 못 채우면
    자동 진행 대신 사용자에게 결합/방향추가/방향폐기/검증진행 중 하나를 요청한다
    (awaiting_conflict_resolution)."""
    directions = state.get("solution_directions") or []
    directions_by_id = {
        direction.get("direction_id"): direction for direction in directions if isinstance(direction, dict)
    }
    materialized_records = [
        record
        for record in (state.get("idea_evolution") or [])
        if isinstance(record, dict)
        and record.get("action_type") in ("merge", "revision")
        and record.get("changed_by") == "planning_expert"
        and record.get("result_direction_id")
    ]
    materialized_change = False
    if materialized_records:
        latest = materialized_records[-1]
        result = directions_by_id.get(latest["result_direction_id"])
        parents = [directions_by_id.get(direction_id) for direction_id in latest.get("target_direction_ids") or []]
        materialized_change = bool(
            result
            and result.get("status") == "active"
            and parents
            and all(parent and parent.get("status") in ("merged", "superseded", "revised") for parent in parents)
        )

    messages = state.get("messages") or []
    speaker_types = [(message.get("speaker_id"), message.get("message_type")) for message in messages]
    has_ordered_messages = bool(
        len(speaker_types) >= 3
        and speaker_types[-1] == ("ideation_facilitator", "summary")
        and speaker_types[-2] == ("planning_expert", "opinion")
        and any(
            speaker == "dev_expert" and message_type == "disagreement"
            for speaker, message_type in speaker_types[:-2]
        )
    )
    return (
        solution_direction_count(state) >= MIN_SOLUTION_DIRECTIONS
        and critique_count(state) >= 1
        and merge_or_revision_count(state) >= 1
        and materialized_change
        and has_ordered_messages
    )


def ready_for_concept_confirmation(state: IdeationConvState) -> bool:
    """concept_confirmation을 사용자에게 제안하기 전에 반드시 충족돼야 하는 조건 —
    기획/개발 검증이 모두 완료됐는지(요청: "기획 관점의 검증이 완료됨" / "개발 관점의
    검증이 완료됨")."""
    return planning_validation_completed(state) and technical_validation_completed(state)


# 용준/Claude(2026-07-27, 요청 1번: "refinement 모드에서도 idea_locked 이전에는 신청서
# 문구나 사업계획서 작성으로 바로 넘어가지 않도록 하는 공통 가드는 적용해도 됩니다" /
# 요청 4번 "concept_confirmation 이전 단계에서는 신청서 작성법·사업계획서 목차 등을 다루지
# 못하게") — LLM이 만든 발화(spoken_text/content)에 확정 이전 단계에서 금지된 표현이
# 섞였는지 키워드로 판정한다. LLM이 "이건 신청서 작성이 아니다"라고 스스로 판단하게
# 맡기지 않고(요청 5번과 동일한 원칙 — 판단을 LLM에 맡기지 않는다), 코드가 결정적으로
# 검사한다. 오탐이 있더라도(예: "제출 양식"이라는 단어가 인용문 안에 있는 경우) "확정
# 전에는 절대 문서 작성으로 새지 않는다"는 안전 쪽으로 보수적으로 판단한다.
PRE_LOCK_BANNED_PHRASES: tuple[str, ...] = (
    "신청서 작성",
    "신청서 문구",
    "신청서 초안",
    "사업계획서 목차",
    "사업계획서 작성",
    "제안서 표현",
    "제출 문서 형식",
    "제출 양식",
    "서비스 소개문",
    "최종 서비스명",
    "기술 스택을 확정",
    "기술스택을 확정",
)

# 용준/Claude(2026-07-27, 후속 요청 3번: "idea_locked=False라는 이유만으로 신청서/사업계획서
# 키워드가 포함된 모든 발화를 교체하지 마세요 — 현재 phase와 발화 의도를 함께 확인") —
# 위 PRE_LOCK_BANNED_PHRASES는 문서 관련 "명사구"만 보므로 "사업계획서 작성 관점에서
# 설득력이 있는가?"처럼 실제로는 검토·질문인 발화도 우연히 걸릴 수 있다("사업계획서 작성"이
# 부분 문자열로 들어있기 때문). 이 마커가 하나라도 있으면 "실제 작성 행위를 하는 중"이
# 아니라 "검토·질문 중"이라는 뜻이므로 차단하지 않는다 — 질문 억양(?/글까요/인가요)이나
# 명시적 검토·평가 표현이 이에 해당한다.
PRE_LOCK_REVIEW_INTENT_MARKERS: tuple[str, ...] = (
    "?",
    "설득력이 있는가",
    "설득력 있는가",
    "설득력이 있을까요",
    "타당한가",
    "타당할까요",
    "적합한가",
    "적합할까요",
    "충분한가",
    "충분할까요",
    "일까요",
    "인가요",
    "필요할까요",
    "괜찮을까요",
    "관점에서 검토",
    "관점에서 보면",
    "검토해 보면",
    "평가해 보면",
    "고려해야 할까요",
)

# 문서 명사구 없이도 그 자체로 "지금 실제로 작성/확정하고 있다"는 것이 명확한 행위
# 표현 — idea_validation처럼 문서 종류를 언급하는 것 자체가 자연스러운 검증 단계(phase)
# 에서는, 이런 명확한 행위 마커가 있을 때만 차단한다(단순 언급만으로는 차단하지 않는다).
PRE_LOCK_DRAFTING_ACTION_MARKERS: tuple[str, ...] = (
    "작성하겠습니다",
    "작성해 드리겠습니다",
    "작성해드리겠습니다",
    "작성할게요",
    "작성해 볼게요",
    "초안을 만들",
    "초안을 준비",
    "목차를 정리하겠습니다",
    "목차부터 정리",
    "제출하겠습니다",
    "완성하겠습니다",
    "확정하겠습니다",
    "작성을 시작",
    "작성을 진행",
)

# 검증 단계(요청: "사업계획서에서 설득력이 있는가?" 같은 검토 질문이 자연스러운 단계)에서는
# 문서 종류를 언급하는 것 자체를 막지 않는다 — 명확한 작성 행위 마커가 있을 때만 차단한다.
_VALIDATION_STAGE_PHASES: frozenset[str] = frozenset(
    {"idea_validation", "awaiting_concept_confirmation", "concept_confirmation"}
)


def contains_pre_lock_banned_content(text: str | None, *, phase: str | None = None) -> bool:
    """확정 전(idea_locked=False) 발화에 신청서/사업계획서 등 문서 작성으로 새는 표현이
    섞였는지 판정한다. 요청 3번 — 단순 키워드 매치가 아니라 현재 phase와 발화 의도를 함께
    본다:
    1. 은행 문서 명사구가 없으면 애초에 검사 대상이 아니다.
    2. 질문/검토 억양(PRE_LOCK_REVIEW_INTENT_MARKERS)이 있으면 "검토 질문"으로 보고
       차단하지 않는다(예: "사업계획서 작성 관점에서 설득력이 있는가?").
    3. idea_validation/concept_confirmation처럼 문서 종류를 언급하는 것 자체가 자연스러운
       검증 단계에서는, 명확한 작성 행위 마커(PRE_LOCK_DRAFTING_ACTION_MARKERS)가 있을
       때만 차단한다 — 단순 언급만으로는 차단하지 않는다.
    4. 그 외(일반 discussion/expert_discussion 등)에서는 기존처럼 보수적으로 차단한다
       ("확정 전에는 절대 문서 작성으로 새지 않는다")."""
    if not text:
        return False
    if not any(phrase in text for phrase in PRE_LOCK_BANNED_PHRASES):
        return False
    if any(marker in text for marker in PRE_LOCK_REVIEW_INTENT_MARKERS):
        return False
    if phase in _VALIDATION_STAGE_PHASES:
        return any(marker in text for marker in PRE_LOCK_DRAFTING_ACTION_MARKERS)
    return True
