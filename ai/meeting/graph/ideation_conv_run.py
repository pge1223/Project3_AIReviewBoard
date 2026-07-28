# 작성자: 용준/Claude(2026-07-20)
# 목적: 대화형 아이디어 발전 회의(ideation-conversation) 실행 엔트리포인트. ideation_run.py
#       (배치형)와 같은 역할(그래프 조립 + State 초기화/갱신 + 실행)을 대화형 그래프에
#       대해 수행하지만, "한 번의 함수 호출 = HTTP 요청 한 번"이 되도록 훨씬 잘게 나뉜다.
#       start_ideation_conversation()은 세션 시작(기획 전문가 첫 질문 하나), reply_to_*는
#       사용자 답변 반영 + 다음 정지 지점까지 실행, finalize_ideation_conversation()은
#       오직 사용자가 확정 버튼을 눌렀을 때만 호출된다(요청 9~10항).
# import: 표준 라이브러리 typing/uuid/datetime, 같은 패키지의 ideation_conv_build/state/llm.

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from prompts import build_ideation_conv_form_draft_prompt

from .application_form_draft import apply_application_form_draft_patch, remaining_content_fields
from .ideation_conv_build import _ENTRY_NODES, assemble_ideation_conversation_graph
from .ideation_conv_discovery import MAX_CANDIDATE_REGENERATIONS, is_regenerate_request
from .ideation_conv_problem import _select_areas_by_action_payload
from .ideation_conv_nodes import (
    PHASE_TO_PENDING_PERSONA,
    REVISION_TRIGGER_STANCES,
    _route_next_expert_turn,
    _runtime_scope_for,
    conversation_context_for,
    generate_expert_delegation_facilitator_recommendation,
    generate_expert_delegation_proposal,
    generate_expert_delegation_review,
    is_expert_delegation_request,
    judge_answer_sufficiency,
    make_clarification_message,
    make_expert_delegation_facilitator_message,
    make_expert_delegation_message,
    make_expert_delegation_review_message,
    make_follow_up_message,
)
from .ideation_conv_state import (
    ConvMessage,
    IdeationCancelled,
    IdeationConvState,
    apply_user_answer,
    contains_pre_lock_banned_content,
    initial_conv_state,
    is_graph_entry_phase,
    request_finalize,
)
from .ideation_llm_log import log_llm_response
from .ideation_nodes import call_evidence_lookup
from .ideation_trace import sanitize_preview, trace_event
from .llm import LLMCall, parse_json_response

IdeationConvProgressCallback = Callable[[dict], None]

# 용준/Claude(2026-07-22, 요청: 사용자 답변을 session target evidence로 반영) — ai/meeting/graph는
# ai.rag를 직접 import하지 않는다(evidence_lookup/ground_claims와 동일한 경계). 실제 색인
# 구현(ai.rag.orchestration.ideation_target_indexing_service)은 backend가 만들어 주입한다.
# kind="user_answer"면 payload={"session_id","user_message_id","answer_text",
# "pending_question","pending_question_topic"}.
IndexTargetEvidenceFn = Callable[[str, dict], dict]

# 짧은 동의·감탄·UI 제어 문구는 target evidence로 색인하지 않는다(요청 17-2번) — 결정적
# 키워드/길이 기준으로만 판단하고 LLM을 다시 부르지 않는다.
_MIN_TARGET_EVIDENCE_CHARS = 12
_SKIP_AS_TARGET_EVIDENCE_PHRASES = (
    "네", "넵", "네네", "좋아요", "좋습니다", "감사합니다", "고맙습니다", "알겠습니다", "확인했습니다",
    "잠시만", "잠깐만",
)


def _should_index_user_message_as_target_evidence(*, message_type: str, phase: str, content: str) -> bool:
    """용준/Claude(2026-07-22, 요청: 인덱싱 대상 메시지 제한) — 진행자의 질문에 대한 구체적인
    답변이나, 기능/사용자/데이터/일정/제약/구현 범위를 추가하는 사용자 개입만 target evidence로
    색인 대상으로 판단한다. message_type/phase/길이/키워드만으로 결정적으로 판단하고, "실제
    아이디어 정보가 포함됐는지"를 LLM으로 판정하지 않는다(안전하게 판단할 방법이 없으면 무조건
    사실로 승격하지 않는다는 원칙 그대로 — 판단이 애매하면 색인하지 않는 쪽으로 보수적으로
    판단한다)."""
    if message_type not in ("answer", "interjection"):
        return False
    if phase == "awaiting_candidate_selection":
        # 후보 선택 응답은 index_selected_candidate_as_target()이 candidate 자체를 색인하므로
        # 사용자의 선택 발화(번호/제목) 자체를 또 색인하지 않는다.
        return False
    normalized = (content or "").strip()
    if len(normalized) < _MIN_TARGET_EVIDENCE_CHARS:
        return False
    if is_regenerate_request(normalized) or is_expert_delegation_request(normalized):
        return False
    if normalized in _SKIP_AS_TARGET_EVIDENCE_PHRASES:
        return False
    return True


def _index_user_answer(
    *,
    state: IdeationConvState,
    answer_message: ConvMessage,
    raw_answer_text: str,
    index_target_evidence: IndexTargetEvidenceFn | None,
) -> None:
    """사용자 답변을 session target evidence로 색인한다(요청 4번). 색인이 끝난 뒤에야 이
    함수가 반환하므로(동기 호출), 호출부가 그 다음에 이어서 그래프를 실행하면 다음 전문가
    검색은 항상 이번 답변이 색인된 뒤의 상태를 본다(요청 9번 순서 보장) — 별도의 백그라운드
    작업이나 큐를 쓰지 않는다. 실패해도 예외를 전파하지 않는다 — 회의 자체는 계속돼야 하고
    (요청 17-4번), 이번 턴은 그냥 이 답변의 target 근거 없이 진행된다."""
    if index_target_evidence is None:
        return
    if not _should_index_user_message_as_target_evidence(
        message_type=answer_message["message_type"], phase=state["phase"], content=raw_answer_text
    ):
        return
    try:
        index_target_evidence(
            "user_answer",
            {
                "session_id": state["session_id"],
                "user_message_id": answer_message["message_id"],
                "answer_text": raw_answer_text,
                "pending_question": state.get("pending_question"),
                "pending_question_topic": state.get("pending_question_topic"),
            },
        )
    except Exception as exc:  # noqa: BLE001 — 색인 실패가 회의를 막으면 안 된다.
        trace_event(
            "IDEATION_TARGET_EVIDENCE_UPSERT_FAILED",
            level=30,
            session_id=state.get("session_id"),
            source_type="user_session_answer",
            user_message_id=answer_message["message_id"],
            error=sanitize_preview(str(exc), limit=100),
        )

# 용준/Claude(2026-07-27, 후속 요청 2번: "action_code가 명시적으로 전달된 경우 자연어
# 키워드 판정으로 조용히 폴백하지 말고 명시적 validation error를 내라") — action_code별로
# 허용되는 previous_state["phase"](=action_code가 의미를 갖는 awaiting_* 정지 지점)를
# 고정한다. 여기 없는 action_code는 전부 미지원으로 간주한다.
_ACTION_CODE_ALLOWED_PHASES: dict[str, frozenset[str]] = {
    "select_problem_focus": frozenset({"awaiting_problem_focus_selection"}),
    "combine_problem_focus": frozenset({"awaiting_problem_focus_selection"}),
    "add_solution_direction": frozenset({"awaiting_conflict_resolution"}),
    "merge_directions": frozenset({"awaiting_conflict_resolution"}),
    "drop_direction": frozenset({"awaiting_conflict_resolution"}),
    "proceed_to_validation": frozenset({"awaiting_conflict_resolution"}),
    "revise_candidate": frozenset({"awaiting_concept_confirmation"}),
    "confirm_concept": frozenset({"awaiting_concept_confirmation"}),
    "choose_another_candidate": frozenset({"awaiting_concept_confirmation"}),
    "return_to_problem_definition": frozenset({"awaiting_conflict_resolution", "awaiting_concept_confirmation"}),
}


def _validate_action_code(state: IdeationConvState, action_code: str, action_payload: dict) -> None:
    """action_code가 명시적으로 전달됐을 때만 호출된다(호출부 참고). 여기서 걸러내지 못한
    문제만 각 노드(ideation_conv_problem.py)의 자연어 폴백으로 넘어간다 — action_code 자체가
    지원 대상이 아니거나, 현재 phase에서 허용되지 않거나, 필수 payload가 없거나, 대상 id가
    존재하지 않거나 이미 폐기/결합된 경우는 전부 여기서 명시적 ValueError로 막는다(요청:
    "프론트 버그나 잘못된 요청이 숨겨지지 않아야 한다"). ValueError는 API 레이어
    (ideation_conversation_preview.py)가 HTTP 400으로 그대로 변환한다."""
    allowed_phases = _ACTION_CODE_ALLOWED_PHASES.get(action_code)
    if allowed_phases is None:
        raise ValueError(f"지원하지 않는 action_code입니다: {action_code!r}")
    phase = state["phase"]
    if phase not in allowed_phases:
        raise ValueError(
            f"action_code {action_code!r}는 현재 phase({phase!r})에서 허용되지 않습니다. "
            f"허용된 phase: {sorted(allowed_phases)}"
        )

    if action_code in ("select_problem_focus", "combine_problem_focus"):
        areas = state.get("problem_areas") or []
        selected = _select_areas_by_action_payload(areas, action_payload)
        if not selected:
            raise ValueError(
                f"action_code {action_code!r}의 action_payload가 유효하지 않습니다. "
                "problem_areas 안에 존재하는 indices(1-based) 또는 area_ids가 필요합니다."
            )
        if action_code == "combine_problem_focus" and len(selected) < 2:
            raise ValueError("combine_problem_focus는 서로 다른 area 2개를 지정해야 합니다.")

    elif action_code in ("merge_directions", "drop_direction"):
        direction_ids = action_payload.get("direction_ids")
        if not isinstance(direction_ids, list) or not direction_ids:
            raise ValueError(f"action_code {action_code!r}는 action_payload.direction_ids(list)가 필요합니다.")
        if action_code == "merge_directions" and len(direction_ids) < 2:
            raise ValueError("merge_directions는 서로 다른 direction_id 2개 이상이 필요합니다.")
        directions_by_id = {d.get("direction_id"): d for d in state.get("solution_directions") or []}
        for direction_id in direction_ids:
            target = directions_by_id.get(direction_id)
            if target is None:
                raise ValueError(f"존재하지 않는 direction_id입니다: {direction_id!r}")
            if target.get("status") != "active":
                raise ValueError(
                    f"이미 폐기되거나 결합된 direction은 지정할 수 없습니다: "
                    f"{direction_id!r}(status={target.get('status')!r})"
                )

    elif action_code == "confirm_concept":
        if not state.get("provisional_idea"):
            raise ValueError("확정할 provisional_idea가 없습니다.")


def validate_ideation_action_code(state: IdeationConvState, action_code: str, action_payload: dict) -> None:
    """_validate_action_code의 공개 래퍼. reply_ideation_conversation은 이 검증을 그래프
    실행 도중(action_code가 주어졌을 때) 자동으로 수행하지만, 스트리밍 API
    (ideation_conversation_preview.py::reply_conversation_stream)는 StreamingResponse를
    시작하기 전에 미리 검증해 HTTP 400을 응답해야 한다(스트림이 이미 200으로 시작된 뒤
    NDJSON error 이벤트로만 실패를 알리면 클라이언트가 이를 정상 200 응답으로 오인할 수
    있다) — 그 목적으로 밑줄 없는 이름으로 별도 노출한다. 검증 로직 자체는 전혀 새로
    만들지 않고 _validate_action_code를 그대로 재사용한다."""
    _validate_action_code(state, action_code, action_payload)


# API가 사용자 입력을 받아도 되는(=그래프를 다시 부르지 않고 멈춰 있어야 하는) phase.
# 용준/Claude(2026-07-21): discovery(아이디어 발굴) 모드의 후보 선택 대기 phase를 추가한다
# — PHASE_TO_PENDING_PERSONA에는 없는 phase이므로 answer_sufficiency 게이트(아래 참고)는
# 자동으로 건너뛰고 apply_user_answer -> candidate_selection 노드로 그대로 이어진다(요청:
# 후보 선택 전에는 refinement 질문이 실행되지 않고, 선택은 재질문 판정 대상도 아니다).
REPLYABLE_PHASES = {
    "awaiting_planning_answer",
    "awaiting_developer_answer",
    "awaiting_user_decision",
    "discussion_complete",
    "awaiting_candidate_selection",
    # 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — discovery
    # 전용 신규 정지 지점 3개.
    "awaiting_problem_focus_selection",
    "awaiting_conflict_resolution",
    "awaiting_concept_confirmation",
}

# 용준/Claude(2026-07-27, 요청 1·4번: "concept_confirmation 이전 단계에서는 신청서 작성법·
# 사업계획서 목차 등을 다루지 못하게" / "refinement 모드에서도 idea_locked 이전에는 신청서
# 문구나 사업계획서 작성으로 바로 넘어가지 않도록") — LLM이 만든 발화에 확정 이전 금지
# 표현(ideation_conv_state.PRE_LOCK_BANNED_PHRASES)이 섞이면, 그 발화를 그대로 사용자에게
# 보여주지 않고 이 안내문으로 교체한다. 사용자 요청 원문 예시를 그대로 쓴다.
_PRE_LOCK_REDIRECT_MESSAGE = (
    "아직 아이디어를 확정하는 단계가 아닙니다. 문서를 작성하기 전에 해결하려는 문제와 "
    "가능한 해결 방향을 더 탐색하겠습니다."
)


def _guard_pre_lock_messages(state: IdeationConvState, baseline_message_count: int) -> IdeationConvState:
    """idea_locked=False인 동안 baseline_message_count 이후에 새로 추가된, 사용자가 아닌
    발화 중 PRE_LOCK_BANNED_PHRASES가 섞인 것을 찾아 교체한다. discovery 모드의 새 단계
    (problem_discovery~concept_confirmation)는 프롬프트 규칙 자체가 신청서/사업계획서
    언급을 요구하지 않지만, refinement 모드(초기 아이디어를 이미 입력해 곧바로
    expert_discussion부터 시작하는 세션)는 idea_locked=False로 finalize 전까지 계속 대화가
    이어지므로 이 후처리 가드가 실질적인 방어선이다(요청 1번) — LLM이 "이건 신청서
    작성이 아니다"라고 스스로 판단하게 맡기지 않고, 이미 나온 발화를 코드가 검사한다."""
    if state.get("idea_locked"):
        return state
    messages = state.get("messages") or []
    if len(messages) <= baseline_message_count:
        return state
    changed = False
    new_messages = list(messages)
    for i in range(baseline_message_count, len(new_messages)):
        message = new_messages[i]
        if message.get("speaker_id") == "user":
            continue
        if not contains_pre_lock_banned_content(message.get("content"), phase=state.get("phase")):
            continue
        changed = True
        redirected = dict(message)
        redirected["content"] = _PRE_LOCK_REDIRECT_MESSAGE
        structured = redirected.get("structured")
        if isinstance(structured, dict):
            redirected["structured"] = {**structured, "spoken_text": _PRE_LOCK_REDIRECT_MESSAGE, "pre_lock_guard_triggered": True}
        new_messages[i] = redirected
        trace_event(
            "IDEATION_PRE_LOCK_CONTENT_GUARD_TRIGGERED",
            level=30,
            session_id=state.get("session_id"),
            speaker_id=message.get("speaker_id"),
            phase=state.get("phase"),
        )
    if not changed:
        return state
    return IdeationConvState(**{**state, "messages": new_messages})

# 같은 쟁점(pending_question)으로 재질문할 수 있는 최대 횟수. 요청 3번(재질문 조건)의 예시
# "재질문이 2회 이상 반복되면... 합리적인 가정을 제시하고 다음 단계로 진행한다"를 그대로
# 코드 상수로 옮긴 값 — LLM 판정과 무관하게 이 값에 도달하면 강제로 다음 단계로 넘어간다
# (요청 5번 "라운드가 끝없이 반복되지 않도록" 무한 재질문 방지, max_rounds와는 별개의 축).
_MAX_ANSWER_RETRY = 2


def _new_user_message(content: str, round_number: int, message_type: str = "answer") -> ConvMessage:
    """사용자 발언 메시지를 조립한다. speaker_id="user"는 persona_cards.json에 없는
    값이라 ideation_conv_nodes.py의 페르소나 기반 헬퍼를 재사용할 수 없어 여기서 직접
    만든다 — message_id/created_at은 다른 메시지와 동일하게 항상 서버가 만든다.

    message_type(용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환)은 기본값 "answer"
    (기존 동작 그대로)에 더해 "interjection"을 받을 수 있다 — 호출부(아래
    _reply_message_type_for)가 "진행자가 실제로 물은 질문에 답한 것"과 "라운드 사이에
    자발적으로 끼어든 것"을 구분해 넘긴다(요청 6번)."""
    return ConvMessage(
        message_id=f"MSG-{uuid.uuid4().hex[:10]}",
        speaker_id="user",
        speaker_name="사용자",
        role="사용자",
        round=round_number,
        message_type=message_type,  # type: ignore[typeddict-item]
        content=content,
        referenced_message_ids=[],
        evidence=[],
        created_at=datetime.now(timezone.utc).isoformat(),
        structured=None,
    )


def _reply_message_type_for(previous_state: IdeationConvState) -> str:
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환): "awaiting_user_decision"
    (라운드테이블이 한 라운드를 마치고 멈춘 지점)에서만 "answer"/"interjection"을 구분한다.
    진행자가 needs_user_decision=True로 실제 질문을 던졌으면(pending_question이 설정됨)
    "answer", 아니면(사용자가 답할 의무 없이 자발적으로 끼어든 것) "interjection"이다.
    다른 phase(awaiting_planning_answer/awaiting_developer_answer/awaiting_candidate_selection
    — 인터뷰·후보 선택 흐름)는 기존 그대로 항상 "answer"다(요청 범위 밖, 동작 변경 없음)."""
    if previous_state["phase"] in {"awaiting_user_decision", "discussion_complete"}:
        return "answer" if previous_state.get("pending_question") else "interjection"
    return "answer"


def _new_facilitator_message(content: str, round_number: int) -> ConvMessage:
    """후보 재생성 상한 안내와 같이 그래프 노드를 거치지 않고 바로
    반환해야 하는 진행자 메시지를 만든다."""
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


def _progress(snapshot: IdeationConvState) -> dict:
    return {
        "phase": snapshot.get("phase"),
        "round": snapshot.get("round"),
        "messages_done": len(snapshot.get("messages") or []),
        "llm_calls_used": snapshot.get("llm_calls_used"),
    }


# 용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소): on_snapshot 콜백 타입은 더 이상 쓰지
# 않지만(아래 _drive_graph 참고 — 매 스냅샷마다 부르는 대신 취소 시점의 마지막 완료 상태만
# 예외에 실어 전달한다), start_ideation_conversation 등 공개 함수 시그니처의 하위 호환을
# 위해 이름은 남겨 둔다.
IdeationConvSnapshotCallback = Callable[[IdeationConvState], None]

# 재인/Claude(2026-07-23, 아바타 페이싱 연동): stop_after_expert_turn=True일 때, 이 화자의
# 발언이 하나 추가되는 즉시 _drive_graph를 멈춘다. _route_next_expert_turn 등 "누가 다음에
# 말할지" 판단 로직 자체는 전혀 건드리지 않았다(아래 continue_ideation_expert_turn이 그
# 함수를 그대로 재사용한다).
_SINGLE_TURN_STOP_SPEAKERS = frozenset({"planning_expert", "dev_expert"})

# 재인/Claude(2026-07-23, 아바타 페이싱 연동): _route_next_expert_turn이 반환할 수 있는 값
# 중, 강제 진입(forced_next_speaker)으로 이어가도 되는 것들 — "failed"는 제외(그래프를 다시
# 부를 이유가 없다). continue_ideation_expert_turn에서만 쓴다.
_FORCED_ENTRY_TARGETS = frozenset({"planning_expert", "dev_expert", "facilitator"})


def _drive_graph(
    graph: Any,
    state: IdeationConvState,
    on_progress: IdeationConvProgressCallback | None = None,
    on_snapshot: IdeationConvSnapshotCallback | None = None,
    stop_after_expert_turn: bool = False,
) -> IdeationConvState:
    """용준/Claude(2026-07-22, 요청: "잠시만" 실제 취소): IdeationCancelled가 이 for 루프
    도중 올라오면, 그 시점까지 완료된 마지막 스냅샷을 예외 객체(exc.partial_state)에 실어
    그대로 상위(worker)까지 전파한다 — "쟁점 A에 대한 발언 2건은 이미 완료, 3번째가
    스트리밍 중 취소"된 경우 앞의 2건은 exc.partial_state에 담기고, 취소된 3번째만
    빠진다(완료된 전문가 주장은 유지, 미완성만 취소 — 요청 14번). 일반 오류(취소가 아닌
    예외)는 그대로 전파하고 partial_state를 붙이지 않는다 — 호출부가 세션 store에 아무것도
    쓰지 않아야 이전 canonical state가 손상되지 않는다(회귀 테스트: 스트리밍 중 일반 LLM
    오류가 나도 세션 state가 손상되지 않아야 한다).

    재인/Claude(2026-07-23, 아바타 페이싱 연동): stop_after_expert_turn=True면
    _SINGLE_TURN_STOP_SPEAKERS 화자의 발언이 새로 추가되는 스냅샷에서 멈춘다(기본값
    False면 기존과 완전히 동일하게 끝까지 돈다 — 기존 호출부는 전혀 영향받지 않는다).

    재인/Claude(2026-07-23, 실측: "선택 직후 진행자 안건 소개랑 기획위원 첫 발언이 한
    응답에 같이 옴"): 위 조건 하나만으로는 부족한 경우가 있다 — ideation_conv_discovery.py
    ::_resolve_selection이 후보 확정 시 [선택 확정, 안건 소개] 메시지 2개를 한 노드
    실행에서 한꺼번에 반환하는데, 마지막 메시지(안건 소개)의 화자가 ideation_facilitator라
    _SINGLE_TURN_STOP_SPEAKERS에 안 걸려서 멈추지 않고 그대로 planning_expert_discussion까지
    같은 호출 안에서 이어져버린다. 그래서 "한 스냅샷에서 새 메시지가 2개 이상 추가됐고
    전부 ideation_facilitator"인 경우도 정지 지점으로 취급한다 — 코드 전체에서
    _resolve_selection이 메시지를 2개 이상 묶어 반환하는 유일한 곳이라(grep으로 확인),
    이 조건이 다른 정상 흐름을 잘못 멈추게 할 위험은 없다.

    재인/Claude(2026-07-24, 실측: "진행자 라운드 정리 발언 + 다음 라운드 기획위원 첫
    발언이 완전히 동시에 나옴"): 진행자가 "정리" 발언을 혼자 하나만 만드는 경우(위 2개
    묶음 케이스와 다름)는 이 시점에 바로 멈추면 안 된다 — discussion_facilitator 바로
    다음 노드인 canvas_update(화면에 안 보이는 캔버스 갱신)가 아직 안 끝난 상태라, 여기서
    끊으면 캔버스 갱신이 통째로 스킵돼버린다. 그렇다고 원래처럼 끝까지 흘려보내면
    continue_round인 경우 다음 라운드 기획위원 발언까지 같은 호출에 묶여버려 원래 문제가
    재현된다. 그래서 "진행자 단독 발언을 봤다"는 사실만 기억해뒀다가, 그 다음 스냅샷
    (canvas_update의 결과 — 새 메시지 유무와 무관하게)에서 멈춘다. continue_round가 아니면
    canvas_update 다음에 그래프가 자연스럽게 끝나므로(END) 이 예약이 발동하기 전에 루프가
    이미 끝나 있는 경우도 있는데, 그때도 결과는 동일하게 올바르다(캔버스까지 반영된 최종
    상태). continue_round인 경우, 다음 라운드 기획위원 발언은 이 호출에 안 끼고, 아바타가
    진행자 발언을 다 재생한 뒤 별도의 continue_ideation_expert_turn 호출로 자연스럽게
    이어받는다(그 함수의 "직전 발언자가 planning/dev가 아니면 라운드 첫 턴이므로
    planning_expert부터"라는 기존 부트스트랩 분기가 그대로 처리해준다 — 아래
    continue_ideation_expert_turn 참고)."""
    final_state: IdeationConvState = state
    previous_message_count = len(state.get("messages") or [])
    stop_after_next_snapshot = False
    try:
        for snapshot in graph.stream(state, stream_mode="values"):
            final_state = snapshot
            snapshot_messages = snapshot.get("messages") or []
            trace_event(
                "IDEATION_GRAPH_SNAPSHOT",
                phase=snapshot.get("phase"),
                message_count=len(snapshot_messages),
                solution_direction_count=len(snapshot.get("solution_directions") or []),
                critique_count=sum(
                    1
                    for item in (snapshot.get("idea_evolution") or [])
                    if item.get("action_type") == "critique"
                ),
                merge_or_revision_count=sum(
                    1
                    for item in (snapshot.get("idea_evolution") or [])
                    if item.get("action_type") in {"merge", "revision"}
                ),
                conflict_round_count=snapshot.get("conflict_round_count", 0),
                stop_after_expert_turn=stop_after_expert_turn,
            )
            if on_progress is not None:
                on_progress(_progress(snapshot))
            if stop_after_expert_turn:
                if stop_after_next_snapshot:
                    break
                messages = snapshot.get("messages") or []
                new_count = len(messages) - previous_message_count
                if new_count > 0:
                    new_messages = messages[len(messages) - new_count :]
                    previous_message_count = len(messages)
                    last_speaker = new_messages[-1].get("speaker_id")
                    # 용준/Claude(2026-07-27, 실측: "discovery 모드에서 기획 의원 발언 1건
                    # 후 회의가 영원히 멈춤") — _SINGLE_TURN_STOP_SPEAKERS는 원래 기존
                    # 라운드테이블(expert_discussion, planning_expert_discussion/
                    # dev_expert_discussion 노드)에서 아바타가 한 발언씩만 재생하도록 만든
                    # 정지 지점이다. 그런데 discovery 모드의 idea_divergence 노드도 같은
                    # persona_id("planning_expert")로 메시지를 만들어서, phase 구분 없이
                    # speaker_id만 보면 idea_divergence 직후에도 잘못 멈춰버린다(그
                    # 노드는 "정지 없이 바로 idea_conflict_and_merge로 이어진다"는 설계다 —
                    # ideation_conv_problem.py::make_idea_divergence_node 참고). discovery
                    # 모드 노드들은 phase를 "expert_discussion"으로 두지 않으므로(성공
                    # 경로에서 phase 키 자체를 반환하지 않아 이전 phase가 그대로 유지된다),
                    # phase가 "expert_discussion"일 때만 이 정지 조건을 적용해 legacy
                    # 라운드테이블에만 국한시킨다. continue_ideation_expert_turn도 정확히
                    # 같은 조건(phase == "expert_discussion")으로만 재개를 허용하므로 이
                    # 정지 지점과 재개 지점의 전제가 항상 일치한다.
                    if last_speaker in _SINGLE_TURN_STOP_SPEAKERS and snapshot.get("phase") == "expert_discussion":
                        break
                    # 용준/Claude(2026-07-28, 실측 후속: "idea_divergence까지는 통과했는데
                    # 그 다음 스냅샷에서 또 멈춤") — 아래 두 facilitator 전용 정지 조건도
                    # 위와 같은 이유(legacy expert_discussion 라운드테이블 전용 설계)로
                    # discovery 모드에서 오작동한다. problem_definition 노드가 만드는
                    # "문제 정의" 요약 메시지도 speaker_id="ideation_facilitator" 단독
                    # 1건이라, 이 조건이 그대로 걸리면 stop_after_next_snapshot이 True가
                    # 되어 바로 다음 스냅샷(idea_divergence 실행 결과)에서 idea_conflict_and_merge
                    # 로 넘어가기도 전에 루프가 끊긴다. 두 조건 모두 phase가
                    # "expert_discussion"일 때만 적용해 legacy 라운드테이블에 국한시킨다.
                    if snapshot.get("phase") == "expert_discussion":
                        if new_count > 1 and all(
                            m.get("speaker_id") == "ideation_facilitator" for m in new_messages
                        ):
                            break
                        if new_count == 1 and last_speaker == "ideation_facilitator":
                            stop_after_next_snapshot = True
    except IdeationCancelled as exc:
        exc.partial_state = final_state if final_state is not state else None
        raise
    return final_state


def start_ideation_conversation(
    *,
    session_id: str,
    notice_and_criteria: dict[str, Any],
    user_idea: dict[str, Any],
    llm_call: LLMCall,
    max_rounds: int = 3,
    evidence_lookup=None,
    ground_claims=None,
    index_target_evidence: IndexTargetEvidenceFn | None = None,
    evidence_planner=None,
    external_evidence_lookup=None,
    on_progress: IdeationConvProgressCallback | None = None,
    on_snapshot: IdeationConvSnapshotCallback | None = None,
    application_form_items: list[dict] | None = None,
) -> IdeationConvState:
    """세션을 시작해 기획 전문가의 첫 질문 하나만 만들고 멈춘다(요청 목표 흐름 1~3번).

    가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입): application_form_items는 순수
    추가 파라미터다(기본값 None) — 넘기지 않으면 기존 호출부와 완전히 동일하게 동작한다.

    용준/Claude(2026-07-27, RAG-007 연결): external_evidence_lookup도 순수 추가 파라미터다
    (기본값 None) — problem_discovery, candidate_planning/candidate_feasibility,
    idea_validation 노드에 전달된다."""
    graph = assemble_ideation_conversation_graph(
        llm_call,
        evidence_lookup=evidence_lookup,
        ground_claims=ground_claims,
        index_target_evidence=index_target_evidence,
        evidence_planner=evidence_planner,
        external_evidence_lookup=external_evidence_lookup,
    )
    state = initial_conv_state(
        session_id, notice_and_criteria, user_idea, max_rounds=max_rounds,
        application_form_items=application_form_items,
    )
    baseline_message_count = len(state["messages"])
    result_state = _drive_graph(graph, state, on_progress, on_snapshot)
    return _guard_pre_lock_messages(result_state, baseline_message_count)


# 용준/Claude(2026-07-28, 요청: "다시 시도"가 전체 회의를 처음부터 다시 실행하지 않게)
# — failed_node -> 그 노드의 진입 phase 역매핑. _ENTRY_NODES(ideation_conv_build.py)가
# "이 phase로 그래프를 시작하면 이 노드가 실행된다"는 정방향 매핑이므로, 그 역방향을 그대로
# 재사용한다(새 매핑 테이블을 따로 관리하지 않는다 — 두 테이블이 어긋날 위험을 없앤다).
_FAILED_NODE_TO_RETRY_PHASE: dict[str, str] = {node_name: phase for phase, node_name in _ENTRY_NODES.items()}


def retry_failed_ideation_conversation_node(
    *,
    previous_state: IdeationConvState,
    llm_call: LLMCall,
    evidence_lookup=None,
    ground_claims=None,
    index_target_evidence: IndexTargetEvidenceFn | None = None,
    evidence_planner=None,
    external_evidence_lookup=None,
    on_progress: IdeationConvProgressCallback | None = None,
    on_snapshot: IdeationConvSnapshotCallback | None = None,
) -> IdeationConvState:
    """phase="failed"인 세션을 failed_node부터 재개한다 — 새 세션을 만들거나 messages/
    problem_definition/idea_evolution 등 기존 state를 지우지 않는다(phase/failed_node만
    되돌린다). reply_ideation_conversation을 그대로 쓸 수 없는 이유: "failed"는
    REPLYABLE_PHASES에 없고(그래프 자체도 "failed"에서는 절대 시작하지 않는다,
    _route_entry 참고) 사용자 메시지 없이 노드를 그냥 재실행해야 하기 때문이다."""
    if previous_state.get("phase") != "failed":
        raise ValueError(f"실패 상태(phase='failed')의 세션만 재시도할 수 있습니다: phase={previous_state.get('phase')!r}")
    failed_node = previous_state.get("failed_node")
    retry_phase = _FAILED_NODE_TO_RETRY_PHASE.get(failed_node or "")
    if retry_phase is None:
        raise ValueError(f"이 노드는 failed_node부터 재시도를 지원하지 않습니다: {failed_node!r}")

    graph = assemble_ideation_conversation_graph(
        llm_call,
        evidence_lookup=evidence_lookup,
        ground_claims=ground_claims,
        index_target_evidence=index_target_evidence,
        evidence_planner=evidence_planner,
        external_evidence_lookup=external_evidence_lookup,
    )
    retry_state = IdeationConvState(**{**previous_state, "phase": retry_phase, "failed_node": None})
    baseline_message_count = len(retry_state.get("messages") or [])
    result_state = _drive_graph(graph, retry_state, on_progress, on_snapshot)
    return _guard_pre_lock_messages(result_state, baseline_message_count)


_DELEGATION_COUNTERPART = {"planning_expert": "dev_expert", "dev_expert": "planning_expert"}


def _delegate_to_expert(
    *,
    previous_state: IdeationConvState,
    persona_id: str,
    pending_question: str,
    llm_call: LLMCall,
    evidence_lookup,
    context: dict[str, Any],
) -> tuple[IdeationConvState, IdeationConvState | None, list[ConvMessage]]:
    """용준/Claude(2026-07-21, 요청: "모르겠다" UX 개선 + 2026-07-21 후속 요청: expert_delegation도
    위원 간 상호 검토로 확장): 사용자가 pending_question에 답하는 대신 전문가 판단에
    위임했을 때(answer_type="expert_delegation") 단일 위원 제안으로 끝내지 않고, 담당
    위원 제안 -> 반대 위원 검토(REVIEW_TRIGGER_STANCES에 속하면 담당 위원의 수정까지) ->
    진행자 최종 권고안까지 만든다. 같은 질문을 반복하지 않고 다음 단계로 진행해야 하므로
    (요청 사항), 이 처리는 answer_type="answer"와 같은 원칙을 공유한다 — pending_question_topic을
    resolved_topics에 추가하고 answer_retry_count를 리셋한다(reply_ideation_conversation이
    이어서 apply_user_answer로 phase를 넘긴다). 진행자 최종 권고안(facilitator_message)의
    출력 스키마에는 애초에 사용자 재질문 필드가 없어 구조적으로 같은 질문을 반복할 수 없다.

    반환값은 _apply_answer_sufficiency_gate와 같은 3-tuple 계약을 따르되, 세 번째 값이
    이제 리스트다(요청: 여러 위원 발언을 순서대로 끼워 넣어야 하므로) — 성공하면
    (다음에 apply_user_answer에 넘길 previous_state, None, [제안, 검토, (있으면) 수정,
    진행자 권고안] 메시지 목록), 어느 단계든 구조화 검증에 실패하면(재시도 후에도 무효)
    (previous_state, phase="failed"인 최종 state, []) — 다른 콘텐츠 생성 노드(질문/의견
    턴)의 구조화 검증 실패와 동일한 정책이다."""
    query = pending_question or _topic_query_fallback(previous_state)
    used = previous_state.get("llm_calls_used", 0)

    def _fail(node_name: str, attempts: int) -> tuple[IdeationConvState, IdeationConvState | None, list[ConvMessage]]:
        nonlocal used
        used += attempts
        failed_state = IdeationConvState(
            **{**previous_state, "phase": "failed", "failed_node": node_name, "llm_calls_used": used}
        )
        return previous_state, failed_state, []

    # 1) 담당 위원의 최초 제안.
    runtime_scope = _runtime_scope_for(previous_state)
    owner_retrieved = call_evidence_lookup(evidence_lookup, persona_id, query, runtime_scope=runtime_scope)
    proposal_raw, ok, attempts = generate_expert_delegation_proposal(
        llm_call,
        persona_id,
        pending_question,
        previous_state["notice_and_criteria"],
        previous_state["user_idea"],
        owner_retrieved,
        context,
    )
    if not ok or proposal_raw is None:
        return _fail(f"expert_delegation__{persona_id}", attempts)
    used += attempts

    known_ids = {m["message_id"] for m in previous_state["messages"]}
    proposal_message = make_expert_delegation_message(
        persona_id=persona_id,
        round_number=previous_state["round"],
        spoken_text=proposal_raw.get("spoken_text", ""),
        proposal=proposal_raw["proposal"],
        reason=proposal_raw["reason"],
        assumption=proposal_raw["assumption"],
        referenced_message_ids=proposal_raw.get("referenced_message_ids"),
        # 용준/Claude(2026-07-22, RAG 근거 유실 수정): proposal_raw.get("evidence")는 LLM이
        # 자발적으로 되돌려준 검증되지 않는 필드라 대부분 비어 있었다 — owner_retrieved(위에서
        # 실제 RAG 검색으로 얻어 프롬프트에 주입한 근거)를 그대로 저장해야 한다.
        evidence=owner_retrieved,
        known_message_ids=known_ids,
    )
    messages = [proposal_message]
    known_ids = known_ids | {proposal_message["message_id"]}

    # 2) 반대 위원의 검토(용준/Claude(2026-07-21), 요청: expert_delegation도 위원 간 상호
    #    검토로 확장) — 사용자가 아니라 동료 전문가로서 제안을 검토한다.
    counterpart_id = _DELEGATION_COUNTERPART[persona_id]
    counterpart_retrieved = call_evidence_lookup(evidence_lookup, counterpart_id, query, runtime_scope=runtime_scope)
    review_raw, ok, attempts = generate_expert_delegation_review(
        llm_call,
        counterpart_id,
        pending_question,
        previous_state["notice_and_criteria"],
        previous_state["user_idea"],
        counterpart_retrieved,
        context,
        proposal_raw,
    )
    if not ok or review_raw is None:
        return _fail(f"expert_delegation_review__{counterpart_id}", attempts)
    used += attempts

    review_message = make_expert_delegation_review_message(
        persona_id=counterpart_id,
        round_number=previous_state["round"],
        raw=review_raw,
        known_message_ids=known_ids,
        evidence=counterpart_retrieved,
    )
    messages.append(review_message)
    known_ids = known_ids | {review_message["message_id"]}

    # 3) 반대 위원의 stance가 REVISION_TRIGGER_STANCES(반박/조건부_동의/대안_제시)에 속할
    #    때만 담당 위원이 수정/유지 의견을 낸다(요청 6번과 동일한 비용 절감 원칙 — 새 분류
    #    LLM 호출 없이 이미 나온 stance 필드만으로 결정적으로 게이팅한다).
    revision_raw: dict | None = None
    if review_raw.get("stance") in REVISION_TRIGGER_STANCES:
        revision_retrieved = call_evidence_lookup(evidence_lookup, persona_id, query, runtime_scope=runtime_scope)
        revision_raw, ok, attempts = generate_expert_delegation_proposal(
            llm_call,
            persona_id,
            pending_question,
            previous_state["notice_and_criteria"],
            previous_state["user_idea"],
            revision_retrieved,
            context,
            stage="revision",
            counterpart_review=review_raw,
        )
        if not ok or revision_raw is None:
            return _fail(f"expert_delegation__{persona_id}", attempts)
        used += attempts

        revision_message = make_expert_delegation_message(
            persona_id=persona_id,
            round_number=previous_state["round"],
            spoken_text=revision_raw.get("spoken_text", ""),
            proposal=revision_raw["proposal"],
            reason=revision_raw["reason"],
            assumption=revision_raw["assumption"],
            referenced_message_ids=revision_raw.get("referenced_message_ids"),
            # 용준/Claude(2026-07-22, RAG 근거 유실 수정): revision_retrieved(이번 수정 턴에
            # 다시 검색한 근거)를 그대로 저장한다 — 위 proposal_message와 동일한 이유.
            evidence=revision_retrieved,
            known_message_ids=known_ids,
            responding_to=revision_raw.get("responding_to"),
            revision=revision_raw.get("revision"),
        )
        messages.append(revision_message)

    # 4) 진행자 최종 권고안 — 스키마에 사용자 재질문 필드가 아예 없어 같은 질문을 반복할 수
    #    없다(요청: "다시 사용자에게 같은 질문을 넘기면 안 됩니다").
    facilitator_raw, ok, attempts = generate_expert_delegation_facilitator_recommendation(
        llm_call,
        previous_state["notice_and_criteria"],
        pending_question,
        proposal_raw,
        review_raw,
        revision_raw,
    )
    if not ok or facilitator_raw is None:
        return _fail("expert_delegation_facilitator", attempts)
    used += attempts

    facilitator_message = make_expert_delegation_facilitator_message(
        round_number=previous_state["round"], raw=facilitator_raw
    )
    messages.append(facilitator_message)

    resolved_topics = list(previous_state.get("resolved_topics") or [])
    pending_topic = previous_state.get("pending_question_topic")
    if pending_topic and pending_topic not in resolved_topics:
        resolved_topics = resolved_topics + [pending_topic]

    # 임시 가정 기록은 이제 진행자의 최종 권고안을 기준으로 남긴다(요청 사항 그대로 보존 —
    # 여전히 unresolved_issues에 "임시 가정"으로 추가되지만, 여러 위원이 검토한 뒤의
    # 최종 결론을 담아야 다음 질문 프롬프트가 더 정확한 맥락을 받는다).
    assumption_note = (
        f"{persona_id}: '{pending_question}'에 대해 사용자가 판단을 위임해 위원 간 검토를 거친 "
        f"다음 임시 가정으로 진행합니다 — {facilitator_raw['final_recommendation']}"
    )
    updated_unresolved = previous_state["unresolved_issues"]
    if assumption_note not in updated_unresolved:
        updated_unresolved = updated_unresolved + [assumption_note]

    next_previous_state = IdeationConvState(
        **{
            **previous_state,
            "resolved_topics": resolved_topics,
            "unresolved_issues": updated_unresolved,
            "llm_calls_used": used,
        }
    )
    return next_previous_state, None, messages


def _topic_query_fallback(state: IdeationConvState) -> str:
    """pending_question이 비어 있을 때만 쓰는 대체 RAG 질의문 — user_idea를 그대로
    텍스트로 이어붙인다(ideation_conv_nodes.py::_topic_query와 같은 원칙이지만, 그 함수는
    비공개 헬퍼라 이 모듈에서 다시 만든다)."""
    idea = state.get("user_idea")
    if isinstance(idea, dict):
        return " ".join(str(v) for v in idea.values() if v)
    return str(idea or "")


def _apply_answer_sufficiency_gate(
    *,
    previous_state: IdeationConvState,
    persona_id: str,
    user_message: str,
    llm_call: LLMCall,
    evidence_lookup=None,
) -> tuple[IdeationConvState, IdeationConvState | None, list[ConvMessage]]:
    """사용자가 pending_question에 방금 남긴 메시지가 답변인지, 설명 요청인지, 불충분한
    답변인지, 전문가에게 판단을 위임한 것인지 판정하고 그에 맞게 처리한다(요청 3번 + 용어
    설명 요청을 재질문으로 오판하지 않기 + "모르겠다" UX 개선).

    반환값은 (다음에 apply_user_answer에 넘길 previous_state, 그래프를 돌리지 않고 즉시
    끝낼 최종 state-또는-None, apply_user_answer 직후 메시지 목록에 추가로 끼워 넣을 전문가
    위임 제안 메시지-또는-None) 3-tuple이다:
      - answer_type="answer"이거나 같은 쟁점 재질문이 이미 상한(_MAX_ANSWER_RETRY)에
        도달했으면 두 번째 값은 None이다 — 호출부가 이어서 apply_user_answer + 그래프 실행을
        정상 진행한다(요청 5번: 상한 도달 시 판정 결과와 무관하게 강제로 다음 단계로 진행,
        이때 판정이 여전히 insufficient_answer였다면 그 사실을 unresolved_issues에 "합리적
        가정"으로 남겨 둔다). 세 번째 값은 None이다.
      - answer_type="clarification_request"이면, 사용자의 요청 메시지와 설명+선택지+재질문을
        담은 명확화 응답을 메시지로 추가한 최종 state를 두 번째 값으로 반환한다. pending_question/
        pending_expected_answer_type/answer_retry_count는 전혀 바뀌지 않는다 — 사용자가 아직
        원래 질문에 답하지 않았을 뿐, 불충분한 답을 한 것이 아니기 때문이다(요청: 재질문
        횟수를 늘리지 않는다). 세 번째 값은 None이다.
      - answer_type="insufficient_answer"이고 아직 상한에 도달하지 않았으면, 사용자의 답변과
        좁혀진 재질문을 메시지로 추가한 최종 state를 두 번째 값으로 반환한다. 세 번째 값은
        None이다.
      - answer_type="expert_delegation"(용준/Claude(2026-07-21), 요청: "모르겠다" UX
        개선)이면 _delegate_to_expert()에 위임한다 — 사용자가 답 대신 전문가 판단에
        맡겼으므로 같은 질문을 반복하지 않고 다음 단계로 진행해야 한다(answer와 같은
        control-flow). 세 번째 값(전문가 제안 메시지)이 채워진다. 제안 생성 자체가
        실패하면 다른 콘텐츠 생성 노드와 동일하게 phase="failed"로 끝난다(두 번째 값).
      호출부는 두 번째 값이 있으면 그래프를 전혀 실행하지 않고(정지 지점을 새로 만들지 않고)
      그 state를 그대로 돌려준다.

    resolved_topics(요청: 질문 주제 구조화)는 answer_type이 "answer" 또는
    "expert_delegation"일 때만 pending_question_topic을 추가한다 — clarification_request/
    insufficient_answer(재질문 진행 중이든 상한 도달로 강제 진행하든)는 사용자가 그 주제에
    실제로 명확히 답하거나 위임한 게 아니므로 resolved로 표시하지 않는다(요청 3번 그대로).

    용준/Claude(2026-07-21, 요청: "모르겠다" UX 개선): 명시적인 위임 표현은 작은 모델이
    잘못 분류하지 않도록 judge_answer_sufficiency(LLM 판정) 호출 자체보다 먼저 결정적
    규칙(is_expert_delegation_request)으로 감지한다 — 매칭되면 sufficiency LLM 호출을
    아예 건너뛴다.
    """
    pending_question = previous_state.get("pending_question") or ""
    expected_answer_type = previous_state.get("pending_expected_answer_type")
    retry_count = previous_state.get("answer_retry_count", 0)
    context = conversation_context_for(previous_state)

    if is_expert_delegation_request(user_message):
        return _delegate_to_expert(
            previous_state=previous_state,
            persona_id=persona_id,
            pending_question=pending_question,
            llm_call=llm_call,
            evidence_lookup=evidence_lookup,
            context=context,
        )

    judgment = judge_answer_sufficiency(
        llm_call,
        persona_id,
        pending_question,
        user_message,
        retry_count,
        context,
        expected_answer_type,
        user_idea=previous_state.get("user_idea"),
        idea_candidates=previous_state.get("idea_candidates"),
    )
    used = previous_state.get("llm_calls_used", 0) + 1
    answer_type = judgment["answer_type"]

    if answer_type == "expert_delegation":
        return _delegate_to_expert(
            previous_state=IdeationConvState(**{**previous_state, "llm_calls_used": used}),
            persona_id=persona_id,
            pending_question=pending_question,
            llm_call=llm_call,
            evidence_lookup=evidence_lookup,
            context=context,
        )

    if answer_type == "clarification_request":
        answer_message = _new_user_message(user_message, previous_state["round"])
        clarification_message = make_clarification_message(
            persona_id=persona_id,
            round_number=previous_state["round"],
            clarification_response=judgment["clarification_response"] or judgment["reason"],
        )
        stop_state = IdeationConvState(
            **{
                **previous_state,
                "messages": previous_state["messages"] + [answer_message, clarification_message],
                "llm_calls_used": used,
                # pending_question/pending_expected_answer_type/answer_retry_count는 그대로
                # 유지한다 — 사용자는 여전히 같은 원래 질문에 답해야 한다.
            }
        )
        return previous_state, stop_state, []

    if answer_type == "answer" or retry_count >= _MAX_ANSWER_RETRY:
        updated_unresolved = previous_state["unresolved_issues"]
        resolved_topics = list(previous_state.get("resolved_topics") or [])
        if answer_type != "answer":
            # 상한 도달로 강제 진행 — 무엇이 불명확한 채로 남았는지 회의록에 남긴다. 이 주제는
            # 실제로 명확히 답해진 게 아니므로 resolved_topics에는 추가하지 않는다.
            note = f"{persona_id}: '{pending_question}'에 대한 답변이 불명확하여 다음 가정으로 진행합니다 — {judgment['reason']}"
            if note not in updated_unresolved:
                updated_unresolved = updated_unresolved + [note]
        else:
            pending_topic = previous_state.get("pending_question_topic")
            if pending_topic and pending_topic not in resolved_topics:
                resolved_topics = resolved_topics + [pending_topic]
        next_previous_state = IdeationConvState(
            **{
                **previous_state,
                "unresolved_issues": updated_unresolved,
                "resolved_topics": resolved_topics,
                "llm_calls_used": used,
            }
        )
        return next_previous_state, None, []

    follow_up_question = judgment["follow_up_question"] or pending_question
    answer_message = _new_user_message(user_message, previous_state["round"])
    follow_up_message = make_follow_up_message(
        persona_id=persona_id,
        round_number=previous_state["round"],
        reason=judgment["reason"],
        follow_up_question=follow_up_question,
    )
    stop_state = IdeationConvState(
        **{
            **previous_state,
            "messages": previous_state["messages"] + [answer_message, follow_up_message],
            "pending_question": follow_up_question,
            "answer_retry_count": retry_count + 1,
            "llm_calls_used": used,
        }
    )
    return previous_state, stop_state, []


def reply_ideation_conversation(
    *,
    previous_state: IdeationConvState,
    user_message: str,
    llm_call: LLMCall,
    evidence_lookup=None,
    ground_claims=None,
    index_target_evidence: IndexTargetEvidenceFn | None = None,
    evidence_planner=None,
    external_evidence_lookup=None,
    on_progress: IdeationConvProgressCallback | None = None,
    on_snapshot: IdeationConvSnapshotCallback | None = None,
    stop_after_expert_turn: bool = False,
    action_code: str | None = None,
    action_payload: dict | None = None,
) -> IdeationConvState:
    """사용자 답변을 반영해 다음 정지 지점까지 그래프를 이어간다.

    phase에 따라 실제로 벌어지는 일이 다르다(같은 함수로 통일해도 되는 이유는 다음에
    실행할 노드가 이미 phase 자체에 인코딩돼 있기 때문 — apply_user_answer 참고):
      - awaiting_planning_answer 중 호출: 개발 전문가 질문 1개만 만들고 다시 멈춘다
        (요청 4~5번 — 개발 전문가는 사용자가 기획 질문에 답하기 전에는 절대 실행되지 않는다).
      - awaiting_developer_answer 중 호출: 두 전문가가 순서대로 보완 의견을 말하고,
        더 물어볼 게 있으면 같은 호출 안에서 다음 질문까지 자동으로 만든 뒤 멈춘다.
      - awaiting_user_decision 중 호출: 사용자가 확정 대신 자유롭게 한 마디 더 남긴
        경우로, 두 전문가가 다시 보완 의견을 말한다.

    재인/Claude(2026-07-23, 아바타 페이싱 연동 — 실측: "진행자 2번·기획 1번·개발 1번이
    2초 간격으로 그냥 다 나왔다"): stop_after_expert_turn=False(기본값)면 위 설명대로 한
    라운드를 끝까지(또는 다음 질문까지) 다 만들고 나서야 반환한다 — 이 함수가 원래 그렇게
    설계됐고 기존 호출부(비-아바타 테스트 등)는 전부 그 동작을 기대하므로 기본값은 절대
    안 바꾼다. True면 continue_ideation_expert_turn과 똑같이 _drive_graph의
    stop_after_expert_turn을 그대로 전달한다 — 즉 "사용자가 방금 답해서 라운드가 새로
    시작되는 바로 그 첫 순간"에도 기획/개발 위원 발언 1건에서 멈춘다. 이래야 라운드의
    첫 발언부터 마지막(진행자 정리)까지 전부 아바타 재생 페이싱(끝나기 3초 전 다음 요청)을
    거치게 된다 — 첫 턴만 통째로 오고 그 다음부터만 끊기는 반쪽짜리 페이싱이 되지 않는다.

    용준/Claude(2026-07-27, 후속 요청 4번: "2차 프론트에서는 action code를 함께 보낼
    예정 — 백엔드는 action code를 우선 사용하고 자연어 키워드 판정은 하위 호환용
    폴백으로 유지") — action_code/action_payload는 순수 추가 파라미터다(기본값 None).
    넘기면 이번 한 번의 그래프 호출에서만 state["pending_user_action"]으로 실려
    problem_focus_selection/conflict_resolution/concept_confirmation 노드가 텍스트
    파싱보다 먼저 확인한다(ideation_conv_problem.py 각 노드 참고). 넘기지 않으면(기존
    클라이언트) 기존과 완전히 동일하게 자연어 파싱만 동작한다."""
    if previous_state["phase"] not in REPLYABLE_PHASES:
        raise ValueError(
            f"사용자 답변을 받을 수 없는 phase입니다: {previous_state['phase']!r}. "
            f"허용된 phase: {sorted(REPLYABLE_PHASES)}"
        )

    # 용준/Claude(2026-07-27, 후속 요청 2번) — action_code가 명시적으로 전달되면 여기서
    # 먼저 검증한다. 실패하면 ValueError를 그대로 던진다(자연어 폴백으로 조용히 넘어가지
    # 않는다) — 아래 재생성 키워드 단축 경로와 자연어 파싱 전부보다 먼저 실행되어야
    # "action_code는 있는데 메시지 텍스트가 우연히 다른 키워드와 겹쳐 엉뚱하게 처리되는"
    # 상황을 막을 수 있다.
    if action_code:
        _validate_action_code(previous_state, action_code, action_payload or {})

    # discovery 세션에서는 후보를 선택한 뒤 기획/개발 질문으로 넘어간
    # 상태에서도 "아이디어 다시 짜줘" 의도를 최우선으로 처리한다. 이 가드가
    # 재질문 충분성 판정보다 먼저 실행되어야 재생성 요청을 "질문에 대한
    # 불충분한 답변"으로 오판해 동일한 질문을 반복하지 않는다. action_code가 명시적으로
    # 전달됐으면 이 자연어 단축 경로 자체를 건너뛴다 — 이미 검증된 구조화 액션이 우선이다.
    if (
        not action_code
        and previous_state.get("ideation_mode") == "discovery"
        and previous_state["phase"] != "awaiting_candidate_selection"
        and is_regenerate_request(user_message)
    ):
        regeneration_count = previous_state.get("candidate_regeneration_count", 0)
        answer_message = _new_user_message(user_message, previous_state["round"])
        if regeneration_count >= MAX_CANDIDATE_REGENERATIONS:
            notice = _new_facilitator_message(
                f"후보 재추천은 최대 {MAX_CANDIDATE_REGENERATIONS}회까지 가능합니다. "
                "현재 아이디어를 계속 발전시키거나 새 회의를 시작해 주세요.",
                previous_state["round"],
            )
            capped_state = IdeationConvState(
                **{
                    **previous_state,
                    "messages": previous_state["messages"] + [answer_message, notice],
                }
            )
            if on_progress is not None:
                on_progress(_progress(capped_state))
            return capped_state

        restart_state = IdeationConvState(
            **{
                **previous_state,
                "messages": previous_state["messages"] + [answer_message],
                "phase": "candidate_generation",
                "round": 1,
                "pending_question": None,
                "pending_expected_answer_type": None,
                "pending_question_topic": None,
                "resolved_topics": [],
                "consensus": [],
                "unresolved_issues": [],
                "idea_proposal": None,
                "failed_node": None,
                "answer_retry_count": 0,
                "selected_idea": None,
                "selected_idea_document_id": None,
                "selection_reason": None,
                "selection_intent": None,
                "user_selection_message": None,
                "source_candidates": [],
                "merge_analysis": None,
                "candidate_regeneration_count": regeneration_count + 1,
            }
        )
        graph = assemble_ideation_conversation_graph(
            llm_call,
            evidence_lookup=evidence_lookup,
            ground_claims=ground_claims,
            index_target_evidence=index_target_evidence,
            evidence_planner=evidence_planner,
            external_evidence_lookup=external_evidence_lookup,
        )
        restart_baseline = len(restart_state["messages"])
        restart_result = _drive_graph(
            graph, restart_state, on_progress, on_snapshot, stop_after_expert_turn=stop_after_expert_turn
        )
        return _guard_pre_lock_messages(restart_result, restart_baseline)

    pending_persona = PHASE_TO_PENDING_PERSONA.get(previous_state["phase"])
    extra_message: ConvMessage | None = None
    if pending_persona is not None:
        # awaiting_planning_answer/awaiting_developer_answer: 지금 답한 것이 특정 질문에
        # 대한 답변이므로 재질문 여부를 먼저 판정한다(요청 3번). awaiting_user_decision(사용자가
        # 확정 대신 자유롭게 남긴 한마디)에는 이 게이트를 적용하지 않는다 — 특정 질문에 대한
        # 답이 아니라 자유 발언이기 때문이다.
        previous_state, follow_up_state, extra_messages = _apply_answer_sufficiency_gate(
            previous_state=previous_state,
            persona_id=pending_persona,
            user_message=user_message,
            llm_call=llm_call,
            evidence_lookup=evidence_lookup,
        )
        if follow_up_state is not None:
            if on_progress is not None:
                on_progress(_progress(follow_up_state))
            return follow_up_state
    else:
        extra_messages = []

    answer_message = _new_user_message(
        user_message, previous_state["round"], message_type=_reply_message_type_for(previous_state)
    )
    # 용준/Claude(2026-07-22, 요청: 사용자 답변을 session target evidence로 반영 + 인덱싱
    # 완료 후 다음 전문가 턴 실행) — apply_user_answer/그래프 실행보다 먼저, 이 답변을 동기적으로
    # target evidence로 색인한다. 이 함수가 반환한 뒤에야 아래 _drive_graph가 다음 전문가 노드를
    # 실행하므로, 색인 전에 다음 전문가 검색이 시작되는 race condition이 없다(요청 9번).
    _index_user_answer(
        state=previous_state,
        answer_message=answer_message,
        raw_answer_text=user_message,
        index_target_evidence=index_target_evidence,
    )
    state = apply_user_answer(previous_state, answer_message)
    if action_code:
        state = IdeationConvState(
            **{**state, "pending_user_action": {"code": action_code, "payload": action_payload or {}}}
        )
    if extra_messages:
        # 전문가 위임 제안 흐름(요청: "모르겠다" UX 개선 + 위원 간 상호 검토 확장) — 사용자의
        # 원문 메시지 바로 다음에 [담당 위원 제안, 반대 위원 검토, (있으면) 수정, 진행자
        # 권고안] 순서로 이어지도록, apply_user_answer가 사용자 메시지를 넣은 직후 그대로
        # 끼워 넣는다. 이후 프롬프트(conversation_context_for의 recent_messages)와 최종
        # 종합(synthesis)이 messages 전체를 그대로 참조하므로, 이 발언들도 자연히 그
        # 컨텍스트에 포함된다(요청 8번).
        state = IdeationConvState(**{**state, "messages": state["messages"] + extra_messages})
    if not is_graph_entry_phase(state["phase"]):
        # 방어적 점검 — apply_user_answer()는 항상 그래프 진입 가능한 phase로만 전이시키므로
        # 정상 흐름에서는 절대 여기 도달하지 않는다.
        raise AssertionError(f"apply_user_answer가 진입 불가능한 phase를 반환했습니다: {state['phase']!r}")
    graph = assemble_ideation_conversation_graph(
        llm_call,
        evidence_lookup=evidence_lookup,
        ground_claims=ground_claims,
        index_target_evidence=index_target_evidence,
        evidence_planner=evidence_planner,
        external_evidence_lookup=external_evidence_lookup,
    )
    reply_baseline = len(state["messages"])
    result_state = _drive_graph(graph, state, on_progress, on_snapshot, stop_after_expert_turn=stop_after_expert_turn)

    if result_state.get("forced_next_speaker") is not None:
        # 2026-07-26 라운드테이블 재설계: apply_user_answer가 awaiting_user_decision 직후
        # forced_next_speaker="facilitator"를 심어 discussion_facilitator로 바로 진입시킨다
        # (사용자가 진행자의 질문/선택지에 막 답한 경우, 전문가를 다시 거치지 않기 위함).
        # discussion_facilitator_node는 이 값을 스스로 리셋하지 않으므로(다른 두 전문가
        # 노드와 달리 원래 forced 진입 대상이 아니었던 노드라서 — continue_ideation_expert_turn의
        # 같은 정리 로직 참고) 다음 요청에 잔류하지 않도록 여기서 확실히 지운다.
        result_state = IdeationConvState(**{**result_state, "forced_next_speaker": None})
    return _guard_pre_lock_messages(result_state, reply_baseline)


def continue_ideation_expert_turn(
    *,
    previous_state: IdeationConvState,
    llm_call: LLMCall,
    evidence_lookup=None,
    ground_claims=None,
    index_target_evidence: IndexTargetEvidenceFn | None = None,
    evidence_planner=None,
    external_evidence_lookup=None,
    on_progress: IdeationConvProgressCallback | None = None,
    on_snapshot: IdeationConvSnapshotCallback | None = None,
) -> IdeationConvState:
    """재인/Claude(2026-07-23, 아바타 페이싱 연동): 새 사용자 입력 없이, 지금 진행 중인
    라운드에서 다음 발언 하나만 더 만들어서 반환한다. 아바타가 방금 발언을 재생하는
    도중(재생 끝나기 3초 전) 다음 위원 영상을 미리 준비시키려고 호출하는 함수 — "누가
    다음에 말할지"는 새로 판단하지 않고 기존 _route_next_expert_turn(그래프 조건부 엣지가
    실제로 쓰는 그 함수)을 그대로 재사용한다. 회의 로직은 바뀌지 않고, 언제 멈추고 언제
    다시 부르는지만 다르다.

    다음 화자가 기획/개발 위원이면 그 발언 1건에서 정확히 멈춘다(_drive_graph의
    stop_after_expert_turn). 다음 화자가 진행자(facilitator)면 forced_next_speaker="facilitator"로
    강제 진입시켜 진행자 발언 + 캔버스 갱신까지만 진행하고 멈춘다(캔버스 갱신은 화면에
    보이는 발언이 없으므로 조용히 같이 반영되지만, 다음 라운드 첫 위원 발언까지 이어서
    만들지는 않는다 — 재인/Claude(2026-07-24, 실측: "진행자 정리 발언과 다음 라운드
    기획위원 발언이 동시에 나옴") _drive_graph 쪽 stop_after_next_snapshot 참고). 다음
    라운드가 자동으로 이어지는 경우(continue_round)에도 그 첫 위원 발언은 이 호출에 안
    끼고, 아바타가 방금 반환된 진행자 발언을 다 재생한 뒤 이 함수가 다시 호출될 때
    아래 "라운드 첫 턴" 부트스트랩 분기가 처리한다 — 결과적으로 어느 경우든 "화면에 보일
    발언 하나"씩 딱딱 끊어서 멈추게 된다.

    previous_state["phase"]가 "expert_discussion"이 아니면 호출할 수 없다(라운드가 이미
    끝났거나 사용자 입력을 기다리는 중이라는 뜻 — 호출부가 먼저 걸러야 한다).

    재인/Claude(2026-07-23, 실측: "선택 직후 진행자 안건 소개 끝나면 기획위원이 먼저
    말해야 하는데 요청이 이상하게 감"): _route_next_expert_turn은 "방금 기획/개발위원이
    말한 직후"에만 불리도록 설계된 함수라(그 함수 자체 주석: "정상 흐름에서는 항상 방금
    전문가 발언 직후에만 이 라우터가 불린다"), 아직 이번 라운드에서 위원이 한 번도 안
    말한 시점에 그대로 부르면 자기 방어 코드(missing_expert_message)가 "facilitator"를
    반환해버려 진행자가 또 진행자를 부르는 잘못된 결과가 나온다. 이 경우는
    _route_next_expert_turn을 아예 부르지 않는다.

    용준/Claude(2026-07-26, 라운드테이블 재설계 — 실측: "후보 선택 직후 고정 1턴 없이
    바로 위원이 말함"): 위 문단은 원래 "이 경우는 무조건 기획위원이 먼저 말한다"였다.
    하지만 candidate_selection이 끝나면 항상 정지하는 _drive_graph의 single_turn 멈춤
    지점(요청: "선택 확정 + 안건 소개"를 한 박자로 보여주기 위해 진행자 메시지 2개가
    같은 스냅샷에 있으면 그래프가 discussion_facilitator를 실행하기도 전에 멈춘다) 때문에,
    이 함수가 프론트의 자동 이어달리기(useEffect 루프, IdeationConversationScreen.jsx)로
    다음 턴을 이어받을 때 이 지점을 항상 통과한다 — 즉 "이번 라운드에서 위원이 아직 말한
    적 없음"은 두 가지 서로 다른 상황을 뭉뚱그리고 있었다: (a) 진행자가 방금 새 주제를
    열어서(continue_round) 위원 차례가 된 경우 — 기획위원이 먼저 말하는 게 맞다, (b) 이
    세션에서 discussion_facilitator가 아직 한 번도 실제 턴을 낸 적이 없는 경우
    (candidate_selection이 붙인 정적 요약/안건 메시지뿐, structured 없음) — "고정 1턴"
    요건(후보 선택 직후 진행자가 전문가 없이 먼저 요약+문제정의를 확인)에 따라 기획위원이
    아니라 진행자가 먼저 말해야 한다. (a)/(b) 구분 없이 무조건 기획위원으로 보내던 게
    바로 이 버그였다."""
    if previous_state.get("phase") != "expert_discussion":
        raise ValueError(
            "continue_ideation_expert_turn은 phase가 'expert_discussion'일 때만 호출할 수 "
            f"있습니다(현재: {previous_state.get('phase')!r})."
        )

    messages = previous_state.get("messages") or []
    last_message = messages[-1] if messages else None
    if last_message is not None and last_message.get("speaker_id") in ("planning_expert", "dev_expert"):
        next_target = _route_next_expert_turn(previous_state)
    elif any(m.get("speaker_id") == "ideation_facilitator" and m.get("structured") for m in messages):
        # (a) 진행자가 이미 실제 턴을 낸 적이 있다 — 방금 새 주제를 연 것이므로 기획위원이
        # 먼저 말한다(그래프 기본 진입 규칙과 동일한 관례).
        next_target = "planning_expert"
    else:
        # (b) 이 세션에서 진행자가 아직 한 번도 실제 턴을 낸 적이 없다 — "고정 1턴"이
        # 먼저다. discussion_facilitator 노드 자신이 이 경우를 감지해(is_first_facilitator_turn)
        # 전문가 없이 사용자에게 바로 묻는다.
        next_target = "facilitator"
    if next_target not in _FORCED_ENTRY_TARGETS:
        # "failed" — 더 진행할 턴이 없다. 그대로 반환(호출부가 phase 등을 보고 처리).
        return previous_state

    state = IdeationConvState(**{**previous_state, "forced_next_speaker": next_target})
    graph = assemble_ideation_conversation_graph(
        llm_call,
        evidence_lookup=evidence_lookup,
        ground_claims=ground_claims,
        index_target_evidence=index_target_evidence,
        evidence_planner=evidence_planner,
        external_evidence_lookup=external_evidence_lookup,
    )
    turn_baseline = len(state["messages"])
    result_state = _drive_graph(graph, state, on_progress, on_snapshot, stop_after_expert_turn=True)

    if result_state.get("forced_next_speaker") is not None:
        # 재인/Claude(2026-07-23): forced_next_speaker="facilitator"로 강제 진입한 뒤 라운드가
        # 그대로 끝나면(await_user_decision) discussion_facilitator_node는 이 값을 리셋하지
        # 않는다 — planning/dev 노드(make_conv_discussion_node)는 매번 스스로 None으로
        # 리셋하지만, facilitator는 원래 forced 진입 대상이 아니었던 노드라 그 리셋 로직이
        # 없다(회의 로직 자체를 건드리지 않으려고 그 노드 코드는 그대로 뒀다). 다음 라운드가
        # 시작될 때 이 값이 그대로 남아있으면 _route_entry가 엉뚱하게 facilitator로 바로
        # 진입해버리므로, 여기서 확실히 지운다.
        result_state = IdeationConvState(**{**result_state, "forced_next_speaker": None})
    return _guard_pre_lock_messages(result_state, turn_baseline)


def finalize_ideation_conversation(
    *,
    previous_state: IdeationConvState,
    llm_call: LLMCall,
    on_progress: IdeationConvProgressCallback | None = None,
    on_snapshot: IdeationConvSnapshotCallback | None = None,
) -> IdeationConvState:
    """사용자가 "주제 확정하고 초안 받기"를 눌렀을 때만 호출된다(요청 9~10항). phase가
    awaiting_user_decision이 아니면 request_finalize()가 ValueError를 던진다 — 호출부
    (API 라우터)가 이를 400으로 변환해야 한다."""
    state = request_finalize(previous_state)
    graph = assemble_ideation_conversation_graph(llm_call)
    return _drive_graph(graph, state, on_progress, on_snapshot)


# 가은/Claude(2026-07-27, 버그 리포트: "target_fields 26개 세션에서 LLM이 매번 11~15개만
# 채우고 나머지를 빠뜨림") — 필드 수가 많을수록 "한 번의 JSON 응답에 전부 빠짐없이"라는
# 요구 자체가 안 지켜지는 빈도가 늘어난다. 한 번에 요청하는 필드 수를 이 상한으로 쪼갠다.
_FORM_DRAFT_BATCH_SIZE = 10


def _chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _fill_form_draft_batch(
    *,
    llm_call: LLMCall,
    notice_and_criteria: Any,
    idea_proposal: Any,
    working_draft: list[dict],
    batch: list[dict],
    session_id: str | None,
) -> tuple[list[dict], int, list[str], int]:
    """batch(최대 _FORM_DRAFT_BATCH_SIZE개 필드)를 채운다. 요청 제안 2번("전부 아니면 실패
    대신 채워진 필드는 그대로 적용하고 누락분만 재시도") 그대로 — _safe_call_structured_json은
    검증 실패 시 raw 자체를 버리므로(all-or-nothing 계약) 여기서는 쓰지 않고, 채워진 필드는
    즉시 적용한 뒤 다음 시도에서 남은 필드만 다시 요청한다. 반환값은
    (updated_working_draft, applied_count, supplement_notes, attempts_used)."""
    remaining = list(batch)
    applied_count = 0
    supplement_notes: list[str] = []
    attempts_used = 0

    for attempt in range(2):  # 최초 1회 + 누락분 재시도 1회(요청 10번과 동일한 상한 정책)
        if not remaining:
            break
        remaining_ids = {str(row.get("field_id")) for row in remaining}
        prompt = build_ideation_conv_form_draft_prompt(
            notice_and_criteria, idea_proposal, working_draft, remaining
        )
        attempts_used += 1
        llm_text = llm_call(prompt)
        try:
            raw = parse_json_response(llm_text)
        except (ValueError, KeyError, TypeError):
            trace_event(
                "IDEATION_FORM_DRAFT_BATCH_PARSE_FAILED",
                level=30,
                session_id=session_id,
                attempt=attempt + 1,
                batch_field_count=len(remaining),
            )
            log_llm_response(
                node_name="form_draft_batch",
                attempt=attempt + 1,
                ok=False,
                reason="json_parse_failed",
                raw_response=llm_text,
            )
            continue

        fields = raw.get("fields") if isinstance(raw, dict) else None
        log_llm_response(
            node_name="form_draft_batch",
            attempt=attempt + 1,
            ok=isinstance(fields, dict),
            reason=None if isinstance(fields, dict) else "fields_missing_or_not_a_dict",
            raw_response=llm_text,
        )
        low_confidence_count = 0
        if isinstance(fields, dict):
            patch = []
            for field_id, field_data in fields.items():
                if not isinstance(field_data, dict):
                    continue
                patch.append({"field_id": field_id, "value": field_data.get("value")})
                if str(field_data.get("confidence") or "").strip().lower() == "low":
                    low_confidence_count += 1
            working_draft, applied = apply_application_form_draft_patch(
                working_draft, patch, confirmable_field_ids=remaining_ids
            )
            applied_ids = {row["field_id"] for row in applied}
            for row in working_draft:
                if row.get("field_id") in applied_ids:
                    row["status"] = "confirmed"
            applied_count += len(applied)
            remaining = [row for row in remaining if str(row.get("field_id")) not in applied_ids]
        if isinstance(raw, dict):
            field_names = {
                str(row.get("field_id")): str(row.get("field_name") or "")
                for row in working_draft
            }
            for note in raw.get("missing_information") or []:
                if not isinstance(note, dict):
                    continue
                missing = str(note.get("missing") or "").strip()
                if not missing:
                    continue
                label = field_names.get(str(note.get("field_id") or "").strip())
                reason = str(note.get("reason") or "").strip()
                text = f"{label}: {missing}" if label else missing
                if reason:
                    text = f"{text} ({reason})"
                supplement_notes.append(text)
        trace_event(
            "IDEATION_FORM_DRAFT_BATCH_PROGRESS",
            session_id=session_id,
            attempt=attempt + 1,
            requested_field_count=len(remaining_ids),
            still_missing_field_count=len(remaining),
            low_confidence_field_count=low_confidence_count,
        )

    return working_draft, applied_count, supplement_notes, attempts_used


def generate_application_form_draft(
    *,
    previous_state: IdeationConvState,
    llm_call: LLMCall,
) -> IdeationConvState:
    """가은/Claude(2026-07-27, 요청: "주제 확정하고 아래에 신청서 초안 버튼 하나 만들어서
    페이지로 하나 띄워주자") — 주제가 확정된(phase="finalized") 세션에서, 사용자가 회의 중
    선택한 신청양식 항목(application_form_items) 중 아직 대화로 확정되지 않은 필드를
    idea_proposal(방금 만든 종합 결과)을 근거로 문서체로 채운다.

    가은/Claude(2026-07-27, 버그 리포트 반영) — target_fields가 많은 세션(예: 26개)에서
    한 번의 JSON 응답에 전부 채우라고 요구하면 매 시도 다른 필드 조합이 빠지는 문제가
    있었다. _FORM_DRAFT_BATCH_SIZE 단위로 나눠 호출하고, 배치 안에서도 채워진 필드는 그대로
    적용한 뒤 누락분만 재시도한다(제안 1·2번 모두 반영) — 그래도 못 채운 필드는 status가
    "empty"로 남을 뿐 요청 전체를 실패시키지 않는다. remaining_content_fields()가 매번
    현재 draft에서 다시 계산하므로, 사용자가 버튼을 다시 누르면 그때 남은 필드만 대상이 된다.

    LangGraph 노드가 아니라 독립 함수다 — 그래프 라운드 진행과 무관한 1회성 후처리라
    ideation_conv_build.py의 그래프 구조(다른 담당자가 계속 손대는 중인 라운드테이블
    재설계)를 건드리지 않는다. status="confirmed"인 필드는 remaining_content_fields()가
    애초에 target_fields에서 제외하므로 이 함수가 그 값을 덮어쓸 방법이 없다."""
    if previous_state["phase"] != "finalized":
        raise ValueError(
            f"신청서 초안은 주제가 확정된 뒤에만 만들 수 있습니다: phase={previous_state['phase']!r}"
        )
    if not previous_state.get("application_form_items"):
        raise ValueError("이 세션에는 선택된 신청 양식 항목이 없습니다.")

    current_draft = previous_state.get("application_form_draft") or []
    target_fields = remaining_content_fields(current_draft)
    if not target_fields:
        # 이미 대화 중 전부 확정됐다 — 새로 만들 것 없이 현재 상태 그대로 반환.
        return previous_state

    session_id = previous_state.get("session_id")
    working_draft = current_draft
    used = previous_state.get("llm_calls_used", 0)
    total_applied = 0
    supplement_notes: list[str] = []

    for batch in _chunked(target_fields, _FORM_DRAFT_BATCH_SIZE):
        working_draft, applied_count, batch_notes, attempts_used = _fill_form_draft_batch(
            llm_call=llm_call,
            notice_and_criteria=previous_state["notice_and_criteria"],
            idea_proposal=previous_state.get("idea_proposal"),
            working_draft=working_draft,
            batch=batch,
            session_id=session_id,
        )
        used += attempts_used
        total_applied += applied_count
        supplement_notes.extend(batch_notes)

    if total_applied == 0:
        trace_event(
            "IDEATION_FORM_DRAFT_GENERATION_FAILED",
            level=30,
            session_id=session_id,
            target_field_count=len(target_fields),
        )
        raise RuntimeError("신청서 초안 생성에 실패했습니다. 다시 시도해 주세요.")

    trace_event(
        "IDEATION_FORM_DRAFT_GENERATED",
        session_id=session_id,
        target_field_count=len(target_fields),
        applied_field_count=total_applied,
        supplement_note_count=len(supplement_notes),
    )
    return IdeationConvState(
        **{
            **previous_state,
            "application_form_draft": working_draft,
            "application_form_supplement_notes": supplement_notes,
            "llm_calls_used": used,
        }
    )

