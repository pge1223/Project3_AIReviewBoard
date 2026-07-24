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

from .ideation_conv_build import assemble_ideation_conversation_graph
from .ideation_conv_discovery import MAX_CANDIDATE_REGENERATIONS, is_regenerate_request
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
from .ideation_nodes import call_evidence_lookup
from .ideation_conv_state import (
    ConvMessage,
    IdeationCancelled,
    IdeationConvState,
    apply_user_answer,
    initial_conv_state,
    is_graph_entry_phase,
    request_finalize,
)
from .ideation_trace import sanitize_preview, trace_event
from .llm import LLMCall

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
}

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
                    if last_speaker in _SINGLE_TURN_STOP_SPEAKERS:
                        break
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
    on_progress: IdeationConvProgressCallback | None = None,
    on_snapshot: IdeationConvSnapshotCallback | None = None,
    application_form_items: list[dict] | None = None,
) -> IdeationConvState:
    """세션을 시작해 기획 전문가의 첫 질문 하나만 만들고 멈춘다(요청 목표 흐름 1~3번).

    가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입): application_form_items는 순수
    추가 파라미터다(기본값 None) — 넘기지 않으면 기존 호출부와 완전히 동일하게 동작한다."""
    graph = assemble_ideation_conversation_graph(
        llm_call,
        evidence_lookup=evidence_lookup,
        ground_claims=ground_claims,
        index_target_evidence=index_target_evidence,
        evidence_planner=evidence_planner,
    )
    state = initial_conv_state(
        session_id, notice_and_criteria, user_idea, max_rounds=max_rounds,
        application_form_items=application_form_items,
    )
    return _drive_graph(graph, state, on_progress, on_snapshot)


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
    on_progress: IdeationConvProgressCallback | None = None,
    on_snapshot: IdeationConvSnapshotCallback | None = None,
    stop_after_expert_turn: bool = False,
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
    거치게 된다 — 첫 턴만 통째로 오고 그 다음부터만 끊기는 반쪽짜리 페이싱이 되지 않는다."""
    if previous_state["phase"] not in REPLYABLE_PHASES:
        raise ValueError(
            f"사용자 답변을 받을 수 없는 phase입니다: {previous_state['phase']!r}. "
            f"허용된 phase: {sorted(REPLYABLE_PHASES)}"
        )

    # discovery 세션에서는 후보를 선택한 뒤 기획/개발 질문으로 넘어간
    # 상태에서도 "아이디어 다시 짜줘" 의도를 최우선으로 처리한다. 이 가드가
    # 재질문 충분성 판정보다 먼저 실행되어야 재생성 요청을 "질문에 대한
    # 불충분한 답변"으로 오판해 동일한 질문을 반복하지 않는다.
    if (
        previous_state.get("ideation_mode") == "discovery"
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
        )
        return _drive_graph(graph, restart_state, on_progress, on_snapshot, stop_after_expert_turn=stop_after_expert_turn)

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
    )
    return _drive_graph(graph, state, on_progress, on_snapshot, stop_after_expert_turn=stop_after_expert_turn)


def continue_ideation_expert_turn(
    *,
    previous_state: IdeationConvState,
    llm_call: LLMCall,
    evidence_lookup=None,
    ground_claims=None,
    index_target_evidence: IndexTargetEvidenceFn | None = None,
    evidence_planner=None,
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
    말한 시점(방금 진행자 안건 소개만 끝난 직후)에 그대로 부르면 자기 방어 코드
    (missing_expert_message)가 "facilitator"를 반환해버려 진행자가 또 진행자를 부르는
    잘못된 결과가 나온다. 이 경우는 _route_next_expert_turn을 아예 부르지 않고, 그래프의
    기본 진입 규칙(_route_entry의 기본값 = planning_expert_discussion, ideation_conv_build.py
    ::_ENTRY_NODES)과 동일하게 "라운드 첫 턴은 항상 기획위원"으로 직접 정한다 — 이것도
    새 판단 로직이 아니라 이미 있는 그래프 관례를 그대로 따르는 것뿐이다."""
    if previous_state.get("phase") != "expert_discussion":
        raise ValueError(
            "continue_ideation_expert_turn은 phase가 'expert_discussion'일 때만 호출할 수 "
            f"있습니다(현재: {previous_state.get('phase')!r})."
        )

    messages = previous_state.get("messages") or []
    last_message = messages[-1] if messages else None
    if last_message is None or last_message.get("speaker_id") not in ("planning_expert", "dev_expert"):
        # 이번 라운드에서 위원이 아직 한 번도 안 말함 — _route_next_expert_turn을 쓸 수
        # 없는 케이스(위 설명 참고). 그래프 기본 진입 규칙과 동일하게 기획위원이 먼저 말한다.
        next_target = "planning_expert"
    else:
        next_target = _route_next_expert_turn(previous_state)
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
    )
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
    return result_state


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


_TARGET_TO_FORCED_SPEAKER = {"planning_expert": "planning_expert", "dev_expert": "dev_expert"}
_INTERJECTION_COUNTERPART = {"planning_expert": "dev_expert", "dev_expert": "planning_expert"}
# expert_discussion 계열 phase(취소 직후에도 canonical state의 phase는 여전히
# "expert_discussion"이다 — 아직 한 라운드가 끝나지 않았으므로) + 기존 REPLYABLE_PHASES(라운드
# 사이 자유 질문에도 대상 지정을 허용하는 자연스러운 확장) 양쪽에서 인터럽션을 받는다.
INTERJECTION_REPLYABLE_PHASES = REPLYABLE_PHASES | {"expert_discussion"}


def reply_to_interjection(
    *,
    previous_state: IdeationConvState,
    user_message: str,
    target_speaker_id: str,
    llm_call: LLMCall,
    opinion_target_speaker_id: str | None = None,
    interrupted_speaker_id: str | None = None,
    evidence_lookup=None,
    ground_claims=None,
    index_target_evidence: IndexTargetEvidenceFn | None = None,
    evidence_planner=None,
    on_progress: IdeationConvProgressCallback | None = None,
    on_snapshot: IdeationConvSnapshotCallback | None = None,
) -> IdeationConvState:
    """용준/Claude(2026-07-22, 요청: "잠시만" 재개 — 지정 위원 우선 응답): 사용자가 "잠시만"
    으로 진행 중이던 발언을 취소한 뒤(또는 라운드 사이 자유롭게) 특정 위원을 지정해 질문했을
    때 호출된다. reply_ideation_conversation과 분리한 이유: phase 게이트가 다르고
    (expert_discussion 자체도 허용해야 한다), "지정 위원이 먼저 답하고 상대가 반드시
    검토한다"를 보장하려면 진입 노드를 강제해야 하기 때문이다(forced_next_speaker).

    target_speaker_id="both"면 active_issue_id가 가리키는 쟁점의 가장 최근 발언자 반대편이
    먼저 답한다(마지막이 planning_expert였다면 dev_expert 먼저) — 찾을 수 없으면 기본값
    planning_expert가 먼저 답한다. 지정 위원 응답 이후 라우팅은 기존
    _route_next_expert_turn을 그대로 타므로(recommended_next_speaker가 상대 위원이면 자동으로
    상대가 검토), "지정 위원 답변 → 상대 검토"가 그래프 구조 변경 없이 보장된다."""
    if previous_state["phase"] not in INTERJECTION_REPLYABLE_PHASES:
        raise ValueError(
            f"이 phase에서는 위원 지정 질문을 받을 수 없습니다: {previous_state['phase']!r}."
        )
    if target_speaker_id not in ("planning_expert", "dev_expert", "both"):
        raise ValueError(f"target_speaker_id가 올바르지 않습니다: {target_speaker_id!r}")
    opinion_target_speaker_id = opinion_target_speaker_id or target_speaker_id
    if opinion_target_speaker_id not in ("planning_expert", "dev_expert", "both"):
        raise ValueError(
            f"opinion_target_speaker_id가 올바르지 않습니다: {opinion_target_speaker_id!r}"
        )
    if interrupted_speaker_id not in (None, "planning_expert", "dev_expert"):
        raise ValueError(f"interrupted_speaker_id가 올바르지 않습니다: {interrupted_speaker_id!r}")

    if target_speaker_id == "both":
        last_expert_message = None
        for msg in reversed(previous_state["messages"]):
            if msg.get("speaker_id") in ("planning_expert", "dev_expert"):
                last_expert_message = msg
                break
        if last_expert_message and last_expert_message["speaker_id"] == "planning_expert":
            forced_speaker = "dev_expert"
        else:
            forced_speaker = "planning_expert"
    else:
        forced_speaker = _TARGET_TO_FORCED_SPEAKER[target_speaker_id]

    opinion_speakers = (
        {"planning_expert", "dev_expert"}
        if opinion_target_speaker_id == "both"
        else {opinion_target_speaker_id}
    )
    # 완료되지 않은 중단 발언은 메시지로 저장하지 않으므로 과거의 같은 위원 발언을 잘못
    # 연결하지 않는다. 나머지 선택 대상은 가장 최근 완료 발언을 명시적으로 참조한다.
    referenced_message_ids: list[str] = []
    for speaker_id in opinion_speakers:
        if speaker_id == interrupted_speaker_id:
            continue
        referenced = next(
            (
                message
                for message in reversed(previous_state["messages"])
                if message.get("speaker_id") == speaker_id
            ),
            None,
        )
        if referenced and referenced.get("message_id"):
            referenced_message_ids.append(referenced["message_id"])

    interjection_message = ConvMessage(
        message_id=f"MSG-{uuid.uuid4().hex[:10]}",
        speaker_id="user",
        speaker_name="사용자",
        role="사용자",
        round=previous_state["round"],
        message_type="interjection",
        content=user_message,
        referenced_message_ids=referenced_message_ids,
        evidence=[],
        created_at=datetime.now(timezone.utc).isoformat(),
        structured={
            "target_speaker_id": target_speaker_id,
            "opinion_target_speaker_id": opinion_target_speaker_id,
            "interrupted_speaker_id": interrupted_speaker_id,
            "active_issue_id": previous_state.get("active_issue_id"),
        },
    )
    # 용준/Claude(2026-07-22, 요청: 사용자 답변을 session target evidence로 반영) — "잠시만"
    # 인터젝션도 구체적인 개입이면 색인 대상이다(reply_ideation_conversation과 동일한 분류
    # 규칙 재사용). 아래 _drive_graph보다 먼저 실행되므로 인덱싱 완료 후에만 다음 전문가
    # 검색이 시작된다(요청 9번).
    _index_user_answer(
        state=previous_state,
        answer_message=interjection_message,
        raw_answer_text=user_message,
        index_target_evidence=index_target_evidence,
    )

    state = IdeationConvState(
        **{
            **previous_state,
            "phase": "expert_discussion",
            "messages": previous_state["messages"] + [interjection_message],
            "forced_next_speaker": forced_speaker,
            "pending_question": None,
            "pending_question_topic": None,
            # 용준/Claude(2026-07-22, 요청: 지정 위원 질문 후 상대 검토 코드 강제) — forced_speaker가
            # 답하고 나면 반드시 반대편(_INTERJECTION_COUNTERPART)이 한 번 더 검토해야 하고,
            # 그 전까지는 _route_next_expert_turn이 facilitator로 이동하지 못한다(그래프
            # 라우팅이 아니라 이 네 필드가 강제한다 — 요청 상태 필드 그대로).
            "interjection_target_speaker_id": target_speaker_id,
            "interjection_response_message_id": None,
            "required_counterpart_speaker_id": _INTERJECTION_COUNTERPART[forced_speaker],
            "counterpart_review_completed": False,
        }
    )
    graph = assemble_ideation_conversation_graph(
        llm_call,
        evidence_lookup=evidence_lookup,
        ground_claims=ground_claims,
        index_target_evidence=index_target_evidence,
        evidence_planner=evidence_planner,
    )
    return _drive_graph(graph, state, on_progress, on_snapshot)
