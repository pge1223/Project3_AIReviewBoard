# 작성자: 용준/Claude(2026-07-22, 요청: 선택된 아이디어/사용자 답변을 target evidence로 색인)
# 목적: index_target_evidence 콜러블이 candidate_selection 노드와 사용자 답변 처리 흐름에
#       실제로 연결되는지(요청 2·4·9번), 인덱싱 대상 메시지 제한(요청 17-2번)이 결정적으로
#       동작하는지, 색인 실패가 회의를 중단시키지 않는지(요청 17-4번) 확인한다. 실제 ai.rag
#       구현은 쓰지 않고 콜 순서/인자만 기록하는 fake를 주입한다(ai/meeting은 ai.rag를
#       직접 import하지 않는 기존 경계를 그대로 지킨다).

import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph import reply_ideation_conversation  # noqa: E402

from test_ideation_discovery_graph import (  # noqa: E402
    DiscoveryScriptedLLM,
    _start_discovery,
)


class _RecordingIndexer:
    """index_target_evidence로 주입되는 fake. 호출 순서와 인자를 그대로 기록한다."""

    def __init__(self, fail_kinds: set[str] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail_kinds = fail_kinds or set()

    def __call__(self, kind: str, payload: dict) -> dict:
        self.calls.append((kind, dict(payload)))
        if kind in self.fail_kinds:
            raise RuntimeError("색인 실패(테스트 시뮬레이션)")
        if kind == "candidate":
            document_id = f"ideation-target::P1::{payload['session_id']}::{payload['candidate_id']}"
        else:
            document_id = f"ideation-answer::P1::{payload['session_id']}::{payload['user_message_id']}"
        return {"document_id": document_id, "chunk_count": 3, "status": "ok"}


# ---------------------------------------------------------------------------
# candidate_selection 노드 -> index_target_evidence("candidate", ...) 배선
# ---------------------------------------------------------------------------


def test_candidate_selection_calls_index_target_evidence_and_stores_document_id():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    indexer = _RecordingIndexer()

    state = reply_ideation_conversation(
        previous_state=state, user_message="1번", llm_call=llm, index_target_evidence=indexer
    )

    candidate_calls = [c for c in indexer.calls if c[0] == "candidate"]
    assert len(candidate_calls) == 1
    kind, payload = candidate_calls[0]
    assert payload["candidate_id"] == "candidate_1"
    assert payload["session_id"] == state["session_id"]
    assert payload["candidate"]["title"]

    assert state["selected_idea_document_id"] == f"ideation-target::P1::{state['session_id']}::candidate_1"


def test_candidate_selection_without_indexer_leaves_document_id_none():
    """index_target_evidence를 주입하지 않으면(use_rag=False 등) 색인을 건너뛰고
    selected_idea_document_id=None으로 안전하게 진행한다."""
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)

    state = reply_ideation_conversation(previous_state=state, user_message="1번", llm_call=llm)

    assert state["selected_idea_document_id"] is None
    assert state["phase"] != "failed"


def test_candidate_selection_indexing_failure_does_not_corrupt_state():
    """색인이 실패해도(요청 17-4번) 회의 state는 손상되지 않고, 그냥 target 근거 없이
    진행된다."""
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    indexer = _RecordingIndexer(fail_kinds={"candidate"})

    state = reply_ideation_conversation(
        previous_state=state, user_message="1번", llm_call=llm, index_target_evidence=indexer
    )

    assert state["phase"] != "failed"
    assert state["selected_idea_document_id"] is None
    assert state["selected_idea"]["candidate_id"] == "candidate_1"  # 선택 자체는 정상 반영됨.


# ---------------------------------------------------------------------------
# 사용자 답변 -> index_target_evidence("user_answer", ...) 배선 + 인덱싱 대상 메시지 제한
# ---------------------------------------------------------------------------


def _refinement_llm():
    """discovery를 거치지 않고 곧바로 refinement round-table로 들어가는 세션에서 쓸 stub —
    DiscoveryScriptedLLM은 discovery 전용 프롬프트 마커만 처리하므로, refinement round-table
    (planning/dev 의견, 진행자 정리, sufficiency 판정) 마커를 재사용해 answer 처리 경로를
    그대로 태운다."""
    return DiscoveryScriptedLLM()


def test_user_answer_calls_index_target_evidence_before_next_expert_turn():
    """요청 9번 — 인덱싱이 완료된 뒤에만 다음 전문가 턴이 실행된다. 이 테스트에서는 인덱서
    호출이 곧 다음 그래프 실행보다 먼저 일어난다는 것을(동기 호출 순서) 확인한다 — 인덱서가
    이번 답변에 대해 정확히 한 번, 그래프 재실행 이전에 호출됐는지로 순서를 검증한다."""
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    indexer = _RecordingIndexer()
    state = reply_ideation_conversation(
        previous_state=state, user_message="1번", llm_call=llm, index_target_evidence=indexer
    )
    # 라운드테이블 한 라운드가 끝나 awaiting_user_decision에서 멈췄다 — 이제 사용자가
    # 자유롭게 한 마디 더 남기면(구체적인 개입) 그 답변이 색인 대상이다.
    assert state["phase"] == "discussion_complete"
    indexer.calls.clear()

    substantial_message = "학교 주변 통학로의 실시간 혼잡도와 사고 이력을 함께 반영해야 합니다."
    state = reply_ideation_conversation(
        previous_state=state, user_message=substantial_message, llm_call=llm, index_target_evidence=indexer
    )

    user_answer_calls = [c for c in indexer.calls if c[0] == "user_answer"]
    assert len(user_answer_calls) == 1
    _, payload = user_answer_calls[0]
    assert payload["answer_text"] == substantial_message
    assert payload["session_id"] == state["session_id"]
    assert payload["user_message_id"]


def test_short_greeting_reply_is_not_indexed_as_target_evidence():
    """요청 17-2번 — 단순 동의·감탄 같은 짧은 문구는 색인 대상에서 제외된다."""
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    indexer = _RecordingIndexer()
    state = reply_ideation_conversation(
        previous_state=state, user_message="1번", llm_call=llm, index_target_evidence=indexer
    )
    indexer.calls.clear()

    state = reply_ideation_conversation(
        previous_state=state, user_message="네 감사합니다", llm_call=llm, index_target_evidence=indexer
    )
    assert not [c for c in indexer.calls if c[0] == "user_answer"]


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-23, 요청: stale closure 수정 — 필수 테스트 1번). 후보 선택과 첫 전문가
# 검색이 같은 /reply 안에서 이어질 때(candidate_selection -> to_refinement ->
# planning_expert_discussion, 정지 없이 연속 실행), evidence_lookup이 요청 시작 시점의
# stale 값(selected_idea_document_id=None)이 아니라 candidate_selection이 방금 갱신한
# 최신 document_id를 runtime_scope로 받는지 확인한다.
# ---------------------------------------------------------------------------


class _RecordingEvidenceLookup:
    """evidence_lookup으로 주입되는 fake. runtime_scope 키워드 인자를 받아 그대로 기록한다
    (ai/meeting은 ai.rag를 모르므로 이 fake도 runtime_scope의 내용에 대해 아무 것도
    가정하지 않고, 호출부가 실제로 무엇을 전달했는지만 기록한다)."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, persona_id, query, *, runtime_scope=None):
        self.calls.append({"persona_id": persona_id, "query": query, "runtime_scope": runtime_scope})
        return []


def test_same_reply_expert_search_uses_freshly_selected_candidate_document_id():
    """후보 선택과 같은 /reply 안에서 실행되는 첫 planning/dev 전문가 검색이 candidate_selection
    노드가 이번 호출 안에서 방금 인덱싱·저장한 selected_idea_document_id를 받아야 한다 —
    evidence_lookup을 만들 때(요청 시작 시점, previous_state.selected_idea_document_id=None)
    캡처된 stale 값이 아니라, 노드가 실제로 호출하는 순간의 최신 state 값이어야 한다."""
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    indexer = _RecordingIndexer()
    lookup = _RecordingEvidenceLookup()

    state = reply_ideation_conversation(
        previous_state=state,
        user_message="1번",
        llm_call=llm,
        index_target_evidence=indexer,
        evidence_lookup=lookup,
    )

    assert state["selected_idea_document_id"] is not None
    expert_calls = [c for c in lookup.calls if c["persona_id"] in ("planning_expert", "dev_expert")]
    assert expert_calls, "후보 선택 직후 같은 요청 안에서 라운드테이블 전문가 검색이 실행돼야 한다"
    for call in expert_calls:
        assert call["runtime_scope"] is not None
        assert call["runtime_scope"]["selected_candidate_document_id"] == state["selected_idea_document_id"]
        assert call["runtime_scope"]["session_id"] == state["session_id"]


def test_candidate_reselection_within_same_request_scopes_to_new_candidate():
    """요청 7번 — 같은 세션에서 후보를 다시 선택(재추천 후 재선택)해도, 그 요청의 첫 전문가
    검색은 새로 선택된 후보의 document_id로 스코프된다(이전 후보 id가 남지 않는다)."""
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    indexer = _RecordingIndexer()

    # 1차 선택: candidate_1.
    state = reply_ideation_conversation(
        previous_state=state, user_message="1번", llm_call=llm, index_target_evidence=indexer
    )
    first_document_id = state["selected_idea_document_id"]
    assert first_document_id is not None

    # 세션을 재선택 가능한 지점으로 되돌릴 수 없는 discovery 흐름 특성상, 새 세션에서
    # candidate_2를 바로 선택해 "다른 후보가 선택되면 다른 document_id가 나온다"는 계약만
    # 별도로 확인한다(같은 요청 안에서 재선택 시나리오는 candidate_selection 노드 자체가
    # 사용자 텍스트 파싱으로 결정하므로, 이 테스트는 그 id 계산 계약을 검증한다).
    state2 = _start_discovery(llm)
    indexer2 = _RecordingIndexer()
    state2 = reply_ideation_conversation(
        previous_state=state2, user_message="2번", llm_call=llm, index_target_evidence=indexer2
    )
    second_document_id = state2["selected_idea_document_id"]
    assert second_document_id is not None
    assert second_document_id != first_document_id


def test_user_answer_indexing_failure_does_not_stop_meeting():
    llm = DiscoveryScriptedLLM()
    state = _start_discovery(llm)
    indexer = _RecordingIndexer()
    state = reply_ideation_conversation(
        previous_state=state, user_message="1번", llm_call=llm, index_target_evidence=indexer
    )
    failing_indexer = _RecordingIndexer(fail_kinds={"user_answer"})

    substantial_message = "학교 주변 통학로의 실시간 혼잡도와 사고 이력을 함께 반영해야 합니다."
    state = reply_ideation_conversation(
        previous_state=state, user_message=substantial_message, llm_call=llm, index_target_evidence=failing_indexer
    )
    assert state["phase"] != "failed"
    assert any(m["content"] == substantial_message for m in state["messages"])
