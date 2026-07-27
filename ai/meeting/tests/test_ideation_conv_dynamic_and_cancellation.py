# 작성자: 용준/Claude(2026-07-22)
# 목적: "동적 전문가 회의로 개편" 요청의 핵심 회귀 테스트 — (1) 발언 순서/횟수가 그래프
#       구조가 아니라 쟁점·반론 여부로 결정되는지(고정 "기획 1회 → 개발 1회 → 진행자"
#       패턴이 아닌지), (2) IdeationCancelled가 재시도 없이 그래프 실행까지 전파되고 취소
#       시점까지 완료된 발언만 partial_state에 남는지를 검증한다.
# import: 표준 라이브러리 json/sys/pathlib, pytest; ai/meeting/graph 패키지.

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph import (  # noqa: E402
    IdeationCancelled,
    MAX_EXPERT_TURNS_PER_ISSUE,
    start_ideation_conversation,
)

NOTICE_AND_CRITERIA = {
    "competition_name": "IT 공공서비스 공모전",
    "notice_document": "실현가능성, 공공성을 평가한다.",
}
USER_IDEA = {
    "description": "공공기관의 정책·지원사업 문서를 RAG로 검색하고, 사용자 상황에 맞는 지원사업을 추천하는 AI 서비스를 만들고 싶습니다."
}


def _persona(prompt: str) -> str:
    if "당신은 AI Review Board의 기획 전문가입니다" in prompt:
        return "planning_expert"
    if "당신은 AI Review Board의 개발 전문가입니다" in prompt:
        return "dev_expert"
    return "ideation_facilitator"


def _discussion_payload(speaker: str, *, active_issue_id="target_user", issue_resolved=False,
                         recommended_next_speaker="dev_expert", needs_counterpart_response=True,
                         new_information="새로 확인된 내용", spoken_text=None) -> dict:
    return {
        "stance": "반박",
        "spoken_text": spoken_text or f"[{speaker}] 발화",
        "judgment": "판단",
        "reason": "근거",
        "suggestion": "제안",
        "interim_conclusion": "임시 결론",
        "responding_to": "상대 발언" if recommended_next_speaker != speaker else None,
        "agreement": "",
        "concern": "우려",
        "confirmed": [],
        "unconfirmed": [],
        "referenced_message_ids": [],
        "evidence": [],
        "next_action": None,
        "active_issue_id": active_issue_id,
        "active_issue_title": "목표 사용자",
        "new_information": [new_information],
        "proposal": "제안",
        "changed_position": False,
        "needs_counterpart_response": needs_counterpart_response,
        "recommended_next_speaker": recommended_next_speaker,
        "issue_resolved": issue_resolved,
        "needs_user_input": False,
        "user_question": None,
    }


def _spoken_text_for_prompt(prompt: str, speaker: str) -> str | None:
    if "사용자 직접 질문 최우선 규칙" not in prompt:
        return None
    if speaker == "planning_expert":
        return (
            "대학생 예비 창업자와 목표 사용자 관점에서 개인정보, 기술 범위, 유지보수 질문을 "
            "운영 비용과 고객 가치 기준으로 직접 검토합니다."
        )
    return (
        "개인정보와 기술 범위, 유지보수 질문을 센서 장애 감지, 예비 부품, 자동 모니터링과 "
        "배포 구조 관점에서 직접 검토합니다."
    )


def _facilitator_payload() -> dict:
    return {
        "agreements": [],
        "disagreements": ["목표 사용자 범위"],
        "facilitator_summary": "목표 사용자 범위를 두고 두 위원이 계속 이견을 보였습니다.",
        "spoken_text": "목표 사용자 범위에 대해 계속 논의가 이어졌습니다.",
        "needs_user_decision": False,
        "user_question": None,
    }


class _EndlessDisagreementLLM:
    """기획/개발이 같은 쟁점(target_user)에 대해 서로 계속 반박하며 상대를 지목하는 stub —
    "기획 1회 → 개발 1회 → 진행자"로 고정되지 않고, per-issue 발언 캡
    (MAX_EXPERT_TURNS_PER_ISSUE)에 도달할 때까지 실제로 여러 차례 주고받는지 검증한다."""

    def __call__(self, prompt: str) -> str:
        if "[판정 규칙]" in prompt:
            return json.dumps({"is_sufficient": True, "reason": "충분", "follow_up_question": None})
        if "[의견 규칙]" in prompt:
            speaker = _persona(prompt)
            counterpart = "dev_expert" if speaker == "planning_expert" else "planning_expert"
            return json.dumps(
                _discussion_payload(
                    speaker,
                    recommended_next_speaker=counterpart,
                    issue_resolved=False,
                    spoken_text=_spoken_text_for_prompt(prompt, speaker),
                ),
                ensure_ascii=False,
            )
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(_facilitator_payload(), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")


def test_dynamic_routing_keeps_bouncing_between_experts_until_issue_cap():
    """요청 핵심 — "기획 1회 → 개발 1회 → 진행자 정리"로 무조건 고정되지 않고, 같은 쟁점에
    대해 반론이 계속되면 캡(MAX_EXPERT_TURNS_PER_ISSUE)에 도달할 때까지 전문가 발언이
    이어져야 한다."""
    llm = _EndlessDisagreementLLM()
    state = start_ideation_conversation(
        session_id="DYNAMIC-1",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        max_rounds=1,
    )

    expert_messages = [m for m in state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
    # 남은 쟁점을 한 번 이월하더라도 각 쟁점 안에서는 캡에 도달할 때까지 실제로
    # 여러 번 주고받아야 한다(2회 고정이 아니라 쟁점별 캡만큼).
    issue_message_counts: dict[str, int] = {}
    for message in expert_messages:
        issue_id = message["structured"]["active_issue_id"]
        issue_message_counts[issue_id] = issue_message_counts.get(issue_id, 0) + 1
    assert len(issue_message_counts) == 2
    assert all(count == MAX_EXPERT_TURNS_PER_ISSUE for count in issue_message_counts.values())
    # 진행자가 "매 발언 쌍마다" 등장하지 않고, 각 쟁점의 캡에 도달한 뒤 한 번씩 정리한다.
    facilitator_messages = [m for m in state["messages"] if m["speaker_id"] == "ideation_facilitator"]
    assert len(facilitator_messages) == 3  # 오프닝 안건 제시(1) + 쟁점별 캡 도달 정리(2).
    assert state.get("stop_reason") == "max_turns_reached"
    # 발언자가 planning/dev를 번갈아 오갔는지(고정 순서가 아니라 상대를 지목하는 동적 교대).
    speakers = [m["speaker_id"] for m in expert_messages]
    assert speakers[0] == "planning_expert"
    assert speakers[1] == "dev_expert"
    assert any(speakers[i] != speakers[i - 1] for i in range(1, len(speakers)))
    # 각 쟁점의 첫 발언은 새 논점의 시작이므로 상호 검토 대상이 없고, 그 뒤 발언부터
    # 직전 상대 전문가 메시지를 정확히 참조한다.
    message_by_id = {message["message_id"]: message for message in state["messages"]}
    seen_issue_ids: set[str] = set()
    for message in expert_messages:
        issue_id = message["structured"]["active_issue_id"]
        target_id = message["structured"]["responding_to_message_id"]
        target_speaker = message["structured"]["responding_to_speaker_id"]
        if issue_id not in seen_issue_ids:
            seen_issue_ids.add(issue_id)
            assert target_id is None
            continue
        assert target_id in message_by_id
        assert message_by_id[target_id]["speaker_id"] == target_speaker
        assert target_speaker != message["speaker_id"]


class _SingleIssueResolvedLLM:
    """쟁점이 1회 교환 만에 해결되는 stub — 발언 캡과 무관하게, 해결되면 그 즉시 진행자가
    정리해야 한다(불필요하게 캡까지 발언을 늘리지 않는다)."""

    def __call__(self, prompt: str) -> str:
        if "[의견 규칙]" in prompt:
            speaker = _persona(prompt)
            if speaker == "planning_expert":
                return json.dumps(
                    _discussion_payload(speaker, recommended_next_speaker="dev_expert", issue_resolved=False),
                    ensure_ascii=False,
                )
            return json.dumps(
                _discussion_payload(
                    speaker, recommended_next_speaker="ideation_facilitator", issue_resolved=True,
                    needs_counterpart_response=False,
                ),
                ensure_ascii=False,
            )
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(_facilitator_payload(), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")


def test_facilitator_does_not_wait_for_turn_cap_when_issue_resolves_quickly():
    """쟁점이 2회 만에 해결되면(개발 위원이 issue_resolved=true) 진행자가 즉시 개입해야
    한다 — 매번 캡까지 채우지 않는다."""
    llm = _SingleIssueResolvedLLM()
    state = start_ideation_conversation(
        session_id="DYNAMIC-2",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        max_rounds=1,
    )
    expert_messages = [m for m in state["messages"] if m["speaker_id"] in ("planning_expert", "dev_expert")]
    assert len(expert_messages) == 2
    assert state.get("stop_reason") == "consensus_reached"
    assert state["open_issues"] == []
    assert state["resolved_issues"]


class _PlanningTriesToSkipDeveloperLLM:
    """기획 위원이 곧바로 진행자를 추천해도 개발 검토가 보장되는지 재현한다."""

    def __call__(self, prompt: str) -> str:
        if "[의견 규칙]" in prompt:
            speaker = _persona(prompt)
            return json.dumps(
                _discussion_payload(
                    speaker,
                    recommended_next_speaker="ideation_facilitator",
                    needs_counterpart_response=False,
                    issue_resolved=speaker == "dev_expert",
                ),
                ensure_ascii=False,
            )
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(_facilitator_payload(), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")


def test_each_round_requires_developer_review_when_planning_recommends_facilitator():
    state = start_ideation_conversation(
        session_id="DYNAMIC-PLANNING-SKIP",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=_PlanningTriesToSkipDeveloperLLM(),
        max_rounds=1,
    )
    expert_speakers = [
        message["speaker_id"]
        for message in state["messages"]
        if message["speaker_id"] in ("planning_expert", "dev_expert")
    ]
    assert expert_speakers[:2] == ["planning_expert", "dev_expert"]


class _CancelAfterNCallsLLM:
    """N번째 호출부터 IdeationCancelled를 던지는 stub — 실제 스트리밍 llm_call이 cancel_event
    감지 시 이 예외를 던지는 것과 동일한 상황을 재현한다."""

    def __init__(self, cancel_after: int):
        self.cancel_after = cancel_after
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if self.calls > self.cancel_after:
            raise IdeationCancelled("SESSION-X", "REQ-1")
        if "[의견 규칙]" in prompt:
            speaker = _persona(prompt)
            counterpart = "dev_expert" if speaker == "planning_expert" else "planning_expert"
            return json.dumps(
                _discussion_payload(speaker, recommended_next_speaker=counterpart, issue_resolved=False),
                ensure_ascii=False,
            )
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")


def test_cancellation_propagates_without_retry_and_does_not_fail_phase():
    """요청 13번 — 취소는 재시도되지 않고(같은 프롬프트로 llm_call이 다시 불리지 않고) 그대로
    전파돼야 하며, phase="failed"로 이어지면 안 된다(일반 오류와 다른 정책)."""
    llm = _CancelAfterNCallsLLM(cancel_after=1)  # 기획 위원의 최초 발언 1회만 허용.
    with pytest.raises(IdeationCancelled) as excinfo:
        start_ideation_conversation(
            session_id="CANCEL-1",
            notice_and_criteria=NOTICE_AND_CRITERIA,
            user_idea=USER_IDEA,
            llm_call=llm,
            max_rounds=1,
        )
    # 재시도가 있었다면 calls가 3(최초 1 + 재시도 1 + 취소 유발 1) 이상이었을 것이다 — 여기서는
    # 정확히 2회(성공 1 + 취소 유발 1)만 호출되어야 한다(재시도 없음).
    assert llm.calls == 2
    exc = excinfo.value
    # 취소 시점까지 완료된 발언(진행자 오프닝 + 기획 위원의 최초 의견)은 partial_state에
    # 남아 있어야 한다 — "완료된 전문가 주장은 유지, 미완성만 취소"(요청 14번).
    assert exc.partial_state is not None
    speakers = [m["speaker_id"] for m in exc.partial_state["messages"]]
    assert speakers == ["ideation_facilitator", "planning_expert"]
    assert exc.partial_state["phase"] != "failed"


def test_cancellation_before_any_expert_turn_completes_has_no_expert_messages():
    """세션 시작 직후(첫 전문가 발언이 완료되기도 전) 취소되면, partial_state가 있더라도
    (진행자의 LLM-미호출 오프닝 메시지는 그래프 진입 전에 이미 있었으므로 남아있을 수
    있다) 전문가(planning_expert/dev_expert) 발언은 하나도 포함되면 안 된다 — 되돌아갈
    "완료된 전문가 주장"이 없기 때문이다."""
    llm = _CancelAfterNCallsLLM(cancel_after=0)
    with pytest.raises(IdeationCancelled) as excinfo:
        start_ideation_conversation(
            session_id="CANCEL-2",
            notice_and_criteria=NOTICE_AND_CRITERIA,
            user_idea=USER_IDEA,
            llm_call=llm,
            max_rounds=1,
        )
    partial = excinfo.value.partial_state
    if partial is not None:
        expert_speakers = {m["speaker_id"] for m in partial["messages"]} & {"planning_expert", "dev_expert"}
        assert not expert_speakers


# 용준/Claude(2026-07-22, 요청: "잠시만" 취소 중 phase 오염 수정) — 아래 테스트들은 실제
# 브라우저에서 재현된 회귀 시나리오를 직접 재현한다: discussion_facilitator가 다음 라운드로
# 자동 진행하기로 결정한 "직후"(그래프 내부적으로는 곧바로 planning_expert_discussion으로
# 이어지는 그 경계)에 취소되면, 예전에는 facilitator가 내부 라우팅 신호로만 쓰려던
# phase="planning_question"이 그대로 partial_state에 찍혀 세션에 저장됐다 — 이후 재개 시점에
# 그 phase를 재개 불가능한 값으로 보고 거부했다(원 보고 그대로).
def _make_cancel_at_round_transition_llm():
    """_EndlessDisagreementLLM과 동일하게 같은 쟁점(target_user)에 대해 계속 반박하며
    라운드 1을 쟁점 발언 캡(MAX_EXPERT_TURNS_PER_ISSUE=6)까지 채운 뒤, facilitator가
    "다음 라운드로 자동 진행"(continue_round)을 결정하고 나서 그 다음 호출(라운드 2의
    첫 기획 위원 발언 생성 시도)부터 IdeationCancelled를 던진다 — 실제 스트리밍
    llm_call이 cancel_event를 감지해 던지는 시점과 동일하다(다음 노드의 llm_call
    진입 시점)."""
    calls = {"n": 0}

    def llm(prompt: str) -> str:
        calls["n"] += 1
        # 라운드 1: 기획/개발이 같은 쟁점으로 6회 주고받는다(호출 1~6) + facilitator 정리
        # 1회(호출 7) = 총 7회는 정상 응답. 8번째 호출(라운드 2의 첫 발언 시도)부터 취소.
        if calls["n"] > 7:
            raise IdeationCancelled("SESSION-ROUND-TRANSITION", "REQ-ROUND-TRANSITION")
        if "[의견 규칙]" in prompt:
            speaker = _persona(prompt)
            counterpart = "dev_expert" if speaker == "planning_expert" else "planning_expert"
            return json.dumps(
                _discussion_payload(speaker, recommended_next_speaker=counterpart, issue_resolved=False),
                ensure_ascii=False,
            )
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(_facilitator_payload(), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")

    llm.calls = calls
    return llm


def _cancelled_at_round_transition():
    """max_rounds=2를 줘서(라운드 1이 쟁점 캡으로 끝나도 round(1) < max_rounds(2)이므로)
    facilitator가 await_user_decision이 아니라 continue_round를 결정하게 만든다 — 바로 그
    직후(라운드 2 진입 중)에 취소가 일어나는 상황을 재현한다."""
    llm = _make_cancel_at_round_transition_llm()
    with pytest.raises(IdeationCancelled) as excinfo:
        start_ideation_conversation(
            session_id="ROUND-TRANSITION-1",
            notice_and_criteria=NOTICE_AND_CRITERIA,
            user_idea=USER_IDEA,
            llm_call=llm,
            max_rounds=2,
        )
    return excinfo.value


def test_cancellation_at_facilitator_round_transition_normalizes_phase():
    """핵심 회귀 검증 — facilitator가 continue_round를 정하고 다음 라운드 노드로 넘어가는
    도중 취소돼도, partial_state의 phase는 그래프 밖에서 의미 없는 내부 신호값
    ("planning_question")이 아니라 항상 재개 가능한 canonical 값("expert_discussion")이어야
    하고, "failed"로도 이어지면 안 된다(요청 3·6·7·11번)."""
    exc = _cancelled_at_round_transition()
    partial = exc.partial_state
    assert partial is not None
    assert partial["phase"] == "expert_discussion"
    assert partial["phase"] != "planning_question"
    assert partial["phase"] != "failed"
    # facilitator가 이미 라운드를 2로 넘겨 놓았어야 한다(취소는 그 다음 노드 실행 중
    # 일어났으므로, facilitator 자신의 update는 이미 완료돼 있다).
    assert partial["round"] == 2
    assert partial["expert_turn_count"] == 0
    # facilitator 정리 메시지까지는 남아 있어야 한다(완료된 발언은 유지 — 요청 14번 원칙).
    assert partial["messages"][-1]["speaker_id"] == "ideation_facilitator"



if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
