from __future__ import annotations

import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MEETING_DIR.parents[1]
sys.path.insert(0, str(MEETING_DIR))
sys.path.insert(0, str(REPO_ROOT))

from ai.rag.evidence_linking.claim_grounding import ground_claims  # noqa: E402
from graph.ideation_conv_nodes import (  # noqa: E402
    _discussion_retry_note,
    _repair_evaluative_expert_judgment_claim,
    _safe_call_structured_json,
    make_conv_discussion_node,
)
from graph.ideation_conv_state import initial_conv_state  # noqa: E402


class _AlwaysInvalidStreamingLLM:
    def __init__(self):
        self.calls = 0
        self.discards: list[tuple[str, str, bool]] = []

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return '{"spoken_text": "필수 필드가 없는 임시 초안"}'

    def discard_streamed_prompt(self, prompt: str, reason: str, will_retry: bool = True) -> None:
        self.discards.append((prompt, reason, will_retry))


def test_each_failed_attempt_is_traced_and_streamed_draft_is_discarded(monkeypatch):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "graph.ideation_conv_nodes.trace_event",
        lambda name, **fields: events.append((name, fields)),
    )
    llm = _AlwaysInvalidStreamingLLM()

    raw, ok, attempts = _safe_call_structured_json(
        llm,
        "same prompt",
        lambda value: "missing_or_empty_field:judgment" if not value.get("judgment") else None,
        "discussion__planning_expert",
    )

    failures = [fields for name, fields in events if name == "IDEATION_STRUCTURED_RESPONSE_VALIDATION_FAILED"]
    assert raw is None
    assert ok is False
    assert attempts == 2
    assert [item["attempt"] for item in failures] == [1, 2]
    assert [item["will_retry"] for item in failures] == [True, False]
    assert [reason for _, reason, _will_retry in llm.discards] == [
        "missing_or_empty_field:judgment",
        "missing_or_empty_field:judgment",
    ]
    assert [will_retry for _, _reason, will_retry in llm.discards] == [True, False]


def test_discussion_retry_receives_the_actual_validation_reason():
    llm = _AlwaysInvalidStreamingLLM()

    _safe_call_structured_json(
        llm,
        "original prompt",
        lambda _value: "spoken_text_issue_drift",
        "discussion__planning_expert",
        retry_note_for=_discussion_retry_note,
    )

    assert llm.calls == 2
    assert llm.discards[0][0] == "original prompt"
    assert "실패 코드: spoken_text_issue_drift" in llm.discards[1][0]
    assert "다음 쟁점의 내용은 제외" in llm.discards[1][0]


def test_evaluative_conclusion_gets_a_separate_expert_judgment_claim():
    raw = {
        "spoken_text": "평가기준에는 문제의 구체성을 요구하지만 현재 설명은 부족합니다.",
        "judgment": "현재 문제 설명은 부족합니다.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "평가기준은 문제의 구체성을 평가한다.",
                "claim_type": "document_fact",
                "evidence_refs": ["E1"],
            }
        ],
    }

    _repair_evaluative_expert_judgment_claim(raw, {"E1": "document_fact"})

    assert [claim["claim_type"] for claim in raw["claims"]] == [
        "document_fact",
        "expert_judgment",
    ]
    assert raw["claims"][1]["text"] == "현재 문제 설명은 부족합니다."
    assert raw["claims"][1]["evidence_refs"] == []


def test_recommendation_language_is_recorded_as_expert_judgment():
    raw = {
        "spoken_text": (
            "도시 문제의 설정이 구체적인지는 평가 기준입니다. "
            "효과를 수치적으로 검증하기 위한 데이터 확보 방안이 필요합니다."
        ),
        "judgment": "효과를 수치적으로 검증하기 위한 데이터 확보 방안이 필요합니다.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "도시 문제의 설정이 구체적인가?",
                "claim_type": "document_fact",
                "evidence_refs": ["E1"],
            }
        ],
    }

    _repair_evaluative_expert_judgment_claim(raw, {"E1": "document_fact"})

    assert [claim["claim_type"] for claim in raw["claims"]] == [
        "document_fact",
        "expert_judgment",
    ]
    assert raw["claims"][1]["text"] == "효과를 수치적으로 검증하기 위한 데이터 확보 방안이 필요합니다."
    assert raw["claims"][1]["evidence_refs"] == []


def test_discussion_uses_safe_expert_judgment_instead_of_failing_session(monkeypatch):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "graph.ideation_conv_nodes.trace_event",
        lambda name, **fields: events.append((name, fields)),
    )
    state = initial_conv_state(
        "SAFE-FALLBACK-1",
        {"competition_name": "테스트 공모전", "notice_document": "평가 기준"},
        {"description": "에너지 사용을 최적화하는 AI 서비스"},
        max_rounds=1,
    )
    llm = _AlwaysInvalidStreamingLLM()
    node = make_conv_discussion_node(
        "planning_expert",
        llm,
        ground_claims=lambda persona_id, claims, evidence: ground_claims(claims, evidence),
    )

    update = node(state)

    assert update.get("phase") != "failed"
    assert update.get("failed_node") is None
    assert update["previous_speaker"] == "planning_expert"
    assert update["expert_turn_count"] == 1
    message = update["messages"][0]
    assert "전문가 판단으로 진행" in message["content"]
    assert "문제 정의를 구체화" in message["content"]
    # 두 번 모두 구조화 검증에 실패한 서버 fallback은 검증 가능한 새 논점이 아니다.
    # 같은 쟁점을 상대 위원에게 다시 넘겨 반복시키지 않고 진행자가 잠정 정리한다.
    assert message["structured"]["recommended_next_speaker"] == "ideation_facilitator"
    assert message["structured"]["needs_counterpart_response"] is False
    assert message["structured"]["repetition_detected"] is True
    assert message["structured"]["issue_resolved"] is False
    assert message["structured"]["safe_fallback"] is True
    assert message["structured"]["safe_fallback_reason"] == "structured_response_validation_failed_twice"
    assert message["claims"][0]["claim_type"] == "expert_judgment"
    assert message["claims"][0]["evidence_refs"] == []
    assert not any(claim["claim_type"] == "document_fact" for claim in message["claims"])
    assert any(name == "IDEATION_STRUCTURED_RESPONSE_SAFE_FALLBACK" for name, _ in events)
