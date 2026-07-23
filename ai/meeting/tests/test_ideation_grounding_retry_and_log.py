# 작성자: 용준/Claude(2026-07-23, 요청: grounding 재시도 프롬프트의 ref 계약 오류 수정 +
#         IDEATION_EVIDENCE_LINKED 로그 매핑 수정)
# 목적: (1) _ground_and_finalize_claims의 재시도 안내 문구가 chunk_id가 아니라 ref만
#       요구하는지, 재시도로 정상 회복되는지, (2) IDEATION_EVIDENCE_LINKED 로그가 claim이
#       인용한 ref와 실제로 연결된 chunk_id를 올바르게 짝지어 기록하는지 확인한다.

import json
import logging
import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
REPO_ROOT = MEETING_DIR.parents[1]

sys.path.insert(0, str(MEETING_DIR))
sys.path.insert(0, str(REPO_ROOT))

from graph.ideation_conv_nodes import _ground_and_finalize_claims  # noqa: E402
from graph.ideation_trace import configure_ideation_trace  # noqa: E402

from ai.rag.evidence_linking.claim_grounding import ground_claims as _ground_claims_impl  # noqa: E402

EVIDENCE = [
    {
        "ref": "E1",
        "chunk_id": "chk_31da000000000000",
        "document_id": "DOC-1",
        "document_name": "WSCE2026 공고문",
        "section": "평가 기준",
        "text": "본 사업은 실현 가능성과 경제성을 중점적으로 평가한다.",
    }
]


def _ground_claims_fn(persona_id, claims, retrieved):
    return _ground_claims_impl(claims, retrieved)


def _validate_always_ok(raw: dict) -> str | None:
    return None


# ---------------------------------------------------------------------------
# 1. grounding 재시도 프롬프트의 ref 계약
# ---------------------------------------------------------------------------


def test_retry_note_requires_ref_not_chunk_id_and_recovers_on_second_response():
    first_raw = {
        "spoken_text": "초안",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
                "claim_type": "document_fact",
                # 존재하지 않는 ref를 인용 -> hard grounding failure -> 재시도 유발.
                "evidence_refs": ["E-does-not-exist"],
            }
        ],
    }
    second_response = json.dumps(
        {
            "spoken_text": "재시도 응답",
            "claims": [
                {
                    "claim_id": "claim_1",
                    "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
                    "claim_type": "document_fact",
                    "evidence_refs": ["E1"],
                }
            ],
        },
        ensure_ascii=False,
    )

    captured_prompts: list[str] = []

    def llm_call(prompt: str) -> str:
        captured_prompts.append(prompt)
        return second_response

    raw, grounding, used = _ground_and_finalize_claims(
        persona_id="planning_expert",
        raw=first_raw,
        retrieved=EVIDENCE,
        prompt="원본 프롬프트",
        llm_call=llm_call,
        validate=_validate_always_ok,
        used=1,
        ground_claims_fn=_ground_claims_fn,
    )

    assert len(captured_prompts) == 1
    retry_prompt = captured_prompts[0]
    # 재시도 안내는 evidence_refs에 넣을 값으로 ref만 지시해야 한다 — 예전 버그 문구처럼
    # "실제 chunk_id만 evidence_refs에 넣고" 같은 지시를 다시 내려서는 안 된다(chunk_id를
    # "쓰지 말라"는 부정 설명은 허용된다).
    assert "chunk_id만 evidence_refs" not in retry_prompt
    assert "실제 chunk_id와 연결" not in retry_prompt
    assert 'evidence_refs에는' in retry_prompt and '"ref"' in retry_prompt

    assert used == 2
    assert grounding["evidence_status"] == "grounded"
    # 서버 쪽 linked_evidence_refs 계약은 항상 실제 chunk_id다(ref가 아니다).
    assert grounding["linked_evidence_refs"] == ["chk_31da000000000000"]
    assert raw["spoken_text"] == "재시도 응답"


def test_retry_note_still_present_when_second_response_also_fails():
    """재시도 후에도 실패하면(존재하지 않는 ref를 계속 인용) fallback 문구로 교체되고,
    grounding은 ungrounded로 남는다 — 재시도는 정확히 한 번만 일어난다."""
    first_raw = {
        "spoken_text": "초안",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "이 사업은 6개월 안에 구축해야 한다.",
                "claim_type": "document_fact",
                "evidence_refs": ["E-does-not-exist"],
            }
        ],
    }
    still_bad_response = json.dumps(
        {
            "spoken_text": "여전히 근거 없는 확정 발언",
            "claims": [
                {
                    "claim_id": "claim_1",
                    "text": "이 사업은 6개월 안에 구축해야 한다.",
                    "claim_type": "document_fact",
                    "evidence_refs": ["E-still-does-not-exist"],
                }
            ],
        },
        ensure_ascii=False,
    )
    call_count = {"n": 0}

    def llm_call(prompt: str) -> str:
        call_count["n"] += 1
        return still_bad_response

    raw, grounding, used = _ground_and_finalize_claims(
        persona_id="planning_expert",
        raw=first_raw,
        retrieved=EVIDENCE,
        prompt="원본 프롬프트",
        llm_call=llm_call,
        validate=_validate_always_ok,
        used=1,
        ground_claims_fn=_ground_claims_fn,
    )

    assert call_count["n"] == 1  # 재시도는 딱 한 번만 일어난다.
    assert used == 2
    assert grounding["evidence_status"] == "ungrounded"
    assert "확인하기 어렵습니다" in raw["spoken_text"]


# ---------------------------------------------------------------------------
# 1-1. 용준/Claude(2026-07-23, 요청: 스트리밍 UX 버그 수정) — grounding 재시도가 이전
# 스트림 말풍선을 명시적으로 discard하는지. 재시도 프롬프트는 원본과 문자열이 달라
# (prompt + retry_note) llm_call 내부의 "동일 prompt 재호출" 감지로는 자동으로 잡히지
# 않으므로, _ground_and_finalize_claims가 재시도 llm_call 전에 반드시
# discard_streamed_prompt(원본 prompt, ...)를 먼저 불러야 프런트에서 이전 초안과 재시도
# 초안이 동시에 남는 문제가 생기지 않는다.
# ---------------------------------------------------------------------------


class _RecordingStreamingLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []
        self.discards: list[tuple[str, str, bool]] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response

    def discard_streamed_prompt(self, prompt: str, reason: str, will_retry: bool = True) -> None:
        self.discards.append((prompt, reason, will_retry))


def test_grounding_retry_discards_previous_stream_before_retry_call():
    first_raw = {
        "spoken_text": "초안",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
                "claim_type": "document_fact",
                "evidence_refs": ["E-does-not-exist"],
            }
        ],
    }
    second_response = json.dumps(
        {
            "spoken_text": "재시도 응답",
            "claims": [
                {
                    "claim_id": "claim_1",
                    "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
                    "claim_type": "document_fact",
                    "evidence_refs": ["E1"],
                }
            ],
        },
        ensure_ascii=False,
    )
    llm = _RecordingStreamingLLM(second_response)

    _ground_and_finalize_claims(
        persona_id="planning_expert",
        raw=first_raw,
        retrieved=EVIDENCE,
        prompt="원본 프롬프트",
        llm_call=llm,
        validate=_validate_always_ok,
        used=1,
        ground_claims_fn=_ground_claims_fn,
    )

    # discard가 원본 prompt에 대해, 재시도 llm_call보다 먼저 정확히 한 번 일어나야 한다.
    assert llm.discards == [("원본 프롬프트", "grounding_retry", True)]
    assert len(llm.calls) == 1
    assert llm.calls[0] != "원본 프롬프트"  # 재시도는 retry_note가 덧붙은 다른 프롬프트다.


def test_grounding_success_on_first_try_never_discards_anything():
    """근거 연결이 처음부터 성공하면 재시도 자체가 없으므로 discard도 호출되지 않는다 —
    정상 스트리밍 말풍선을 잘못 검토 중 상태로 바꾸면 안 된다."""
    raw = {
        "spoken_text": "발언",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
                "claim_type": "document_fact",
                "evidence_refs": ["E1"],
            }
        ],
    }
    llm = _RecordingStreamingLLM("{}")

    _ground_and_finalize_claims(
        persona_id="planning_expert",
        raw=raw,
        retrieved=EVIDENCE,
        prompt="원본 프롬프트",
        llm_call=llm,
        validate=_validate_always_ok,
        used=1,
        ground_claims_fn=_ground_claims_fn,
    )

    assert llm.discards == []
    assert llm.calls == []


# ---------------------------------------------------------------------------
# 2. IDEATION_EVIDENCE_LINKED 로그 매핑
# ---------------------------------------------------------------------------


def test_ideation_evidence_linked_log_records_actual_chunk_id_for_cited_ref(caplog):
    raw = {
        "spoken_text": "발언",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
                "claim_type": "document_fact",
                "evidence_refs": ["E1"],
            }
        ],
    }

    configure_ideation_trace(enabled=True)
    try:
        with caplog.at_level(logging.INFO, logger="ai.meeting.ideation_trace"):
            _ground_and_finalize_claims(
                persona_id="planning_expert",
                raw=raw,
                retrieved=EVIDENCE,
                prompt="원본 프롬프트",
                llm_call=lambda p: "{}",
                validate=_validate_always_ok,
                used=1,
                ground_claims_fn=_ground_claims_fn,
            )
    finally:
        configure_ideation_trace(enabled=None)

    linked_records = [r for r in caplog.records if "[IDEATION_EVIDENCE_LINKED]" in r.getMessage()]
    assert len(linked_records) == 1
    message = linked_records[0].getMessage()
    assert 'claim_id="claim_1"' in message
    assert '"E1"' in message
    assert '"chk_31da000000000000"' in message
    # 로그에 ref 값이 chunk_id 자리에 잘못 들어가지 않았는지 확인한다.
    assert "chunk_ids=[\"E1\"]" not in message


def test_ideation_evidence_linked_log_skips_claims_with_unknown_ref(caplog):
    raw = {
        "spoken_text": "발언",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "WSCE는 실현 가능성과 경제성을 평가한다.",
                "claim_type": "document_fact",
                "evidence_refs": ["E-does-not-exist"],
            }
        ],
    }

    configure_ideation_trace(enabled=True)
    try:
        with caplog.at_level(logging.INFO, logger="ai.meeting.ideation_trace"):
            _ground_and_finalize_claims(
                persona_id="planning_expert",
                raw=raw,
                retrieved=EVIDENCE,
                prompt="원본 프롬프트",
                llm_call=lambda p: "{}",
                validate=_validate_always_ok,
                used=1,
                ground_claims_fn=_ground_claims_fn,
            )
    finally:
        configure_ideation_trace(enabled=None)

    linked_records = [r for r in caplog.records if "[IDEATION_EVIDENCE_LINKED]" in r.getMessage()]
    assert linked_records == []
