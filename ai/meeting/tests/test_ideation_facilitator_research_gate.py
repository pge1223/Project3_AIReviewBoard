from __future__ import annotations

import json
import sys
from pathlib import Path


MEETING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_conv_nodes import (  # noqa: E402
    make_discussion_facilitator_node,
)


class _QuestioningFacilitatorLLM:
    def __call__(self, _prompt: str) -> str:
        return json.dumps(
            {
                "agreements": [],
                "disagreements": [],
                "facilitator_summary": "문서 작성 과정의 불편을 구체화해야 합니다.",
                "spoken_text": "어떤 문서에서 특히 불편을 느꼈는지 알려주시겠어요?",
                "needs_user_decision": True,
                "user_question": "어떤 문서에서 특히 불편을 느꼈는지 알려주시겠어요?",
            },
            ensure_ascii=False,
        )


def _state() -> dict:
    return {
        "session_id": "FACILITATOR-RAG-GATE",
        "phase": "expert_discussion",
        "round": 1,
        "max_rounds": 5,
        "messages": [
            {
                "message_id": "MSG-DEV",
                "speaker_id": "dev_expert",
                "speaker_name": "개발 의원",
                "role": "개발 의원",
                "round": 1,
                "message_type": "opinion",
                "content": "행정 문서 작성 과정의 비효율을 문제로 정의했습니다.",
                "referenced_message_ids": [],
                "evidence": [],
                "created_at": "2026-07-27T00:00:00+00:00",
                "structured": {
                    "active_issue_id": "problem",
                    "needs_user_input": False,
                },
            }
        ],
        "consensus": [],
        "unresolved_issues": [],
        "notice_and_criteria": {},
        "user_idea": {"description": "행정 문서 작성 지원 서비스"},
        "selected_idea": {
            "title": "행정 문서 작성 지원",
            "problem": "반복 작성과 서식 대응에 시간이 오래 걸립니다.",
        },
        "active_issue_id": "problem",
        "open_issues": [
            {
                "issue_id": "problem",
                "title": "문제 정의",
                "status": "open",
                "turns": 1,
                "family": "problem",
            }
        ],
        "resolved_issues": [],
        "resolved_topics": [],
        "expert_turn_count": 1,
        "llm_calls_used": 0,
        "external_evidence": [],
        "external_evidence_meta": {},
    }


def test_internal_rag_fact_resolves_question_before_external_search():
    external_calls: list[str] = []

    def internal_lookup(_persona_id: str, _query: str) -> list[dict]:
        return [
            {
                "document_id": "DOC-REPORT",
                "document_name": "행정업무 실태조사",
                "chunk_id": "CHK-REPORT-1",
                "document_role": "criteria",
                "final_score": 0.82,
                "text": "실태조사 결과 문서 반복 입력과 서식 변환에 평균 3시간이 소요됩니다.",
            }
        ]

    def external_lookup(_persona_id: str, query: str) -> dict:
        external_calls.append(query)
        return {"external_evidence": []}

    update = make_discussion_facilitator_node(
        _QuestioningFacilitatorLLM(),
        evidence_lookup=internal_lookup,
        external_evidence_lookup=external_lookup,
    )(_state())

    message = update["messages"][0]
    assert external_calls == []
    assert update["phase"] == "expert_discussion"
    assert update["next_route"] == "continue_round"
    assert update["pending_question"] is None
    assert "행정업무 실태조사" not in message["content"]
    assert "평균 3시간" not in message["content"]
    assert not message["content"].endswith("?")
    assert message["structured"]["pre_question_research_source"] == "internal"
    assert message["structured"]["pre_question_research_resolved"] is True
    assert message["evidence"][0]["chunk_id"] == "CHK-REPORT-1"
    assert message["evidence"][0]["document_name"] == "행정업무 실태조사"
    assert "평균 3시간" in message["evidence"][0]["quote"]


def test_form_instruction_is_not_an_answer_and_external_rag_is_used_next():
    external_calls: list[str] = []

    def internal_lookup(_persona_id: str, _query: str) -> list[dict]:
        return [
            {
                "document_id": "DOC-FORM",
                "document_name": "신청서 양식",
                "chunk_id": "CHK-FORM-1",
                "document_role": "criteria",
                "final_score": 0.9,
                "text": "<작성 요령> 문서 작성 과정의 비효율성을 구체적으로 작성하십시오.",
            }
        ]

    def external_lookup(_persona_id: str, query: str) -> dict:
        external_calls.append(query)
        return {
            "external_evidence": [
                {
                    "source_id": "SRC-1",
                    "document_id": "EXT-DOC-1",
                    "chunk_id": "EXT-CHK-1",
                    "title": "공공부문 행정업무 조사",
                    "publisher": "공공연구원",
                    "source_url": "https://example.test/report",
                    "reference_date": "2026-01-01",
                    "quote": "행정기관 종사자는 반복적인 문서 입력과 기관별 서식 변환을 주요 업무 부담으로 응답했습니다.",
                    "reference_only": True,
                }
            ],
            "used_dataset_search": True,
            "used_public_api_search": False,
            "warnings": [],
        }

    update = make_discussion_facilitator_node(
        _QuestioningFacilitatorLLM(),
        evidence_lookup=internal_lookup,
        external_evidence_lookup=external_lookup,
    )(_state())

    message = update["messages"][0]
    assert len(external_calls) == 1
    assert update["phase"] == "expert_discussion"
    assert update["pending_question"] is None
    assert "공공부문 행정업무 조사" not in message["content"]
    assert "반복적인 문서 입력" not in message["content"]
    assert message["structured"]["pre_question_research_source"] == "external"
    assert update["external_evidence"][0]["source_id"] == "SRC-1"


def test_hwpx_raw_chunk_is_kept_in_evidence_not_facilitator_message():
    class _RawChunkFacilitatorLLM:
        def __call__(self, _prompt: str) -> str:
            raw = (
                "'붙임2_2026_공공기관_AI_혁신_챌린지_참가_신청_서식_실증·PoC_F.hwpx'에서는\n"
                "○\n-\n※\n<작성 요령> 타 기관 확산 가능성을 구체적으로 작성"
            )
            return json.dumps(
                {
                    "agreements": [],
                    "disagreements": [],
                    "facilitator_summary": raw,
                    "spoken_text": raw,
                    "needs_user_decision": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )

    def internal_lookup(_persona_id: str, _query: str) -> list[dict]:
        return [
            {
                "document_id": "DOC-HWPX",
                "document_name": "붙임2_2026_공공기관_AI_혁신_챌린지_참가_신청_서식_실증·PoC_F.hwpx",
                "chunk_id": "CHK-HWPX-1",
                "section": "타 기관 확산 계획 및 노력",
                "document_role": "criteria",
                "quote": (
                    "2. 타 기관 확산 계획 및 노력 ○ - ※ "
                    "<작성 요령> 타 기관 확산 가능성을 구체적으로 작성 "
                    "3. 대국민 체감 효과"
                ),
            }
        ]

    update = make_discussion_facilitator_node(
        _RawChunkFacilitatorLLM(),
        evidence_lookup=internal_lookup,
    )(_state())

    message = update["messages"][0]
    assert message["content"] == (
        "관련 공모전 근거를 확인했습니다. 해당 기준을 바탕으로 후보를 검증하겠습니다."
    )
    assert ".hwpx" not in message["content"]
    assert "<작성 요령>" not in message["content"]
    assert message["evidence"][0]["document_name"].endswith(".hwpx")
    assert "<작성 요령>" in message["evidence"][0]["quote"]
    assert message["evidence"][0]["chunk_id"] == "CHK-HWPX-1"


def test_user_is_asked_only_when_internal_and_external_research_are_empty():
    def internal_lookup(_persona_id: str, _query: str) -> list[dict]:
        return []

    def external_lookup(_persona_id: str, _query: str) -> dict:
        return {
            "external_evidence": [],
            "used_dataset_search": False,
            "used_public_api_search": False,
            "warnings": [],
        }

    update = make_discussion_facilitator_node(
        _QuestioningFacilitatorLLM(),
        evidence_lookup=internal_lookup,
        external_evidence_lookup=external_lookup,
    )(_state())

    message = update["messages"][0]
    assert update["phase"] == "awaiting_user_decision"
    assert update["pending_question"]
    assert message["content"].endswith("?")
    assert message["structured"]["pre_question_research_resolved"] is False


def test_real_user_choice_skips_research_and_keeps_the_gated_question():
    lookup_calls: list[str] = []
    state = _state()
    state["messages"][-1]["structured"].update(
        {
            "user_decision_required": True,
            "user_question": "우선 적용 지역을 서울과 부산 중 어디로 정할까요?",
            "decision_options": [
                {"label": "서울", "detail": "서울부터 적용"},
                {"label": "부산", "detail": "부산부터 적용"},
            ],
        }
    )

    def internal_lookup(_persona_id: str, query: str) -> list[dict]:
        lookup_calls.append(f"internal:{query}")
        return []

    def external_lookup(_persona_id: str, query: str) -> dict:
        lookup_calls.append(f"external:{query}")
        return {"external_evidence": []}

    update = make_discussion_facilitator_node(
        _QuestioningFacilitatorLLM(),
        evidence_lookup=internal_lookup,
        external_evidence_lookup=external_lookup,
    )(state)

    message = update["messages"][0]
    assert lookup_calls == []
    assert update["phase"] == "awaiting_user_decision"
    assert update["pending_question"] == "우선 적용 지역을 서울과 부산 중 어디로 정할까요?"
    assert message["structured"]["pre_question_research_query"] is None
    assert [choice["label"] for choice in message["structured"]["choices"]] == ["서울", "부산"]
