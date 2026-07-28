# 작성자: 용준/Claude(2026-07-21)
# 목적: /ideation-conversation/start API가 user_idea 없이도(discovery 모드) 400을
#       반환하지 않는지, competition_name은 계속 필수인지, 응답에 discovery 관련 필드
#       (ideation_mode/idea_candidates 등)가 포함되는지, 기존 요청 형식(초기 아이디어를
#       채워 보내는 방식)과의 하위 호환이 유지되는지를 검증한다.
#
#       app.main 전체를 띄우지 않고(Mongo/KURE 임베더 로딩 등 무거운 초기화를 피하기 위해,
#       backend/tests/test_meetings_followup.py가 app.api.routes.meetings를 직접 import하는
#       것과 같은 패턴) ideation_conversation_preview.router만 담은 최소 FastAPI 앱을 쓴다.
#       OpenAI 실제 호출은 _build_llm_call을 monkeypatch해 대체한다(네트워크 호출 없음).
# import: fastapi.testclient, pytest; app.api.routes.ideation_conversation_preview 모듈.

import json
import re
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config import settings  # noqa: E402
import app.api.routes.ideation_conversation_preview as conv_route  # noqa: E402

_REMAINING_TOPICS_RE = re.compile(
    r"\[아직 확인되지 않은 주제\(우선순위 순\) remaining_topics\]\n(.*?)\n\n", re.S
)


def _topic_from_prompt(prompt: str) -> str:
    match = _REMAINING_TOPICS_RE.search(prompt)
    if not match:
        return "problem"
    try:
        remaining = json.loads(match.group(1))
    except (ValueError, TypeError):
        return "problem"
    return remaining[0] if remaining else "problem"


def _stub_llm_call(session_id: str, model: str):
    def llm_call(prompt: str) -> str:
        # 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) —
        # candidate_planning보다 앞선 신규 단계 5종의 stub. 최소 조건(방향 3개 이상/반론
        # 1회 이상/수정·결합 1회 이상)을 1라운드 안에 채워, 아래 기존 [후보 생성 규칙] 이하
        # 스텁으로 곧바로 이어지게 한다(기존 테스트가 기대하는 후보 내용/개수는 그대로 유지).
        if "[문제 영역 생성 규칙]" in prompt:
            return json.dumps(
                {
                    "problem_areas": [
                        {"area_id": "area_1", "title": "문제 영역 1", "summary": "요약1", "who_is_affected": "대상1"},
                        {"area_id": "area_2", "title": "문제 영역 2", "summary": "요약2", "who_is_affected": "대상2"},
                    ]
                },
                ensure_ascii=False,
            )
        if "[문제 정의 규칙]" in prompt:
            return json.dumps(
                {
                    "problem_definition": {
                        "problem": "문제 정의",
                        "target_user": "대상 사용자",
                        "user_context": "사용자 상황",
                        "root_cause": "원인",
                        "existing_solution": "기존 방식",
                        "existing_limitations": "기존 한계",
                    }
                },
                ensure_ascii=False,
            )
        if "[발산 규칙]" in prompt:
            return json.dumps(
                {
                    "solution_directions": [
                        {
                            "direction_id": f"direction_{i}",
                            "title": f"방향 {i}",
                            "core_principle": f"원리 {i}",
                            "mechanism": f"작동 방식 {i}",
                            "target_user_fit": f"적합성 {i}",
                        }
                        for i in (1, 2, 3, 4)
                    ]
                },
                ensure_ascii=False,
            )
        if "[반론·결합 규칙]" in prompt:
            return json.dumps(
                {
                    "round_events": [
                        {
                            "issued_by": "dev_expert",
                            "action_type": "critique",
                            "target_direction_ids": ["direction_1", "direction_3"],
                            "spoken_text": "방향 1과 3은 그대로 구현하기 어렵습니다.",
                            "detail": "상세",
                        },
                        {
                            "issued_by": "planning_expert",
                            "action_type": "merge",
                            "target_direction_ids": ["direction_1", "direction_3"],
                            "spoken_text": "방향 1과 3을 결합하겠습니다.",
                            "detail": "상세",
                        },
                    ],
                    "resulting_direction": {
                        "direction_id": "direction_merged_1",
                        "title": "결합 방향",
                        "core_principle": "결합 원리",
                        "mechanism": "결합 작동 방식",
                        "target_user_fit": "결합 적합성",
                        "parent_direction_ids": ["direction_1", "direction_3"],
                        "strengths": ["결합 장점"],
                        "open_assumptions": ["검증 가정"],
                    },
                },
                ensure_ascii=False,
            )
        if "[검증 규칙]" in prompt:
            return json.dumps(
                {
                    "planning": {
                        "value_worth_solving": "가치 있음",
                        "target_user_clarity": "명확함",
                        "differentiation": "차별성 있음",
                        "usage_motivation": "동기 있음",
                        "contest_alignment": "기준 부합",
                        "concerns": [],
                    },
                    "technical": {
                        "data_availability": "확보 가능",
                        "feasibility": "가능",
                        "ai_necessity": "필요",
                        "privacy_or_security_risks": "낮음",
                        "prototype_feasibility": "가능",
                        "concerns": [],
                    },
                    "unresolved_assumptions": [],
                },
                ensure_ascii=False,
            )
        if "[후보 생성 규칙]" in prompt:
            return json.dumps(
                {
                    "contest_analysis": {
                        "purpose": "p",
                        "key_criteria": ["a"],
                        "required_tech_or_theme": ["b"],
                        "suitable_problem_domains": ["c"],
                        "constraints": ["d"],
                        "unknown_from_notice": ["e"],
                    },
                    "candidates": [
                        {
                            "candidate_id": "candidate_1",
                            "title": "후보1",
                            "problem": "문제1",
                            "target_user": "사용자1",
                            "usage_scenario": "상황1",
                            "core_value": "가치1",
                            "solution": "해결1",
                            "main_features": ["기능1"],
                            "differentiation": "차별1",
                            "contest_fit": "적합1",
                            "success_metrics": ["지표1"],
                            "source_direction_ids": ["direction_2"],
                            "reflected_evolution_ids": [],
                        },
                        {
                            "candidate_id": "candidate_2",
                            "title": "후보2",
                            "problem": "문제2",
                            "target_user": "사용자2",
                            "usage_scenario": "상황2",
                            "core_value": "가치2",
                            "solution": "해결2",
                            "main_features": ["기능2"],
                            "differentiation": "차별2",
                            "contest_fit": "적합2",
                            "success_metrics": ["지표2"],
                            "source_direction_ids": ["direction_merged_1"],
                            "reflected_evolution_ids": [],
                        },
                    ],
                },
                ensure_ascii=False,
            )
        if "[검토 규칙]" in prompt:
            return json.dumps(
                {
                    "candidate_reviews": [
                        {
                            "candidate_id": "candidate_1",
                            "required_data": ["d1"],
                            "technical_approach": "t1",
                            "mvp_scope": "m1",
                            "feasibility": "high",
                            "risks": ["r1"],
                            "dev_notes": None,
                        },
                        {
                            "candidate_id": "candidate_2",
                            "required_data": ["d2"],
                            "technical_approach": "t2",
                            "mvp_scope": "m2",
                            "feasibility": "medium",
                            "risks": ["r2"],
                            "dev_notes": None,
                        },
                    ]
                },
                ensure_ascii=False,
            )
        if "[질문 규칙]" in prompt:
            is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
            speaker = "planning_expert" if is_planning else "dev_expert"
            payload = {
                "spoken_text": f"[{speaker}] 발화 질문",
                "judgment": f"[{speaker}] 판단",
                "question": f"[{speaker}] 질문",
                "question_topic": _topic_from_prompt(prompt),
                "referenced_message_ids": [],
                "evidence": [],
            }
            # 후보 결합 직후 첫 질문(require_combine_structure=true)이면 요청된 4단 구조
            # 필드도 채운다 — ai/meeting/tests/test_ideation_discovery_graph.py의
            # _CombineAwareScriptedLLM과 같은 최소 대응.
            if "[결합 직후 첫 메시지 여부 require_combine_structure]\ntrue" in prompt:
                payload["user_selection_summary"] = f"[{speaker}] 사용자 선택 반영 요약"
                payload["proposal"] = f"[{speaker}] 제안"
            return json.dumps(payload, ensure_ascii=False)
        if "[판정 규칙]" in prompt:
            return json.dumps(
                {"answer_type": "answer", "reason": "충분", "follow_up_question": None, "clarification_response": None},
                ensure_ascii=False,
            )
        if "[제안 규칙]" in prompt:
            # 용준/Claude(2026-07-21, 요청: "모르겠다" UX 개선) — 전문가 위임(expert_delegation)
            # 제안 생성 프롬프트.
            is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
            speaker = "planning_expert" if is_planning else "dev_expert"
            return json.dumps(
                {
                    "spoken_text": f"[{speaker}] 발화 제안",
                    "proposal": f"[{speaker}] 임시 제안",
                    "reason": f"[{speaker}] 제안 이유",
                    "assumption": f"[{speaker}] 이 방향으로 진행",
                    "responding_to": None,
                    "revision": None,
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )
        if "[위임 검토 규칙]" in prompt:
            # 용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장) —
            # stance="보완"(REVISION_TRIGGER_STANCES 밖)이라 수정 턴은 추가로 실행되지 않는다.
            is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
            reviewer = "planning_expert" if is_planning else "dev_expert"
            return json.dumps(
                {
                    "stance": "보완",
                    "spoken_text": f"[{reviewer}] 발화 검토",
                    "judgment": f"[{reviewer}] 검토 판단",
                    "reason": f"[{reviewer}] 검토 근거",
                    "responding_to": "상대 전문가의 임시 제안",
                    "agreement": f"[{reviewer}] 동의 지점",
                    "concern": "",
                    "recommendation": f"[{reviewer}] 채택 가능",
                    "referenced_message_ids": [],
                    "evidence": [],
                },
                ensure_ascii=False,
            )
        if "[위임 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "agreements": ["제안 방향에 합의"],
                    "considerations": [],
                    "final_recommendation": "이 방향으로 진행하겠습니다.",
                    "spoken_text": "이 방향으로 진행하겠습니다.",
                },
                ensure_ascii=False,
            )
        if "[의견 규칙]" in prompt:
            # 용준/Claude(2026-07-21, 요청: 위원 간 실제 회의로 개편) — 사용자가 기획/개발
            # 두 질문에 모두 정상 답변하면(위임 없이) expert_discussion이 실행된다.
            is_planning = "당신은 AI Review Board의 기획 전문가입니다" in prompt
            is_dev = not is_planning
            speaker = "planning_expert" if is_planning else "dev_expert"
            is_response_stage = "[discussion_stage]\nresponse" in prompt
            return json.dumps(
                {
                    "stance": "보완",
                    "spoken_text": f"[{speaker}] 발화 판단",
                    "judgment": f"[{speaker}] 판단",
                    "reason": f"[{speaker}] 근거",
                    "suggestion": f"[{speaker}] 제안",
                    "interim_conclusion": f"[{speaker}] 현재 임시 결론",
                    "responding_to": "상대 발언" if is_response_stage else None,
                    "agreement": "동의 지점" if is_response_stage else "",
                    "concern": "",
                    "confirmed": [],
                    "unconfirmed": [],
                    "referenced_message_ids": [],
                    "evidence": [],
                    "next_action": None,
                    "active_issue_id": "mvp_scope",
                    "active_issue_title": "MVP 범위",
                    "new_information": [f"[{speaker}] 새로 확인된 내용"],
                    "proposal": f"[{speaker}] 제안",
                    "changed_position": False,
                    "needs_counterpart_response": not is_dev,
                    "recommended_next_speaker": "ideation_facilitator" if is_dev else "dev_expert",
                    "issue_resolved": bool(is_dev),
                    "needs_user_input": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(
                {
                    "agreements": [],
                    "disagreements": [],
                    "facilitator_summary": "두 전문가가 이번 라운드 의견을 정리했습니다.",
                    "spoken_text": "두 위원이 이번 라운드 의견을 정리했습니다.",
                    "needs_user_decision": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )
        if "[캔버스 갱신 규칙]" in prompt:
            return json.dumps(
                {
                    "problem": "문의 응대 부담",
                    "target_user": "소상공인",
                    "core_value": "응대 시간 절감",
                    "solution": "FAQ 자동 응답",
                    "differentiation": "저비용 구축",
                    "feasibility": "medium",
                    "risks": ["오답 위험"],
                    "contest_fit": "실현가능성 기준 대응",
                },
                ensure_ascii=False,
            )
        if "[해석 규칙]" in prompt:
            # 용준/Claude(2026-07-21, /board 실 연동): 후보 결합(combine) 응답 — 이번
            # 확장(원본 후보/결합 분석을 API 응답에 노출)을 검증하는 테스트에서만 실제로
            # 호출된다("1번과 2번 결합" 같은 자연어 요청).
            return json.dumps(
                {
                    "resolution": "combine",
                    "selected_candidate_ids": ["candidate_1", "candidate_2"],
                    "selection_reason": "두 후보의 장점을 결합",
                    "combined_idea": {
                        "title": "결합 아이디어",
                        "problem": "결합된 문제",
                        "target_user": "결합된 사용자",
                        "usage_scenario": "결합 상황",
                        "core_value": "결합 가치",
                        "solution": "결합 해결책",
                        "main_features": ["결합 기능"],
                        "required_data": ["결합 데이터"],
                        "technical_approach": "결합 기술",
                        "mvp_scope": "결합 MVP",
                        "differentiation": "결합 차별성",
                        "contest_fit": "결합 적합성",
                        "success_metrics": ["결합 지표"],
                    },
                    "merge_analysis": {
                        "common_problem": "반복 업무 부담",
                        "common_value": "사장님의 시간 절약",
                        "fit": "high",
                        "primary_features": ["기능1"],
                        "secondary_features": ["기능2"],
                        "conflicts": [],
                        "open_questions": [],
                    },
                    "unverified_assumptions": ["결합 가정1"],
                    "clarifying_question": None,
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")

    return llm_call


@pytest.fixture(autouse=True)
def _enable_preview_and_stub_llm(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_IDEATION_PREVIEW", True)
    monkeypatch.setattr(conv_route, "_build_llm_call", _stub_llm_call)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(conv_route.router)
    return TestClient(app)


def _start_discovery_and_select_problem(client: TestClient, **start_json_overrides) -> dict:
    """용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — discovery
    모드는 이제 /start가 곧바로 후보를 만들지 않고 "problem_discovery"에서 멈춘다. 이
    헬퍼는 문제 영역 1번을 선택해 기존 계약(즉 /start 직후 후보 2개가 있던 지점)까지
    자동으로 진행시킨다."""
    body = {"competition_name": "데모 공모전", "competition_document": "실현가능성을 평가한다."}
    body.update(start_json_overrides)
    start_resp = client.post("/ideation-conversation/start", json=body)
    assert start_resp.status_code == 200
    assert start_resp.json()["phase"] == "awaiting_problem_focus_selection"
    session_id = start_resp.json()["session_id"]
    reply_resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "1번"})
    assert reply_resp.status_code == 200
    assert reply_resp.json()["phase"] == "awaiting_candidate_selection"
    return reply_resp.json()


def _force_session_to_awaiting_planning_answer(session_id: str, llm_call) -> None:
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) 보존 검증용 헬퍼 — 새 세션은
    더 이상 phase="awaiting_planning_answer"(1:1 인터뷰 진입점)로 시작하지 않지만
    (start_ideation_conversation이 곧바로 라운드테이블로 진입한다), planning_question
    노드와 그 정지 지점 자체는 하위 호환을 위해 코드에 그대로 남아 있다. 과거에 저장된
    세션이 이 phase에 멈춰 있었다고 가정하고 세션 스토어에 직접 주입해, /reply API가 그
    보존된 경로를 여전히 올바르게 처리하는지 검증한다."""
    from graph.ideation_conv_nodes import make_conv_question_node
    from graph.ideation_conv_state import initial_conv_state

    state = dict(
        initial_conv_state(
            session_id,
            {"competition_name": "데모 공모전"},
            {"description": "소상공인이 손님 문의에 자동으로 답하는 챗봇"},
        )
    )
    state["phase"] = "planning_question"
    state["messages"] = []
    node = make_conv_question_node("planning_expert", "awaiting_planning_answer", llm_call)
    update = node(state)
    state = {**state, **update, "messages": state["messages"] + update.get("messages", [])}
    conv_route._store.update(session_id, state)


def test_start_without_user_idea_returns_200_and_discovery_mode(client: TestClient):
    resp = client.post(
        "/ideation-conversation/start",
        json={"competition_name": "데모 공모전", "competition_document": "실현가능성을 평가한다."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ideation_mode"] == "discovery"
    # 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — discovery
    # 모드는 이제 candidate_generation 이전에 문제 발견 단계(problem_discovery)부터
    # 시작한다 — /start만으로는 아직 후보가 만들어지지 않는다.
    assert body["phase"] == "awaiting_problem_focus_selection"
    assert body["active_stage"] == "candidate_discovery"

    selected_body = _start_discovery_and_select_problem(client)
    assert len(selected_body["idea_candidates"]) >= 2
    assert selected_body["active_stage"] == "candidate_discovery"


def test_active_stage_switches_to_refinement_after_candidate_selection(client: TestClient):
    """discovery로 시작해 후보를 선택하면 ideation_mode는 "discovery"로 유지되지만,
    확정 전(잠정 선택)에는 active_stage가 "candidate_selection"이고, 사용자가 실제로
    확정해야만 "refinement"로 바뀐다(요청 2·3번 — 선택 즉시 확정되지 않는다)."""
    body = _start_discovery_and_select_problem(client)
    session_id = body["session_id"]

    reply_resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "1번"})
    assert reply_resp.status_code == 200
    body = reply_resp.json()
    assert body["ideation_mode"] == "discovery"
    assert body["phase"] == "awaiting_concept_confirmation"
    assert body["active_stage"] == "candidate_selection"

    confirm_resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "확정할게요"})
    assert confirm_resp.status_code == 200
    confirm_body = confirm_resp.json()
    assert confirm_body["phase"] == "discussion_complete"
    assert confirm_body["active_stage"] == "refinement"


def test_start_with_whitespace_only_user_idea_returns_200_and_discovery_mode(client: TestClient):
    resp = client.post(
        "/ideation-conversation/start",
        json={"competition_name": "데모 공모전", "user_idea": "   "},
    )
    assert resp.status_code == 200
    assert resp.json()["ideation_mode"] == "discovery"


def test_start_without_competition_name_still_returns_400(client: TestClient):
    # competition_name은 계속 필수다 — 빈 문자열이면 _clamp_text가 400을 반환해야 한다.
    resp = client.post("/ideation-conversation/start", json={"competition_name": "", "user_idea": ""})
    assert resp.status_code == 400


def test_start_with_user_idea_still_returns_refinement_mode_backward_compatible(client: TestClient):
    """기존 요청 형식(초기 아이디어를 채워 보내는 방식)이 그대로 동작하는지 — 하위 호환성."""
    resp = client.post(
        "/ideation-conversation/start",
        json={
            "competition_name": "데모 공모전",
            "competition_document": "실현가능성을 평가한다.",
            "user_idea": "소상공인이 손님 문의에 자동으로 답하는 챗봇",
            "max_rounds": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ideation_mode"] == "refinement"
    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 초기 아이디어가 있으면
    # 곧바로 라운드테이블 한 라운드까지 실행된 뒤 "awaiting_user_decision"으로 멈춘다.
    assert body["phase"] == "discussion_complete"
    # 기존 필드가 하나도 빠지지 않았다.
    for field in (
        "session_id",
        "phase",
        "round",
        "max_rounds",
        "messages",
        "consensus",
        "unresolved_issues",
        "idea_proposal",
        "ideation_mode",
        "active_stage",
        "idea_candidates",
        "selected_idea",
        "selection_reason",
        "error",
    ):
        assert field in body
    # 용준/Claude(2026-07-21, 질문 주제 구조화): resolved_topics/pending_question_topic은
    # 순수 추가 필드다 — 요청 16번(기존 필드 유지) + 신규 필드 노출을 함께 확인한다.
    assert body["resolved_topics"] == []
    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — pending_question_topic은
    # 진행자가 실제로 사용자에게 직접 질문을 던졌을 때만 채워진다(discussion_facilitator
    # stub은 needs_user_decision=False를 반환하므로 None이다). 1:1 인터뷰 질문 노드가 만드는
    # "problem"이라는 고정값은 더 이상 첫 라운드에서 나오지 않는다.
    assert body["pending_question_topic"] is None


def test_reply_response_includes_topic_fields_after_first_answer(client: TestClient):
    """/reply 응답에도 resolved_topics/pending_question_topic이 포함되고, 답변이 answer로
    판정되면 resolved_topics가 실제로 갱신되는지 확인한다."""
    start_resp = client.post(
        "/ideation-conversation/start",
        json={
            "competition_name": "데모 공모전",
            "user_idea": "소상공인이 손님 문의에 자동으로 답하는 챗봇",
        },
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]
    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 라운드테이블은 진행자가
    # 실제로 질문을 던졌을 때만 pending_question_topic을 채운다(stub은 needs_user_decision
    # =False를 반환).
    assert start_resp.json()["pending_question_topic"] is None

    reply_resp = client.post(
        f"/ideation-conversation/{session_id}/reply",
        json={"message": "반복 문의 응대 부담을 해결하고 싶습니다"},
    )
    assert reply_resp.status_code == 200
    body = reply_resp.json()
    assert "resolved_topics" in body
    assert "pending_question_topic" in body


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-21, /board 실 연동): _serialize_state에 순수 추가한
# original_idea_candidates/selection_intent/user_selection_message/source_candidates/
# merge_analysis 필드가 실제 API 응답에 노출되는지 검증한다 — /board의 결과 화면과 후보
# 결합 컨텍스트 표시가 이 필드들에 의존한다.
# ---------------------------------------------------------------------------


def test_start_response_includes_original_candidates_field(client: TestClient):
    body = _start_discovery_and_select_problem(client)
    assert len(body["original_idea_candidates"]) >= 2
    assert body["selection_intent"] is None
    assert body["source_candidates"] == []
    assert body["merge_analysis"] is None


def test_start_response_exposes_updated_idea_canvas(client: TestClient):
    resp = client.post(
        "/ideation-conversation/start",
        json={
            "competition_name": "데모 공모전",
            "competition_document": "실현가능성을 평가한다.",
            "user_idea": "소상공인 문의를 자동화하고 싶습니다.",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["phase"] == "discussion_complete"
    assert body["idea_canvas"]["problem"] == "문의 응대 부담"
    assert body["idea_canvas"]["feasibility"] == "medium"


# ---------------------------------------------------------------------------
# 가은/Claude(2026-07-22, 요청: 신청양식 항목 약한 주입) — StartRequest.application_form_items가
# 실제로 그래프까지 전달돼 discussion 프롬프트에 주입되는지 검증한다.
# ---------------------------------------------------------------------------


def test_start_with_application_form_items_injects_into_discussion_prompt(client: TestClient, monkeypatch):
    captured_prompts: list[str] = []
    base_llm = _stub_llm_call("CAPTURE-SESSION", "gpt-4o-mini")

    def _capturing_llm_call(session_id: str, model: str):
        def llm_call(prompt: str) -> str:
            captured_prompts.append(prompt)
            return base_llm(prompt)

        return llm_call

    monkeypatch.setattr(conv_route, "_build_llm_call", _capturing_llm_call)

    resp = client.post(
        "/ideation-conversation/start",
        json={
            "competition_name": "데모 공모전",
            "competition_document": "실현가능성을 평가한다.",
            "user_idea": "소상공인이 손님 문의에 자동으로 답하는 챗봇",  # refinement 모드 -> 곧바로 discussion 호출
            "application_form_items": [
                {"field_name": "문제 정의", "description": "해결하려는 문제", "char_limit": 300},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["application_form_items"] == [
        {"field_name": "문제 정의", "description": "해결하려는 문제", "char_limit": 300}
    ]

    discussion_prompts = [p for p in captured_prompts if "[의견 규칙]" in p]
    assert discussion_prompts
    assert any("문제 정의" in p and "해결하려는 문제" in p for p in discussion_prompts)


def test_start_without_application_form_items_defaults_to_empty_list(client: TestClient):
    """필드를 아예 안 보내면(기존 클라이언트) 기본값 빈 리스트로 동작하고, 응답에도
    빈 리스트로 노출된다 — 순수 추가 필드라 하위 호환이 유지된다."""
    resp = client.post(
        "/ideation-conversation/start",
        json={"competition_name": "데모 공모전", "competition_document": "실현가능성을 평가한다."},
    )
    assert resp.status_code == 200
    assert resp.json()["application_form_items"] == []


def test_reply_response_includes_merge_context_fields_after_combine(client: TestClient):
    body = _start_discovery_and_select_problem(client)
    session_id = body["session_id"]

    reply_resp = client.post(
        f"/ideation-conversation/{session_id}/reply", json={"message": "1번과 2번 결합해줘"}
    )
    assert reply_resp.status_code == 200
    body = reply_resp.json()

    assert body["selection_intent"] == "combine"
    assert body["user_selection_message"] == "1번과 2번 결합해줘"
    assert {c["candidate_id"] for c in body["source_candidates"]} == {"candidate_1", "candidate_2"}
    assert body["merge_analysis"]["fit"] == "high"
    assert body["merge_analysis"]["common_problem"] == "반복 업무 부담"
    # 용준/Claude(2026-07-27, 요청: "발산 전에 완성, 선택 즉시 확정" 구조 개편) — 결합
    # 확정 직후에도 selected_idea는 아직 None이다(provisional_idea만 채워진다). 사용자가
    # 실제로 확정해야 selected_idea가 채워지고 라운드테이블이 시작된다.
    assert body["selected_idea"] is None
    assert body["provisional_idea"]["title"] == "결합 아이디어"
    assert body["phase"] == "awaiting_concept_confirmation"

    confirm_resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "확정할게요"})
    assert confirm_resp.status_code == 200
    confirm_body = confirm_resp.json()
    assert confirm_body["selected_idea"]["title"] == "결합 아이디어"
    assert confirm_body["idea_locked"] is True
    # 용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — 확정 직후 라운드테이블이
    # 같은 요청 안에서 곧바로 끝까지 실행돼 "discussion_complete"로 멈춘다.
    assert confirm_body["phase"] == "discussion_complete"


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-21, 요청: "모르겠다" UX 개선): /reply API 계층에서도 사용자가
# "잘 모르겠어" 같은 표현으로 답하면 같은 질문을 반복하지 않고, 담당 전문가의 제안
# 메시지가 생성되며 다음 단계로 진행되는지 확인한다(그래프 레벨 상세 검증은
# ai/meeting/tests/test_ideation_conv_graph.py가 담당한다).
# ---------------------------------------------------------------------------


def test_reply_with_dont_know_advances_instead_of_repeating_question(client: TestClient):
    """용준/Claude(2026-07-21, 요청: 전문가 라운드테이블 전환) — expert_delegation 위임
    흐름은 여전히 phase="awaiting_planning_answer"/"awaiting_developer_answer"(1:1 인터뷰
    진입점)에서만 동작한다(PHASE_TO_PENDING_PERSONA). 새 세션은 더 이상 그 phase로
    시작하지 않으므로, 레거시 세션(과거에 그 phase에 멈춰 있던 세션)을 세션 스토어에 직접
    주입해 /reply API가 그 보존된 경로를 여전히 올바르게 처리하는지 검증한다."""
    start_resp = client.post(
        "/ideation-conversation/start",
        json={
            "competition_name": "데모 공모전",
            "user_idea": "소상공인이 손님 문의에 자동으로 답하는 챗봇",
        },
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]
    _force_session_to_awaiting_planning_answer(session_id, _stub_llm_call(session_id, "gpt-test"))

    reply_resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "잘 모르겠어"})
    assert reply_resp.status_code == 200
    body = reply_resp.json()

    assert body["phase"] == "awaiting_developer_answer"
    # 용준/Claude(2026-07-21, 요청: expert_delegation도 위원 간 상호 검토로 확장) — 단일
    # 위원 제안으로 끝나지 않고 [담당(기획) 제안, 반대(개발) 검토, 진행자 권고안, 다음
    # 질문(개발)] 순서로 이어진다.
    proposal_message = next(
        m for m in body["messages"] if m["speaker_id"] == "planning_expert" and m["message_type"] == "opinion"
    )
    # 용준/Claude(2026-07-22, 요청: 보고서형 메시지 → 자연스러운 회의 발화 전환) — content는
    # 이제 spoken_text + 고정 안내 문구다([기획 전문가 제안] 헤더는 더 이상 붙지 않는다).
    # proposal/reason/assumption 원문은 structured에서 확인한다.
    assert proposal_message["structured"]["proposal"]
    assert "이 가정은 언제든 수정할 수 있습니다" in proposal_message["content"]
    assert any(m["speaker_id"] == "ideation_facilitator" and m["message_type"] == "summary" for m in body["messages"])


def test_session_recovery_preserves_discussion_rounds_and_message_order(client: TestClient):
    """요청 2026-07-21 후속 5번 — 세션 복구(GET /ideation-conversation/{session_id}) 후에도
    discussion_rounds와 메시지 순서가 그대로 보존되는지 확인한다. 문제 선택 -> 후보 선택 ->
    확정(concept_confirmation, 요청 2·3번 — 선택만으로는 라운드테이블이 열리지 않는다) ->
    자유 발언 두 번(expert_discussion 라운드 반복)까지 실제로 진행한 뒤, GET 응답이 마지막
    POST /reply 응답과 완전히 동일한지 비교한다."""
    body = _start_discovery_and_select_problem(client)
    session_id = body["session_id"]

    candidate_selected = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "1번"}).json()
    assert candidate_selected["phase"] == "awaiting_concept_confirmation"
    confirmed = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "확정할게요"}).json()
    assert confirmed["idea_locked"] is True

    client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "타깃은 동네 카페 사장님입니다"})
    reply_resp = client.post(
        f"/ideation-conversation/{session_id}/reply", json={"message": "카카오톡 채널 API를 쓰려 합니다"}
    )
    assert reply_resp.status_code == 200
    reply_body = reply_resp.json()
    assert reply_body["discussion_rounds"], "discussion_rounds가 비어 있으면 안 된다"

    get_resp = client.get(f"/ideation-conversation/{session_id}")
    assert get_resp.status_code == 200
    get_body = get_resp.json()

    # 메시지 순서와 speaker_id 시퀀스가 완전히 동일해야 한다(세션 복구 후에도 회의 발언
    # 순서가 보존된다).
    assert [m["message_id"] for m in get_body["messages"]] == [m["message_id"] for m in reply_body["messages"]]
    assert [m["speaker_id"] for m in get_body["messages"]] == [m["speaker_id"] for m in reply_body["messages"]]
    assert get_body["discussion_rounds"] == reply_body["discussion_rounds"]
    assert get_body["phase"] == reply_body["phase"]


# ---------------------------------------------------------------------------
# 후속 요청(2026-07-27) 4번 — /reply의 action_code/action_payload가 실제로 그래프까지
# 전달돼 자연어 파싱보다 우선 적용되는지 확인한다.
# ---------------------------------------------------------------------------


def test_reply_action_code_confirm_concept_locks_idea_via_api(client: TestClient):
    body = _start_discovery_and_select_problem(client)
    session_id = body["session_id"]

    selected_resp = client.post(f"/ideation-conversation/{session_id}/reply", json={"message": "1번"})
    assert selected_resp.json()["phase"] == "awaiting_concept_confirmation"

    # 텍스트만 보면 확정 키워드가 전혀 없지만, action_code가 우선 적용돼 확정된다.
    confirm_resp = client.post(
        f"/ideation-conversation/{session_id}/reply",
        json={"message": "음...", "action_code": "confirm_concept"},
    )
    assert confirm_resp.status_code == 200
    assert confirm_resp.json()["idea_locked"] is True


def test_reply_action_code_invalid_payload_returns_400(client: TestClient):
    start_resp = client.post(
        "/ideation-conversation/start",
        json={"competition_name": "데모 공모전", "competition_document": "실현가능성을 평가한다."},
    )
    session_id = start_resp.json()["session_id"]

    resp = client.post(
        f"/ideation-conversation/{session_id}/reply",
        json={"message": "1번", "action_code": "select_problem_focus", "action_payload": {"indices": [99]}},
    )
    assert resp.status_code == 400
    assert "action_payload" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-27, 요청: "구버전 후보 화면이 브라우저 세션 재개로 계속 다시 뜬다 —
# 자동 폐기") — 문제 발견 단계 개편(2026-07-27) 이전에는 discovery 세션이 candidate_generation
# 부터 곧바로 시작해 problem_definition을 전혀 채우지 않았다. 그런 구버전 세션이
# MongoDB/메모리 캐시에 남아 있으면, session_id 기반 재개(GET /{session_id})와 프로젝트
# 최근 회의 재개(GET /project/{id}/latest) 둘 다 사용자를 그 구버전 후보 화면으로 돌려보내야
# 했다. 이제는 두 조회 API가 problem_definition이 비어 있는 candidate_generation/
# awaiting_candidate_selection 세션을 "없는 세션"(404)처럼 취급해, 프론트가 새
# problem_discovery 세션을 자동으로 시작하게 한다 — 반대로 problem_definition을 거쳐
# 정상적으로 같은 phase에 도달한 신규 세션은 그대로 재개돼야 한다(오탐 방지).
# ---------------------------------------------------------------------------


def _seed_session(session_id: str, *, phase: str, problem_definition, project_id: str = "proj-legacy-test"):
    from app.api.routes.meetings import GUEST_USER_EMAIL

    state = {
        "session_id": session_id,
        "ideation_mode": "discovery",
        "phase": phase,
        "problem_definition": problem_definition,
        "idea_candidates": [{"title": "AI 기반 민원 처리 자동화 시스템"}],
        "messages": [],
        "round": 1,
        "max_rounds": 3,
        "notice_and_criteria": {},
        "user_idea": {},
        "consensus": [],
        "unresolved_issues": [],
    }
    conv_route._store.create(state, use_rag=False, project_id=project_id, user_email=GUEST_USER_EMAIL)
    return state


def test_get_session_hides_legacy_pre_problem_stage_discovery_session(client: TestClient):
    _seed_session("LEGACY-CANDIDATE-SESSION", phase="awaiting_candidate_selection", problem_definition=None)
    resp = client.get("/ideation-conversation/LEGACY-CANDIDATE-SESSION")
    assert resp.status_code == 404


def test_get_session_still_resumes_new_pipeline_session_at_same_phase(client: TestClient):
    _seed_session(
        "REAL-CANDIDATE-SESSION",
        phase="awaiting_candidate_selection",
        problem_definition={"problem": "실제 문제 정의"},
    )
    resp = client.get("/ideation-conversation/REAL-CANDIDATE-SESSION")
    assert resp.status_code == 200
    assert resp.json()["phase"] == "awaiting_candidate_selection"


def test_get_latest_project_conversation_hides_legacy_session(client: TestClient, monkeypatch):
    _seed_session(
        "LEGACY-LATEST-SESSION", phase="candidate_generation", problem_definition=None, project_id="proj-legacy-latest"
    )

    async def fake_find_latest(project_id, user_email):
        return {"session_id": "LEGACY-LATEST-SESSION"}

    monkeypatch.setattr(conv_route._session_repo, "find_latest_by_project", fake_find_latest)
    resp = client.get("/ideation-conversation/project/proj-legacy-latest/latest")
    assert resp.status_code == 404


def test_get_latest_project_conversation_still_resumes_new_pipeline_session(client: TestClient, monkeypatch):
    _seed_session(
        "REAL-LATEST-SESSION",
        phase="awaiting_candidate_selection",
        problem_definition={"problem": "실제 문제 정의"},
        project_id="proj-real-latest",
    )

    async def fake_find_latest(project_id, user_email):
        return {"session_id": "REAL-LATEST-SESSION"}

    monkeypatch.setattr(conv_route._session_repo, "find_latest_by_project", fake_find_latest)
    resp = client.get("/ideation-conversation/project/proj-real-latest/latest")
    assert resp.status_code == 200
    assert resp.json()["phase"] == "awaiting_candidate_selection"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
