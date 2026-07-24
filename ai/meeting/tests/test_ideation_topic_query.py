# 작성자: 용준/Claude(2026-07-23, 요청: RAG 근거 실제 활용 강화 — query 품질 개선)
# 목적: ideation_conv_nodes.py::_topic_query가 (1) 후보 dict 전체를 이어붙이던 이전 방식과
#       달리 핵심 필드만 요약하고, (2) 현재 active_issue를 실제로 반영하며, (3) persona_id에
#       따라 기획/개발 검색어가 달라지는지 확인한다. 이 세 가지가 실측 문제(linked_evidence_
#       count=0, evidence_status="expert_judgment_only")의 근본 원인이었다 — 검색 자체는
#       "성공"으로 보여도 검색어가 지금 쟁점과 무관하게 광범위해서 근거의 관련성 검증을
#       통과하지 못했다.

import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting

sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_conv_nodes import (  # noqa: E402
    _active_issue_title,
    _idea_core_summary,
    _topic_query,
    resolve_effective_issue,
    resolve_retrieval_issue,
)
from graph.ideation_conv_state import initial_conv_state  # noqa: E402


def _state_with_issue(*, issue_id="differentiation", issue_title="차별성과 고객 가치"):
    state = initial_conv_state(
        session_id="S1",
        notice_and_criteria={"competition_name": "테스트 공모전"},
        user_idea={
            "title": "AI 기반 환경 모니터링 플랫폼",
            "problem": "도시 대기오염과 환경 문제를 실시간으로 파악하고 대응하기 어렵다",
            "target_user": "지자체 담당자",
            "solution": "IoT 센서로 대기질을 실시간 수집하고 AI로 분석해 시민에게 알린다",
            "main_features": ["IoT 센서 기반 데이터 수집", "실시간 대기질 모니터링", "시민 알림", "AI 분석", "데이터 시각화"],
            "tech_approach": "MQTT 기반 IoT 게이트웨이 + 시계열 DB",
            "mvp": "핵심 지역 3곳 시범 운영",
        },
    )
    state["active_issue_id"] = issue_id
    state["open_issues"] = [
        {
            "issue_id": issue_id,
            "title": issue_title,
            "status": "open",
            "planning_position": None,
            "development_position": None,
            "resolution": None,
            "turns": 0,
        }
    ]
    return state


def test_idea_core_summary_uses_only_title_problem_solution_not_every_field():
    """이전에는 user_idea dict의 모든 값(주요 기능·기술 접근·MVP까지)을 이어붙였다 — 이제는
    title/problem/solution 세 필드만 요약해, 검색어가 무관한 필드로 희석되지 않는다."""
    idea = {
        "title": "T",
        "problem": "P",
        "target_user": "U",
        "solution": "S",
        "main_features": ["a", "b", "c"],
        "tech_approach": "X",
        "mvp": "M",
    }
    summary = _idea_core_summary(idea)
    assert "T" in summary and "P" in summary and "S" in summary
    assert "a" not in summary and "X" not in summary and "M" not in summary


def test_topic_query_reflects_active_issue_title_not_just_idea_dump():
    state = _state_with_issue(issue_id="differentiation", issue_title="차별성과 고객 가치")
    query = _topic_query(state, "planning_expert")
    assert "차별성과 고객 가치" in query


def test_topic_query_changes_when_active_issue_changes():
    state_a = _state_with_issue(issue_id="differentiation", issue_title="차별성과 고객 가치")
    state_b = _state_with_issue(issue_id="data_freshness", issue_title="데이터 갱신 주기")
    query_a = _topic_query(state_a, "planning_expert")
    query_b = _topic_query(state_b, "planning_expert")
    assert query_a != query_b
    assert "데이터 갱신 주기" in query_b
    assert "데이터 갱신 주기" not in query_a


def test_user_interjection_overrides_existing_issue_and_leads_retrieval_query():
    state = _state_with_issue(issue_id="problem", issue_title="문제 정의")
    question = "유지보수 문제를 어떻게 해결해야 하지?"
    state["messages"].append(
        {
            "message_id": "MSG-user-maintenance",
            "speaker_id": "user",
            "speaker_name": "사용자",
            "role": "사용자",
            "round": 1,
            "message_type": "interjection",
            "content": question,
            "referenced_message_ids": [],
            "evidence": [],
            "created_at": "2026-07-23T00:00:00+00:00",
            "structured": {"target_speaker_id": "dev_expert", "active_issue_id": "problem"},
        }
    )
    state["interjection_target_speaker_id"] = "dev_expert"
    state["required_counterpart_speaker_id"] = "planning_expert"
    state["counterpart_review_completed"] = False

    effective = resolve_effective_issue(state, "dev_expert")
    query = _topic_query(state, "dev_expert")

    assert effective["issue_id"] == "problem"
    assert effective["title"] == question
    assert effective["source"] == "user_interjection"
    assert query.startswith(f"사용자 직접 질문: {question}")
    assert f"현재 쟁점: {question}" in query


def test_topic_query_differs_between_planning_and_dev_for_same_state():
    """같은 state(같은 아이디어, 같은 active_issue)라도 persona_id가 다르면 검색어의 역할별
    검토 관점 부분이 달라야 한다 — 두 역할이 사실상 같은 broad query를 받아 순서만 다른
    결과를 받던 문제를 막는다."""
    state = _state_with_issue()
    planning_query = _topic_query(state, "planning_expert")
    dev_query = _topic_query(state, "dev_expert")
    assert planning_query != dev_query
    assert "차별성" in planning_query or "심사 기준" in planning_query
    assert "데이터" in dev_query or "구현 가능성" in dev_query


def test_topic_query_without_persona_id_has_no_role_focus_backward_compatible():
    """persona_id를 넘기지 않으면(구버전 호출부) 역할별 관점 없이 이슈/아이디어 요약만
    반환한다 — 기존 동작과 호환된다."""
    state = _state_with_issue()
    query = _topic_query(state)
    assert "검토 관점" not in query


def test_active_issue_title_falls_back_to_issue_id_when_no_record_yet():
    state = _state_with_issue()
    state["active_issue_id"] = "hallucination_risk"
    state["open_issues"] = []
    state["resolved_issues"] = []
    assert _active_issue_title(state) == "hallucination_risk"


def test_active_issue_title_none_when_no_active_issue():
    state = _state_with_issue()
    state["active_issue_id"] = None
    assert _active_issue_title(state) is None


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-23, 요청: 첫 전문가 턴에도 실제 검토 쟁점 반영 / 역할별 target query
# 필드 개선) — planning의 첫 발언 시점에는 active_issue_id가 아직 None이다(쟁점은 그 발언이
# 끝난 뒤에야 열린다). resolve_retrieval_issue가 그 순간에도 구체적인 검토 주제를 고르는지,
# 그리고 idea 요약이 역할별로 다른 필드를 쓰는지 확인한다.
# ---------------------------------------------------------------------------


def _state_without_active_issue():
    state = _state_with_issue()
    state["active_issue_id"] = None
    state["open_issues"] = []
    state["resolved_issues"] = []
    state["unresolved_issues"] = []
    state["resolved_topics"] = []
    return state


def test_resolve_retrieval_issue_uses_active_issue_title_when_present():
    state = _state_with_issue(issue_id="differentiation", issue_title="차별성과 고객 가치")
    assert resolve_retrieval_issue(state, "planning_expert") == "차별성과 고객 가치"


def test_resolve_retrieval_issue_without_active_issue_still_returns_specific_topic_for_first_turn():
    """active_issue_id가 아직 없는 planning 첫 턴에도 빈 문자열이나 아이디어 요약뿐인
    검색어가 아니라, 구체적인 검토 주제(TOPIC_PRIORITY 첫 항목)가 나와야 한다."""
    state = _state_without_active_issue()
    topic = resolve_retrieval_issue(state, "planning_expert")
    assert topic
    assert topic != ""
    query = _topic_query(state, "planning_expert")
    assert f"현재 쟁점: {topic}" in query


def test_resolve_retrieval_issue_prefers_unresolved_issues_over_topic_priority():
    state = _state_without_active_issue()
    state["unresolved_issues"] = ["시민 알림의 실행 가능성"]
    assert resolve_retrieval_issue(state, "planning_expert") == "시민 알림의 실행 가능성"


def test_resolve_retrieval_issue_skips_resolved_topics():
    state = _state_without_active_issue()
    state["resolved_topics"] = ["problem", "target_user", "core_value", "contest_fit"]
    # TOPIC_PRIORITY 순서상 다음은 "differentiation" -> "차별성과 고객 가치"여야 한다.
    assert resolve_retrieval_issue(state, "planning_expert") == "차별성과 고객 가치"


def test_resolve_retrieval_issue_skips_resolved_issue_titles_too():
    """resolved_topics(질문 흐름)뿐 아니라 resolved_issues(토론 흐름)에 이미 있는 제목도
    다시 검색 주제로 고르지 않는다."""
    state = _state_without_active_issue()
    state["resolved_issues"] = [
        {
            "issue_id": "problem",
            "title": "문제 정의",
            "status": "resolved",
            "planning_position": None,
            "development_position": None,
            "resolution": "해결됨",
            "turns": 2,
        }
    ]
    topic = resolve_retrieval_issue(state, "planning_expert")
    assert topic != "문제 정의"


def test_resolve_retrieval_issue_falls_back_to_role_default_when_topics_exhausted():
    state = _state_without_active_issue()
    state["resolved_topics"] = [
        "problem", "target_user", "core_value", "contest_fit", "differentiation",
        "mvp", "data", "ai_role", "roadmap",
    ]
    assert resolve_retrieval_issue(state, "planning_expert") == "차별성과 고객 가치"
    assert resolve_retrieval_issue(state, "dev_expert") == "기술 구현 가능성"


def test_dev_first_turn_query_uses_issue_opened_by_planning():
    """planning이 쟁점을 이미 열어(active_issue_id 세팅) 놓았으면, 뒤이은 dev 검색은 그
    쟁점을 그대로 이어받는다(planning이 스스로 새 쟁점을 여는 게 아니라)."""
    state = _state_with_issue(issue_id="feasibility", issue_title="기술 실현 가능성")
    dev_topic = resolve_retrieval_issue(state, "dev_expert")
    assert dev_topic == "기술 실현 가능성"


def test_idea_core_summary_planning_includes_target_user_and_differentiation():
    idea = {
        "title": "T",
        "problem": "P",
        "target_user": "지자체 담당자",
        "solution": "S",
        "differentiation": "기존 서비스 대비 실시간 예측 제공",
        "main_features": ["a", "b", "c"],
        "technical_approach": "MQTT",
    }
    summary = _idea_core_summary(idea, "planning_expert")
    assert "지자체 담당자" in summary
    assert "기존 서비스 대비 실시간 예측 제공" in summary
    assert "MQTT" not in summary  # dev 전용 필드는 planning 요약에 섞이지 않는다.


def test_idea_core_summary_dev_includes_technical_fields_not_planning_only_fields():
    idea = {
        "title": "T",
        "problem": "P",
        "target_user": "지자체 담당자",
        "solution": "S",
        "differentiation": "기존 서비스 대비 실시간 예측 제공",
        "main_features": ["IoT 센서 수집", "실시간 모니터링", "AI 분석", "시각화", "알림"],
        "required_data": "대기질 센서 데이터",
        "technical_approach": "MQTT 기반 IoT 게이트웨이",
        "mvp_scope": "핵심 지역 3곳",
        "risks": ["센서 오작동", "네트워크 단절"],
    }
    summary = _idea_core_summary(idea, "dev_expert")
    assert "MQTT 기반 IoT 게이트웨이" in summary
    assert "대기질 센서 데이터" in summary
    assert "센서 오작동" in summary
    assert "지자체 담당자" not in summary  # planning 전용 필드는 dev 요약에 섞이지 않는다.


def test_idea_core_summary_list_field_capped_to_a_few_items():
    idea = {"title": "T", "solution": "S", "main_features": ["f1", "f2", "f3", "f4", "f5"]}
    summary = _idea_core_summary(idea, "dev_expert")
    assert "f1" in summary and "f2" in summary and "f3" in summary
    assert "f4" not in summary and "f5" not in summary


def test_idea_core_summary_empty_fields_are_skipped_without_duplication():
    idea = {"title": "T", "problem": "", "target_user": None, "solution": "T"}
    summary = _idea_core_summary(idea, "planning_expert")
    # "T"가 title과 solution에 중복으로 들어 있어도 한 번만 포함된다.
    assert summary.count("T") == 1
