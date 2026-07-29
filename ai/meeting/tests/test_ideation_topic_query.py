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
    _deterministic_final_direction_prefix,
    _deterministic_session_state_answer,
    _idea_core_summary,
    _topic_query,
    classify_query_type,
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


def test_resolve_effective_issue_anchors_expert_analysis_query_to_current_question():
    """용준/Claude(2026-07-29, 실측 버그 수정) — 실 사이트에서 expert_analysis_query
    ("MVP 구현 시 가장 큰 기술 위험 3개를 데이터/개인정보/음성 기능에 한정해서 답해달라")
    응답에 이전 턴들이 남긴 unresolved_issues 누적 문구("'None'은 문서 근거만으로
    확정할 수 없어...", "'로드맵'은...")가 그대로 노출됐다. active_issue_id가 비어 있어도
    expert_analysis_query는 canonical 주제 로테이션이나 unresolved_issues 누적 문구가 아니라
    이번 사용자 질문 원문 자체를 쟁점으로 삼아야 한다."""
    state = _state_without_active_issue()
    state["request_type"] = "expert_analysis_query"
    state["current_user_input"] = (
        "현재 확정된 아이디어만 기준으로, MVP 구현 시 가장 큰 기술 위험 3개와 각각의 대응 "
        "방법을 제시해 주세요. 데이터 확보, 개인정보 보호, 음성·쉬운 문장 기능 구현 "
        "가능성에 한정해서 답해 주세요."
    )
    state["unresolved_issues"] = [
        "planning_expert: 'None'은 문서 근거만으로 확정할 수 없어 전문가 판단(가정)으로 진행합니다 — ...",
        "dev_expert: '로드맵'은 문서 근거만으로 확정할 수 없어 전문가 판단(가정)으로 진행합니다 — ...",
    ]
    issue = resolve_effective_issue(state, "planning_expert")
    assert issue["source"] == "current_user_input"
    assert "None" not in issue["title"]
    assert "로드맵" not in issue["title"]
    assert issue["title"] == state["current_user_input"]


def test_resolve_effective_issue_ignores_current_user_input_for_other_request_types():
    """expert_analysis_query가 아니면 기존 unresolved_issues 우선순위 동작이 그대로
    유지돼야 한다(회귀 없음 — 위 새 분기는 expert_analysis_query에만 적용)."""
    state = _state_without_active_issue()
    state["current_user_input"] = "아무 질문"
    state["unresolved_issues"] = ["시민 알림의 실행 가능성"]
    issue = resolve_effective_issue(state, "planning_expert")
    assert issue["source"] == "unresolved_issues"
    assert issue["title"] == "시민 알림의 실행 가능성"


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


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-29, 요청: RAG 품질 개선 3단계 — dual-query 방식으로 전환) — 2단계에서는
# 키워드 목록에 매칭되는 질문만 topic_query에서 쟁점/역할 관점을 뺐는데, 목록에 없는 표현은
# 여전히 희석됐다(B02 실측). 이제 topic_query는 항상 "idea_summary | 현재 쟁점 | 검토 관점"
# 형식을 유지하고(사용자 질문 원문이 항상 맨 앞), 희석 문제는 RAG 쪽(
# ai/rag/orchestration/ideation_evidence_service.py)이 " | " 앞부분만 별도로 plain 검색해
# 역할 확장 검색과 rank fusion으로 합치는 dual-query 방식으로 해결한다 — 여기서는 topic_query
# 조립 자체(순서·형식)만 검증한다.
# ---------------------------------------------------------------------------


def _state_with_description_only(description: str, *, issue_id="problem", issue_title="문제 정의"):
    state = initial_conv_state(
        session_id="S1",
        notice_and_criteria={"competition_name": "테스트 공모전"},
        user_idea={"description": description},
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


def test_admin_fact_question_still_includes_issue_and_role_focus():
    """3단계에서는 키워드 기반 예외 처리를 없앴다 — 행정 사실 질문도 다른 질문과 동일하게
    쟁점/역할 관점이 붙는다(희석 문제는 RAG 쪽 dual-query가 담당)."""
    state = _state_with_description_only("이 공모전에서 결격 사유에는 어떤 것들이 있나요?")
    query = _topic_query(state, "dev_expert")
    assert query.startswith("이 공모전에서 결격 사유에는 어떤 것들이 있나요?")
    assert "현재 쟁점: 문제 정의" in query
    assert "검토 관점" in query


def test_topic_query_always_puts_raw_question_first():
    """RAG 쪽이 topic_query.split(" | ", 1)[0]로 사용자 질문 원문만 뽑아 별도 plain 검색에
    쓴다(dual-query Query A) — 이 계약이 깨지지 않는지 검증한다."""
    state = _state_with_description_only("참가 자격 요건이 어떻게 되나요?")
    query = _topic_query(state, "dev_expert")
    assert query.split(" | ", 1)[0] == "참가 자격 요건이 어떻게 되나요?"


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-29, 요청: RAG 품질 개선 3단계 — 정확한 상태값은 코드가 결정적으로
# 출력) — "최종 확정한 아이디어가 뭔가요" 질문에서 LLM이 재구성하다 명칭을 바꾸거나 검토
# 중인 후보를 확정안으로 착각하는 문제(B05 회귀)를 코드 레벨에서 막는다.
# ---------------------------------------------------------------------------


def test_deterministic_prefix_none_for_non_final_direction_query():
    state = _state_with_description_only("결격 사유가 뭔가요?")
    assert _deterministic_final_direction_prefix(state, "결격 사유가 뭔가요?") is None


def test_deterministic_prefix_uses_exact_title_when_locked():
    state = _state_with_description_only("최종 확정한 아이디어가 뭔가요?")
    state["idea_locked"] = True
    state["selected_idea"] = {"title": "상대적 기준 검토 방식", "problem": "P"}
    prefix = _deterministic_final_direction_prefix(state, "최종 확정한 아이디어가 뭔가요?")
    assert prefix == "사용자가 최종 확정한 아이디어는 '상대적 기준 검토 방식'입니다."


def test_deterministic_prefix_says_not_confirmed_when_idea_not_locked():
    state = _state_with_description_only("최종 확정한 아이디어가 뭔가요?")
    state["idea_locked"] = False
    state["selected_idea"] = None
    prefix = _deterministic_final_direction_prefix(state, "최종 확정한 아이디어가 뭔가요?")
    assert prefix == "아직 최종 확정된 아이디어가 없습니다."


def test_deterministic_prefix_says_not_confirmed_when_locked_but_title_missing():
    """idea_locked=True인데 title이 비어 있으면(데이터 이상) 빈 문자열을 정답인 것처럼
    내보내지 않고 "없다"고 말한다 — 값을 지어내지 않는다."""
    state = _state_with_description_only("최종 확정한 아이디어가 뭔가요?")
    state["idea_locked"] = True
    state["selected_idea"] = {"title": "", "problem": "P"}
    prefix = _deterministic_final_direction_prefix(state, "최종 확정한 아이디어가 뭔가요?")
    assert prefix == "아직 최종 확정된 아이디어가 없습니다."


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-29, 요청: 생성 품질 개선 2단계 — session_state_query 일반화 +
# 질문 유형 라우팅). _deterministic_final_direction_prefix는 문구·동작 그대로 유지되므로
# (위 4개 테스트) 여기서는 새로 추가한 idea_locked/current_phase/provisional 분기와
# classify_query_type만 검증한다.
# ---------------------------------------------------------------------------


def test_session_state_answer_falls_back_to_final_direction_first():
    state = _state_with_description_only("최종 확정한 아이디어가 뭔가요?")
    state["idea_locked"] = True
    state["selected_idea"] = {"title": "제목", "problem": "P"}
    assert (
        _deterministic_session_state_answer(state, "최종 확정한 아이디어가 뭔가요?")
        == "사용자가 최종 확정한 아이디어는 '제목'입니다."
    )


def test_session_state_answer_idea_locked_query():
    state = _state_with_description_only("아이디어가 확정됐나요?")
    state["idea_locked"] = False
    assert _deterministic_session_state_answer(state, "아이디어가 확정됐나요?") == "아이디어가 아직 최종 확정되지 않았습니다."
    state["idea_locked"] = True
    assert _deterministic_session_state_answer(state, "아이디어가 확정됐나요?") == "아이디어가 최종 확정되어 잠겼습니다."


def test_session_state_answer_current_phase_query():
    state = _state_with_description_only("지금 어떤 단계인가요?")
    state["phase"] = "idea_validation"
    assert _deterministic_session_state_answer(state, "지금 어떤 단계인가요?") == "현재 회의는 '아이디어 검증' 단계입니다."


def test_session_state_answer_provisional_idea_query():
    state = _state_with_description_only("지금 검토 중인 아이디어가 뭔가요?")
    state["provisional_idea"] = {"title": "잠정안"}
    assert (
        _deterministic_session_state_answer(state, "지금 검토 중인 아이디어가 뭔가요?")
        == "현재 검증 중인 잠정 후보는 '잠정안'입니다."
    )
    state["provisional_idea"] = None
    assert (
        _deterministic_session_state_answer(state, "지금 검토 중인 아이디어가 뭔가요?")
        == "현재 검증 중인 잠정 후보가 없습니다."
    )


def test_session_state_answer_none_for_unrelated_query():
    state = _state_with_description_only("결격 사유가 뭔가요?")
    assert _deterministic_session_state_answer(state, "결격 사유가 뭔가요?") is None


def test_classify_query_type_document_fact():
    assert classify_query_type("이 공모전 참가 자격 요건이 어떻게 되나요?") == "document_fact_query"
    assert classify_query_type("제출 서류는 뭐가 필요한가요?") == "document_fact_query"


def test_classify_query_type_session_state():
    assert classify_query_type("최종 확정한 아이디어가 뭔가요?") == "session_state_query"
    assert classify_query_type("아이디어가 확정됐나요?") == "session_state_query"


def test_classify_query_type_expert_analysis():
    assert classify_query_type("이 기능의 구현 가능성이 어때요?") == "expert_analysis_query"
    assert classify_query_type("데이터 확보 방안과 기술 위험을 개선 제안 해주세요") == "expert_analysis_query"


def test_classify_query_type_fallback_for_ambiguous_text():
    """용준/Claude(2026-07-29, 요청: 4단계 — fallback 명명) — None 대신 명시적인
    ideation_discussion_request를 반환한다."""
    assert classify_query_type("안녕하세요") == "ideation_discussion_request"


def test_classify_query_type_ideation_discussion_request_for_discussion_ask():
    """아이디어를 자유롭게 토론해 달라는 요청은 특정 유형(사실/상태/전문가판단) 신호가
    없어 ideation_discussion_request(기존 일반 discussion 경로)로 분류된다."""
    assert classify_query_type("이 아이디어에 대해 자유롭게 토론해 주세요") == "ideation_discussion_request"


def test_classify_query_type_document_fact_boosted_by_criteria_evidence():
    """키워드 신호가 약해도(임계값 미만) evidence의 document_role 구성이 criteria 위주면
    document_fact_query 쪽으로 점수가 보정된다(요청: "키워드 몇 개만 하드코딩하지 말고
    구조화 분류")."""
    evidence = [
        {"document_role": "criteria", "text": "심사 기준 안내"},
        {"document_role": "criteria", "text": "제출 안내"},
        {"document_role": "target", "text": "아이디어 설명"},
    ]
    # "요건" 하나만으로는 임계값(3) 미만이라 evidence 보정이 없으면 fallback이어야 한다.
    assert classify_query_type("요건이 뭔가요", evidence=None) == "ideation_discussion_request"
    assert classify_query_type("요건이 뭔가요", evidence=evidence) == "document_fact_query"


# ---------------------------------------------------------------------------
# 용준/Claude(2026-07-29, 요청: 4단계 — B02 회귀 수정 검증 + 전체 검증 매트릭스). 실제
# 6-케이스 평가셋 원문 질의를 그대로 사용해 분류가 요청한 매트릭스와 일치하는지 확인한다.
# ---------------------------------------------------------------------------


def test_classify_query_type_verification_matrix():
    cases = {
        # A01/A03/A04/B01/B02 원문 그대로 — document_fact_query여야 한다.
        "실증·PoC 부문과 우수사례 부문의 평가 항목별 배점은 어떻게 다른가요?": "document_fact_query",
        "이 공모전에 컨소시엄으로 참여할 때 참여기관 요건은 어떻게 되나요?": "document_fact_query",
        "실증·PoC 수행보고서에서 데이터 관련 항목은 어떤 내용을 작성해야 하나요?": "document_fact_query",
        "이 공모전에서 결격 사유에는 어떤 것들이 있나요?": "document_fact_query",
        "1차 서면심사에서는 몇 개 과제가 선정되나요?": "document_fact_query",
        # B05
        "이 회의에서 사용자가 최종적으로 확정한 해결 방향은 무엇인가요?": "session_state_query",
        # 구현 가능성·MVP 위험 질문
        "이 기능의 구현 가능성과 MVP 단계 기술 위험이 어느 정도인가요?": "expert_analysis_query",
        # 자유 토론 요청
        "이 아이디어를 자유롭게 토론해 주세요": "ideation_discussion_request",
    }
    for query, expected in cases.items():
        assert classify_query_type(query) == expected, f"{query!r} -> expected {expected!r}"
