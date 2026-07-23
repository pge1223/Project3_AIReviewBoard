# 작성자: 용준/Claude(2026-07-23, 요청: "웹 E2E 전 4가지 보완")
# 목적: (1) 발언 상한으로 쟁점이 강제 종료된 뒤 진행자 노드가 active_issue_id/open_issues를
#       실제로 다음 공식 쟁점으로 갱신하는지, (2) 진행자 프롬프트가 로컬로 갱신된
#       open_issues/resolved_issues(closed_reason/resolution_kind 포함)와 다음 쟁점 힌트를
#       받는지, (3) resolve_canonical_issue_family가 포괄적인 단일 키워드보다 구체적인
#       표현에 더 높은 가중치를 줘서 problem/data, problem/mvp, core_value/differentiation
#       충돌을 올바르게 해소하는지, (4) decision_options 라벨이 화자 이름("기획 위원
#       제안")이 아니라 실제 발언 내용에서 뽑은 짧은 제목이고, 텍스트 선택지 계약(진행자
#       질문 문자열에 2~3개 선택지가 온전히 표시됨)이 지켜지는지 검증한다.
# import: 표준 라이브러리 json/sys/pathlib; ai/meeting/graph 패키지.

from __future__ import annotations

import json
import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph.ideation_conv_nodes import (  # noqa: E402
    MAX_EXPERT_TURNS_PER_ISSUE,
    extract_distinct_alternatives,
    make_discussion_facilitator_node,
    resolve_canonical_issue_family,
    _compose_decision_question,
)
from graph.ideation_conv_state import TOPIC_PRIORITY  # noqa: E402


# ---------------------------------------------------------------------------
# 1) + 2) 로테이션 강제 반영 — active_issue_id/open_issues/resolved_issues/prompt
# ---------------------------------------------------------------------------


class _FacilitatorLLM:
    """진행자 구조화 응답만 반환하는 stub. 실제로 전달된 프롬프트를 검사할 수 있도록
    last_prompt에 저장한다."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def __call__(self, prompt: str) -> str:
        self.last_prompt = prompt
        return json.dumps(
            {
                "agreements": [],
                "disagreements": [],
                "facilitator_summary": "이번 쟁점은 발언 상한으로 정리하고 다음 쟁점으로 넘어갑니다.",
                "spoken_text": "다음 쟁점으로 넘어가겠습니다.",
                "needs_user_decision": False,
                "user_question": None,
            },
            ensure_ascii=False,
        )


def _facilitator_state(*, active_issue_id, open_issues, resolved_issues) -> dict:
    return {
        "session_id": "ROTATE-TEST",
        "phase": "expert_discussion",
        "round": 1,
        "max_rounds": 5,
        "messages": [
            {
                "message_id": "MSG-LAST",
                "speaker_id": "dev_expert",
                "structured": {"needs_user_input": False},
            }
        ],
        "consensus": [],
        "unresolved_issues": [],
        "notice_and_criteria": {},
        "active_issue_id": active_issue_id,
        "open_issues": open_issues,
        "resolved_issues": resolved_issues,
        "expert_turn_count": 1,
        "resolved_topics": [],
        "llm_calls_used": 0,
    }


def _capped_data_issue() -> dict:
    return {
        "issue_id": "issue_data_1",
        "title": "데이터 확보 방안",
        "family": "data",
        "status": "open",
        "turns": MAX_EXPERT_TURNS_PER_ISSUE,
        "planning_position": "기획 의견",
        "development_position": "개발 의견",
    }


def test_facilitator_forces_rotation_into_active_issue_id_and_open_issues():
    """테스트 1 핵심 — next_issue_family를 로그로만 남기지 않고, active_issue_id와
    open_issues를 실제로 다음 공식 쟁점(problem — data보다 TOPIC_PRIORITY 우선순위가 높고
    아직 안 다룬 축)으로 갱신해야 한다."""
    llm = _FacilitatorLLM()
    node = make_discussion_facilitator_node(llm)
    state = _facilitator_state(
        active_issue_id="issue_data_1", open_issues=[_capped_data_issue()], resolved_issues=[]
    )

    update = node(state)

    assert update["active_issue_id"] == "topic_problem"
    new_open = update["open_issues"]
    assert any(
        issue["issue_id"] == "topic_problem" and issue["family"] == "problem" and issue["status"] == "open"
        for issue in new_open
    )
    assert all(issue["issue_id"] != "issue_data_1" for issue in new_open)


def test_facilitator_reuses_existing_open_issue_of_next_family_instead_of_duplicating():
    """다음 공식 family에 해당하는 쟁점이 이미 open_issues에 있다면(다른 전문가가 먼저
    열어둔 경우), 새 topic_<family> 레코드를 또 만들지 않고 그 issue_id를 재사용해야
    한다."""
    llm = _FacilitatorLLM()
    node = make_discussion_facilitator_node(llm)
    already_open_mvp = {
        "issue_id": "issue_mvp_custom",
        "title": "MVP 범위 재확인",
        "family": "mvp",
        "status": "open",
        "turns": 1,
    }
    state = _facilitator_state(
        active_issue_id="issue_data_1",
        open_issues=[_capped_data_issue(), already_open_mvp],
        resolved_issues=[
            {"issue_id": f"issue_{t}", "title": t, "family": t, "status": "resolved", "resolution_kind": "agreed_resolution"}
            for t in ("problem", "target_user", "core_value", "contest_fit", "differentiation")
        ],
    )

    update = node(state)

    assert update["active_issue_id"] == "issue_mvp_custom"
    mvp_records = [issue for issue in update["open_issues"] if issue.get("family") == "mvp"]
    assert len(mvp_records) == 1


def test_facilitator_clears_active_issue_when_no_official_topics_remain():
    """모든 공식 평가축이 이미 resolved면 로테이션할 다음 쟁점이 없다 — active_issue_id는
    None이어야 하고(더 이상 강제로 잇지 않음) 회의는 사용자 결정을 기다려야 한다."""
    llm = _FacilitatorLLM()
    node = make_discussion_facilitator_node(llm)
    resolved_all_but_data = [
        {"issue_id": f"issue_{t}", "title": t, "family": t, "status": "resolved", "resolution_kind": "agreed_resolution"}
        for t in TOPIC_PRIORITY
        if t != "data"
    ]
    state = _facilitator_state(
        active_issue_id="issue_data_1",
        open_issues=[_capped_data_issue()],
        resolved_issues=resolved_all_but_data,
    )

    update = node(state)

    assert update["active_issue_id"] is None
    assert update.get("next_route") != "continue_round"


def test_facilitator_prompt_receives_local_resolved_issues_with_closed_reason_and_next_hint():
    """테스트 2 핵심 — 진행자 프롬프트는 이번 턴에 로컬로 갱신된 resolved_issues(방금
    강제 종료된 쟁점의 closed_reason/resolution_kind 포함)와 다음 공식 쟁점 힌트를 받아야
    한다. 갱신 전 state.get("resolved_issues")(빈 배열)를 그대로 넘기면 이 정보가
    누락된다."""
    llm = _FacilitatorLLM()
    node = make_discussion_facilitator_node(llm)
    state = _facilitator_state(
        active_issue_id="issue_data_1", open_issues=[_capped_data_issue()], resolved_issues=[]
    )

    node(state)

    prompt = llm.last_prompt
    assert prompt is not None
    assert "max_issue_turns_reached" in prompt
    assert "parked_expert_judgment" in prompt
    assert "issue_data_1" in prompt
    # next_issue_hint(사람이 읽는 다음 쟁점 제목)도 프롬프트에 그대로 드러나야 한다.
    assert "문제 정의" in prompt


# ---------------------------------------------------------------------------
# 3) canonical family 다중 키워드 충돌 — 첫 매칭이 아니라 가중치 합으로 판정
# ---------------------------------------------------------------------------


def test_resolve_canonical_issue_family_prefers_specific_data_keywords_over_generic_problem_word():
    """"문제"라는 포괄적 단일 단어가 먼저 걸려도, "실시간 데이터"/"데이터 수집"처럼 구체적인
    표현이 더 많으면 problem이 아니라 data로 분류돼야 한다."""
    family = resolve_canonical_issue_family("문제 해결을 위한 실시간 데이터 수집 방안")
    assert family == "data"


def test_resolve_canonical_issue_family_problem_mvp_conflict_resolves_by_weight():
    text = "문제 해결을 위해 MVP 핵심 기능부터 우선순위 기능 범위를 정해야 한다"
    assert resolve_canonical_issue_family(text) == "mvp"


def test_resolve_canonical_issue_family_core_value_differentiation_conflict_resolves_by_weight():
    text = "핵심 가치 제안이 명확한 반면 차별점은 아직 불명확하다"
    assert resolve_canonical_issue_family(text) == "core_value"


def test_resolve_canonical_issue_family_ties_break_by_topic_priority():
    """같은 가중치로 두 family가 동점이면 TOPIC_PRIORITY 순서(더 앞선 축)로 정한다 —
    target_user가 differentiation보다 앞선다."""
    text = "타겟과 차별 요소를 함께 검토"
    assert resolve_canonical_issue_family(text) == "target_user"


def test_resolve_canonical_issue_family_prefers_official_issue_id_over_text_guess():
    """공식 issue_id("topic_<family>")가 있으면, 텍스트가 다른 family를 강하게 시사해도
    issue_id의 family를 그대로 쓴다."""
    family = resolve_canonical_issue_family("문제 정의가 시급합니다", issue_id="topic_data")
    assert family == "data"


# ---------------------------------------------------------------------------
# 4) decision_options — 화자 이름이 아니라 실제 내용 기반 라벨, 텍스트 선택지 계약
# ---------------------------------------------------------------------------


def _alternative_messages() -> list[dict]:
    return [
        {
            "speaker_id": "planning_expert",
            "structured": {
                "active_issue_id": "issue_data_1",
                "proposal": "공공 API 중심으로 데이터를 확보하는 방향을 제안합니다.",
            },
        },
        {
            "speaker_id": "dev_expert",
            "structured": {
                "active_issue_id": "issue_data_1",
                "proposal": "민간 데이터 제휴를 통해 실시간성을 확보하는 방향이 낫습니다.",
            },
        },
    ]


def test_decision_options_labels_are_content_based_not_speaker_role_names():
    """테스트 4 핵심 — 선택지 라벨이 "기획 위원 제안"/"개발 위원 제안"이 아니라 실제 제안
    내용에서 추출한 짧은 제목이어야 한다."""
    alternatives = extract_distinct_alternatives(
        messages=_alternative_messages(), active_issue_id="issue_data_1"
    )
    assert len(alternatives) == 2

    _question, options, _default = _compose_decision_question(
        topic="product_direction_choice",
        issue_title="데이터 확보 방안",
        missing_information=[],
        distinct_alternatives=alternatives,
    )

    assert len(options) == 2
    for option in options:
        assert "기획 위원 제안" not in option["label"]
        assert "개발 위원 제안" not in option["label"]
    # 실제 제안 원문의 핵심 단어가 라벨에 남아 있어야 한다(축약이라도 화자 이름 대체가
    # 아니라 내용 기반이어야 함을 보장).
    assert any("api" in options[0]["label"].lower() or "API" in options[0]["label"] for _ in [0])
    assert "공공" in options[0]["label"] or "api" in options[0]["label"].lower()
    assert "민간" in options[1]["label"] or "데이터" in options[1]["label"]


def test_decision_options_detail_shows_raw_proposal_text_not_fabricated_pros_cons():
    """장단점이 실제 발언에 없으면 임의로 만들지 않고 detail에 제안 원문만 담아야 한다."""
    alternatives = extract_distinct_alternatives(
        messages=_alternative_messages(), active_issue_id="issue_data_1"
    )
    _question, options, _default = _compose_decision_question(
        topic="product_direction_choice",
        issue_title="데이터 확보 방안",
        missing_information=[],
        distinct_alternatives=alternatives,
    )
    assert options[0]["detail"] == "공공 API 중심으로 데이터를 확보하는 방향을 제안합니다."
    assert options[1]["detail"] == "민간 데이터 제휴를 통해 실시간성을 확보하는 방향이 낫습니다."


def test_decision_question_text_displays_two_or_more_real_alternatives_fully():
    """텍스트 선택지 계약(이번 패치 범위: 옵션 A) — 프론트 버튼 없이도, 진행자
    content/pending_question으로 쓰이는 question 문자열 자체에 2~3개의 실제 선택지가
    온전히(라벨+detail 전부) 표시돼야 한다."""
    alternatives = extract_distinct_alternatives(
        messages=_alternative_messages(), active_issue_id="issue_data_1"
    )
    question, options, default_label = _compose_decision_question(
        topic="product_direction_choice",
        issue_title="데이터 확보 방안",
        missing_information=[],
        distinct_alternatives=alternatives,
    )
    assert 2 <= len(options) <= 3
    for option in options:
        assert option["label"] in question
        assert option["detail"] in question
    assert default_label in question
