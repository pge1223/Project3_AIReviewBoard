# 작성자: 용준/Claude(2026-07-23, 요청: "동일 쟁점 표현 변경 반복 루프" 수정)
# 목적: 실측 세션(IDEA-CONV-754a75ab)에서 재현된 "쟁점 발언 상한 도달 -> 진행자가 문장만
#       바꾼 같은 쟁점을 새 issue_id로 다시 여는" 무한 루프를 결정론적 코드가 막는지
#       검증한다. (1) normalize_issue_text/resolve_canonical_issue_family/
#       resolve_issue_duplicate/_select_next_issue_family 순수 함수 단위 검증, (2)
#       make_conv_discussion_node를 통한 실제 중복 억제 통합 검증, (3) 발언 상한 이후
#       실제로 서로 다른 공식 평가축(TOPIC_PRIORITY)으로 로테이션하고 회의가 정상 종료되는지
#       전체 그래프 실행으로 검증, (4) classify_user_decision_topic/
#       extract_distinct_alternatives — near_issue_cap만으로 사용자 질문을 만들지 않고
#       실제 서로 다른 대안이 있을 때만 실제 발언 내용으로 질문을 구성하는지 검증한다.
# import: 표준 라이브러리 json/sys/pathlib; ai/meeting/graph 패키지.

from __future__ import annotations

import json
import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph import start_ideation_conversation  # noqa: E402
from graph.ideation_conv_nodes import (  # noqa: E402
    classify_user_decision_topic,
    extract_distinct_alternatives,
    is_duplicate_issue,
    normalize_issue_text,
    resolve_canonical_issue_family,
    resolve_issue_duplicate,
    _compose_decision_question,
    _select_next_issue_family,
)
from graph.ideation_conv_state import TOPIC_PRIORITY  # noqa: E402

NOTICE_AND_CRITERIA = {
    "competition_name": "IT 공공서비스 공모전",
    "notice_document": "실현가능성, 공공성을 평가한다.",
}
USER_IDEA = {
    "description": "실시간 교통 데이터를 활용해 대중교통 혼잡도를 예측하는 서비스를 만들고 싶습니다."
}

# 실제 재현 세션(IDEA-CONV-754a75ab)에서 관찰된, 사실상 같은 쟁점(데이터 확보)의 서로 다른
# 문장 표현 4개.
DATA_ISSUE_PHRASINGS = [
    "정확한 실시간 데이터 확보의 가능성",
    "추가적으로 필요한 데이터 소스",
    "데이터 소스의 구체적인 확보 방법을 명시할 필요가 있다",
    "API 및 외부 데이터 제공자 연계 방안",
]


def test_automatic_issue_rotation_skips_contest_fit():
    next_family = _select_next_issue_family(
        excluded_family="core_value",
        open_issues=[],
        resolved_issues=[
            {"issue_id": "topic_problem", "title": "문제 정의", "family": "problem"},
            {"issue_id": "topic_target_user", "title": "목표 사용자", "family": "target_user"},
            {"issue_id": "topic_core_value", "title": "핵심 가치", "family": "core_value"},
        ],
        resolved_topics=["problem", "target_user", "core_value"],
    )
    assert next_family == "differentiation"


# ---------------------------------------------------------------------------
# 1. 순수 함수 단위 테스트 — normalize_issue_text / resolve_canonical_issue_family
# ---------------------------------------------------------------------------


def test_normalize_issue_text_strips_particles_and_generic_words():
    a = normalize_issue_text("데이터 소스의 구체적인 확보 방법을 명시할 필요가 있다")
    b = normalize_issue_text("데이터 소스 확보")
    # 조사·일반 표현("구체적", "방법", "필요", "확보")을 지운 뒤 "데이터"/"소스"가 남아야
    # 두 표현 모두에 공통으로 포함된다.
    assert "데이터" in a
    assert "소스" in a
    assert a and b


def test_resolve_canonical_issue_family_maps_all_data_phrasings_to_same_family():
    """테스트 A 핵심 — 실제 재현 세션의 4가지 표현이 모두 같은 canonical family(data)로
    매핑돼야 한다."""
    families = {resolve_canonical_issue_family(text) for text in DATA_ISSUE_PHRASINGS}
    assert families == {"data"}


def test_resolve_canonical_issue_family_keeps_genuinely_different_topics_distinct():
    """테스트 D 핵심 — 실제로 다른 쟁점(문제 정의/차별성/기술 구현 가능성/MVP 범위/
    개인정보·보안 리스크)은 절대 같은 family로 합쳐지면 안 된다."""
    titles = ["문제 정의", "차별성", "기술 구현 가능성", "MVP 범위", "개인정보·보안 리스크"]
    families = [resolve_canonical_issue_family(title) for title in titles]
    assert len(set(families)) == len(titles)
    assert families[0] == "problem"
    assert families[1] == "differentiation"
    assert families[3] == "mvp"


def test_is_duplicate_issue_requires_non_empty_matching_family():
    assert is_duplicate_issue("data", "data") is True
    assert is_duplicate_issue("data", "mvp") is False
    assert is_duplicate_issue(None, None) is False
    assert is_duplicate_issue("", "") is False


# ---------------------------------------------------------------------------
# 2. 순수 함수 단위 테스트 — resolve_issue_duplicate / _select_next_issue_family
# ---------------------------------------------------------------------------


def test_resolve_issue_duplicate_reuses_active_issue_under_different_wording():
    """같은 쟁점이 지금 active인 상태에서 표현만 바뀐 새 후보가 오면, 새 레코드를 만들지
    않고 기존 active 쟁점 id를 그대로 재사용해야 한다."""
    open_issues = [
        {"issue_id": "issue_data_1", "title": DATA_ISSUE_PHRASINGS[0], "family": "data", "status": "open"}
    ]
    result = resolve_issue_duplicate(
        candidate_issue_id="issue_data_2",
        candidate_issue_title=DATA_ISSUE_PHRASINGS[1],
        current_active_issue_id="issue_data_1",
        open_issues=open_issues,
        resolved_issues=[],
    )
    assert result["duplicate"] is True
    assert result["duplicate_source"] == "active"
    assert result["issue_id"] == "issue_data_1"
    assert result["rotated"] is False


def test_resolve_issue_duplicate_merges_into_other_open_issue_by_family():
    """다른(현재 active는 아닌) open 쟁점과 family가 같으면 그 쟁점 id로 합쳐야 한다."""
    open_issues = [
        {"issue_id": "issue_problem_1", "title": "문제 정의", "family": "problem", "status": "open"},
        {"issue_id": "issue_data_1", "title": DATA_ISSUE_PHRASINGS[0], "family": "data", "status": "open"},
    ]
    result = resolve_issue_duplicate(
        candidate_issue_id="issue_data_2",
        candidate_issue_title=DATA_ISSUE_PHRASINGS[1],
        current_active_issue_id="issue_problem_1",
        open_issues=open_issues,
        resolved_issues=[],
    )
    assert result["duplicate"] is True
    assert result["duplicate_source"] == "open"
    assert result["issue_id"] == "issue_data_1"
    assert result["rotated"] is False


def test_resolve_issue_duplicate_suppresses_reopening_parked_family_and_rotates():
    """테스트 B/C 핵심 — 발언 상한으로 강제 종료(parked_expert_judgment)된 family가 새
    표현으로 다시 제안되면, 그 family를 재등록하지 않고 아직 안 다룬 다른 공식 평가축으로
    로테이션해야 한다."""
    resolved_issues = [
        {
            "issue_id": "issue_data_1",
            "title": DATA_ISSUE_PHRASINGS[0],
            "family": "data",
            "status": "resolved",
            "resolution_kind": "parked_expert_judgment",
        }
    ]
    result = resolve_issue_duplicate(
        candidate_issue_id="issue_data_new",
        candidate_issue_title=DATA_ISSUE_PHRASINGS[2],
        current_active_issue_id=None,
        open_issues=[],
        resolved_issues=resolved_issues,
        resolved_topics=[],
    )
    assert result["duplicate"] is True
    assert result["duplicate_source"] == "parked"
    assert result["rotated"] is True
    assert result["canonical_family"] != "data"
    # TOPIC_PRIORITY에서 "data"보다 우선순위가 높고 아직 다루지 않은 첫 번째 공식 축이어야
    # 한다("data"는 problem/target_user/core_value/contest_fit/differentiation/mvp 다음이다).
    assert result["canonical_family"] == "problem"
    assert result["issue_id"] == "topic_problem"


def test_resolve_issue_duplicate_distinguishes_consensus_resolved_from_parked():
    """강제 종료가 아니라 실제 합의로 끝난 쟁점(agreed_resolution)도 재등록하지 않되,
    duplicate_source가 "resolved"로 구분돼야 한다(강제 종료와 합의 완료 구분)."""
    resolved_issues = [
        {
            "issue_id": "issue_problem_1",
            "title": "문제 정의",
            "family": "problem",
            "status": "resolved",
            "resolution_kind": "agreed_resolution",
        }
    ]
    result = resolve_issue_duplicate(
        candidate_issue_id="issue_problem_2",
        candidate_issue_title="문제를 다시 정의할 필요가 있다",
        current_active_issue_id=None,
        open_issues=[],
        resolved_issues=resolved_issues,
        resolved_topics=[],
    )
    assert result["duplicate"] is True
    assert result["duplicate_source"] == "resolved"


def test_resolve_issue_duplicate_accepts_genuinely_new_family():
    """실제로 새로운 쟁점(기존 open/resolved 어디와도 family가 겹치지 않음)은 그대로
    받아들여야 한다(테스트 D — 다른 쟁점을 잘못 억제하면 안 됨)."""
    result = resolve_issue_duplicate(
        candidate_issue_id="issue_mvp_1",
        candidate_issue_title="MVP 범위",
        current_active_issue_id=None,
        open_issues=[{"issue_id": "issue_problem_1", "title": "문제 정의", "family": "problem", "status": "open"}],
        resolved_issues=[],
    )
    assert result["duplicate"] is False
    assert result["issue_id"] == "issue_mvp_1"


def test_select_next_issue_family_prefers_open_issue_over_topic_priority():
    open_issues = [{"issue_id": "issue_x", "title": "MVP 범위", "family": "mvp"}]
    next_family = _select_next_issue_family(
        excluded_family="data", open_issues=open_issues, resolved_issues=[], resolved_topics=[]
    )
    assert next_family == "mvp"


def test_select_next_issue_family_returns_none_when_all_topics_exhausted():
    """테스트 E 핵심 — TOPIC_PRIORITY 9개 공식 축이 전부 resolved_issues에 있으면 더 이상
    로테이션할 대상이 없어야 한다(회의를 정리해야 한다는 신호)."""
    resolved_issues = [
        {"issue_id": f"issue_{topic}", "title": topic, "family": topic, "status": "resolved"}
        for topic in TOPIC_PRIORITY
    ]
    next_family = _select_next_issue_family(
        excluded_family=None, open_issues=[], resolved_issues=resolved_issues, resolved_topics=[]
    )
    assert next_family is None


# ---------------------------------------------------------------------------
# 3. 실제 전체 회의 그래프 실행 — 표현만 바뀐 같은 쟁점이 반복돼도 무한 루프 없이
#    다른 공식 평가축으로 넘어가고 phase="failed"로 끝나지 않는지 검증한다.
# ---------------------------------------------------------------------------


def _persona(prompt: str) -> str:
    if "당신은 AI Review Board의 기획 전문가입니다" in prompt:
        return "planning_expert"
    if "당신은 AI Review Board의 개발 전문가입니다" in prompt:
        return "dev_expert"
    return "ideation_facilitator"


def _facilitator_payload() -> dict:
    return {
        "agreements": [],
        "disagreements": ["데이터 확보 방안에 대한 이견이 계속되고 있습니다"],
        "facilitator_summary": "데이터 확보 방안에 대해 계속 논의가 이어졌습니다.",
        "spoken_text": "데이터 확보 방안에 대해 계속 논의가 이어졌습니다.",
        "needs_user_decision": False,
        "user_question": None,
    }


def _canvas_payload() -> dict:
    return {
        "problem": "문제 정의",
        "target_user": "목표 사용자",
        "core_value": "핵심 가치",
        "solution": "해결 방안",
        "differentiation": "차별점",
        "contest_fit": "공모전 적합성",
        "feasibility": "medium",
        "risks": [],
    }


class _RepeatedRewordedIssueLLM:
    """같은 쟁점(데이터 확보)을 문장만 바꿔가며 계속 반박하는 stub — 실제 재현 세션의
    핵심 증상(문장만 바뀐 같은 쟁점이 새 issue_id로 계속 재생성됨)을 재현한다. 진행자가
    강제 종료해도 다음 발언에서 또 같은 의미를 다른 표현으로 제안한다."""

    def __init__(self) -> None:
        self.expert_call_count = 0

    def __call__(self, prompt: str) -> str:
        if "[의견 규칙]" in prompt:
            speaker = _persona(prompt)
            counterpart = "dev_expert" if speaker == "planning_expert" else "planning_expert"
            phrasing = DATA_ISSUE_PHRASINGS[self.expert_call_count % len(DATA_ISSUE_PHRASINGS)]
            self.expert_call_count += 1
            return json.dumps(
                {
                    "stance": "반박",
                    "spoken_text": f"[{speaker}] {phrasing}",
                    "judgment": "판단",
                    "reason": "근거",
                    "suggestion": "제안",
                    "interim_conclusion": "임시 결론",
                    "responding_to": "상대 발언",
                    "agreement": "",
                    "concern": "우려",
                    "confirmed": [],
                    "unconfirmed": [],
                    "referenced_message_ids": [],
                    "evidence": [],
                    "next_action": None,
                    "active_issue_id": f"issue_data_{self.expert_call_count}",
                    "active_issue_title": phrasing,
                    "new_information": [phrasing],
                    "proposal": phrasing,
                    "changed_position": False,
                    "needs_counterpart_response": True,
                    "recommended_next_speaker": counterpart,
                    "issue_resolved": False,
                    "needs_user_input": False,
                    "user_question": None,
                },
                ensure_ascii=False,
            )
        if "[진행자 정리 규칙]" in prompt:
            return json.dumps(_facilitator_payload(), ensure_ascii=False)
        if "[캔버스 갱신 규칙]" in prompt:
            return json.dumps(_canvas_payload(), ensure_ascii=False)
        raise AssertionError(f"예상하지 못한 프롬프트: {prompt[:150]}")


def test_reworded_duplicate_issue_does_not_explode_open_issue_records():
    """테스트 A+E 통합 — 표현만 바뀐 같은 쟁점이 계속 제안돼도 open_issues/resolved_issues에
    "data" family 레코드가 여러 개 쌓이지 않고, 회의가 phase="failed"로 끝나지 않는다."""
    llm = _RepeatedRewordedIssueLLM()
    state = start_ideation_conversation(
        session_id="DEDUP-1",
        notice_and_criteria=NOTICE_AND_CRITERIA,
        user_idea=USER_IDEA,
        llm_call=llm,
        max_rounds=3,
    )
    assert state["phase"] != "failed"

    all_issues = list(state.get("open_issues") or []) + list(state.get("resolved_issues") or [])
    data_family_records = [
        issue
        for issue in all_issues
        if (issue.get("family") or resolve_canonical_issue_family(issue.get("title"))) == "data"
    ]
    # 실측 세션 증상 — 문장만 바뀐 같은 쟁점마다 새 레코드가 쌓이면 4개 이상이 된다. 중복
    # 억제가 동작하면 "data" family 레코드는 정확히 1개(최초 open -> 발언 상한으로 강제
    # 종료된 그 레코드)만 남아야 한다.
    assert len(data_family_records) == 1
    assert data_family_records[0]["status"] == "resolved"
    assert data_family_records[0].get("resolution_kind") == "parked_expert_judgment"
    assert data_family_records[0].get("closed_reason") == "max_issue_turns_reached"

    # 발언 상한 도달 뒤 다음 활성 쟁점(있다면)은 "data"가 아닌 다른 공식 평가축이어야 한다
    # (테스트 B — 표현 변경 버전으로 다시 이동하면 실패).
    active_issue_id = state.get("active_issue_id")
    if active_issue_id:
        active_record = next(
            issue for issue in (state.get("open_issues") or []) if issue["issue_id"] == active_issue_id
        )
        assert active_record.get("family") != "data"


# ---------------------------------------------------------------------------
# 4. 사용자 선택 게이트 — near_issue_cap만으로 product_direction_choice를 만들지 않고,
#    실제 서로 다른 대안이 있을 때만 실제 발언 내용으로 질문한다.
# ---------------------------------------------------------------------------


def test_classify_user_decision_topic_ignores_near_issue_cap_without_alternatives():
    """대안이 없으면 near_issue_cap이어도 product_direction_choice를 만들지 않는다."""
    topic = classify_user_decision_topic(
        missing_information=[], issue_title="데이터 확보 방안", near_issue_cap=True, distinct_alternatives=None
    )
    assert topic is None

    topic_empty = classify_user_decision_topic(
        missing_information=[], issue_title="데이터 확보 방안", near_issue_cap=True, distinct_alternatives=[]
    )
    assert topic_empty is None


def test_classify_user_decision_topic_requires_two_distinct_alternatives():
    """대안이 하나뿐이면 여전히 사용자에게 묻지 않는다."""
    one_alt = [{"speaker": "planning_expert", "text": "공공 API 중심으로 구축"}]
    assert (
        classify_user_decision_topic(
            missing_information=[], issue_title="데이터 확보 방안", near_issue_cap=True, distinct_alternatives=one_alt
        )
        is None
    )

    two_alts = one_alt + [{"speaker": "dev_expert", "text": "자체 센서로 직접 수집"}]
    assert (
        classify_user_decision_topic(
            missing_information=[], issue_title="데이터 확보 방안", near_issue_cap=True, distinct_alternatives=two_alts
        )
        == "product_direction_choice"
    )


def test_classify_user_decision_topic_generic_info_gap_is_not_a_decision_topic():
    """단순 정보 부족(일반 기술 질문)은 사용자 선택 질문으로 바뀌면 안 된다 — near_issue_cap이
    아니면 애초에 대안 개수와 무관하게 None이어야 한다."""
    topic = classify_user_decision_topic(
        missing_information=["센서 설치 방식이 구체적으로 문서에 없습니다"],
        issue_title="센서 연동",
        near_issue_cap=False,
        distinct_alternatives=None,
    )
    assert topic is None


def test_extract_distinct_alternatives_dedups_paraphrased_proposals():
    """같은 제안을 표현만 바꾼 경우 서로 다른 대안으로 중복 생성하지 않는다."""
    messages = [
        {
            "speaker_id": "planning_expert",
            "structured": {"active_issue_id": "issue_data", "proposal": "공공 API 중심으로 데이터를 확보합니다"},
        },
        {
            "speaker_id": "dev_expert",
            "structured": {"active_issue_id": "issue_data", "proposal": "공공 API를 중심으로 데이터 확보 필요"},
        },
    ]
    alternatives = extract_distinct_alternatives(messages=messages, active_issue_id="issue_data")
    assert len(alternatives) == 1


def test_extract_distinct_alternatives_keeps_real_distinct_proposals():
    """실제로 다른 3개 대안(공공 API / 민간 제휴 / 자체 센서)이 있으면 모두 보존해야 한다."""
    messages = [
        {
            "speaker_id": "planning_expert",
            "structured": {"active_issue_id": "issue_data", "proposal": "공공 API 중심으로 데이터를 확보합니다"},
        },
        {
            "speaker_id": "dev_expert",
            "structured": {"active_issue_id": "issue_data", "proposal": "민간 데이터 제휴로 정확도를 높입니다"},
        },
        {
            "speaker_id": "planning_expert",
            "structured": {"active_issue_id": "issue_data", "proposal": "자체 센서로 직접 수집하는 방안도 있습니다"},
        },
    ]
    alternatives = extract_distinct_alternatives(messages=messages, active_issue_id="issue_data")
    assert len(alternatives) == 3
    assert all("제안 A" not in alt["text"] and "제안 B" not in alt["text"] for alt in alternatives)


def test_compose_decision_question_uses_real_alternative_text_not_generic_labels():
    """decision_options에 실제 발언 내용이 담기고 "제안 A/B" 같은 범용 문구가 나오지 않는다."""
    alternatives = [
        {"speaker": "planning_expert", "text": "공공 API 중심으로 데이터를 확보합니다"},
        {"speaker": "dev_expert", "text": "자체 센서로 직접 수집하는 방안도 있습니다"},
    ]
    question, options, default_label = _compose_decision_question(
        topic="product_direction_choice",
        issue_title="데이터 확보 방안",
        missing_information=[],
        distinct_alternatives=alternatives,
    )
    assert len(options) == 2
    option_texts = " ".join(opt["detail"] for opt in options)
    assert "공공 API 중심으로 데이터를 확보합니다" in option_texts
    assert "자체 센서로 직접 수집하는 방안도 있습니다" in option_texts
    assert "제안 A" not in question and "제안 B" not in question
    assert default_label
    assert "선택해 주세요" in question
