# 작성자: 경이
# 목적: 점수 설명 카드(RPT-006) 검증 — score_result의 항목별 점수를 rubric·위원 근거·
#       감점과 연결하고, 설명 문구/총점 산식이 계산값과 정확히 일치(LLM 불일치 방지)하는지,
#       결정론적으로 동일한지 확인한다.
# import: 표준 라이브러리 json/sys/pathlib, pytest; ai/meeting/scoring 패키지.

import copy
import json
import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from scoring import build_score_explanation, calculate_score  # noqa: E402

FIXTURE = MEETING_DIR / "tests" / "fixtures" / "final_meeting_result.v2.json"


def _load_data() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["data"]


def test_explanation_total_formula_matches_score_result():
    data = _load_data()
    explanation = build_score_explanation(
        data["score_result"], data["rubric"], data["reviewer_results"]
    )
    total = explanation["total"]
    assert total["total_score"] == 61
    assert total["criteria_sum"] == 61
    assert total["penalty_sum"] == 0
    assert total["formula"] == "61 - 0 = 61"
    assert total["calculation_method"] == "criterion_owner"


def test_explanation_links_each_criterion_to_reviewer_basis():
    data = _load_data()
    explanation = build_score_explanation(
        data["score_result"], data["rubric"], data["reviewer_results"]
    )
    cards = {c["criterion_id"]: c for c in explanation["criteria"]}

    # rubric 4개 항목 모두 카드가 있어야 한다
    assert set(cards) == {"marketability", "technical_feasibility", "differentiation", "execution_plan"}

    mk = cards["marketability"]
    assert mk["criterion_name"] == "시장성"
    assert mk["raw_score"] == 20
    assert mk["max_score"] == 30
    assert mk["aggregation"] == "single"
    assert mk["scored_by"] == ["REV-MOCK-001"]
    # 위원 근거(약점)가 카드에 연결되어야 한다
    assert mk["reviewer_scores"][0]["issues"]
    # 설명 문구에 계산값이 그대로 들어가야 한다(LLM 불일치 방지)
    assert "30점 만점에 20점" in mk["basis"]


def test_explanation_reflects_missing_required_penalty():
    """필수항목(marketability)을 아무도 채점하지 않으면, 카드에 감점 사유와 문구가 나타난다."""
    data = _load_data()
    reviewers = copy.deepcopy(data["reviewer_results"])
    for r in reviewers:
        r["rubric_scores"] = [s for s in r["rubric_scores"] if s["criterion_id"] != "marketability"]

    score_result = calculate_score(data["rubric"], reviewers, missing_required_penalty=5)
    explanation = build_score_explanation(score_result, data["rubric"], reviewers)
    cards = {c["criterion_id"]: c for c in explanation["criteria"]}

    mk = cards["marketability"]
    assert mk["aggregation"] == "none"
    assert mk["penalty"] == 5
    assert mk["penalty_reasons"] and mk["penalty_reasons"][0]["type"] == "missing_required"
    assert "채점한 위원이 없어" in mk["basis"]
    assert "감점 5점" in mk["basis"]


def test_explanation_is_deterministic():
    data = _load_data()
    e1 = build_score_explanation(data["score_result"], data["rubric"], data["reviewer_results"])
    e2 = build_score_explanation(data["score_result"], data["rubric"], data["reviewer_results"])
    assert e1 == e2


def test_explanation_discloses_deterministic_score_cap():
    rubric = {
        "rubric_id": "R-CAP",
        "total_max_score": 20,
        "criteria": [
            {
                "criterion_id": "data_utilization",
                "criterion_name": "데이터 활용성",
                "max_score": 20,
                "required": True,
            }
        ],
    }
    reviewers = [
        {
            "review_id": "REV-CAP",
            "rubric_scores": [{"criterion_id": "data_utilization", "score": 15}],
        }
    ]
    submission = {
        "text": (
            "다양한 공공데이터를 활용할 예정이다. 구체적인 출처는 나중에 정할 계획이다. "
            "데이터를 활용해 추천 서비스를 만들 것이다. 세부적인 가공 방법도 추후 마련할 예정이다."
        )
    }
    score_result = calculate_score(rubric, reviewers, submission=submission)
    explanation = build_score_explanation(score_result, rubric, reviewers)
    card = explanation["criteria"][0]

    assert card["raw_score"] == 2
    assert card["calibration"]["original_score"] == 15
    assert "위원 제안 점수 15점" in card["basis"]
    assert "문서 근거 상한 2점" in card["basis"]
