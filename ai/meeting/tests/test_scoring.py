# 작성자: 경이
# 목적: 점수 엔진(ai/meeting/scoring) 검증 — mock 재현, 결정론(동일입력=동일출력, TST-002 기반),
#       필수항목 누락 감점(MTG-003) 동작 확인.
# import: 표준 라이브러리 copy/json/sys/pathlib, pytest; ai/meeting/scoring 패키지.

import copy
import json
import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from scoring import calculate_score  # noqa: E402

FIXTURE = MEETING_DIR / "tests" / "fixtures" / "final_meeting_result.v2.json"

CALIBRATION_RUBRIC = {
    "rubric_id": "R-CALIBRATION",
    "total_max_score": 100,
    "criteria": [
        {"criterion_id": "ai_innovation", "criterion_name": "AI 혁신성", "max_score": 25, "required": True},
        {
            "criterion_id": "data_utilization",
            "criterion_name": "데이터 활용성",
            "max_score": 20,
            "required": True,
        },
        {"criterion_id": "feasibility", "criterion_name": "실현 가능성", "max_score": 20, "required": True},
        {
            "criterion_id": "creativity_differentiation",
            "criterion_name": "창의성·차별성",
            "max_score": 15,
            "required": True,
        },
        {"criterion_id": "expected_effect", "criterion_name": "기대 효과성", "max_score": 20, "required": True},
    ],
}


def _calibration_reviewers(scores: list[float | int]) -> list[dict]:
    return [
        {
            "review_id": "CALIBRATION-REVIEWER",
            "rubric_scores": [
                {"criterion_id": criterion["criterion_id"], "score": score}
                for criterion, score in zip(CALIBRATION_RUBRIC["criteria"], scores, strict=True)
            ],
        }
    ]


def _load_data() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["data"]


def test_reproduces_mock_score_result():
    """rubric + reviewer_results 로 계산한 score_result 가 mock 의 값과 정확히 일치한다."""
    data = _load_data()
    result = calculate_score(data["rubric"], data["reviewer_results"])
    assert result == data["score_result"]


def test_deterministic_same_input_same_output():
    """동일 입력을 두 번 계산하면 완전히 동일한 결과가 나온다(MTG-003 검수 기준)."""
    data = _load_data()
    r1 = calculate_score(data["rubric"], data["reviewer_results"])
    r2 = calculate_score(data["rubric"], data["reviewer_results"])
    assert r1 == r2
    assert r1["total_score"] == 61


def test_missing_required_criterion_penalty():
    """필수 항목(marketability)을 아무도 채점하지 않으면 raw 0 + 누락 감점 표식이 생긴다."""
    data = _load_data()
    reviewers = copy.deepcopy(data["reviewer_results"])
    for r in reviewers:
        r["rubric_scores"] = [s for s in r["rubric_scores"] if s["criterion_id"] != "marketability"]

    result = calculate_score(data["rubric"], reviewers)

    mk = next(b for b in result["breakdown"] if b["criterion_id"] == "marketability")
    assert mk["raw_score"] == 0
    assert mk["source_review_ids"] == []
    assert any(
        p["type"] == "missing_required" and p["criterion_id"] == "marketability"
        for p in result["penalties"]
    )
    # 61 - marketability(20) = 41
    assert result["total_score"] == 41


def test_non_required_missing_has_no_penalty():
    """비필수 항목(differentiation) 누락은 감점 표식을 만들지 않는다."""
    data = _load_data()
    reviewers = copy.deepcopy(data["reviewer_results"])
    for r in reviewers:
        r["rubric_scores"] = [s for s in r["rubric_scores"] if s["criterion_id"] != "differentiation"]

    result = calculate_score(data["rubric"], reviewers)

    assert result["penalties"] == []
    # 61 - differentiation(11) = 50
    assert result["total_score"] == 50


def test_multiple_reviewers_same_criterion_average():
    """같은 기준을 두 위원이 채점하면 평균으로 집계하고 source_review_ids 를 모두 기록한다."""
    rubric = {
        "rubric_id": "R1",
        "total_max_score": 100,
        "criteria": [{"criterion_id": "c1", "criterion_name": "기준1", "max_score": 100, "required": True}],
    }
    reviewers = [
        {"review_id": "A", "rubric_scores": [{"criterion_id": "c1", "score": 80}]},
        {"review_id": "B", "rubric_scores": [{"criterion_id": "c1", "score": 70}]},
    ]
    result = calculate_score(rubric, reviewers)
    assert result["total_score"] == 75
    assert result["breakdown"][0]["source_review_ids"] == ["A", "B"]


def test_low_quality_document_is_capped_to_expected_band():
    """출처·수치·방법 없이 예정만 반복한 문서는 LLM이 만점을 줘도 30점 이하로 보정된다."""
    vague_passage = (
        "고용 문제를 해결하는 서비스를 만들 예정이다. "
        "최신 AI 기술을 활용하여 정확한 추천이 가능할 것이다. "
        "고용노동 관련 다양한 공공데이터를 활용할 예정이다. "
        "사용자에게 맞춤형 정보를 제공할 계획이다. "
        "기존 서비스와 차별화된 기능을 구축할 예정이다. "
        "사회적으로 긍정적인 효과가 기대된다. "
        "구체적인 출처와 구현 절차는 향후 정할 예정이다. "
    )
    submission = {
        "document_name": "파일명은_채점근거가_아님_50점대테스트.docx",
        # 실제 A유형처럼 분량은 길지만 같은 추상 문장을 반복하는 경우를 재현한다.
        "text": vague_passage * 20,
    }
    result = calculate_score(
        CALIBRATION_RUBRIC,
        _calibration_reviewers([25, 20, 20, 15, 20]),
        submission=submission,
    )
    breakdown = {item["criterion_id"]: item for item in result["breakdown"]}

    assert result["calculation_version"] == "score_v3"
    assert 25 <= result["total_score"] <= 31
    assert result["total_score"] == sum(item["raw_score"] for item in result["breakdown"])
    assert breakdown["data_utilization"]["raw_score"] == 5
    assert {s["code"] for s in breakdown["data_utilization"]["calibration"]["signals"]} >= {
        "S1",
        "S2",
        "S3",
        "MULTI",
    }
    assert breakdown["ai_innovation"]["raw_score"] == 6.25
    assert "S4" in {
        signal["code"] for signal in breakdown["ai_innovation"]["calibration"]["signals"]
    }


def test_mid_quality_document_keeps_scores_when_caps_do_not_apply():
    """출처·방법·수치가 있는 중간 문서는 보정기가 정상 점수를 임의로 깎지 않는다."""
    passage = (
        "HRD-Net과 워크넷, 고용보험 데이터를 API로 수집한다. "
        "훈련 과정과 채용 정보를 정규화하는 2단계 파이프라인을 구성한다. "
        "AI 모델로 유사 이력을 찾아 추천하되 세부 필드 설계는 보완이 필요하다. "
        "기존 서비스보다 데이터를 함께 비교한다는 차이가 있다. "
        "시범 운영 후 취업 성과를 측정한다. "
    )
    submission = {
        "document_name": "65점대테스트.docx",
        "text": passage * 30,
    }
    expected = [15, 13, 12, 10, 13]
    result = calculate_score(
        CALIBRATION_RUBRIC,
        _calibration_reviewers(expected),
        submission=submission,
    )

    assert result["total_score"] == 63
    assert [item["raw_score"] for item in result["breakdown"]] == expected
    assert all("calibration" not in item for item in result["breakdown"])


def test_high_quality_document_is_not_reduced():
    """출처·필드·정량값·방법이 풍부한 우수 문서는 기존 80점 앵커를 유지한다."""
    passage = (
        "HRD-Net 훈련과정별 수료율·취업률, 워크넷 직무별 구인배율, 고용보험 통계를 결합한다. "
        "국가기술자격과 내일배움카드의 직무 필드를 KECO 코드로 정규화한다. "
        "임베딩 검색으로 후보 Top 5를 뽑고 LLM이 재정렬하는 2단계 AI 파이프라인을 사용한다. "
        "수료생 5명 미만 표본은 제외하고 구인배율 편차가 3~4배인 직무를 별도로 표시한다. "
        "API 수집, 데이터베이스 적재, 모델 추론, 품질 검증의 구현 단계를 정의했다. "
    )
    submission = {
        "document_name": "A유형.docx",
        "text": passage * 25,
    }
    expected = [20, 18, 15, 12, 15]
    result = calculate_score(
        CALIBRATION_RUBRIC,
        _calibration_reviewers(expected),
        submission=submission,
    )

    assert result["total_score"] == 80
    assert [item["raw_score"] for item in result["breakdown"]] == expected
    assert all("calibration" not in item for item in result["breakdown"])


def test_topic_only_document_uses_bottom_anchor():
    """거의 백지인 실제 문서는 placeholder와 달리 최하단 10~20점 밴드로 제한한다."""
    submission = {"document_name": "주제만.docx", "text": "AI로 취업 문제를 해결하는 서비스입니다."}
    result = calculate_score(
        CALIBRATION_RUBRIC,
        _calibration_reviewers([25, 20, 20, 15, 20]),
        submission=submission,
    )

    assert result["total_score"] == 10
    assert all(
        any(signal["code"] == "S0" for signal in item["calibration"]["signals"])
        for item in result["breakdown"]
    )


def test_required_keywords_cap_only_activates_when_rubric_declares_them():
    """S5는 공고문에서 필수 키워드 메타데이터를 준 동적 rubric에만 적용한다."""
    rubric = {
        "rubric_id": "R-REQUIRED",
        "total_max_score": 20,
        "criteria": [
            {
                "criterion_id": "required_feature",
                "criterion_name": "필수 기능",
                "max_score": 20,
                "required": True,
                "required_keywords": ["안전성 검증"],
            }
        ],
    }
    passage = (
            "한국고용정보원 데이터를 API로 수집하고 2단계 파이프라인으로 정규화한다. "
            "사용자 100명을 대상으로 효과를 측정하며 이미 시범 구현을 완료했다."
            "운영 환경에서는 데이터베이스 적재와 모델 추론 결과를 매주 확인한다. "
            "품질 지표와 장애 대응 절차도 수치로 기록해 구현 가능성을 검증했다."
    )
    submission = {"text": passage * 8}
    result = calculate_score(
        rubric,
        [{"review_id": "A", "rubric_scores": [{"criterion_id": "required_feature", "score": 18}]}],
        submission=submission,
    )

    assert result["total_score"] == 10
    assert result["breakdown"][0]["calibration"]["signals"] == [
        {"code": "S5", "reason": "공고문에서 지정한 필수 요소가 제출문서에 없음"}
    ]


def test_portal_name_and_contest_ordinal_are_not_evidence():
    """공공데이터포털과 제목의 '제5회'만으로 출처·정량 근거 상한이 풀리지 않는다."""
    text = (
        "제5회 공공데이터 AI 공모전 아이디어 제안서\n"
        "2) 데이터 활용 방안\n"
        "공공데이터포털에서 다양한 데이터를 찾아 활용할 예정이다. "
        "구체적인 데이터셋과 필드는 추후 정할 계획이다."
    )
    result = calculate_score(
        CALIBRATION_RUBRIC,
        _calibration_reviewers([25, 20, 20, 15, 20]),
        submission={"text": text},
    )
    data = next(b for b in result["breakdown"] if b["criterion_id"] == "data_utilization")
    codes = {signal["code"] for signal in data["calibration"]["signals"]}

    assert {"S1", "S2"} <= codes
    assert data["raw_score"] <= 5


def test_short_specific_document_ranks_below_long_vague_document():
    """실측 AA(짧고 항목별 깊이 부족)는 긴 A(추상적)보다 낮은 상한을 갖는다."""
    long_vague = (
        "1) 제안 배경\n취업 문제를 해결하는 AI 서비스를 만들 예정이다. "
        "2) 데이터 활용 방안\n다양한 공공데이터를 활용하며 세부 출처는 추후 정한다. "
        "3) AI 활용 방안\nAI가 알아서 추천하고 방법은 향후 정한다. "
        "4) 상세 설명\n일정과 자원은 앞으로 계획한다. "
        "5) 창의성\n차별화 기능을 추가할 예정이다. "
        "6) 기대효과\n많은 도움이 될 것으로 기대된다. "
    ) * 20
    short_specific = (
        "1) 제안 배경\n취업 훈련 정보의 분산 문제를 해결한다. "
        "2) 데이터 활용 방안\nHRD-Net과 워크넷의 취업률 필드를 사용한다. "
        "3) AI 활용 방안\n생성형 AI가 추천 이유를 설명한다. "
        "4) 상세 설명\n이력 입력 후 결과를 보여준다. "
        "5) 창의성\n검색 대신 개인별 우선순위를 제시한다. "
        "6) 기대효과\n훈련 선택 편의를 높인다. "
    ) * 4
    reviewers = _calibration_reviewers([25, 20, 20, 15, 20])

    vague_result = calculate_score(
        CALIBRATION_RUBRIC, reviewers, submission={"text": long_vague}
    )
    short_result = calculate_score(
        CALIBRATION_RUBRIC, reviewers, submission={"text": short_specific}
    )

    assert short_result["total_score"] <= 20
    assert short_result["total_score"] < vague_result["total_score"]
