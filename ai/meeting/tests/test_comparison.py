# 작성자: 경이
# 목적: 수정 전후 비교 리포트(RPT-004) 검증 — 항목별 점수 증감·해결/신규/잔존 지적
#       계산, 평가기준 변경 시 직접 비교 제한 처리를 확인한다.
# import: 표준 라이브러리 sys/pathlib; ai/meeting/scoring 패키지.

import sys
from pathlib import Path

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from scoring import build_revision_comparison, build_version_history  # noqa: E402


def _doc(meeting_id: str, criteria_meta: list[dict], breakdown: list[dict], reviewer_scores: list[dict], total: int) -> dict:
    """비교 함수가 읽는 필드만 담은 최소 v2 문서."""
    return {
        "meeting_id": meeting_id,
        "rubric": {"rubric_id": "R", "total_max_score": 100, "criteria": criteria_meta},
        "reviewer_results": [{"persona_id": "p1", "rubric_scores": reviewer_scores}],
        "score_result": {"total_score": total, "breakdown": breakdown},
    }


_CRIT_AB = [
    {"criterion_id": "A", "criterion_name": "항목A", "max_score": 50, "required": True},
    {"criterion_id": "B", "criterion_name": "항목B", "max_score": 50, "required": True},
]


def _before():
    return _doc(
        "MTG-BEFORE",
        _CRIT_AB,
        breakdown=[
            {"criterion_id": "A", "raw_score": 20, "max_score": 50},
            {"criterion_id": "B", "raw_score": 30, "max_score": 50},
        ],
        reviewer_scores=[
            {"criterion_id": "A", "judgment": "needs_improvement", "issues": ["A문제1", "A문제2"]},
            {"criterion_id": "B", "judgment": "acceptable", "issues": ["B문제1"]},
        ],
        total=50,
    )


def test_comparison_reports_score_delta_and_resolved_issues():
    before = _before()
    after = _doc(
        "MTG-AFTER",
        _CRIT_AB,
        breakdown=[
            {"criterion_id": "A", "raw_score": 40, "max_score": 50},
            {"criterion_id": "B", "raw_score": 30, "max_score": 50},
        ],
        reviewer_scores=[
            {"criterion_id": "A", "judgment": "strong", "issues": ["A문제2"]},
            {"criterion_id": "B", "judgment": "acceptable", "issues": ["B문제1", "B문제신규"]},
        ],
        total=70,
    )

    report = build_revision_comparison(before, after)

    assert report["rubric_changed"] is False
    assert report["direct_comparison_limited"] is False
    assert report["total"] == {"before": 50, "after": 70, "delta": 20}

    cards = {c["criterion_id"]: c for c in report["criteria"]}
    a = cards["A"]
    assert a["delta"] == 20
    assert a["before_judgment"] == "needs_improvement"
    assert a["after_judgment"] == "strong"
    assert a["resolved_issues"] == ["A문제1"]  # 해결됨
    assert a["persisting_issues"] == ["A문제2"]  # 잔존
    assert a["new_issues"] == []

    b = cards["B"]
    assert b["delta"] == 0
    assert b["resolved_issues"] == []
    assert b["new_issues"] == ["B문제신규"]


def test_comparison_flags_limited_when_rubric_criteria_changed():
    before = _before()
    # 평가기준 변경: B 삭제, C 추가
    after = _doc(
        "MTG-AFTER",
        [
            {"criterion_id": "A", "criterion_name": "항목A", "max_score": 50, "required": True},
            {"criterion_id": "C", "criterion_name": "항목C", "max_score": 50, "required": True},
        ],
        breakdown=[
            {"criterion_id": "A", "raw_score": 40, "max_score": 50},
            {"criterion_id": "C", "raw_score": 25, "max_score": 50},
        ],
        reviewer_scores=[
            {"criterion_id": "A", "judgment": "strong", "issues": []},
            {"criterion_id": "C", "judgment": "acceptable", "issues": ["C문제"]},
        ],
        total=65,
    )

    report = build_revision_comparison(before, after)

    assert report["rubric_changed"] is True
    assert report["direct_comparison_limited"] is True
    assert report["warnings"]
    assert [c["criterion_id"] for c in report["criteria"]] == ["A"]  # 공통 항목만 비교
    assert [c["criterion_id"] for c in report["added_criteria"]] == ["C"]
    assert [c["criterion_id"] for c in report["removed_criteria"]] == ["B"]


def test_comparison_flags_max_score_change_as_not_comparable():
    before = _before()
    # A 배점 50 -> 40 으로 변경
    after = _doc(
        "MTG-AFTER",
        [
            {"criterion_id": "A", "criterion_name": "항목A", "max_score": 40, "required": True},
            {"criterion_id": "B", "criterion_name": "항목B", "max_score": 50, "required": True},
        ],
        breakdown=[
            {"criterion_id": "A", "raw_score": 35, "max_score": 40},
            {"criterion_id": "B", "raw_score": 30, "max_score": 50},
        ],
        reviewer_scores=[
            {"criterion_id": "A", "judgment": "strong", "issues": []},
            {"criterion_id": "B", "judgment": "acceptable", "issues": ["B문제1"]},
        ],
        total=65,
    )

    report = build_revision_comparison(before, after)

    cards = {c["criterion_id"]: c for c in report["criteria"]}
    assert cards["A"]["comparable"] is False
    assert cards["A"]["max_score_changed"] is True
    assert cards["B"]["comparable"] is True
    assert report["rubric_changed"] is True
    assert any("배점" in w for w in report["warnings"])


def test_version_history_preserves_score_calibration_trace():
    document = _before()
    calibration = {
        "original_score": 35,
        "cap_score": 20,
        "cap_ratio": 0.4,
        "signals": [{"code": "S2", "reason": "숫자 또는 정량 표현이 없음"}],
    }
    document["score_result"]["breakdown"][0]["raw_score"] = 20
    document["score_result"]["breakdown"][0]["calibration"] = calibration

    versions = build_version_history([document])
    criterion = next(c for c in versions[0]["criteria"] if c["criterion_id"] == "A")

    assert criterion["score"] == 20
    assert criterion["calibration"] == calibration
