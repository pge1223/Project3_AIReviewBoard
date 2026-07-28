# 작성자: 경이
# 목적: 위원 일관성 테스트 하네스(TST-002) 검증 — 반복 평가 편차 지표(총점/항목/판단/핵심
#       지적) 계산과 허용범위 위반 판정, 그리고 실제 파이프라인(run_meeting)이 동일 LLM
#       출력에 대해 결정론적임(편차 0)을 baseline으로 확인한다.
# import: 표준 라이브러리 copy/json/sys/pathlib, pytest; ai/meeting quality/graph 패키지.

import copy
import json
import sys
from pathlib import Path

import pytest

MEETING_DIR = Path(__file__).resolve().parents[1]  # ai/meeting
sys.path.insert(0, str(MEETING_DIR))

from graph import build_routing, run_meeting  # noqa: E402
from quality import ConsistencyTolerance, run_consistency_check, summarize_consistency  # noqa: E402

COMPETITION_MAPPING_PATH = MEETING_DIR / "personas" / "rubric_mapping_competition.json"

_PERSONA_NAMES = {
    "planning_expert": "기획 전문가",
    "dev_expert": "개발 전문가",
}


def _doc(total: int, criteria: list[dict]) -> dict:
    """일관성 하네스가 읽는 필드만 담은 최소 v2 문서.
    criteria: [{criterion_id, raw_score, judgment, issues}]"""
    return {
        "score_result": {
            "total_score": total,
            "breakdown": [{"criterion_id": c["criterion_id"], "raw_score": c["raw_score"]} for c in criteria],
        },
        "reviewer_results": [
            {
                "persona_id": "p",
                "rubric_scores": [
                    {"criterion_id": c["criterion_id"], "judgment": c["judgment"], "issues": c.get("issues", [])}
                    for c in criteria
                ],
            }
        ],
    }


def _crit(cid: str, score: int, judgment: str, issues=None) -> dict:
    return {"criterion_id": cid, "raw_score": score, "judgment": judgment, "issues": issues or []}


# ---------------------------------------------------------------------------
# 분석기(summarize_consistency)
# ---------------------------------------------------------------------------


def test_identical_runs_have_zero_variance():
    doc = _doc(60, [_crit("A", 30, "strong", ["문제1"]), _crit("B", 30, "acceptable", ["문제2"])])
    report = summarize_consistency([copy.deepcopy(doc) for _ in range(4)])

    assert report["within_tolerance"] is True
    assert report["total_score"]["range"] == 0
    assert report["criteria"]["A"]["judgment_agreement"] == 1.0
    assert report["key_issue_jaccard_mean"] == 1.0
    assert report["violations"] == []


def test_small_jitter_within_tolerance():
    docs = [
        _doc(60, [_crit("A", 30, "strong", ["문제1"]), _crit("B", 30, "acceptable")]),
        _doc(62, [_crit("A", 32, "strong", ["문제1"]), _crit("B", 30, "acceptable")]),
        _doc(61, [_crit("A", 31, "strong", ["문제1"]), _crit("B", 30, "acceptable")]),
    ]
    report = summarize_consistency(docs)  # 기본 허용치(총점 10, 항목 8)
    assert report["within_tolerance"] is True
    assert report["total_score"]["range"] == 2
    assert report["criteria"]["A"]["score"]["range"] == 2


def test_excess_total_score_range_flagged():
    docs = [
        _doc(40, [_crit("A", 40, "critical_risk")]),
        _doc(80, [_crit("A", 80, "strong")]),
    ]
    report = summarize_consistency(docs, ConsistencyTolerance(max_total_score_range=10, max_criterion_score_range=8))
    assert report["within_tolerance"] is False
    assert any("총점 편차" in v for v in report["violations"])
    assert any("[A] 점수 편차" in v for v in report["violations"])


def test_judgment_disagreement_flagged():
    # A의 판단이 실행마다 뒤집힘 → 일치율 낮음
    docs = [
        _doc(30, [_crit("A", 30, "strong")]),
        _doc(30, [_crit("A", 30, "critical_risk")]),
        _doc(30, [_crit("A", 30, "needs_improvement")]),
    ]
    report = summarize_consistency(docs, ConsistencyTolerance(min_judgment_agreement=0.6))
    assert report["within_tolerance"] is False
    assert any("judgment 일치율" in v for v in report["violations"])
    assert report["criteria"]["A"]["judgment_agreement"] < 0.6


def test_scored_ratio_tracks_gating_variance():
    # A가 어떤 실행에선 채점되고 어떤 실행에선 미채점(게이팅) → scored_ratio로 드러남
    docs = [
        _doc(30, [_crit("A", 30, "strong")]),
        _doc(0, []),  # A 미채점
    ]
    report = summarize_consistency(docs)
    assert report["criteria"]["A"]["scored_ratio"] == 0.5


# ---------------------------------------------------------------------------
# baseline: 실제 run_meeting 파이프라인은 동일 LLM 출력에 대해 결정론적
# ---------------------------------------------------------------------------


def _make_raw_reviewer(persona_id, criteria, score):
    """criteria: [(criterion_id, criterion_name), ...] — 주담당 항목 전부 같은 점수로 채점."""
    return {
        "review_id": f"REV-{persona_id}",
        "persona_id": persona_id,
        "persona_name": _PERSONA_NAMES[persona_id],
        "review_round": 1,
        "review_summary": "요약",
        "review_items": [
            {
                "criterion_id": cid,
                "criterion_name": cname,
                "max_score": 25,
                "score_recommendation": score,
                "judgment": "adequate",
                "confidence": "high",
                "strengths": [],
                "weaknesses": ["개선 필요"],
                "evidence_refs": [],
                "improvement_actions": [],
            }
            for cid, cname in criteria
        ],
        "cross_reviews": [],
        "out_of_scope": [],
    }


_RAW_CHAIR = {
    "chair_id": "review_chair",
    "overall_assessment": "종합",
    "consensus": [],
    "disagreements": [],
    "top_strengths": [],
    "top_risks": [],
    "final_priority_actions": [],
    "final_decision": None,
    "decision_note": None,
}


def test_run_meeting_pipeline_is_deterministic_baseline():
    """동일 LLM 출력(stub)이면 run_meeting을 몇 번 돌려도 편차 0이어야 한다(파이프라인 결정론).
    실제 모델이 붙으면 같은 하네스로 편차를 측정한다."""
    mapping = json.loads(COMPETITION_MAPPING_PATH.read_text(encoding="utf-8"))
    routing = build_routing(mapping)
    criteria_by_id = {c["criterion_id"]: c for c in mapping["rubric"]}
    owned: dict[str, list[str]] = {}
    for cid, r in routing.items():
        owned.setdefault(r["primary"], []).append(cid)

    raw_by_marker = {
        f"{_PERSONA_NAMES[pid]}입니다": _make_raw_reviewer(
            pid, [(cid, criteria_by_id[cid]["criterion_name"]) for cid in cids], 20
        )
        for pid, cids in owned.items()
    }
    raw_by_marker["위원장(review_chair)입니다"] = _RAW_CHAIR

    def stub(prompt: str) -> str:
        for marker, raw in raw_by_marker.items():
            if marker in prompt:
                return json.dumps(raw, ensure_ascii=False)
        raise AssertionError("마커 못 찾음")

    def run_once() -> dict:
        return run_meeting(
            meeting_id="MTG-CONSIST-001",
            project_id="PRJ-CONSIST-001",
            document_id="DOC-CONSIST-001",
            title="일관성 baseline",
            rubric_mapping=mapping,
            submission={"document_name": "t.pdf", "text": "..."},
            retrieved_evidence=[],
            llm_call=stub,
        )

    report = run_consistency_check(run_once, n_runs=3)

    assert report["within_tolerance"] is True
    assert report["total_score"]["range"] == 0
    assert report["total_score"]["mean"] == 80
    assert all(c["judgment_agreement"] == 1.0 for c in report["criteria"].values())
    assert report["key_issue_jaccard_mean"] == 1.0


def test_run_consistency_check_rejects_zero_runs():
    with pytest.raises(ValueError):
        run_consistency_check(lambda: _doc(0, []), n_runs=0)
