# 작성자: 경이
# 목적: 점수 설명 카드(RPT-006). score_result(계산 로그)의 항목별 점수를 rubric 기준·
#       위원별 근거(강점/약점/판단/근거ID)·감점 규칙과 연결해, 프론트가 "각 점수에 계산
#       근거 표시"(검수 기준)를 렌더링할 수 있는 구조화 데이터로 조립한다.
#       예외사항 "LLM 자연어와 계산값 불일치 방지"에 따라, 설명 문구(basis/formula)는
#       LLM이 아니라 score_result의 숫자에서 결정론적으로 생성한다 — 카드의 어떤 수치도
#       계산 엔진(M2) 출력과 어긋날 수 없다.
# import: 표준 라이브러리 collections/decimal; 같은 패키지의 deductions._num.

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from .deductions import _num


def _index_reviewer_scores(reviewer_results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """criterion_id -> [위원별 채점 근거] 로 모은다(위원 원본 근거를 카드에 연결하기 위함)."""
    by_criterion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in reviewer_results:
        review_id = r.get("review_id") or r.get("persona_id") or "unknown"
        for s in r.get("rubric_scores", []):
            by_criterion[s["criterion_id"]].append(
                {
                    "review_id": review_id,
                    "persona_id": r.get("persona_id"),
                    "score": s.get("score"),
                    "judgment": s.get("judgment"),
                    "strengths": s.get("strengths", []),
                    "issues": s.get("issues", []),
                    "suggestions": s.get("suggestions", []),
                    "evidence_ids": s.get("evidence_ids", []),
                    "evidence_status": s.get("evidence_status"),
                }
            )
    return by_criterion


def _basis_text(criterion_name: str, breakdown: dict[str, Any], contributors: list[dict], penalty: Decimal) -> str:
    """항목 점수가 어떻게 나왔는지 계산값만으로 서술한다(LLM 미사용)."""
    raw = breakdown["raw_score"]
    max_score = breakdown["max_score"]
    calibration = breakdown.get("calibration")
    n = len(contributors)
    if n == 0:
        text = f"'{criterion_name}' 항목을 채점한 위원이 없어 {_num(Decimal(str(raw)))}점으로 처리되었습니다."
    elif calibration:
        original = calibration["original_score"]
        cap = calibration["cap_score"]
        text = (
            f"위원 제안 점수 {original}점에 문서 근거 상한 {cap}점을 적용해 "
            f"{max_score}점 만점에 {raw}점으로 집계했습니다."
        )
    elif n == 1:
        text = f"담당 위원 1명이 {max_score}점 만점에 {raw}점을 부여했습니다."
    else:
        scores = ", ".join(str(c["score"]) for c in contributors)
        text = f"위원 {n}명의 점수({scores})를 평균해 {max_score}점 만점에 {raw}점으로 집계했습니다."
    if penalty > 0:
        text += f" 필수항목 관련 감점 {_num(penalty)}점이 적용되었습니다."
    return text


def build_score_explanation(
    score_result: dict[str, Any],
    rubric: dict[str, Any],
    reviewer_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """score_result + rubric + 위원 평가를 점수 설명 카드 데이터로 조립한다(RPT-006).

    반환 구조:
    - total: 총점 산식(항목 합 - 감점 합 = 총점)과 계산 방식/버전
    - criteria: 항목별 카드(점수·배점·집계 방식·위원별 근거·감점·계산 근거 문구)
    - penalties: score_result의 감점 목록 원본
    """
    criteria_meta = {c["criterion_id"]: c for c in rubric["criteria"]}
    reviewer_scores = _index_reviewer_scores(reviewer_results)

    penalties_by_criterion: dict[Any, list[dict]] = defaultdict(list)
    for p in score_result.get("penalties", []):
        penalties_by_criterion[p.get("criterion_id")].append(p)

    cards: list[dict[str, Any]] = []
    criteria_sum = Decimal(0)
    for b in score_result["breakdown"]:
        cid = b["criterion_id"]
        meta = criteria_meta.get(cid, {})
        contributors = reviewer_scores.get(cid, [])
        pens = penalties_by_criterion.get(cid, [])
        penalty_amount = sum((Decimal(str(p["amount"])) for p in pens), Decimal(0))
        criteria_sum += Decimal(str(b["raw_score"]))

        if not contributors:
            aggregation = "none"
        elif len(contributors) == 1:
            aggregation = "single"
        else:
            aggregation = "average"

        cards.append(
            {
                "criterion_id": cid,
                "criterion_name": meta.get("criterion_name", cid),
                "required": meta.get("required"),
                "raw_score": b["raw_score"],
                "max_score": b["max_score"],
                "weight": b.get("weight"),
                "weighted_score": b.get("weighted_score"),
                "aggregation": aggregation,
                "scored_by": b.get("source_review_ids", []),
                "reviewer_scores": contributors,
                "calibration": b.get("calibration"),
                "penalty": _num(penalty_amount),
                "penalty_reasons": pens,
                "basis": _basis_text(meta.get("criterion_name", cid), b, contributors, penalty_amount),
            }
        )

    penalty_sum = sum(
        (Decimal(str(p["amount"])) for p in score_result.get("penalties", [])), Decimal(0)
    )
    total_score = score_result["total_score"]
    total = {
        "total_score": total_score,
        "max_score": score_result["max_score"],
        "criteria_sum": _num(criteria_sum),
        "penalty_sum": _num(penalty_sum),
        "formula": f"{_num(criteria_sum)} - {_num(penalty_sum)} = {total_score}",
        "calculation_method": score_result["calculation_method"],
        "calculation_version": score_result["calculation_version"],
    }
    return {"total": total, "criteria": cards, "penalties": score_result.get("penalties", [])}
