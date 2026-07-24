# 작성자: 경이
# 목적: 위원별 구조화 점수 + rubric 을 받아 평가기준 충족도(score_result)를 결정론적으로 계산한다(MTG-003).
#       LLM 이 아니라 Python 규칙으로 계산하며, 동일 입력에는 항상 동일 결과를 낸다.
#       출력은 contracts/schemas/review_output.schema.json(v2) 의 scoreResult 형태를 따른다.
# import: 표준 라이브러리 decimal, collections; 같은 패키지의 weights, deductions.

from __future__ import annotations

from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .calibration import build_score_cap
from .deductions import compute_penalties, _num
from .weights import criterion_max_scores, resolve_weights, total_max_score

SCORE_LABEL = "evaluation_criteria_alignment"
_QUANT = Decimal("0.01")


def _collect_scores(reviewer_results: list[dict[str, Any]]) -> dict[str, list[tuple[str, Decimal]]]:
    """criterion_id -> [(review_id, score)] 로 위원 점수를 모은다. score 가 없는(null) 항목은 건너뛴다."""
    by_criterion: dict[str, list[tuple[str, Decimal]]] = defaultdict(list)
    for r in reviewer_results:
        review_id = r.get("review_id") or r.get("persona_id") or "unknown"
        for item in r.get("rubric_scores", []):
            score = item.get("score")
            if score is None:
                continue
            by_criterion[item["criterion_id"]].append((review_id, Decimal(str(score))))
    return by_criterion


def calculate_score(
    rubric: dict[str, Any],
    reviewer_results: list[dict[str, Any]],
    *,
    submission: dict[str, Any] | None = None,
    calculation_version: str | None = None,
    method: str = "criterion_owner",
    missing_required_penalty: float | int | Decimal = 0,
) -> dict[str, Any]:
    """rubric + 위원 평가로 score_result(v2) 를 계산한다.

    criterion_owner: 각 기준을 담당 위원이 채점(보통 1명). 여러 위원이 같은 기준을 채점하면 평균한다.
    총점 = Σ(항목 점수) - Σ(penalty). 배점(max_score)이 곧 가중치다.
    """
    effective_version = calculation_version or ("score_v3" if submission is not None else "score_v2")
    max_scores = criterion_max_scores(rubric)
    weights = resolve_weights(rubric, method)
    by_criterion = _collect_scores(reviewer_results)
    penalty_amount = Decimal(str(missing_required_penalty))

    breakdown: list[dict[str, Any]] = []
    running_total = Decimal(0)

    # rubric 기준 순서를 유지해 결정론적 출력을 보장한다.
    for c in rubric["criteria"]:
        cid = c["criterion_id"]
        max_score = max_scores[cid]
        entries = by_criterion.get(cid, [])

        if entries:
            proposed_raw = sum((s for _, s in entries), Decimal(0)) / Decimal(len(entries))
            proposed_raw = proposed_raw.quantize(_QUANT, rounding=ROUND_HALF_UP)
            proposed_raw = min(max(proposed_raw, Decimal(0)), max_score)  # [0, max_score] 로 클램프
            source_ids = [rid for rid, _ in entries]
        else:
            proposed_raw = Decimal(0)
            source_ids = []

        calibration = build_score_cap(rubric, c, submission)
        if calibration is not None:
            raw = min(proposed_raw, calibration["cap_score"])
        else:
            raw = proposed_raw

        running_total += raw
        item = {
            "criterion_id": cid,
            "raw_score": _num(raw),
            "max_score": _num(max_score),
            "weight": _num(weights[cid]),
            "weighted_score": _num(raw),
            "penalty": 0,
            "source_review_ids": source_ids,
        }
        if calibration is not None and raw < proposed_raw:
            item["calibration"] = {
                "original_score": _num(proposed_raw),
                "cap_score": _num(calibration["cap_score"]),
                "cap_ratio": _num(calibration["cap_ratio"]),
                "signals": [
                    {"code": signal["code"], "reason": signal["reason"]}
                    for signal in calibration["signals"]
                ],
            }
        breakdown.append(item)

    penalties = compute_penalties(rubric, set(by_criterion.keys()), penalty_amount)
    penalty_total = sum((Decimal(str(p["amount"])) for p in penalties), Decimal(0))

    total = running_total - penalty_total
    total = max(total, Decimal(0))

    return {
        "total_score": _num(total),
        "max_score": _num(total_max_score(rubric)),
        "score_label": SCORE_LABEL,
        "calculation_version": effective_version,
        "calculation_method": method,
        "breakdown": breakdown,
        "penalties": penalties,
    }
