# 작성자: 경이
# 목적: 수정 전후 비교 리포트(RPT-004). 이전/신규 회의 결과(review_output v2 문서) 둘을
#       받아 항목별 점수 증감과 해결/신규/잔존 지적을 비교한다(검수 기준 "항목별 점수
#       증감 및 해결된 지적 표시"). 두 회의를 ID로 꺼내오는 조회는 backend(윤한 DB)의
#       몫이고, 이 함수는 문서 2개를 입력으로 받는 순수 비교 로직이다. 평가기준(rubric)이
#       달라지면 직접 비교를 제한하고 경고를 남긴다(예외사항 "평가기준 버전이 다르면
#       직접 비교 제한"). 프론트 화면(React 비교 리포트)은 가은 몫이다.
# import: 표준 라이브러리 collections/decimal; 같은 패키지의 deductions._num.

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from .deductions import _num
from .personalization import TECHNICAL_PERSONA_IDS

# 판단 심각도(높을수록 심각). 한 항목을 여러 위원이 채점하면 가장 심각한 판단을 대표로 쓴다.
_JUDGMENT_SEVERITY = {"strong": 0, "acceptable": 1, "needs_improvement": 2, "critical_risk": 3}

# 한 항목을 여러 위원이 겹쳐 채점할 때 "그 항목의 실제 담당 위원"을 고르는 우선순위.
# 프론트 VersionTrackerTestPage.personaRank와 동일하게 맞춘다: 기술(개발) > 전문 > 종합(완성도).
# 이 우선순위가 곧 항목의 소속 탭(dev/planning)을 결정한다.
_GENERALIST_PERSONA_IDS = {"presentation_completeness"}


def _persona_rank(persona_id: str | None) -> int:
    if persona_id in TECHNICAL_PERSONA_IDS:
        return 2
    if persona_id in _GENERALIST_PERSONA_IDS:
        return 0
    return 1


def _issues_by_criterion(document: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in document.get("reviewer_results", []):
        for s in r.get("rubric_scores", []):
            for issue in s.get("issues", []):
                out[s["criterion_id"]].add(issue)
    return out


def _representative_judgment_by_criterion(document: dict[str, Any]) -> dict[str, str]:
    worst: dict[str, str] = {}
    for r in document.get("reviewer_results", []):
        for s in r.get("rubric_scores", []):
            cid, j = s["criterion_id"], s.get("judgment")
            if j is None:
                continue
            if cid not in worst or _JUDGMENT_SEVERITY.get(j, 0) > _JUDGMENT_SEVERITY.get(worst[cid], 0):
                worst[cid] = j
    return worst


def _scores_by_criterion(document: dict[str, Any]) -> dict[str, Any]:
    return {b["criterion_id"]: b["raw_score"] for b in document["score_result"]["breakdown"]}


def _criteria_meta(document: dict[str, Any]) -> dict[str, dict]:
    return {c["criterion_id"]: c for c in document["rubric"]["criteria"]}


def _detail_by_criterion(document: dict[str, Any]) -> dict[str, dict]:
    """항목마다 "대표 위원"(우선순위 최상위)의 지적·개선안·소속을 고른다.
    한 항목을 여러 위원이 겹쳐 채점하면 _persona_rank가 높은 위원을 대표로 삼는다."""
    detail: dict[str, dict] = {}
    for r in document.get("reviewer_results", []):
        pid = r.get("persona_id")
        for s in r.get("rubric_scores", []):
            cid = s.get("criterion_id")
            if cid is None:
                continue
            cur = detail.get(cid)
            if cur is None or _persona_rank(pid) > _persona_rank(cur["persona_id"]):
                detail[cid] = {
                    "persona_id": pid,
                    "issues": list(s.get("issues", []) or []),
                    "suggestions": list(s.get("suggestions", []) or []),
                }
    return detail


def build_version_history(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """오래된→최신 순 회의(meeting) 문서 리스트를 v1.0, v1.1, v1.2 … 버전 히스토리로 만든다.

    "다음 수정본 제출"로 회의가 하나씩 쌓일 때마다 버전이 하나씩 늘어난다(덮어쓰지 않는다).
    각 버전은 총점 + 항목별(점수·판정·소속 위원·지적·개선안)과, 직전 버전 대비 지적 상태
    (new=이번에 새로 생김 / resolved=이번에 해결됨)를 담아 프론트가 그대로 그릴 수 있게 한다.
    build_revision_comparison(2개 비교)과 달리 N개 전체 히스토리를 다룬다."""
    versions: list[dict[str, Any]] = []
    prev_issues: dict[str, set[str]] | None = None
    for idx, doc in enumerate(documents):
        meta = _criteria_meta(doc)
        scores = _scores_by_criterion(doc)
        issues = _issues_by_criterion(doc)
        judgments = _representative_judgment_by_criterion(doc)
        detail = _detail_by_criterion(doc)
        sr = doc.get("score_result") or {}

        criteria: list[dict[str, Any]] = []
        for b in sr.get("breakdown", []) or []:
            cid = b.get("criterion_id")
            m = meta.get(cid, {})
            d = detail.get(cid, {})
            cur_iss = issues.get(cid, set())
            prev_iss = prev_issues.get(cid, set()) if prev_issues is not None else set()
            new_iss = (cur_iss - prev_iss) if prev_issues is not None else set()
            resolved_iss = (prev_iss - cur_iss) if prev_issues is not None else set()
            committee = "dev" if d.get("persona_id") in TECHNICAL_PERSONA_IDS else "planning"
            criterion_version = {
                    "criterion_id": cid,
                    "criterion_name": m.get("criterion_name", cid),
                    "committee": committee,
                    "score": b.get("raw_score", 0),
                    "max": b.get("max_score", m.get("max_score")),
                    "judgment": judgments.get(cid) or "acceptable",
                    "issues": d.get("issues", []),
                    "suggestions": d.get("suggestions", []),
                    "new_issues": sorted(new_iss),
                    "resolved_issues": sorted(resolved_iss),
                }
            if b.get("calibration"):
                criterion_version["calibration"] = b["calibration"]
            criteria.append(criterion_version)

        versions.append(
            {
                "version": f"v1.{idx}",
                "index": idx,
                "label": "최초 제출" if idx == 0 else f"{idx}차 수정본",
                "meeting_id": doc.get("meeting_id"),
                "submitted_at": doc.get("created_at"),
                "total_score": sr.get("total_score", 0),
                "max_score": sr.get("max_score", 100),
                "criteria": criteria,
            }
        )
        prev_issues = issues
    return versions


def build_revision_comparison(
    before_document: dict[str, Any],
    after_document: dict[str, Any],
) -> dict[str, Any]:
    """수정 전(before)·수정 후(after) 회의 결과를 비교한 리포트를 만든다(RPT-004).

    반환 구조:
    - total: 총점 before/after/delta
    - criteria: 두 회의에 공통으로 있는 항목별 비교(점수 증감, 판단 변화, 해결/신규/잔존 지적)
    - added_criteria / removed_criteria: 평가기준이 바뀌어 한쪽에만 있는 항목
    - rubric_changed / direct_comparison_limited / warnings: 평가기준 변경 감지(예외사항)
    """
    before_meta = _criteria_meta(before_document)
    after_meta = _criteria_meta(after_document)

    before_scores = _scores_by_criterion(before_document)
    after_scores = _scores_by_criterion(after_document)
    before_issues = _issues_by_criterion(before_document)
    after_issues = _issues_by_criterion(after_document)
    before_judgment = _representative_judgment_by_criterion(before_document)
    after_judgment = _representative_judgment_by_criterion(after_document)

    common_ids = [cid for cid in before_meta if cid in after_meta]
    removed_ids = [cid for cid in before_meta if cid not in after_meta]
    added_ids = [cid for cid in after_meta if cid not in before_meta]

    warnings: list[str] = []
    criteria: list[dict[str, Any]] = []
    for cid in common_ids:
        b_max = before_meta[cid].get("max_score")
        a_max = after_meta[cid].get("max_score")
        max_changed = b_max != a_max
        if max_changed:
            warnings.append(
                f"'{after_meta[cid].get('criterion_name', cid)}' 배점이 {b_max}→{a_max}로 바뀌어 "
                f"점수 증감을 직접 비교하기 어렵습니다."
            )

        b_score = before_scores.get(cid, 0)
        a_score = after_scores.get(cid, 0)
        delta = _num(Decimal(str(a_score)) - Decimal(str(b_score)))
        b_iss, a_iss = before_issues.get(cid, set()), after_issues.get(cid, set())

        criteria.append(
            {
                "criterion_id": cid,
                "criterion_name": after_meta[cid].get("criterion_name", cid),
                "comparable": not max_changed,
                "before_score": b_score,
                "after_score": a_score,
                "delta": delta,
                "before_judgment": before_judgment.get(cid),
                "after_judgment": after_judgment.get(cid),
                "resolved_issues": sorted(b_iss - a_iss),
                "new_issues": sorted(a_iss - b_iss),
                "persisting_issues": sorted(b_iss & a_iss),
                "max_score_changed": max_changed,
            }
        )

    if removed_ids or added_ids:
        warnings.append(
            "평가기준 항목 구성이 달라졌습니다(추가/삭제된 항목이 있어 총점은 직접 비교가 제한됩니다)."
        )

    rubric_changed = bool(removed_ids or added_ids) or any(c["max_score_changed"] for c in criteria)

    before_total = before_document["score_result"]["total_score"]
    after_total = after_document["score_result"]["total_score"]

    return {
        "meeting_before": before_document.get("meeting_id"),
        "meeting_after": after_document.get("meeting_id"),
        "rubric_changed": rubric_changed,
        "direct_comparison_limited": rubric_changed,
        "warnings": warnings,
        "total": {
            "before": before_total,
            "after": after_total,
            "delta": _num(Decimal(str(after_total)) - Decimal(str(before_total))),
        },
        "criteria": criteria,
        "added_criteria": [
            {"criterion_id": cid, "criterion_name": after_meta[cid].get("criterion_name", cid)}
            for cid in added_ids
        ],
        "removed_criteria": [
            {"criterion_id": cid, "criterion_name": before_meta[cid].get("criterion_name", cid)}
            for cid in removed_ids
        ],
    }
