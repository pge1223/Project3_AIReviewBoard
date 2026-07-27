from __future__ import annotations


_PHASE_FIELD_TERMS = {
    "problem_definition": ("현황", "문제", "필요성", "배경", "불편", "problem"),
    "target_and_context": ("대상", "사용자", "수혜", "적용 현장", "도입", "target", "user"),
    "purpose_and_value": ("목적", "목표", "핵심 가치", "가치", "purpose", "value"),
    "solution_scenario": ("제안 내용", "해결", "추진 방법", "서비스", "적용 방식", "solution"),
    "ai_role_and_mvp": ("제안 내용", "추진 방법", "기술", "ai", "인공지능", "mvp", "데이터"),
    "effect_and_validation": ("기대 효과", "효과", "성과", "검증", "측정", "effect"),
    "differentiation_and_risk": ("차별", "혁신", "리스크", "위험", "실현 가능", "risk"),
}


def _text(value) -> str:
    return str(value or "").strip()


def _phase_value(phase: str, confirmed: dict) -> str:
    if phase == "problem_definition":
        return _text(confirmed.get("problem"))
    if phase == "target_and_context":
        parts = []
        target_user = _text(confirmed.get("target_user") or confirmed.get("target"))
        operator = _text(confirmed.get("operator"))
        context = _text(confirmed.get("context"))
        if target_user:
            parts.append(f"핵심 대상은 {target_user}입니다.")
        if operator:
            parts.append(f"도입·운영 주체는 {operator}입니다.")
        if context:
            parts.append(f"적용 현장은 {context}입니다.")
        return " ".join(parts)
    if phase == "purpose_and_value":
        return _text(confirmed.get("value") or confirmed.get("purpose"))
    if phase in {"solution_scenario", "ai_role_and_mvp"}:
        parts = []
        solution = _text(confirmed.get("solution"))
        ai_role = _text(confirmed.get("ai_role"))
        mvp_scope = _text(confirmed.get("mvp_scope"))
        required_data = _text(confirmed.get("required_data"))
        if solution:
            parts.append(solution)
        if ai_role:
            parts.append(f"AI는 {ai_role} 역할을 담당합니다.")
        if mvp_scope:
            parts.append(f"초기 구현 범위는 {mvp_scope}입니다.")
        if required_data:
            parts.append(f"필요 데이터는 {required_data}입니다.")
        return " ".join(parts)
    if phase == "effect_and_validation":
        effect = _text(confirmed.get("expected_effect"))
        validation = _text(confirmed.get("validation_method"))
        if effect and validation:
            return f"{effect} 효과를 기대하며, {validation} 방식으로 검증합니다."
        return effect or validation
    if phase == "differentiation_and_risk":
        differentiation = _text(confirmed.get("differentiation"))
        risk = _text(confirmed.get("risk"))
        parts = []
        if differentiation:
            parts.append(differentiation)
        if risk:
            parts.append(f"주요 리스크는 {risk}입니다.")
        return " ".join(parts)
    return ""


def _best_field(draft: list[dict], phase: str, preferred_field_ids: list[str]) -> dict | None:
    by_id = {str(row.get("field_id")): row for row in draft}
    terms = _PHASE_FIELD_TERMS.get(phase, ())
    ranked: list[tuple[int, int, dict]] = []
    for index, row in enumerate(draft):
        haystack = f"{row.get('field_name', '')} {row.get('description', '')}".lower()
        score = sum(3 if term.lower() in _text(row.get("field_name")).lower() else 1 for term in terms if term.lower() in haystack)
        if str(row.get("field_id")) in preferred_field_ids:
            score += 2
        if score:
            ranked.append((score, -index, row))
    if ranked:
        return max(ranked, key=lambda item: (item[0], item[1]))[2]
    for field_id in preferred_field_ids:
        if field_id in by_id:
            return by_id[field_id]
    return draft[0] if draft else None


def incremental_draft_patches(
    *,
    state: dict,
    phase: str,
    state_update: dict,
    candidate_state_update: dict,
    preferred_field_ids: list[str],
    skip_field_ids: set[str],
) -> list[dict]:
    draft = state.get("application_form_draft") or []
    candidate_phase = None
    candidate_value = ""
    for candidate_key, mapped_phase in (
        ("solution_candidate", "solution_scenario"),
        ("target_candidate", "target_and_context"),
        ("operator_candidate", "target_and_context"),
        ("context_candidate", "target_and_context"),
        ("value_candidate", "purpose_and_value"),
    ):
        candidate_value = _text(candidate_state_update.get(candidate_key))
        if candidate_value:
            candidate_phase = mapped_phase
            break

    mapping_phase = phase if state_update or not candidate_phase else candidate_phase
    row = _best_field(
        draft,
        mapping_phase,
        preferred_field_ids if mapping_phase == phase else [],
    )
    if row is None:
        return []
    field_id = str(row.get("field_id"))
    if field_id in skip_field_ids:
        return []

    value = _phase_value(phase, state.get("confirmed_state") or {}) if state_update else ""
    if value:
        char_limit = row.get("char_limit")
        if isinstance(char_limit, int) and char_limit > 0:
            value = value[:char_limit]
        return [
            {
                "field_id": field_id,
                "form_field": _text(row.get("field_name")),
                "value": value,
                "provisional_value": "",
                "source": "conversation",
                "status": "draft",
                "confidence": "medium",
                "editable": True,
            }
        ]

    if candidate_value:
        return [
            {
                "field_id": field_id,
                "form_field": _text(row.get("field_name")),
                "value": "",
                "provisional_value": f"작성 중 — {candidate_value} 후보가 논의되었습니다.",
                "source": "conversation",
                "status": "in_progress",
                "confidence": "low",
                "editable": True,
            }
        ]
    return []


def provisional_phase_patch(
    *,
    state: dict,
    phase: str,
    value: str,
    preferred_field_ids: list[str],
) -> list[dict]:
    row = _best_field(
        state.get("application_form_draft") or [],
        phase,
        preferred_field_ids,
    )
    normalized = _text(value)
    if row is None or not normalized:
        return []
    return [
        {
            "field_id": str(row.get("field_id")),
            "form_field": _text(row.get("field_name")),
            "value": "",
            "provisional_value": f"작성 중 - 전문가 권고안: {normalized}",
            "source": "expert_delegation",
            "status": "in_progress",
            "confidence": "low",
            "editable": True,
        }
    ]
