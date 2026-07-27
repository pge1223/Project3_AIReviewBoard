from __future__ import annotations


PHASE_ORDER = (
    "problem_definition",
    "target_and_context",
    "purpose_and_value",
    "solution_scenario",
    "ai_role_and_mvp",
    "effect_and_validation",
    "differentiation_and_risk",
    "application_drafting",
)

PHASE_REQUIREMENTS = {
    "problem_definition": ("problem",),
    "target_and_context": ("target",),
    "purpose_and_value": ("purpose",),
    "solution_scenario": ("solution",),
    "ai_role_and_mvp": ("ai_role", "mvp_scope"),
    "effect_and_validation": ("expected_effect",),
    "differentiation_and_risk": ("differentiation", "risk"),
    "application_drafting": (),
}

FOUNDATION_STATE_KEYS = ("problem", "target", "purpose", "solution")

PHASE_LABELS = {
    "problem_definition": "문제 상황",
    "target_and_context": "대상과 적용 현장",
    "purpose_and_value": "목적과 핵심 가치",
    "solution_scenario": "해결 상황과 방식",
    "ai_role_and_mvp": "AI 역할·MVP·데이터",
    "effect_and_validation": "기대 효과와 검증 방법",
    "differentiation_and_risk": "차별점과 리스크",
    "application_drafting": "신청서 평가항목 초안",
}

_PHASE_FIELD_KEYWORDS = {
    "problem_definition": ("기술 설명", "문제", "현황", "필요성", "배경", "제안 내용"),
    "target_and_context": ("기술 설명", "대상", "사용자", "수혜", "적용"),
    "purpose_and_value": ("기술 설명", "목적", "가치"),
    "solution_scenario": ("기술 설명", "해결", "서비스", "제안 내용", "적용 방식"),
    "ai_role_and_mvp": ("기술 이름", "기술 설명", "ai", "인공지능", "mvp", "데이터"),
    "effect_and_validation": ("효과", "성과", "검증", "사회적 가치"),
    "differentiation_and_risk": ("혁신", "차별", "실현가능", "위험", "리스크"),
    "application_drafting": ("기술", "혁신", "확장", "적용", "실현가능", "사회적 가치", "평가"),
}

_ADMIN_KEYWORDS = ("성명", "이름", "연락처", "전화", "이메일", "주소", "소속", "서명", "날짜")


def is_coachable_field(row: dict) -> bool:
    haystack = f"{row.get('field_name', '')} {row.get('description', '')}".lower()
    return not any(keyword in haystack for keyword in _ADMIN_KEYWORDS)


def first_phase() -> str:
    return PHASE_ORDER[0]


def next_phase(current_phase: str) -> str | None:
    try:
        index = PHASE_ORDER.index(current_phase)
    except ValueError:
        return first_phase()
    return PHASE_ORDER[index + 1] if index + 1 < len(PHASE_ORDER) else None


def phase_complete(phase: str, confirmed_state: dict) -> bool:
    return all(str(confirmed_state.get(key) or "").strip() for key in PHASE_REQUIREMENTS.get(phase, ()))


def connected_fields(draft: list[dict], phase: str, *, limit: int = 3) -> list[dict]:
    candidates = [row for row in draft if is_coachable_field(row)]
    if not candidates:
        return []
    keywords = _PHASE_FIELD_KEYWORDS.get(phase, ())
    matched = []
    for row in candidates:
        haystack = f"{row.get('field_name', '')} {row.get('description', '')}".lower()
        if any(keyword.lower() in haystack for keyword in keywords):
            matched.append(row)
    return (matched or candidates[:1])[:limit]


def foundation_ready(confirmed_state: dict) -> bool:
    return all(str(confirmed_state.get(key) or "").strip() for key in FOUNDATION_STATE_KEYS)
