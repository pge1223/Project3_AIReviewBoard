"""Application-form draft helpers for the facilitator v02 prompt.

Rollback boundary:
- Switching prompt_loader.IDEATION_CONV_DISCUSSION_FACILITATOR_TEMPLATE back to
  ideation_conv_discussion_facilitator.txt disables new draft patches.
- These helpers are additive and keep legacy sessions without draft state valid.
"""

from __future__ import annotations

import re
from typing import Any


_FIELD_ID_SAFE = re.compile(r"[^a-z0-9_]+")

# 2026-07-26 라운드테이블 재설계 — frontend/src/pages/board/ideationConversationHelpers.js의
# ADMINISTRATIVE_FIELD_KEYWORDS/isAdministrativeFormField와 반드시 동일하게 유지한다(둘 중
# 하나만 바뀌면 "회의에서 다룰 항목 선택 모달"과 진행자가 다음 필드를 고르는 기준이
# 어긋난다). 진행자가 고정 9단계 체크리스트 대신 이 신청서에 실제로 있는 "남은 내용 항목"을
# 대상으로 다음 질문을 고르도록 바꾸면서, 그 후보군에서 행정/개인정보 항목을 코드로 먼저
# 제외하기 위해 추가했다(예전에는 이 필터링을 LLM 프롬프트 지시문에만 맡겼다).
# 가은/Claude(2026-07-27, 요청: "과제번호, email 이런 게 다 체크되어 있다" 버그 확인) —
# frontend/src/pages/board/ideationConversationHelpers.js의 ADMINISTRATIVE_FIELD_KEYWORDS와
# 반드시 동일하게 유지한다(위 모듈 docstring 참고). 실사용 신청서에서 이 목록에 없어
# 걸러지지 않았던 개인정보/행정 항목을 추가했다: 핸드폰, 영문 e-mail, 주민등록번호·생년월일,
# 개인정보, 소속(기관), 책임자, 참여기관/주관기관/기관명.
_ADMINISTRATIVE_FIELD_KEYWORDS = (
    "담당자", "성명", "전화", "핸드폰", "휴대폰", "휴대전화", "이메일", "메일", "e-mail", "팩스", "홈페이지",
    "사업자등록번호", "법인등록번호", "부서", "직위", "대표자", "책임자",
    "설립연도", "매출액", "매출", "경영실적", "자본금", "종업원", "고용 인원",
    "신청 기관", "신청기관", "참여기관", "주관기관", "기관명", "소속", "도시명", "주소", "기업(법인)명",
    "주민등록번호", "생년월일", "개인정보",
)


def is_administrative_form_field(field_name: str) -> bool:
    name = (field_name or "").strip().lower()
    if not name:
        return False
    return any(keyword.lower() in name for keyword in _ADMINISTRATIVE_FIELD_KEYWORDS)


def remaining_content_fields(draft: list[dict] | None) -> list[dict]:
    """아직 확정되지 않았고 행정/개인정보 항목이 아닌 draft row만, 신청서 원래 순서대로
    반환한다. 진행자가 고정 9단계 대신 이 리스트에서 다음 필드를 고른다(순서 자체를
    강제하지 않고 후보군만 좁힌다 — 어떤 걸 먼저 다룰지는 진행자 프롬프트의 판단)."""
    return [
        row
        for row in (draft or [])
        if isinstance(row, dict)
        and str(row.get("status") or "") != "confirmed"
        and not is_administrative_form_field(row.get("field_name"))
    ]


def _field_id(item: dict, index: int) -> str:
    provided = str(item.get("field_id") or "").strip().lower()
    if provided:
        normalized = _FIELD_ID_SAFE.sub("_", provided).strip("_")
        if normalized:
            return normalized
    return f"form_field_{index + 1}"


def initialize_application_form_draft(items: list[dict] | None) -> list[dict]:
    """Build stable, empty draft rows without modifying extracted form items."""
    rows: list[dict] = []
    for index, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        field_name = str(item.get("field_name") or "").strip()
        if not field_name:
            continue
        rows.append(
            {
                "field_id": _field_id(item, index),
                "field_name": field_name,
                "description": str(item.get("description") or "").strip(),
                "char_limit": item.get("char_limit"),
                "value": "",
                "status": "empty",
            }
        )
    return rows


def apply_application_form_draft_patch(
    current_draft: list[dict] | None,
    raw_patch: Any,
    confirmable_field_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Apply only patches targeting known fields and return sanitized applied rows."""
    draft = [dict(row) for row in (current_draft or []) if isinstance(row, dict)]
    by_id = {str(row.get("field_id")): row for row in draft}
    applied: list[dict] = []
    if not isinstance(raw_patch, list):
        return draft, applied

    for patch in raw_patch:
        if not isinstance(patch, dict):
            continue
        field_id = str(patch.get("field_id") or "").strip()
        value = str(patch.get("value") or "").strip()
        row = by_id.get(field_id)
        if row is None or not value:
            continue

        char_limit = row.get("char_limit")
        if isinstance(char_limit, int) and char_limit > 0:
            value = value[:char_limit]

        status = str(patch.get("status") or "draft").strip().lower()
        if status not in {"draft", "confirmed"}:
            status = "draft"
        if (
            status == "confirmed"
            and confirmable_field_ids is not None
            and field_id not in confirmable_field_ids
        ):
            status = "draft"
        row["value"] = value
        row["status"] = status
        applied.append({"field_id": field_id, "value": value, "status": status})

    return draft, applied
