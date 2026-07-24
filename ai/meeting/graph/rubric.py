# 작성자: 경이
# 목적: ai/meeting/personas/rubric_mapping_*.json(가은 PER-002 산출물)을 review_output.schema.json
#       v2의 rubric 객체와, 위원 라우팅 정보(어느 criterion을 어느 persona가 채점하는지)로
#       변환한다(M4). 도메인 무관하게 동작한다 — government_support/competition 매핑 모두
#       같은 persona_rubric_mapping 스키마를 쓰기 때문이다.
# import: 표준 라이브러리 typing만 사용.

from __future__ import annotations

from typing import Any


def combine_criteria_documents(
    documents: list[dict[str, Any]],
    *,
    max_chars: int,
) -> tuple[str, list[str]]:
    """URL·공문 파일 등 여러 criteria 문서를 같은 비중으로 합친다.

    첫 문서가 긴 웹페이지라는 이유로 뒤에 업로드한 배점표가 잘리지 않도록 문서별 문자
    예산을 균등 배분한다. 반환 ID 목록은 캐시 무효화에도 사용한다.
    """
    usable = [document for document in documents if document.get("parsed_text")]
    if not usable:
        return "", []
    per_document = max(500, max_chars // len(usable))
    parts: list[str] = []
    source_ids: list[str] = []
    for document in usable:
        source_id = str(document.get("_id") or document.get("document_id") or "")
        source_ids.append(source_id)
        source_name = (
            document.get("original_filename")
            or document.get("source_url")
            or source_id
            or "공고 자료"
        )
        excerpt = str(document["parsed_text"])[:per_document]
        parts.append(f"[출처: {source_name}]\n{excerpt}")
    return "\n\n".join(parts)[:max_chars], source_ids


def build_rubric(mapping: dict[str, Any]) -> dict[str, Any]:
    """rubric_mapping_*.json의 rubric[] 배열을 v2 rubric 객체로 변환한다.

    default_supplementary_perspectives는 채점 대상이 아니므로 포함하지 않는다.
    source_document_id는 mapping["meta"]에 값이 있으면(공고문에서 동적으로 추출된
    rubric, build_dynamic_rubric_mapping() 참고) 그 값을 쓰고, 없으면(정적 템플릿)
    None이다.
    """
    criteria = []
    for item in mapping["rubric"]:
        criterion = {
            "criterion_id": item["criterion_id"],
            "criterion_name": item["criterion_name"],
            "max_score": item["max_score"],
            "required": item["required"],
        }
        if item.get("description"):
            criterion["description"] = item["description"]
        # 공고문에 명시된 필수 요소가 동적 rubric 추출 결과에 있을 때만 전달한다.
        # 값이 없는 정적 rubric에는 필드를 만들지 않아 기존 v2 결과를 그대로 유지한다.
        if item.get("required_keywords"):
            criterion["required_keywords"] = list(item["required_keywords"])
        if item.get("required_keyword_groups"):
            criterion["required_keyword_groups"] = [
                list(group) for group in item["required_keyword_groups"]
            ]
        criteria.append(criterion)
    domain = mapping["meta"]["domain"]
    rubric = {
        "rubric_id": f"RUBRIC-{domain.upper()}",
        "source_document_id": mapping.get("meta", {}).get("source_document_id"),
        "total_max_score": mapping["total_max_score"],
        "criteria": criteria,
    }
    if mapping.get("meta", {}).get("source_document_ids"):
        rubric["source_document_ids"] = list(mapping["meta"]["source_document_ids"])
    if mapping.get("bonus_rules"):
        rubric["bonus_rules"] = [dict(rule) for rule in mapping["bonus_rules"]]
        rubric["bonus_max_score"] = mapping.get("bonus_max_score", 0)
    return rubric


# 가은/Claude(2026-07-18): PER-002 동적 rubric — 공고문(criteria 문서)에서 LLM으로
# 추출한 평가항목을 rubric_mapping 형태로 병합한다. 경이 리뷰(팀 승인 답변, 아래 조건
# 그대로 반영):
#   - 새 persona를 만들지 않고 base_mapping["committee"](고정 4인)에만 배정한다.
#   - 배점 합계는 LLM 출력을 신뢰하지 않고 항상 서버에서 재계산한다
#     (weights.total_max_score()가 rubric["total_max_score"]와 criteria 배점 합이
#     다르면 예외를 던지므로, 애초에 항상 일치하게 만든다).
#   - criterion_id 중복은 거부한다.
#   - primary_persona_id/secondary_persona_id는 committee 소속이어야 하고,
#     primary_perspective_id는 그 persona의 실제 evaluation_perspectives에 있는
#     값이어야 한다(persona_cards.json — LLM이 존재하지 않는 관점을 지어내는 걸 막는
#     화이트리스트 검증).
# 검증 실패 시 ValueError를 던진다 — 호출부(backend/app/api/routes/meetings.py)가
# 잡아서 정적 템플릿(base_mapping)으로 폴백한다.
def build_dynamic_rubric_mapping(
    base_mapping: dict[str, Any],
    extracted_items: list[dict[str, Any]],
    source_document_id: str,
    persona_cards: dict[str, dict[str, Any]],
    bonus_rules: list[dict[str, Any]] | None = None,
    bonus_max_score: float | int = 0,
    source_document_ids: list[str] | None = None,
) -> dict[str, Any]:
    """base_mapping(정적 템플릿)의 committee/default_supplementary_perspectives는
    그대로 두고, rubric[]만 extracted_items로 교체한 새 mapping을 반환한다."""
    committee = set(base_mapping["committee"])
    if not extracted_items:
        raise ValueError("추출된 평가항목이 비어 있습니다.")

    perspective_whitelist: dict[str, set[str]] = {
        pid: {p["perspective_id"] for p in persona_cards.get(pid, {}).get("evaluation_perspectives", [])}
        for pid in committee
    }

    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for item in extracted_items:
        criterion_id = item.get("criterion_id")
        criterion_name = item.get("criterion_name")
        description = item.get("description")
        max_score = item.get("max_score")
        primary_persona_id = item.get("primary_persona_id")
        primary_perspective_id = item.get("primary_perspective_id")
        secondary_persona_id = item.get("secondary_persona_id")
        required_keywords = item.get("required_keywords") or []
        required_keyword_groups = item.get("required_keyword_groups") or []

        if not isinstance(criterion_id, str) or not criterion_id:
            raise ValueError(f"criterion_id가 올바르지 않습니다: {item!r}")
        if criterion_id in seen_ids:
            raise ValueError(f"criterion_id가 중복되었습니다: {criterion_id!r}")
        seen_ids.add(criterion_id)

        if not isinstance(criterion_name, str) or not criterion_name:
            raise ValueError(f"criterion_name이 올바르지 않습니다: {item!r}")
        if description is not None and (not isinstance(description, str) or not description.strip()):
            raise ValueError(f"description이 올바르지 않습니다: {item!r}")
        if not isinstance(max_score, (int, float)) or max_score <= 0:
            raise ValueError(f"max_score가 올바르지 않습니다: {item!r}")
        if not isinstance(required_keywords, list) or not all(
            isinstance(keyword, str) and keyword.strip() for keyword in required_keywords
        ):
            raise ValueError(f"required_keywords가 올바르지 않습니다: {item!r}")
        if not isinstance(required_keyword_groups, list) or not all(
            isinstance(group, list)
            and group
            and all(isinstance(keyword, str) and keyword.strip() for keyword in group)
            for group in required_keyword_groups
        ):
            raise ValueError(f"required_keyword_groups가 올바르지 않습니다: {item!r}")

        if primary_persona_id not in committee:
            raise ValueError(
                f"primary_persona_id({primary_persona_id!r})가 committee({sorted(committee)})에 없습니다."
            )
        if secondary_persona_id is not None and secondary_persona_id not in committee:
            raise ValueError(
                f"secondary_persona_id({secondary_persona_id!r})가 committee({sorted(committee)})에 없습니다."
            )
        if primary_perspective_id not in perspective_whitelist.get(primary_persona_id, set()):
            raise ValueError(
                f"primary_perspective_id({primary_perspective_id!r})가 {primary_persona_id!r}의 "
                f"evaluation_perspectives에 없습니다."
            )

        normalized_item = {
                "criterion_id": criterion_id,
                "criterion_name": criterion_name,
                "max_score": max_score,
                "required": bool(item.get("required", True)),
                "source": "notice",
                "weight_origin": "notice_extracted",
                "weight_origin_note": "공고문에서 LLM으로 자동 추출한 평가항목입니다.",
                "primary_persona_id": primary_persona_id,
                "primary_perspective_id": primary_perspective_id,
                "secondary_persona_id": secondary_persona_id,
            }
        if description:
            normalized_item["description"] = description.strip()
        if required_keywords:
            normalized_item["required_keywords"] = required_keywords
        if required_keyword_groups:
            normalized_item["required_keyword_groups"] = required_keyword_groups
        normalized.append(normalized_item)

    total_max_score = sum(item["max_score"] for item in normalized)
    normalized_bonus_rules: list[dict[str, Any]] = []
    seen_bonus_ids: set[str] = set()
    for rule in bonus_rules or []:
        bonus_id = rule.get("bonus_id")
        name = rule.get("name")
        points = rule.get("points")
        if not isinstance(bonus_id, str) or not bonus_id or bonus_id in seen_bonus_ids:
            raise ValueError(f"bonus_id가 올바르지 않거나 중복되었습니다: {rule!r}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"가점 name이 올바르지 않습니다: {rule!r}")
        if not isinstance(points, (int, float)) or points <= 0:
            raise ValueError(f"가점 points가 올바르지 않습니다: {rule!r}")
        seen_bonus_ids.add(bonus_id)
        normalized_bonus_rules.append(
            {
                "bonus_id": bonus_id,
                "name": name,
                "points": points,
                "description": str(rule.get("description") or ""),
                "evidence_keywords": [
                    str(keyword)
                    for keyword in (rule.get("evidence_keywords") or [])
                    if str(keyword).strip()
                ],
                # 가점은 데이터 언급만으로 확정하지 않고 증빙 확인을 기본으로 한다.
                "requires_verification": bool(rule.get("requires_verification", True)),
            }
        )
    if not isinstance(bonus_max_score, (int, float)) or bonus_max_score < 0:
        raise ValueError(f"bonus_max_score가 올바르지 않습니다: {bonus_max_score!r}")
    if normalized_bonus_rules and bonus_max_score <= 0:
        raise ValueError("bonus_rules가 있으면 bonus_max_score는 0보다 커야 합니다.")

    result = {
        **base_mapping,
        "meta": {
            **base_mapping["meta"],
            "source_document_id": source_document_id,
            "source_document_ids": source_document_ids or [source_document_id],
            "dynamic": True,
            "rubric_extraction_version": 2,
        },
        "total_max_score": total_max_score,
        "rubric": normalized,
    }
    if normalized_bonus_rules:
        result["bonus_rules"] = normalized_bonus_rules
        result["bonus_max_score"] = bonus_max_score
    return result


def build_routing(mapping: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    """criterion_id -> {"primary": persona_id, "secondary": persona_id 또는 None} 매핑을 만든다.

    노드/그래프 조립이 "이 기준은 누가 채점하는가"를 알아야 할 때 쓴다(현재 reviewer
    노드는 위원 전체가 rubric 전체를 보고 자기 전문 범위만 채점하므로 라우팅을 강제하진
    않지만, 배정 근거를 추적하거나 향후 위원별 rubric 부분집합을 넘길 때 재사용한다).
    """
    return {
        item["criterion_id"]: {
            "primary": item["primary_persona_id"],
            "secondary": item.get("secondary_persona_id"),
        }
        for item in mapping["rubric"]
    }
