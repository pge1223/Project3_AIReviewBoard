"""
Persona -> RAG-003 role_id Resolver
========================================
회의 rubric_mapping의 persona_id(가은 PER-002 산출물, 예: business_strategy)와
RAG-003 RoleRegistry의 role_id(ai.rag.role_retrieval.roles, 예: finance)는 서로 다른
값 체계라 자동으로 대응되지 않는다. 이 모듈이 domain별 매핑을 한 곳에서만 관리한다.

매핑이 없는 persona_id를 조용히 role_id=None(semantic-only 검색)으로 넘기지 않는다 —
기본 정책은 strict(PersonaRoleMappingError)이며, allow_semantic_fallback=True를 명시적으로
설정했을 때만 완화되고 그때도 반드시 warning 로그를 남긴다(팀 정책, 2026-07-16 확정).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class PersonaRoleMappingError(ValueError):
    """domain/persona_id(/criterion_id)에 대응하는 RAG-003 role_id 매핑을 찾지 못했을 때 발생."""


# domain -> persona_id -> role_id (2순위: persona 단위 기본 매핑).
# competition 도메인은 committee 4인(ai/meeting/personas/rubric_mapping_competition.json)에
# 대해 팀이 확정한 매핑이다:
#   creativity_originality   -> marketing   (차별성, 사용자 가치, 경쟁 대비 독창성)
#   technical_feasibility    -> technology  (기술 구성, 구현 가능성, 보안·운영)
#   business_strategy        -> finance     (수익 모델, 비용, 예산, 사업 리스크)
#   presentation_completeness -> planning   (문서 구조, 논리 연결, 일정, 지표, 완성도)
#
# government_support(rubric_mapping_government_support.json)는 committee가
# policy_fit/business_strategy/technical_feasibility/budget_execution으로 competition과
# 다르다 — 팀이 확정한 매핑(2026-07-17):
#   policy_fit             -> policy            (정책 목표, 공공성, 지원요건, 정책 부합성)
#   business_strategy      -> planning          (사업 필요성 등 기획·사업 관점 criterion 기본)
#   technical_feasibility  -> technology        (기술성 및 수행역량)
#   budget_execution       -> budget_execution  (예산 편성, 집행계획, 정산, 위험 대응)
# business_strategy는 criterion에 따라 성격이 갈려(사업 필요성 vs 사업화 가능성=시장·차별성)
# feasibility criterion만 _CRITERION_OVERRIDES로 marketing으로 재배정한다.
_DOMAIN_PERSONA_ROLE_MAPPING: dict[str, dict[str, str]] = {
    "competition": {
        "creativity_originality": "marketing",
        "technical_feasibility": "technology",
        "business_strategy": "finance",
        "presentation_completeness": "planning",
        "planning_expert": "planning",
        "dev_expert": "technology",
    },
    "government_support": {
        "policy_fit": "policy",
        "business_strategy": "planning",
        "technical_feasibility": "technology",
        "budget_execution": "budget_execution",
    },
}

# domain -> persona_id -> criterion_id -> role_id (1순위: criterion 단위 override).
# 동일 persona가 성격이 다른 criterion을 맡게 되면(예: business_strategy가 재무성 criterion과
# 정책부합성 criterion을 동시에 담당) 여기에 override를 추가한다.
# government_support의 business_strategy는 necessity(사업 필요성, 문제 정의 -> planning
# 기본 매핑 그대로)와 feasibility(사업화 가능성, 시장·차별성 성격 -> marketing)를 동시에
# 맡아 feasibility criterion만 override한다(rubric_mapping_government_support.json 기준).
_CRITERION_OVERRIDES: dict[str, dict[str, dict[str, str]]] = {
    "government_support": {
        "business_strategy": {
            "feasibility": "marketing",
        },
    },
}


@dataclass(frozen=True)
class RoleMappingConfig:
    """resolve_role_id()의 매핑 누락 처리 정책. 기본값은 semantic-only fallback 비활성화(strict)."""

    allow_semantic_fallback: bool = False


def resolve_role_id(
    *,
    domain: str,
    persona_id: str,
    criterion_id: Optional[str] = None,
    config: Optional[RoleMappingConfig] = None,
) -> Optional[str]:
    """domain·persona_id(·criterion_id)에 대응하는 RAG-003 role_id를 찾는다.

    우선순위:
      1) domain + persona_id + criterion_id override (_CRITERION_OVERRIDES)
      2) domain + persona_id 기본 매핑 (_DOMAIN_PERSONA_ROLE_MAPPING)
      3) 매핑 없음 -> config.allow_semantic_fallback이 True면 None을 반환하며 warning 로그를
         남기고(semantic-only 검색으로 대체), False(기본값)면 PersonaRoleMappingError.
    """
    effective_config = config or RoleMappingConfig()

    criterion_override = _CRITERION_OVERRIDES.get(domain, {}).get(persona_id, {})
    if criterion_id is not None and criterion_id in criterion_override:
        return criterion_override[criterion_id]

    domain_mapping = _DOMAIN_PERSONA_ROLE_MAPPING.get(domain)
    if domain_mapping is not None and persona_id in domain_mapping:
        return domain_mapping[persona_id]

    if effective_config.allow_semantic_fallback:
        logger.warning(
            "[PERSONA_ROLE_MAPPING_FALLBACK] domain=%s persona_id=%s criterion_id=%s -- "
            "role_id 매핑을 찾지 못해 semantic-only(role_id=None) 검색으로 대체합니다.",
            domain,
            persona_id,
            criterion_id,
        )
        return None

    raise PersonaRoleMappingError(
        f"domain={domain!r} persona_id={persona_id!r}(criterion_id={criterion_id!r})에 대응하는 "
        "RAG-003 role_id 매핑이 없습니다. role_mapping.py의 _DOMAIN_PERSONA_ROLE_MAPPING/"
        "_CRITERION_OVERRIDES에 추가하거나, RoleMappingConfig(allow_semantic_fallback=True)로 "
        "명시적으로 완화하세요."
    )


def supported_domains() -> list[str]:
    """persona_id -> role_id 매핑이 하나 이상 정의된 domain 목록."""
    return sorted(_DOMAIN_PERSONA_ROLE_MAPPING)


__all__ = [
    "PersonaRoleMappingError",
    "RoleMappingConfig",
    "resolve_role_id",
    "supported_domains",
]
