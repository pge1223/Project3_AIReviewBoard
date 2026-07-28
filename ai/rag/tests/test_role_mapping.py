"""
Unit Tests for ai.rag.orchestration.role_mapping
(순수 함수 — 실제 Chroma/KURE/LLM/LangGraph 없음)
"""

import logging

import pytest

from ai.rag.orchestration.role_mapping import (
    PersonaRoleMappingError,
    RoleMappingConfig,
    resolve_role_id,
    supported_domains,
)


class TestCompetitionMapping:
    @pytest.mark.parametrize(
        "persona_id, expected_role_id",
        [
            ("creativity_originality", "marketing"),
            ("technical_feasibility", "technology"),
            ("business_strategy", "finance"),
            ("presentation_completeness", "planning"),
            ("planning_expert", "planning"),
            ("dev_expert", "technology"),
        ],
    )
    def test_competition_persona_maps_to_expected_role(self, persona_id, expected_role_id):
        assert resolve_role_id(domain="competition", persona_id=persona_id) == expected_role_id

    def test_criterion_id_does_not_change_default_mapping_when_no_override(self):
        # override가 없으면 criterion_id를 무엇을 넘겨도 persona 기본 매핑으로 떨어진다.
        assert (
            resolve_role_id(domain="competition", persona_id="business_strategy", criterion_id="contribution")
            == "finance"
        )
        assert (
            resolve_role_id(domain="competition", persona_id="business_strategy", criterion_id="other_criterion")
            == "finance"
        )


class TestGovernmentSupportMapping:
    @pytest.mark.parametrize(
        "persona_id, expected_role_id",
        [
            ("policy_fit", "policy"),
            ("business_strategy", "planning"),
            ("technical_feasibility", "technology"),
            ("budget_execution", "budget_execution"),
        ],
    )
    def test_government_support_persona_maps_to_expected_role(self, persona_id, expected_role_id):
        assert resolve_role_id(domain="government_support", persona_id=persona_id) == expected_role_id

    def test_business_strategy_feasibility_criterion_overrides_to_marketing(self):
        assert (
            resolve_role_id(
                domain="government_support", persona_id="business_strategy", criterion_id="feasibility"
            )
            == "marketing"
        )

    def test_business_strategy_necessity_criterion_uses_default_planning(self):
        assert (
            resolve_role_id(
                domain="government_support", persona_id="business_strategy", criterion_id="necessity"
            )
            == "planning"
        )


class TestUnknownDomainAndPersona:
    def test_unknown_domain_raises_without_fallback(self):
        with pytest.raises(PersonaRoleMappingError):
            resolve_role_id(domain="unknown_domain", persona_id="business_strategy")

    def test_unknown_persona_in_known_domain_raises_without_fallback(self):
        with pytest.raises(PersonaRoleMappingError):
            resolve_role_id(domain="competition", persona_id="unknown_persona")

    def test_unknown_persona_in_government_support_raises_without_fallback(self):
        with pytest.raises(PersonaRoleMappingError):
            resolve_role_id(domain="government_support", persona_id="unknown_persona")


class TestStrictDefaultAndFallback:
    def test_default_config_does_not_allow_semantic_fallback(self):
        config = RoleMappingConfig()
        assert config.allow_semantic_fallback is False
        with pytest.raises(PersonaRoleMappingError):
            resolve_role_id(domain="unknown_domain", persona_id="unknown_persona", config=config)

    def test_fallback_enabled_returns_none_role_id(self):
        config = RoleMappingConfig(allow_semantic_fallback=True)
        role_id = resolve_role_id(domain="unknown_domain", persona_id="unknown_persona", config=config)
        assert role_id is None

    def test_fallback_enabled_logs_warning(self, caplog):
        config = RoleMappingConfig(allow_semantic_fallback=True)
        with caplog.at_level(logging.WARNING, logger="ai.rag.orchestration.role_mapping"):
            resolve_role_id(domain="unknown_domain", persona_id="unknown_persona", config=config)
        assert any("PERSONA_ROLE_MAPPING_FALLBACK" in record.message for record in caplog.records)

    def test_known_mapping_takes_priority_over_fallback(self):
        # 매핑이 있으면 allow_semantic_fallback=True여도 fallback을 타지 않고 실제 매핑을 쓴다.
        config = RoleMappingConfig(allow_semantic_fallback=True)
        role_id = resolve_role_id(domain="competition", persona_id="business_strategy", config=config)
        assert role_id == "finance"


class TestSupportedDomains:
    def test_competition_listed(self):
        assert "competition" in supported_domains()

    def test_government_support_listed(self):
        assert "government_support" in supported_domains()
