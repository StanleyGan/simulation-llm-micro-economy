"""Tests for prompt translation from DGP features to LLM personas."""

from __future__ import annotations

from micro_economy.dgp import DGPFeatures
from micro_economy.models import Good
from micro_economy.prompt_translator import (
    features_to_narrative_prompt,
    features_to_numeric_prompt,
    features_to_persona,
)


def _make_features(**overrides) -> DGPFeatures:
    defaults = {
        "name": "test_agent_0",
        "archetype": "test",
        "risk_appetite": 0.8,
        "budget": 200.0,
        "preferences": {Good.FOOD: 0.4, Good.TOOLS: 0.3, Good.LUXURY: 0.2, Good.MEDICINE: 0.1},
        "patience": 0.3,
        "production_skill": Good.FOOD,
    }
    defaults.update(overrides)
    return DGPFeatures(**defaults)


class TestNumericPrompt:
    def test_contains_trait_labels(self):
        prompt = features_to_numeric_prompt(_make_features())
        assert "Risk appetite" in prompt
        assert "Patience" in prompt
        assert "Good preferences" in prompt

    def test_drop_risk(self):
        prompt = features_to_numeric_prompt(_make_features(), drop_feature="risk")
        assert "Risk appetite" not in prompt
        assert "Patience" in prompt

    def test_drop_patience(self):
        prompt = features_to_numeric_prompt(_make_features(), drop_feature="patience")
        assert "Patience" not in prompt
        assert "Risk appetite" in prompt

    def test_drop_preferences(self):
        prompt = features_to_numeric_prompt(_make_features(), drop_feature="preferences")
        assert "Good preferences" not in prompt

    def test_drop_production_skill(self):
        prompt = features_to_numeric_prompt(_make_features(), drop_feature="production_skill")
        assert "produce" not in prompt.lower()

    def test_display_name_override(self):
        prompt = features_to_numeric_prompt(_make_features(), display_name="Agent_A_0")
        assert "Agent_A_0" in prompt
        assert "test_agent_0" not in prompt


class TestNarrativePrompt:
    def test_high_risk_aggressive_language(self):
        prompt = features_to_narrative_prompt(_make_features(risk_appetite=0.9))
        assert "aggressive" in prompt.lower() or "bold" in prompt.lower()

    def test_low_risk_cautious_language(self):
        prompt = features_to_narrative_prompt(_make_features(risk_appetite=0.1))
        assert "cautious" in prompt.lower()

    def test_drop_feature_omits_section(self):
        prompt = features_to_narrative_prompt(_make_features(), drop_feature="risk")
        assert "cautious" not in prompt.lower()
        assert "aggressive" not in prompt.lower()


class TestFeaturesToPersona:
    def test_neutral_label_overrides_name(self):
        persona = features_to_persona(_make_features(), neutral_label="Agent_C_3")
        assert persona.name == "Agent_C_3"
        assert "Agent_C_3" in persona.personality

    def test_strategy_guide_included(self):
        persona = features_to_persona(_make_features(), with_strategy=True)
        assert persona.strategy_guide != ""
        assert "STRATEGY GUIDE" in persona.strategy_guide

    def test_strategy_guide_excluded_by_default(self):
        persona = features_to_persona(_make_features())
        assert persona.strategy_guide == ""
