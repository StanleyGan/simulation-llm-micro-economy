"""Tests for archetype feature sampling."""

from __future__ import annotations

import random

from micro_economy.dgp_personas import ALL_ARCHETYPES, sample_agent, sample_population
from micro_economy.models import Good


class TestSampleAgent:
    def test_name_format(self):
        random.seed(42)
        agent = sample_agent(ALL_ARCHETYPES[0], 3)
        assert agent.name == f"{ALL_ARCHETYPES[0].archetype}_3"

    def test_features_in_range(self):
        random.seed(42)
        arch = ALL_ARCHETYPES[0]
        agent = sample_agent(arch, 0)
        assert arch.risk_range[0] <= agent.risk_appetite <= arch.risk_range[1]
        assert arch.budget_range[0] <= agent.budget <= arch.budget_range[1]
        assert arch.patience_range[0] <= agent.patience <= arch.patience_range[1]
        assert agent.production_skill == arch.production_skill

    def test_preferences_sum_to_one(self):
        random.seed(42)
        agent = sample_agent(ALL_ARCHETYPES[1], 0)
        total = sum(agent.preferences.values())
        assert abs(total - 1.0) < 1e-6


class TestSamplePopulation:
    def test_correct_count(self):
        random.seed(42)
        pop = sample_population(ALL_ARCHETYPES, samples_per_archetype=5)
        assert len(pop) == len(ALL_ARCHETYPES) * 5

    def test_all_archetypes_present(self):
        random.seed(42)
        pop = sample_population(ALL_ARCHETYPES, samples_per_archetype=3)
        archetypes_found = {f.archetype for f in pop}
        expected = {a.archetype for a in ALL_ARCHETYPES}
        assert archetypes_found == expected

    def test_all_goods_in_preferences(self):
        random.seed(42)
        pop = sample_population(ALL_ARCHETYPES, samples_per_archetype=1)
        for f in pop:
            assert set(f.preferences.keys()) == set(Good)
