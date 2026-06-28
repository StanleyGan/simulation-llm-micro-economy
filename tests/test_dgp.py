"""Tests for DGP utility math and decision-making."""

from __future__ import annotations

import math
import random

import pytest

from micro_economy.dgp import DGPAgent, DGPFeatures, crra_utility, dgp_decision, total_utility
from micro_economy.models import Good, MarketState


def _make_features(**overrides) -> DGPFeatures:
    defaults = {
        "name": "test_0",
        "archetype": "test",
        "risk_appetite": 0.5,
        "budget": 200.0,
        "preferences": {Good.FOOD: 0.4, Good.TOOLS: 0.3, Good.LUXURY: 0.2, Good.MEDICINE: 0.1},
        "patience": 0.5,
        "production_skill": Good.FOOD,
    }
    defaults.update(overrides)
    return DGPFeatures(**defaults)


class TestCRRAUtility:
    def test_zero_quantity_returns_zero(self):
        assert crra_utility(0, 0.5) == 0.0

    def test_negative_quantity_returns_zero(self):
        assert crra_utility(-1, 0.5) == 0.0

    def test_risk_zero_linear(self):
        assert crra_utility(4.0, 0.0) == pytest.approx(4.0)

    def test_risk_one_log(self):
        assert crra_utility(math.e, 1.0) == pytest.approx(1.0)

    def test_diminishing_returns(self):
        u1 = crra_utility(1, 0.7)
        u5 = crra_utility(5, 0.7)
        u10 = crra_utility(10, 0.7)
        assert u5 - u1 > u10 - u5


class TestTotalUtility:
    def test_empty_inventory(self):
        prefs = {Good.FOOD: 0.5, Good.TOOLS: 0.5}
        assert total_utility({}, prefs, 0.5) == 0.0

    def test_positive_with_inventory(self):
        inv = {Good.FOOD: 5, Good.TOOLS: 3}
        prefs = {Good.FOOD: 0.6, Good.TOOLS: 0.4}
        assert total_utility(inv, prefs, 0.5) > 0.0

    def test_higher_preference_weight_increases_utility(self):
        inv = {Good.FOOD: 5}
        u_low = total_utility(inv, {Good.FOOD: 0.2, Good.TOOLS: 0.8}, 0.5)
        u_high = total_utility(inv, {Good.FOOD: 0.8, Good.TOOLS: 0.2}, 0.5)
        assert u_high > u_low


class TestDGPDecision:
    def test_returns_valid_orders(self):
        random.seed(42)
        features = _make_features()
        agent = DGPAgent(features=features)
        market = MarketState()
        orders = dgp_decision(agent, market, round_num=1)
        for o in orders:
            assert o.action in ("buy", "sell")
            assert o.quantity > 0

    def test_respects_budget(self):
        random.seed(42)
        features = _make_features(budget=10.0)
        agent = DGPAgent(features=features)
        market = MarketState()
        orders = dgp_decision(agent, market, round_num=1)
        sell_revenue = sum(o.quantity * market.prices[Good(o.good)] for o in orders if o.action == "sell")
        buy_cost = sum(o.quantity * market.prices[Good(o.good)] for o in orders if o.action == "buy")
        effective_budget = agent.budget + sell_revenue
        assert buy_cost <= effective_budget * 1.5
