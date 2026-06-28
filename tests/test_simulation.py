"""Tests for the market matching engine."""

from __future__ import annotations

from micro_economy.models import (
    Agent,
    AgentPersona,
    Good,
    Inventory,
    MarketOrder,
    MarketState,
)
from micro_economy.simulation import execute_trades, match_orders


def _make_agent(name: str, budget: float = 200.0) -> Agent:
    persona = AgentPersona(
        name=name,
        personality="test",
        preferred_goods=[Good.FOOD],
        production_skill=Good.FOOD,
        risk_tolerance=0.5,
    )
    return Agent(persona=persona, budget=budget, inventory=Inventory())


class TestMatchOrders:
    def test_compatible_orders_produce_trade(self):
        buy = MarketOrder(agent_name="buyer", action="buy", good="food", quantity=2, max_price=12.0)
        sell = MarketOrder(agent_name="seller", action="sell", good="food", quantity=2, max_price=8.0)
        trades = match_orders([buy], [sell], MarketState(), round_num=1)
        assert len(trades) == 1
        assert trades[0].price == 10.0  # midpoint
        assert trades[0].quantity == 2

    def test_incompatible_prices_no_trade(self):
        buy = MarketOrder(agent_name="buyer", action="buy", good="food", quantity=2, max_price=5.0)
        sell = MarketOrder(agent_name="seller", action="sell", good="food", quantity=2, max_price=15.0)
        trades = match_orders([buy], [sell], MarketState(), round_num=1)
        assert len(trades) == 0

    def test_same_agent_no_trade(self):
        buy = MarketOrder(agent_name="alice", action="buy", good="food", quantity=2, max_price=15.0)
        sell = MarketOrder(agent_name="alice", action="sell", good="food", quantity=2, max_price=5.0)
        trades = match_orders([buy], [sell], MarketState(), round_num=1)
        assert len(trades) == 0

    def test_partial_fill(self):
        buy = MarketOrder(agent_name="buyer", action="buy", good="food", quantity=5, max_price=12.0)
        sell = MarketOrder(agent_name="seller", action="sell", good="food", quantity=2, max_price=8.0)
        trades = match_orders([buy], [sell], MarketState(), round_num=1)
        assert len(trades) == 1
        assert trades[0].quantity == 2


class TestExecuteTrades:
    def test_updates_budgets_and_inventory(self):
        buyer = _make_agent("buyer", budget=100.0)
        seller = _make_agent("seller", budget=50.0)
        seller.inventory.add(Good.FOOD, 5)

        from micro_economy.models import Trade

        trade = Trade(buyer="buyer", seller="seller", good=Good.FOOD, quantity=2, price=10.0, round_num=1)
        execute_trades([trade], {"buyer": buyer, "seller": seller})

        assert buyer.budget == 80.0
        assert seller.budget == 70.0
        assert buyer.inventory.holdings[Good.FOOD] == 2
        assert seller.inventory.holdings[Good.FOOD] == 3
