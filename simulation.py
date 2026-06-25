"""Core simulation engine — orchestrates rounds of trading."""

from __future__ import annotations
import asyncio
import random
from typing import AsyncGenerator, Any

from models import (
    Agent, AgentPersona, Good, Inventory, MarketOrder, MarketState, Trade,
    DEFAULT_PERSONAS, GOOD_LIST,
)
from llm_agent import get_agent_decision


def create_agents(personas: list[AgentPersona] | None = None, starting_budget: float = 200.0) -> list[Agent]:
    """Initialize agents with starting budgets and some initial inventory."""
    personas = personas or DEFAULT_PERSONAS
    agents = []
    for p in personas:
        inv = Inventory()
        # Give each agent some of their production good
        inv.add(p.production_skill, random.randint(3, 6))
        # Small random amounts of other goods
        for g in GOOD_LIST:
            if g != p.production_skill:
                inv.add(g, random.randint(0, 2))
        agents.append(Agent(persona=p, budget=starting_budget, inventory=inv))
    return agents


def match_orders(buys: list[MarketOrder], sells: list[MarketOrder], market: MarketState, round_num: int) -> list[Trade]:
    """Simple order matching: pair compatible buy/sell orders."""
    trades = []
    random.shuffle(buys)
    random.shuffle(sells)

    for buy in buys:
        for sell in sells:
            if sell.quantity <= 0:
                continue
            if buy.good != sell.good:
                continue
            if buy.agent_name == sell.agent_name:
                continue
            # Check price compatibility
            if buy.max_price and sell.max_price and buy.max_price < sell.max_price:
                continue
            # Trade at midpoint price
            if buy.max_price and sell.max_price:
                price = (buy.max_price + sell.max_price) / 2
            else:
                good_enum = Good(buy.good)
                price = market.prices[good_enum]

            qty = min(buy.quantity, sell.quantity)
            if qty <= 0:
                continue

            trades.append(Trade(
                buyer=buy.agent_name,
                seller=sell.agent_name,
                good=Good(buy.good),
                quantity=qty,
                price=round(price, 2),
                round_num=round_num,
            ))
            buy.quantity -= qty
            sell.quantity -= qty
            if buy.quantity <= 0:
                break

    return trades


def execute_trades(trades: list[Trade], agents_map: dict[str, Agent]):
    """Apply trades to agent budgets and inventories."""
    for t in trades:
        buyer = agents_map[t.buyer]
        seller = agents_map[t.seller]
        total_cost = t.price * t.quantity

        if buyer.budget >= total_cost and seller.inventory.holdings.get(t.good, 0) >= t.quantity:
            buyer.budget -= total_cost
            seller.budget += total_cost
            seller.inventory.remove(t.good, t.quantity)
            buyer.inventory.add(t.good, t.quantity)
            buyer.trade_history.append({"round": t.round_num, "action": "bought", "good": t.good.value, "qty": t.quantity, "price": t.price})
            seller.trade_history.append({"round": t.round_num, "action": "sold", "good": t.good.value, "qty": t.quantity, "price": t.price})


async def run_simulation(
    num_rounds: int = 20,
    agents: list[Agent] | None = None,
    delay: float = 1.0,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the simulation, yielding events for each round."""
    if agents is None:
        agents = create_agents()
    market = MarketState()
    agents_map = {a.name: a for a in agents}

    # Initial state
    yield _snapshot_event("init", 0, agents, market)

    for round_num in range(1, num_rounds + 1):
        # Collect orders from all agents
        all_orders: list[MarketOrder] = []
        round_thoughts: list[dict] = []

        for agent in agents:
            orders = get_agent_decision(agent, market, round_num)
            all_orders.extend(orders)
            if agent.thoughts:
                round_thoughts.append({
                    "agent": agent.name,
                    "thought": agent.thoughts[-1] if agent.thoughts else "",
                })

        # Update demand based on buy orders
        for order in all_orders:
            if order.action == "buy":
                good = Good(order.good)
                market.demand[good] = market.demand.get(good, 0) + order.quantity

        # Match and execute
        buys = [o for o in all_orders if o.action == "buy"]
        sells = [o for o in all_orders if o.action == "sell"]
        trades = match_orders(buys, sells, market, round_num)
        execute_trades(trades, agents_map)
        market.trade_log.extend(trades)

        # Update prices
        market.update_prices()
        market.record_prices()

        # Yield round event
        yield {
            "type": "round",
            "round": round_num,
            "trades": [
                {"buyer": t.buyer, "seller": t.seller, "good": t.good.value,
                 "quantity": t.quantity, "price": t.price}
                for t in trades
            ],
            "thoughts": round_thoughts,
            "prices": market.prices_dict(),
            "price_history": market.price_history,
            "agents": [a.to_state_dict(market.prices) for a in agents],
            "orders": [
                {"agent": o.agent_name, "action": o.action, "good": o.good,
                 "quantity": o.quantity, "max_price": o.max_price}
                for o in all_orders
            ],
        }

        await asyncio.sleep(delay)

    # Final summary
    yield {
        "type": "complete",
        "round": num_rounds,
        "prices": market.prices_dict(),
        "price_history": market.price_history,
        "agents": [a.to_state_dict(market.prices) for a in agents],
        "total_trades": len(market.trade_log),
    }


def run_simulation_sync(
    num_rounds: int = 20,
    personas: list[AgentPersona] | None = None,
    starting_budget: float = 200.0,
) -> tuple[list[Agent], MarketState]:
    """Synchronous simulation for batch experiments. Returns (agents, market)."""
    agents = create_agents(personas=personas, starting_budget=starting_budget)
    market = MarketState()
    agents_map = {a.name: a for a in agents}

    for round_num in range(1, num_rounds + 1):
        all_orders: list[MarketOrder] = []
        for agent in agents:
            orders = get_agent_decision(agent, market, round_num)
            all_orders.extend(orders)

        for order in all_orders:
            if order.action == "buy":
                good = Good(order.good)
                market.demand[good] = market.demand.get(good, 0) + order.quantity

        buys = [o for o in all_orders if o.action == "buy"]
        sells = [o for o in all_orders if o.action == "sell"]
        trades = match_orders(buys, sells, market, round_num)
        execute_trades(trades, agents_map)
        market.trade_log.extend(trades)
        market.update_prices()
        market.record_prices()

    return agents, market


def _snapshot_event(event_type: str, round_num: int, agents: list[Agent], market: MarketState) -> dict:
    return {
        "type": event_type,
        "round": round_num,
        "prices": market.prices_dict(),
        "price_history": market.price_history,
        "agents": [a.to_state_dict(market.prices) for a in agents],
    }
