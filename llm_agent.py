"""LLM-powered agent decision-making using OpenAI API."""

from __future__ import annotations
import json
import os
import math
import random
from typing import Optional

from models import Agent, Good, MarketOrder, MarketState, GOOD_LIST


def _poisson_sample(lam: float) -> int:
    """Simple Poisson sampling (matches DGP production)."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p < L:
            return k - 1


# ---------------------------------------------------------------------------
# Mock mode: when no API key is set, agents use simple heuristic strategies
# ---------------------------------------------------------------------------

def _mock_decision(agent: Agent, market: MarketState, round_num: int) -> list[MarketOrder]:
    """Simple heuristic trading — used when no API key is available."""
    orders: list[MarketOrder] = []
    persona = agent.persona

    # Produce goods based on skill
    production_good = persona.production_skill
    produce_qty = _poisson_sample(2)
    agent.inventory.add(production_good, produce_qty)

    # Sell surplus of production good
    held = agent.inventory.holdings.get(production_good, 0)
    if held > 3:
        sell_qty = random.randint(1, held - 2)
        min_price = market.prices[production_good] * (0.8 + persona.risk_tolerance * 0.3)
        orders.append(MarketOrder(
            agent_name=agent.name,
            action="sell",
            good=production_good.value,
            quantity=sell_qty,
            max_price=round(min_price, 2),
            reasoning=f"Selling surplus {production_good.value}",
        ))

    # Buy preferred goods if affordable
    for good in persona.preferred_goods:
        if good == production_good:
            continue
        price = market.prices[good]
        if agent.budget > price * 2:
            buy_qty = random.randint(1, min(3, int(agent.budget / price)))
            max_pay = price * (1.0 + persona.risk_tolerance * 0.2)
            orders.append(MarketOrder(
                agent_name=agent.name,
                action="buy",
                good=good.value,
                quantity=buy_qty,
                max_price=round(max_pay, 2),
                reasoning=f"Buying needed {good.value}",
            ))
            break  # one buy per round in mock mode

    thought = (f"[Round {round_num}] Produced {produce_qty} {production_good.value}. "
               f"Budget: ${agent.budget:.2f}. "
               f"Strategy: {'aggressive' if persona.risk_tolerance > 0.5 else 'conservative'}.")
    agent.thoughts.append(thought)
    return orders


# ---------------------------------------------------------------------------
# OpenAI API mode
# ---------------------------------------------------------------------------

_OPENAI_CLIENT = None

def _get_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


DECISION_SYSTEM_PROMPT = """\
You are {name}, a trader in a small marketplace economy.

Your personality: {personality}

RULES:
- You produce {production_skill} efficiently (you get free units each round).
- You prefer: {preferred_goods}.
- Risk tolerance: {risk_tolerance}/10.
- You can BUY or SELL goods. You cannot trade more than you have (for sells) or afford (for buys).
- TRADING CONSTRAINT: Each round you may do ONE of the following:
  (a) Trade in a single good (buy or sell one type of good), OR
  (b) Swap between two goods (sell one type, buy another type).
  You cannot trade more than 5 units of any single good per round.
- Think step-by-step about what trades benefit you, then output your decisions.

Respond with ONLY a JSON object (no markdown fences):
{{
  "thinking": "your brief internal reasoning (1-2 sentences)",
  "orders": [
    {{"action": "buy"|"sell", "good": "food"|"tools"|"luxury"|"medicine", "quantity": int, "max_price": float}}
  ]
}}

max_price = the most you'll pay (buy) or the minimum you'll accept (sell).
You may place 1-2 orders per round (one buy and/or one sell). Be strategic!
"""


def _llm_decision(agent: Agent, market: MarketState, round_num: int) -> list[MarketOrder]:
    """Use OpenAI to decide trades."""
    client = _get_client()
    persona = agent.persona

    # Produce goods
    production_good = persona.production_skill
    produce_qty = _poisson_sample(2)
    agent.inventory.add(production_good, produce_qty)

    # Build user prompt with current state
    recent_trades = market.trade_log[-10:] if market.trade_log else []
    trade_summary = "\n".join(
        f"  {t.buyer} bought {t.quantity} {t.good.value} from {t.seller} @ ${t.price:.2f}"
        for t in recent_trades
    ) or "  No recent trades."

    user_msg = f"""Round {round_num}. You just produced {produce_qty} {production_good.value}.

CURRENT MARKET PRICES:
{json.dumps(market.prices_dict(), indent=2)}

YOUR STATE:
- Budget: ${agent.budget:.2f}
- Inventory: {json.dumps(agent.inventory.to_dict())}

RECENT TRADES:
{trade_summary}

What do you want to do this round?"""

    system = DECISION_SYSTEM_PROMPT.format(
        name=persona.name,
        personality=persona.personality,
        production_skill=persona.production_skill.value,
        preferred_goods=", ".join(g.value for g in persona.preferred_goods),
        risk_tolerance=int(persona.risk_tolerance * 10),
    )
    if persona.strategy_guide:
        system += persona.strategy_guide

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)
        thinking = data.get("thinking", "")
        agent.thoughts.append(f"[Round {round_num}] {thinking}")

        orders = []
        for o in data.get("orders", []):
            orders.append(MarketOrder(
                agent_name=agent.name,
                action=o["action"],
                good=o["good"],
                quantity=int(o["quantity"]),
                max_price=float(o.get("max_price", 0)),
                reasoning=thinking,
            ))
        return orders

    except Exception as e:
        agent.thoughts.append(f"[Round {round_num}] LLM error: {e}. Falling back to heuristic.")
        return _mock_decision(agent, market, round_num)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_agent_decision(agent: Agent, market: MarketState, round_num: int) -> list[MarketOrder]:
    """Get trading decisions from an agent. Uses OpenAI if API key is set, else mock."""
    if os.environ.get("OPENAI_API_KEY"):
        return _llm_decision(agent, market, round_num)
    return _mock_decision(agent, market, round_num)
