"""Data-Generating Process (DGP) — utility-maximizing agents with known causal structure.

Each agent is defined by numeric features (risk, budget, preferences, patience).
Decisions are made by maximizing CRRA utility subject to budget/inventory constraints.
This produces ground-truth outcomes for validating LLM persona simulations.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

from micro_economy.models import GOOD_LIST, Good, MarketOrder, MarketState

# ---------------------------------------------------------------------------
# DGP Agent definition
# ---------------------------------------------------------------------------


@dataclass
class DGPFeatures:
    """Numeric features that fully define an agent's behavior."""

    name: str
    archetype: str  # e.g. "aggressive", "cautious"
    risk_appetite: float  # r ∈ [0, 1], higher = more aggressive
    budget: float  # starting money
    preferences: dict[Good, float]  # weights over goods, sum to 1
    patience: float  # δ ∈ [0, 1], higher = more patient
    production_skill: Good  # which good they produce

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "archetype": self.archetype,
            "risk_appetite": round(self.risk_appetite, 4),
            "budget": round(self.budget, 2),
            "preferences": {g.value: round(v, 4) for g, v in self.preferences.items()},
            "patience": round(self.patience, 4),
            "production_skill": self.production_skill.value,
        }


@dataclass
class DGPAgent:
    """An agent that makes decisions via utility maximization."""

    features: DGPFeatures
    budget: float = 0.0
    inventory: dict[Good, int] = field(default_factory=lambda: {g: 0 for g in Good})
    wealth_history: list[float] = field(default_factory=list)
    decision_history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.budget == 0.0:
            self.budget = self.features.budget

        # Large starting inventory of production good (surplus to sell)
        self.inventory[self.features.production_skill] = random.randint(8, 15)
        # Small amounts of other goods
        for g in GOOD_LIST:
            if g != self.features.production_skill:
                self.inventory[g] = random.randint(0, 2)

    @property
    def name(self) -> str:
        return self.features.name

    def net_worth(self, prices: dict[Good, float]) -> float:
        inv_value = sum(prices.get(g, 0) * q for g, q in self.inventory.items())
        return self.budget + inv_value


# ---------------------------------------------------------------------------
# CRRA Utility
# ---------------------------------------------------------------------------


def crra_utility(quantity: float, risk: float) -> float:
    """CRRA utility for a single good.

    U(x) = x^(1-r) / (1-r)  for r != 1
    U(x) = ln(x)             for r == 1
    """
    if quantity <= 0:
        return 0.0
    if abs(risk - 1.0) < 1e-6:
        return math.log(quantity)
    exponent = 1.0 - risk
    return (quantity**exponent) / exponent


def total_utility(inventory: dict[Good, int], preferences: dict[Good, float], risk: float) -> float:
    """Total utility across all goods: Σⱼ pⱼ · U(xⱼ)."""
    total = 0.0
    for good in Good:
        qty = inventory.get(good, 0)
        pref = preferences.get(good, 0.0)
        if qty > 0 and pref > 0:
            total += pref * crra_utility(qty, risk)
    return total


# ---------------------------------------------------------------------------
# Optimal trade finder
# ---------------------------------------------------------------------------

MAX_TRADE_PER_GOOD = 5  # cap per good per round to keep search tractable


def find_optimal_trade(
    agent: DGPAgent,
    prices: dict[Good, float],
    noise_std: float = 0.05,
) -> dict[Good, int]:
    """Find the trade vector that maximizes utility.

    Enumerates feasible trade combinations (bounded by MAX_TRADE_PER_GOOD)
    and picks the one with highest utility.

    Returns dict mapping Good -> int (positive = buy, negative = sell).
    """
    features = agent.features
    current_utility = total_utility(agent.inventory, features.preferences, features.risk_appetite)

    # Noise term for bounded rationality
    budget_noise = 1.0 + random.gauss(0, noise_std)
    effective_budget = agent.budget * max(0.5, budget_noise)

    best_trade = {g: 0 for g in Good}
    best_utility = current_utility

    # Generate candidate trades for each good
    # For efficiency, consider each good independently first, then combine top candidates
    good_candidates: dict[Good, list[int]] = {}
    for good in Good:
        held = agent.inventory.get(good, 0)
        max_sell = min(held, MAX_TRADE_PER_GOOD)
        max_buy = min(MAX_TRADE_PER_GOOD, int(effective_budget / max(prices[good], 0.01)))
        good_candidates[good] = list(range(-max_sell, max_buy + 1))

    # For tractability: evaluate single-good trades first, then try pairwise
    # Phase 1: single-good trades
    for good in Good:
        for qty in good_candidates[good]:
            if qty == 0:
                continue
            trade = {g: 0 for g in Good}
            trade[good] = qty
            util = _evaluate_trade(agent, trade, prices, effective_budget)
            if util is not None and util > best_utility:
                best_utility = util
                best_trade = trade.copy()

    # Phase 2: pairwise trades (sell one good, buy another)
    for sell_good, buy_good in itertools.permutations(Good, 2):
        for sell_qty in range(1, min(agent.inventory.get(sell_good, 0), MAX_TRADE_PER_GOOD) + 1):
            sell_revenue = sell_qty * prices[sell_good]
            max_buy = min(MAX_TRADE_PER_GOOD, int((effective_budget + sell_revenue) / max(prices[buy_good], 0.01)))
            for buy_qty in range(1, max_buy + 1):
                trade = {g: 0 for g in Good}
                trade[sell_good] = -sell_qty
                trade[buy_good] = buy_qty
                util = _evaluate_trade(agent, trade, prices, effective_budget)
                if util is not None and util > best_utility:
                    best_utility = util
                    best_trade = trade.copy()

    return best_trade


def _evaluate_trade(
    agent: DGPAgent,
    trade: dict[Good, int],
    prices: dict[Good, float],
    effective_budget: float,
) -> float | None:
    """Evaluate a candidate trade. Returns utility if feasible, None if not.

    Includes cash utility so spending money has a real opportunity cost.
    """
    features = agent.features
    new_inventory = {}
    cost = 0.0

    for good in Good:
        new_qty = agent.inventory.get(good, 0) + trade.get(good, 0)
        if new_qty < 0:
            return None  # can't sell more than we have
        new_inventory[good] = new_qty

        qty_change = trade.get(good, 0)
        if qty_change > 0:
            cost += qty_change * prices[good]
        elif qty_change < 0:
            cost += qty_change * prices[good]  # negative = revenue

    if cost > effective_budget:
        return None  # can't afford

    new_budget = agent.budget - cost

    # Inventory utility
    immediate_util = total_utility(new_inventory, features.preferences, features.risk_appetite)

    # Cash utility: money has value (can buy things later)
    # Scale by patience — patient agents value savings more
    cash_weight = 0.3 + features.patience * 0.4  # 0.3 to 0.7
    if new_budget > 0:
        cash_util = cash_weight * crra_utility(new_budget / 10.0, features.risk_appetite)
    else:
        cash_util = 0.0

    # Patient agents get a bonus for holding diverse inventory (proxy for future value)
    diversity_bonus = 0.0
    if features.patience > 0.3:
        non_zero = sum(1 for g in Good if new_inventory.get(g, 0) > 0)
        diversity_bonus = features.patience * 0.1 * non_zero

    return immediate_util + cash_util + diversity_bonus


# ---------------------------------------------------------------------------
# Generate orders from optimal trade
# ---------------------------------------------------------------------------


def generate_orders(
    agent: DGPAgent,
    trade: dict[Good, int],
    prices: dict[Good, float],
) -> list[MarketOrder]:
    """Convert an optimal trade vector into market orders with limit prices."""
    orders = []
    r = agent.features.risk_appetite

    for good in Good:
        qty = trade.get(good, 0)
        if qty > 0:
            # Buying: max price scales with risk appetite
            max_price = prices[good] * (1.0 + r * 0.2)
            orders.append(
                MarketOrder(
                    agent_name=agent.name,
                    action="buy",
                    good=good.value,
                    quantity=qty,
                    max_price=round(max_price, 2),
                    reasoning="DGP: utility-maximizing buy",
                )
            )
        elif qty < 0:
            # Selling: min price scales inversely with risk appetite
            min_price = prices[good] * (1.0 - r * 0.2)
            orders.append(
                MarketOrder(
                    agent_name=agent.name,
                    action="sell",
                    good=good.value,
                    quantity=abs(qty),
                    max_price=round(min_price, 2),
                    reasoning="DGP: utility-maximizing sell",
                )
            )

    return orders


# ---------------------------------------------------------------------------
# DGP decision function (drop-in replacement for LLM decisions)
# ---------------------------------------------------------------------------


def dgp_decision(agent: DGPAgent, market: MarketState, round_num: int) -> list[MarketOrder]:
    """Make a trading decision using the DGP utility maximization.

    This function has the same interface as get_agent_decision() in llm_agent.py,
    so it can be used as a drop-in replacement in the simulation loop.
    """
    # Step 1: Production
    production_good = agent.features.production_skill
    produce_qty = _poisson_sample(2)  # λ=2
    agent.inventory[production_good] = agent.inventory.get(production_good, 0) + produce_qty

    # Step 2: Find optimal trade
    trade = find_optimal_trade(agent, market.prices)

    # Step 3: Generate orders
    orders = generate_orders(agent, trade, market.prices)

    # Record decision
    agent.decision_history.append(
        {
            "round": round_num,
            "produced": produce_qty,
            "production_good": production_good.value,
            "trade": {g.value: v for g, v in trade.items()},
            "orders": len(orders),
        }
    )

    return orders


def _poisson_sample(lam: float) -> int:
    """Simple Poisson sampling."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p < L:
            return k - 1
