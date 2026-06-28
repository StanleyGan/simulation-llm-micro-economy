"""Economic metrics for analyzing simulation outcomes."""

from __future__ import annotations

import math
from typing import Any

from micro_economy.models import Agent, Good, MarketState, Trade


def gini_coefficient(values: list[float]) -> float:
    """Compute Gini coefficient (0 = perfect equality, 1 = max inequality)."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    numerator = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
    denominator = n * sum(sorted_vals)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def price_volatility(price_history: list[dict[str, float]], good: str) -> float:
    """Coefficient of variation for a good's price over time."""
    prices = [h[good] for h in price_history if good in h]
    if len(prices) < 2:
        return 0.0
    mean = sum(prices) / len(prices)
    if mean == 0:
        return 0.0
    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
    return math.sqrt(variance) / mean


def price_volatility_all(price_history: list[dict[str, float]]) -> dict[str, float]:
    """Volatility for each good."""
    if not price_history:
        return {}
    goods = list(price_history[0].keys())
    return {g: round(price_volatility(price_history, g), 4) for g in goods}


def avg_price_volatility(price_history: list[dict[str, float]]) -> float:
    """Average volatility across all goods."""
    vols = price_volatility_all(price_history)
    if not vols:
        return 0.0
    return round(sum(vols.values()) / len(vols), 4)


def wealth_distribution(agents: list[Agent], prices: dict[Good, float]) -> list[float]:
    """Get net worth of each agent."""
    return [a.net_worth(prices) for a in agents]


def wealth_stats(agents: list[Agent], prices: dict[Good, float]) -> dict[str, float]:
    """Summary statistics on wealth."""
    worths = wealth_distribution(agents, prices)
    if not worths:
        return {}
    return {
        "min": round(min(worths), 2),
        "max": round(max(worths), 2),
        "mean": round(sum(worths) / len(worths), 2),
        "spread": round(max(worths) - min(worths), 2),
        "gini": round(gini_coefficient(worths), 4),
    }


def trade_volume_stats(trade_log: list[Trade]) -> dict[str, Any]:
    """Aggregate trade statistics."""
    if not trade_log:
        return {"total_trades": 0, "total_value": 0, "by_good": {}}

    total_value = sum(t.price * t.quantity for t in trade_log)
    by_good: dict[str, dict] = {}
    for t in trade_log:
        g = t.good.value
        if g not in by_good:
            by_good[g] = {"count": 0, "volume": 0, "total_value": 0.0}
        by_good[g]["count"] += 1
        by_good[g]["volume"] += t.quantity
        by_good[g]["total_value"] += t.price * t.quantity

    for g in by_good:
        by_good[g]["total_value"] = round(by_good[g]["total_value"], 2)

    return {
        "total_trades": len(trade_log),
        "total_value": round(total_value, 2),
        "by_good": by_good,
    }


def market_efficiency(price_history: list[dict[str, float]]) -> float:
    """Measure how quickly prices stabilize. Lower = more efficient.

    Uses mean absolute round-over-round price change in the second half
    of the simulation, normalized by average price level.
    """
    if len(price_history) < 4:
        return 0.0

    half = len(price_history) // 2
    later = price_history[half:]
    goods = list(later[0].keys())

    total_change = 0.0
    total_price = 0.0
    count = 0

    for i in range(1, len(later)):
        for g in goods:
            total_change += abs(later[i][g] - later[i - 1][g])
            total_price += later[i][g]
            count += 1

    if total_price == 0:
        return 0.0
    return round(total_change / total_price, 4)


def compute_all_metrics(
    agents: list[Agent],
    market: MarketState,
) -> dict[str, Any]:
    """Compute all metrics for a completed simulation run."""
    return {
        "wealth": wealth_stats(agents, market.prices),
        "gini": round(gini_coefficient(wealth_distribution(agents, market.prices)), 4),
        "price_volatility": price_volatility_all(market.price_history),
        "avg_volatility": avg_price_volatility(market.price_history),
        "trade_volume": trade_volume_stats(market.trade_log),
        "market_efficiency": market_efficiency(market.price_history),
        "final_prices": market.prices_dict(),
    }
