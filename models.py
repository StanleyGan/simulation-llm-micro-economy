"""Data models for the micro-economy simulation."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random


class Good(str, Enum):
    FOOD = "food"
    TOOLS = "tools"
    LUXURY = "luxury"
    MEDICINE = "medicine"


GOOD_LIST = list(Good)


@dataclass
class AgentPersona:
    name: str
    personality: str  # fed to LLM as system context
    preferred_goods: list[Good]  # goods this agent values more
    production_skill: Good  # good this agent produces efficiently
    risk_tolerance: float  # 0-1, influences trading style
    strategy_guide: str = ""  # optional CRRA strategy rules appended to system prompt


@dataclass
class Inventory:
    holdings: dict[Good, int] = field(default_factory=lambda: {g: 0 for g in Good})

    def add(self, good: Good, qty: int):
        self.holdings[good] = self.holdings.get(good, 0) + qty

    def remove(self, good: Good, qty: int) -> bool:
        if self.holdings.get(good, 0) >= qty:
            self.holdings[good] -= qty
            return True
        return False

    def total_items(self) -> int:
        return sum(self.holdings.values())

    def to_dict(self) -> dict[str, int]:
        return {g.value: q for g, q in self.holdings.items()}


@dataclass
class Agent:
    persona: AgentPersona
    budget: float
    inventory: Inventory = field(default_factory=Inventory)
    trade_history: list[dict] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.persona.name

    def net_worth(self, prices: dict[Good, float]) -> float:
        inv_value = sum(prices.get(g, 0) * q for g, q in self.inventory.holdings.items())
        return self.budget + inv_value

    def to_state_dict(self, prices: dict[Good, float]) -> dict:
        return {
            "name": self.name,
            "budget": round(self.budget, 2),
            "inventory": self.inventory.to_dict(),
            "net_worth": round(self.net_worth(prices), 2),
        }


@dataclass
class MarketOrder:
    agent_name: str
    action: str  # "buy" or "sell"
    good: str
    quantity: int
    max_price: Optional[float] = None  # max willing to pay (buy) or min accept (sell)
    reasoning: str = ""


@dataclass
class Trade:
    buyer: str
    seller: str
    good: Good
    quantity: int
    price: float
    round_num: int


@dataclass
class MarketState:
    prices: dict[Good, float] = field(default_factory=lambda: {
        Good.FOOD: 10.0,
        Good.TOOLS: 25.0,
        Good.LUXURY: 50.0,
        Good.MEDICINE: 30.0,
    })
    supply: dict[Good, int] = field(default_factory=lambda: {g: 100 for g in Good})
    demand: dict[Good, int] = field(default_factory=lambda: {g: 0 for g in Good})
    price_history: list[dict[str, float]] = field(default_factory=list)
    trade_log: list[Trade] = field(default_factory=list)

    def record_prices(self):
        self.price_history.append({g.value: p for g, p in self.prices.items()})

    def update_prices(self):
        """Adjust prices based on supply/demand imbalance."""
        for good in Good:
            d = self.demand.get(good, 0)
            s = self.supply.get(good, 0)
            if s > 0:
                ratio = d / s
                # Price moves toward equilibrium
                adjustment = (ratio - 1.0) * 0.1
                adjustment += random.uniform(-0.02, 0.02)  # noise
                self.prices[good] = max(1.0, self.prices[good] * (1 + adjustment))
                self.prices[good] = round(self.prices[good], 2)
        # Reset demand counters
        self.demand = {g: 0 for g in Good}

    def prices_dict(self) -> dict[str, float]:
        return {g.value: p for g, p in self.prices.items()}


# --- Default agent personas ---

DEFAULT_PERSONAS = [
    AgentPersona(
        name="Alice",
        personality="A cautious farmer who prioritizes food security and steady income. "
                    "Dislikes risk and prefers small, safe trades. Tends to hoard food.",
        preferred_goods=[Good.FOOD, Good.MEDICINE],
        production_skill=Good.FOOD,
        risk_tolerance=0.2,
    ),
    AgentPersona(
        name="Bob",
        personality="An ambitious merchant who loves luxury goods and big deals. "
                    "Willing to take risks for high returns. Always looking for arbitrage.",
        preferred_goods=[Good.LUXURY, Good.TOOLS],
        production_skill=Good.TOOLS,
        risk_tolerance=0.8,
    ),
    AgentPersona(
        name="Clara",
        personality="A pragmatic doctor who values medicine above all. "
                    "Trades strategically but fairly. Believes in building long-term relationships.",
        preferred_goods=[Good.MEDICINE, Good.FOOD],
        production_skill=Good.MEDICINE,
        risk_tolerance=0.5,
    ),
    AgentPersona(
        name="Drake",
        personality="A shrewd speculator who tries to corner markets and manipulate prices. "
                    "Buys low, sells high. Has no loyalty to other traders.",
        preferred_goods=[Good.LUXURY, Good.MEDICINE],
        production_skill=Good.LUXURY,
        risk_tolerance=0.9,
    ),
    AgentPersona(
        name="Eve",
        personality="A community-minded toolmaker who believes in fair trade. "
                    "Tries to keep prices stable and help others. Will sacrifice profit for fairness.",
        preferred_goods=[Good.TOOLS, Good.FOOD],
        production_skill=Good.TOOLS,
        risk_tolerance=0.3,
    ),
    AgentPersona(
        name="Frank",
        personality="A survivalist who stockpiles essential goods (food, medicine). "
                    "Distrusts the market and only trades when necessary. Very conservative.",
        preferred_goods=[Good.FOOD, Good.MEDICINE],
        production_skill=Good.FOOD,
        risk_tolerance=0.1,
    ),
]
