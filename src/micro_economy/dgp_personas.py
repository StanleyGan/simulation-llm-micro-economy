"""Persona archetypes with feature sampling distributions.

Each archetype defines ranges for the 4 features. We sample N agents per archetype,
producing a population with known feature distributions for ground-truth validation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from micro_economy.dgp import DGPFeatures
from micro_economy.models import Good


@dataclass
class ArchetypeDistribution:
    """Defines how to sample features for a persona archetype."""

    archetype: str
    risk_range: tuple[float, float]  # (low, high) for risk_appetite
    budget_range: tuple[float, float]  # (low, high) for starting budget
    preference_weights: dict[Good, tuple[float, float]]  # (low, high) per good, normalized after
    patience_range: tuple[float, float]  # (low, high) for patience
    production_skill: Good  # fixed per archetype


def _sample_preferences(weights: dict[Good, tuple[float, float]]) -> dict[Good, float]:
    """Sample preference weights from ranges and normalize to sum to 1."""
    raw = {g: random.uniform(lo, hi) for g, (lo, hi) in weights.items()}
    total = sum(raw.values())
    return {g: v / total for g, v in raw.items()}


def sample_agent(archetype: ArchetypeDistribution, agent_id: int) -> DGPFeatures:
    """Sample one agent from an archetype distribution."""
    return DGPFeatures(
        name=f"{archetype.archetype}_{agent_id}",
        archetype=archetype.archetype,
        risk_appetite=random.uniform(*archetype.risk_range),
        budget=random.uniform(*archetype.budget_range),
        preferences=_sample_preferences(archetype.preference_weights),
        patience=random.uniform(*archetype.patience_range),
        production_skill=archetype.production_skill,
    )


def sample_population(archetypes: list[ArchetypeDistribution], samples_per_archetype: int = 10) -> list[DGPFeatures]:
    """Sample a full population from multiple archetypes."""
    population = []
    for arch in archetypes:
        for i in range(samples_per_archetype):
            population.append(sample_agent(arch, i))
    return population


# ---------------------------------------------------------------------------
# 6 Archetypes matching our original simulation personas
# ---------------------------------------------------------------------------

# Alice — cautious farmer
CAUTIOUS_FARMER = ArchetypeDistribution(
    archetype="cautious_farmer",
    risk_range=(0.05, 0.25),  # very risk-averse
    budget_range=(120, 200),  # moderate budget
    preference_weights={
        Good.FOOD: (0.4, 0.6),  # strongly prefers food
        Good.TOOLS: (0.05, 0.15),
        Good.LUXURY: (0.02, 0.08),
        Good.MEDICINE: (0.2, 0.35),  # also values medicine
    },
    patience_range=(0.6, 0.9),  # very patient
    production_skill=Good.FOOD,
)

# Bob — aggressive merchant
AGGRESSIVE_MERCHANT = ArchetypeDistribution(
    archetype="aggressive_merchant",
    risk_range=(0.7, 0.95),  # very aggressive
    budget_range=(200, 320),  # higher budget
    preference_weights={
        Good.FOOD: (0.05, 0.15),
        Good.TOOLS: (0.15, 0.3),
        Good.LUXURY: (0.4, 0.6),  # loves luxury
        Good.MEDICINE: (0.05, 0.15),
    },
    patience_range=(0.1, 0.35),  # impatient, wants quick profits
    production_skill=Good.TOOLS,
)

# Clara — pragmatic doctor
PRAGMATIC_DOCTOR = ArchetypeDistribution(
    archetype="pragmatic_doctor",
    risk_range=(0.35, 0.55),  # moderate risk
    budget_range=(160, 240),  # moderate budget
    preference_weights={
        Good.FOOD: (0.15, 0.3),
        Good.TOOLS: (0.05, 0.15),
        Good.LUXURY: (0.05, 0.15),
        Good.MEDICINE: (0.4, 0.6),  # strongly prefers medicine
    },
    patience_range=(0.4, 0.65),  # moderately patient
    production_skill=Good.MEDICINE,
)

# Drake — shrewd speculator
SHREWD_SPECULATOR = ArchetypeDistribution(
    archetype="shrewd_speculator",
    risk_range=(0.8, 0.98),  # highest risk
    budget_range=(240, 400),  # high budget to speculate with
    preference_weights={
        Good.FOOD: (0.05, 0.1),
        Good.TOOLS: (0.1, 0.2),
        Good.LUXURY: (0.35, 0.55),  # targets luxury
        Good.MEDICINE: (0.2, 0.35),  # and medicine
    },
    patience_range=(0.05, 0.25),  # very impatient, trades constantly
    production_skill=Good.LUXURY,
)

# Eve — fair toolmaker
FAIR_TOOLMAKER = ArchetypeDistribution(
    archetype="fair_toolmaker",
    risk_range=(0.15, 0.35),  # conservative
    budget_range=(120, 200),  # moderate budget
    preference_weights={
        Good.FOOD: (0.2, 0.35),
        Good.TOOLS: (0.3, 0.45),  # values tools
        Good.LUXURY: (0.05, 0.15),
        Good.MEDICINE: (0.1, 0.2),
    },
    patience_range=(0.5, 0.75),  # patient, steady
    production_skill=Good.TOOLS,
)

# Frank — survivalist
SURVIVALIST = ArchetypeDistribution(
    archetype="survivalist",
    risk_range=(0.02, 0.15),  # extremely risk-averse
    budget_range=(80, 160),  # lower budget (doesn't trust money)
    preference_weights={
        Good.FOOD: (0.4, 0.55),  # hoards food
        Good.TOOLS: (0.05, 0.15),
        Good.LUXURY: (0.01, 0.05),  # doesn't care about luxury
        Good.MEDICINE: (0.3, 0.45),  # hoards medicine
    },
    patience_range=(0.7, 0.95),  # extremely patient, waits it out
    production_skill=Good.FOOD,
)

# Registry
ALL_ARCHETYPES = [
    CAUTIOUS_FARMER,
    AGGRESSIVE_MERCHANT,
    PRAGMATIC_DOCTOR,
    SHREWD_SPECULATOR,
    FAIR_TOOLMAKER,
    SURVIVALIST,
]
