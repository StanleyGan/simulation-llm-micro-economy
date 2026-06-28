"""Persona configurations for experiments."""

from __future__ import annotations

from micro_economy.models import AgentPersona, Good


def _make_persona(name: str, personality: str, preferred: list[Good], skill: Good, risk: float) -> AgentPersona:
    return AgentPersona(
        name=name,
        personality=personality,
        preferred_goods=preferred,
        production_skill=skill,
        risk_tolerance=risk,
    )


# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------

ALL_CONSERVATIVE = [
    _make_persona(
        "Agent1",
        "Very cautious trader. Avoids risk, prefers small safe trades. Hoards essentials.",
        [Good.FOOD, Good.MEDICINE],
        Good.FOOD,
        0.1,
    ),
    _make_persona(
        "Agent2",
        "Risk-averse craftsman. Only trades when necessary. Values stability.",
        [Good.TOOLS, Good.FOOD],
        Good.TOOLS,
        0.15,
    ),
    _make_persona(
        "Agent3",
        "Careful doctor. Trades conservatively, never overpays.",
        [Good.MEDICINE, Good.FOOD],
        Good.MEDICINE,
        0.1,
    ),
    _make_persona(
        "Agent4",
        "Timid artisan. Produces luxury goods but sells reluctantly.",
        [Good.LUXURY, Good.TOOLS],
        Good.LUXURY,
        0.2,
    ),
    _make_persona(
        "Agent5", "Prudent farmer. Stockpiles food, trades only surplus.", [Good.FOOD, Good.MEDICINE], Good.FOOD, 0.1
    ),
    _make_persona(
        "Agent6",
        "Conservative toolmaker. Steady, predictable trading patterns.",
        [Good.TOOLS, Good.FOOD],
        Good.TOOLS,
        0.15,
    ),
]

ALL_AGGRESSIVE = [
    _make_persona(
        "Agent1",
        "Ruthless speculator. Buys everything cheap, manipulates prices. Maximum profit.",
        [Good.LUXURY, Good.MEDICINE],
        Good.FOOD,
        0.95,
    ),
    _make_persona(
        "Agent2",
        "Aggressive merchant. Takes huge risks for big returns. Loves arbitrage.",
        [Good.TOOLS, Good.LUXURY],
        Good.TOOLS,
        0.9,
    ),
    _make_persona(
        "Agent3",
        "Cutthroat dealer. Corners markets, hoards to drive up prices.",
        [Good.MEDICINE, Good.LUXURY],
        Good.MEDICINE,
        0.85,
    ),
    _make_persona(
        "Agent4",
        "Greedy tycoon. Buys in bulk, sells at premium. No mercy.",
        [Good.LUXURY, Good.TOOLS],
        Good.LUXURY,
        0.95,
    ),
    _make_persona(
        "Agent5", "Bold farmer. Overproduces and dumps to crash competitors.", [Good.FOOD, Good.TOOLS], Good.FOOD, 0.9
    ),
    _make_persona(
        "Agent6", "Reckless gambler. Makes wild bets on price movements.", [Good.TOOLS, Good.MEDICINE], Good.TOOLS, 0.85
    ),
]

MIXED_BALANCED = [
    _make_persona(
        "Alice",
        "Cautious farmer. Prioritizes food security, dislikes risk.",
        [Good.FOOD, Good.MEDICINE],
        Good.FOOD,
        0.2,
    ),
    _make_persona(
        "Bob", "Ambitious merchant. Loves luxury goods, takes risks.", [Good.LUXURY, Good.TOOLS], Good.TOOLS, 0.8
    ),
    _make_persona(
        "Clara", "Pragmatic doctor. Fair trader, builds relationships.", [Good.MEDICINE, Good.FOOD], Good.MEDICINE, 0.5
    ),
    _make_persona(
        "Drake", "Shrewd speculator. Buys low, sells high. No loyalty.", [Good.LUXURY, Good.MEDICINE], Good.LUXURY, 0.9
    ),
    _make_persona(
        "Eve",
        "Community toolmaker. Believes in fair trade, stabilizes prices.",
        [Good.TOOLS, Good.FOOD],
        Good.TOOLS,
        0.3,
    ),
    _make_persona(
        "Frank",
        "Survivalist. Stockpiles essentials, trades only when needed.",
        [Good.FOOD, Good.MEDICINE],
        Good.FOOD,
        0.1,
    ),
]

MOSTLY_CONSERVATIVE_ONE_SHARK = [
    _make_persona(
        "Agent1", "Very cautious farmer. Avoids risk at all costs.", [Good.FOOD, Good.MEDICINE], Good.FOOD, 0.1
    ),
    _make_persona(
        "Agent2", "Risk-averse craftsman. Steady and predictable.", [Good.TOOLS, Good.FOOD], Good.TOOLS, 0.15
    ),
    _make_persona(
        "Agent3", "Careful doctor. Never overpays, trades small.", [Good.MEDICINE, Good.FOOD], Good.MEDICINE, 0.1
    ),
    _make_persona(
        "Agent4", "Timid artisan. Sells reluctantly, buys carefully.", [Good.LUXURY, Good.TOOLS], Good.LUXURY, 0.2
    ),
    _make_persona("Agent5", "Prudent farmer. Only trades surplus.", [Good.FOOD, Good.MEDICINE], Good.FOOD, 0.1),
    _make_persona(
        "Shark",
        "Ruthless predator. Exploits naive traders. Corners markets. Maximum aggression.",
        [Good.LUXURY, Good.MEDICINE],
        Good.TOOLS,
        0.95,
    ),
]

ALL_COOPERATIVE = [
    _make_persona(
        "Agent1",
        "Altruistic farmer. Shares food freely, sells below market to help others.",
        [Good.FOOD, Good.MEDICINE],
        Good.FOOD,
        0.3,
    ),
    _make_persona(
        "Agent2",
        "Generous toolmaker. Prioritizes community welfare over profit.",
        [Good.TOOLS, Good.FOOD],
        Good.TOOLS,
        0.25,
    ),
    _make_persona(
        "Agent3",
        "Charitable doctor. Provides medicine cheaply. Values fairness above all.",
        [Good.MEDICINE, Good.FOOD],
        Good.MEDICINE,
        0.3,
    ),
    _make_persona(
        "Agent4",
        "Fair artisan. Prices luxury goods reasonably. Cares about equality.",
        [Good.LUXURY, Good.TOOLS],
        Good.LUXURY,
        0.2,
    ),
    _make_persona(
        "Agent5", "Community farmer. Grows extra food to stabilize supply.", [Good.FOOD, Good.TOOLS], Good.FOOD, 0.25
    ),
    _make_persona(
        "Agent6", "Egalitarian trader. Aims for fair deals for everyone.", [Good.TOOLS, Good.MEDICINE], Good.TOOLS, 0.3
    ),
]

SPECIALISTS = [
    _make_persona(
        "FoodKing",
        "Food monopolist. Only produces and trades food. Tries to control food prices.",
        [Good.FOOD],
        Good.FOOD,
        0.7,
    ),
    _make_persona(
        "ToolBaron", "Tool specialist. Dominates tools market. Refuses to diversify.", [Good.TOOLS], Good.TOOLS, 0.7
    ),
    _make_persona(
        "LuxuryLord", "Luxury specialist. Hoards and controls all luxury goods.", [Good.LUXURY], Good.LUXURY, 0.7
    ),
    _make_persona(
        "MedChief", "Medicine specialist. Controls medicine supply for leverage.", [Good.MEDICINE], Good.MEDICINE, 0.7
    ),
    _make_persona(
        "FoodKing2", "Second food producer. Competes with FoodKing for food dominance.", [Good.FOOD], Good.FOOD, 0.6
    ),
    _make_persona(
        "Generalist",
        "Jack of all trades. Buys whatever is cheap, sells whatever is dear.",
        [Good.FOOD, Good.TOOLS, Good.LUXURY, Good.MEDICINE],
        Good.TOOLS,
        0.5,
    ),
]


# Registry of all configurations
EXPERIMENT_CONFIGS: dict[str, list[AgentPersona]] = {
    "all_conservative": ALL_CONSERVATIVE,
    "all_aggressive": ALL_AGGRESSIVE,
    "mixed_balanced": MIXED_BALANCED,
    "conservative_plus_shark": MOSTLY_CONSERVATIVE_ONE_SHARK,
    "all_cooperative": ALL_COOPERATIVE,
    "specialists": SPECIALISTS,
}
