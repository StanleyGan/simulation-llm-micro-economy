"""Translate DGP numeric features into LLM persona prompts.

Two styles:
  - Numeric: gives the LLM explicit numbers ("risk appetite: 8.2/10")
  - Narrative: translates features into natural language descriptions

Both use the same underlying DGPFeatures so comparisons are apples-to-apples.
"""

from __future__ import annotations

from models import AgentPersona, Good
from dgp import DGPFeatures


# ---------------------------------------------------------------------------
# Numeric prompt style
# ---------------------------------------------------------------------------

def _to_range(score: float, spread: float = 1.0) -> str:
    """Convert a score to a range string like '7-9 out of 10'.

    Clamps to [0, 10] and rounds to integers.
    """
    low = max(0, round(score - spread))
    high = min(10, round(score + spread))
    if low == high:
        return f"{low} out of 10"
    return f"{low}-{high} out of 10"


def features_to_numeric_prompt(
    f: DGPFeatures,
    display_name: str | None = None,
    drop_feature: str | None = None,
) -> str:
    """Translate features into a numeric prompt with ranges.

    Args:
        f: The DGP feature set
        display_name: Override for the agent name shown in the prompt
        drop_feature: Feature to omit ("risk", "patience", "preferences", "production_skill")
    """
    name = display_name or f.name
    risk_score = f.risk_appetite * 10
    patience_score = f.patience * 10

    lines = [f"You are {name}, a trader in a small marketplace.\n"]
    lines.append("YOUR TRAITS (use these to guide your decisions):")

    if drop_feature != "risk":
        lines.append(
            f"- Risk appetite: {_to_range(risk_score)} "
            f"({'very aggressive' if risk_score > 7 else 'moderate' if risk_score > 4 else 'very cautious'})"
        )

    if drop_feature != "patience":
        lines.append(
            f"- Patience: {_to_range(patience_score)} "
            f"({'very patient, prefers long-term gains' if patience_score > 7 else 'moderate patience' if patience_score > 4 else 'impatient, wants quick profits'})"
        )

    if drop_feature != "preferences":
        pref_lines = ", ".join(
            f"{g.value} {round(w * 100)}%"
            for g, w in sorted(f.preferences.items(), key=lambda x: -x[1])
        )
        lines.append(f"- Good preferences: {pref_lines}")

    if drop_feature != "production_skill":
        lines.append(f"- You produce {f.production_skill.value} efficiently.")

    lines.append(f"- Starting budget: ${f.budget:.0f}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Narrative prompt style
# ---------------------------------------------------------------------------

_RISK_DESCRIPTIONS = [
    (0.2, "extremely cautious. You hate taking risks and prefer the safety of what you know. "
          "You'd rather miss an opportunity than lose what you have."),
    (0.4, "fairly conservative. You take calculated risks only when the odds are clearly in your favor. "
          "Stability matters more than big gains."),
    (0.6, "moderately risk-tolerant. You're willing to take reasonable risks when you see opportunity, "
          "but you don't gamble recklessly."),
    (0.8, "quite aggressive. You actively seek out profitable opportunities and don't mind uncertainty. "
          "You believe higher risk means higher reward."),
    (1.0, "very aggressive and bold. You thrive on high-stakes trades and love the thrill of big moves. "
          "You'll take large positions to maximize your gains."),
]

_PATIENCE_DESCRIPTIONS = [
    (0.2, "You are very impatient — you want profits now, not later. "
          "Quick trades and immediate returns are your priority."),
    (0.4, "You lean toward short-term thinking. "
          "You'd rather lock in a decent gain today than wait for a better one tomorrow."),
    (0.6, "You have moderate patience. "
          "You can wait for the right opportunity but won't sit idle for too long."),
    (0.8, "You are quite patient. "
          "You're happy to hold your position and wait for the market to come to you."),
    (1.0, "You are extremely patient and think long-term. "
          "You believe in building wealth slowly and steadily, never rushing into trades."),
]



def _describe(value: float, descriptions: list[tuple[float, str]]) -> str:
    """Pick the description bracket for a value."""
    for threshold, desc in descriptions:
        if value <= threshold:
            return desc
    return descriptions[-1][1]


def _describe_preferences(prefs: dict[Good, float]) -> str:
    """Turn preference weights into natural language."""
    sorted_prefs = sorted(prefs.items(), key=lambda x: -x[1])
    top = sorted_prefs[0]
    second = sorted_prefs[1]
    bottom = sorted_prefs[-1]

    parts = [f"You value {top[0].value} the most"]
    if second[1] > 0.2:
        parts.append(f"followed closely by {second[0].value}")
    if bottom[1] < 0.1:
        parts.append(f"and have little interest in {bottom[0].value}")

    return ". ".join(parts) + "."


def features_to_narrative_prompt(
    f: DGPFeatures,
    display_name: str | None = None,
    drop_feature: str | None = None,
) -> str:
    """Translate features into a natural language narrative prompt.

    Args:
        f: The DGP feature set
        display_name: Override for the agent name shown in the prompt
        drop_feature: Feature to omit ("risk", "patience", "preferences", "production_skill")
    """
    name = display_name or f.name

    parts = [f"You are {name}, a trader in a small marketplace.\n"]
    parts.append("YOUR PERSONALITY:")

    if drop_feature != "risk":
        risk_desc = _describe(f.risk_appetite, _RISK_DESCRIPTIONS)
        parts.append(f"You are {risk_desc}")

    if drop_feature != "patience":
        patience_desc = _describe(f.patience, _PATIENCE_DESCRIPTIONS)
        parts.append(patience_desc)

    if drop_feature != "preferences":
        pref_desc = _describe_preferences(f.preferences)
        parts.append(pref_desc)

    if drop_feature != "production_skill":
        parts.append(f"You are skilled at producing {f.production_skill.value} and always have a steady supply of it.")

    parts.append(f"You start with a budget of ${f.budget:.0f}.")

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Strategy guide (CRRA optimization rules in natural language)
# ---------------------------------------------------------------------------

def build_strategy_guide(f: DGPFeatures) -> str:
    """Build agent-specific strategy rules derived from CRRA utility.

    Describes diminishing returns, cash management, diversity bonus,
    and trade evaluation in natural language using the agent's actual
    parameter values. Appended to the system prompt as a STRATEGY GUIDE.
    """
    # Diminishing returns intensity from risk appetite
    if f.risk_appetite > 0.7:
        dr_desc = (
            "You experience sharply diminishing returns from accumulating any single good. "
            "Having 2 of something is much better than 1, but having 5 vs 4 barely matters. "
            "Diversify across goods rather than stockpiling."
        )
    elif f.risk_appetite > 0.4:
        dr_desc = (
            "You experience moderate diminishing returns from accumulating goods. "
            "More of a good is always better, but the benefit tapers off. "
            "Balance between concentrating on favorites and diversifying."
        )
    else:
        dr_desc = (
            "You experience mild diminishing returns from accumulating goods. "
            "Holding larger quantities of preferred goods is reasonable."
        )

    # Cash value from patience
    cash_weight = 0.3 + f.patience * 0.4
    if f.patience > 0.7:
        cash_desc = (
            f"Holding cash is valuable to you (weight: {cash_weight:.1f}). "
            "Avoid spending all your money — keep a buffer for future rounds. "
            "Only trade when the deal is clearly worthwhile."
        )
    elif f.patience > 0.4:
        cash_desc = (
            f"Cash has moderate value to you (weight: {cash_weight:.1f}). "
            "Balance spending on goods with keeping some cash on hand."
        )
    else:
        cash_desc = (
            f"Cash has low value to you (weight: {cash_weight:.1f}). "
            "Prioritize converting cash into goods you value."
        )

    # Diversity bonus
    if f.patience > 0.3:
        diversity_desc = (
            f"You benefit from holding diverse goods "
            f"(bonus: {f.patience * 0.1:.2f} per good type held). "
            "Try to hold at least some of each type."
        )
    else:
        diversity_desc = "Focus on accumulating the goods you value most."

    return (
        f"\nSTRATEGY GUIDE (use these principles to evaluate trades):\n"
        f"- DIMINISHING RETURNS: {dr_desc}\n"
        f"- CASH MANAGEMENT: {cash_desc}\n"
        f"- DIVERSITY: {diversity_desc}\n"
        f"- TRADE EVALUATION: A trade is good when the benefit of gaining a good "
        f"(considering your preference weight and how much you already have) "
        f"exceeds the cost of spending cash or giving up inventory.\n"
    )


# ---------------------------------------------------------------------------
# Convert to AgentPersona (used by existing LLM simulation)
# ---------------------------------------------------------------------------

def features_to_persona(
    f: DGPFeatures,
    style: str = "numeric",
    with_strategy: bool = False,
    neutral_label: str | None = None,
    drop_feature: str | None = None,
) -> AgentPersona:
    """Convert DGPFeatures into an AgentPersona for use with the LLM agent.

    Args:
        f: The DGP feature set
        style: "numeric" or "narrative"
        with_strategy: if True, include CRRA strategy guide in the system prompt
        neutral_label: if set, use this as the display name instead of the archetype-based name
        drop_feature: feature to omit from the prompt ("risk", "patience", "preferences", "production_skill")
    """
    display_name = neutral_label or f.name

    if style == "numeric":
        personality = features_to_numeric_prompt(f, display_name=display_name, drop_feature=drop_feature)
    elif style == "narrative":
        personality = features_to_narrative_prompt(f, display_name=display_name, drop_feature=drop_feature)
    else:
        raise ValueError(f"Unknown style: {style}. Use 'numeric' or 'narrative'.")

    strategy = build_strategy_guide(f) if with_strategy else ""

    # Derive preferred_goods from top 2 preferences
    sorted_prefs = sorted(f.preferences.items(), key=lambda x: -x[1])
    preferred = [g for g, _ in sorted_prefs[:2]]

    return AgentPersona(
        name=display_name,
        personality=personality,
        preferred_goods=preferred,
        production_skill=f.production_skill,
        risk_tolerance=f.risk_appetite,
        strategy_guide=strategy,
    )
