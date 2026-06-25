# LLM simulation benchmark report

## Overview

This report evaluates whether LLM persona simulations produce economically faithful outcomes compared to a structural data-generating process (DGP) with known causal structure. The DGP uses CRRA utility maximization with brute-force trade enumeration. The LLM simulation uses gpt-4o-mini with persona prompts derived from the same DGP features. Both share the same market matching engine — decision-making is the only variable being tested.

## Experiment design

### Agents

Six archetypes, 10 agents per archetype, 60 agents trading together per run. Archetypes: cautious farmer, aggressive merchant, pragmatic doctor, shrewd speculator, fair toolmaker, survivalist. Each agent's behavior is fully defined by numeric features: risk appetite, patience, budget, good preferences, and production skill.

### Prompt styles

Two translation methods convert DGP features into LLM prompts:

- **Numeric**: explicit ranges ("risk appetite: 7-9 out of 10, patience: 1-3 out of 10, preference weights: food 50%, medicine 34%...")
- **Narrative**: natural language ("You are extremely cautious. You hate taking risks... You value food the most, followed closely by medicine.")

Both include starting budget and trading constraints matching the DGP action space.

### Runs

DGP: 10 runs × 10 rounds × 60 agents (seeds 42–51). LLM: 5 runs × 10 rounds × 60 agents (seeds 42–46). Comparison uses the 5 overlapping seeds (300 matched agents).

---

## Results

### Per-run Spearman ρ

| Run (seed) | Numeric ρ | Narrative ρ |
|------------|-----------|-------------|
| 0 (42) | 0.723 | 0.737 |
| 1 (43) | 0.661 | 0.712 |
| 2 (44) | 0.758 | 0.735 |
| 3 (45) | 0.717 | 0.717 |
| 4 (46) | 0.701 | 0.722 |
| **Mean ± std** | **0.712 ± 0.031** | **0.725 ± 0.010** |

All runs exceed ρ = 0.65, with no negative or near-zero outliers. The LLM reliably preserves the broad wealth ranking across 60 agents.

### Archetype-level comparison

| Archetype | GT rank | Numeric rank | Narrative rank | GT wealth | Numeric wealth | Narrative wealth |
|-----------|---------|--------------|----------------|-----------|----------------|------------------|
| shrewd speculator | 8.0 | 5.8 | 5.9 | $854 | $634 | $634 |
| pragmatic doctor | 14.1 | 18.5 | 18.2 | $726 | $410 | $411 |
| fair toolmaker | 34.9 | 31.3 | 31.4 | $533 | $326 | $322 |
| cautious farmer | 38.4 | 49.3 | 49.7 | $507 | $204 | $198 |
| aggressive merchant | 41.6 | 27.0 | 26.6 | $487 | $353 | $350 |
| survivalist | 46.0 | 51.1 | 51.2 | $461 | $191 | $188 |

Archetype-level Spearman ρ = 0.829 for both styles. The LLM correctly identifies shrewd speculator and pragmatic doctor as top performers, and survivalist as bottom.

### Systematic biases

**Aggressive merchant overvaluation**: GT rank 41.6 (below median) → LLM rank 27.0 (above median). This is the largest rank discrepancy. The LLM interprets "aggressive" as a positive trading trait, while in the CRRA model, high risk appetite with moderate patience actually leads to suboptimal trades.

**Wealth deflation**: LLM agents produce 26–61% less wealth than GT across all archetypes. The LLM trades less efficiently than the utility-maximizing DGP — it doesn't find the mathematically optimal trade each round.

**Cautious/survivalist penalty**: These archetypes drop from mid-pack (GT rank 38–46) to near-bottom (LLM rank 49–51). The LLM interprets "cautious" and "survivalist" as reasons to avoid trading, which in a market economy means missing profitable opportunities.

---

## Trajectory analysis

### Volatility comparison (coefficient of variation)

| Archetype | GT CV | Numeric CV | Narrative CV |
|-----------|-------|------------|--------------|
| aggressive merchant | 0.0958 | 0.0848 | 0.0883 |
| cautious farmer | 0.0961 | 0.0907 | 0.0897 |
| fair toolmaker | 0.0613 | 0.0942 | 0.0994 |
| pragmatic doctor | 0.0670 | 0.0787 | 0.0733 |
| shrewd speculator | 0.0760 | 0.0909 | 0.0950 |
| survivalist | 0.1089 | 0.0843 | 0.0825 |

GT CV range is 0.061–0.109 vs LLM range 0.073–0.099. The LLM compresses volatility into a narrower band — fair toolmaker and shrewd speculator show higher LLM volatility than GT, while survivalist shows lower. The LLM doesn't fully differentiate risk-taking behavior by archetype.

---

## Key findings

1. **The LLM produces strongly correlated wealth rankings** (ρ = 0.71–0.74 per run, archetype ρ = 0.83). With 60 agents and proper statistical power, the LLM reliably captures which archetypes perform well and which don't.

2. **Textual anchoring bias dominates optimization rules**: "aggressive merchant" is consistently overvalued and "cautious farmer"/"survivalist" are undervalued. Adding explicit CRRA strategy rules to the system prompt does not fix these biases — personality labels override strategic instructions. The LLM maps persona descriptions to trading outcomes through its own prior associations, not through the optimization logic provided.

3. **Strategy guide helps within-archetype, not between**: Adding CRRA rules improves per-agent ρ by +0.017 to +0.024, but archetype-level ρ (0.829) and systematic biases are unchanged. The LLM uses the rules for finer-grained differentiation among agents with similar personas, not for correcting its core archetype-level behavior.

4. **Prompt style matters less than expected**: Numeric and narrative produce nearly identical results (ρ = 0.712 vs 0.725, archetype ρ = 0.829 for both). Narrative has lower variance (± 0.010 vs ± 0.031).

5. **LLM decisions are less strategic than GT**: wealth is systematically deflated (21–61%), trajectories converge too quickly, and volatility patterns don't fully differentiate by archetype. The LLM doesn't find optimal trades — it follows a simplified heuristic shaped by persona descriptions.

6. **Ordinal faithfulness, not cardinal**: The LLM gets the ranking mostly right but not the magnitudes. This is consistent with LLMs being better at relative reasoning ("who should do better?") than quantitative optimization.

---

## Ablation A1: strategy guide

### Design

The CRRA utility function's optimization logic is described in natural language and appended to the system prompt as a "STRATEGY GUIDE" section. The persona prompt (numeric or narrative) is unchanged — the strategy guide is additive. Four agent-specific rules are included:

1. **Diminishing returns**: intensity varies by risk appetite (sharp/moderate/mild)
2. **Cash management**: cash weight = 0.3 + patience × 0.4, with spending guidance
3. **Diversity**: bonus of patience × 0.1 per good type held (if patience > 0.3)
4. **Trade evaluation**: compare marginal utility gained vs lost before trading

This tests whether teaching the LLM *how the DGP thinks about trades* improves faithfulness, while keeping the personality framing intact.

### Results

| Condition | Mean ρ ± std | Archetype ρ |
|-----------|-------------|-------------|
| Numeric (baseline) | 0.712 ± 0.031 | 0.829 |
| Narrative (baseline) | 0.725 ± 0.010 | 0.829 |
| Numeric + strategy | 0.736 ± 0.036 | 0.829 |
| Narrative + strategy | 0.742 ± 0.026 | 0.829 |

Per-run detail (strategy conditions):

| Run (seed) | Numeric+strategy ρ | Narrative+strategy ρ |
|------------|-------------------|---------------------|
| 0 (42) | 0.685 | 0.762 |
| 1 (43) | 0.773 | 0.744 |
| 2 (44) | 0.780 | 0.775 |
| 3 (45) | 0.723 | 0.699 |
| 4 (46) | 0.717 | 0.731 |

### Archetype comparison (strategy conditions)

| Archetype | GT rank | Num+strat rank | Narr+strat rank | GT wealth | Num+strat wealth | Narr+strat wealth |
|-----------|---------|----------------|-----------------|-----------|------------------|-------------------|
| shrewd speculator | 8.0 | 5.8 | 5.9 | $854 | $674 | $643 |
| pragmatic doctor | 14.1 | 16.5 | 16.8 | $726 | $461 | $461 |
| fair toolmaker | 34.9 | 31.3 | 30.3 | $533 | $330 | $330 |
| cautious farmer | 38.4 | 49.1 | 49.0 | $507 | $221 | $216 |
| aggressive merchant | 41.6 | 29.2 | 29.5 | $487 | $345 | $334 |
| survivalist | 46.0 | 51.1 | 51.5 | $461 | $211 | $203 |

### Analysis

The strategy guide produces a modest improvement in per-agent rank correlation: numeric +0.024, narrative +0.017. The improvement is within-archetype — agents with different feature values are ranked more accurately relative to each other. Archetype-level ρ stays at 0.829.

However, the core biases are unchanged:

- **Aggressive merchant** still overvalued (GT rank 41.6 → ~29). The LLM reads the personality description ("aggressive", "risk-seeking") and treats it as a positive trading signal, overriding the strategy guide's diminishing-returns logic.
- **Cautious/survivalist** still penalized (~49-51). The LLM interprets "cautious" as a reason to avoid trading even when the strategy guide says cash has low value.
- **Wealth deflation** similar (21–57%). The strategy guide doesn't close the efficiency gap with the DGP optimizer.

The conclusion: the LLM can incorporate optimization rules to make marginally better within-archetype differentiations, but personality labels dominate archetype-level behavior. The textual anchoring effect is stronger than explicit strategic instructions.

---

## Suggested further ablation studies

**A2. Label ablation**: Replace archetype labels with neutral names (Agent_A through Agent_F). Tests whether the textual anchoring bias disappears when "aggressive" and "cautious" labels are removed.

**A3. Feature dropout**: Systematically remove one feature at a time (risk appetite, patience, preferences, budget) from the prompt. Measures which features the LLM actually uses for decision-making vs which are ignored.

**A4. Contradictory signals**: Give an agent "aggressive" personality text but cautious numeric parameters (low risk, high patience). Tests whether the LLM anchors on text descriptions or numeric values when they conflict.

**A5. Temperature variation**: Run the same agents at temperature 0.0, 0.5, 1.0. Tests whether LLM stochasticity affects trajectory diversity and rank convergence.
