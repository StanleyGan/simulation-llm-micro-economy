# LLM micro-economy simulation: benchmark report

## 1. Overview

This report evaluates whether LLM persona simulations produce economically faithful outcomes compared to a structural data-generating process (DGP) with known causal structure. The DGP uses CRRA utility maximization with brute-force trade enumeration. The LLM simulation uses gpt-4o-mini with persona prompts derived from the same DGP features. Both share the same market matching engine — decision-making is the only variable being tested.

The motivation draws from causal machine learning: when we have a DGP with known causal structure, we can use it to test how faithfully a model recovers ground-truth outcomes. Applied to LLM simulations, this framework lets us measure what an LLM actually captures about economic behavior, what it distorts, and what prompt-level interventions can (and cannot) fix.

## 2. Experiment design

### 2.1 Agents

Six archetypes, 10 agents per archetype, 60 agents trading together per run:

| Archetype | Risk appetite | Patience | Production | Key preference |
|---|---|---|---|---|
| cautious_farmer | 0.05–0.25 | 0.6–0.9 | food | food |
| aggressive_merchant | 0.7–0.95 | 0.1–0.35 | tools | luxury |
| pragmatic_doctor | 0.35–0.55 | 0.4–0.65 | medicine | medicine |
| shrewd_speculator | 0.8–0.98 | 0.05–0.25 | luxury | luxury |
| fair_toolmaker | 0.15–0.35 | 0.5–0.75 | tools | tools |
| survivalist | 0.02–0.15 | 0.7–0.95 | food | food |

Each agent's behavior is fully defined by five numeric features: risk appetite, patience, budget, good preferences (over food/tools/luxury/medicine), and production skill. Features are sampled from archetype-specific distributions defined in `dgp_personas.py`.

### 2.2 DGP utility function

The DGP uses constant relative risk aversion (CRRA) utility:

- Per-good utility: U(qty) = qty^(1-r) / (1-r), where r = risk_appetite
- Total utility: Σ pref_j × U(qty_j) + cash_utility + diversity_bonus
- Cash utility: (0.3 + patience × 0.4) × CRRA(budget/10, risk)
- Diversity bonus: patience × 0.1 per good type held (if patience > 0.3)
- Trade search: brute-force single-good + pairwise enumeration, max 5 units per good per trade

Higher risk appetite produces sharper diminishing returns, favoring diversification. Higher patience increases cash-holding value and diversity bonus. The DGP makes the mathematically optimal trade each round given its utility function.

### 2.3 Prompt styles

Two translation methods convert DGP features into LLM persona prompts:

- **Numeric**: explicit ranges and percentages ("risk appetite: 7–9 out of 10, patience: 1–3 out of 10, preference weights: food 50%, medicine 34%...")
- **Narrative**: natural language descriptions ("You are extremely cautious. You hate taking risks... You value food the most, followed closely by medicine.")

Both include starting budget, production skill, and trading constraints matching the DGP action space.

### 2.4 Experimental conditions

| Condition | Description | Runs |
|---|---|---|
| Numeric baseline | Numeric prompt, no extras | 5 |
| Narrative baseline | Narrative prompt, no extras | 5 |
| Numeric + strategy | Numeric + CRRA strategy guide | 5 |
| Narrative + strategy | Narrative + CRRA strategy guide | 5 |
| Neutral labels (A2) | Numeric prompt, archetype names replaced with Agent_A–F | 5 |
| Drop risk (A5) | Numeric prompt, risk appetite omitted | 5 |
| Drop patience (A5) | Numeric prompt, patience omitted | 5 |
| Drop preferences (A5) | Numeric prompt, good preferences omitted | 5 |
| Drop production (A5) | Numeric prompt, production skill omitted | 5 |

DGP: 10 runs × 10 rounds × 60 agents (seeds 42–51). All LLM conditions: 5 runs × 10 rounds × 60 agents (seeds 42–46). Comparisons use the 5 overlapping seeds (300 matched agents per condition).

---

## 3. Baseline results

### 3.1 Per-run Spearman rank correlation

| Run (seed) | Numeric ρ | Narrative ρ |
|---|---|---|
| 0 (42) | 0.723 | 0.737 |
| 1 (43) | 0.661 | 0.712 |
| 2 (44) | 0.758 | 0.735 |
| 3 (45) | 0.717 | 0.717 |
| 4 (46) | 0.701 | 0.722 |
| **Mean ± std** | **0.712 ± 0.031** | **0.725 ± 0.010** |

All runs exceed ρ = 0.65, with no negative or near-zero outliers. The LLM reliably preserves the broad wealth ranking across 60 agents. Narrative has slightly higher mean ρ and substantially lower variance.

### 3.2 Archetype-level comparison

| Archetype | GT rank | Numeric rank | Narrative rank | GT wealth | Numeric wealth | Narrative wealth |
|---|---|---|---|---|---|---|
| shrewd_speculator | 8.0 | 5.8 | 5.9 | $854 | $634 | $634 |
| pragmatic_doctor | 14.1 | 18.5 | 18.2 | $726 | $410 | $411 |
| fair_toolmaker | 34.9 | 31.3 | 31.4 | $533 | $326 | $322 |
| cautious_farmer | 38.4 | 49.3 | 49.7 | $507 | $204 | $198 |
| aggressive_merchant | 41.6 | 27.0 | 26.6 | $487 | $353 | $350 |
| survivalist | 46.0 | 51.1 | 51.2 | $461 | $191 | $188 |

Archetype-level Spearman ρ = 0.829 for both styles. The LLM correctly identifies shrewd_speculator and pragmatic_doctor as top performers, and survivalist as bottom.

### 3.3 Systematic biases

**Aggressive merchant overvaluation**: GT rank 41.6 (below median) → LLM rank 27.0 (above median). This is the largest rank discrepancy. The LLM interprets "aggressive" as a positive trading trait, while in the CRRA model, high risk appetite with low patience leads to suboptimal trades due to sharp diminishing returns and undervalued cash/diversity.

**Cautious/survivalist penalty**: These archetypes drop from mid-pack (GT rank 38–46) to near-bottom (LLM rank 49–51). The LLM interprets "cautious" and "survivalist" as reasons to avoid trading, which in a market economy means missing profitable opportunities. In the DGP, cautious agents trade selectively but effectively — their low risk appetite means mild diminishing returns, so concentrating on preferred goods is actually near-optimal.

**Wealth deflation**: LLM agents produce 24–61% less wealth than GT across all archetypes. The LLM trades less efficiently than the utility-maximizing DGP.

---

## 4. Ablation A1: strategy guide

### 4.1 Design

The CRRA utility function's optimization logic is described in natural language and appended to the system prompt. Four agent-specific rules:

1. **Diminishing returns**: intensity varies by risk appetite (sharp/moderate/mild)
2. **Cash management**: cash weight = 0.3 + patience × 0.4, with spending guidance
3. **Diversity**: bonus of patience × 0.1 per good type held (if patience > 0.3)
4. **Trade evaluation**: compare marginal utility gained vs lost before trading

### 4.2 Results

| Condition | Mean ρ ± std | Archetype ρ |
|---|---|---|
| Numeric baseline | 0.712 ± 0.031 | 0.829 |
| Narrative baseline | 0.725 ± 0.010 | 0.829 |
| Numeric + strategy | 0.736 ± 0.036 | 0.829 |
| Narrative + strategy | 0.742 ± 0.026 | 0.829 |

### 4.3 Analysis

The strategy guide produces a modest improvement in per-agent rank correlation (+0.024 numeric, +0.017 narrative). The improvement is within-archetype — agents with different feature values are ranked more accurately relative to each other. Archetype-level ρ remains 0.829.

The core biases are unchanged: aggressive_merchant still overvalued (rank ~29), cautious/survivalist still penalized (rank ~49–51), wealth deflation similar (21–57%). Personality labels dominate archetype-level behavior. The textual anchoring effect is stronger than explicit strategic instructions.

---

## 5. Ablation A2: label ablation

### 5.1 Design

Replace archetype-derived agent names (e.g., "cautious_farmer_3") with neutral identifiers ("Agent_A_3"). The mapping is: cautious_farmer → A, aggressive_merchant → B, pragmatic_doctor → C, shrewd_speculator → D, fair_toolmaker → E, survivalist → F. All other prompt content (numeric features, trait descriptions) is unchanged. Numeric style only, no strategy guide.

This tests whether the textual anchoring bias originates from the archetype labels themselves or from the feature descriptions that accompany them.

### 5.2 Results

| Condition | Mean ρ ± std | Archetype ρ |
|---|---|---|
| Numeric baseline | 0.712 ± 0.031 | 0.829 |
| **Neutral labels** | **0.866 ± 0.011** | **0.829** |

Per-run detail:

| Run (seed) | Baseline ρ | Neutral labels ρ |
|---|---|---|
| 0 (42) | 0.723 | 0.861 |
| 1 (43) | 0.661 | 0.874 |
| 2 (44) | 0.758 | 0.882 |
| 3 (45) | 0.717 | 0.861 |
| 4 (46) | 0.701 | 0.852 |

Archetype-level comparison:

| Archetype | GT rank | Baseline rank | Neutral rank | GT wealth | Baseline wealth | Neutral wealth |
|---|---|---|---|---|---|---|
| shrewd_speculator | 8.0 | 5.8 | 5.7 | $854 | $634 | $641 |
| pragmatic_doctor | 14.1 | 18.5 | 16.9 | $726 | $410 | $426 |
| fair_toolmaker | 34.9 | 31.3 | 30.5 | $533 | $326 | $325 |
| cautious_farmer | 38.4 | 49.3 | 49.5 | $507 | $204 | $204 |
| aggressive_merchant | 41.6 | 27.0 | 29.2 | $487 | $353 | $336 |
| survivalist | 46.0 | 51.1 | 51.3 | $461 | $191 | $192 |

### 5.3 Analysis

Neutral labels produced the single largest improvement in per-agent rank correlation across all experiments: +0.154 over baseline (0.712 → 0.866), with the lowest variance (± 0.011). This is 6.4× larger than the strategy guide improvement (+0.024).

However, two critical observations:

1. **Archetype-level ρ remains 0.829.** The between-archetype ranking is unchanged. Aggressive_merchant is still overvalued (rank 29.2 vs GT 41.6), cautious_farmer/survivalist still penalized (rank ~50 vs GT ~38–46).

2. **Wealth recovery ratios are nearly identical to baseline.** Cautious_farmer recovers 40.5% of DGP wealth in both conditions. Shrewd_speculator recovers 76.1% baseline vs 77.0% neutral.

This reveals that label removal improved **within-archetype agent differentiation** — the LLM became much better at ranking agents within the same archetype relative to each other — but did not fix the systematic bias in how it values different archetype profiles. The archetype-level bias is driven by the feature distributions themselves (high risk + low patience profile), not solely by the label "aggressive."

---

## 6. Ablation A5: feature dropout

### 6.1 Design

Remove one feature at a time from the numeric prompt. Four conditions: drop risk appetite, drop patience, drop preferences, drop production skill. Budget is always included. Numeric style, no strategy guide, no neutral labels.

### 6.2 Results

| Condition | Mean ρ ± std | Δ vs baseline |
|---|---|---|
| Numeric baseline | 0.712 ± 0.031 | — |
| Drop risk | 0.707 ± 0.025 | -0.005 |
| Drop patience | 0.736 ± 0.035 | +0.024 |
| Drop preferences | 0.695 ± 0.025 | -0.017 |
| Drop production | 0.697 ± 0.055 | -0.015 |

Archetype-level ρ = 0.829 across all dropout conditions.

### 6.3 Per-archetype wealth recovery (LLM wealth / DGP wealth)

| Archetype | Baseline | -Risk | -Patience | -Preferences | -Production |
|---|---|---|---|---|---|
| shrewd_speculator | 76.1% | 79.9% | 77.1% | 81.8% | 80.8% |
| aggressive_merchant | 71.6% | 76.7% | 70.1% | 73.6% | 72.7% |
| pragmatic_doctor | 57.2% | 59.3% | 64.9% | 59.4% | 58.6% |
| fair_toolmaker | 62.4% | 64.9% | 63.2% | 63.2% | 65.4% |
| cautious_farmer | 40.5% | 41.7% | 41.9% | 39.1% | 40.1% |
| survivalist | 41.0% | 42.3% | 44.0% | 40.6% | 42.4% |

### 6.4 Analysis

**Risk appetite is noise, not signal.** Dropping risk from the prompt has virtually no effect on rank correlation (-0.005). This is surprising because risk appetite is the defining CRRA parameter. The LLM doesn't use explicit risk values to calibrate trade aggressiveness — it relies on archetype labels and other correlated features instead.

**Patience introduces distortion.** Dropping patience *improves* correlation (+0.024), matching the strategy guide's improvement. The biggest beneficiary is pragmatic_doctor, whose wealth recovery jumps from 57.2% to 64.9%. The doctor has moderate patience (0.4–0.65), which the LLM apparently misinterprets as "wait more, trade less." Without that signal, the LLM trades more actively for this archetype, which happens to be closer to the DGP's optimum.

**Preferences are the most informative feature.** Dropping good preferences produces the largest negative impact (-0.017). Preferences are the feature the LLM uses most effectively for within-agent differentiation — they provide concrete guidance on *what* to buy and sell.

**Production skill provides stability.** Dropping production skill has a moderate negative impact (-0.015) but doubles variance (± 0.055 vs ± 0.031). Without knowing what they produce, the LLM's decisions become more erratic across runs.

---

## 7. Cross-condition analysis

### 7.1 Feature-wealth correlations

A key question: does the LLM recover the correct causal relationships between features and wealth? We compute Pearson correlation between each DGP feature and final wealth.

| Feature | DGP |r| | LLM |r| (range across conditions) |
|---|---|---|
| risk_appetite | 0.50 | 0.73–0.80 |
| patience | 0.49 | 0.71–0.77 |
| budget_initial | 0.60 | 0.71–0.77 |

**The LLM over-amplifies every feature-wealth relationship.** In the DGP, these correlations are moderate (0.49–0.60) because the CRRA utility function is nonlinear and features interact — high risk + low patience doesn't simply add up. The LLM produces correlations of 0.71–0.80 across all conditions, suggesting it applies a near-linear mapping from features to behavior.

This has an important implication: the LLM may appear to "understand" the features (strong correlations), but it does so through an oversimplified heuristic. It captures the direction of each effect (more risk → different trading) but misses the nonlinear interactions that define the DGP (diminishing returns on risk, patience-cash interactions, diversity bonuses).

### 7.2 Within-archetype coefficient of variation

| Archetype | DGP CV | LLM baseline CV | Neutral labels CV |
|---|---|---|---|
| cautious_farmer | 0.131 | 0.175 | 0.175 |
| aggressive_merchant | 0.120 | 0.111 | 0.136 |
| pragmatic_doctor | 0.108 | 0.119 | 0.102 |
| shrewd_speculator | 0.137 | 0.129 | 0.121 |
| fair_toolmaker | 0.129 | 0.146 | 0.117 |
| survivalist | 0.135 | 0.163 | 0.155 |

The LLM generally inflates within-archetype wealth variance for low-performing archetypes (cautious_farmer: 0.131 → 0.175) and compresses it for high-performing ones (shrewd_speculator: 0.137 → 0.129). Neutral labels bring some archetypes closer to DGP variance (pragmatic_doctor, fair_toolmaker) while leaving others inflated.

### 7.3 Wealth growth patterns

A revealing asymmetry in the DGP:

| Archetype | DGP growth | LLM baseline growth |
|---|---|---|
| cautious_farmer | **+33.5%** | -13.7% |
| survivalist | **+40.7%** | -15.3% |
| aggressive_merchant | **-23.6%** | -12.5% |
| shrewd_speculator | **-12.5%** | -9.2% |
| pragmatic_doctor | +15.4% | -9.4% |
| fair_toolmaker | -1.5% | -17.5% |

In the DGP, patient agents (cautious_farmer, survivalist) *gain* wealth over rounds by waiting for good prices and trading selectively. Aggressive agents *lose* wealth because CRRA diminishing returns penalize concentrated trading. The LLM inverts this pattern: all archetypes show negative growth, but aggressive archetypes lose *less*. The LLM has an **activity bias** — it equates trading frequency with success, rather than modeling the quality of each trade.

### 7.4 The archetype ρ = 0.829 ceiling

Archetype-level Spearman ρ is exactly 0.829 across all 9 conditions tested. No prompt-level intervention changes the between-archetype ranking. The LLM consistently produces:

> speculator > doctor > toolmaker ≈ merchant > farmer > survivalist

The DGP ordering is:

> speculator > doctor > farmer ≈ toolmaker > merchant > survivalist

The LLM swaps aggressive_merchant and cautious_farmer. In the DGP, the merchant's high risk appetite and low patience produce sharp diminishing returns and undervalued cash reserves — an objectively disadvantageous profile. But the LLM maps "high budget + active trading" to success, which happens to be the merchant's profile. This swap is robust across all prompt styles, strategy guides, label changes, and feature dropouts.

---

## 8. Summary of findings

### 8.1 What LLM simulations get right

1. **Ordinal faithfulness at scale.** Per-agent rank correlation of 0.71–0.87 across 60 agents means the LLM captures the broad wealth hierarchy. Researchers can trust relative comparisons ("archetype X outperforms Y") for most archetype pairs.

2. **Feature responsiveness.** The LLM responds to all provided features in the expected direction. Agents with higher budgets, more aggressive risk profiles, and stronger good preferences produce differentiated outcomes.

3. **Within-archetype differentiation.** When labels are removed, per-agent ρ reaches 0.866, suggesting the LLM can make fine-grained distinctions between agents with similar profiles based on their numeric features alone.

### 8.2 What LLM simulations get wrong

1. **Textual anchoring bias.** Archetype labels create a fixed mapping between persona names and economic outcomes that overrides numeric features and explicit optimization rules. This is the dominant source of error.

2. **Linear feature heuristics.** The LLM over-amplifies feature-wealth correlations (0.73–0.80 vs DGP's 0.49–0.60), applying a near-linear mapping where the DGP has nonlinear interactions. It doesn't capture diminishing returns, feature interactions, or the nonlinear relationship between risk aversion and optimal trading strategy.

3. **Activity bias.** The LLM equates trading with success. Patient agents who should *gain* from selective trading instead lose wealth. Aggressive agents who should be penalized by diminishing returns are instead rewarded for high activity.

4. **Cardinal infidelity.** Wealth magnitudes are systematically deflated (39–82% of DGP wealth depending on archetype). The LLM does not find optimal trades — it follows a simplified heuristic.

5. **Compressed behavioral dynamics.** Trajectory volatility and wealth growth patterns are compressed into a narrow band. The LLM doesn't fully differentiate risk-taking behavior over time.

### 8.3 Practical recommendations for LLM simulation researchers

1. **Use neutral labels.** Replacing descriptive archetype names with neutral identifiers (Agent_A, Agent_B) is the single most effective intervention, improving per-agent correlation by +0.154 at zero cost.

2. **Trust rankings, not magnitudes.** LLM simulations are ordinally faithful but cardinally unreliable. Report results as relative rankings or comparative statements, not absolute wealth values.

3. **Be cautious with personality-laden prompts.** Adjectives like "aggressive," "cautious," and "shrewd" carry strong LLM priors that may override the numeric parameters intended to drive behavior. Prefer factual descriptions over personality framing.

4. **Feature dropout as prompt debugging.** Systematically removing features can reveal which parameters the LLM actually uses vs. ignores. In our case, risk appetite had near-zero marginal contribution, while removing patience actually improved fidelity.

5. **Validate against structural models.** The DGP comparison framework used here can be applied to any LLM simulation: define the ground-truth decision function, generate both DGP and LLM outcomes, and measure correlation at multiple levels of aggregation.

---

## 9. Condition comparison summary

| Condition | Mean ρ ± std | Archetype ρ | Key observation |
|---|---|---|---|
| Numeric baseline | 0.712 ± 0.031 | 0.829 | Reference condition |
| Narrative baseline | 0.725 ± 0.010 | 0.829 | Lower variance than numeric |
| Numeric + strategy | 0.736 ± 0.036 | 0.829 | +0.024, within-archetype only |
| Narrative + strategy | 0.742 ± 0.026 | 0.829 | Best with labels, +0.030 |
| **Neutral labels (A2)** | **0.866 ± 0.011** | **0.829** | **Best overall, +0.154** |
| Drop risk (A5) | 0.707 ± 0.025 | 0.829 | Risk feature unused |
| Drop patience (A5) | 0.736 ± 0.035 | 0.829 | Patience was hurting |
| Drop preferences (A5) | 0.695 ± 0.025 | 0.829 | Most important feature |
| Drop production (A5) | 0.697 ± 0.055 | 0.829 | Highest variance |

---

## 10. Planned experiments

**A3. Contradictory signals**: Give agents personality text that conflicts with their numeric parameters (e.g., "aggressive" narrative + cautious numeric features). Directly quantifies the anchoring weight between text descriptions and numeric specifications. This is the core identification experiment: when the LLM's textual prior conflicts with the prompt's numeric specification, which dominates?

**A4. Temperature variation**: Run at temperature 0.0, 0.5, 1.0. Tests whether LLM stochasticity affects trajectory diversity, rank convergence, and the activity bias.

**Causal effect recovery**: Reframe results as treatment effect estimation. The DGP defines the true causal effect of each feature on wealth; measure how well the LLM recovers these effects, including heterogeneous treatment effects (does the LLM capture that risk appetite affects cautious agents differently than aggressive agents?).
