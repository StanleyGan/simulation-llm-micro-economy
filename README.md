# LLM Micro-Economy Simulation

Validating whether LLM persona simulations produce economically faithful outcomes compared to a structural data-generating process (DGP) with known causal structure.

The DGP uses CRRA utility maximization; the LLM uses `gpt-4o-mini` with persona prompts derived from the same features. Both share the same market matching engine — decision-making is the only variable.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
make install          # or: uv sync --group dev
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## Project Structure

```
src/micro_economy/
├── models.py              # Shared data models (Good, Agent, MarketOrder, etc.)
├── simulation.py          # Market engine: order matching and trade execution
├── dgp.py                 # CRRA utility maximization (ground truth decisions)
├── dgp_personas.py        # 6 archetype definitions with feature distributions
├── llm_agent.py           # LLM decision-making via OpenAI API
├── prompt_translator.py   # DGPFeatures → AgentPersona (numeric/narrative styles)
├── experiments.py         # Preset persona configurations
└── metrics.py             # Economic metrics (Gini, volatility, efficiency)

tests/
├── test_dgp.py            # CRRA utility math and decision tests
├── test_dgp_personas.py   # Archetype sampling tests
├── test_prompt_translator.py  # Prompt generation and ablation tests
└── test_simulation.py     # Market matching engine tests

run_dgp.py                 # DGP experiment runner
run_llm_benchmark.py       # LLM experiment runner with ablation flags
compare_benchmark.py       # Spearman ρ comparison and chart generation
server.py                  # FastAPI web app (optional)
```

## Development

```bash
make test             # Run tests
make lint             # Check linting and formatting
make format           # Auto-fix lint issues and format
make ci               # Run lint + tests (same as CI)
```

## Running Experiments

### DGP Ground Truth

```bash
uv run python run_dgp.py --runs 10 --seed 42 --output dgp_results_v3
```

### LLM Benchmark

```bash
# Numeric prompt style
uv run python run_llm_benchmark.py --style numeric --runs 5 --seed 42 --output llm_numeric_v3

# Narrative prompt style
uv run python run_llm_benchmark.py --style narrative --runs 5 --seed 42 --output llm_narrative_v3

# With CRRA strategy guide
uv run python run_llm_benchmark.py --style numeric --with-strategy --runs 5 --seed 42 --output llm_numeric_strategy_v3
```

### Compare Results

```bash
uv run python compare_benchmark.py \
  --dgp dgp_results_v3/dgp_ground_truth.json \
  --llm llm_numeric_v3/llm_numeric_results.json \
  --output comparison_numeric_v3
```

## Ablation Studies

### A2: Label Ablation

Replace archetype labels (e.g., "cautious_farmer") with neutral names ("Agent_A") to test whether textual anchoring bias comes from labels vs. feature descriptions.

```bash
uv run python run_llm_benchmark.py --style numeric --neutral-labels --runs 5 --seed 42 --output llm_neutral_numeric_v3
uv run python run_llm_benchmark.py --style narrative --neutral-labels --runs 5 --seed 42 --output llm_neutral_narrative_v3
```

### A5: Feature Dropout

Remove one feature at a time from prompts to measure each feature's contribution to decision quality.

```bash
uv run python run_llm_benchmark.py --style numeric --drop-feature risk --runs 5 --seed 42 --output llm_drop_risk_v3
uv run python run_llm_benchmark.py --style numeric --drop-feature patience --runs 5 --seed 42 --output llm_drop_patience_v3
uv run python run_llm_benchmark.py --style numeric --drop-feature preferences --runs 5 --seed 42 --output llm_drop_preferences_v3
uv run python run_llm_benchmark.py --style numeric --drop-feature production_skill --runs 5 --seed 42 --output llm_drop_production_v3
```

## Archetypes

| Archetype | Risk | Patience | Production | Key Preference |
|-----------|------|----------|------------|----------------|
| cautious_farmer | 0.05–0.25 | 0.6–0.9 | food | food |
| aggressive_merchant | 0.7–0.95 | 0.1–0.35 | tools | luxury |
| pragmatic_doctor | 0.35–0.55 | 0.4–0.65 | medicine | medicine |
| shrewd_speculator | 0.8–0.98 | 0.05–0.25 | luxury | luxury |
| fair_toolmaker | 0.15–0.35 | 0.5–0.75 | tools | tools |
| survivalist | 0.02–0.15 | 0.7–0.95 | food | food |

## Current Results (v3)

60 agents per run, 10 rounds. DGP: 10 runs (seeds 42–51). LLM: 5 runs (seeds 42–46).

| Condition | Mean Spearman ρ ± std | Archetype ρ |
|-----------|----------------------|-------------|
| Numeric | 0.712 ± 0.031 | 0.829 |
| Narrative | 0.725 ± 0.010 | 0.829 |
| Numeric + strategy | 0.736 ± 0.036 | 0.829 |
| Narrative + strategy | 0.742 ± 0.026 | 0.829 |

### Key Findings

1. **Strong ordinal faithfulness**: The LLM reliably preserves wealth rankings across 60 agents (ρ = 0.71–0.74 per run, archetype ρ = 0.83), correctly identifying top and bottom performers. However, it captures rankings, not magnitudes — wealth is systematically deflated by 21–61%.

2. **Textual anchoring bias overrides optimization rules**: "Aggressive merchant" is consistently overvalued (GT rank 41.6 → LLM rank ~27), while "cautious farmer" and "survivalist" are penalized (GT rank 38–46 → LLM rank 49–51). The LLM maps persona labels to trading outcomes through its own prior associations rather than the provided optimization logic.

3. **Strategy guide helps within-archetype, not between**: Adding explicit CRRA optimization rules to the system prompt improves per-agent correlation by +0.02 but leaves archetype-level biases and ranking (ρ = 0.829) unchanged. The LLM uses the rules for finer-grained differentiation among similar agents, not for correcting its core archetype-level behavior.

4. **Prompt style matters less than expected**: Numeric and narrative produce nearly identical results. Narrative has slightly higher mean ρ (0.725 vs 0.712) and lower variance (±0.010 vs ±0.031).

5. **Compressed volatility**: LLM trajectory volatility falls into a narrower band (CV 0.073–0.099) than DGP (0.061–0.109), indicating the LLM doesn't fully differentiate risk-taking behavior by archetype.

For the full report including per-run breakdowns, archetype-level comparisons, trajectory analysis, and strategy guide ablation results, see [benchmark_report_v3.md](benchmark_report_v3.md).
