"""Run LLM benchmark simulation against DGP ground truth.

Uses the same seed, same population, same market engine as run_dgp.py,
but uses LLM decisions instead of DGP utility maximization.
Outputs results in the same format for comparison against ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from micro_economy.dgp import DGPFeatures
from micro_economy.dgp_personas import ALL_ARCHETYPES, sample_population
from micro_economy.llm_agent import get_agent_decision
from micro_economy.models import GOOD_LIST, Agent, Good, Inventory, MarketOrder, MarketState
from micro_economy.prompt_translator import features_to_persona
from micro_economy.simulation import execute_trades, match_orders

_NEUTRAL_GROUP_LABELS = ["A", "B", "C", "D", "E", "F"]


def create_llm_agents_from_features(
    features_list: list[DGPFeatures],
    style: str = "numeric",
    with_strategy: bool = False,
    neutral_labels: bool = False,
    drop_feature: str | None = None,
) -> list[Agent]:
    """Create LLM Agent objects from DGP features.

    Uses the same starting conditions as run_dgp.py's create_dgp_agents:
    - Budget at 15% of sampled value
    - Large production inventory, small amounts of others
    """
    archetype_counter: dict[str, int] = {}
    archetype_to_letter: dict[str, str] = {}

    agents = []
    for f in features_list:
        neutral_label = None
        if neutral_labels:
            if f.archetype not in archetype_to_letter:
                idx = len(archetype_to_letter)
                archetype_to_letter[f.archetype] = _NEUTRAL_GROUP_LABELS[idx]
            letter = archetype_to_letter[f.archetype]
            count = archetype_counter.get(f.archetype, 0)
            archetype_counter[f.archetype] = count + 1
            neutral_label = f"Agent_{letter}_{count}"

        persona = features_to_persona(
            f,
            style=style,
            with_strategy=with_strategy,
            neutral_label=neutral_label,
            drop_feature=drop_feature,
        )
        inv = Inventory()
        budget = f.budget * 0.15
        inv.add(f.production_skill, random.randint(8, 15))
        for g in GOOD_LIST:
            if g != f.production_skill:
                inv.add(g, random.randint(0, 2))
        agents.append(Agent(persona=persona, budget=budget, inventory=inv))
    return agents


def run_llm_simulation(
    agents: list[Agent],
    num_rounds: int = 20,
) -> tuple[list[Agent], MarketState]:
    """Run a single simulation with LLM agents (synchronous)."""
    market = MarketState()
    agents_map = {a.name: a for a in agents}

    # Record initial wealth
    wealth_tracker = {a.name: [a.net_worth(market.prices)] for a in agents}

    for round_num in range(1, num_rounds + 1):
        all_orders: list[MarketOrder] = []

        for agent in agents:
            orders = get_agent_decision(agent, market, round_num)
            all_orders.extend(orders)

        # Update demand
        for order in all_orders:
            if order.action == "buy":
                good = Good(order.good)
                market.demand[good] = market.demand.get(good, 0) + order.quantity

        # Match and execute using shared market engine
        buys = [o for o in all_orders if o.action == "buy"]
        sells = [o for o in all_orders if o.action == "sell"]
        trades = match_orders(buys, sells, market, round_num)
        execute_trades(trades, agents_map)
        market.trade_log.extend(trades)

        market.update_prices()
        market.record_prices()

        # Record wealth
        for agent in agents:
            wealth_tracker[agent.name].append(agent.net_worth(market.prices))

    return agents, market, wealth_tracker


def run_experiment(
    agents_per_archetype: int = 10,
    num_rounds: int = 10,
    num_runs: int = 10,
    style: str = "numeric",
    output_dir: str = "llm_results",
    base_seed: int = 42,
    with_strategy: bool = False,
    neutral_labels: bool = False,
    drop_feature: str | None = None,
) -> list[dict]:
    """Run the full LLM experiment matching DGP design.

    Each run: all agents (agents_per_archetype × 6 archetypes) trade together.
    Multiple runs use different seeds matching the DGP seeds.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    num_archetypes = len(ALL_ARCHETYPES)
    total_agents = agents_per_archetype * num_archetypes

    print(f"Prompt style: {style}")
    print(f"{agents_per_archetype} agents × {num_archetypes} archetypes = {total_agents} agents per run")
    print(f"Running {num_runs} runs × {num_rounds} rounds (seeds {base_seed}–{base_seed + num_runs - 1})\n")

    all_results = []

    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        random.seed(seed)

        # Same population as DGP (same seed)
        population = sample_population(ALL_ARCHETYPES, samples_per_archetype=agents_per_archetype)
        agents = create_llm_agents_from_features(
            population,
            style=style,
            with_strategy=with_strategy,
            neutral_labels=neutral_labels,
            drop_feature=drop_feature,
        )

        print(f"Run {run_idx + 1}/{num_runs} (seed {seed}): ", end="", flush=True)
        agents, market, wealth_tracker = run_llm_simulation(agents, num_rounds=num_rounds)

        # Collect results
        run_results = []
        for agent, feat in zip(agents, population):
            trajectory = wealth_tracker[agent.name]

            result = {
                "run_id": run_idx,
                "seed": seed,
                "name": agent.name,
                "archetype": feat.archetype,
                "risk_appetite": feat.risk_appetite,
                "budget_initial": feat.budget,
                "patience": feat.patience,
                "production_skill": feat.production_skill.value,
                "preferences": {g.value: round(v, 4) for g, v in feat.preferences.items()},
                "wealth_trajectory": [round(w, 2) for w in trajectory],
                "final_wealth": round(trajectory[-1], 2),
                "wealth_growth": round((trajectory[-1] - trajectory[0]) / max(trajectory[0], 1), 4),
                "final_budget": round(agent.budget, 2),
                "prompt_style": style,
            }
            run_results.append(result)

        # Rank within run (all 60 agents ranked together)
        run_results.sort(key=lambda r: r["final_wealth"], reverse=True)
        for rank, r in enumerate(run_results):
            r["rank_in_run"] = rank + 1

        all_results.extend(run_results)

        wealths = [r["final_wealth"] for r in run_results]
        print(f"done — wealth range: ${min(wealths):.0f}–${max(wealths):.0f}")

    # Save results
    _save_results(all_results, output_path, style)
    _print_summary(all_results, style)

    return all_results


def _save_results(results: list[dict], output_path: Path, style: str):
    """Save results to JSON and CSV."""
    prefix = f"llm_{style}"

    json_path = output_path / f"{prefix}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: {json_path}")

    csv_path = output_path / f"{prefix}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "name",
                "archetype",
                "risk_appetite",
                "budget_initial",
                "patience",
                "production_skill",
                "pref_food",
                "pref_tools",
                "pref_luxury",
                "pref_medicine",
                "final_wealth",
                "wealth_growth",
                "rank_in_run",
                "final_budget",
                "prompt_style",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["run_id"],
                    r["name"],
                    r["archetype"],
                    r["risk_appetite"],
                    r["budget_initial"],
                    r["patience"],
                    r["production_skill"],
                    r["preferences"]["food"],
                    r["preferences"]["tools"],
                    r["preferences"]["luxury"],
                    r["preferences"]["medicine"],
                    r["final_wealth"],
                    r["wealth_growth"],
                    r["rank_in_run"],
                    r["final_budget"],
                    r["prompt_style"],
                ]
            )
    print(f"Summary CSV: {csv_path}")

    traj_path = output_path / f"{prefix}_trajectories.csv"
    with open(traj_path, "w", newline="") as f:
        writer = csv.writer(f)
        max_rounds = max(len(r["wealth_trajectory"]) for r in results)
        header = ["run_id", "name", "archetype"] + [f"round_{t}" for t in range(max_rounds)]
        writer.writerow(header)
        for r in results:
            row = [r["run_id"], r["name"], r["archetype"]] + r["wealth_trajectory"]
            writer.writerow(row)
    print(f"Trajectories: {traj_path}")


def _print_summary(results: list[dict], style: str):
    """Print archetype-level summary."""
    from collections import defaultdict

    by_archetype = defaultdict(list)
    for r in results:
        by_archetype[r["archetype"]].append(r)

    print(f"\n{'=' * 70}")
    print(f"LLM Results (style: {style})")
    print(f"{'=' * 70}")
    print(f"{'Archetype':<22} {'Wealth (mean±std)':<22} {'Growth':<14} {'Rank (mean)'}")
    print(f"{'-' * 70}")

    for archetype, agents in sorted(by_archetype.items()):
        wealths = [a["final_wealth"] for a in agents]
        growths = [a["wealth_growth"] for a in agents]
        ranks = [a["rank_in_run"] for a in agents]

        w_mean = sum(wealths) / len(wealths)
        w_std = (sum((w - w_mean) ** 2 for w in wealths) / len(wealths)) ** 0.5
        g_mean = sum(growths) / len(growths)
        r_mean = sum(ranks) / len(ranks)

        print(f"{archetype:<22} ${w_mean:>7.0f} ± ${w_std:>6.0f}   {g_mean:>+8.1%}     {r_mean:>5.1f}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM simulation with DGP features")
    parser.add_argument(
        "--agents-per-archetype",
        type=int,
        default=10,
        help="Number of agents per archetype (default: 10, total = this × 6)",
    )
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per simulation")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs with different seeds")
    parser.add_argument(
        "--style", type=str, default="numeric", choices=["numeric", "narrative"], help="Prompt translation style"
    )
    parser.add_argument("--output", type=str, default="llm_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--with-strategy", action="store_true", help="Include CRRA strategy guide in system prompt")
    parser.add_argument(
        "--neutral-labels",
        action="store_true",
        help="A2 ablation: replace archetype names with neutral labels (Agent_A, Agent_B, ...)",
    )
    parser.add_argument(
        "--drop-feature",
        type=str,
        default=None,
        choices=["risk", "patience", "preferences", "production_skill"],
        help="A5 ablation: omit a feature from the prompt",
    )
    args = parser.parse_args()

    run_experiment(
        agents_per_archetype=args.agents_per_archetype,
        num_rounds=args.rounds,
        num_runs=args.runs,
        style=args.style,
        output_dir=args.output,
        base_seed=args.seed,
        with_strategy=args.with_strategy,
        neutral_labels=args.neutral_labels,
        drop_feature=args.drop_feature,
    )
