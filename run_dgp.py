"""Run the DGP simulation and produce ground-truth data.

Samples 10 agents per archetype (6 archetypes = 60 agents).
Runs each group of 6 agents (one per archetype) through 20 rounds of trading.
Outputs features, decisions, and wealth trajectories to CSV and JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from micro_economy.dgp import DGPAgent, DGPFeatures, dgp_decision
from micro_economy.dgp_personas import ALL_ARCHETYPES, sample_population
from micro_economy.models import Good, MarketState
from micro_economy.simulation import match_orders


def run_dgp_simulation(
    agents: list[DGPAgent],
    num_rounds: int = 1,
) -> tuple[list[DGPAgent], MarketState]:
    """Run a single simulation with DGP agents."""
    market = MarketState()
    agents_map = {a.name: a for a in agents}

    # Record initial wealth
    for agent in agents:
        agent.wealth_history.append(agent.net_worth(market.prices))

    for round_num in range(1, num_rounds + 1):
        all_orders = []
        for agent in agents:
            orders = dgp_decision(agent, market, round_num)
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

        # Execute trades (need adapter since DGPAgent has different inventory structure)
        for t in trades:
            buyer = agents_map[t.buyer]
            seller = agents_map[t.seller]
            total_cost = t.price * t.quantity

            if buyer.budget >= total_cost and seller.inventory.get(t.good, 0) >= t.quantity:
                buyer.budget -= total_cost
                seller.budget += total_cost
                seller.inventory[t.good] = seller.inventory.get(t.good, 0) - t.quantity
                buyer.inventory[t.good] = buyer.inventory.get(t.good, 0) + t.quantity

        market.trade_log.extend(trades)
        market.update_prices()
        market.record_prices()

        # Record wealth
        for agent in agents:
            agent.wealth_history.append(agent.net_worth(market.prices))

    return agents, market


def create_dgp_agents(features_list: list[DGPFeatures]) -> list[DGPAgent]:
    """Create DGP agents from feature specs with initial inventory.

    Agents start with large production inventory and small budgets
    to force trading (selling surplus to buy what they need).
    """
    agents = []
    for f in features_list:
        # Override budget to be small — forces agents to sell to buy
        agent = DGPAgent(features=f)
        agents.append(agent)
    return agents


def run_experiment(
    agents_per_archetype: int = 10,
    num_rounds: int = 10,
    num_runs: int = 10,
    output_dir: str = "dgp_results",
    base_seed: int = 42,
) -> list[dict]:
    """Run the full DGP experiment.

    Each run: all agents (agents_per_archetype × 6 archetypes) trade together
    in one simulation. Multiple runs use different seeds for statistical power.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    num_archetypes = len(ALL_ARCHETYPES)
    total_agents = agents_per_archetype * num_archetypes

    print(f"{agents_per_archetype} agents × {num_archetypes} archetypes = {total_agents} agents per run")
    print(f"Running {num_runs} runs × {num_rounds} rounds (seeds {base_seed}–{base_seed + num_runs - 1})\n")

    all_results = []

    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        random.seed(seed)

        population = sample_population(ALL_ARCHETYPES, samples_per_archetype=agents_per_archetype)
        agents = create_dgp_agents(population)

        print(f"Run {run_idx + 1}/{num_runs} (seed {seed}): ", end="", flush=True)
        agents, market = run_dgp_simulation(agents, num_rounds=num_rounds)

        # Collect results
        run_results = []
        for agent in agents:
            result = {
                "run_id": run_idx,
                "seed": seed,
                "features": agent.features.to_dict(),
                "name": agent.name,
                "archetype": agent.features.archetype,
                "risk_appetite": agent.features.risk_appetite,
                "budget_initial": agent.features.budget,
                "patience": agent.features.patience,
                "production_skill": agent.features.production_skill.value,
                "preferences": {g.value: round(v, 4) for g, v in agent.features.preferences.items()},
                "wealth_trajectory": [round(w, 2) for w in agent.wealth_history],
                "final_wealth": round(agent.wealth_history[-1], 2),
                "wealth_growth": round(
                    (agent.wealth_history[-1] - agent.wealth_history[0]) / max(agent.wealth_history[0], 1), 4
                ),
                "wealth_volatility": round(_volatility(agent.wealth_history), 4),
                "num_decisions": len(agent.decision_history),
                "final_inventory": {g.value: agent.inventory.get(g, 0) for g in Good},
                "final_budget": round(agent.budget, 2),
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
    _save_results(all_results, output_path)

    # Print summary
    _print_summary(all_results)

    return all_results


def _volatility(values: list[float]) -> float:
    """Coefficient of variation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (variance ** 0.5) / mean


def _save_results(results: list[dict], output_path: Path):
    """Save results to JSON and CSV."""
    # Full JSON
    json_path = output_path / "dgp_ground_truth.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: {json_path}")

    # Flat CSV (one row per agent)
    csv_path = output_path / "dgp_ground_truth.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "name", "archetype",
            "risk_appetite", "budget_initial", "patience", "production_skill",
            "pref_food", "pref_tools", "pref_luxury", "pref_medicine",
            "final_wealth", "wealth_growth", "wealth_volatility",
            "rank_in_run", "final_budget",
        ])
        for r in results:
            writer.writerow([
                r["run_id"], r["name"], r["archetype"],
                r["risk_appetite"], r["budget_initial"], r["patience"],
                r["production_skill"],
                r["preferences"]["food"], r["preferences"]["tools"],
                r["preferences"]["luxury"], r["preferences"]["medicine"],
                r["final_wealth"], r["wealth_growth"], r["wealth_volatility"],
                r["rank_in_run"], r["final_budget"],
            ])
    print(f"Summary CSV: {csv_path}")

    # Wealth trajectories CSV (for time-series analysis)
    traj_path = output_path / "dgp_wealth_trajectories.csv"
    with open(traj_path, "w", newline="") as f:
        writer = csv.writer(f)
        max_rounds = max(len(r["wealth_trajectory"]) for r in results)
        header = ["run_id", "name", "archetype"] + [f"round_{t}" for t in range(max_rounds)]
        writer.writerow(header)
        for r in results:
            row = [r["run_id"], r["name"], r["archetype"]] + r["wealth_trajectory"]
            writer.writerow(row)
    print(f"Trajectories: {traj_path}")

    # Features-only CSV (for later LLM prompt generation)
    feat_path = output_path / "dgp_features.csv"
    with open(feat_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "name", "archetype",
            "risk_appetite", "budget_initial", "patience", "production_skill",
            "pref_food", "pref_tools", "pref_luxury", "pref_medicine",
        ])
        for r in results:
            writer.writerow([
                r["run_id"], r["name"], r["archetype"],
                r["risk_appetite"], r["budget_initial"], r["patience"],
                r["production_skill"],
                r["preferences"]["food"], r["preferences"]["tools"],
                r["preferences"]["luxury"], r["preferences"]["medicine"],
            ])
    print(f"Features only: {feat_path}")


def _print_summary(results: list[dict]):
    """Print archetype-level summary statistics."""
    from collections import defaultdict

    by_archetype = defaultdict(list)
    for r in results:
        by_archetype[r["archetype"]].append(r)

    print(f"\n{'='*70}")
    print(f"{'Archetype':<22} {'Wealth (mean±std)':<22} {'Growth':<14} {'Rank (mean)':<12} {'Volatility'}")
    print(f"{'='*70}")

    for archetype, agents in sorted(by_archetype.items()):
        wealths = [a["final_wealth"] for a in agents]
        growths = [a["wealth_growth"] for a in agents]
        ranks = [a["rank_in_run"] for a in agents]
        vols = [a["wealth_volatility"] for a in agents]

        w_mean = sum(wealths) / len(wealths)
        w_std = (sum((w - w_mean)**2 for w in wealths) / len(wealths)) ** 0.5
        g_mean = sum(growths) / len(growths)
        r_mean = sum(ranks) / len(ranks)
        v_mean = sum(vols) / len(vols)

        print(f"{archetype:<22} ${w_mean:>7.0f} ± ${w_std:>6.0f}   {g_mean:>+8.1%}     {r_mean:>5.1f}       {v_mean:.4f}")

    print(f"{'='*70}")


def _plot_charts(results: list[dict], output_path: Path):
    """Generate matplotlib charts and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    from collections import defaultdict

    import matplotlib.pyplot as plt

    ARCH_COLORS = {
        'cautious_farmer': '#22c55e',
        'aggressive_merchant': '#ef4444',
        'pragmatic_doctor': '#3b82f6',
        'shrewd_speculator': '#f59e0b',
        'fair_toolmaker': '#8b5cf6',
        'survivalist': '#06b6d4',
    }

    dark = {
        'figure.facecolor': '#0f1117',
        'axes.facecolor': '#1a1d27',
        'axes.edgecolor': '#2e3345',
        'axes.labelcolor': '#9ca3af',
        'text.color': '#e4e4e7',
        'xtick.color': '#9ca3af',
        'ytick.color': '#9ca3af',
        'grid.color': '#2e3345',
        'legend.facecolor': '#1a1d27',
        'legend.edgecolor': '#2e3345',
    }
    plt.rcParams.update(dark)

    by_archetype = defaultdict(list)
    for r in results:
        by_archetype[r["archetype"]].append(r)

    archetypes = sorted(by_archetype.keys())

    # --- Chart 1: Wealth trajectories (all agents, colored by archetype) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    legend_added = set()
    for arch in archetypes:
        color = ARCH_COLORS.get(arch, '#888')
        for agent in by_archetype[arch]:
            traj = agent["wealth_trajectory"]
            label = arch.replace('_', ' ') if arch not in legend_added else None
            ax.plot(traj, color=color, alpha=0.3, linewidth=1, label=label)
            legend_added.add(arch)
    ax.set_xlabel("Round")
    ax.set_ylabel("Wealth ($)")
    ax.set_title("Wealth Trajectories by Archetype")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "trajectories.png", dpi=150)
    plt.close(fig)
    print(f"Chart: {output_path / 'trajectories.png'}")

    # --- Chart 2: Mean final wealth bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    means = [sum(a["final_wealth"] for a in by_archetype[arch]) / len(by_archetype[arch]) for arch in archetypes]
    stds = [
        (sum((a["final_wealth"] - means[i])**2 for a in by_archetype[arch]) / len(by_archetype[arch])) ** 0.5
        for i, arch in enumerate(archetypes)
    ]
    colors = [ARCH_COLORS.get(a, '#888') for a in archetypes]
    bars = ax.bar([a.replace('_', ' ') for a in archetypes], means, yerr=stds,
                  color=colors, alpha=0.7, edgecolor=colors, capsize=5)
    ax.set_ylabel("Mean Final Wealth ($)")
    ax.set_title("Final Wealth by Archetype")
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right')
    fig.tight_layout()
    fig.savefig(output_path / "wealth_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Chart: {output_path / 'wealth_comparison.png'}")

    # --- Chart 3: Mean rank bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_ranks = [sum(a["rank_in_run"] for a in by_archetype[arch]) / len(by_archetype[arch]) for arch in archetypes]
    bars = ax.bar([a.replace('_', ' ') for a in archetypes], mean_ranks,
                  color=colors, alpha=0.7, edgecolor=colors)
    ax.set_ylabel("Mean Rank (1 = best)")
    ax.set_title("Average Rank by Archetype")
    ax.invert_yaxis()
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right')
    fig.tight_layout()
    fig.savefig(output_path / "rank_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Chart: {output_path / 'rank_comparison.png'}")

    # --- Chart 4: Risk vs Final Wealth scatter ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for arch in archetypes:
        color = ARCH_COLORS.get(arch, '#888')
        risks = [a["risk_appetite"] for a in by_archetype[arch]]
        wealths = [a["final_wealth"] for a in by_archetype[arch]]
        ax.scatter(risks, wealths, c=color, label=arch.replace('_', ' '),
                   alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax.set_xlabel("Risk Appetite")
    ax.set_ylabel("Final Wealth ($)")
    ax.set_title("Risk Appetite vs Final Wealth")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "risk_vs_wealth.png", dpi=150)
    plt.close(fig)
    print(f"Chart: {output_path / 'risk_vs_wealth.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DGP ground-truth simulation")
    parser.add_argument("--agents-per-archetype", type=int, default=10,
                        help="Number of agents per archetype (default: 10, total = this × 6)")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per simulation")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs with different seeds")
    parser.add_argument("--output", type=str, default="dgp_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    results = run_experiment(
        agents_per_archetype=args.agents_per_archetype,
        num_rounds=args.rounds,
        num_runs=args.runs,
        output_dir=args.output,
        base_seed=args.seed,
    )

    if not args.no_charts:
        _plot_charts(results, Path(args.output))
