"""Batch experiment runner — runs multiple persona configs x N runs, collects metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from micro_economy.experiments import EXPERIMENT_CONFIGS
from micro_economy.metrics import compute_all_metrics
from micro_economy.simulation import run_simulation_sync


def run_single(config_name: str, personas, num_rounds: int, run_id: int) -> dict:
    """Run one simulation and return metrics."""
    print(f"  Run {run_id + 1}: ", end="", flush=True)
    t0 = time.time()
    agents, market = run_simulation_sync(num_rounds=num_rounds, personas=personas)
    elapsed = time.time() - t0
    m = compute_all_metrics(agents, market)
    m["config"] = config_name
    m["run_id"] = run_id
    m["rounds"] = num_rounds
    m["elapsed_s"] = round(elapsed, 2)
    print(f"done ({elapsed:.1f}s) — gini={m['gini']}, trades={m['trade_volume']['total_trades']}, "
          f"avg_vol={m['avg_volatility']}")
    return m


def run_experiment(
    configs: dict[str, list] | None = None,
    num_runs: int = 10,
    num_rounds: int = 20,
    output_dir: str = "results",
) -> list[dict]:
    """Run all configurations and collect results."""
    configs = configs or EXPERIMENT_CONFIGS
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    all_results = []

    for config_name, personas in configs.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name} ({len(personas)} agents, {num_rounds} rounds, {num_runs} runs)")
        print(f"{'='*60}")

        config_results = []
        for run_id in range(num_runs):
            result = run_single(config_name, personas, num_rounds, run_id)
            config_results.append(result)

        # Summary
        ginis = [r["gini"] for r in config_results]
        vols = [r["avg_volatility"] for r in config_results]
        trades = [r["trade_volume"]["total_trades"] for r in config_results]
        efficiencies = [r["market_efficiency"] for r in config_results]

        print(f"\n  Summary for {config_name}:")
        print(
            f"    Gini:       mean={_mean(ginis):.4f}  std={_std(ginis):.4f}"
            f"  range=[{min(ginis):.4f}, {max(ginis):.4f}]"
        )
        print(f"    Volatility: mean={_mean(vols):.4f}  std={_std(vols):.4f}")
        print(f"    Trades:     mean={_mean(trades):.1f}  std={_std(trades):.1f}")
        print(f"    Efficiency: mean={_mean(efficiencies):.4f}  std={_std(efficiencies):.4f}")

        all_results.extend(config_results)

    # Save raw results as JSON
    json_path = output_path / "experiment_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {json_path}")

    # Save summary CSV
    csv_path = output_path / "experiment_summary.csv"
    _write_summary_csv(all_results, csv_path)
    print(f"Summary CSV saved to {csv_path}")

    # Generate charts
    try:
        from charts import generate_all_charts
        generate_all_charts(all_results, output_path)
        print(f"Charts saved to {output_path}/")
    except ImportError:
        print("(matplotlib not installed — skipping charts)")

    return all_results


def _write_summary_csv(results: list[dict], path: Path):
    """Write a flat CSV of key metrics per run."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "run_id", "rounds", "gini", "avg_volatility",
            "market_efficiency", "total_trades", "total_trade_value",
            "wealth_min", "wealth_max", "wealth_mean", "wealth_spread",
            "elapsed_s",
        ])
        for r in results:
            writer.writerow([
                r["config"], r["run_id"], r["rounds"], r["gini"],
                r["avg_volatility"], r["market_efficiency"],
                r["trade_volume"]["total_trades"],
                r["trade_volume"]["total_value"],
                r["wealth"]["min"], r["wealth"]["max"],
                r["wealth"]["mean"], r["wealth"]["spread"],
                r["elapsed_s"],
            ])


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run micro-economy experiments")
    parser.add_argument("--runs", type=int, default=10, help="Runs per configuration")
    parser.add_argument("--rounds", type=int, default=20, help="Rounds per simulation")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Specific configs to run (default: all)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    configs = EXPERIMENT_CONFIGS
    if args.configs:
        configs = {k: v for k, v in EXPERIMENT_CONFIGS.items() if k in args.configs}
        if not configs:
            print(f"No matching configs. Available: {list(EXPERIMENT_CONFIGS.keys())}")
            sys.exit(1)

    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    print(f"Mode: {'OpenAI (gpt-4o-mini)' if has_key else 'Mock (heuristic)'}")
    print(f"Configs: {list(configs.keys())}")
    print(f"Runs per config: {args.runs}")
    print(f"Rounds per run: {args.rounds}")

    run_experiment(configs=configs, num_runs=args.runs, num_rounds=args.rounds,
                   output_dir=args.output)
