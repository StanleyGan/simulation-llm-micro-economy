"""Compare DGP ground truth against LLM benchmark results.

Loads results from run_dgp.py (ground truth) and run_llm_benchmark.py
(LLM simulation), computes comparison metrics, and generates charts.

Designed for multi-run experiments with 60 agents per run (10 per archetype).
Uses Spearman rank correlation (appropriate for n=60).
"""

from __future__ import annotations
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(path: Path) -> list[dict]:
    """Load results from a JSON file."""
    with open(path) as f:
        return json.load(f)


def spearman_rank_correlation(ranks_a: list[float], ranks_b: list[float]) -> float:
    """Compute Spearman rank correlation between two rank lists."""
    n = len(ranks_a)
    if n < 2:
        return 0.0
    d_sq = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    return 1 - (6 * d_sq) / (n * (n**2 - 1))


def _to_ordinal_ranks(values: list[float]) -> list[int]:
    """Convert raw values to ordinal ranks (1 = smallest value)."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, idx in enumerate(indexed):
        ranks[idx] = rank + 1
    return ranks


def compare_results(dgp_results: list[dict], llm_results: list[dict]) -> dict:
    """Compare DGP and LLM results across multiple runs.

    Each run has 60 agents (10 per archetype) all trading together.
    Computes Spearman ρ per run, then reports mean ± std across runs.
    """
    # Index by (run_id, name) for matching
    dgp_by_key = {}
    for r in dgp_results:
        key = (r["run_id"], r["name"])
        dgp_by_key[key] = r

    llm_by_key = {}
    for r in llm_results:
        key = (r["run_id"], r["name"])
        llm_by_key[key] = r

    # Find common keys
    common_keys = set(dgp_by_key.keys()) & set(llm_by_key.keys())
    if not common_keys:
        print("ERROR: No matching (run_id, name) pairs found!")
        return {}

    print(f"Matched {len(common_keys)} agent results across both simulations\n")

    # --- Per-run Spearman rank correlation ---
    runs = sorted(set(k[0] for k in common_keys))
    rank_correlations = []

    for run_id in runs:
        run_keys = sorted([k for k in common_keys if k[0] == run_id], key=lambda k: k[1])
        dgp_ranks = [dgp_by_key[k]["rank_in_run"] for k in run_keys]
        llm_ranks = [llm_by_key[k]["rank_in_run"] for k in run_keys]
        rho = spearman_rank_correlation(dgp_ranks, llm_ranks)
        rank_correlations.append(rho)
        n_agents = len(run_keys)
        print(f"  Run {run_id} (n={n_agents}): Spearman ρ = {rho:.3f}")

    mean_rho = sum(rank_correlations) / len(rank_correlations)
    std_rho = math.sqrt(sum((r - mean_rho)**2 for r in rank_correlations) / len(rank_correlations))
    print(f"\n  Spearman ρ across {len(runs)} runs: {mean_rho:.3f} ± {std_rho:.3f}")
    if mean_rho > 0.7:
        print("  → Strong agreement: LLM rankings closely match ground truth")
    elif mean_rho > 0.3:
        print("  → Moderate agreement: LLM partially captures ground truth ranking structure")
    else:
        print("  → Weak agreement: LLM rankings diverge significantly from ground truth")

    # --- Archetype-level comparison (aggregated across all runs) ---
    print(f"\n{'='*80}")
    print(f"{'Archetype':<22} {'GT Rank':<12} {'LLM Rank':<12} {'GT Wealth':<14} {'LLM Wealth':<14} {'Diff'}")
    print(f"{'='*80}")

    dgp_by_arch = defaultdict(list)
    llm_by_arch = defaultdict(list)
    for k in common_keys:
        arch = dgp_by_key[k]["archetype"]
        dgp_by_arch[arch].append(dgp_by_key[k])
        llm_by_arch[arch].append(llm_by_key[k])

    arch_comparison = []
    for arch in sorted(dgp_by_arch.keys()):
        dgp_agents = dgp_by_arch[arch]
        llm_agents = llm_by_arch[arch]

        dgp_rank = sum(a["rank_in_run"] for a in dgp_agents) / len(dgp_agents)
        llm_rank = sum(a["rank_in_run"] for a in llm_agents) / len(llm_agents)
        dgp_wealth = sum(a["final_wealth"] for a in dgp_agents) / len(dgp_agents)
        llm_wealth = sum(a["final_wealth"] for a in llm_agents) / len(llm_agents)
        wealth_diff = ((llm_wealth - dgp_wealth) / dgp_wealth * 100) if dgp_wealth else 0

        print(f"{arch:<22} {dgp_rank:>8.1f}     {llm_rank:>8.1f}     ${dgp_wealth:>9.0f}     ${llm_wealth:>9.0f}   {wealth_diff:>+6.1f}%")
        arch_comparison.append({
            "archetype": arch,
            "dgp_rank": dgp_rank, "llm_rank": llm_rank,
            "dgp_wealth": dgp_wealth, "llm_wealth": llm_wealth,
        })

    print(f"{'='*80}")

    # --- Overall archetype ranking correlation ---
    # Re-rank the mean ranks to ordinal (1-6) for Spearman
    dgp_mean_ranks = [a["dgp_rank"] for a in arch_comparison]
    llm_mean_ranks = [a["llm_rank"] for a in arch_comparison]
    dgp_ordinal = _to_ordinal_ranks(dgp_mean_ranks)
    llm_ordinal = _to_ordinal_ranks(llm_mean_ranks)
    overall_rho = spearman_rank_correlation(dgp_ordinal, llm_ordinal)
    print(f"\nArchetype-level Spearman ρ: {overall_rho:.3f}")

    return {
        "per_run_rho": rank_correlations,
        "mean_rho": mean_rho,
        "std_rho": std_rho,
        "archetype_rho": overall_rho,
        "archetype_comparison": arch_comparison,
        "num_runs": len(runs),
        "agents_per_run": len(run_keys) if runs else 0,
    }


def plot_comparison(dgp_results: list[dict], llm_results: list[dict],
                    comparison: dict, output_path: Path):
    """Generate comparison charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ARCH_COLORS = {
        'cautious_farmer': '#22c55e', 'aggressive_merchant': '#ef4444',
        'pragmatic_doctor': '#3b82f6', 'shrewd_speculator': '#f59e0b',
        'fair_toolmaker': '#8b5cf6', 'survivalist': '#06b6d4',
    }

    dark = {
        'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
        'axes.edgecolor': '#2e3345', 'axes.labelcolor': '#9ca3af',
        'text.color': '#e4e4e7', 'xtick.color': '#9ca3af',
        'ytick.color': '#9ca3af', 'grid.color': '#2e3345',
        'legend.facecolor': '#1a1d27', 'legend.edgecolor': '#2e3345',
    }
    plt.rcParams.update(dark)

    arch_comp = comparison["archetype_comparison"]
    archetypes = [a["archetype"].replace('_', ' ') for a in arch_comp]

    # --- Chart 1: Side-by-side wealth comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(archetypes))
    width = 0.35
    ax.bar([i - width/2 for i in x],
           [a["dgp_wealth"] for a in arch_comp],
           width, label='Ground Truth', color='#6366f1', alpha=0.8)
    ax.bar([i + width/2 for i in x],
           [a["llm_wealth"] for a in arch_comp],
           width, label='LLM', color='#f59e0b', alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(archetypes, rotation=20, ha='right')
    ax.set_ylabel("Mean Final Wealth ($)")
    ax.set_title(f"Ground Truth vs LLM: Final Wealth by Archetype (ρ = {comparison['archetype_rho']:.3f})")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "ground_truth_vs_llm_wealth.png", dpi=150)
    plt.close(fig)
    print(f"\nChart: {output_path / 'ground_truth_vs_llm_wealth.png'}")

    # --- Chart 2: Side-by-side rank comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x],
           [a["dgp_rank"] for a in arch_comp],
           width, label='Ground Truth', color='#6366f1', alpha=0.8)
    ax.bar([i + width/2 for i in x],
           [a["llm_rank"] for a in arch_comp],
           width, label='LLM', color='#f59e0b', alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(archetypes, rotation=20, ha='right')
    ax.set_ylabel("Mean Rank (1 = best)")
    ax.set_title("Ground Truth vs LLM: Average Rank by Archetype")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "ground_truth_vs_llm_rank.png", dpi=150)
    plt.close(fig)
    print(f"Chart: {output_path / 'ground_truth_vs_llm_rank.png'}")

    # --- Chart 3: Per-run rank correlation distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    rhos = comparison["per_run_rho"]
    ax.bar(range(len(rhos)), rhos, color='#6366f1', alpha=0.7, edgecolor='#6366f1')
    ax.axhline(y=comparison["mean_rho"], color='#f59e0b', linestyle='--',
               label=f'Mean ρ = {comparison["mean_rho"]:.3f} ± {comparison["std_rho"]:.3f}')
    ax.set_xlabel("Run")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Rank Correlation per Run (Ground Truth vs LLM)")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "ground_truth_vs_llm_correlation.png", dpi=150)
    plt.close(fig)
    print(f"Chart: {output_path / 'ground_truth_vs_llm_correlation.png'}")

    # --- Chart 4: Ground Truth rank vs LLM rank scatter (per agent) ---
    fig, ax = plt.subplots(figsize=(8, 8))
    dgp_by_key = {(r["run_id"], r["name"]): r for r in dgp_results}
    llm_by_key = {(r["run_id"], r["name"]): r for r in llm_results}
    common = set(dgp_by_key.keys()) & set(llm_by_key.keys())

    n_agents = comparison.get("agents_per_run", 60)
    for k in common:
        arch = dgp_by_key[k]["archetype"]
        color = ARCH_COLORS.get(arch, '#888')
        ax.scatter(dgp_by_key[k]["rank_in_run"], llm_by_key[k]["rank_in_run"],
                   c=color, alpha=0.3, s=20, edgecolors='white', linewidth=0.3)

    # Add legend
    for arch, color in ARCH_COLORS.items():
        ax.scatter([], [], c=color, label=arch.replace('_', ' '), s=40)
    ax.plot([0.5, n_agents + 0.5], [0.5, n_agents + 0.5], 'w--', alpha=0.3, label='Perfect agreement')
    ax.set_xlabel("GT Rank")
    ax.set_ylabel("LLM Rank")
    ax.set_title("Per-Agent Rank: Ground Truth vs LLM")
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(0.5, n_agents + 0.5)
    ax.set_ylim(0.5, n_agents + 0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "ground_truth_vs_llm_rank_scatter.png", dpi=150)
    plt.close(fig)
    print(f"Chart: {output_path / 'ground_truth_vs_llm_rank_scatter.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Ground Truth vs LLM results")
    parser.add_argument("--dgp", type=str, default="dgp_results/dgp_ground_truth.json",
                        help="Path to DGP results JSON")
    parser.add_argument("--llm", type=str, default="llm_results/llm_numeric_results.json",
                        help="Path to LLM results JSON")
    parser.add_argument("--output", type=str, default="comparison_results",
                        help="Output directory for charts")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)

    print("Loading results...")
    dgp_results = load_results(Path(args.dgp))
    llm_results = load_results(Path(args.llm))

    print(f"Ground truth: {len(dgp_results)} agents")
    print(f"LLM: {len(llm_results)} agents\n")

    comparison = compare_results(dgp_results, llm_results)

    # Save comparison
    with open(output_path / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved: {output_path / 'comparison.json'}")

    if not args.no_charts:
        plot_comparison(dgp_results, llm_results, comparison, output_path)
