"""Generate comparison charts from experiment results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Color palette
COLORS = ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#22c55e", "#3b82f6"]


def _group_by_config(results: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["config"]].append(r)
    return dict(groups)


def _setup_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#1a1d27",
        "axes.edgecolor": "#2e3345",
        "grid.color": "#2e3345",
        "text.color": "#e4e4e7",
        "xtick.color": "#9ca3af",
        "ytick.color": "#9ca3af",
        "axes.labelcolor": "#9ca3af",
        "font.size": 11,
    })


def plot_gini_comparison(results: list[dict], output_path: Path):
    """Box plot of Gini coefficient across configurations."""
    _setup_style()
    groups = _group_by_config(results)
    configs = list(groups.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [[r["gini"] for r in groups[c]] for c in configs]
    bp = ax.boxplot(data, labels=configs, patch_artist=True, widths=0.6)

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("#e4e4e7")

    ax.set_title("Wealth Inequality (Gini Coefficient) by Configuration", fontweight="bold", fontsize=13)
    ax.set_ylabel("Gini Coefficient")
    ax.set_xlabel("Configuration")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "gini_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved gini_comparison.png")


def plot_volatility_comparison(results: list[dict], output_path: Path):
    """Box plot of average price volatility across configurations."""
    _setup_style()
    groups = _group_by_config(results)
    configs = list(groups.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [[r["avg_volatility"] for r in groups[c]] for c in configs]
    bp = ax.boxplot(data, labels=configs, patch_artist=True, widths=0.6)

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("#e4e4e7")

    ax.set_title("Price Volatility by Configuration", fontweight="bold", fontsize=13)
    ax.set_ylabel("Avg Coefficient of Variation")
    ax.set_xlabel("Configuration")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "volatility_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved volatility_comparison.png")


def plot_trade_volume(results: list[dict], output_path: Path):
    """Bar chart of trade volume across configurations."""
    _setup_style()
    groups = _group_by_config(results)
    configs = list(groups.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    means = [sum(r["trade_volume"]["total_trades"] for r in groups[c]) / len(groups[c]) for c in configs]
    ax.bar(configs, means, color=COLORS[:len(configs)], alpha=0.8, edgecolor="#2e3345")

    # Add error bars
    stds = []
    for c in configs:
        vals = [r["trade_volume"]["total_trades"] for r in groups[c]]
        m = sum(vals) / len(vals)
        stds.append((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5)
    ax.errorbar(configs, means, yerr=stds, fmt="none", ecolor="#e4e4e7", capsize=4)

    ax.set_title("Total Trades by Configuration", fontweight="bold", fontsize=13)
    ax.set_ylabel("Number of Trades (mean ± std)")
    ax.set_xlabel("Configuration")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "trade_volume.png", dpi=150)
    plt.close(fig)
    print("  Saved trade_volume.png")


def plot_wealth_spread(results: list[dict], output_path: Path):
    """Box plot of wealth spread (max - min net worth)."""
    _setup_style()
    groups = _group_by_config(results)
    configs = list(groups.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [[r["wealth"]["spread"] for r in groups[c]] for c in configs]
    bp = ax.boxplot(data, labels=configs, patch_artist=True, widths=0.6)

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("#e4e4e7")

    ax.set_title("Wealth Spread (Max - Min Net Worth) by Configuration", fontweight="bold", fontsize=13)
    ax.set_ylabel("Wealth Spread ($)")
    ax.set_xlabel("Configuration")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "wealth_spread.png", dpi=150)
    plt.close(fig)
    print("  Saved wealth_spread.png")


def plot_efficiency_comparison(results: list[dict], output_path: Path):
    """Box plot of market efficiency across configurations."""
    _setup_style()
    groups = _group_by_config(results)
    configs = list(groups.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [[r["market_efficiency"] for r in groups[c]] for c in configs]
    bp = ax.boxplot(data, labels=configs, patch_artist=True, widths=0.6)

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("#e4e4e7")

    ax.set_title("Market Efficiency by Configuration (lower = more stable)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Avg Price Change (normalized)")
    ax.set_xlabel("Configuration")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "market_efficiency.png", dpi=150)
    plt.close(fig)
    print("  Saved market_efficiency.png")


def plot_summary_dashboard(results: list[dict], output_path: Path):
    """Combined 2x2 dashboard of key metrics."""
    _setup_style()
    groups = _group_by_config(results)
    configs = list(groups.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Micro-Economy Experiment Dashboard", fontsize=16, fontweight="bold", y=0.98)

    metrics = [
        ("gini", "Gini Coefficient", "Wealth Inequality"),
        ("avg_volatility", "Coefficient of Variation", "Price Volatility"),
        ("market_efficiency", "Normalized Price Change", "Market Efficiency"),
    ]

    for idx, (key, ylabel, title) in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]
        data = [[r[key] for r in groups[c]] for c in configs]
        bp = ax.boxplot(data, labels=configs, patch_artist=True, widths=0.5)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("#e4e4e7")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Trade volume bar chart
    ax = axes[1][1]
    means = [sum(r["trade_volume"]["total_trades"] for r in groups[c]) / len(groups[c]) for c in configs]
    ax.bar(configs, means, color=COLORS[:len(configs)], alpha=0.8)
    ax.set_title("Trade Volume", fontweight="bold")
    ax.set_ylabel("Total Trades (mean)")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "dashboard.png", dpi=150)
    plt.close(fig)
    print("  Saved dashboard.png")


def generate_all_charts(results: list[dict], output_path: Path):
    """Generate all comparison charts."""
    print("\nGenerating charts...")
    plot_gini_comparison(results, output_path)
    plot_volatility_comparison(results, output_path)
    plot_trade_volume(results, output_path)
    plot_wealth_spread(results, output_path)
    plot_efficiency_comparison(results, output_path)
    plot_summary_dashboard(results, output_path)
