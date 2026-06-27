"""Deep quantitative analysis across DGP and LLM conditions."""
import json
import math
import os
from collections import defaultdict

# --- Config ---
BASE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = {
    "DGP (ground truth)": "dgp_results_v3/dgp_ground_truth.json",
    "Numeric baseline": "llm_numeric_v3/llm_numeric_results.json",
    "Narrative baseline": "llm_narrative_v3/llm_narrative_results.json",
    "Neutral labels": "llm_neutral_numeric_v3/llm_numeric_results.json",
    "Drop risk": "llm_drop_risk_v3/llm_numeric_results.json",
    "Drop patience": "llm_drop_patience_v3/llm_numeric_results.json",
    "Drop preferences": "llm_drop_preferences_v3/llm_numeric_results.json",
    "Drop production": "llm_drop_production_v3/llm_numeric_results.json",
}

def _mean(xs):
    return sum(xs) / len(xs)

def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def _pearsonr(xs, ys):
    n = len(xs)
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx, sy = _std(xs), _std(ys)
    if sx == 0 or sy == 0:
        return 0.0, 1.0
    r = cov / (sx * sy)
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # two-tailed p approximation
    df = n - 2
    p = 2 * (1 - _t_cdf(abs(t), df))
    return r, p

def _t_cdf(t, df):
    x = df / (df + t * t)
    return 1 - 0.5 * _betainc(df / 2, 0.5, x)

def _betainc(a, b, x, steps=200):
    from functools import reduce
    # numerical integration via Simpson's rule
    def f(u):
        if u <= 0 or u >= 1:
            return 0.0
        try:
            return u ** (a - 1) * (1 - u) ** (b - 1)
        except (OverflowError, ValueError):
            return 0.0
    h = x / steps
    s = f(0) + f(x)
    for i in range(1, steps):
        coeff = 4 if i % 2 else 2
        s += coeff * f(i * h)
    integral = s * h / 3
    # beta function via log-gamma
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return integral / math.exp(log_beta)


ARCHETYPES = [
    "cautious_farmer",
    "aggressive_merchant",
    "pragmatic_doctor",
    "shrewd_speculator",
    "fair_toolmaker",
    "survivalist",
]


def load_data():
    all_data = {}
    for label, path in CONDITIONS.items():
        fpath = os.path.join(BASE, path)
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found, skipping {label}")
            continue
        with open(fpath) as f:
            all_data[label] = json.load(f)
        print(f"Loaded {label}: {len(all_data[label])} agent-records")
    return all_data


def per_archetype_stats(all_data):
    print("\n" + "=" * 100)
    print("SECTION 1: PER-ARCHETYPE STATISTICS")
    print("=" * 100)

    for cond, records in all_data.items():
        print(f"\n--- {cond} ---")
        header = f"{'Archetype':<25} {'N':>5} {'Mean Wealth':>12} {'Std Wealth':>12} {'Mean Rank':>10} {'Mean Growth':>12}"
        print(header)
        print("-" * len(header))

        by_arch = defaultdict(list)
        for r in records:
            by_arch[r["archetype"]].append(r)

        for arch in ARCHETYPES:
            agents = by_arch.get(arch, [])
            if not agents:
                print(f"{arch:<25} {'N/A':>5}")
                continue
            wealths = [a["final_wealth"] for a in agents]
            ranks = [a["rank_in_run"] for a in agents]
            growths = [a.get("wealth_growth", 0) for a in agents]
            print(
                f"{arch:<25} {len(agents):>5} {_mean(wealths):>12.2f} {_std(wealths):>12.2f} "
                f"{_mean(ranks):>10.1f} {_mean(growths):>12.4f}"
            )


def within_archetype_cv(all_data):
    print("\n" + "=" * 100)
    print("SECTION 2: WITHIN-ARCHETYPE COEFFICIENT OF VARIATION (CV = std/mean)")
    print("=" * 100)

    # Print header
    header_parts = [f"{'Archetype':<25}"]
    cond_names = list(all_data.keys())
    for c in cond_names:
        short = c[:15]
        header_parts.append(f"{short:>16}")
    print("".join(header_parts))
    print("-" * (25 + 16 * len(cond_names)))

    for arch in ARCHETYPES:
        row = [f"{arch:<25}"]
        for cond in cond_names:
            records = all_data[cond]
            agents = [r for r in records if r["archetype"] == arch]
            if not agents:
                row.append(f"{'N/A':>16}")
                continue
            wealths = [a["final_wealth"] for a in agents]
            mean_w = _mean(wealths)
            std_w = _std(wealths)
            cv = std_w / mean_w if mean_w != 0 else float("inf")
            row.append(f"{cv:>16.4f}")
        print("".join(row))

    # Also show per-run within-archetype CV (averaged across runs)
    print("\n--- Per-run within-archetype CV (averaged across runs) ---")
    header_parts = [f"{'Archetype':<25}"]
    for c in cond_names:
        short = c[:15]
        header_parts.append(f"{short:>16}")
    print("".join(header_parts))
    print("-" * (25 + 16 * len(cond_names)))

    for arch in ARCHETYPES:
        row = [f"{arch:<25}"]
        for cond in cond_names:
            records = all_data[cond]
            by_run = defaultdict(list)
            for r in records:
                if r["archetype"] == arch:
                    by_run[r["run_id"]].append(r["final_wealth"])
            if not by_run:
                row.append(f"{'N/A':>16}")
                continue
            run_cvs = []
            for run_id, wealths in by_run.items():
                m = _mean(wealths)
                s = _std(wealths)
                if m > 0:
                    run_cvs.append(s / m)
            row.append(f"{_mean(run_cvs):>16.4f}")
        print("".join(row))


def feature_correlations(all_data):
    print("\n" + "=" * 100)
    print("SECTION 3: PEARSON CORRELATION OF DGP FEATURES vs FINAL_WEALTH")
    print("=" * 100)

    features = ["risk_appetite", "patience", "budget_initial"]

    for feat in features:
        print(f"\n--- Feature: {feat} ---")
        header = f"{'Condition':<25} {'Pearson r':>10} {'p-value':>12} {'N':>6}"
        print(header)
        print("-" * len(header))

        for cond, records in all_data.items():
            xs = []
            ys = []
            for r in records:
                val = r.get(feat)
                if val is None and "features" in r:
                    val = r["features"].get(feat)
                if val is not None:
                    xs.append(val)
                    ys.append(r["final_wealth"])
            if len(xs) < 3:
                print(f"{cond:<25} {'N/A':>10} {'N/A':>12} {len(xs):>6}")
                continue
            r_val, p_val = _pearsonr(xs, ys)
            print(f"{cond:<25} {r_val:>10.4f} {p_val:>12.2e} {len(xs):>6}")

    # Also do per-run correlations
    print("\n--- Per-run Pearson r (mean +/- std across runs) ---")
    for feat in features:
        print(f"\nFeature: {feat}")
        header = f"{'Condition':<25} {'Mean r':>10} {'Std r':>10} {'Runs':>6}"
        print(header)
        print("-" * len(header))

        for cond, records in all_data.items():
            by_run = defaultdict(lambda: ([], []))
            for r in records:
                val = r.get(feat)
                if val is None and "features" in r:
                    val = r["features"].get(feat)
                if val is not None:
                    xs, ys = by_run[r["run_id"]]
                    xs.append(val)
                    ys.append(r["final_wealth"])
            run_rs = []
            for run_id, (xs, ys) in by_run.items():
                if len(xs) >= 3:
                    r_val, _ = _pearsonr(xs, ys)
                    run_rs.append(r_val)
            if not run_rs:
                print(f"{cond:<25} {'N/A':>10}")
                continue
            print(
                f"{cond:<25} {_mean(run_rs):>10.4f} {_std(run_rs):>10.4f} {len(run_rs):>6}"
            )


def rank_comparison(all_data):
    """Compare mean rank per archetype across conditions."""
    print("\n" + "=" * 100)
    print("SECTION 4: MEAN RANK PER ARCHETYPE (lower = wealthier)")
    print("=" * 100)

    cond_names = list(all_data.keys())
    header_parts = [f"{'Archetype':<25}"]
    for c in cond_names:
        short = c[:15]
        header_parts.append(f"{short:>16}")
    print("".join(header_parts))
    print("-" * (25 + 16 * len(cond_names)))

    for arch in ARCHETYPES:
        row = [f"{arch:<25}"]
        for cond in cond_names:
            records = all_data[cond]
            agents = [r for r in records if r["archetype"] == arch]
            if not agents:
                row.append(f"{'N/A':>16}")
                continue
            ranks = [a["rank_in_run"] for a in agents]
            row.append(f"{_mean(ranks):>16.1f}")
        print("".join(row))


def wealth_ratio_vs_dgp(all_data):
    """LLM wealth as fraction of DGP wealth, per archetype."""
    print("\n" + "=" * 100)
    print("SECTION 5: LLM WEALTH AS FRACTION OF DGP WEALTH (per archetype)")
    print("=" * 100)

    if "DGP (ground truth)" not in all_data:
        print("No DGP data, skipping.")
        return

    dgp_records = all_data["DGP (ground truth)"]
    dgp_means = {}
    for arch in ARCHETYPES:
        agents = [r for r in dgp_records if r["archetype"] == arch]
        if agents:
            dgp_means[arch] = _mean([a["final_wealth"] for a in agents])

    cond_names = [c for c in all_data.keys() if c != "DGP (ground truth)"]
    header_parts = [f"{'Archetype':<25} {'DGP Mean':>12}"]
    for c in cond_names:
        short = c[:13]
        header_parts.append(f"{short:>14}")
    print("".join(header_parts))
    print("-" * (37 + 14 * len(cond_names)))

    for arch in ARCHETYPES:
        dgp_m = dgp_means.get(arch, 0)
        row = [f"{arch:<25} {dgp_m:>12.2f}"]
        for cond in cond_names:
            records = all_data[cond]
            agents = [r for r in records if r["archetype"] == arch]
            if not agents or dgp_m == 0:
                row.append(f"{'N/A':>14}")
                continue
            llm_m = _mean([a["final_wealth"] for a in agents])
            ratio = llm_m / dgp_m
            row.append(f"{ratio:>14.3f}")
        print("".join(row))


def main():
    all_data = load_data()
    per_archetype_stats(all_data)
    within_archetype_cv(all_data)
    feature_correlations(all_data)
    rank_comparison(all_data)
    wealth_ratio_vs_dgp(all_data)


if __name__ == "__main__":
    main()
