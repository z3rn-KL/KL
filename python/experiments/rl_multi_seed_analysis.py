"""
Summary of rl_multi_seed_benchmark.py (and, when present, the lambda runs of
rl_guided_lambda_benchmark.py): gap to the exact optimum over 10 seeds and 20
workloads, seed-to-seed stability, training time, convergence curves and paired
Wilcoxon tests on the per-workload mean gap.

python experiments/rl_multi_seed_analysis.py
"""

from itertools import combinations
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "python"))

from experiments.rl_wilcoxon_pairwise import holm  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_PATH = RESULTS_DIR / "rl_multi_seed_runs.csv"
CURVES_PATH = RESULTS_DIR / "rl_multi_seed_curves.csv"
LAMBDA_PATH = RESULTS_DIR / "rl_guided_lambda_runs.csv"
HEADROOM_PATH = RESULTS_DIR / "benchmark_headroom_108_details.csv"

SUMMARY_PATH = RESULTS_DIR / "rl_multi_seed_summary.csv"
TESTS_PATH = RESULTS_DIR / "rl_multi_seed_tests.csv"
FINDINGS_PATH = RESULTS_DIR / "rl_multi_seed_findings.txt"
CURVE_FIGURE = RESULTS_DIR / "rl_multi_seed_convergence.png"
GAP_FIGURE = RESULTS_DIR / "rl_multi_seed_gaps.png"

LABEL = {
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
    "dqn": "DQN",
    "nearest_neighbor": "Nearest Neighbor",
    "clarke_wright": "Clarke-Wright",
    "nn_search": "NN + local search",
}
RL = ("q_learning", "sarsa", "dqn")
REFERENCES = ("nearest_neighbor", "clarke_wright", "nn_search")

BOOTSTRAP = 5000
GROUPS = ((5, 7), (8, 10), (11, 16))


def bootstrap(values: np.ndarray, rng) -> tuple[float, float]:
    means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(BOOTSTRAP)
    ]

    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    rng = np.random.default_rng(42)

    runs = pd.read_csv(RUNS_PATH)
    head = pd.read_csv(HEADROOM_PATH).set_index("workload_id")

    workloads = sorted(runs.workload_id.unique())
    seeds = [int(v) for v in sorted(runs.seed.unique())]

    # Per-workload mean gap (over seeds) of each RL algorithm, and the gap of
    # each classical reference on the same workloads.
    gap = {}

    for metric, column, ref_column in (
        ("km", "gap_km_pct", "{}_km"),
        ("min", "gap_minutes_pct", "{}_min"),
    ):
        table = (
            runs.groupby(["workload_id", "algorithm"])[column]
            .mean()
            .unstack("algorithm")
            .loc[workloads]
        )

        optimum = head.loc[workloads, f"optimal_{metric}"]

        for ref in REFERENCES:
            value = head.loc[workloads, ref_column.format(ref)]
            table[ref] = (value - optimum) / optimum * 100.0

        gap[metric] = table

    rows = []

    for name in (*RL, *REFERENCES):
        part = runs[runs.algorithm == name]
        row = {"method": LABEL[name], "kind": "rl" if name in RL else "classical"}

        for metric in ("km", "min"):
            per_workload = gap[metric][name].to_numpy()
            low, high = bootstrap(per_workload, rng)
            row[f"mean_gap_{metric}_pct"] = float(per_workload.mean())
            row[f"ci_low_{metric}"] = low
            row[f"ci_high_{metric}"] = high

        if name in RL:
            by_run = part.gap_km_pct.to_numpy()
            seed_std = part.groupby("workload_id").gap_km_pct.std(ddof=1)
            same_route = part.groupby("workload_id").greedy_route.nunique() == 1

            per_seed = part.groupby("seed").gap_km_pct.mean()
            row["seed_mean_gap_min_pct"] = float(per_seed.min())
            row["seed_mean_gap_max_pct"] = float(per_seed.max())
            row["seed_42_mean_gap_pct"] = float(per_seed.get(42, np.nan))

            row["runs"] = len(part)
            row["share_runs_at_optimum_km"] = float((by_run < 1e-6).mean())
            row["worst_run_gap_km_pct"] = float(by_run.max())
            row["seed_std_gap_km_pct"] = float(seed_std.mean())
            row["workloads_same_route_all_seeds"] = int(same_route.sum())
            row["late_per_run"] = float(part.late.mean())
            row["training_seconds"] = float(part.training_seconds.mean())

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    # Paired Wilcoxon on the per-workload mean gap.
    tests = []

    for metric in ("km", "min"):
        family = []

        for a, b in combinations((*RL, *REFERENCES), 2):
            if a in REFERENCES and b in REFERENCES:
                continue

            diff = (gap[metric][a] - gap[metric][b]).to_numpy()
            nonzero = diff[np.abs(diff) > 1e-9]

            p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else np.nan

            family.append(
                {
                    "metric": metric,
                    "a": LABEL[a],
                    "b": LABEL[b],
                    "workloads": len(diff),
                    "a_better": int((diff < -1e-9).sum()),
                    "ties": int((np.abs(diff) <= 1e-9).sum()),
                    "b_better": int((diff > 1e-9).sum()),
                    "mean_diff_pct_points": float(diff.mean()),
                    "p_value": p_value,
                }
            )

        for row, adjusted in zip(family, holm([r["p_value"] for r in family])):
            row["p_holm"] = adjusted

        tests += family

    pd.DataFrame(tests).to_csv(TESTS_PATH, index=False)

    # By size group (km gap, mean over workloads of the per-workload mean).
    by_size = {}

    for low, high in GROUPS:
        ids = [w for w in workloads if low <= head.loc[w, "n"] <= high]
        by_size[(low, high)] = (len(ids), gap["km"].loc[ids].mean())

    lines = [
        "MULTI-SEED BENCHMARK OF Q-LEARNING, SARSA AND DQN",
        "",
        f"{len(workloads)} workloads (sizes {int(head.loc[workloads, 'n'].min())} to "
        f"{int(head.loc[workloads, 'n'].max())}), {len(seeds)} seeds {seeds}, "
        f"{len(runs)} runs, 5,000 training episodes each.",
        "Gap = how much longer than the exact optimum (mean of the per-workload mean over seeds).",
        "The classical methods are single deterministic runs on the same workloads.",
        "",
        "DISTANCE GAP TO OPTIMUM (%)",
        f"  {'method':20s} {'mean':>7s} {'95% CI':>16s} {'at optimum':>11s} {'worst run':>10s} "
        f"{'seed std':>9s} {'same route':>11s} {'train s':>8s}",
    ]

    for r in summary.itertuples():
        extra = ""

        if r.kind == "rl":
            extra = (
                f"{r.share_runs_at_optimum_km * 100:10.1f}% {r.worst_run_gap_km_pct:10.2f} "
                f"{r.seed_std_gap_km_pct:9.2f} {int(r.workloads_same_route_all_seeds):8d}/{len(workloads)} "
                f"{r.training_seconds:8.1f}"
            )

        lines.append(
            f"  {r.method:20s} {r.mean_gap_km_pct:7.2f} "
            f"{r.ci_low_km:6.2f} to {r.ci_high_km:5.2f} {extra}"
        )

    lines += ["", "MEAN DISTANCE GAP OF EACH SEED OVER THE WORKLOADS (%): what one seed alone would report"]

    for r in summary[summary.kind == "rl"].itertuples():
        lines.append(
            f"  {r.method:20s} best seed {r.seed_mean_gap_min_pct:6.2f}  worst seed "
            f"{r.seed_mean_gap_max_pct:6.2f}  seed 42 {r.seed_42_mean_gap_pct:6.2f}  "
            f"all seeds {r.mean_gap_km_pct:6.2f}"
        )

    lines += ["", "TRAVEL-TIME GAP TO OPTIMUM (%)"]

    for r in summary.itertuples():
        lines.append(
            f"  {r.method:20s} {r.mean_gap_min_pct:7.2f} {r.ci_low_min:6.2f} to {r.ci_high_min:5.2f}"
        )

    lines += ["", "DISTANCE GAP BY WORKLOAD SIZE (%, mean per-workload gap)"]
    header = "  " + " ".join(f"{LABEL[n]:>17s}" for n in (*RL, *REFERENCES))
    lines.append(f"  {'size (workloads)':18s}" + header)

    for (low, high), (count, series) in by_size.items():
        values = " ".join(f"{series[n]:17.2f}" for n in (*RL, *REFERENCES))
        lines.append(f"  {f'{low}-{high} ({count})':18s}  {values}")

    lines += [
        "",
        "PAIRED WILCOXON ON THE PER-WORKLOAD MEAN DISTANCE GAP (Holm-corrected)",
        f"  {'A':18s} {'B':18s} {'A<B':>4s} {'tie':>4s} {'A>B':>4s} {'mean diff':>10s} {'p':>9s} {'p Holm':>9s}",
    ]

    for t in tests:
        if t["metric"] == "km":
            lines.append(
                f"  {t['a']:18s} {t['b']:18s} {t['a_better']:4d} {t['ties']:4d} {t['b_better']:4d} "
                f"{t['mean_diff_pct_points']:10.2f} {t['p_value']:9.2e} {t['p_holm']:9.2e}"
            )

    # Lambda runs, when available.
    if LAMBDA_PATH.exists():
        guided = pd.read_csv(LAMBDA_PATH)
        used = [int(v) for v in sorted(guided.seed.unique())]

        plain = runs[runs.algorithm == "dqn"].copy()
        plain = plain[plain.seed.isin(used) & plain.workload_id.isin(guided.workload_id.unique())]

        labels = pd.read_csv(RESULTS_DIR / "xedu_kmeans_clusters.csv")[["id", "cluster_id"]].astype({"id": str})
        cluster_of = dict(zip(labels["id"], labels["cluster_id"]))

        def plain_switches(route: str) -> tuple[int, float]:
            ids = route.split()
            count = sum(cluster_of[a] != cluster_of[b] for a, b in zip(ids, ids[1:]))

            return count, count / max(1, len(ids) - 1)

        plain[["cluster_switches", "switch_rate"]] = plain.greedy_route.apply(
            lambda r: pd.Series(plain_switches(r))
        )
        plain["penalty"] = 0.0

        table = pd.concat([plain, guided], ignore_index=True)

        lambda_rows = []

        for penalty, part in table.groupby("penalty"):
            lambda_rows.append(
                {
                    "penalty": penalty,
                    "runs": len(part),
                    "mean_gap_km_pct": part.groupby("workload_id").gap_km_pct.mean().mean(),
                    "mean_gap_min_pct": part.groupby("workload_id").gap_minutes_pct.mean().mean(),
                    "mean_switch_rate": part.switch_rate.mean(),
                    "late_per_run": part.late.mean(),
                    "training_seconds": part.training_seconds.mean(),
                }
            )

        pd.DataFrame(lambda_rows).to_csv(RESULTS_DIR / "rl_guided_lambda_summary.csv", index=False)

        # Paired Wilcoxon of each lambda against lambda = 0, per workload.
        per_workload = table.groupby(["penalty", "workload_id"])[["gap_km_pct", "switch_rate"]].mean()
        lambda_tests = []

        for penalty in sorted(p for p in table.penalty.unique() if p > 0):
            for metric in ("gap_km_pct", "switch_rate"):
                diff = (
                    per_workload.xs(penalty, level="penalty")[metric]
                    - per_workload.xs(0.0, level="penalty")[metric]
                ).to_numpy()
                nonzero = diff[np.abs(diff) > 1e-9]

                lambda_tests.append(
                    {
                        "penalty": penalty,
                        "metric": metric,
                        "workloads": len(diff),
                        "lower": int((diff < -1e-9).sum()),
                        "ties": int((np.abs(diff) <= 1e-9).sum()),
                        "higher": int((diff > 1e-9).sum()),
                        "mean_diff": float(diff.mean()),
                        "p_value": float(wilcoxon(nonzero).pvalue) if len(nonzero) else np.nan,
                    }
                )

        # Holm over the three lambdas of the same metric.
        for metric in ("gap_km_pct", "switch_rate"):
            family = [t for t in lambda_tests if t["metric"] == metric]

            for t, adjusted in zip(family, holm([t["p_value"] for t in family])):
                t["p_holm"] = adjusted

        pd.DataFrame(lambda_tests).to_csv(RESULTS_DIR / "rl_guided_lambda_tests.csv", index=False)

        lines += [
            "",
            f"K-MEANS-GUIDED DQN: lambda on {guided.workload_id.nunique()} workloads, seeds {used}",
            f"  {'lambda':>7s} {'runs':>5s} {'km gap %':>9s} {'min gap %':>10s} {'switch rate':>12s} {'late/run':>9s}",
        ]

        for r in lambda_rows:
            lines.append(
                f"  {r['penalty']:7.2f} {r['runs']:5d} {r['mean_gap_km_pct']:9.2f} "
                f"{r['mean_gap_min_pct']:10.2f} {r['mean_switch_rate']:12.3f} {r['late_per_run']:9.2f}"
            )

        lines += [
            "",
            "  Paired Wilcoxon of each lambda against lambda = 0 (per-workload mean; lower = smaller than lambda 0)",
            f"  {'lambda':>7s} {'metric':>12s} {'lower':>6s} {'tie':>4s} {'higher':>7s} {'mean diff':>10s} {'p':>8s} {'p Holm':>8s}",
        ]

        for t in lambda_tests:
            lines.append(
                f"  {t['penalty']:7.2f} {t['metric']:>12s} {t['lower']:6d} {t['ties']:4d} {t['higher']:7d} "
                f"{t['mean_diff']:10.3f} {t['p_value']:8.4f} {t['p_holm']:8.4f}"
            )

    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # Figure 1: gap per algorithm, one dot per workload.
    fig, ax = plt.subplots(figsize=(7, 3.8))
    order = (*RL, *REFERENCES)

    for k, name in enumerate(order):
        values = gap["km"][name].to_numpy()
        ax.scatter(
            np.full(len(values), k) + rng.uniform(-0.12, 0.12, len(values)),
            values,
            s=14,
            alpha=0.55,
            color="#2b5f9e" if name in RL else "#3a8a5a",
        )
        ax.plot([k - 0.28, k + 0.28], [values.mean()] * 2, color="#b8433c", lw=2)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL[n].replace(" ", "\n", 1) for n in order], fontsize=8)
    ax.set_ylabel("Dài hơn tối ưu (%), mỗi chấm một đợt")
    ax.grid(alpha=0.25)
    fig.savefig(GAP_FIGURE, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: convergence (reward normalised inside each workload).
    curves = pd.read_csv(CURVES_PATH)
    lo = curves.groupby("workload_id").mean_reward.transform("min")
    hi = curves.groupby("workload_id").mean_reward.transform("max")
    curves["scaled"] = (curves.mean_reward - lo) / (hi - lo).replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(7, 3.8))

    for name, color in zip(RL, ("#d9822b", "#7b5ea7", "#2b5f9e")):
        part = curves[curves.algorithm == name].groupby("episodes_done").scaled
        mean, std = part.mean(), part.std()
        ax.plot(mean.index, mean.values, color=color, label=LABEL[name])
        ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Số episode huấn luyện")
    ax.set_ylabel("Phần thưởng trung bình (chuẩn hóa theo đợt)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(CURVE_FIGURE, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
