"""
Paired Wilcoxon signed-rank tests between the routing algorithms, workload by
workload, on the results already saved by the benchmarks.

    1. 108 workloads: Nearest Neighbor, Clarke-Wright, Q-Learning, SARSA, DQN
    2. 12 final-validation workloads: + K-Means-guided DQN
    3. 16 held-out test workloads: Proposed RL against the others

For every pair and metric the file gives wins, ties and losses, the mean and
median difference, the Wilcoxon p-value (pairs with a zero difference are
dropped, as the test requires) and the Holm-corrected p-value over all the
comparisons of the same source and metric.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

OUT_PATH = RESULTS_DIR / "rl_wilcoxon_pairwise.csv"
FINDINGS_PATH = RESULTS_DIR / "rl_wilcoxon_pairwise_findings.txt"

TOLERANCE = 1e-9


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values; NaN stays NaN."""

    p = np.array(p_values, dtype=float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    valid = int((~np.isnan(p)).sum())
    adjusted = np.full(len(p), np.nan)
    running = 0.0

    for rank, index in enumerate(order[:valid]):
        running = max(running, (valid - rank) * p[index])
        adjusted[index] = min(1.0, running)

    return adjusted.tolist()


def compare(table: pd.DataFrame, source: str, metric: str, names, label):
    """
    table: one row per workload, one column per algorithm (lower is better).
    """

    rows = []

    for a, b in combinations(names, 2):
        diff = (table[a] - table[b]).to_numpy(dtype=float)
        nonzero = diff[np.abs(diff) > TOLERANCE]

        p_value = np.nan

        if len(nonzero) >= 1:
            p_value = float(wilcoxon(nonzero).pvalue)

        rows.append(
            {
                "source": source,
                "metric": metric,
                "a": label[a],
                "b": label[b],
                "workloads": len(diff),
                "a_better": int((diff < -TOLERANCE).sum()),
                "ties": int((np.abs(diff) <= TOLERANCE).sum()),
                "b_better": int((diff > TOLERANCE).sum()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                "p_value": p_value,
            }
        )

    adjusted = holm([r["p_value"] for r in rows])

    for row, value in zip(rows, adjusted):
        row["p_holm"] = value

    return rows


def main() -> None:
    rows = []

    # 1. The 540-run benchmark.
    bench = pd.read_csv(RESULTS_DIR / "dataset_algorithm_all_results.csv")
    names = ["nearest_neighbor", "clarke_wright", "q_learning", "sarsa", "dqn"]
    label = {
        "nearest_neighbor": "Nearest Neighbor",
        "clarke_wright": "Clarke-Wright",
        "q_learning": "Q-Learning",
        "sarsa": "SARSA",
        "dqn": "DQN",
    }

    for metric, column in (
        ("distance_km", "total_distance_km"),
        ("travel_minutes", "total_travel_time_minutes"),
    ):
        wide = bench.pivot(index="workload_id", columns="algorithm", values=column)
        rows += compare(wide, "108 workloads", metric, names, label)

    # 2. Final validation, with the K-Means-guided DQN.
    final = pd.read_csv(RESULTS_DIR / "final_guided_dqn_four_method_workloads.csv")
    names = ["nearest_neighbor", "clarke_wright", "vanilla_dqn", "guided_dqn"]
    label = {
        "nearest_neighbor": "Nearest Neighbor",
        "clarke_wright": "Clarke-Wright",
        "vanilla_dqn": "DQN",
        "guided_dqn": "K-Means-guided DQN",
    }

    for metric, suffix in (
        ("distance_km", "distance_km"),
        ("travel_minutes", "travel_time_minutes"),
    ):
        wide = pd.DataFrame({n: final[f"{n}_{suffix}"] for n in names})
        rows += compare(wide, "12 validation workloads", metric, names, label)

    # 3. Held-out test of Proposed RL.
    test = pd.read_csv(RESULTS_DIR / "proposed_rl_final_locked_objective_details.csv")
    names = ["NN", "CW", "Q-Learning", "SARSA", "DQN", "Proposed RL"]
    label = {n: n for n in names}

    for metric, column in (
        ("distance_km", "total_distance_km"),
        ("travel_minutes", "total_travel_time_minutes"),
        ("objective_J", "locked_objective"),
    ):
        wide = test.pivot(index="workload_id", columns="algorithm", values=column)[names]
        rows += compare(wide, "16 test workloads", metric, names, label)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_PATH, index=False)

    lines = [
        "PAIRED WILCOXON TESTS BETWEEN ALGORITHMS (lower distance, time or J is better)",
        "",
        "a_better / ties / b_better count workloads. p_holm corrects for the number of",
        "pairs compared in the same source and metric.",
        "",
    ]

    for (source, metric), group in result.groupby(["source", "metric"], sort=False):
        lines.append(f"[{source}] {metric}")
        lines.append(
            f"  {'A':20s} {'B':20s} {'A<B':>4s} {'tie':>4s} {'A>B':>4s} "
            f"{'mean diff':>10s} {'p':>9s} {'p Holm':>9s}"
        )

        for r in group.itertuples():
            lines.append(
                f"  {r.a:20s} {r.b:20s} {r.a_better:4d} {r.ties:4d} {r.b_better:4d} "
                f"{r.mean_diff:10.3f} {r.p_value:9.2e} {r.p_holm:9.2e}"
            )

        lines.append("")

    FINDINGS_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
