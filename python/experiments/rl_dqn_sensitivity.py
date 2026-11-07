"""
Hyperparameter sensitivity of the DQN, one parameter at a time.

Base configuration (the benchmark's): learning rate 0.001, gamma 0.95, epsilon
decay 0.998, hidden size 128, 5,000 episodes. Each non-base value below is
trained with 5 seeds on 4 workloads (sizes 8, 10, 12 and 13). The base runs are
taken from rl_multi_seed_benchmark.py (same workloads, same seeds, same CPU
training), so they are not repeated here.

python experiments/rl_dqn_sensitivity.py [workers]

Writes results/rl_dqn_sensitivity_runs.csv (one row per run, resumable) and
results/rl_dqn_sensitivity_summary.csv / _findings.txt.
"""

from multiprocessing import Pool
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "python"))

import experiments.rl_multi_seed_benchmark as ms  # noqa: E402
from routing.open_path import path_cost  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_PATH = RESULTS_DIR / "rl_dqn_sensitivity_runs.csv"
SUMMARY_PATH = RESULTS_DIR / "rl_dqn_sensitivity_summary.csv"
FINDINGS_PATH = RESULTS_DIR / "rl_dqn_sensitivity_findings.txt"

WORKLOADS = (19, 4, 50, 63)  # sizes 8, 10, 12, 13
SEEDS = ms.SEEDS[:5]

# (parameter, DQNConfig field, value); the base value of each is not listed.
VARIANTS = (
    ("learning_rate", "learning_rate", 0.0003),
    ("learning_rate", "learning_rate", 0.003),
    ("gamma", "discount_factor", 0.80),
    ("gamma", "discount_factor", 0.99),
    ("epsilon_decay", "epsilon_decay", 0.995),
    ("epsilon_decay", "epsilon_decay", 0.9995),
    ("hidden_size", "hidden_dim", 64),
    ("hidden_size", "hidden_dim", 256),
    ("episodes", "episodes", 1000),
    ("episodes", "episodes", 2500),
    ("episodes", "episodes", 10000),
)

BASE = {
    "learning_rate": 0.001,
    "gamma": 0.95,
    "epsilon_decay": 0.998,
    "hidden_size": 128,
    "episodes": 5000,
}


def run_one(task):
    workload_id, parameter, field, value, seed = task
    work = ms._DATA[workload_id]

    environment = ms.base.RoutingEnvironment(
        deliveries=work["deliveries"],
        road_matrix=work["road_matrix"],
        start_time=work["start_time"],
    )

    config = ms.base.make_dqn_config()
    config = type(config)(
        **{**config.__dict__, "random_seed": seed, "device": "cpu", field: value}
    )
    agent = ms.base.DQNAgent(delivery_count=len(work["ids"]), config=config)

    tic = perf_counter()
    agent.train(environment)
    seconds = perf_counter() - tic

    order = list(agent.greedy_rollout(environment).delivery_order)
    position = {i: k for k, i in enumerate(work["ids"])}
    sequence = [position[i] for i in order]

    km = path_cost(work["distance"], sequence, closed=True)
    minutes = path_cost(work["travel"], sequence, closed=True)
    late, _ = ms.lateness(order, work)

    return {
        "workload_id": workload_id,
        "n": len(work["ids"]),
        "parameter": parameter,
        "value": value,
        "seed": seed,
        "gap_km_pct": (km - work["optimal_km"]) / work["optimal_km"] * 100.0,
        "gap_minutes_pct": (minutes - work["optimal_min"]) / work["optimal_min"] * 100.0,
        "late": late,
        "training_seconds": seconds,
    }


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(ms.RUNS_PATH)
    base = base[
        (base.algorithm == "dqn")
        & base.workload_id.isin(WORKLOADS)
        & base.seed.isin(SEEDS)
    ].copy()

    rows = []

    base_rows = []

    for parameter, value in BASE.items():
        part = base.assign(parameter=parameter, value=value)
        base_rows.append(part[list(runs.columns)])

    table = pd.concat([runs, *base_rows], ignore_index=True)

    for (parameter, value), part in table.groupby(["parameter", "value"]):
        per_workload = part.groupby("workload_id").gap_km_pct.mean()
        per_workload_min = part.groupby("workload_id").gap_minutes_pct.mean()

        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "is_base": bool(BASE[parameter] == value),
                "runs": len(part),
                "mean_gap_km_pct": per_workload.mean(),
                "mean_gap_minutes_pct": per_workload_min.mean(),
                "runs_at_optimum": int((part.gap_km_pct < 1e-6).sum()),
                "seed_std_gap_km_pct": part.groupby("workload_id").gap_km_pct.std(ddof=1).mean(),
                "late_per_run": part.late.mean(),
                "training_seconds": part.training_seconds.mean(),
            }
        )

    order = {p: i for i, p in enumerate(BASE)}

    return pd.DataFrame(rows).sort_values(
        ["parameter", "value"], key=lambda c: c.map(order) if c.name == "parameter" else c
    )


def report() -> None:
    runs = pd.read_csv(RUNS_PATH)
    summary = summarise(runs)
    summary.to_csv(SUMMARY_PATH, index=False)

    lines = [
        "DQN HYPERPARAMETER SENSITIVITY (one parameter at a time)",
        "",
        f"Workloads {list(WORKLOADS)} (sizes 8, 10, 12, 13), seeds {list(SEEDS)}; base = "
        "lr 0.001, gamma 0.95, epsilon decay 0.998, hidden 128, 5,000 episodes.",
        "Gap = % longer than the exact optimum, mean of the per-workload mean over seeds.",
        "",
        f"  {'parameter':14s} {'value':>8s} {'runs':>5s} {'km gap %':>9s} {'min gap %':>10s} "
        f"{'at optimum':>11s} {'seed std':>9s} {'late/run':>9s} {'train s':>8s}",
    ]

    for r in summary.itertuples():
        mark = "*" if r.is_base else " "
        lines.append(
            f"  {r.parameter:14s} {r.value:8g}{mark}{r.runs:5d} {r.mean_gap_km_pct:9.2f} "
            f"{r.mean_gap_minutes_pct:10.2f} {r.runs_at_optimum:6d}/{r.runs:<4d} "
            f"{r.seed_std_gap_km_pct:9.2f} {r.late_per_run:9.2f} {r.training_seconds:8.1f}"
        )

    lines += ["", "* = base value"]
    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    started = perf_counter()
    data = ms.build_data()
    data = {w: data[w] for w in WORKLOADS}

    done = set()
    rows = []

    if RUNS_PATH.exists():
        old = pd.read_csv(RUNS_PATH)
        rows = old.to_dict("records")
        done = set(zip(old.workload_id, old.parameter, old.value, old.seed))

    tasks = [
        (w, parameter, field, value, s)
        for parameter, field, value in VARIANTS
        for w in WORKLOADS
        for s in SEEDS
        if (w, parameter, value, s) not in done
    ]

    # Longest first (10,000 episodes), so the pool ends together.
    tasks.sort(key=lambda t: -(t[3] if t[1] == "episodes" else 5000))

    print(f"{len(tasks)} runs to do ({len(done)} saved), {workers} workers", flush=True)

    with Pool(workers, initializer=ms.init_worker, initargs=(data,)) as pool:
        for index, row in enumerate(pool.imap_unordered(run_one, tasks), start=1):
            rows.append(row)

            if index % 10 == 0 or index == len(tasks):
                pd.DataFrame(rows).to_csv(RUNS_PATH, index=False)
                print(f"{index}/{len(tasks)} runs, {perf_counter() - started:.0f}s", flush=True)

    report()
    print(f"finished in {perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        main()
