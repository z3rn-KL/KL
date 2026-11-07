"""
Cluster-switch penalty lambda of the K-Means-guided DQN, on many workloads and
seeds.

lambda = 0.25, 0.5 and 1.0 are trained here on the 20 workloads of
rl_multi_seed_benchmark.py with its first 5 seeds. lambda = 0 is the plain DQN
of that benchmark (same configuration and seeds), so it is not trained again.

python experiments/rl_guided_lambda_benchmark.py [workers]

Writes results/rl_guided_lambda_runs.csv (one row per run, resumable).
"""

from multiprocessing import Pool
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "python"))

import experiments.rl_multi_seed_benchmark as ms  # noqa: E402
from rl.routing_environment import RewardConfig  # noqa: E402
from routing.open_path import path_cost  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_PATH = RESULTS_DIR / "rl_guided_lambda_runs.csv"
KMEANS_PATH = RESULTS_DIR / "xedu_kmeans_clusters.csv"

LAMBDAS = (0.25, 0.5, 1.0)
SEEDS = ms.SEEDS[:5]


def switches(order, labels_by_id):
    """Consecutive deliveries that lie in different K-Means clusters."""

    return sum(
        labels_by_id[a] != labels_by_id[b] for a, b in zip(order, order[1:])
    )


def run_one(task):
    workload_id, penalty, seed = task
    work = ms._DATA[workload_id]

    environment = ms.base.RoutingEnvironment(
        deliveries=work["deliveries"],
        road_matrix=work["road_matrix"],
        start_time=work["start_time"],
        reward_config=RewardConfig(cluster_switch_penalty=penalty),
        delivery_cluster_ids=work["clusters"],
    )

    config = ms.base.make_dqn_config()
    config = type(config)(**{**config.__dict__, "random_seed": seed, "device": "cpu"})
    agent = ms.base.DQNAgent(delivery_count=len(work["ids"]), config=config)

    tic = perf_counter()
    agent.train(environment)
    seconds = perf_counter() - tic

    order = list(agent.greedy_rollout(environment).delivery_order)

    position = {i: k for k, i in enumerate(work["ids"])}
    sequence = [position[i] for i in order]

    km = path_cost(work["distance"], sequence, closed=True)
    minutes = path_cost(work["travel"], sequence, closed=True)
    late, late_minutes = ms.lateness(order, work)
    labels = dict(zip(work["ids"], work["clusters"]))
    count = switches(order, labels)

    return {
        "workload_id": workload_id,
        "n": len(work["ids"]),
        "penalty": penalty,
        "seed": seed,
        "distance_km": km,
        "travel_minutes": minutes,
        "optimal_km": work["optimal_km"],
        "optimal_minutes": work["optimal_min"],
        "gap_km_pct": (km - work["optimal_km"]) / work["optimal_km"] * 100.0,
        "gap_minutes_pct": (minutes - work["optimal_min"]) / work["optimal_min"] * 100.0,
        "late": late,
        "late_minutes": late_minutes,
        "cluster_switches": count,
        "switch_rate": count / max(1, len(order) - 1),
        "training_seconds": seconds,
        "greedy_route": " ".join(order),
    }


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    started = perf_counter()
    data = ms.build_data()

    labels = pd.read_csv(KMEANS_PATH)[["id", "cluster_id"]].astype({"id": str})
    cluster_of = dict(zip(labels["id"], labels["cluster_id"].astype(int)))

    for work in data.values():
        work["clusters"] = [cluster_of[i] for i in work["ids"]]

    done = set()
    rows = []

    if RUNS_PATH.exists():
        old = pd.read_csv(RUNS_PATH)
        rows = old.to_dict("records")
        done = set(zip(old.workload_id, old.penalty, old.seed))

    tasks = [
        (w, p, s)
        for p in LAMBDAS
        for w in sorted(data, key=lambda k: -len(data[k]["ids"]))
        for s in SEEDS
        if (w, p, s) not in done
    ]

    print(f"{len(tasks)} runs to do ({len(done)} saved), {workers} workers", flush=True)

    with Pool(workers, initializer=ms.init_worker, initargs=(data,)) as pool:
        for index, row in enumerate(pool.imap_unordered(run_one, tasks), start=1):
            rows.append(row)

            if index % 10 == 0 or index == len(tasks):
                pd.DataFrame(rows).to_csv(RUNS_PATH, index=False)
                print(f"{index}/{len(tasks)} runs, {perf_counter() - started:.0f}s", flush=True)

    print(f"finished in {perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
