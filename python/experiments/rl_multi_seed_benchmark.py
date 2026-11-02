"""
Multi-seed, multi-workload benchmark of Q-Learning, SARSA and DQN.

Every algorithm is trained with 10 random seeds on 20 of the 108 benchmark
workloads (all sizes 5 to 16), with the benchmark's own configuration (5,000
episodes, learning rates and epsilon schedule of core_algorithm_full_benchmark).
Each run is compared with the exact optimum of its workload, so the result is a
distribution of gaps, not one lucky seed.

python experiments/rl_multi_seed_benchmark.py [workers]

Writes results/rl_multi_seed_runs.csv (one row per run, resumable) and
results/rl_multi_seed_curves.csv (training reward per block of episodes).
"""

from multiprocessing import Pool
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "python"))

import experiments.core_algorithm_full_benchmark as base  # noqa: E402
import experiments.daily_planning_experiment as daily  # noqa: E402
from experiments.core_algorithm_dataset_benchmark import (  # noqa: E402
    select_dataset_workloads,
)
from routing import RoadMatrixResult  # noqa: E402
from routing.open_path import optimal_path, path_cost  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_PATH = RESULTS_DIR / "rl_multi_seed_runs.csv"
CURVES_PATH = RESULTS_DIR / "rl_multi_seed_curves.csv"

SEEDS = (1, 7, 13, 21, 42, 99, 123, 2026, 27182, 31415)
ALGORITHMS = ("q_learning", "sarsa", "dqn")
EPISODES = base.TRAINING_EPISODES
CURVE_BLOCK = 100

# How many workloads to take at each size (20 in total, sizes 5 to 16).
PICK = {5: 3, 6: 2, 7: 2, 8: 2, 9: 1, 10: 2, 11: 1, 12: 2, 13: 2, 14: 1, 15: 1, 16: 1}

_DATA: dict = {}


def pick_workloads(workloads: list[pd.DataFrame]) -> list[int]:
    """
    Workload ids (1-based, as in the 108-workload benchmark), evenly spread
    inside each size class.
    """

    by_size: dict[int, list[int]] = {}

    for workload_id, frame in enumerate(workloads, start=1):
        by_size.setdefault(len(frame), []).append(workload_id)

    chosen = []

    for size, count in PICK.items():
        ids = by_size[size]
        positions = np.linspace(0, len(ids) - 1, count).round().astype(int)
        chosen += [ids[p] for p in positions]

    return sorted(chosen)


def build_data():
    """
    Everything a worker needs, built once in the parent: deliveries, road
    matrix cut from the global matrix, deadlines and the exact optimum.
    """

    delivery_df, lookup = base.load_delivery_dataframe()
    snapped = base.load_snapped_nodes()
    frame = base.prepare_dataframe(delivery_df=delivery_df, snapped_df=snapped)
    workloads = select_dataset_workloads(frame)

    chosen = pick_workloads(workloads)

    pool = daily.load_pool()
    index_of = dict(zip(pool["id"].astype(str), pool["node"].astype(int)))
    expected_of = dict(zip(pool["id"].astype(str), pool["expected"]))

    matrix = np.load(daily.MATRIX_PATH)
    road_nodes = [int(n) for n in matrix["road_nodes"]]

    data = {}

    for workload_id in chosen:
        workload_df = workloads[workload_id - 1]
        deliveries, delivery_nodes = base.build_workload(
            workload_df=workload_df, delivery_lookup=lookup
        )
        ids = [str(d.delivery_id) for d in deliveries]
        rows = [0, *(index_of[i] for i in ids)]

        index = np.ix_(rows, rows)
        distance = matrix["distance_km"][index]
        travel = matrix["travel_time_minutes"][index]

        node_ids = tuple(road_nodes[r] for r in rows)
        wave_end = workload_df["wave_end"].iloc[0]

        deadlines = [
            None
            if pd.isna(expected_of[i])
            else (expected_of[i] - pd.to_datetime(wave_end, utc=True)).total_seconds() / 60.0
            for i in ids
        ]

        data[workload_id] = {
            "deliveries": deliveries,
            "ids": ids,
            "road_matrix": RoadMatrixResult(
                node_ids=node_ids, distance_km=distance, travel_time_minutes=travel
            ),
            "start_time": wave_end,
            "deadlines": deadlines,
            "distance": distance.tolist(),
            "travel": travel.tolist(),
            "optimal_km": optimal_path(distance.tolist(), closed=True)[0],
            "optimal_min": optimal_path(travel.tolist(), closed=True)[0],
        }

    return data


def init_worker(data):
    import torch

    torch.set_num_threads(1)

    _DATA.update(data)


def lateness(order_ids, work):
    position = {i: k for k, i in enumerate(work["ids"])}
    clock, previous, late, minutes = 0.0, 0, 0, 0.0

    for i in order_ids:
        k = position[i]
        clock += work["travel"][previous][k + 1]
        previous = k + 1
        deadline = work["deadlines"][k]

        if deadline is not None and clock > deadline:
            late += 1
            minutes += clock - deadline

    return late, minutes


def run_one(task):
    workload_id, algorithm, seed = task
    work = _DATA[workload_id]

    environment = base.RoutingEnvironment(
        deliveries=work["deliveries"],
        road_matrix=work["road_matrix"],
        start_time=work["start_time"],
    )

    if algorithm == "q_learning":
        config = base.make_q_config()
        config = type(config)(**{**config.__dict__, "random_seed": seed})
        agent = base.QLearningAgent(config)
    elif algorithm == "sarsa":
        config = base.make_sarsa_config()
        config = type(config)(**{**config.__dict__, "random_seed": seed})
        agent = base.SarsaAgent(config)
    else:
        config = base.make_dqn_config()
        config = type(config)(
            **{**config.__dict__, "random_seed": seed, "device": "cpu"}
        )
        agent = base.DQNAgent(
            delivery_count=len(work["deliveries"]), config=config
        )

    tic = perf_counter()
    training = agent.train(environment)
    seconds = perf_counter() - tic

    rollout = agent.greedy_rollout(environment)
    order = list(rollout.delivery_order)

    position = {i: k for k, i in enumerate(work["ids"])}
    sequence = [position[i] for i in order]

    km = path_cost(work["distance"], sequence, closed=True)
    minutes = path_cost(work["travel"], sequence, closed=True)
    late, late_minutes = lateness(order, work)

    rewards = np.array([e.total_reward for e in training.episodes])
    blocks = rewards[: len(rewards) // CURVE_BLOCK * CURVE_BLOCK].reshape(-1, CURVE_BLOCK).mean(axis=1)

    row = {
        "workload_id": workload_id,
        "n": len(work["ids"]),
        "algorithm": algorithm,
        "seed": seed,
        "distance_km": km,
        "travel_minutes": minutes,
        "optimal_km": work["optimal_km"],
        "optimal_minutes": work["optimal_min"],
        "gap_km_pct": (km - work["optimal_km"]) / work["optimal_km"] * 100.0,
        "gap_minutes_pct": (minutes - work["optimal_min"]) / work["optimal_min"] * 100.0,
        "late": late,
        "late_minutes": late_minutes,
        "training_seconds": seconds,
        "greedy_route": " ".join(order),
    }

    curve = [
        {
            "workload_id": workload_id,
            "algorithm": algorithm,
            "seed": seed,
            "episodes_done": (b + 1) * CURVE_BLOCK,
            "mean_reward": float(v),
        }
        for b, v in enumerate(blocks)
    ]

    return row, curve


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    started = perf_counter()
    data = build_data()

    print(
        f"{len(data)} workloads {sorted(data)}; sizes "
        f"{sorted(len(v['ids']) for v in data.values())}; "
        f"{len(SEEDS)} seeds x {len(ALGORITHMS)} algorithms; {workers} workers",
        flush=True,
    )

    done = set()

    if RUNS_PATH.exists():
        old = pd.read_csv(RUNS_PATH)
        done = set(zip(old.workload_id, old.algorithm, old.seed))

    # Long DQN runs first so that the pool stays busy to the end.
    tasks = [
        (w, a, s)
        for a in ("dqn", "sarsa", "q_learning")
        for w in sorted(data, key=lambda k: -len(data[k]["ids"]))
        for s in SEEDS
        if (w, a, s) not in done
    ]

    print(f"{len(tasks)} runs to do ({len(done)} already saved)", flush=True)

    rows = [] if not RUNS_PATH.exists() else pd.read_csv(RUNS_PATH).to_dict("records")
    curves = [] if not CURVES_PATH.exists() else pd.read_csv(CURVES_PATH).to_dict("records")

    with Pool(workers, initializer=init_worker, initargs=(data,)) as pool:
        for index, (row, curve) in enumerate(
            pool.imap_unordered(run_one, tasks), start=1
        ):
            rows.append(row)
            curves += curve

            if index % 10 == 0 or index == len(tasks):
                pd.DataFrame(rows).to_csv(RUNS_PATH, index=False)
                pd.DataFrame(curves).to_csv(CURVES_PATH, index=False)

                print(
                    f"{index}/{len(tasks)} runs, {perf_counter() - started:.0f}s",
                    flush=True,
                )

    print(f"finished in {perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
