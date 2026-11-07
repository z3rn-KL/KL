import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

import experiments.daily_planning_experiment as daily  # noqa: E402
from rl.shift.instances import ShiftSampler, split_days  # noqa: E402
from rl.shift.objective import ShiftObjectiveConfig  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results"
KMEANS_PATH = RESULTS_DIR / "xedu_kmeans_clusters.csv"

# SHIFT_OBJECTIVE=wait adds the customer waiting cost (time from the start of
# the shift to each delivery) to the shift score and writes every result
# under a "_wait" name, next to the untouched original results. The cost is
# an assumed scenario value, like every money coefficient.
WAIT = os.environ.get("SHIFT_OBJECTIVE", "base") == "wait"
WAIT_COST_PER_MINUTE = 50.0
NAME = "_wait" if WAIT else ""
OBJECTIVE = ShiftObjectiveConfig(
    wait_cost_per_minute=WAIT_COST_PER_MINUTE if WAIT else 0.0
)


def load_sampler(**sampler_kwargs):
    """
    Real orders with their K-Means cluster, the global road matrix and a
    day-level train / validation / test split.

    Needs results/daily_planning_road_matrix.npz
    (python experiments/daily_planning_prepare_matrix.py).
    """

    pool = daily.load_pool()

    clusters = pd.read_csv(KMEANS_PATH)[["id", "cluster_id"]].astype(
        {"id": str}
    )

    pool = pool.merge(clusters, on="id", validate="one_to_one")

    matrix = np.load(daily.MATRIX_PATH)

    sampler_kwargs.setdefault("config", OBJECTIVE)

    sampler = ShiftSampler(
        pool,
        matrix["distance_km"],
        matrix["travel_time_minutes"],
        **sampler_kwargs,
    )

    return sampler, split_days(pool["local_day"])
