from __future__ import annotations

import ast
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

import experiments.reproducibility_validation as repro

from rl.proposed.objective import (
    ObjectiveConfig,
    ObjectiveContext,
    compute_route_objective,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

SPLIT_PATH = (
    RESULTS_DIR
    / "proposed_rl_workload_split.csv"
)

TEST_CACHE_DIR = (
    RESULTS_DIR
    / "proposed_rl_test_road_cache"
)

PROPOSED_DETAILS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_test_details.csv"
)

RECOVERY_PATH = (
    RESULTS_DIR
    / "proposed_rl_baseline_route_recovery.csv"
)

FINAL_DETAILS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_locked_objective_details.csv"
)

FINAL_SUMMARY_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_locked_objective_summary.csv"
)

FINAL_PAIRWISE_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_locked_objective_pairwise.csv"
)

FINDINGS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_locked_objective_findings.txt"
)


# ============================================================
# FROZEN EXPERIMENT DESIGN
# ============================================================

RANDOM_SEED = 42

EXPECTED_TEST_WORKLOADS = 16
EXPECTED_TEST_DELIVERIES = 131

ALGORITHMS = (
    "nearest_neighbor",
    "clarke_wright",
    "q_learning",
    "sarsa",
    "dqn",
)

DISPLAY_NAMES = {
    "nearest_neighbor": "NN",
    "clarke_wright": "CW",
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
    "dqn": "DQN",
}

FINAL_ALGORITHM_ORDER = (
    "NN",
    "CW",
    "Q-Learning",
    "SARSA",
    "DQN",
    "Proposed RL",
)


# ============================================================
# LOCKED PROPOSED OBJECTIVE
# ============================================================

LOCKED_OBJECTIVE = ObjectiveConfig(
    distance_weight=0.375,
    travel_time_weight=0.375,
    late_delivery_weight=0.10,
    lateness_weight=0.10,
    cluster_switch_weight=0.05,
)


# ============================================================
# VERIFICATION TOLERANCES
# ============================================================

# Reproducibility experiment trước đã đạt exact match
# ngoại trừ sai số floating-point rất nhỏ.
FLOAT_ATOL = 1e-6
RATE_ATOL = 1e-9


# ============================================================
# ROAD MATRIX COMPATIBILITY OBJECT
# ============================================================

@dataclass(frozen=True)
class CachedRoadMatrix:
    """
    Fallback object compatible with the fields used by
    the existing RL benchmark.

    Matrix convention:
        row/column 0 = depot
        row/column 1..N = deliveries
    """

    node_ids: tuple[int, ...]
    distance_km: tuple[tuple[float, ...], ...]
    travel_time_minutes: tuple[
        tuple[float, ...],
        ...
    ]


# ============================================================
# LOAD FROZEN TEST SPLIT
# ============================================================

def load_test_split() -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_PATH}"
        )

    split = pd.read_csv(
        SPLIT_PATH
    )

    required = {
        "workload_id",
        "number_of_deliveries",
        "split",
    }

    missing = (
        required
        - set(split.columns)
    )

    if missing:
        raise ValueError(
            "Split file missing columns: "
            f"{sorted(missing)}"
        )

    split["workload_id"] = (
        split["workload_id"]
        .astype(int)
    )

    split["number_of_deliveries"] = (
        split["number_of_deliveries"]
        .astype(int)
    )

    test = (
        split[
            split["split"].astype(str)
            == "test"
        ]
        .copy()
        .sort_values(
            [
                "number_of_deliveries",
                "workload_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(test)
        != EXPECTED_TEST_WORKLOADS
    ):
        raise ValueError(
            f"Expected {EXPECTED_TEST_WORKLOADS} "
            f"Test workloads, found {len(test)}."
        )

    total_deliveries = int(
        test[
            "number_of_deliveries"
        ].sum()
    )

    if (
        total_deliveries
        != EXPECTED_TEST_DELIVERIES
    ):
        raise ValueError(
            f"Expected {EXPECTED_TEST_DELIVERIES} "
            f"Test deliveries, "
            f"found {total_deliveries}."
        )

    return test


# ============================================================
# TEST CACHE
# ============================================================

def load_test_cache(
    workload_id: int,
) -> dict[str, object]:
    path = (
        TEST_CACHE_DIR
        / f"workload_{workload_id}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Test road cache: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        result = {
            "workload_id": int(
                data[
                    "workload_id"
                ][0]
            ),

            "number_of_deliveries": int(
                data[
                    "number_of_deliveries"
                ][0]
            ),

            "depot_node": int(
                data[
                    "depot_node"
                ][0]
            ),

            "delivery_nodes": tuple(
                int(value)
                for value in data[
                    "delivery_nodes"
                ].tolist()
            ),

            "delivery_ids": tuple(
                str(value)
                for value in data[
                    "delivery_ids"
                ].tolist()
            ),

            "cluster_ids": tuple(
                int(value)
                for value in data[
                    "cluster_ids"
                ].tolist()
            ),

            "deadlines_minutes": np.asarray(
                data[
                    "deadlines_minutes"
                ],
                dtype=float,
            ),

            "deadline_valid": np.asarray(
                data[
                    "deadline_valid"
                ],
                dtype=bool,
            ),

            "distance_km": np.asarray(
                data[
                    "distance_km"
                ],
                dtype=float,
            ),

            "travel_time_minutes": np.asarray(
                data[
                    "travel_time_minutes"
                ],
                dtype=float,
            ),
        }

    return result


# ============================================================
# ROAD MATRIX FOR EXISTING BENCHMARK
# ============================================================

def build_road_matrix_for_order(
    *,
    cache: dict[str, object],
    delivery_ids: Sequence[str],
    delivery_nodes: Sequence[int],
):
    """
    Reorder the cached road matrix so that its
    delivery ordering exactly matches prepare_workload().

    Important:
    RoutingEnvironment expects distance_km and
    travel_time_minutes to be NumPy arrays because
    it validates their .shape attributes.
    """

    from routing.road_matrix import (
        RoadMatrixResult,
    )

    cached_ids = tuple(
        str(value)
        for value in cache[
            "delivery_ids"
        ]
    )

    desired_ids = tuple(
        str(value)
        for value in delivery_ids
    )

    if set(
        cached_ids
    ) != set(
        desired_ids
    ):
        raise ValueError(
            "Cached delivery IDs do not match "
            "prepare_workload() delivery IDs."
        )

    cached_index = {
        delivery_id: index
        for index, delivery_id
        in enumerate(
            cached_ids
        )
    }

    # Road matrix convention:
    #
    # index 0     = depot
    # index 1..N  = deliveries
    permutation = [
        0,
        *[
            cached_index[
                delivery_id
            ]
            + 1
            for delivery_id
            in desired_ids
        ],
    ]

    cached_distance = np.asarray(
        cache[
            "distance_km"
        ],
        dtype=np.float64,
    )

    cached_travel_time = np.asarray(
        cache[
            "travel_time_minutes"
        ],
        dtype=np.float64,
    )

    reordered_distance = (
        cached_distance[
            np.ix_(
                permutation,
                permutation,
            )
        ]
        .copy()
    )

    reordered_travel_time = (
        cached_travel_time[
            np.ix_(
                permutation,
                permutation,
            )
        ]
        .copy()
    )

    expected_size = (
        len(
            desired_ids
        )
        + 1
    )

    expected_shape = (
        expected_size,
        expected_size,
    )

    if (
        reordered_distance.shape
        != expected_shape
    ):
        raise ValueError(
            "Unexpected reordered distance "
            f"matrix shape: "
            f"{reordered_distance.shape}; "
            f"expected {expected_shape}."
        )

    if (
        reordered_travel_time.shape
        != expected_shape
    ):
        raise ValueError(
            "Unexpected reordered travel-time "
            f"matrix shape: "
            f"{reordered_travel_time.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(
        reordered_distance
    ).all():
        raise ValueError(
            "Reordered distance matrix contains "
            "non-finite values."
        )

    if not np.isfinite(
        reordered_travel_time
    ).all():
        raise ValueError(
            "Reordered travel-time matrix contains "
            "non-finite values."
        )

    depot_node = int(
        cache[
            "depot_node"
        ]
    )

    node_ids = tuple(
        [
            depot_node,
            *[
                int(value)
                for value in delivery_nodes
            ],
        ]
    )

    if len(
        node_ids
    ) != expected_size:
        raise ValueError(
            "node_ids length does not match "
            "road matrix size."
        )

    return RoadMatrixResult(
        node_ids=(
            node_ids
        ),

        distance_km=(
            reordered_distance
        ),

        travel_time_minutes=(
            reordered_travel_time
        ),
    )


# ============================================================
# REORDER CACHE MATRICES FOR ROUTE EVALUATION
# ============================================================

def reordered_cache_arrays(
    *,
    cache: dict[str, object],
    desired_delivery_ids: Sequence[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
]:
    cached_ids = tuple(
        str(value)
        for value in cache[
            "delivery_ids"
        ]
    )

    desired_ids = tuple(
        str(value)
        for value in desired_delivery_ids
    )

    cached_index = {
        delivery_id: index
        for index, delivery_id
        in enumerate(
            cached_ids
        )
    }

    permutation = [
        0,
        *[
            cached_index[
                delivery_id
            ]
            + 1
            for delivery_id
            in desired_ids
        ],
    ]

    delivery_permutation = [
        cached_index[
            delivery_id
        ]
        for delivery_id
        in desired_ids
    ]

    distance = np.asarray(
        cache[
            "distance_km"
        ],
        dtype=float,
    )[
        np.ix_(
            permutation,
            permutation,
        )
    ]

    travel_time = np.asarray(
        cache[
            "travel_time_minutes"
        ],
        dtype=float,
    )[
        np.ix_(
            permutation,
            permutation,
        )
    ]

    deadlines = np.asarray(
        cache[
            "deadlines_minutes"
        ],
        dtype=float,
    )[
        delivery_permutation
    ]

    deadline_valid = np.asarray(
        cache[
            "deadline_valid"
        ],
        dtype=bool,
    )[
        delivery_permutation
    ]

    original_clusters = tuple(
        int(value)
        for value in cache[
            "cluster_ids"
        ]
    )

    clusters = tuple(
        original_clusters[
            index
        ]
        for index in delivery_permutation
    )

    return (
        distance,
        travel_time,
        deadlines,
        deadline_valid,
        clusters,
    )


# ============================================================
# EVALUATE RECOVERED ROUTE
# ============================================================

def evaluate_delivery_order(
    *,
    delivery_order: Sequence[str],
    canonical_delivery_ids: Sequence[str],
    cache: dict[str, object],
) -> dict[str, object]:
    """
    Evaluate a recovered baseline route using the same
    cached road matrices used in final Test.

    Service duration is not added, matching the
    existing benchmark convention.
    """

    canonical_ids = tuple(
        str(value)
        for value
        in canonical_delivery_ids
    )

    route_ids = tuple(
        str(value)
        for value
        in delivery_order
    )

    n = len(
        canonical_ids
    )

    if len(
        route_ids
    ) != n:
        raise ValueError(
            "Recovered route length mismatch."
        )

    if set(
        route_ids
    ) != set(
        canonical_ids
    ):
        raise ValueError(
            "Recovered route delivery IDs "
            "do not match workload."
        )

    (
        distance,
        travel_time,
        deadlines,
        deadline_valid,
        clusters,
    ) = reordered_cache_arrays(
        cache=cache,
        desired_delivery_ids=(
            canonical_ids
        ),
    )

    delivery_index = {
        delivery_id: index
        for index, delivery_id
        in enumerate(
            canonical_ids
        )
    }

    actions = tuple(
        delivery_index[
            delivery_id
        ]
        for delivery_id
        in route_ids
    )

    if set(
        actions
    ) != set(
        range(
            n
        )
    ):
        raise ValueError(
            "Recovered route is not a "
            "complete permutation."
        )

    total_distance = 0.0
    total_travel_time = 0.0

    elapsed = 0.0

    late_deliveries = 0
    total_lateness = 0.0
    max_lateness = 0.0

    current_matrix_index = 0

    cluster_switches = 0
    previous_cluster = None

    for action in actions:
        next_matrix_index = (
            action + 1
        )

        leg_distance = float(
            distance[
                current_matrix_index,
                next_matrix_index,
            ]
        )

        leg_time = float(
            travel_time[
                current_matrix_index,
                next_matrix_index,
            ]
        )

        total_distance += (
            leg_distance
        )

        total_travel_time += (
            leg_time
        )

        elapsed += (
            leg_time
        )

        if bool(
            deadline_valid[
                action
            ]
        ):
            deadline = float(
                deadlines[
                    action
                ]
            )

            lateness = max(
                0.0,
                elapsed - deadline,
            )

            if lateness > 0.0:
                late_deliveries += 1

            total_lateness += (
                lateness
            )

            max_lateness = max(
                max_lateness,
                lateness,
            )

        cluster = int(
            clusters[
                action
            ]
        )

        # Depot -> first delivery does not count.
        if (
            previous_cluster
            is not None
            and cluster
            != previous_cluster
        ):
            cluster_switches += 1

        previous_cluster = (
            cluster
        )

        current_matrix_index = (
            next_matrix_index
        )

    # Closed route: final delivery -> depot.
    total_distance += float(
        distance[
            current_matrix_index,
            0,
        ]
    )

    return_time = float(
        travel_time[
            current_matrix_index,
            0,
        ]
    )

    total_travel_time += (
        return_time
    )

    elapsed += (
        return_time
    )

    valid_deadlines = int(
        deadline_valid.sum()
    )

    on_time_deliveries = (
        valid_deadlines
        - late_deliveries
    )

    if valid_deadlines == 0:
        on_time_rate = 1.0
    else:
        on_time_rate = (
            on_time_deliveries
            / valid_deadlines
        )

    return {
        "number_of_deliveries": (
            n
        ),

        "total_distance_km": (
            total_distance
        ),

        "total_travel_time_minutes": (
            total_travel_time
        ),

        "valid_deadlines": (
            valid_deadlines
        ),

        "on_time_deliveries": (
            on_time_deliveries
        ),

        "late_deliveries": (
            late_deliveries
        ),

        "on_time_rate": (
            on_time_rate
        ),

        "total_lateness_minutes": (
            total_lateness
        ),

        "max_lateness_minutes": (
            max_lateness
        ),

        "cluster_switches": (
            cluster_switches
        ),

        "cluster_switch_rate": (
            cluster_switches
            / max(
                1,
                n - 1,
            )
        ),

        "route_actions": (
            actions
        ),
    }


# ============================================================
# CANONICAL VERIFICATION
# ============================================================

def close_enough(
    actual: float,
    expected: float,
    *,
    tolerance: float = FLOAT_ATOL,
) -> bool:
    return (
        abs(
            float(actual)
            - float(expected)
        )
        <= tolerance
    )


def verify_against_reference(
    *,
    evaluation: dict[str, object],
    reference: pd.Series,
) -> dict[str, object]:
    distance_match = close_enough(
        evaluation[
            "total_distance_km"
        ],
        reference[
            "total_distance_km"
        ],
    )

    time_match = close_enough(
        evaluation[
            "total_travel_time_minutes"
        ],
        reference[
            "total_travel_time_minutes"
        ],
    )

    late_match = (
        int(
            evaluation[
                "late_deliveries"
            ]
        )
        == int(
            reference[
                "late_deliveries"
            ]
        )
    )

    on_time_rate_match = close_enough(
        evaluation[
            "on_time_rate"
        ],
        reference[
            "on_time_rate"
        ],
        tolerance=RATE_ATOL,
    )

    lateness_match = close_enough(
        evaluation[
            "total_lateness_minutes"
        ],
        reference[
            "total_lateness_minutes"
        ],
    )

    verified = all(
        (
            distance_match,
            time_match,
            late_match,
            on_time_rate_match,
            lateness_match,
        )
    )

    return {
        "distance_match": (
            distance_match
        ),

        "travel_time_match": (
            time_match
        ),

        "late_deliveries_match": (
            late_match
        ),

        "on_time_rate_match": (
            on_time_rate_match
        ),

        "lateness_match": (
            lateness_match
        ),

        "verified_exact": (
            verified
        ),
    }


# ============================================================
# LOCKED OBJECTIVE
# ============================================================

def calculate_locked_objective(
    *,
    number_of_deliveries: int,
    valid_deadlines: int,
    total_distance_km: float,
    total_travel_time_minutes: float,
    late_deliveries: int,
    total_lateness_minutes: float,
    cluster_switches: int,
) -> float:
    context = ObjectiveContext(
        number_of_deliveries=(
            number_of_deliveries
        ),

        number_of_valid_deadlines=(
            valid_deadlines
        ),
    )

    breakdown = (
        compute_route_objective(
            total_distance_km=(
                total_distance_km
            ),

            total_travel_time_minutes=(
                total_travel_time_minutes
            ),

            late_deliveries=(
                late_deliveries
            ),

            total_lateness_minutes=(
                total_lateness_minutes
            ),

            cluster_switches=(
                cluster_switches
            ),

            context=(
                context
            ),

            config=(
                LOCKED_OBJECTIVE
            ),
        )
    )

    return float(
        breakdown.total_cost
    )


# ============================================================
# EXISTING RECOVERY / RESUME
# ============================================================

def load_existing_recovery() -> pd.DataFrame:
    if not RECOVERY_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        RECOVERY_PATH
    )

    if dataframe.empty:
        return dataframe

    dataframe[
        "workload_id"
    ] = dataframe[
        "workload_id"
    ].astype(int)

    return dataframe


def verified_keys(
    dataframe: pd.DataFrame,
) -> set[tuple[int, str]]:
    if dataframe.empty:
        return set()

    if (
        "verified_exact"
        not in dataframe.columns
    ):
        return set()

    verified = dataframe[
        dataframe[
            "verified_exact"
        ].astype(bool)
    ]

    return {
        (
            int(row.workload_id),
            str(row.algorithm),
        )
        for row in verified.itertuples(
            index=False
        )
    }


def autosave(
    rows: list[dict[str, object]],
) -> None:
    pd.DataFrame(
        rows
    ).to_csv(
        RECOVERY_PATH,
        index=False,
    )


# ============================================================
# RECOVER FIVE BASELINE ROUTES
# ============================================================

def recover_baseline_routes(
    *,
    test_split: pd.DataFrame,
) -> pd.DataFrame:
    print()
    print("=" * 90)
    print(
        "BASELINE ROUTE RECOVERY"
    )
    print("=" * 90)

    print()
    print(
        "Algorithms:"
    )

    for algorithm in ALGORITHMS:
        print(
            f"  {algorithm}"
        )

    canonical = (
        repro.load_canonical_results()
    )

    manifest = (
        repro.load_manifest()
    )

    existing = (
        load_existing_recovery()
    )

    rows: list[
        dict[str, object]
    ]

    if existing.empty:
        rows = []
    else:
        rows = (
            existing
            .to_dict(
                "records"
            )
        )

    completed = (
        verified_keys(
            existing
        )
    )

    total_required = (
        EXPECTED_TEST_WORKLOADS
        * len(
            ALGORITHMS
        )
    )

    print()
    print(
        f"Required recoveries: "
        f"{total_required}"
    )

    print(
        f"Already verified: "
        f"{len(completed)}"
    )

    # Run cheap methods first.
    # DQN therefore runs last.
    run_number = (
        len(completed)
    )

    for algorithm in ALGORITHMS:
        print()
        print(
            "-" * 90
        )

        print(
            f"ALGORITHM: "
            f"{algorithm}"
        )

        print(
            "-" * 90
        )

        for split_row in test_split.itertuples(
            index=False
        ):
            workload_id = int(
                split_row.workload_id
            )

            key = (
                workload_id,
                algorithm,
            )

            if key in completed:
                print(
                    f"workload={workload_id:3d} | "
                    f"{algorithm:16s} | "
                    "SKIP VERIFIED"
                )

                continue

            run_number += 1

            print()
            print(
                f"[{run_number:2d}/{total_required}] "
                f"workload={workload_id:3d} | "
                f"algorithm={algorithm}"
            )

            # ------------------------------------------------
            # Canonical workload
            # ------------------------------------------------

            (
                delivery_ids,
                deliveries,
                delivery_nodes,
                wave_start,
                wave_end,
            ) = repro.prepare_workload(
                workload_id,
                manifest,
            )

            n = len(
                delivery_ids
            )

            if (
                n
                != int(
                    split_row
                    .number_of_deliveries
                )
            ):
                raise ValueError(
                    f"Workload {workload_id}: "
                    "size mismatch."
                )

            # ------------------------------------------------
            # Frozen Test road cache
            # ------------------------------------------------

            cache = (
                load_test_cache(
                    workload_id
                )
            )

            depot_node = int(
                cache[
                    "depot_node"
                ]
            )

            road_matrix = (
                build_road_matrix_for_order(
                    cache=cache,

                    delivery_ids=(
                        delivery_ids
                    ),

                    delivery_nodes=(
                        delivery_nodes
                    ),
                )
            )

            # ------------------------------------------------
            # Same seed as canonical benchmark
            # ------------------------------------------------

            repro.reset_global_seeds(
                RANDOM_SEED
            )

            # Existing benchmark convention:
            # routing begins at wave_end.
            start_time = (
                wave_end
            )

            # ------------------------------------------------
            # Recover route using the original implementation
            # ------------------------------------------------

            (
                delivery_order,
                runtime_seconds,
            ) = repro.run_algorithm(
                algorithm,

                depot_node=(
                    depot_node
                ),

                delivery_ids=(
                    delivery_ids
                ),

                delivery_nodes=(
                    delivery_nodes
                ),

                deliveries=(
                    deliveries
                ),

                road_matrix=(
                    road_matrix
                ),

                start_time=(
                    start_time
                ),
            )

            # ------------------------------------------------
            # Independently re-evaluate recovered route
            # ------------------------------------------------

            evaluation = (
                evaluate_delivery_order(
                    delivery_order=(
                        delivery_order
                    ),

                    canonical_delivery_ids=(
                        delivery_ids
                    ),

                    cache=(
                        cache
                    ),
                )
            )

            # ------------------------------------------------
            # Original canonical benchmark result
            # ------------------------------------------------

            reference = (
                repro.get_reference_row(
                    canonical,
                    workload_id,
                    algorithm,
                )
            )

            verification = (
                verify_against_reference(
                    evaluation=(
                        evaluation
                    ),

                    reference=(
                        reference
                    ),
                )
            )

            locked_objective = (
                calculate_locked_objective(
                    number_of_deliveries=(
                        n
                    ),

                    valid_deadlines=int(
                        evaluation[
                            "valid_deadlines"
                        ]
                    ),

                    total_distance_km=float(
                        evaluation[
                            "total_distance_km"
                        ]
                    ),

                    total_travel_time_minutes=float(
                        evaluation[
                            "total_travel_time_minutes"
                        ]
                    ),

                    late_deliveries=int(
                        evaluation[
                            "late_deliveries"
                        ]
                    ),

                    total_lateness_minutes=float(
                        evaluation[
                            "total_lateness_minutes"
                        ]
                    ),

                    cluster_switches=int(
                        evaluation[
                            "cluster_switches"
                        ]
                    ),
                )
            )

            result_row = {
                "workload_id": (
                    workload_id
                ),

                "algorithm": (
                    algorithm
                ),

                "algorithm_display": (
                    DISPLAY_NAMES[
                        algorithm
                    ]
                ),

                "number_of_deliveries": (
                    n
                ),

                "rerun_delivery_order": str(
                    list(
                        delivery_order
                    )
                ),

                "route_actions": str(
                    evaluation[
                        "route_actions"
                    ]
                ),

                "rerun_runtime_seconds": float(
                    runtime_seconds
                ),

                "total_distance_km": float(
                    evaluation[
                        "total_distance_km"
                    ]
                ),

                "reference_distance_km": float(
                    reference[
                        "total_distance_km"
                    ]
                ),

                "total_travel_time_minutes": float(
                    evaluation[
                        "total_travel_time_minutes"
                    ]
                ),

                "reference_travel_time_minutes": float(
                    reference[
                        "total_travel_time_minutes"
                    ]
                ),

                "valid_deadlines": int(
                    evaluation[
                        "valid_deadlines"
                    ]
                ),

                "on_time_deliveries": int(
                    evaluation[
                        "on_time_deliveries"
                    ]
                ),

                "late_deliveries": int(
                    evaluation[
                        "late_deliveries"
                    ]
                ),

                "reference_late_deliveries": int(
                    reference[
                        "late_deliveries"
                    ]
                ),

                "on_time_rate": float(
                    evaluation[
                        "on_time_rate"
                    ]
                ),

                "reference_on_time_rate": float(
                    reference[
                        "on_time_rate"
                    ]
                ),

                "total_lateness_minutes": float(
                    evaluation[
                        "total_lateness_minutes"
                    ]
                ),

                "reference_total_lateness_minutes": float(
                    reference[
                        "total_lateness_minutes"
                    ]
                ),

                "max_lateness_minutes": float(
                    evaluation[
                        "max_lateness_minutes"
                    ]
                ),

                "cluster_switches": int(
                    evaluation[
                        "cluster_switches"
                    ]
                ),

                "cluster_switch_rate": float(
                    evaluation[
                        "cluster_switch_rate"
                    ]
                ),

                "locked_objective": (
                    locked_objective
                ),

                **verification,
            }

            # Remove an old failed row for the same key
            # before writing the new attempt.
            rows = [
                row
                for row in rows
                if not (
                    int(
                        row[
                            "workload_id"
                        ]
                    )
                    == workload_id
                    and str(
                        row[
                            "algorithm"
                        ]
                    )
                    == algorithm
                )
            ]

            rows.append(
                result_row
            )

            autosave(
                rows
            )

            status = (
                "VERIFIED"
                if verification[
                    "verified_exact"
                ]
                else "MISMATCH"
            )

            print(
                f"  distance = "
                f"{evaluation['total_distance_km']:.6f} km"
            )

            print(
                f"  time     = "
                f"{evaluation['total_travel_time_minutes']:.6f} min"
            )

            print(
                f"  late     = "
                f"{evaluation['late_deliveries']}"
            )

            print(
                f"  switches = "
                f"{evaluation['cluster_switches']}"
            )

            print(
                f"  J_locked = "
                f"{locked_objective:.6f}"
            )

            print(
                f"  status   = "
                f"{status}"
            )

            # ------------------------------------------------
            # We refuse to use a recovered route if it does
            # not reproduce the canonical benchmark.
            # ------------------------------------------------

            if not verification[
                "verified_exact"
            ]:
                print()
                print(
                    "Recovery mismatch detected."
                )

                print(
                    "The route was saved for diagnosis "
                    "but WILL NOT be accepted."
                )

                raise RuntimeError(
                    f"Baseline recovery mismatch: "
                    f"workload={workload_id}, "
                    f"algorithm={algorithm}. "
                    f"Inspect {RECOVERY_PATH}"
                )

            completed.add(
                key
            )

            # Existing reproducibility cleanup.
            try:
                repro.cleanup_memory()
            except Exception:
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    final = pd.DataFrame(
        rows
    )

    verified = final[
        final[
            "verified_exact"
        ].astype(bool)
    ]

    if len(
        verified
    ) != total_required:
        raise RuntimeError(
            "Route recovery did not produce "
            f"{total_required} verified rows. "
            f"Found {len(verified)}."
        )

    return (
        verified
        .sort_values(
            [
                "algorithm",
                "workload_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PROPOSED RL FINAL TEST
# ============================================================

def parse_action_tuple(
    value: object,
) -> tuple[int, ...]:
    if isinstance(
        value,
        tuple,
    ):
        return tuple(
            int(item)
            for item in value
        )

    if isinstance(
        value,
        list,
    ):
        return tuple(
            int(item)
            for item in value
        )

    parsed = ast.literal_eval(
        str(
            value
        )
    )

    if not isinstance(
        parsed,
        (
            tuple,
            list,
        ),
    ):
        raise ValueError(
            "route_actions must be a list/tuple."
        )

    return tuple(
        int(item)
        for item in parsed
    )


def load_proposed_rows(
    test_split: pd.DataFrame,
) -> pd.DataFrame:
    if not PROPOSED_DETAILS_PATH.exists():
        raise FileNotFoundError(
            "Missing Proposed RL final Test details: "
            f"{PROPOSED_DETAILS_PATH}"
        )

    dataframe = pd.read_csv(
        PROPOSED_DETAILS_PATH
    )

    required = {
        "workload_id",
        "number_of_deliveries",
        "distance_km",
        "travel_time_minutes",
        "late_deliveries",
        "valid_deadlines",
        "on_time_deliveries",
        "on_time_rate",
        "lateness_minutes",
        "cluster_switches",
        "route_actions",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "Proposed RL final Test details "
            "missing columns: "
            f"{sorted(missing)}"
        )

    expected_ids = set(
        test_split[
            "workload_id"
        ].astype(int)
    )

    actual_ids = set(
        dataframe[
            "workload_id"
        ].astype(int)
    )

    if actual_ids != expected_ids:
        raise ValueError(
            "Proposed RL Test workload IDs "
            "do not match frozen Test split."
        )

    rows: list[
        dict[str, object]
    ] = []

    for source in dataframe.itertuples(
        index=False
    ):
        workload_id = int(
            source.workload_id
        )

        n = int(
            source.number_of_deliveries
        )

        actions = (
            parse_action_tuple(
                source.route_actions
            )
        )

        if (
            len(actions) != n
            or set(actions)
            != set(
                range(
                    n
                )
            )
        ):
            raise ValueError(
                f"Invalid Proposed RL route "
                f"for workload {workload_id}."
            )

        cache = (
            load_test_cache(
                workload_id
            )
        )

        cluster_ids = tuple(
            int(value)
            for value in cache[
                "cluster_ids"
            ]
        )

        cluster_switches = sum(
            1
            for previous, current
            in zip(
                actions[:-1],
                actions[1:],
            )
            if (
                cluster_ids[
                    previous
                ]
                != cluster_ids[
                    current
                ]
            )
        )

        stored_switches = int(
            source.cluster_switches
        )

        if (
            cluster_switches
            != stored_switches
        ):
            raise ValueError(
                f"Proposed RL cluster-switch "
                f"verification failed for "
                f"workload {workload_id}: "
                f"stored={stored_switches}, "
                f"recomputed={cluster_switches}"
            )

        locked_objective = (
            calculate_locked_objective(
                number_of_deliveries=(
                    n
                ),

                valid_deadlines=int(
                    source.valid_deadlines
                ),

                total_distance_km=float(
                    source.distance_km
                ),

                total_travel_time_minutes=float(
                    source.travel_time_minutes
                ),

                late_deliveries=int(
                    source.late_deliveries
                ),

                total_lateness_minutes=float(
                    source.lateness_minutes
                ),

                cluster_switches=(
                    cluster_switches
                ),
            )
        )

        rows.append(
            {
                "workload_id": (
                    workload_id
                ),

                "algorithm": (
                    "Proposed RL"
                ),

                "number_of_deliveries": (
                    n
                ),

                "total_distance_km": float(
                    source.distance_km
                ),

                "total_travel_time_minutes": float(
                    source.travel_time_minutes
                ),

                "valid_deadlines": int(
                    source.valid_deadlines
                ),

                "on_time_deliveries": int(
                    source.on_time_deliveries
                ),

                "late_deliveries": int(
                    source.late_deliveries
                ),

                "on_time_rate": float(
                    source.on_time_rate
                ),

                "total_lateness_minutes": float(
                    source.lateness_minutes
                ),

                "cluster_switches": (
                    cluster_switches
                ),

                "cluster_switch_rate": (
                    cluster_switches
                    / max(
                        1,
                        n - 1,
                    )
                ),

                "locked_objective": (
                    locked_objective
                ),

                "route_actions": str(
                    actions
                ),
            }
        )

    proposed = pd.DataFrame(
        rows
    )

    if (
        len(proposed)
        != EXPECTED_TEST_WORKLOADS
    ):
        raise ValueError(
            "Expected 16 Proposed RL Test rows."
        )

    return proposed


# ============================================================
# NORMALIZE RECOVERED BASELINES
# ============================================================

def baseline_final_rows(
    recovery: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "workload_id": (
                recovery[
                    "workload_id"
                ].astype(int)
            ),

            "algorithm": (
                recovery[
                    "algorithm_display"
                ].astype(str)
            ),

            "number_of_deliveries": (
                recovery[
                    "number_of_deliveries"
                ].astype(int)
            ),

            "total_distance_km": (
                recovery[
                    "total_distance_km"
                ].astype(float)
            ),

            "total_travel_time_minutes": (
                recovery[
                    "total_travel_time_minutes"
                ].astype(float)
            ),

            "valid_deadlines": (
                recovery[
                    "valid_deadlines"
                ].astype(int)
            ),

            "on_time_deliveries": (
                recovery[
                    "on_time_deliveries"
                ].astype(int)
            ),

            "late_deliveries": (
                recovery[
                    "late_deliveries"
                ].astype(int)
            ),

            "on_time_rate": (
                recovery[
                    "on_time_rate"
                ].astype(float)
            ),

            "total_lateness_minutes": (
                recovery[
                    "total_lateness_minutes"
                ].astype(float)
            ),

            "cluster_switches": (
                recovery[
                    "cluster_switches"
                ].astype(int)
            ),

            "cluster_switch_rate": (
                recovery[
                    "cluster_switch_rate"
                ].astype(float)
            ),

            "locked_objective": (
                recovery[
                    "locked_objective"
                ].astype(float)
            ),

            "route_actions": (
                recovery[
                    "route_actions"
                ].astype(str)
            ),
        }
    )


# ============================================================
# SUMMARY
# ============================================================

def build_final_summary(
    details: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for algorithm in (
        FINAL_ALGORITHM_ORDER
    ):
        group = details[
            details[
                "algorithm"
            ]
            == algorithm
        ]

        if (
            len(group)
            != EXPECTED_TEST_WORKLOADS
        ):
            raise ValueError(
                f"{algorithm}: expected "
                f"{EXPECTED_TEST_WORKLOADS} rows, "
                f"found {len(group)}."
            )

        total_valid = int(
            group[
                "valid_deadlines"
            ].sum()
        )

        total_late = int(
            group[
                "late_deliveries"
            ].sum()
        )

        weighted_sla = (
            1.0
            if total_valid == 0
            else (
                (
                    total_valid
                    - total_late
                )
                / total_valid
            )
        )

        rows.append(
            {
                "algorithm": (
                    algorithm
                ),

                "workloads": (
                    len(group)
                ),

                "deliveries": int(
                    group[
                        "number_of_deliveries"
                    ].sum()
                ),

                "mean_distance_km": float(
                    group[
                        "total_distance_km"
                    ].mean()
                ),

                "mean_travel_time_minutes": float(
                    group[
                        "total_travel_time_minutes"
                    ].mean()
                ),

                "late_deliveries": (
                    total_late
                ),

                "weighted_sla": (
                    weighted_sla
                ),

                "total_lateness_minutes": float(
                    group[
                        "total_lateness_minutes"
                    ].sum()
                ),

                "mean_cluster_switches": float(
                    group[
                        "cluster_switches"
                    ].mean()
                ),

                "mean_cluster_switch_rate": float(
                    group[
                        "cluster_switch_rate"
                    ].mean()
                ),

                "mean_locked_objective": float(
                    group[
                        "locked_objective"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROPOSED VS BASELINES
# ============================================================

def build_pairwise(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    proposed = (
        summary[
            summary[
                "algorithm"
            ]
            == "Proposed RL"
        ]
        .iloc[0]
    )

    rows: list[
        dict[str, object]
    ] = []

    for baseline_name in (
        "NN",
        "CW",
        "Q-Learning",
        "SARSA",
        "DQN",
    ):
        baseline = (
            summary[
                summary[
                    "algorithm"
                ]
                == baseline_name
            ]
            .iloc[0]
        )

        baseline_j = float(
            baseline[
                "mean_locked_objective"
            ]
        )

        proposed_j = float(
            proposed[
                "mean_locked_objective"
            ]
        )

        if baseline_j > 0.0:
            objective_improvement = (
                (
                    baseline_j
                    - proposed_j
                )
                / baseline_j
                * 100.0
            )
        else:
            objective_improvement = (
                math.nan
            )

        baseline_switches = float(
            baseline[
                "mean_cluster_switches"
            ]
        )

        proposed_switches = float(
            proposed[
                "mean_cluster_switches"
            ]
        )

        if baseline_switches > 0.0:
            switch_reduction = (
                (
                    baseline_switches
                    - proposed_switches
                )
                / baseline_switches
                * 100.0
            )
        else:
            switch_reduction = (
                math.nan
            )

        rows.append(
            {
                "baseline_algorithm": (
                    baseline_name
                ),

                "baseline_mean_locked_objective": (
                    baseline_j
                ),

                "proposed_mean_locked_objective": (
                    proposed_j
                ),

                # Positive means Proposed has lower J.
                "locked_objective_improvement_pct": (
                    objective_improvement
                ),

                "baseline_mean_cluster_switches": (
                    baseline_switches
                ),

                "proposed_mean_cluster_switches": (
                    proposed_switches
                ),

                # Positive means Proposed has fewer switches.
                "cluster_switch_reduction_pct": (
                    switch_reduction
                ),

                "sla_change_percentage_points": (
                    (
                        float(
                            proposed[
                                "weighted_sla"
                            ]
                        )
                        - float(
                            baseline[
                                "weighted_sla"
                            ]
                        )
                    )
                    * 100.0
                ),

                "late_delivery_change": (
                    int(
                        proposed[
                            "late_deliveries"
                        ]
                    )
                    - int(
                        baseline[
                            "late_deliveries"
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FINDINGS
# ============================================================

def write_findings(
    *,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    lines = [
        (
            "PROPOSED RL FINAL LOCKED "
            "OBJECTIVE COMPARISON"
        ),

        "",

        (
            "Held-out Test:"
        ),

        (
            "workloads = 16"
        ),

        (
            "deliveries = 131"
        ),

        "",

        (
            "Locked objective:"
        ),

        (
            "J = 0.375*D_bar + "
            "0.375*T_bar + "
            "0.10*M_bar + "
            "0.10*L_bar + "
            "0.05*S_bar"
        ),

        "",

        (
            "Baseline delivery orders were recovered "
            "using the original benchmark "
            "implementations with seed 42."
        ),

        (
            "Recovered routes were accepted only "
            "when distance, travel time, SLA/late "
            "deliveries and lateness exactly matched "
            "the canonical benchmark within numerical "
            "floating-point tolerance."
        ),

        "",

        (
            "No objective tuning, model selection "
            "or Test-set training was performed."
        ),

        "",

        "SUMMARY",
        "",
    ]

    for row in summary.itertuples(
        index=False
    ):
        lines.extend(
            [
                (
                    f"[{row.algorithm}]"
                ),

                (
                    "mean_distance_km = "
                    f"{row.mean_distance_km:.9f}"
                ),

                (
                    "mean_travel_time_minutes = "
                    f"{row.mean_travel_time_minutes:.9f}"
                ),

                (
                    f"late_deliveries = "
                    f"{row.late_deliveries}"
                ),

                (
                    f"weighted_sla = "
                    f"{row.weighted_sla:.9f}"
                ),

                (
                    "total_lateness_minutes = "
                    f"{row.total_lateness_minutes:.9f}"
                ),

                (
                    "mean_cluster_switches = "
                    f"{row.mean_cluster_switches:.9f}"
                ),

                (
                    "mean_cluster_switch_rate = "
                    f"{row.mean_cluster_switch_rate:.9f}"
                ),

                (
                    "mean_locked_objective = "
                    f"{row.mean_locked_objective:.9f}"
                ),

                "",
            ]
        )

    lines.extend(
        [
            (
                "PROPOSED RL VS BASELINES"
            ),

            "",
        ]
    )

    for row in pairwise.itertuples(
        index=False
    ):
        lines.extend(
            [
                (
                    f"[vs {row.baseline_algorithm}]"
                ),

                (
                    "locked_objective_improvement_pct = "
                    f"{row.locked_objective_improvement_pct:.6f}"
                ),

                (
                    "cluster_switch_reduction_pct = "
                    f"{row.cluster_switch_reduction_pct:.6f}"
                ),

                (
                    "sla_change_percentage_points = "
                    f"{row.sla_change_percentage_points:.6f}"
                ),

                (
                    "late_delivery_change = "
                    f"{row.late_delivery_change}"
                ),

                "",
            ]
        )

    FINDINGS_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)

    print(
        "PROPOSED RL - FINAL ROUTE RECOVERY "
        "AND LOCKED OBJECTIVE COMPARISON"
    )

    print("=" * 90)

    print()
    print(
        "This experiment performs:"
    )

    print(
        "  baseline route recovery"
    )

    print(
        "  canonical-result verification"
    )

    print(
        "  cluster-switch evaluation"
    )

    print(
        "  same-objective evaluation"
    )

    print()
    print(
        "It performs NO Proposed RL training "
        "or Test tuning."
    )

    # --------------------------------------------------------
    # Frozen Test split
    # --------------------------------------------------------

    test_split = (
        load_test_split()
    )

    print()
    print(
        "Frozen Test:"
    )

    print(
        f"  workloads  = "
        f"{len(test_split)}"
    )

    print(
        "  deliveries = "
        f"{test_split['number_of_deliveries'].sum()}"
    )

    # --------------------------------------------------------
    # Recover all 80 baseline routes
    # --------------------------------------------------------

    recovery = (
        recover_baseline_routes(
            test_split=(
                test_split
            )
        )
    )

    print()
    print("=" * 90)

    print(
        "ROUTE RECOVERY SUMMARY"
    )

    print("=" * 90)

    recovery_summary = (
        recovery
        .groupby(
            "algorithm_display",
            as_index=False,
        )
        .agg(
            workloads=(
                "workload_id",
                "count",
            ),

            exact_matches=(
                "verified_exact",
                "sum",
            ),

            mean_cluster_switches=(
                "cluster_switches",
                "mean",
            ),
        )
    )

    print(
        recovery_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Combine baselines + locked Proposed result
    # --------------------------------------------------------

    baseline_rows = (
        baseline_final_rows(
            recovery
        )
    )

    proposed_rows = (
        load_proposed_rows(
            test_split
        )
    )

    details = pd.concat(
        [
            baseline_rows,
            proposed_rows,
        ],
        ignore_index=True,
    )

    expected_final_rows = (
        EXPECTED_TEST_WORKLOADS
        * len(
            FINAL_ALGORITHM_ORDER
        )
    )

    if (
        len(details)
        != expected_final_rows
    ):
        raise RuntimeError(
            f"Expected {expected_final_rows} "
            f"final rows, found {len(details)}."
        )

    details.to_csv(
        FINAL_DETAILS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        build_final_summary(
            details
        )
    )

    summary.to_csv(
        FINAL_SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Proposed-vs-baseline
    # --------------------------------------------------------

    pairwise = (
        build_pairwise(
            summary
        )
    )

    pairwise.to_csv(
        FINAL_PAIRWISE_PATH,
        index=False,
    )

    write_findings(
        summary=(
            summary
        ),

        pairwise=(
            pairwise
        ),
    )

    # --------------------------------------------------------
    # Final display
    # --------------------------------------------------------

    print()
    print("=" * 100)

    print(
        "FINAL LOCKED OBJECTIVE SUMMARY"
    )

    print("=" * 100)

    display = (
        summary.copy()
    )

    display[
        "weighted_sla"
    ] = (
        display[
            "weighted_sla"
        ]
        * 100.0
    )

    print(
        display[
            [
                "algorithm",
                "mean_distance_km",
                "mean_travel_time_minutes",
                "late_deliveries",
                "weighted_sla",
                "total_lateness_minutes",
                "mean_cluster_switches",
                "mean_cluster_switch_rate",
                "mean_locked_objective",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 100)

    print(
        "PROPOSED RL - LOCKED OBJECTIVE "
        "VS BASELINES"
    )

    print("=" * 100)

    print(
        pairwise.to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  locked_objective_improvement_pct > 0"
    )

    print(
        "      => Proposed RL has lower/better J."
    )

    print(
        "  cluster_switch_reduction_pct > 0"
    )

    print(
        "      => Proposed RL changes cluster less often."
    )

    print()
    print(
        f"Recovery: "
        f"{RECOVERY_PATH}"
    )

    print(
        f"Final details: "
        f"{FINAL_DETAILS_PATH}"
    )

    print(
        f"Final summary: "
        f"{FINAL_SUMMARY_PATH}"
    )

    print(
        f"Pairwise: "
        f"{FINAL_PAIRWISE_PATH}"
    )

    print(
        f"Findings: "
        f"{FINDINGS_PATH}"
    )


if __name__ == "__main__":
    main()