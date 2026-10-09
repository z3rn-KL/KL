"""
Controlled reproducibility validation.

Purpose
-------
Re-run a small representative subset of the canonical
dataset-scale benchmark using the same project implementation,
road graph, workload definitions and RL configurations.

Selected workloads
------------------
29 -> N=5
71 -> N=10
90 -> N=13

Algorithms
----------
- Nearest Neighbor
- Clarke-Wright
- Q-Learning
- SARSA
- DQN

Important
---------
This experiment does NOT overwrite the canonical 540-run benchmark.

It creates independent reproducibility outputs and compares
the rerun metrics with:

    results/dataset_algorithm_all_results.csv

RL algorithms reuse the exact benchmark helper functions:

    run_q_learning(...)
    run_sarsa(...)
    run_dqn(...)

The experiment is resumable. Results are saved after every
algorithm run.
"""

from __future__ import annotations

import gc
import random
import time

from pathlib import Path

import numpy as np
import pandas as pd

import api.dependencies as deps

import experiments.core_algorithm_full_benchmark as benchmark

from routing import (
    NearestNeighborRouter,
    RoadMatrixBuilder,
)

from routing.clarke_wright import (
    ClarkeWrightRouter,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

CANONICAL_RESULTS_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_all_results.csv"
)

WORKLOAD_MANIFEST_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_workloads.csv"
)

OUTPUT_RESULTS_PATH = (
    RESULTS_DIR
    / "reproducibility_validation_results.csv"
)

OUTPUT_SUMMARY_PATH = (
    RESULTS_DIR
    / "reproducibility_validation_summary.csv"
)

OUTPUT_FINDINGS_PATH = (
    RESULTS_DIR
    / "reproducibility_validation_findings.txt"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

WORKLOAD_IDS = [
    29,
    71,
    90,
]

ALGORITHMS = [
    "nearest_neighbor",
    "clarke_wright",
    "q_learning",
    "sarsa",
    "dqn",
]

RANDOM_SEED = 42


# Exact numerical comparison.
EXACT_ATOL = 1e-6


# Practical comparison thresholds.
PRACTICAL_RELATIVE_TOLERANCE_PCT = 1.0

SECONDARY_RELATIVE_TOLERANCE_PCT = 5.0


# ============================================================
# UTILITIES
# ============================================================

def reset_global_seeds(
    seed: int = RANDOM_SEED,
) -> None:
    """
    Reset common RNGs before each RL experiment.

    Individual benchmark agents also use their own configured seed.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    try:
        import torch

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

    except ImportError:
        pass


def cleanup_memory() -> None:
    """
    Reduce memory accumulation between RL runs.
    """

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except ImportError:
        pass


def safe_save_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save through a temporary file to reduce the risk
    of losing results if the process is interrupted.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        path
    )


def relative_difference_pct(
    rerun_value: float,
    original_value: float,
) -> float:
    """
    Signed relative difference:

        (rerun - original) / original * 100
    """

    if np.isclose(
        original_value,
        0.0,
    ):
        if np.isclose(
            rerun_value,
            0.0,
        ):
            return 0.0

        return float(
            "nan"
        )

    return float(
        (
            rerun_value
            - original_value
        )
        / original_value
        * 100.0
    )


def absolute_relative_difference_pct(
    rerun_value: float,
    original_value: float,
) -> float:
    value = (
        relative_difference_pct(
            rerun_value=(
                rerun_value
            ),
            original_value=(
                original_value
            ),
        )
    )

    if np.isnan(
        value
    ):
        return float(
            "nan"
        )

    return abs(
        value
    )


def parse_delivery_ids(
    value,
) -> list[str]:
    return [
        item.strip()

        for item
        in str(
            value
        ).split(
            "->"
        )

        if item.strip()
    ]


def extract_delivery_order(
    run_result,
    expected_delivery_ids: list[str],
) -> list[str]:
    """
    Extract delivery order without assuming the exact
    return tuple layout of QL/SARSA/DQN helpers.

    This keeps the validation compatible with the existing
    benchmark helper functions.
    """

    expected = set(
        str(
            value
        )
        for value
        in expected_delivery_ids
    )

    expected_count = len(
        expected_delivery_ids
    )

    candidates = []

    if hasattr(
        run_result,
        "delivery_order",
    ):
        candidates.append(
            getattr(
                run_result,
                "delivery_order",
            )
        )

    if isinstance(
        run_result,
        dict,
    ):
        for key in [
            "delivery_order",
            "route",
            "route_order",
            "greedy_route",
        ]:
            if key in run_result:
                candidates.append(
                    run_result[
                        key
                    ]
                )

    if isinstance(
        run_result,
        (
            tuple,
            list,
        ),
    ):
        candidates.extend(
            run_result
        )

    for candidate in candidates:
        if not isinstance(
            candidate,
            (
                list,
                tuple,
            ),
        ):
            continue

        normalized = [
            str(
                value
            )
            for value
            in candidate
        ]

        if (
            len(
                normalized
            )
            == expected_count
            and set(
                normalized
            )
            == expected
        ):
            return normalized

    raise RuntimeError(
        "Could not extract delivery order "
        "from algorithm result."
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_manifest(
) -> pd.DataFrame:
    dataframe = pd.read_csv(
        WORKLOAD_MANIFEST_PATH
    )

    dataframe[
        "workload_id"
    ] = pd.to_numeric(
        dataframe[
            "workload_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    return dataframe


def load_canonical_results(
) -> pd.DataFrame:
    dataframe = pd.read_csv(
        CANONICAL_RESULTS_PATH
    )

    dataframe[
        "workload_id"
    ] = pd.to_numeric(
        dataframe[
            "workload_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    return dataframe


def load_existing_results(
) -> pd.DataFrame:
    if not OUTPUT_RESULTS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        OUTPUT_RESULTS_PATH
    )


def completed_keys(
    dataframe: pd.DataFrame,
) -> set[
    tuple[
        int,
        str,
    ]
]:
    if dataframe.empty:
        return set()

    return {
        (
            int(
                row[
                    "workload_id"
                ]
            ),
            str(
                row[
                    "algorithm"
                ]
            ),
        )

        for _, row
        in dataframe.iterrows()
    }


# ============================================================
# WORKLOAD PREPARATION
# ============================================================

def prepare_workload(
    workload_id: int,
    manifest: pd.DataFrame,
):
    matched = manifest[
        manifest[
            "workload_id"
        ]
        == workload_id
    ]

    if (
        len(
            matched
        )
        != 1
    ):
        raise ValueError(
            f"Expected exactly one manifest row "
            f"for workload {workload_id}."
        )

    row = matched.iloc[
        0
    ]

    delivery_ids = (
        parse_delivery_ids(
            row[
                "delivery_ids"
            ]
        )
    )

    expected_size = int(
        row[
            "number_of_deliveries"
        ]
    )

    if (
        len(
            delivery_ids
        )
        != expected_size
    ):
        raise ValueError(
            f"Workload {workload_id}: "
            "delivery count does not match manifest."
        )

    (
        deliveries,
        delivery_nodes,
    ) = (
        deps.resolve_deliveries(
            delivery_ids
        )
    )

    wave_start = (
        pd.Timestamp(
            row[
                "wave_start"
            ]
        )
        .to_pydatetime()
    )

    wave_end = (
        pd.Timestamp(
            row[
                "wave_end"
            ]
        )
        .to_pydatetime()
    )

    return (
        delivery_ids,
        deliveries,
        delivery_nodes,
        wave_start,
        wave_end,
    )


# ============================================================
# ALGORITHM EXECUTION
# ============================================================

def run_nearest_neighbor(
    depot_node: int,
    delivery_ids: list[str],
    delivery_nodes: list[int],
):
    router = (
        NearestNeighborRouter(
            road_network=(
                deps
                .get_road_network()
            ),
            metric="distance",
        )
    )

    return router.build_route(
        depot_node=(
            depot_node
        ),
        delivery_ids=(
            delivery_ids
        ),
        delivery_nodes=(
            delivery_nodes
        ),
        return_to_depot=True,
    )


def run_clarke_wright(
    depot_node: int,
    delivery_ids: list[str],
    delivery_nodes: list[int],
):
    router = (
        ClarkeWrightRouter(
            road_network=(
                deps
                .get_road_network()
            )
        )
    )

    return router.build_route(
        depot_node=(
            depot_node
        ),
        delivery_ids=(
            delivery_ids
        ),
        delivery_nodes=(
            delivery_nodes
        ),
        return_to_depot=True,
    )


def run_algorithm(
    algorithm: str,
    *,
    depot_node: int,
    delivery_ids: list[str],
    delivery_nodes: list[int],
    deliveries,
    road_matrix,
    start_time,
):
    """
    Run one algorithm and return:

        delivery_order,
        execution_runtime_seconds

    RL algorithms reuse exactly the helper functions from
    core_algorithm_full_benchmark.py.

    Their return structure is:

        rollout,
        training,
        training_runtime,
        route_runtime

    and the final route is stored in:

        rollout.delivery_order
    """

    reset_global_seeds()

    start = time.perf_counter()

    # ========================================================
    # NEAREST NEIGHBOR
    # ========================================================

    if (
        algorithm
        == "nearest_neighbor"
    ):
        result = (
            run_nearest_neighbor(
                depot_node=(
                    depot_node
                ),
                delivery_ids=(
                    delivery_ids
                ),
                delivery_nodes=(
                    delivery_nodes
                ),
            )
        )

        delivery_order = [
            str(
                value
            )
            for value
            in result.delivery_order
        ]

    # ========================================================
    # CLARKE-WRIGHT
    # ========================================================

    elif (
        algorithm
        == "clarke_wright"
    ):
        result = (
            run_clarke_wright(
                depot_node=(
                    depot_node
                ),
                delivery_ids=(
                    delivery_ids
                ),
                delivery_nodes=(
                    delivery_nodes
                ),
            )
        )

        delivery_order = [
            str(
                value
            )
            for value
            in result.delivery_order
        ]

    # ========================================================
    # Q-LEARNING
    # ========================================================

    elif (
        algorithm
        == "q_learning"
    ):
        (
            rollout,
            training,
            training_runtime,
            route_runtime,
        ) = (
            benchmark.run_q_learning(
                deliveries=deliveries,
                road_matrix=road_matrix,
                start_time=start_time,
            )
        )

        delivery_order = [
            str(
                value
            )
            for value
            in rollout.delivery_order
        ]

    # ========================================================
    # SARSA
    # ========================================================

    elif (
        algorithm
        == "sarsa"
    ):
        (
            rollout,
            training,
            training_runtime,
            route_runtime,
        ) = (
            benchmark.run_sarsa(
                deliveries=deliveries,
                road_matrix=road_matrix,
                start_time=start_time,
            )
        )

        delivery_order = [
            str(
                value
            )
            for value
            in rollout.delivery_order
        ]

    # ========================================================
    # DQN
    # ========================================================

    elif (
        algorithm
        == "dqn"
    ):
        (
            rollout,
            training,
            training_runtime,
            route_runtime,
        ) = (
            benchmark.run_dqn(
                deliveries=deliveries,
                road_matrix=road_matrix,
                start_time=start_time,
            )
        )

        delivery_order = [
            str(
                value
            )
            for value
            in rollout.delivery_order
        ]

    else:
        raise ValueError(
            f"Unsupported algorithm: "
            f"{algorithm}"
        )

    runtime = (
        time.perf_counter()
        - start
    )

    # ========================================================
    # SAFETY VALIDATION
    # ========================================================

    expected_ids = {
        str(
            value
        )
        for value
        in delivery_ids
    }

    actual_ids = set(
        delivery_order
    )

    if (
        len(
            delivery_order
        )
        != len(
            delivery_ids
        )
    ):
        raise RuntimeError(
            f"{algorithm}: delivery-order "
            "length does not match workload."
        )

    if (
        actual_ids
        != expected_ids
    ):
        raise RuntimeError(
            f"{algorithm}: delivery-order "
            "IDs do not match workload IDs."
        )

    return (
        delivery_order,
        float(
            runtime
        ),
    )


# ============================================================
# COMPARISON
# ============================================================

def get_reference_row(
    canonical: pd.DataFrame,
    workload_id: int,
    algorithm: str,
) -> pd.Series:
    matched = canonical[
        (
            canonical[
                "workload_id"
            ]
            == workload_id
        )
        &
        (
            canonical[
                "algorithm"
            ]
            == algorithm
        )
    ]

    if (
        len(
            matched
        )
        != 1
    ):
        raise ValueError(
            f"Canonical result missing or duplicated: "
            f"workload={workload_id}, "
            f"algorithm={algorithm}"
        )

    return matched.iloc[
        0
    ]


def build_result_row(
    *,
    workload_id: int,
    number_of_deliveries: int,
    algorithm: str,
    wave_start,
    wave_end,
    delivery_order: list[str],
    evaluation,
    runtime_seconds: float,
    reference: pd.Series,
) -> dict:
    original_distance = float(
        reference[
            "total_distance_km"
        ]
    )

    original_time = float(
        reference[
            "total_travel_time_minutes"
        ]
    )

    original_on_time_rate = float(
        reference[
            "on_time_rate"
        ]
    )

    original_late = int(
        reference[
            "late_deliveries"
        ]
    )

    rerun_distance = float(
        evaluation
        .total_distance_km
    )

    rerun_time = float(
        evaluation
        .total_travel_time_minutes
    )

    rerun_on_time_rate = float(
        evaluation
        .on_time_rate
    )

    rerun_late = int(
        evaluation
        .late_deliveries
    )

    distance_difference = (
        rerun_distance
        - original_distance
    )

    time_difference = (
        rerun_time
        - original_time
    )

    distance_relative_pct = (
        relative_difference_pct(
            rerun_value=(
                rerun_distance
            ),
            original_value=(
                original_distance
            ),
        )
    )

    time_relative_pct = (
        relative_difference_pct(
            rerun_value=(
                rerun_time
            ),
            original_value=(
                original_time
            ),
        )
    )

    distance_exact = bool(
        np.isclose(
            rerun_distance,
            original_distance,
            atol=EXACT_ATOL,
            rtol=0.0,
        )
    )

    time_exact = bool(
        np.isclose(
            rerun_time,
            original_time,
            atol=EXACT_ATOL,
            rtol=0.0,
        )
    )

    sla_exact = bool(
        np.isclose(
            rerun_on_time_rate,
            original_on_time_rate,
            atol=EXACT_ATOL,
            rtol=0.0,
        )
    )

    late_exact = bool(
        rerun_late
        == original_late
    )

    exact_core_match = bool(
        distance_exact
        and time_exact
        and sla_exact
        and late_exact
    )

    distance_abs_pct = (
        absolute_relative_difference_pct(
            rerun_value=(
                rerun_distance
            ),
            original_value=(
                original_distance
            ),
        )
    )

    time_abs_pct = (
        absolute_relative_difference_pct(
            rerun_value=(
                rerun_time
            ),
            original_value=(
                original_time
            ),
        )
    )

    within_1pct = bool(
        (
            np.isnan(
                distance_abs_pct
            )
            or distance_abs_pct
            <= PRACTICAL_RELATIVE_TOLERANCE_PCT
        )
        and
        (
            np.isnan(
                time_abs_pct
            )
            or time_abs_pct
            <= PRACTICAL_RELATIVE_TOLERANCE_PCT
        )
        and late_exact
    )

    within_5pct = bool(
        (
            np.isnan(
                distance_abs_pct
            )
            or distance_abs_pct
            <= SECONDARY_RELATIVE_TOLERANCE_PCT
        )
        and
        (
            np.isnan(
                time_abs_pct
            )
            or time_abs_pct
            <= SECONDARY_RELATIVE_TOLERANCE_PCT
        )
        and late_exact
    )

    return {
        "workload_id":
            workload_id,

        "number_of_deliveries":
            number_of_deliveries,

        "algorithm":
            algorithm,

        "wave_start":
            wave_start,

        "wave_end":
            wave_end,

        "seed":
            RANDOM_SEED,

        "rerun_delivery_order":
            " -> ".join(
                delivery_order
            ),

        "original_distance_km":
            original_distance,

        "rerun_distance_km":
            rerun_distance,

        "distance_difference_km":
            distance_difference,

        "distance_relative_difference_pct":
            distance_relative_pct,

        "original_travel_time_minutes":
            original_time,

        "rerun_travel_time_minutes":
            rerun_time,

        "travel_time_difference_minutes":
            time_difference,

        "travel_time_relative_difference_pct":
            time_relative_pct,

        "original_on_time_rate":
            original_on_time_rate,

        "rerun_on_time_rate":
            rerun_on_time_rate,

        "on_time_rate_difference":
            (
                rerun_on_time_rate
                - original_on_time_rate
            ),

        "original_late_deliveries":
            original_late,

        "rerun_late_deliveries":
            rerun_late,

        "late_delivery_difference":
            (
                rerun_late
                - original_late
            ),

        "exact_distance_match":
            distance_exact,

        "exact_travel_time_match":
            time_exact,

        "exact_sla_match":
            sla_exact,

        "exact_late_delivery_match":
            late_exact,

        "exact_core_match":
            exact_core_match,

        "within_1pct_core_metrics":
            within_1pct,

        "within_5pct_core_metrics":
            within_5pct,

        "rerun_runtime_seconds":
            runtime_seconds,
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for algorithm in (
        ALGORITHMS
    ):
        subset = results[
            results[
                "algorithm"
            ]
            == algorithm
        ]

        if subset.empty:
            continue

        rows.append(
            {
                "algorithm":
                    algorithm,

                "runs":
                    int(
                        len(
                            subset
                        )
                    ),

                "exact_core_matches":
                    int(
                        subset[
                            "exact_core_match"
                        ]
                        .sum()
                    ),

                "exact_core_match_rate":
                    float(
                        subset[
                            "exact_core_match"
                        ]
                        .mean()
                    ),

                "within_1pct_matches":
                    int(
                        subset[
                            "within_1pct_core_metrics"
                        ]
                        .sum()
                    ),

                "within_1pct_match_rate":
                    float(
                        subset[
                            "within_1pct_core_metrics"
                        ]
                        .mean()
                    ),

                "within_5pct_matches":
                    int(
                        subset[
                            "within_5pct_core_metrics"
                        ]
                        .sum()
                    ),

                "within_5pct_match_rate":
                    float(
                        subset[
                            "within_5pct_core_metrics"
                        ]
                        .mean()
                    ),

                "mean_abs_distance_difference_pct":
                    float(
                        subset[
                            "distance_relative_difference_pct"
                        ]
                        .abs()
                        .mean()
                    ),

                "max_abs_distance_difference_pct":
                    float(
                        subset[
                            "distance_relative_difference_pct"
                        ]
                        .abs()
                        .max()
                    ),

                "mean_abs_travel_time_difference_pct":
                    float(
                        subset[
                            "travel_time_relative_difference_pct"
                        ]
                        .abs()
                        .mean()
                    ),

                "max_abs_travel_time_difference_pct":
                    float(
                        subset[
                            "travel_time_relative_difference_pct"
                        ]
                        .abs()
                        .max()
                    ),

                "mean_runtime_seconds":
                    float(
                        subset[
                            "rerun_runtime_seconds"
                        ]
                        .mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def write_findings(
    results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    exact_count = int(
        results[
            "exact_core_match"
        ]
        .sum()
    )

    one_pct_count = int(
        results[
            "within_1pct_core_metrics"
        ]
        .sum()
    )

    five_pct_count = int(
        results[
            "within_5pct_core_metrics"
        ]
        .sum()
    )

    total = len(
        results
    )

    lines = [
        "CONTROLLED REPRODUCIBILITY VALIDATION",
        "=" * 72,
        "",
        "Scope:",
        (
            f"- Workloads: "
            f"{WORKLOAD_IDS}"
        ),
        (
            "- Workload sizes: "
            "N=5, N=10, N=13"
        ),
        (
            f"- Algorithms: "
            f"{', '.join(ALGORITHMS)}"
        ),
        (
            f"- Random seed: "
            f"{RANDOM_SEED}"
        ),
        (
            f"- Expected reruns: "
            f"{len(WORKLOAD_IDS) * len(ALGORITHMS)}"
        ),
        "",
        "Overall comparison:",
        (
            f"- Completed comparisons: "
            f"{total}"
        ),
        (
            f"- Exact core matches: "
            f"{exact_count}/{total}"
        ),
        (
            f"- Within 1% core metrics: "
            f"{one_pct_count}/{total}"
        ),
        (
            f"- Within 5% core metrics: "
            f"{five_pct_count}/{total}"
        ),
        "",
        "Per algorithm:",
    ]

    for _, row in (
        summary.iterrows()
    ):
        lines.extend(
            [
                (
                    f"- {row['algorithm']}: "
                    f"exact "
                    f"{int(row['exact_core_matches'])}/"
                    f"{int(row['runs'])}; "
                    f"within 1% "
                    f"{int(row['within_1pct_matches'])}/"
                    f"{int(row['runs'])}; "
                    f"mean |distance delta| "
                    f"{row['mean_abs_distance_difference_pct']:.4f}%; "
                    f"mean |time delta| "
                    f"{row['mean_abs_travel_time_difference_pct']:.4f}%."
                )
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation note:",
            (
                "- Exact equality is expected to be strongest "
                "for deterministic heuristics."
            ),
            (
                "- RL reruns use the same benchmark helper "
                "functions, seed and hyperparameter configuration."
            ),
            (
                "- GPU-backed DQN may still exhibit small "
                "numerical or training differences depending "
                "on CUDA/PyTorch execution."
            ),
            (
                "- This validation does not replace the canonical "
                "540-run benchmark; it is a controlled reproducibility "
                "check on representative workloads."
            ),
            "",
        ]
    )

    OUTPUT_FINDINGS_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("=" * 72)
    print(
        "CONTROLLED REPRODUCIBILITY VALIDATION"
    )
    print("=" * 72)

    print(
        f"Workloads: {WORKLOAD_IDS}"
    )

    print(
        f"Algorithms: {ALGORITHMS}"
    )

    print(
        f"Seed: {RANDOM_SEED}"
    )

    print(
        "Resume mode: ENABLED"
    )

    manifest = (
        load_manifest()
    )

    canonical = (
        load_canonical_results()
    )

    results = (
        load_existing_results()
    )

    completed = (
        completed_keys(
            results
        )
    )

    road_network = (
        deps.get_road_network()
    )

    depot = (
        deps.get_depot_context()
    )

    evaluator = (
        deps.get_route_evaluator()
    )

    matrix_builder = (
        RoadMatrixBuilder(
            road_network
        )
    )

    expected_runs = (
        len(
            WORKLOAD_IDS
        )
        * len(
            ALGORITHMS
        )
    )

    run_number = len(
        completed
    )

    for workload_id in (
        WORKLOAD_IDS
    ):
        (
            delivery_ids,
            deliveries,
            delivery_nodes,
            wave_start,
            wave_end,
        ) = (
            prepare_workload(
                workload_id=(
                    workload_id
                ),
                manifest=(
                    manifest
                ),
            )
        )

        number_of_deliveries = (
            len(
                deliveries
            )
        )

        print()
        print("=" * 72)

        print(
            f"WORKLOAD {workload_id}"
        )

        print(
            f"N={number_of_deliveries}"
        )

        print(
            f"wave_start={wave_start}"
        )

        print(
            f"wave_end={wave_end}"
        )

        print(
            "Preparing road matrix..."
        )

        matrix_start = (
            time.perf_counter()
        )

        road_matrix = (
            matrix_builder.build(
                [
                    depot.road_node,
                    *delivery_nodes,
                ]
            )
        )

        matrix_runtime = (
            time.perf_counter()
            - matrix_start
        )

        print(
            f"Road matrix: "
            f"{matrix_runtime:.3f}s"
        )

        for algorithm in (
            ALGORITHMS
        ):
            key = (
                workload_id,
                algorithm,
            )

            if key in completed:
                print(
                    f"SKIP "
                    f"workload={workload_id}, "
                    f"algorithm={algorithm}"
                )

                continue

            run_number += 1

            print()
            print(
                "-" * 72
            )

            print(
                f"Run "
                f"{run_number}/"
                f"{expected_runs}"
            )

            print(
                f"workload={workload_id}, "
                f"N={number_of_deliveries}, "
                f"algorithm={algorithm}"
            )

            (
                delivery_order,
                runtime_seconds,
            ) = (
                run_algorithm(
                    algorithm=(
                        algorithm
                    ),
                    depot_node=(
                        depot.road_node
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
                        wave_end
                    ),
                )
            )

            evaluation = (
                benchmark.evaluate(
                    evaluator=(
                        evaluator
                    ),
                    algorithm=(
                        algorithm
                    ),
                    depot_node=(
                        depot.road_node
                    ),
                    deliveries=(
                        deliveries
                    ),
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    delivery_order=(
                        delivery_order
                    ),
                    start_time=(
                        wave_end
                    ),
                )
            )

            reference = (
                get_reference_row(
                    canonical=(
                        canonical
                    ),
                    workload_id=(
                        workload_id
                    ),
                    algorithm=(
                        algorithm
                    ),
                )
            )

            row = (
                build_result_row(
                    workload_id=(
                        workload_id
                    ),
                    number_of_deliveries=(
                        number_of_deliveries
                    ),
                    algorithm=(
                        algorithm
                    ),
                    wave_start=(
                        wave_start
                    ),
                    wave_end=(
                        wave_end
                    ),
                    delivery_order=(
                        delivery_order
                    ),
                    evaluation=(
                        evaluation
                    ),
                    runtime_seconds=(
                        runtime_seconds
                    ),
                    reference=(
                        reference
                    ),
                )
            )

            row[
                "matrix_preparation_seconds"
            ] = float(
                matrix_runtime
            )

            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        [
                            row
                        ]
                    ),
                ],
                ignore_index=True,
            )

            safe_save_csv(
                results,
                OUTPUT_RESULTS_PATH,
            )

            completed.add(
                key
            )

            print(
                f"Original distance: "
                f"{row['original_distance_km']:.4f} km"
            )

            print(
                f"Rerun distance   : "
                f"{row['rerun_distance_km']:.4f} km"
            )

            print(
                f"Distance delta   : "
                f"{row['distance_relative_difference_pct']:+.4f}%"
            )

            print(
                f"Original time    : "
                f"{row['original_travel_time_minutes']:.4f} min"
            )

            print(
                f"Rerun time       : "
                f"{row['rerun_travel_time_minutes']:.4f} min"
            )

            print(
                f"Time delta       : "
                f"{row['travel_time_relative_difference_pct']:+.4f}%"
            )

            print(
                f"Exact match      : "
                f"{row['exact_core_match']}"
            )

            print(
                f"Within 1%        : "
                f"{row['within_1pct_core_metrics']}"
            )

            print(
                f"Runtime          : "
                f"{runtime_seconds:.2f}s"
            )

            cleanup_memory()

    if results.empty:
        raise RuntimeError(
            "No reproducibility results were produced."
        )

    results = (
        results
        .sort_values(
            [
                "workload_id",
                "algorithm",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    safe_save_csv(
        results,
        OUTPUT_RESULTS_PATH,
    )

    summary = (
        build_summary(
            results
        )
    )

    safe_save_csv(
        summary,
        OUTPUT_SUMMARY_PATH,
    )

    write_findings(
        results=(
            results
        ),
        summary=(
            summary
        ),
    )

    print()
    print("=" * 72)
    print(
        "REPRODUCIBILITY SUMMARY"
    )
    print("=" * 72)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Created:"
    )

    print(
        f"- {OUTPUT_RESULTS_PATH}"
    )

    print(
        f"- {OUTPUT_SUMMARY_PATH}"
    )

    print(
        f"- {OUTPUT_FINDINGS_PATH}"
    )

    print()
    print(
        "Reproducibility validation complete."
    )
    print()


if __name__ == "__main__":
    main()