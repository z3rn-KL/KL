"""
Controlled reproducibility validation for K-Means-guided DQN.

Representative workloads:
    workload 101 -> N=5
    workload 106 -> N=10
    workload 97  -> N=13

Configuration:
    cluster_switch_penalty = 0.50
    training episodes      = 5000
    random seed            = 42

The experiment reuses the exact Guided-DQN implementation from:

    experiments.kmeans_guided_dqn_sensitivity

and compares the reruns against the stored final-validation results from:

    experiments.final_kmeans_guided_dqn_validation

The original final-validation results are never overwritten.

The experiment supports:
    - autosave after every completed workload
    - resume after interruption
    - exact metric comparison
    - practical 1% / 5% comparison
    - exact route comparison
    - cluster-switch comparison
"""

from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import api.dependencies as deps

import experiments.final_kmeans_guided_dqn_validation as final_validation
import experiments.kmeans_guided_dqn_sensitivity as guided_experiment

from routing import RoadMatrixBuilder


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_RESULTS_PATH = (
    RESULTS_DIR
    / "guided_dqn_reproducibility_results.csv"
)

OUTPUT_SUMMARY_PATH = (
    RESULTS_DIR
    / "guided_dqn_reproducibility_summary.csv"
)

OUTPUT_FINDINGS_PATH = (
    RESULTS_DIR
    / "guided_dqn_reproducibility_findings.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

WORKLOAD_IDS = [
    101,  # N=5
    106,  # N=10
    97,   # N=13
]

GUIDED_LAMBDA = 0.50

RANDOM_SEED = 42

TRAINING_EPISODES = 5000

EXACT_ATOL = 1e-6

PRACTICAL_TOLERANCE_PCT = 1.0

SECONDARY_TOLERANCE_PCT = 5.0


# ============================================================
# CONFIGURATION CHECK
# ============================================================

def validate_configuration() -> None:
    final_lambda = float(
        final_validation.GUIDED_LAMBDA
    )

    experiment_episodes = int(
        guided_experiment.TRAINING_EPISODES
    )

    experiment_seed = int(
        guided_experiment.RANDOM_SEED
    )

    if not np.isclose(
        final_lambda,
        GUIDED_LAMBDA,
    ):
        raise ValueError(
            "GUIDED_LAMBDA mismatch: "
            f"final validation={final_lambda}, "
            f"reproducibility={GUIDED_LAMBDA}"
        )

    if (
        experiment_episodes
        != TRAINING_EPISODES
    ):
        raise ValueError(
            "TRAINING_EPISODES mismatch: "
            f"experiment={experiment_episodes}, "
            f"reproducibility={TRAINING_EPISODES}"
        )

    if (
        experiment_seed
        != RANDOM_SEED
    ):
        raise ValueError(
            "RANDOM_SEED mismatch: "
            f"experiment={experiment_seed}, "
            f"reproducibility={RANDOM_SEED}"
        )


# ============================================================
# RANDOM SEEDS
# ============================================================

def reset_global_seeds() -> None:
    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    try:
        import torch

        torch.manual_seed(
            RANDOM_SEED
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                RANDOM_SEED
            )

    except ImportError:
        pass


# ============================================================
# SAVE / RESUME
# ============================================================

def safe_save_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        path
    )


def load_existing_results() -> pd.DataFrame:
    if not OUTPUT_RESULTS_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        OUTPUT_RESULTS_PATH
    )

    if dataframe.empty:
        return dataframe

    if (
        "workload_id"
        not in dataframe.columns
    ):
        raise ValueError(
            "Existing reproducibility CSV has an "
            "unexpected schema. Delete this file and run again:\n"
            f"{OUTPUT_RESULTS_PATH}"
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


# ============================================================
# DELIVERY ID HELPERS
# ============================================================

def normalize_delivery_id_column(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    if (
        "delivery_id"
        not in result.columns
        and
        "id"
        in result.columns
    ):
        result = result.rename(
            columns={
                "id":
                    "delivery_id"
            }
        )

    if (
        "delivery_id"
        not in result.columns
    ):
        raise ValueError(
            "Could not find delivery_id or id column."
        )

    result[
        "delivery_id"
    ] = (
        result[
            "delivery_id"
        ]
        .astype(
            str
        )
    )

    return result


def ensure_cluster_ids(
    workload_dataframe: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
) -> pd.DataFrame:
    workload = (
        normalize_delivery_id_column(
            workload_dataframe
        )
    )

    if (
        "cluster_id"
        in workload.columns
    ):
        workload[
            "cluster_id"
        ] = pd.to_numeric(
            workload[
                "cluster_id"
            ],
            errors="raise",
        ).astype(
            int
        )

        return workload

    clusters = (
        normalize_delivery_id_column(
            cluster_assignments
        )
    )

    if (
        "cluster_id"
        not in clusters.columns
    ):
        raise ValueError(
            "Cluster assignments do not "
            "contain cluster_id."
        )

    clusters = (
        clusters[
            [
                "delivery_id",
                "cluster_id",
            ]
        ]
        .drop_duplicates(
            subset=[
                "delivery_id"
            ]
        )
        .copy()
    )

    clusters[
        "cluster_id"
    ] = pd.to_numeric(
        clusters[
            "cluster_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    result = workload.merge(
        clusters,
        on="delivery_id",
        how="left",
        validate="one_to_one",
    )

    if (
        result[
            "cluster_id"
        ]
        .isna()
        .any()
    ):
        missing = (
            result.loc[
                result[
                    "cluster_id"
                ]
                .isna(),
                "delivery_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Missing cluster assignments for: "
            f"{missing}"
        )

    result[
        "cluster_id"
    ] = (
        result[
            "cluster_id"
        ]
        .astype(
            int
        )
    )

    return result


# ============================================================
# FINAL WORKLOAD PIPELINE
# ============================================================

def load_final_workloads() -> pd.DataFrame:
    """
    Reconstruct exactly the workload table used by the
    final Guided-DQN validation.

    Important:
    build_workload_dataframe_from_manifest() requires
    manifest_delivery_ids, which only exists after
    select_final_workloads().
    """

    manifest = (
        final_validation
        .load_workload_manifest()
        .copy()
    )

    baseline_dqn = (
        final_validation
        .load_baseline_dqn(
            manifest
        )
    )

    selected = (
        final_validation
        .select_final_workloads(
            manifest=manifest,
            baseline_dqn=baseline_dqn,
        )
        .copy()
    )

    required_columns = [
        "workload_id",
        "number_of_deliveries",
        "manifest_delivery_ids",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in selected.columns
    ]

    if missing_columns:
        raise ValueError(
            "Final workload table is missing: "
            f"{missing_columns}"
        )

    selected[
        "workload_id"
    ] = pd.to_numeric(
        selected[
            "workload_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    available_ids = set(
        selected[
            "workload_id"
        ]
        .tolist()
    )

    missing_workloads = [
        workload_id
        for workload_id
        in WORKLOAD_IDS
        if workload_id
        not in available_ids
    ]

    if missing_workloads:
        raise ValueError(
            "Reproducibility workloads were not found "
            "in final validation: "
            f"{missing_workloads}"
        )

    return selected


# ============================================================
# REFERENCE FINAL-VALIDATION RESULTS
# ============================================================

def load_reference_results() -> pd.DataFrame:
    dataframe = (
        final_validation
        .load_existing_results()
        .copy()
    )

    if dataframe.empty:
        raise ValueError(
            "Stored Guided-DQN final-validation "
            "results are empty."
        )

    required_columns = [
        "workload_id",
        "number_of_deliveries",
        "cluster_switch_penalty",
        "total_distance_km",
        "total_travel_time_minutes",
        "on_time_rate",
        "late_deliveries",
        "cluster_switches",
        "cluster_switch_rate",
        "delivery_order",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Stored final-validation results "
            "are missing columns: "
            f"{missing_columns}"
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

    dataframe[
        "cluster_switch_penalty"
    ] = pd.to_numeric(
        dataframe[
            "cluster_switch_penalty"
        ],
        errors="raise",
    )

    return dataframe


def get_reference_row(
    reference_results: pd.DataFrame,
    workload_id: int,
) -> pd.Series:
    matched = reference_results[
        (
            reference_results[
                "workload_id"
            ]
            == workload_id
        )
        &
        (
            np.isclose(
                reference_results[
                    "cluster_switch_penalty"
                ],
                GUIDED_LAMBDA,
            )
        )
    ]

    if (
        len(
            matched
        )
        != 1
    ):
        raise ValueError(
            "Expected exactly one Guided-DQN "
            "reference row for workload "
            f"{workload_id}; found {len(matched)}."
        )

    return matched.iloc[
        0
    ]


# ============================================================
# DELIVERY ORDER / CLUSTER SWITCHES
# ============================================================

def parse_delivery_order(
    value: object,
) -> list[str]:
    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            str(item)
            for item
            in value
        ]

    text = str(
        value
    ).strip()

    if not text:
        return []

    if "->" in text:
        return [
            item.strip()
            for item
            in text.split(
                "->"
            )
            if item.strip()
        ]

    if "|" in text:
        return [
            item.strip()
            for item
            in text.split(
                "|"
            )
            if item.strip()
        ]

    if "," in text:
        return [
            item.strip()
            for item
            in text.split(
                ","
            )
            if item.strip()
        ]

    return [
        text
    ]


def calculate_cluster_switches(
    delivery_order: list[str],
    cluster_lookup: dict[str, int],
) -> tuple[
    int,
    float,
    list[int],
]:
    cluster_sequence = []

    for delivery_id in delivery_order:
        key = str(
            delivery_id
        )

        if key not in cluster_lookup:
            raise ValueError(
                "Missing cluster ID for delivery "
                f"{key}"
            )

        cluster_sequence.append(
            int(
                cluster_lookup[
                    key
                ]
            )
        )

    if (
        len(
            cluster_sequence
        )
        <= 1
    ):
        return (
            0,
            0.0,
            cluster_sequence,
        )

    switches = sum(
        1
        for previous, current
        in zip(
            cluster_sequence,
            cluster_sequence[
                1:
            ],
        )
        if previous != current
    )

    switch_rate = (
        switches
        / (
            len(
                cluster_sequence
            )
            - 1
        )
    )

    return (
        int(
            switches
        ),
        float(
            switch_rate
        ),
        cluster_sequence,
    )


# ============================================================
# COMPARISON HELPERS
# ============================================================

def relative_difference_pct(
    rerun_value: float,
    reference_value: float,
) -> float:
    if np.isclose(
        reference_value,
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
            - reference_value
        )
        / reference_value
        * 100.0
    )


def exact_numeric_match(
    rerun_value: float,
    reference_value: float,
) -> bool:
    return bool(
        np.isclose(
            rerun_value,
            reference_value,
            atol=EXACT_ATOL,
            rtol=0.0,
        )
    )


# ============================================================
# GUIDED DQN RETURN HANDLING
# ============================================================

def run_guided_dqn(
    *,
    deliveries,
    road_matrix,
    start_time,
    cluster_ids: list[int],
):
    """
    The current Guided-DQN helper returns five values.

    Reproducibility only requires:
        [0] rollout
        [1] training
        [2] training runtime
        [3] route runtime

    Any additional metadata is deliberately ignored.
    """

    run_result = (
        guided_experiment.run_dqn(
            deliveries=deliveries,
            road_matrix=road_matrix,
            start_time=start_time,
            cluster_ids=cluster_ids,
            cluster_switch_penalty=(
                GUIDED_LAMBDA
            ),
        )
    )

    if not isinstance(
        run_result,
        (
            tuple,
            list,
        ),
    ):
        raise RuntimeError(
            "run_dqn() returned an unexpected "
            f"type: {type(run_result)}"
        )

    if (
        len(
            run_result
        )
        < 4
    ):
        raise RuntimeError(
            "run_dqn() returned fewer than "
            "four values."
        )

    rollout = run_result[0]

    training = run_result[1]

    training_runtime = float(
        run_result[2]
    )

    route_runtime = float(
        run_result[3]
    )

    return (
        rollout,
        training,
        training_runtime,
        route_runtime,
    )


# ============================================================
# RUN ONE WORKLOAD
# ============================================================

def run_one_workload(
    *,
    workload_id: int,
    final_workloads: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    reference_results: pd.DataFrame,
    matrix_builder: RoadMatrixBuilder,
    evaluator,
    depot_node: int,
) -> dict:
    selected = final_workloads[
        final_workloads[
            "workload_id"
        ]
        == workload_id
    ]

    if (
        len(
            selected
        )
        != 1
    ):
        raise ValueError(
            f"Workload {workload_id} was not "
            "found exactly once."
        )

    selected_row = (
        selected.iloc[
            0
        ]
    )

    # --------------------------------------------------------
    # Rebuild the exact workload used in final validation.
    # --------------------------------------------------------

    workload_dataframe = (
        final_validation
        .build_workload_dataframe_from_manifest(
            prepared_dataframe=(
                prepared_dataframe
            ),
            selected_row=(
                selected_row
            ),
        )
    )

    workload_dataframe = (
        ensure_cluster_ids(
            workload_dataframe=(
                workload_dataframe
            ),
            cluster_assignments=(
                cluster_assignments
            ),
        )
    )

    delivery_ids = (
        workload_dataframe[
            "delivery_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    cluster_ids = (
        workload_dataframe[
            "cluster_id"
        ]
        .astype(
            int
        )
        .tolist()
    )

    cluster_lookup = dict(
        zip(
            delivery_ids,
            cluster_ids,
        )
    )

    (
        deliveries,
        delivery_nodes,
    ) = (
        deps.resolve_deliveries(
            delivery_ids
        )
    )

    number_of_deliveries = (
        len(
            deliveries
        )
    )

    expected_size = int(
        selected_row[
            "number_of_deliveries"
        ]
    )

    if (
        number_of_deliveries
        != expected_size
    ):
        raise ValueError(
            f"Workload {workload_id}: "
            f"expected N={expected_size}, "
            f"got N={number_of_deliveries}."
        )

    start_time = (
        pd.Timestamp(
            selected_row[
                "wave_end"
            ]
        )
        .to_pydatetime()
    )

    # --------------------------------------------------------
    # Road matrix
    # --------------------------------------------------------

    print(
        "Preparing road matrix..."
    )

    matrix_timer = (
        time.perf_counter()
    )

    road_matrix = (
        matrix_builder.build(
            [
                depot_node,
                *delivery_nodes,
            ]
        )
    )

    matrix_runtime = (
        time.perf_counter()
        - matrix_timer
    )

    print(
        "Road matrix: "
        f"{matrix_runtime:.3f}s"
    )

    # --------------------------------------------------------
    # Guided DQN training
    # --------------------------------------------------------

    reset_global_seeds()

    print(
        "Training K-Means-guided DQN..."
    )

    (
        rollout,
        training,
        training_runtime,
        route_runtime,
    ) = (
        run_guided_dqn(
            deliveries=deliveries,
            road_matrix=road_matrix,
            start_time=start_time,
            cluster_ids=cluster_ids,
        )
    )

    delivery_order = [
        str(delivery_id)
        for delivery_id
        in rollout.delivery_order
    ]

    if (
        len(
            delivery_order
        )
        != number_of_deliveries
    ):
        raise RuntimeError(
            "Guided-DQN route length does not "
            "match workload size."
        )

    if (
        set(
            delivery_order
        )
        != set(
            delivery_ids
        )
    ):
        raise RuntimeError(
            "Guided-DQN delivery order does not "
            "match workload delivery IDs."
        )

    # --------------------------------------------------------
    # Route evaluation
    # --------------------------------------------------------

    evaluation = (
        guided_experiment
        .evaluate_route(
            evaluator=evaluator,
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=delivery_order,
            start_time=start_time,
        )
    )

    (
        rerun_switches,
        rerun_switch_rate,
        rerun_cluster_sequence,
    ) = (
        calculate_cluster_switches(
            delivery_order=(
                delivery_order
            ),
            cluster_lookup=(
                cluster_lookup
            ),
        )
    )

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = (
        get_reference_row(
            reference_results=(
                reference_results
            ),
            workload_id=(
                workload_id
            ),
        )
    )

    reference_distance = float(
        reference[
            "total_distance_km"
        ]
    )

    reference_time = float(
        reference[
            "total_travel_time_minutes"
        ]
    )

    reference_sla = float(
        reference[
            "on_time_rate"
        ]
    )

    reference_late = int(
        reference[
            "late_deliveries"
        ]
    )

    reference_switches = int(
        reference[
            "cluster_switches"
        ]
    )

    reference_switch_rate = float(
        reference[
            "cluster_switch_rate"
        ]
    )

    reference_delivery_order = (
        parse_delivery_order(
            reference[
                "delivery_order"
            ]
        )
    )

    # --------------------------------------------------------
    # Rerun metrics
    # --------------------------------------------------------

    rerun_distance = float(
        evaluation
        .total_distance_km
    )

    rerun_time = float(
        evaluation
        .total_travel_time_minutes
    )

    rerun_sla = float(
        evaluation
        .on_time_rate
    )

    rerun_late = int(
        evaluation
        .late_deliveries
    )

    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    distance_difference_pct = (
        relative_difference_pct(
            rerun_value=(
                rerun_distance
            ),
            reference_value=(
                reference_distance
            ),
        )
    )

    time_difference_pct = (
        relative_difference_pct(
            rerun_value=(
                rerun_time
            ),
            reference_value=(
                reference_time
            ),
        )
    )

    switch_rate_difference_pct = (
        relative_difference_pct(
            rerun_value=(
                rerun_switch_rate
            ),
            reference_value=(
                reference_switch_rate
            ),
        )
    )

    # --------------------------------------------------------
    # Exact comparisons
    # --------------------------------------------------------

    exact_distance = (
        exact_numeric_match(
            rerun_distance,
            reference_distance,
        )
    )

    exact_time = (
        exact_numeric_match(
            rerun_time,
            reference_time,
        )
    )

    exact_sla = (
        exact_numeric_match(
            rerun_sla,
            reference_sla,
        )
    )

    exact_late = bool(
        rerun_late
        == reference_late
    )

    exact_switches = bool(
        rerun_switches
        == reference_switches
    )

    exact_switch_rate = (
        exact_numeric_match(
            rerun_switch_rate,
            reference_switch_rate,
        )
    )

    exact_route = bool(
        delivery_order
        == reference_delivery_order
    )

    exact_core_match = bool(
        exact_distance
        and exact_time
        and exact_sla
        and exact_late
        and exact_switches
        and exact_switch_rate
    )

    # --------------------------------------------------------
    # Practical comparisons
    # --------------------------------------------------------

    distance_abs_pct = abs(
        distance_difference_pct
    )

    time_abs_pct = abs(
        time_difference_pct
    )

    within_1pct = bool(
        distance_abs_pct
        <= PRACTICAL_TOLERANCE_PCT
        and
        time_abs_pct
        <= PRACTICAL_TOLERANCE_PCT
        and
        exact_sla
        and
        exact_late
    )

    within_5pct = bool(
        distance_abs_pct
        <= SECONDARY_TOLERANCE_PCT
        and
        time_abs_pct
        <= SECONDARY_TOLERANCE_PCT
        and
        exact_sla
        and
        exact_late
    )

    return {
        "workload_id":
            workload_id,

        "number_of_deliveries":
            number_of_deliveries,

        "cluster_switch_penalty":
            GUIDED_LAMBDA,

        "seed":
            RANDOM_SEED,

        "training_episodes":
            TRAINING_EPISODES,

        "reference_distance_km":
            reference_distance,

        "rerun_distance_km":
            rerun_distance,

        "distance_difference_km":
            (
                rerun_distance
                - reference_distance
            ),

        "distance_difference_pct":
            distance_difference_pct,

        "reference_travel_time_minutes":
            reference_time,

        "rerun_travel_time_minutes":
            rerun_time,

        "travel_time_difference_minutes":
            (
                rerun_time
                - reference_time
            ),

        "travel_time_difference_pct":
            time_difference_pct,

        "reference_on_time_rate":
            reference_sla,

        "rerun_on_time_rate":
            rerun_sla,

        "reference_late_deliveries":
            reference_late,

        "rerun_late_deliveries":
            rerun_late,

        "reference_cluster_switches":
            reference_switches,

        "rerun_cluster_switches":
            rerun_switches,

        "reference_cluster_switch_rate":
            reference_switch_rate,

        "rerun_cluster_switch_rate":
            rerun_switch_rate,

        "cluster_switch_rate_difference_pct":
            switch_rate_difference_pct,

        "reference_delivery_order":
            " -> ".join(
                reference_delivery_order
            ),

        "rerun_delivery_order":
            " -> ".join(
                delivery_order
            ),

        "rerun_cluster_sequence":
            " -> ".join(
                str(value)
                for value
                in rerun_cluster_sequence
            ),

        "exact_distance_match":
            exact_distance,

        "exact_travel_time_match":
            exact_time,

        "exact_sla_match":
            exact_sla,

        "exact_late_delivery_match":
            exact_late,

        "exact_cluster_switch_match":
            exact_switches,

        "exact_cluster_switch_rate_match":
            exact_switch_rate,

        "exact_route_match":
            exact_route,

        "exact_core_match":
            exact_core_match,

        "within_1pct_core_metrics":
            within_1pct,

        "within_5pct_core_metrics":
            within_5pct,

        "matrix_preparation_seconds":
            float(
                matrix_runtime
            ),

        "training_runtime_seconds":
            float(
                training_runtime
            ),

        "route_runtime_seconds":
            float(
                route_runtime
            ),
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "workloads":
                    int(
                        len(results)
                    ),

                "deliveries":
                    int(
                        results[
                            "number_of_deliveries"
                        ]
                        .sum()
                    ),

                "exact_core_matches":
                    int(
                        results[
                            "exact_core_match"
                        ]
                        .sum()
                    ),

                "exact_core_match_rate":
                    float(
                        results[
                            "exact_core_match"
                        ]
                        .mean()
                    ),

                "exact_route_matches":
                    int(
                        results[
                            "exact_route_match"
                        ]
                        .sum()
                    ),

                "exact_route_match_rate":
                    float(
                        results[
                            "exact_route_match"
                        ]
                        .mean()
                    ),

                "within_1pct_matches":
                    int(
                        results[
                            "within_1pct_core_metrics"
                        ]
                        .sum()
                    ),

                "within_1pct_match_rate":
                    float(
                        results[
                            "within_1pct_core_metrics"
                        ]
                        .mean()
                    ),

                "within_5pct_matches":
                    int(
                        results[
                            "within_5pct_core_metrics"
                        ]
                        .sum()
                    ),

                "within_5pct_match_rate":
                    float(
                        results[
                            "within_5pct_core_metrics"
                        ]
                        .mean()
                    ),

                "mean_abs_distance_difference_pct":
                    float(
                        results[
                            "distance_difference_pct"
                        ]
                        .abs()
                        .mean()
                    ),

                "max_abs_distance_difference_pct":
                    float(
                        results[
                            "distance_difference_pct"
                        ]
                        .abs()
                        .max()
                    ),

                "mean_abs_travel_time_difference_pct":
                    float(
                        results[
                            "travel_time_difference_pct"
                        ]
                        .abs()
                        .mean()
                    ),

                "max_abs_travel_time_difference_pct":
                    float(
                        results[
                            "travel_time_difference_pct"
                        ]
                        .abs()
                        .max()
                    ),

                "mean_training_runtime_seconds":
                    float(
                        results[
                            "training_runtime_seconds"
                        ]
                        .mean()
                    ),

                "cluster_switch_penalty":
                    GUIDED_LAMBDA,

                "seed":
                    RANDOM_SEED,

                "training_episodes":
                    TRAINING_EPISODES,
            }
        ]
    )


# ============================================================
# FINDINGS
# ============================================================

def write_findings(
    results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    row = (
        summary.iloc[
            0
        ]
    )

    lines = [
        "K-MEANS-GUIDED DQN REPRODUCIBILITY VALIDATION",
        "=" * 72,
        "",
        "Scope:",
        (
            f"- Workloads: {WORKLOAD_IDS}"
        ),
        (
            "- Workload sizes: N=5, N=10, N=13"
        ),
        (
            f"- Cluster-switch penalty: "
            f"{GUIDED_LAMBDA:.2f}"
        ),
        (
            f"- Random seed: "
            f"{RANDOM_SEED}"
        ),
        (
            f"- Training episodes: "
            f"{TRAINING_EPISODES}"
        ),
        "",
        "Summary:",
        (
            "- Exact core matches: "
            f"{int(row['exact_core_matches'])}/"
            f"{int(row['workloads'])}"
        ),
        (
            "- Exact route matches: "
            f"{int(row['exact_route_matches'])}/"
            f"{int(row['workloads'])}"
        ),
        (
            "- Within 1% core metrics: "
            f"{int(row['within_1pct_matches'])}/"
            f"{int(row['workloads'])}"
        ),
        (
            "- Within 5% core metrics: "
            f"{int(row['within_5pct_matches'])}/"
            f"{int(row['workloads'])}"
        ),
        (
            "- Mean absolute distance difference: "
            f"{row['mean_abs_distance_difference_pct']:.6f}%"
        ),
        (
            "- Mean absolute travel-time difference: "
            f"{row['mean_abs_travel_time_difference_pct']:.6f}%"
        ),
        "",
        "Per workload:",
    ]

    for result in (
        results.itertuples(
            index=False
        )
    ):
        lines.append(
            (
                f"- workload {result.workload_id} "
                f"(N={result.number_of_deliveries}): "
                f"distance delta="
                f"{result.distance_difference_pct:+.6f}%; "
                f"time delta="
                f"{result.travel_time_difference_pct:+.6f}%; "
                f"exact core="
                f"{result.exact_core_match}; "
                f"exact route="
                f"{result.exact_route_match}."
            )
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            (
                "- Guided DQN was independently re-trained "
                "on three representative workloads that were "
                "newly trained in the original final validation."
            ),
            (
                "- The same workload definitions, road graph, "
                "cluster assignments, reward configuration, "
                "lambda, random seed and training episode count "
                "were reused."
            ),
            (
                "- Exact equality is reported separately from "
                "practical 1% and 5% metric agreement because "
                "GPU-backed neural-network training may exhibit "
                "numerical nondeterminism."
            ),
            (
                "- This controlled validation does not claim "
                "that all 12 final-validation workloads were "
                "retrained."
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
# MEMORY CLEANUP
# ============================================================

def cleanup_memory(
    matrix_builder: RoadMatrixBuilder,
) -> None:
    matrix_builder.clear_cache()

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except ImportError:
        pass


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("=" * 72)
    print(
        "K-MEANS-GUIDED DQN REPRODUCIBILITY"
    )
    print("=" * 72)

    validate_configuration()

    print(
        f"Workloads: {WORKLOAD_IDS}"
    )

    print(
        f"Lambda: {GUIDED_LAMBDA}"
    )

    print(
        f"Episodes: {TRAINING_EPISODES}"
    )

    print(
        f"Seed: {RANDOM_SEED}"
    )

    print(
        "Resume mode: ENABLED"
    )

    # --------------------------------------------------------
    # Exact final-validation workload pipeline
    # --------------------------------------------------------

    final_workloads = (
        load_final_workloads()
    )

    selected_display = (
        final_workloads[
            final_workloads[
                "workload_id"
            ]
            .isin(
                WORKLOAD_IDS
            )
        ]
        .copy()
    )

    print()
    print(
        "Selected final-validation workloads:"
    )

    display_columns = [
        column
        for column
        in [
            "workload_id",
            "number_of_deliveries",
            "selection_role",
            "manifest_delivery_ids",
        ]
        if column
        in selected_display.columns
    ]

    print(
        selected_display[
            display_columns
        ]
        .sort_values(
            "number_of_deliveries"
        )
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Experiment data
    # --------------------------------------------------------

    (
        prepared_dataframe,
        _delivery_lookup,
    ) = (
        guided_experiment
        .prepare_experiment_dataframe()
    )

    cluster_assignments = (
        guided_experiment
        .load_cluster_assignments()
    )

    reference_results = (
        load_reference_results()
    )

    results = (
        load_existing_results()
    )

    if results.empty:
        completed = set()

    else:
        completed = set(
            results[
                "workload_id"
            ]
            .astype(
                int
            )
            .tolist()
        )

    # --------------------------------------------------------
    # Shared routing services
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Three reproducibility runs
    # --------------------------------------------------------

    for position, workload_id in enumerate(
        WORKLOAD_IDS,
        start=1,
    ):
        print()
        print(
            "-" * 72
        )

        print(
            f"Run {position}/"
            f"{len(WORKLOAD_IDS)}"
        )

        print(
            f"workload_id="
            f"{workload_id}"
        )

        if (
            workload_id
            in completed
        ):
            print(
                "Already completed -> SKIP"
            )

            continue

        result = (
            run_one_workload(
                workload_id=(
                    workload_id
                ),
                final_workloads=(
                    final_workloads
                ),
                prepared_dataframe=(
                    prepared_dataframe
                ),
                cluster_assignments=(
                    cluster_assignments
                ),
                reference_results=(
                    reference_results
                ),
                matrix_builder=(
                    matrix_builder
                ),
                evaluator=(
                    evaluator
                ),
                depot_node=(
                    depot.road_node
                ),
            )
        )

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    [
                        result
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
            workload_id
        )

        print()
        print(
            f"N="
            f"{result['number_of_deliveries']}"
        )

        print(
            "Reference distance : "
            f"{result['reference_distance_km']:.4f} km"
        )

        print(
            "Rerun distance     : "
            f"{result['rerun_distance_km']:.4f} km"
        )

        print(
            "Distance delta     : "
            f"{result['distance_difference_pct']:+.6f}%"
        )

        print(
            "Reference time     : "
            f"{result['reference_travel_time_minutes']:.4f} min"
        )

        print(
            "Rerun time         : "
            f"{result['rerun_travel_time_minutes']:.4f} min"
        )

        print(
            "Time delta         : "
            f"{result['travel_time_difference_pct']:+.6f}%"
        )

        print(
            "Reference switches : "
            f"{result['reference_cluster_switches']}"
        )

        print(
            "Rerun switches     : "
            f"{result['rerun_cluster_switches']}"
        )

        print(
            "Exact core match   : "
            f"{result['exact_core_match']}"
        )

        print(
            "Exact route match  : "
            f"{result['exact_route_match']}"
        )

        print(
            "Within 1%          : "
            f"{result['within_1pct_core_metrics']}"
        )

        print(
            "Within 5%          : "
            f"{result['within_5pct_core_metrics']}"
        )

        print(
            "Training runtime   : "
            f"{result['training_runtime_seconds']:.2f}s"
        )

        cleanup_memory(
            matrix_builder
        )

    # --------------------------------------------------------
    # Verify complete result set
    # --------------------------------------------------------

    expected_ids = set(
        WORKLOAD_IDS
    )

    produced_ids = set(
        results[
            "workload_id"
        ]
        .astype(
            int
        )
        .tolist()
    )

    if (
        produced_ids
        != expected_ids
    ):
        raise RuntimeError(
            "Reproducibility output does not "
            "contain exactly the expected workloads.\n"
            f"Expected: {sorted(expected_ids)}\n"
            f"Produced: {sorted(produced_ids)}"
        )

    if (
        len(
            results
        )
        != len(
            WORKLOAD_IDS
        )
    ):
        raise RuntimeError(
            "Unexpected number of output rows: "
            f"{len(results)}"
        )

    results = (
        results
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
        results=results,
        summary=summary,
    )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "GUIDED DQN REPRODUCIBILITY SUMMARY"
    )
    print("=" * 72)

    result_columns = [
        "workload_id",
        "number_of_deliveries",
        "distance_difference_pct",
        "travel_time_difference_pct",
        "reference_cluster_switch_rate",
        "rerun_cluster_switch_rate",
        "exact_core_match",
        "exact_route_match",
        "within_1pct_core_metrics",
        "within_5pct_core_metrics",
    ]

    print(
        results[
            result_columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "OVERALL:"
    )

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
        "Guided-DQN reproducibility "
        "validation complete."
    )
    print()


if __name__ == "__main__":
    main()