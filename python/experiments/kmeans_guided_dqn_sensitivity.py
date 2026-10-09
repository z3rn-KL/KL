from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from clustering.cluster_alignment import align_cluster_ids
from evaluation import RouteEvaluator

from rl import (
    DQNAgent,
    DQNConfig,
)

from rl.routing_environment import (
    RewardConfig,
    RoutingEnvironment,
)

from routing import RoadMatrixBuilder

from experiments.q_learning_single_workload_experiment import (
    build_workload,
    get_depot_node,
    load_delivery_dataframe,
    load_road_network,
    load_snapped_nodes,
    prepare_dataframe,
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

CLUSTER_PATH = (
    RESULTS_DIR
    / "xedu_kmeans_clusters.csv"
)

DETAIL_OUTPUT_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_results.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_summary.csv"
)

PAIRWISE_OUTPUT_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_pairwise.csv"
)

WORKLOAD_OUTPUT_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_workloads.csv"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

WORKLOAD_SIZES = [
    5,
    6,
    8,
    10,
    12,
    13,
]

CLUSTER_SWITCH_PENALTIES = [
    0.00,
    0.10,
    0.25,
    0.50,
]

TRAINING_EPISODES = 5000

RANDOM_SEED = 42

DQN_LEARNING_RATE = 0.001

DISCOUNT_FACTOR = 0.95

EPSILON_START = 1.00

EPSILON_MIN = 0.05

EPSILON_DECAY = 0.998


# ============================================================
# DQN CONFIG
# ============================================================

def make_dqn_config() -> DQNConfig:
    return DQNConfig(
        learning_rate=(
            DQN_LEARNING_RATE
        ),
        discount_factor=(
            DISCOUNT_FACTOR
        ),
        epsilon_start=(
            EPSILON_START
        ),
        epsilon_min=(
            EPSILON_MIN
        ),
        epsilon_decay=(
            EPSILON_DECAY
        ),
        episodes=(
            TRAINING_EPISODES
        ),
        batch_size=64,
        replay_capacity=10000,
        min_replay_size=128,
        target_update_interval=100,
        hidden_dim=128,
        gradient_clip_norm=5.0,
        elapsed_time_bucket_scale=36.0,
        random_seed=(
            RANDOM_SEED
        ),
        device="auto",
    )


# ============================================================
# LOAD K-MEANS
# ============================================================

def load_cluster_assignments() -> pd.DataFrame:
    if not CLUSTER_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: "
            f"{CLUSTER_PATH}"
        )

    dataframe = pd.read_csv(
        CLUSTER_PATH
    )

    if "delivery_id" not in dataframe.columns:
        if "id" in dataframe.columns:
            dataframe = (
                dataframe.rename(
                    columns={
                        "id": "delivery_id"
                    }
                )
            )
        else:
            raise ValueError(
                "K-Means CSV không có "
                "id hoặc delivery_id."
            )

    required = [
        "delivery_id",
        "cluster_id",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "K-Means CSV thiếu: "
            + ", ".join(missing)
        )

    result = (
        dataframe[
            [
                "delivery_id",
                "cluster_id",
            ]
        ]
        .copy()
    )

    result[
        "delivery_id"
    ] = (
        result[
            "delivery_id"
        ]
        .astype(str)
        .str.strip()
    )

    labels = pd.to_numeric(result["cluster_id"], errors="raise")
    if labels.isna().any() or not np.isfinite(labels.to_numpy(dtype=float)).all():
        raise ValueError("K-Means CSV contains missing or non-finite cluster IDs")
    if not (labels == np.floor(labels)).all():
        raise ValueError("K-Means CSV contains non-integer cluster IDs")
    result["cluster_id"] = labels.astype(int)

    if result[
        "delivery_id"
    ].duplicated().any():
        raise ValueError(
            "K-Means CSV có delivery ID trùng."
        )

    return result


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_experiment_dataframe(
) -> tuple[
    pd.DataFrame,
    dict,
]:
    (
        delivery_df,
        delivery_lookup,
    ) = load_delivery_dataframe()

    snapped_df = (
        load_snapped_nodes()
    )

    dataframe = prepare_dataframe(
        delivery_df=delivery_df,
        snapped_df=snapped_df,
    )

    dataframe[
        "delivery_id"
    ] = (
        dataframe[
            "delivery_id"
        ]
        .astype(str)
        .str.strip()
    )

    clusters = (
        load_cluster_assignments()
    )

    dataframe = dataframe.merge(
        clusters,
        on="delivery_id",
        how="left",
        validate="one_to_one",
    )

    missing_clusters = int(
        dataframe[
            "cluster_id"
        ].isna().sum()
    )

    if missing_clusters != 0:
        raise ValueError(
            f"{missing_clusters} deliveries "
            "không có cluster_id."
        )

    dataframe[
        "cluster_id"
    ] = dataframe[
        "cluster_id"
    ].astype(int)

    return (
        dataframe,
        delivery_lookup,
    )


# ============================================================
# SELECT REPRESENTATIVE WORKLOADS
# ============================================================

def select_representative_workloads(
    dataframe: pd.DataFrame,
) -> list[pd.DataFrame]:
    """
    Select one deterministic representative workload
    for each requested workload size.

    We use the chronological median workload for each
    size instead of the first workload, which avoids
    selecting only an early part of the dataset.
    """

    selected = []

    for workload_size in WORKLOAD_SIZES:
        candidates = []

        for (
            wave_start,
            group,
        ) in dataframe.groupby(
            "wave_start"
        ):
            if len(group) != workload_size:
                continue

            group = (
                group
                .sort_values(
                    "delivery_id"
                )
                .copy()
            )

            candidates.append(
                (
                    wave_start,
                    group,
                )
            )

        candidates.sort(
            key=lambda item: item[0]
        )

        if not candidates:
            raise ValueError(
                "Không tìm thấy workload "
                f"N={workload_size}."
            )

        median_index = (
            len(candidates) // 2
        )

        selected_group = (
            candidates[
                median_index
            ][1]
        )

        selected.append(
            selected_group
        )

    return selected


# ============================================================
# CLUSTER METRICS
# ============================================================

def get_cluster_ids_for_deliveries(
    deliveries,
    workload_df: pd.DataFrame,
) -> list[int]:
    ids = workload_df["delivery_id"].astype(str).str.strip()
    if ids.duplicated().any():
        raise ValueError("Duplicate delivery IDs in clustering workload")
    if workload_df["cluster_id"].isna().any():
        raise ValueError("Missing cluster ID in clustering workload")
    labels = pd.to_numeric(workload_df["cluster_id"], errors="raise")
    if not np.isfinite(labels.to_numpy(dtype=float)).all() or not (labels == np.floor(labels)).all():
        raise ValueError("Invalid cluster IDs in workload")
    lookup = dict(zip(ids, labels.astype(int)))
    return align_cluster_ids(deliveries, lookup)


def calculate_cluster_statistics(
    cluster_ids: list[int],
) -> dict:
    counts = (
        pd.Series(
            cluster_ids
        )
        .value_counts()
    )

    number_of_clusters = int(
        len(counts)
    )

    dominant_cluster_size = int(
        counts.max()
    )

    dominant_cluster_share = (
        dominant_cluster_size
        / len(cluster_ids)
    )

    return {
        "number_of_clusters":
            number_of_clusters,

        "dominant_cluster_size":
            dominant_cluster_size,

        "dominant_cluster_share":
            dominant_cluster_share,
    }


def calculate_route_cluster_metrics(
    delivery_order,
    delivery_cluster_lookup: dict[str, int],
) -> tuple[
    int,
    float,
    str,
]:
    """
    Calculate cluster switches from the final greedy route.

    Depot legs are excluded.
    """

    cluster_sequence = []

    for delivery_id in delivery_order:
        key = str(
            delivery_id
        )

        if key not in delivery_cluster_lookup:
            raise ValueError(
                "Không tìm thấy cluster của "
                f"delivery {key}."
            )

        cluster_sequence.append(
            int(
                delivery_cluster_lookup[
                    key
                ]
            )
        )

    switches = 0

    for index in range(
        1,
        len(cluster_sequence),
    ):
        if (
            cluster_sequence[index]
            != cluster_sequence[
                index - 1
            ]
        ):
            switches += 1

    transitions = max(
        0,
        len(cluster_sequence) - 1,
    )

    if transitions == 0:
        switch_rate = 0.0
    else:
        switch_rate = (
            switches
            / transitions
        )

    sequence_text = " -> ".join(
        str(cluster_id)
        for cluster_id
        in cluster_sequence
    )

    return (
        switches,
        switch_rate,
        sequence_text,
    )


# ============================================================
# DQN RUN
# ============================================================

def run_dqn(
    deliveries,
    road_matrix,
    start_time,
    cluster_ids: list[int],
    cluster_switch_penalty: float,
):
    reward_config = RewardConfig(
        cluster_switch_penalty=(
            cluster_switch_penalty
        )
    )

    environment = RoutingEnvironment(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=start_time,
        reward_config=reward_config,
        delivery_cluster_ids=(
            cluster_ids
        ),
    )

    agent = DQNAgent(
        delivery_count=len(
            deliveries
        ),
        config=make_dqn_config(),
    )

    timer = perf_counter()

    training = agent.train(
        environment
    )

    training_runtime = (
        perf_counter()
        - timer
    )

    timer = perf_counter()

    rollout = agent.greedy_rollout(
        environment
    )

    route_runtime = (
        perf_counter()
        - timer
    )

    return (
        rollout,
        training,
        training_runtime,
        route_runtime,
        environment,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_route(
    evaluator,
    depot_node: int,
    deliveries,
    delivery_nodes,
    delivery_order,
    start_time,
):
    return evaluator.evaluate(
        algorithm="dqn",
        depot_node=depot_node,
        deliveries=deliveries,
        delivery_nodes=delivery_nodes,
        delivery_order=delivery_order,
        start_time=start_time,
        return_to_depot=True,
    )


# ============================================================
# RESUME
# ============================================================

def load_existing_results() -> pd.DataFrame:
    if not DETAIL_OUTPUT_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        DETAIL_OUTPUT_PATH
    )

    print(
        "Existing sensitivity rows:",
        len(dataframe),
    )

    return dataframe


def completed_keys(
    dataframe: pd.DataFrame,
) -> set[tuple]:
    if dataframe.empty:
        return set()

    keys = set()

    for _, row in dataframe.iterrows():
        keys.add(
            (
                int(
                    row[
                        "workload_size"
                    ]
                ),
                str(
                    row[
                        "wave_start"
                    ]
                ),
                round(
                    float(
                        row[
                            "cluster_switch_penalty"
                        ]
                    ),
                    6,
                ),
            )
        )

    return keys


def append_result(
    row: dict,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_df = pd.DataFrame(
        [row]
    )

    if DETAIL_OUTPUT_PATH.exists():
        row_df.to_csv(
            DETAIL_OUTPUT_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        row_df.to_csv(
            DETAIL_OUTPUT_PATH,
            index=False,
        )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    detail_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        penalty,
        group,
    ) in detail_df.groupby(
        "cluster_switch_penalty"
    ):
        rows.append(
            {
                "cluster_switch_penalty":
                    float(penalty),

                "runs":
                    len(group),

                "mean_distance_km":
                    group[
                        "total_distance_km"
                    ].mean(),

                "mean_travel_time_minutes":
                    group[
                        "total_travel_time_minutes"
                    ].mean(),

                "overall_on_time_rate":
                    (
                        group[
                            "on_time_deliveries"
                        ].sum()
                        / group[
                            "number_of_deliveries"
                        ].sum()
                    ),

                "total_late_deliveries":
                    int(
                        group[
                            "late_deliveries"
                        ].sum()
                    ),

                "mean_cluster_switches":
                    group[
                        "cluster_switches"
                    ].mean(),

                "mean_cluster_switch_rate":
                    group[
                        "cluster_switch_rate"
                    ].mean(),

                "mean_training_runtime_seconds":
                    group[
                        "training_runtime_seconds"
                    ].mean(),

                "mean_route_runtime_seconds":
                    group[
                        "route_runtime_seconds"
                    ].mean(),

                "mean_greedy_reward":
                    group[
                        "greedy_reward"
                    ].mean(),

                "mean_reward_per_delivery":
                    group[
                        "reward_per_delivery"
                    ].mean(),

                "mean_model_size":
                    group[
                        "model_size"
                    ].mean(),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "cluster_switch_penalty"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PAIRED COMPARISON AGAINST LAMBDA = 0
# ============================================================

def build_pairwise_summary(
    detail_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline = detail_df[
        np.isclose(
            detail_df[
                "cluster_switch_penalty"
            ],
            0.0,
        )
    ].copy()

    baseline = baseline[
        [
            "workload_size",
            "wave_start",
            "total_distance_km",
            "total_travel_time_minutes",
            "on_time_rate",
            "cluster_switches",
            "cluster_switch_rate",
        ]
    ].rename(
        columns={
            "total_distance_km":
                "baseline_distance_km",

            "total_travel_time_minutes":
                "baseline_travel_time_minutes",

            "on_time_rate":
                "baseline_on_time_rate",

            "cluster_switches":
                "baseline_cluster_switches",

            "cluster_switch_rate":
                "baseline_cluster_switch_rate",
        }
    )

    rows = []

    penalties = sorted(
        detail_df[
            "cluster_switch_penalty"
        ].unique()
    )

    for penalty in penalties:
        if np.isclose(
            penalty,
            0.0,
        ):
            continue

        candidate = detail_df[
            np.isclose(
                detail_df[
                    "cluster_switch_penalty"
                ],
                penalty,
            )
        ].copy()

        merged = candidate.merge(
            baseline,
            on=[
                "workload_size",
                "wave_start",
            ],
            how="inner",
            validate="one_to_one",
        )

        if merged.empty:
            continue

        merged[
            "distance_improvement_pct"
        ] = (
            (
                merged[
                    "baseline_distance_km"
                ]
                - merged[
                    "total_distance_km"
                ]
            )
            / merged[
                "baseline_distance_km"
            ]
            * 100.0
        )

        merged[
            "time_improvement_pct"
        ] = (
            (
                merged[
                    "baseline_travel_time_minutes"
                ]
                - merged[
                    "total_travel_time_minutes"
                ]
            )
            / merged[
                "baseline_travel_time_minutes"
            ]
            * 100.0
        )

        merged[
            "cluster_switch_reduction"
        ] = (
            merged[
                "baseline_cluster_switches"
            ]
            - merged[
                "cluster_switches"
            ]
        )

        merged[
            "cluster_switch_rate_reduction"
        ] = (
            merged[
                "baseline_cluster_switch_rate"
            ]
            - merged[
                "cluster_switch_rate"
            ]
        )

        rows.append(
            {
                "cluster_switch_penalty":
                    float(penalty),

                "workloads":
                    len(merged),

                "mean_distance_improvement_pct":
                    merged[
                        "distance_improvement_pct"
                    ].mean(),

                "distance_win_rate":
                    (
                        merged[
                            "distance_improvement_pct"
                        ]
                        > 0
                    ).mean(),

                "mean_time_improvement_pct":
                    merged[
                        "time_improvement_pct"
                    ].mean(),

                "time_win_rate":
                    (
                        merged[
                            "time_improvement_pct"
                        ]
                        > 0
                    ).mean(),

                "mean_cluster_switch_reduction":
                    merged[
                        "cluster_switch_reduction"
                    ].mean(),

                "mean_cluster_switch_rate_reduction":
                    merged[
                        "cluster_switch_rate_reduction"
                    ].mean(),

                "mean_on_time_rate_change":
                    (
                        merged[
                            "on_time_rate"
                        ]
                        - merged[
                            "baseline_on_time_rate"
                        ]
                    ).mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# WORKLOAD MANIFEST
# ============================================================

def save_workload_manifest(
    workloads: list[pd.DataFrame],
) -> None:
    rows = []

    for workload_df in workloads:
        wave_start = (
            workload_df[
                "wave_start"
            ].iloc[0]
        )

        wave_end = (
            workload_df[
                "wave_end"
            ].iloc[0]
        )

        cluster_counts = (
            workload_df[
                "cluster_id"
            ]
            .value_counts()
            .sort_index()
        )

        rows.append(
            {
                "workload_size":
                    len(workload_df),

                "wave_start":
                    wave_start,

                "wave_end":
                    wave_end,

                "number_of_clusters":
                    workload_df[
                        "cluster_id"
                    ].nunique(),

                "dominant_cluster_share":
                    (
                        cluster_counts.max()
                        / len(workload_df)
                    ),

                "cluster_distribution":
                    "; ".join(
                        f"{int(cluster_id)}:"
                        f"{int(count)}"
                        for (
                            cluster_id,
                            count,
                        )
                        in cluster_counts.items()
                    ),

                "delivery_ids":
                    " -> ".join(
                        workload_df[
                            "delivery_id"
                        ].astype(str)
                    ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        WORKLOAD_OUTPUT_PATH,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print(
        "=" * 72
    )
    print(
        "K-MEANS-GUIDED DQN SENSITIVITY"
    )
    print(
        "=" * 72
    )

    print(
        "Workload sizes:",
        WORKLOAD_SIZES,
    )

    print(
        "Cluster penalties:",
        CLUSTER_SWITCH_PENALTIES,
    )

    print(
        "Training episodes:",
        TRAINING_EPISODES,
    )

    print(
        "Random seed:",
        RANDOM_SEED,
    )

    print(
        "Resume mode: ENABLED"
    )

    service = (
        load_road_network()
    )

    (
        dataframe,
        delivery_lookup,
    ) = (
        prepare_experiment_dataframe()
    )

    workloads = (
        select_representative_workloads(
            dataframe
        )
    )

    save_workload_manifest(
        workloads
    )

    (
        _,
        depot_node,
    ) = get_depot_node(
        service
    )

    matrix_builder = (
        RoadMatrixBuilder(
            service
        )
    )

    evaluator = RouteEvaluator(
        road_network=service
    )

    existing = (
        load_existing_results()
    )

    completed = (
        completed_keys(
            existing
        )
    )

    total_runs = (
        len(workloads)
        * len(
            CLUSTER_SWITCH_PENALTIES
        )
    )

    run_number = 0

    for workload_df in workloads:
        workload_size = len(
            workload_df
        )

        wave_start = (
            workload_df[
                "wave_start"
            ].iloc[0]
        )

        wave_end = (
            workload_df[
                "wave_end"
            ].iloc[0]
        )

        (
            deliveries,
            delivery_nodes,
        ) = build_workload(
            workload_df=workload_df,
            delivery_lookup=(
                delivery_lookup
            ),
        )

        delivery_ids = [
            str(
                delivery.delivery_id
            )
            for delivery
            in deliveries
        ]

        cluster_ids = (
            get_cluster_ids_for_deliveries(
                deliveries=deliveries,
                workload_df=workload_df,
            )
        )

        cluster_stats = (
            calculate_cluster_statistics(
                cluster_ids
            )
        )

        cluster_lookup = dict(
            zip(
                delivery_ids,
                cluster_ids,
            )
        )

        all_nodes = (
            depot_node,
            *delivery_nodes,
        )

        print()
        print(
            "=" * 72
        )

        print(
            f"Preparing workload "
            f"N={workload_size}"
        )

        print(
            f"Wave: {wave_start}"
        )

        print(
            "Clusters:",
            cluster_stats[
                "number_of_clusters"
            ],
        )

        timer = perf_counter()

        road_matrix = (
            matrix_builder.build(
                all_nodes
            )
        )

        matrix_runtime = (
            perf_counter()
            - timer
        )

        print(
            f"Road matrix: "
            f"{matrix_runtime:.3f}s"
        )

        for penalty in (
            CLUSTER_SWITCH_PENALTIES
        ):
            run_number += 1

            key = (
                int(
                    workload_size
                ),
                str(
                    wave_start
                ),
                round(
                    float(
                        penalty
                    ),
                    6,
                ),
            )

            print()
            print(
                "-" * 72
            )

            print(
                f"Run "
                f"{run_number}/{total_runs}"
            )

            print(
                f"N={workload_size}, "
                f"lambda={penalty:.2f}"
            )

            if key in completed:
                print(
                    "Already completed -> skip."
                )
                continue

            (
                rollout,
                training,
                training_runtime,
                route_runtime,
                environment,
            ) = run_dqn(
                deliveries=deliveries,
                road_matrix=road_matrix,
                start_time=wave_end,
                cluster_ids=cluster_ids,
                cluster_switch_penalty=(
                    penalty
                ),
            )

            evaluation = (
                evaluate_route(
                    evaluator=evaluator,
                    depot_node=depot_node,
                    deliveries=deliveries,
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    delivery_order=(
                        rollout.delivery_order
                    ),
                    start_time=wave_end,
                )
            )

            (
                cluster_switches,
                cluster_switch_rate,
                cluster_sequence,
            ) = (
                calculate_route_cluster_metrics(
                    delivery_order=(
                        rollout.delivery_order
                    ),
                    delivery_cluster_lookup=(
                        cluster_lookup
                    ),
                )
            )

            # Cross-check the environment counter.
            if (
                cluster_switches
                != environment.cluster_switches
            ):
                raise ValueError(
                    "Cluster-switch metric mismatch: "
                    f"route={cluster_switches}, "
                    f"environment="
                    f"{environment.cluster_switches}"
                )

            row = {
                "workload_size":
                    workload_size,

                "wave_start":
                    wave_start,

                "wave_end":
                    wave_end,

                "cluster_switch_penalty":
                    penalty,

                "number_of_clusters":
                    cluster_stats[
                        "number_of_clusters"
                    ],

                "dominant_cluster_size":
                    cluster_stats[
                        "dominant_cluster_size"
                    ],

                "dominant_cluster_share":
                    cluster_stats[
                        "dominant_cluster_share"
                    ],

                "number_of_deliveries":
                    evaluation
                    .number_of_deliveries,

                "total_distance_km":
                    evaluation
                    .total_distance_km,

                "total_travel_time_minutes":
                    evaluation
                    .total_travel_time_minutes,

                "on_time_deliveries":
                    evaluation
                    .on_time_deliveries,

                "late_deliveries":
                    evaluation
                    .late_deliveries,

                "on_time_rate":
                    evaluation
                    .on_time_rate,

                "total_lateness_minutes":
                    evaluation
                    .total_lateness_minutes,

                "max_lateness_minutes":
                    evaluation
                    .max_lateness_minutes,

                "training_runtime_seconds":
                    training_runtime,

                "route_runtime_seconds":
                    route_runtime,

                "matrix_preparation_seconds":
                    matrix_runtime,

                "greedy_reward":
                    rollout.total_reward,

                "reward_per_delivery":
                    (
                        rollout.total_reward
                        / workload_size
                    ),

                "model_size":
                    training.parameter_count,

                "final_epsilon":
                    training.final_epsilon,

                "device":
                    training.device,

                "cluster_switches":
                    cluster_switches,

                "cluster_switch_rate":
                    cluster_switch_rate,

                "cluster_sequence":
                    cluster_sequence,

                "delivery_order":
                    " -> ".join(
                        str(delivery_id)
                        for delivery_id
                        in rollout.delivery_order
                    ),
            }

            append_result(
                row
            )

            completed.add(
                key
            )

            print(
                f"distance="
                f"{evaluation.total_distance_km:.4f} km"
            )

            print(
                f"time="
                f"{evaluation.total_travel_time_minutes:.4f} min"
            )

            print(
                f"on_time="
                f"{evaluation.on_time_rate:.4f}"
            )

            print(
                f"cluster_switches="
                f"{cluster_switches}"
            )

            print(
                f"switch_rate="
                f"{cluster_switch_rate:.4f}"
            )

            print(
                f"reward="
                f"{rollout.total_reward:.4f}"
            )

            print(
                f"train="
                f"{training_runtime:.2f}s"
            )

            print(
                f"device="
                f"{training.device}"
            )

            # Help free GPU/CPU memory between DQN runs.
            del environment
            del training
            del rollout

    detail_df = pd.read_csv(
        DETAIL_OUTPUT_PATH
    )

    summary_df = (
        build_summary(
            detail_df
        )
    )

    pairwise_df = (
        build_pairwise_summary(
            detail_df
        )
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    pairwise_df.to_csv(
        PAIRWISE_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "SENSITIVITY SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "=" * 72
    )

    print(
        "PAIRED COMPARISON VS LAMBDA=0"
    )

    print(
        "=" * 72
    )

    if pairwise_df.empty:
        print(
            "No paired results yet."
        )
    else:
        print(
            pairwise_df.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.4f}"
                ),
            )
        )

    print()
    print(
        "Created:"
    )

    for path in [
        DETAIL_OUTPUT_PATH,
        SUMMARY_OUTPUT_PATH,
        PAIRWISE_OUTPUT_PATH,
        WORKLOAD_OUTPUT_PATH,
    ]:
        print(
            "-",
            path,
        )

    print()
    print(
        "Sensitivity experiment complete."
    )


if __name__ == "__main__":
    main()