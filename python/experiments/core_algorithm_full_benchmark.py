import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from evaluation import (
    RouteEvaluator,
)

from rl import (
    DQNAgent,
    DQNConfig,
    QLearningAgent,
    QLearningConfig,
    RoutingEnvironment,
    SarsaAgent,
    SarsaConfig,
)

from routing import (
    NearestNeighborRouter,
    RoadMatrixBuilder,
)

from routing.clarke_wright import (
    ClarkeWrightRouter,
)

from experiments.q_learning_single_workload_experiment import (
    build_result_row,
    build_workload,
    get_depot_node,
    load_delivery_dataframe,
    load_road_network,
    load_snapped_nodes,
    prepare_dataframe,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MASTER_OUTPUT_PATH = (
    RESULTS_DIR
    / "core_algorithm_all_results.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "core_algorithm_final_summary.csv"
)

WORKLOAD_OUTPUT_PATH = (
    RESULTS_DIR
    / "core_algorithm_full_workloads.csv"
)


WORKLOAD_SIZE = 5

TRAINING_EPISODES = 5000

RANDOM_SEED = 42

TABULAR_LEARNING_RATE = 0.20

DQN_LEARNING_RATE = 0.001

DISCOUNT_FACTOR = 0.95

EPSILON_START = 1.00

EPSILON_MIN = 0.05

EPSILON_DECAY = 0.998


ALGORITHMS = (
    "nearest_neighbor",
    "clarke_wright",
    "q_learning",
    "sarsa",
    "dqn",
)


def select_all_workloads(
    dataframe: pd.DataFrame,
) -> list[pd.DataFrame]:
    """
    Select every wave containing exactly
    WORKLOAD_SIZE deliveries.

    Workloads are ordered chronologically,
    so workload_id remains stable when the
    benchmark is resumed.
    """

    workloads = []

    for (
        _,
        group,
    ) in dataframe.groupby(
        "wave_start"
    ):
        if len(
            group
        ) != WORKLOAD_SIZE:
            continue

        group = (
            group
            .sort_values(
                "delivery_id"
            )
            .copy()
        )

        workloads.append(
            group
        )

    workloads.sort(
        key=lambda group: (
            group[
                "wave_start"
            ].iloc[
                0
            ]
        )
    )

    return workloads


def make_q_config(
) -> QLearningConfig:
    return QLearningConfig(
        learning_rate=(
            TABULAR_LEARNING_RATE
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
        random_seed=(
            RANDOM_SEED
        ),
    )


def make_sarsa_config(
) -> SarsaConfig:
    return SarsaConfig(
        learning_rate=(
            TABULAR_LEARNING_RATE
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
        random_seed=(
            RANDOM_SEED
        ),
    )


def make_dqn_config(
) -> DQNConfig:
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


def safe_save_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save through a temporary file first.

    This reduces the risk of losing the
    benchmark CSV if execution is
    interrupted during a write.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = Path(
        str(path)
        + ".tmp"
    )

    dataframe.to_csv(
        temp_path,
        index=False,
    )

    temp_path.replace(
        path
    )


def load_existing_results(
) -> pd.DataFrame:
    if not MASTER_OUTPUT_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        MASTER_OUTPUT_PATH
    )

    print()
    print(
        f"Existing benchmark rows: "
        f"{len(dataframe)}"
    )

    return dataframe


def completed_keys(
    dataframe: pd.DataFrame,
) -> set[tuple[int, str]]:
    if dataframe.empty:
        return set()

    return {
        (
            int(
                row.workload_id
            ),
            str(
                row.algorithm
            ),
        )
        for row
        in dataframe.itertuples()
    }


def append_result(
    results_df: pd.DataFrame,
    row: dict,
) -> pd.DataFrame:
    new_row_df = pd.DataFrame(
        [
            row
        ]
    )

    if results_df.empty:
        result = new_row_df

    else:
        result = pd.concat(
            [
                results_df,
                new_row_df,
            ],
            ignore_index=True,
        )

    algorithm_order = {
        name: index
        for index, name
        in enumerate(
            ALGORITHMS
        )
    }

    result[
        "_algorithm_order"
    ] = (
        result[
            "algorithm"
        ]
        .map(
            algorithm_order
        )
    )

    result = (
        result
        .sort_values(
            [
                "workload_id",
                "_algorithm_order",
            ]
        )
        .drop(
            columns=[
                "_algorithm_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    safe_save_csv(
        result,
        MASTER_OUTPUT_PATH,
    )

    return result


def evaluate(
    evaluator,
    algorithm: str,
    depot_node: int,
    deliveries,
    delivery_nodes,
    delivery_order,
    start_time,
):
    return evaluator.evaluate(
        algorithm=algorithm,
        depot_node=depot_node,
        deliveries=deliveries,
        delivery_nodes=delivery_nodes,
        delivery_order=delivery_order,
        start_time=start_time,
        return_to_depot=True,
    )


def make_result_row(
    workload_id: int,
    wave_start,
    wave_end,
    evaluation,
    route_runtime: float,
    matrix_runtime: (
        float
        | None
    ) = None,
    training_runtime: (
        float
        | None
    ) = None,
    reward: (
        float
        | None
    ) = None,
    model_size: (
        int
        | None
    ) = None,
    final_epsilon: (
        float
        | None
    ) = None,
    device: (
        str
        | None
    ) = None,
) -> dict:
    base = build_result_row(
        evaluation=evaluation,
        runtime_seconds=(
            route_runtime
        ),
    )

    return {
        "scope":
            "all_exact_5",

        "workload_id":
            workload_id,

        "wave_start":
            wave_start,

        "wave_end":
            wave_end,

        "algorithm":
            base[
                "algorithm"
            ],

        "number_of_deliveries":
            base[
                "number_of_deliveries"
            ],

        "on_time_deliveries":
            base[
                "on_time_deliveries"
            ],

        "late_deliveries":
            base[
                "late_deliveries"
            ],

        "on_time_rate":
            base[
                "on_time_rate"
            ],

        "total_lateness_minutes":
            base[
                "total_lateness_minutes"
            ],

        "max_lateness_minutes":
            base[
                "max_lateness_minutes"
            ],

        "total_distance_km":
            base[
                "total_distance_km"
            ],

        "total_travel_time_minutes":
            base[
                "total_travel_time_minutes"
            ],

        "route_runtime_seconds":
            route_runtime,

        "matrix_preparation_seconds":
            matrix_runtime,

        "training_runtime_seconds":
            training_runtime,

        "greedy_reward":
            reward,

        "model_size":
            model_size,

        "final_epsilon":
            final_epsilon,

        "device":
            device,
    }


def run_q_learning(
    deliveries,
    road_matrix,
    start_time,
):
    environment = RoutingEnvironment(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=start_time,
    )

    agent = QLearningAgent(
        make_q_config()
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
    )


def run_sarsa(
    deliveries,
    road_matrix,
    start_time,
):
    environment = RoutingEnvironment(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=start_time,
    )

    agent = SarsaAgent(
        make_sarsa_config()
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
    )


def run_dqn(
    deliveries,
    road_matrix,
    start_time,
):
    environment = RoutingEnvironment(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=start_time,
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
    )


def build_final_summary(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for algorithm in (
        ALGORITHMS
    ):
        group = (
            results_df[
                results_df[
                    "algorithm"
                ]
                == algorithm
            ]
            .copy()
        )

        if group.empty:
            continue

        rows.append(
            {
                "algorithm":
                    algorithm,

                "workloads":
                    len(
                        group
                    ),

                "deliveries":
                    int(
                        group[
                            "number_of_deliveries"
                        ].sum()
                    ),

                "mean_distance_km":
                    group[
                        "total_distance_km"
                    ].mean(),

                "std_distance_km":
                    group[
                        "total_distance_km"
                    ].std(),

                "median_distance_km":
                    group[
                        "total_distance_km"
                    ].median(),

                "mean_travel_time_minutes":
                    group[
                        "total_travel_time_minutes"
                    ].mean(),

                "std_travel_time_minutes":
                    group[
                        "total_travel_time_minutes"
                    ].std(),

                "mean_on_time_rate":
                    group[
                        "on_time_rate"
                    ].mean(),

                "min_on_time_rate":
                    group[
                        "on_time_rate"
                    ].min(),

                "total_late_deliveries":
                    int(
                        group[
                            "late_deliveries"
                        ].sum()
                    ),

                "total_lateness_minutes":
                    group[
                        "total_lateness_minutes"
                    ].sum(),

                "max_lateness_minutes":
                    group[
                        "max_lateness_minutes"
                    ].max(),

                "mean_route_runtime_seconds":
                    group[
                        "route_runtime_seconds"
                    ].mean(),

                "mean_matrix_preparation_seconds":
                    group[
                        "matrix_preparation_seconds"
                    ].mean(),

                "mean_training_runtime_seconds":
                    group[
                        "training_runtime_seconds"
                    ].mean(),

                "total_training_runtime_seconds":
                    group[
                        "training_runtime_seconds"
                    ].sum(),

                "mean_greedy_reward":
                    group[
                        "greedy_reward"
                    ].mean(),

                "std_greedy_reward":
                    group[
                        "greedy_reward"
                    ].std(),

                "mean_model_size":
                    group[
                        "model_size"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def save_workload_manifest(
    workloads,
    delivery_lookup,
) -> None:
    rows = []

    for workload_id, workload_df in enumerate(
        workloads,
        start=1,
    ):
        deliveries, _ = build_workload(
            workload_df=workload_df,
            delivery_lookup=(
                delivery_lookup
            ),
        )

        rows.append(
            {
                "workload_id":
                    workload_id,

                "wave_start":
                    workload_df[
                        "wave_start"
                    ].iloc[
                        0
                    ],

                "wave_end":
                    workload_df[
                        "wave_end"
                    ].iloc[
                        0
                    ],

                "number_of_deliveries":
                    len(
                        deliveries
                    ),

                "delivery_ids":
                    " -> ".join(
                        str(
                            delivery.delivery_id
                        )
                        for delivery
                        in deliveries
                    ),
            }
        )

    safe_save_csv(
        pd.DataFrame(
            rows
        ),
        WORKLOAD_OUTPUT_PATH,
    )


def main():
    print()
    print(
        "========================================"
    )

    print(
        "Core Algorithm Full Benchmark"
    )

    print(
        "========================================"
    )

    print(
        "Resume mode: ENABLED"
    )

    service = (
        load_road_network()
    )

    (
        delivery_df,
        delivery_lookup,
    ) = (
        load_delivery_dataframe()
    )

    snapped_df = (
        load_snapped_nodes()
    )

    dataframe = (
        prepare_dataframe(
            delivery_df=delivery_df,
            snapped_df=snapped_df,
        )
    )

    workloads = (
        select_all_workloads(
            dataframe
        )
    )

    print()
    print(
        f"Exact-{WORKLOAD_SIZE} "
        f"workloads found: "
        f"{len(workloads)}"
    )

    print(
        f"Algorithms: "
        f"{len(ALGORITHMS)}"
    )

    print(
        f"Expected total runs: "
        f"{len(workloads) * len(ALGORITHMS)}"
    )

    save_workload_manifest(
        workloads=workloads,
        delivery_lookup=(
            delivery_lookup
        ),
    )

    results_df = (
        load_existing_results()
    )

    finished = (
        completed_keys(
            results_df
        )
    )

    print(
        f"Already completed runs: "
        f"{len(finished)}"
    )

    (
        _,
        depot_node,
    ) = get_depot_node(
        service
    )

    evaluator = RouteEvaluator(
        road_network=service
    )

    nn_router = (
        NearestNeighborRouter(
            road_network=service,
            metric="distance",
        )
    )

    cw_router = (
        ClarkeWrightRouter(
            road_network=service
        )
    )

    matrix_builder = (
        RoadMatrixBuilder(
            service
        )
    )

    total_runs = (
        len(workloads)
        * len(ALGORITHMS)
    )

    for workload_id, workload_df in enumerate(
        workloads,
        start=1,
    ):
        expected_keys = {
            (
                workload_id,
                algorithm,
            )
            for algorithm
            in ALGORITHMS
        }

        if expected_keys.issubset(
            finished
        ):
            print()
            print(
                f"Workload "
                f"{workload_id}/"
                f"{len(workloads)} "
                f"already complete. "
                f"Skipping."
            )

            continue

        wave_start = (
            workload_df[
                "wave_start"
            ].iloc[
                0
            ]
        )

        wave_end = (
            workload_df[
                "wave_end"
            ].iloc[
                0
            ]
        )

        print()
        print(
            "========================================"
        )

        print(
            f"Workload "
            f"{workload_id}/"
            f"{len(workloads)}"
        )

        print(
            "========================================"
        )

        print(
            f"Wave: "
            f"{wave_start}"
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

        # -------------------------
        # Road matrix
        # -------------------------

        rl_needed = any(
            (
                workload_id,
                algorithm,
            )
            not in finished
            for algorithm
            in (
                "q_learning",
                "sarsa",
                "dqn",
            )
        )

        road_matrix = None
        matrix_runtime = None

        if rl_needed:
            all_nodes = (
                depot_node,
                *delivery_nodes,
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

        # -------------------------
        # Nearest Neighbor
        # -------------------------

        key = (
            workload_id,
            "nearest_neighbor",
        )

        if key not in finished:
            print(
                "Running Nearest Neighbor..."
            )

            timer = perf_counter()

            nn_route = (
                nn_router.build_route(
                    depot_node=depot_node,
                    delivery_ids=delivery_ids,
                    delivery_nodes=delivery_nodes,
                    return_to_depot=True,
                )
            )

            route_runtime = (
                perf_counter()
                - timer
            )

            evaluation = evaluate(
                evaluator=evaluator,
                algorithm=(
                    "nearest_neighbor"
                ),
                depot_node=depot_node,
                deliveries=deliveries,
                delivery_nodes=delivery_nodes,
                delivery_order=(
                    nn_route.delivery_order
                ),
                start_time=wave_end,
            )

            row = make_result_row(
                workload_id=(
                    workload_id
                ),
                wave_start=wave_start,
                wave_end=wave_end,
                evaluation=evaluation,
                route_runtime=(
                    route_runtime
                ),
            )

            results_df = append_result(
                results_df,
                row,
            )

            finished.add(
                key
            )

        # -------------------------
        # Clarke-Wright
        # -------------------------

        key = (
            workload_id,
            "clarke_wright",
        )

        if key not in finished:
            print(
                "Running Clarke-Wright..."
            )

            timer = perf_counter()

            cw_route = (
                cw_router.build_route(
                    depot_node=depot_node,
                    delivery_ids=delivery_ids,
                    delivery_nodes=delivery_nodes,
                    return_to_depot=True,
                )
            )

            route_runtime = (
                perf_counter()
                - timer
            )

            evaluation = evaluate(
                evaluator=evaluator,
                algorithm=(
                    "clarke_wright"
                ),
                depot_node=depot_node,
                deliveries=deliveries,
                delivery_nodes=delivery_nodes,
                delivery_order=(
                    cw_route.delivery_order
                ),
                start_time=wave_end,
            )

            row = make_result_row(
                workload_id=(
                    workload_id
                ),
                wave_start=wave_start,
                wave_end=wave_end,
                evaluation=evaluation,
                route_runtime=(
                    route_runtime
                ),
            )

            results_df = append_result(
                results_df,
                row,
            )

            finished.add(
                key
            )

        # -------------------------
        # Q-Learning
        # -------------------------

        key = (
            workload_id,
            "q_learning",
        )

        if key not in finished:
            print(
                "Running Q-Learning..."
            )

            (
                rollout,
                training,
                training_runtime,
                route_runtime,
            ) = run_q_learning(
                deliveries=deliveries,
                road_matrix=road_matrix,
                start_time=wave_end,
            )

            evaluation = evaluate(
                evaluator=evaluator,
                algorithm="q_learning",
                depot_node=depot_node,
                deliveries=deliveries,
                delivery_nodes=delivery_nodes,
                delivery_order=(
                    rollout.delivery_order
                ),
                start_time=wave_end,
            )

            row = make_result_row(
                workload_id=(
                    workload_id
                ),
                wave_start=wave_start,
                wave_end=wave_end,
                evaluation=evaluation,
                route_runtime=(
                    route_runtime
                ),
                matrix_runtime=(
                    matrix_runtime
                ),
                training_runtime=(
                    training_runtime
                ),
                reward=(
                    rollout.total_reward
                ),
                model_size=(
                    training.q_table_size
                ),
                final_epsilon=(
                    training.final_epsilon
                ),
                device="cpu",
            )

            results_df = append_result(
                results_df,
                row,
            )

            finished.add(
                key
            )

            print(
                f"  reward="
                f"{rollout.total_reward:.4f}, "
                f"train="
                f"{training_runtime:.2f}s"
            )

        # -------------------------
        # SARSA
        # -------------------------

        key = (
            workload_id,
            "sarsa",
        )

        if key not in finished:
            print(
                "Running SARSA..."
            )

            (
                rollout,
                training,
                training_runtime,
                route_runtime,
            ) = run_sarsa(
                deliveries=deliveries,
                road_matrix=road_matrix,
                start_time=wave_end,
            )

            evaluation = evaluate(
                evaluator=evaluator,
                algorithm="sarsa",
                depot_node=depot_node,
                deliveries=deliveries,
                delivery_nodes=delivery_nodes,
                delivery_order=(
                    rollout.delivery_order
                ),
                start_time=wave_end,
            )

            row = make_result_row(
                workload_id=(
                    workload_id
                ),
                wave_start=wave_start,
                wave_end=wave_end,
                evaluation=evaluation,
                route_runtime=(
                    route_runtime
                ),
                matrix_runtime=(
                    matrix_runtime
                ),
                training_runtime=(
                    training_runtime
                ),
                reward=(
                    rollout.total_reward
                ),
                model_size=(
                    training.q_table_size
                ),
                final_epsilon=(
                    training.final_epsilon
                ),
                device="cpu",
            )

            results_df = append_result(
                results_df,
                row,
            )

            finished.add(
                key
            )

            print(
                f"  reward="
                f"{rollout.total_reward:.4f}, "
                f"train="
                f"{training_runtime:.2f}s"
            )

        # -------------------------
        # DQN
        # -------------------------

        key = (
            workload_id,
            "dqn",
        )

        if key not in finished:
            print(
                "Running DQN..."
            )

            (
                rollout,
                training,
                training_runtime,
                route_runtime,
            ) = run_dqn(
                deliveries=deliveries,
                road_matrix=road_matrix,
                start_time=wave_end,
            )

            evaluation = evaluate(
                evaluator=evaluator,
                algorithm="dqn",
                depot_node=depot_node,
                deliveries=deliveries,
                delivery_nodes=delivery_nodes,
                delivery_order=(
                    rollout.delivery_order
                ),
                start_time=wave_end,
            )

            row = make_result_row(
                workload_id=(
                    workload_id
                ),
                wave_start=wave_start,
                wave_end=wave_end,
                evaluation=evaluation,
                route_runtime=(
                    route_runtime
                ),
                matrix_runtime=(
                    matrix_runtime
                ),
                training_runtime=(
                    training_runtime
                ),
                reward=(
                    rollout.total_reward
                ),
                model_size=(
                    training.parameter_count
                ),
                final_epsilon=(
                    training.final_epsilon
                ),
                device=(
                    training.device
                ),
            )

            results_df = append_result(
                results_df,
                row,
            )

            finished.add(
                key
            )

            print(
                f"  reward="
                f"{rollout.total_reward:.4f}, "
                f"train="
                f"{training_runtime:.2f}s, "
                f"device="
                f"{training.device}"
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(
            f"Completed runs: "
            f"{len(finished)}/"
            f"{total_runs}"
        )

    # -------------------------
    # Final summary
    # -------------------------

    summary_df = (
        build_final_summary(
            results_df
        )
    )

    safe_save_csv(
        summary_df,
        SUMMARY_OUTPUT_PATH,
    )

    print()
    print(
        "========================================"
    )

    print(
        "Core Algorithm Final Summary"
    )

    print(
        "========================================"
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
        "========================================"
    )

    print(
        "Benchmark Complete"
    )

    print(
        "========================================"
    )

    print(
        MASTER_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        WORKLOAD_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()