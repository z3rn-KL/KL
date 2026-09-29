from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

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

DETAIL_OUTPUT_PATH = (
    RESULTS_DIR
    / "core_algorithm_benchmark_details.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "core_algorithm_benchmark_summary.csv"
)

WORKLOAD_OUTPUT_PATH = (
    RESULTS_DIR
    / "core_algorithm_benchmark_workloads.csv"
)


WORKLOAD_SIZE = 5

NUMBER_OF_WORKLOADS = 5

TRAINING_EPISODES = 5000

RANDOM_SEED = 42

TABULAR_LEARNING_RATE = 0.20

DQN_LEARNING_RATE = 0.001

DISCOUNT_FACTOR = 0.95

EPSILON_START = 1.00

EPSILON_MIN = 0.05

EPSILON_DECAY = 0.998


def select_benchmark_workloads(
    dataframe: pd.DataFrame,
) -> list[pd.DataFrame]:
    """
    Select exactly five 5-delivery waves
    distributed across the dataset period.

    This avoids selecting only adjacent
    workloads from one short time period.
    """

    groups = []

    for (
        wave_start,
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

        groups.append(
            (
                wave_start,
                group,
            )
        )

    groups.sort(
        key=lambda item: item[
            0
        ]
    )

    if len(
        groups
    ) < NUMBER_OF_WORKLOADS:
        raise ValueError(
            "Not enough exact-size "
            "benchmark workloads."
        )

    selected_indices = (
        np.linspace(
            0,
            len(groups) - 1,
            NUMBER_OF_WORKLOADS,
        )
        .round()
        .astype(int)
    )

    selected = [
        groups[
            int(index)
        ][
            1
        ]
        for index
        in selected_indices
    ]

    return selected


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


def evaluate_route(
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


def result_row(
    workload_id: int,
    wave_start,
    wave_end,
    evaluation,
    route_runtime: float,
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
    return {
        "workload_id":
            workload_id,

        "wave_start":
            wave_start,

        "wave_end":
            wave_end,

        "algorithm":
            evaluation.algorithm,

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

        "route_runtime_seconds":
            route_runtime,

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


def build_summary(
    detail_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for algorithm, group in (
        detail_df.groupby(
            "algorithm"
        )
    ):
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

                "total_lateness_minutes":
                    group[
                        "total_lateness_minutes"
                    ].sum(),

                "mean_route_runtime_seconds":
                    group[
                        "route_runtime_seconds"
                    ].mean(),

                "mean_training_runtime_seconds":
                    group[
                        "training_runtime_seconds"
                    ].mean(),

                "mean_greedy_reward":
                    group[
                        "greedy_reward"
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
            "algorithm"
        )
        .reset_index(
            drop=True
        )
    )


def main():
    print()
    print(
        "========================================"
    )
    print(
        "Core Algorithm Benchmark"
    )
    print(
        "========================================"
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
        select_benchmark_workloads(
            dataframe
        )
    )

    (
        depot,
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

    detail_rows = []

    workload_rows = []

    for workload_id, workload_df in enumerate(
        workloads,
        start=1,
    ):
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

        workload_rows.append(
            {
                "workload_id":
                    workload_id,

                "wave_start":
                    wave_start,

                "wave_end":
                    wave_end,

                "number_of_deliveries":
                    len(
                        deliveries
                    ),

                "delivery_ids":
                    " -> ".join(
                        delivery_ids
                    ),
            }
        )

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
            f"Matrix: "
            f"{matrix_runtime:.3f} sec"
        )

        # -------------------------
        # Nearest Neighbor
        # -------------------------

        timer = perf_counter()

        nn_route = (
            nn_router.build_route(
                depot_node=depot_node,
                delivery_ids=delivery_ids,
                delivery_nodes=delivery_nodes,
                return_to_depot=True,
            )
        )

        nn_runtime = (
            perf_counter()
            - timer
        )

        nn_eval = evaluate_route(
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

        detail_rows.append(
            result_row(
                workload_id=(
                    workload_id
                ),
                wave_start=wave_start,
                wave_end=wave_end,
                evaluation=nn_eval,
                route_runtime=(
                    nn_runtime
                ),
            )
        )

        # -------------------------
        # Clarke-Wright
        # -------------------------

        timer = perf_counter()

        cw_route = (
            cw_router.build_route(
                depot_node=depot_node,
                delivery_ids=delivery_ids,
                delivery_nodes=delivery_nodes,
                return_to_depot=True,
            )
        )

        cw_runtime = (
            perf_counter()
            - timer
        )

        cw_eval = evaluate_route(
            evaluator=evaluator,
            algorithm="clarke_wright",
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=(
                cw_route.delivery_order
            ),
            start_time=wave_end,
        )

        detail_rows.append(
            result_row(
                workload_id=(
                    workload_id
                ),
                wave_start=wave_start,
                wave_end=wave_end,
                evaluation=cw_eval,
                route_runtime=(
                    cw_runtime
                ),
            )
        )

        # -------------------------
        # Q-Learning
        # -------------------------

        (
            q_rollout,
            q_training,
            q_training_runtime,
            q_runtime,
        ) = run_q_learning(
            deliveries=deliveries,
            road_matrix=road_matrix,
            start_time=wave_end,
        )

        q_eval = evaluate_route(
            evaluator=evaluator,
            algorithm="q_learning",
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=(
                q_rollout.delivery_order
            ),
            start_time=wave_end,
        )

        detail_rows.append(
            result_row(
                workload_id=(
                    workload_id
                ),
                wave_start=wave_start,
                wave_end=wave_end,
                evaluation=q_eval,
                route_runtime=q_runtime,
                training_runtime=(
                    q_training_runtime
                ),
                reward=(
                    q_rollout.total_reward
                ),
                model_size=(
                    q_training.q_table_size
                ),
                final_epsilon=(
                    q_training.final_epsilon
                ),
                device="cpu",
            )
        )

        print(
            f"Q-Learning: "
            f"reward="
            f"{q_rollout.total_reward:.4f}, "
            f"train="
            f"{q_training_runtime:.2f}s"
        )

        # -------------------------
        # SARSA
        # -------------------------

        (
            sarsa_rollout,
            sarsa_training,
            sarsa_training_runtime,
            sarsa_runtime,
        ) = run_sarsa(
            deliveries=deliveries,
            road_matrix=road_matrix,
            start_time=wave_end,
        )

        sarsa_eval = evaluate_route(
            evaluator=evaluator,
            algorithm="sarsa",
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=(
                sarsa_rollout
                .delivery_order
            ),
            start_time=wave_end,
        )

        detail_rows.append(
            result_row(
                workload_id=(
                    workload_id
                ),
                wave_start=wave_start,
                wave_end=wave_end,
                evaluation=sarsa_eval,
                route_runtime=(
                    sarsa_runtime
                ),
                training_runtime=(
                    sarsa_training_runtime
                ),
                reward=(
                    sarsa_rollout
                    .total_reward
                ),
                model_size=(
                    sarsa_training
                    .q_table_size
                ),
                final_epsilon=(
                    sarsa_training
                    .final_epsilon
                ),
                device="cpu",
            )
        )

        print(
            f"SARSA: "
            f"reward="
            f"{sarsa_rollout.total_reward:.4f}, "
            f"train="
            f"{sarsa_training_runtime:.2f}s"
        )

        # -------------------------
        # DQN
        # -------------------------

        (
            dqn_rollout,
            dqn_training,
            dqn_training_runtime,
            dqn_runtime,
        ) = run_dqn(
            deliveries=deliveries,
            road_matrix=road_matrix,
            start_time=wave_end,
        )

        dqn_eval = evaluate_route(
            evaluator=evaluator,
            algorithm="dqn",
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=(
                dqn_rollout.delivery_order
            ),
            start_time=wave_end,
        )

        detail_rows.append(
            result_row(
                workload_id=(
                    workload_id
                ),
                wave_start=wave_start,
                wave_end=wave_end,
                evaluation=dqn_eval,
                route_runtime=dqn_runtime,
                training_runtime=(
                    dqn_training_runtime
                ),
                reward=(
                    dqn_rollout.total_reward
                ),
                model_size=(
                    dqn_training
                    .parameter_count
                ),
                final_epsilon=(
                    dqn_training
                    .final_epsilon
                ),
                device=(
                    dqn_training.device
                ),
            )
        )

        print(
            f"DQN: "
            f"reward="
            f"{dqn_rollout.total_reward:.4f}, "
            f"train="
            f"{dqn_training_runtime:.2f}s, "
            f"device="
            f"{dqn_training.device}"
        )

    detail_df = pd.DataFrame(
        detail_rows
    )

    summary_df = (
        build_summary(
            detail_df
        )
    )

    workload_df = pd.DataFrame(
        workload_rows
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_df.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    workload_df.to_csv(
        WORKLOAD_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "========================================"
    )

    print(
        "Core Algorithm Benchmark Summary"
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
        "Results:"
    )

    print(
        DETAIL_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        WORKLOAD_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()