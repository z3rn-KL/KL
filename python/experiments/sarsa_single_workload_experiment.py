from pathlib import Path
from time import perf_counter

import pandas as pd

from evaluation import (
    RouteEvaluator,
)

from rl.q_learning_agent import (
    QLearningAgent,
    QLearningConfig,
)

from rl.routing_environment import (
    RoutingEnvironment,
)

from rl.sarsa_agent import (
    SarsaAgent,
    SarsaConfig,
)

from routing import (
    RoadMatrixBuilder,
)

from experiments.q_learning_single_workload_experiment import (
    build_result_row,
    build_route_dataframe,
    build_training_dataframe,
    build_workload,
    get_depot_node,
    load_delivery_dataframe,
    load_road_network,
    load_snapped_nodes,
    prepare_dataframe,
    select_small_real_workload,
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

SARSA_TRAINING_OUTPUT_PATH = (
    RESULTS_DIR
    / "sarsa_single_workload_training.csv"
)

SARSA_ROUTE_OUTPUT_PATH = (
    RESULTS_DIR
    / "sarsa_single_workload_route.csv"
)

SARSA_SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "sarsa_single_workload_summary.csv"
)

COMPARISON_OUTPUT_PATH = (
    RESULTS_DIR
    / "q_learning_vs_sarsa_single_workload_comparison.csv"
)


TRAINING_EPISODES = 5000

LEARNING_RATE = 0.20

DISCOUNT_FACTOR = 0.95

EPSILON_START = 1.00

EPSILON_MIN = 0.05

EPSILON_DECAY = 0.998

RANDOM_SEED = 42


def create_q_learning_config(
) -> QLearningConfig:
    return QLearningConfig(
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        episodes=TRAINING_EPISODES,
        random_seed=RANDOM_SEED,
    )


def create_sarsa_config(
) -> SarsaConfig:
    return SarsaConfig(
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        episodes=TRAINING_EPISODES,
        random_seed=RANDOM_SEED,
    )


def train_q_learning(
    deliveries,
    road_matrix,
    start_time,
):
    environment = RoutingEnvironment(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=start_time,
    )

    config = (
        create_q_learning_config()
    )

    agent = QLearningAgent(
        config
    )

    print()
    print(
        "========================================"
    )

    print(
        "Training Q-Learning"
    )

    print(
        "========================================"
    )

    print(
        f"Episodes: "
        f"{TRAINING_EPISODES}"
    )

    timer = perf_counter()

    training_result = (
        agent.train(
            environment
        )
    )

    training_runtime = (
        perf_counter()
        - timer
    )

    timer = perf_counter()

    rollout = (
        agent.greedy_rollout(
            environment
        )
    )

    rollout_runtime = (
        perf_counter()
        - timer
    )

    print(
        f"Training runtime: "
        f"{training_runtime:.4f} sec"
    )

    print(
        f"Rollout runtime: "
        f"{rollout_runtime:.6f} sec"
    )

    print(
        f"Q-table size: "
        f"{training_result.q_table_size}"
    )

    print(
        f"Final epsilon: "
        f"{training_result.final_epsilon:.4f}"
    )

    print(
        f"Greedy reward: "
        f"{rollout.total_reward:.4f}"
    )

    print(
        f"Distance: "
        f"{rollout.total_distance_km:.3f} km"
    )

    print(
        f"Travel time: "
        f"{rollout.total_travel_time_minutes:.2f} min"
    )

    return (
        config,
        training_result,
        training_runtime,
        rollout,
        rollout_runtime,
    )


def train_sarsa(
    deliveries,
    road_matrix,
    start_time,
):
    environment = RoutingEnvironment(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=start_time,
    )

    config = (
        create_sarsa_config()
    )

    agent = SarsaAgent(
        config
    )

    print()
    print(
        "========================================"
    )

    print(
        "Training SARSA"
    )

    print(
        "========================================"
    )

    print(
        f"Episodes: "
        f"{TRAINING_EPISODES}"
    )

    timer = perf_counter()

    training_result = (
        agent.train(
            environment
        )
    )

    training_runtime = (
        perf_counter()
        - timer
    )

    timer = perf_counter()

    rollout = (
        agent.greedy_rollout(
            environment
        )
    )

    rollout_runtime = (
        perf_counter()
        - timer
    )

    print(
        f"Training runtime: "
        f"{training_runtime:.4f} sec"
    )

    print(
        f"Rollout runtime: "
        f"{rollout_runtime:.6f} sec"
    )

    print(
        f"Q-table size: "
        f"{training_result.q_table_size}"
    )

    print(
        f"Final epsilon: "
        f"{training_result.final_epsilon:.4f}"
    )

    print(
        f"Greedy reward: "
        f"{rollout.total_reward:.4f}"
    )

    print(
        f"Distance: "
        f"{rollout.total_distance_km:.3f} km"
    )

    print(
        f"Travel time: "
        f"{rollout.total_travel_time_minutes:.2f} min"
    )

    return (
        config,
        training_result,
        training_runtime,
        rollout,
        rollout_runtime,
    )


def print_route(
    name: str,
    rollout,
) -> None:
    print()
    print(
        f"{name} learned route:"
    )

    print(
        "Depot"
    )

    for (
        sequence,
        delivery_id,
    ) in enumerate(
        rollout.delivery_order,
        start=1,
    ):
        print(
            f"  -> {sequence}. "
            f"{delivery_id}"
        )

    print(
        "  -> Depot"
    )


def print_comparison(
    dataframe: pd.DataFrame,
) -> None:
    print()
    print(
        "========================================"
    )

    print(
        "Q-Learning vs SARSA"
    )

    print(
        "========================================"
    )

    columns = [
        "algorithm",
        "number_of_deliveries",
        "total_distance_km",
        "total_travel_time_minutes",
        "on_time_deliveries",
        "late_deliveries",
        "on_time_rate",
        "total_lateness_minutes",
        "max_lateness_minutes",
        "runtime_seconds",
    ]

    print(
        dataframe[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


def main():
    print()
    print(
        "========================================"
    )

    print(
        "XeDu Q-Learning vs SARSA Experiment"
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

    workload_df = (
        select_small_real_workload(
            dataframe
        )
    )

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
        f"Selected wave: "
        f"{wave_start}"
    )

    print(
        f"Planning start: "
        f"{wave_end}"
    )

    print(
        f"Workload size: "
        f"{len(workload_df)}"
    )

    (
        deliveries,
        delivery_nodes,
    ) = (
        build_workload(
            workload_df=workload_df,
            delivery_lookup=delivery_lookup,
        )
    )

    (
        depot,
        depot_node,
    ) = (
        get_depot_node(
            service
        )
    )

    print()
    print(
        "Depot:"
    )

    print(
        f"  ID: "
        f"{depot.depot_id}"
    )

    print(
        f"  Road node: "
        f"{depot_node}"
    )

    all_nodes = (
        depot_node,
        *delivery_nodes,
    )

    matrix_builder = (
        RoadMatrixBuilder(
            service
        )
    )

    print()
    print(
        "Preparing shared road matrix..."
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
        f"Matrix preparation: "
        f"{matrix_runtime:.4f} sec"
    )

    evaluator = RouteEvaluator(
        road_network=service
    )

    (
        q_config,
        q_training_result,
        q_training_runtime,
        q_rollout,
        q_rollout_runtime,
    ) = train_q_learning(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=wave_end,
    )

    (
        sarsa_config,
        sarsa_training_result,
        sarsa_training_runtime,
        sarsa_rollout,
        sarsa_rollout_runtime,
    ) = train_sarsa(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=wave_end,
    )

    print_route(
        "Q-Learning",
        q_rollout,
    )

    print_route(
        "SARSA",
        sarsa_rollout,
    )

    q_evaluation = (
        evaluator.evaluate(
            algorithm="q_learning",
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=(
                q_rollout.delivery_order
            ),
            start_time=wave_end,
            return_to_depot=True,
        )
    )

    sarsa_evaluation = (
        evaluator.evaluate(
            algorithm="sarsa",
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=(
                sarsa_rollout.delivery_order
            ),
            start_time=wave_end,
            return_to_depot=True,
        )
    )

    comparison_df = pd.DataFrame(
        [
            build_result_row(
                q_evaluation,
                q_rollout_runtime,
            ),
            build_result_row(
                sarsa_evaluation,
                sarsa_rollout_runtime,
            ),
        ]
    )

    print_comparison(
        comparison_df
    )

    q_training_df = (
        build_training_dataframe(
            q_training_result
        )
    )

    sarsa_training_df = (
        build_training_dataframe(
            sarsa_training_result
        )
    )

    sarsa_route_df = (
        build_route_dataframe(
            sarsa_rollout
        )
    )

    q_last_100 = (
        q_training_df.tail(
            100
        )
    )

    sarsa_last_100 = (
        sarsa_training_df.tail(
            100
        )
    )

    same_route = (
        q_rollout.delivery_order
        == sarsa_rollout.delivery_order
    )

    summary_df = pd.DataFrame(
        [
            {
                "algorithm":
                    "q_learning",

                "wave_start":
                    wave_start,

                "wave_end":
                    wave_end,

                "number_of_deliveries":
                    len(
                        deliveries
                    ),

                "training_episodes":
                    TRAINING_EPISODES,

                "learning_rate":
                    q_config.learning_rate,

                "discount_factor":
                    q_config.discount_factor,

                "epsilon_start":
                    q_config.epsilon_start,

                "epsilon_min":
                    q_config.epsilon_min,

                "epsilon_decay":
                    q_config.epsilon_decay,

                "final_epsilon":
                    q_training_result.final_epsilon,

                "q_table_size":
                    q_training_result.q_table_size,

                "training_runtime_seconds":
                    q_training_runtime,

                "rollout_runtime_seconds":
                    q_rollout_runtime,

                "greedy_reward":
                    q_rollout.total_reward,

                "greedy_distance_km":
                    q_rollout.total_distance_km,

                "greedy_travel_time_minutes":
                    q_rollout.total_travel_time_minutes,

                "mean_last_100_reward":
                    q_last_100[
                        "total_reward"
                    ].mean(),

                "max_last_100_reward":
                    q_last_100[
                        "total_reward"
                    ].max(),

                "same_route":
                    same_route,
            },
            {
                "algorithm":
                    "sarsa",

                "wave_start":
                    wave_start,

                "wave_end":
                    wave_end,

                "number_of_deliveries":
                    len(
                        deliveries
                    ),

                "training_episodes":
                    TRAINING_EPISODES,

                "learning_rate":
                    sarsa_config.learning_rate,

                "discount_factor":
                    sarsa_config.discount_factor,

                "epsilon_start":
                    sarsa_config.epsilon_start,

                "epsilon_min":
                    sarsa_config.epsilon_min,

                "epsilon_decay":
                    sarsa_config.epsilon_decay,

                "final_epsilon":
                    (
                        sarsa_training_result
                        .final_epsilon
                    ),

                "q_table_size":
                    (
                        sarsa_training_result
                        .q_table_size
                    ),

                "training_runtime_seconds":
                    sarsa_training_runtime,

                "rollout_runtime_seconds":
                    sarsa_rollout_runtime,

                "greedy_reward":
                    sarsa_rollout.total_reward,

                "greedy_distance_km":
                    (
                        sarsa_rollout
                        .total_distance_km
                    ),

                "greedy_travel_time_minutes":
                    (
                        sarsa_rollout
                        .total_travel_time_minutes
                    ),

                "mean_last_100_reward":
                    sarsa_last_100[
                        "total_reward"
                    ].mean(),

                "max_last_100_reward":
                    sarsa_last_100[
                        "total_reward"
                    ].max(),

                "same_route":
                    same_route,
            },
        ]
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sarsa_training_df.to_csv(
        SARSA_TRAINING_OUTPUT_PATH,
        index=False,
    )

    sarsa_route_df.to_csv(
        SARSA_ROUTE_OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SARSA_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "========================================"
    )

    print(
        "Direct RL Comparison"
    )

    print(
        "========================================"
    )

    print(
        f"Same greedy route: "
        f"{same_route}"
    )

    print(
        f"Q-Learning reward: "
        f"{q_rollout.total_reward:.4f}"
    )

    print(
        f"SARSA reward: "
        f"{sarsa_rollout.total_reward:.4f}"
    )

    print(
        f"Q-Learning training: "
        f"{q_training_runtime:.4f} sec"
    )

    print(
        f"SARSA training: "
        f"{sarsa_training_runtime:.4f} sec"
    )

    print(
        f"Q-Learning Q-table: "
        f"{q_training_result.q_table_size}"
    )

    print(
        f"SARSA Q-table: "
        f"{sarsa_training_result.q_table_size}"
    )

    print()
    print(
        f"Shared matrix preparation: "
        f"{matrix_runtime:.4f} sec"
    )

    print()
    print(
        "========================================"
    )

    print(
        "Results Saved"
    )

    print(
        "========================================"
    )

    print(
        SARSA_TRAINING_OUTPUT_PATH
    )

    print(
        SARSA_ROUTE_OUTPUT_PATH
    )

    print(
        SARSA_SUMMARY_OUTPUT_PATH
    )

    print(
        COMPARISON_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()