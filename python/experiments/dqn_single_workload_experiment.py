from pathlib import Path
from time import perf_counter

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

DQN_TRAINING_OUTPUT_PATH = (
    RESULTS_DIR
    / "dqn_single_workload_training.csv"
)

DQN_ROUTE_OUTPUT_PATH = (
    RESULTS_DIR
    / "dqn_single_workload_route.csv"
)

COMPARISON_OUTPUT_PATH = (
    RESULTS_DIR
    / "rl_three_algorithm_single_workload_comparison.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "rl_three_algorithm_single_workload_summary.csv"
)


TRAINING_EPISODES = 5000

RANDOM_SEED = 42

LEARNING_RATE_TABULAR = 0.20

LEARNING_RATE_DQN = 0.001

DISCOUNT_FACTOR = 0.95

EPSILON_START = 1.00

EPSILON_MIN = 0.05

EPSILON_DECAY = 0.998


def create_q_learning_config(
) -> QLearningConfig:
    return QLearningConfig(
        learning_rate=LEARNING_RATE_TABULAR,
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
        learning_rate=LEARNING_RATE_TABULAR,
        discount_factor=DISCOUNT_FACTOR,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        episodes=TRAINING_EPISODES,
        random_seed=RANDOM_SEED,
    )


def create_dqn_config(
) -> DQNConfig:
    return DQNConfig(
        learning_rate=LEARNING_RATE_DQN,
        discount_factor=DISCOUNT_FACTOR,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        episodes=TRAINING_EPISODES,
        batch_size=64,
        replay_capacity=10000,
        min_replay_size=128,
        target_update_interval=100,
        hidden_dim=128,
        gradient_clip_norm=5.0,
        elapsed_time_bucket_scale=36.0,
        random_seed=RANDOM_SEED,
        device="auto",
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

    return (
        config,
        training_result,
        training_runtime,
        rollout,
        rollout_runtime,
    )


def train_dqn(
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
        create_dqn_config()
    )

    agent = DQNAgent(
        delivery_count=len(
            deliveries
        ),
        config=config,
    )

    print()
    print(
        "========================================"
    )
    print(
        "Training DQN"
    )
    print(
        "========================================"
    )

    print(
        f"Device: "
        f"{agent.device}"
    )

    print(
        f"Input dimension: "
        f"{agent.input_dim}"
    )

    print(
        f"Action dimension: "
        f"{agent.action_dim}"
    )

    print(
        f"Parameters: "
        f"{agent.parameter_count}"
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
        f"Replay size: "
        f"{training_result.replay_size}"
    )

    print(
        f"Optimization steps: "
        f"{training_result.optimization_steps}"
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
        f"{rollout.total_distance_km:.4f} km"
    )

    print(
        f"Travel time: "
        f"{rollout.total_travel_time_minutes:.4f} min"
    )

    return (
        config,
        training_result,
        training_runtime,
        rollout,
        rollout_runtime,
    )


def build_dqn_training_dataframe(
    training_result,
) -> pd.DataFrame:
    rows = []

    for episode in (
        training_result.episodes
    ):
        rows.append(
            {
                "episode":
                    episode.episode,

                "epsilon":
                    episode.epsilon,

                "total_reward":
                    episode.total_reward,

                "total_distance_km":
                    episode.total_distance_km,

                "total_travel_time_minutes":
                    (
                        episode
                        .total_travel_time_minutes
                    ),

                "steps":
                    episode.steps,

                "mean_loss":
                    episode.mean_loss,

                "delivery_order":
                    " -> ".join(
                        episode.delivery_order
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def print_route(
    algorithm: str,
    rollout,
) -> None:
    print()
    print(
        f"{algorithm} route:"
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


def main():
    print()
    print(
        "========================================"
    )
    print(
        "XeDu RL Three-Algorithm Experiment"
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
        f"Depot: "
        f"{depot.depot_id}"
    )

    print(
        f"Depot road node: "
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
        q_training,
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
        sarsa_training,
        sarsa_training_runtime,
        sarsa_rollout,
        sarsa_rollout_runtime,
    ) = train_sarsa(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=wave_end,
    )

    (
        dqn_config,
        dqn_training,
        dqn_training_runtime,
        dqn_rollout,
        dqn_rollout_runtime,
    ) = train_dqn(
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

    print_route(
        "DQN",
        dqn_rollout,
    )

    q_evaluation = evaluator.evaluate(
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

    sarsa_evaluation = evaluator.evaluate(
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

    dqn_evaluation = evaluator.evaluate(
        algorithm="dqn",
        depot_node=depot_node,
        deliveries=deliveries,
        delivery_nodes=delivery_nodes,
        delivery_order=(
            dqn_rollout.delivery_order
        ),
        start_time=wave_end,
        return_to_depot=True,
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
            build_result_row(
                dqn_evaluation,
                dqn_rollout_runtime,
            ),
        ]
    )

    print()
    print(
        "========================================"
    )
    print(
        "Q-Learning vs SARSA vs DQN"
    )
    print(
        "========================================"
    )

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    q_training_df = (
        build_training_dataframe(
            q_training
        )
    )

    sarsa_training_df = (
        build_training_dataframe(
            sarsa_training
        )
    )

    dqn_training_df = (
        build_dqn_training_dataframe(
            dqn_training
        )
    )

    dqn_route_df = (
        build_route_dataframe(
            dqn_rollout
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

    dqn_last_100 = (
        dqn_training_df.tail(
            100
        )
    )

    summary_df = pd.DataFrame(
        [
            {
                "algorithm":
                    "q_learning",

                "episodes":
                    TRAINING_EPISODES,

                "learning_rate":
                    q_config.learning_rate,

                "discount_factor":
                    q_config.discount_factor,

                "final_epsilon":
                    q_training.final_epsilon,

                "training_runtime_seconds":
                    q_training_runtime,

                "greedy_reward":
                    q_rollout.total_reward,

                "distance_km":
                    q_rollout.total_distance_km,

                "travel_time_minutes":
                    (
                        q_rollout
                        .total_travel_time_minutes
                    ),

                "mean_last_100_reward":
                    q_last_100[
                        "total_reward"
                    ].mean(),

                "model_size":
                    q_training.q_table_size,
            },
            {
                "algorithm":
                    "sarsa",

                "episodes":
                    TRAINING_EPISODES,

                "learning_rate":
                    sarsa_config.learning_rate,

                "discount_factor":
                    sarsa_config.discount_factor,

                "final_epsilon":
                    sarsa_training.final_epsilon,

                "training_runtime_seconds":
                    sarsa_training_runtime,

                "greedy_reward":
                    sarsa_rollout.total_reward,

                "distance_km":
                    sarsa_rollout.total_distance_km,

                "travel_time_minutes":
                    (
                        sarsa_rollout
                        .total_travel_time_minutes
                    ),

                "mean_last_100_reward":
                    sarsa_last_100[
                        "total_reward"
                    ].mean(),

                "model_size":
                    sarsa_training.q_table_size,
            },
            {
                "algorithm":
                    "dqn",

                "episodes":
                    TRAINING_EPISODES,

                "learning_rate":
                    dqn_config.learning_rate,

                "discount_factor":
                    dqn_config.discount_factor,

                "final_epsilon":
                    dqn_training.final_epsilon,

                "training_runtime_seconds":
                    dqn_training_runtime,

                "greedy_reward":
                    dqn_rollout.total_reward,

                "distance_km":
                    dqn_rollout.total_distance_km,

                "travel_time_minutes":
                    (
                        dqn_rollout
                        .total_travel_time_minutes
                    ),

                "mean_last_100_reward":
                    dqn_last_100[
                        "total_reward"
                    ].mean(),

                "model_size":
                    dqn_training.parameter_count,
            },
        ]
    )

    same_q_sarsa = (
        q_rollout.delivery_order
        == sarsa_rollout.delivery_order
    )

    same_q_dqn = (
        q_rollout.delivery_order
        == dqn_rollout.delivery_order
    )

    same_sarsa_dqn = (
        sarsa_rollout.delivery_order
        == dqn_rollout.delivery_order
    )

    print()
    print(
        "========================================"
    )
    print(
        "Direct RL Summary"
    )
    print(
        "========================================"
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
        f"DQN reward: "
        f"{dqn_rollout.total_reward:.4f}"
    )

    print()

    print(
        f"Q-Learning == SARSA route: "
        f"{same_q_sarsa}"
    )

    print(
        f"Q-Learning == DQN route: "
        f"{same_q_dqn}"
    )

    print(
        f"SARSA == DQN route: "
        f"{same_sarsa_dqn}"
    )

    print()

    print(
        f"Q-Learning training: "
        f"{q_training_runtime:.4f} sec"
    )

    print(
        f"SARSA training: "
        f"{sarsa_training_runtime:.4f} sec"
    )

    print(
        f"DQN training: "
        f"{dqn_training_runtime:.4f} sec"
    )

    print(
        f"DQN device: "
        f"{dqn_training.device}"
    )

    print(
        f"DQN parameters: "
        f"{dqn_training.parameter_count}"
    )

    print(
        f"DQN optimization steps: "
        f"{dqn_training.optimization_steps}"
    )

    print(
        f"Shared matrix preparation: "
        f"{matrix_runtime:.4f} sec"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dqn_training_df.to_csv(
        DQN_TRAINING_OUTPUT_PATH,
        index=False,
    )

    dqn_route_df.to_csv(
        DQN_ROUTE_OUTPUT_PATH,
        index=False,
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
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
        DQN_TRAINING_OUTPUT_PATH
    )

    print(
        DQN_ROUTE_OUTPUT_PATH
    )

    print(
        COMPARISON_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()