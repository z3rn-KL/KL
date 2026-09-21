from pathlib import Path
from time import perf_counter

import pandas as pd

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)

from evaluation import (
    RouteEvaluator,
)

from rl import (
    QLearningAgent,
    QLearningConfig,
    RoutingEnvironment,
)

from routing import (
    NearestNeighborRouter,
    RoadMatrixBuilder,
    RoadNetworkService,
)

from routing.clarke_wright import (
    ClarkeWrightRouter,
)

from routing.priority_nearest_neighbor import (
    PriorityNearestNeighborRouter,
)

from simulation.scenario import (
    ScenarioFactory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)

GRAPH_PATH = (
    PROJECT_ROOT
    / "maps"
    / "xedu_drive.graphml"
)

SNAPPED_PATH = (
    PROJECT_ROOT
    / "results"
    / "xedu_delivery_road_nodes_scc.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

TRAINING_OUTPUT_PATH = (
    RESULTS_DIR
    / "q_learning_single_workload_training.csv"
)

ROUTE_OUTPUT_PATH = (
    RESULTS_DIR
    / "q_learning_single_workload_route.csv"
)

COMPARISON_OUTPUT_PATH = (
    RESULTS_DIR
    / "four_algorithm_single_workload_comparison.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "q_learning_single_workload_summary.csv"
)


WAVE_MINUTES = 120
MIN_WORKLOAD_SIZE = 5

TRAINING_EPISODES = 5000


def load_road_network() -> RoadNetworkService:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing road graph: {GRAPH_PATH}"
        )

    print()
    print(
        "Loading cached road graph..."
    )

    service = (
        RoadNetworkService.load_graphml(
            GRAPH_PATH
        )
    )

    print(
        f"Road nodes: "
        f"{service.graph.number_of_nodes()}"
    )

    print(
        f"Road edges: "
        f"{service.graph.number_of_edges()}"
    )

    return service


def load_delivery_dataframe():
    dataframe = (
        DataLoader.load_csv(
            DATASET_PATH
        )
    )

    dataframe = (
        DataPreprocessor
        .prepare_deliveries(
            dataframe
        )
    )

    deliveries = (
        DeliveryMapper
        .from_dataframe(
            dataframe
        )
    )

    delivery_lookup = {}

    rows = []

    for delivery in deliveries:
        delivery_id = str(
            delivery.delivery_id
        )

        delivery_lookup[
            delivery_id
        ] = delivery

        if delivery.created_at is None:
            continue

        created_at = pd.Timestamp(
            delivery.created_at
        )

        rows.append(
            {
                "delivery_id":
                    delivery_id,

                "created_at":
                    created_at,

                "expected_delivery_time":
                    delivery.expected_delivery_time,

                "service_type":
                    delivery.service_type,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "wave_start"
    ] = (
        result[
            "created_at"
        ]
        .dt.floor(
            f"{WAVE_MINUTES}min"
        )
    )

    result[
        "wave_end"
    ] = (
        result[
            "wave_start"
        ]
        + pd.Timedelta(
            minutes=WAVE_MINUTES
        )
    )

    return (
        result,
        delivery_lookup,
    )


def load_snapped_nodes() -> pd.DataFrame:
    if not SNAPPED_PATH.exists():
        raise FileNotFoundError(
            f"Missing SCC road-node file: "
            f"{SNAPPED_PATH}"
        )

    dataframe = pd.read_csv(
        SNAPPED_PATH
    )

    dataframe[
        "delivery_id"
    ] = dataframe[
        "delivery_id"
    ].astype(
        str
    )

    dataframe[
        "road_node"
    ] = dataframe[
        "road_node"
    ].astype(
        int
    )

    return dataframe[
        [
            "delivery_id",
            "road_node",
        ]
    ]


def prepare_dataframe(
    delivery_df: pd.DataFrame,
    snapped_df: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = delivery_df.merge(
        snapped_df,
        on="delivery_id",
        how="left",
    )

    missing_nodes = int(
        dataframe[
            "road_node"
        ].isna().sum()
    )

    if missing_nodes > 0:
        raise ValueError(
            f"{missing_nodes} deliveries "
            "do not have SCC road nodes."
        )

    dataframe[
        "road_node"
    ] = dataframe[
        "road_node"
    ].astype(
        int
    )

    return dataframe


def select_small_real_workload(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select a deterministic real XeDu wave.

    Strategy:
        1. only waves with at least 5 orders
        2. choose the smallest eligible size
        3. if tied, choose the earliest wave

    This keeps tabular Q-Learning tractable
    for the first real-data validation.
    """

    sizes = (
        dataframe
        .groupby(
            "wave_start"
        )
        .size()
    )

    eligible = sizes[
        sizes >= MIN_WORKLOAD_SIZE
    ]

    if eligible.empty:
        raise ValueError(
            "No eligible workload found."
        )

    minimum_size = int(
        eligible.min()
    )

    candidate_waves = (
        eligible[
            eligible
            == minimum_size
        ]
        .sort_index()
    )

    selected_wave_start = (
        candidate_waves.index[
            0
        ]
    )

    workload = (
        dataframe[
            dataframe[
                "wave_start"
            ]
            == selected_wave_start
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return workload


def build_workload(
    workload_df: pd.DataFrame,
    delivery_lookup: dict,
):
    deliveries = []
    delivery_nodes = []

    for _, row in (
        workload_df.iterrows()
    ):
        delivery_id = str(
            row[
                "delivery_id"
            ]
        )

        delivery = (
            delivery_lookup.get(
                delivery_id
            )
        )

        if delivery is None:
            raise ValueError(
                f"Delivery not found: "
                f"{delivery_id}"
            )

        deliveries.append(
            delivery
        )

        delivery_nodes.append(
            int(
                row[
                    "road_node"
                ]
            )
        )

    return (
        deliveries,
        delivery_nodes,
    )


def get_depot_node(
    service: RoadNetworkService,
):
    scenario = (
        ScenarioFactory
        .create_single_depot_scenario()
    )

    depot = scenario.depots[
        0
    ]

    depot_node = (
        service.nearest_node(
            latitude=depot.latitude,
            longitude=depot.longitude,
        )
    )

    return (
        depot,
        int(
            depot_node
        ),
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


def build_result_row(
    evaluation,
    runtime_seconds: float,
) -> dict:
    return {
        "algorithm":
            evaluation.algorithm,

        "number_of_deliveries":
            evaluation.number_of_deliveries,

        "on_time_deliveries":
            evaluation.on_time_deliveries,

        "late_deliveries":
            evaluation.late_deliveries,

        "on_time_rate":
            evaluation.on_time_rate,

        "total_lateness_minutes":
            evaluation.total_lateness_minutes,

        "max_lateness_minutes":
            evaluation.max_lateness_minutes,

        "total_distance_km":
            evaluation.total_distance_km,

        "total_travel_time_minutes":
            evaluation.total_travel_time_minutes,

        "runtime_seconds":
            runtime_seconds,
    }


def build_training_dataframe(
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
                    episode.total_travel_time_minutes,

                "steps":
                    episode.steps,

                "delivery_order":
                    " -> ".join(
                        episode.delivery_order
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_route_dataframe(
    rollout,
) -> pd.DataFrame:
    rows = []

    for sequence, (
        delivery_id,
        action,
    ) in enumerate(
        zip(
            rollout.delivery_order,
            rollout.action_order,
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "sequence":
                    sequence,

                "delivery_id":
                    delivery_id,

                "action":
                    action,
            }
        )

    return pd.DataFrame(
        rows
    )


def print_comparison(
    comparison_df: pd.DataFrame,
) -> None:
    print()
    print(
        "========================================"
    )

    print(
        "Four-algorithm Single-workload Comparison"
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
        comparison_df[
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
        "XeDu Q-Learning Single-workload Experiment"
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
    ) = load_delivery_dataframe()

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
    ) = build_workload(
        workload_df=workload_df,
        delivery_lookup=delivery_lookup,
    )

    delivery_ids = [
        str(
            delivery.delivery_id
        )
        for delivery
        in deliveries
    ]

    (
        depot,
        depot_node,
    ) = get_depot_node(
        service
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

    matrix_start = (
        perf_counter()
    )

    road_matrix = (
        matrix_builder.build(
            all_nodes
        )
    )

    matrix_runtime = (
        perf_counter()
        - matrix_start
    )

    print(
        f"Matrix preparation: "
        f"{matrix_runtime:.4f} sec"
    )

    nn_router = (
        NearestNeighborRouter(
            road_network=service,
            metric="distance",
        )
    )

    priority_router = (
        PriorityNearestNeighborRouter(
            road_network=service
        )
    )

    cw_router = (
        ClarkeWrightRouter(
            road_network=service
        )
    )

    evaluator = (
        RouteEvaluator(
            road_network=service
        )
    )

    nn_router.matrix_builder = (
        matrix_builder
    )

    priority_router.matrix_builder = (
        matrix_builder
    )

    cw_router.matrix_builder = (
        matrix_builder
    )

    evaluator.matrix_builder = (
        matrix_builder
    )

    print()
    print(
        "Running baseline algorithms..."
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

    nn_runtime = (
        perf_counter()
        - timer
    )

    timer = perf_counter()

    priority_route = (
        priority_router.build_route(
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            start_time=wave_end,
            return_to_depot=True,
        )
    )

    priority_runtime = (
        perf_counter()
        - timer
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

    cw_runtime = (
        perf_counter()
        - timer
    )

    environment = (
        RoutingEnvironment(
            deliveries=deliveries,
            road_matrix=road_matrix,
            start_time=wave_end,
        )
    )

    config = QLearningConfig(
        learning_rate=0.20,
        discount_factor=0.95,
        epsilon_start=1.00,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        episodes=TRAINING_EPISODES,
        random_seed=42,
    )

    agent = QLearningAgent(
        config
    )

    print()
    print(
        f"Training Q-Learning for "
        f"{TRAINING_EPISODES} episodes..."
    )

    training_start = (
        perf_counter()
    )

    training_result = (
        agent.train(
            environment
        )
    )

    training_runtime = (
        perf_counter()
        - training_start
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

    rollout_start = (
        perf_counter()
    )

    rollout = (
        agent.greedy_rollout(
            environment
        )
    )

    rollout_runtime = (
        perf_counter()
        - rollout_start
    )

    print()
    print(
        "Learned Q-Learning route:"
    )

    print(
        "Depot"
    )

    for sequence, delivery_id in enumerate(
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

    print()
    print(
        f"Greedy rollout reward: "
        f"{rollout.total_reward:.4f}"
    )

    print(
        f"Greedy rollout distance: "
        f"{rollout.total_distance_km:.3f} km"
    )

    print(
        f"Greedy rollout travel time: "
        f"{rollout.total_travel_time_minutes:.2f} min"
    )

    nn_evaluation = evaluate_route(
        evaluator=evaluator,
        algorithm="nearest_neighbor",
        depot_node=depot_node,
        deliveries=deliveries,
        delivery_nodes=delivery_nodes,
        delivery_order=nn_route.delivery_order,
        start_time=wave_end,
    )

    priority_evaluation = (
        evaluate_route(
            evaluator=evaluator,
            algorithm=(
                "priority_nearest_neighbor"
            ),
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=(
                priority_route.delivery_order
            ),
            start_time=wave_end,
        )
    )

    cw_evaluation = evaluate_route(
        evaluator=evaluator,
        algorithm="clarke_wright",
        depot_node=depot_node,
        deliveries=deliveries,
        delivery_nodes=delivery_nodes,
        delivery_order=cw_route.delivery_order,
        start_time=wave_end,
    )

    q_evaluation = evaluate_route(
        evaluator=evaluator,
        algorithm="q_learning",
        depot_node=depot_node,
        deliveries=deliveries,
        delivery_nodes=delivery_nodes,
        delivery_order=rollout.delivery_order,
        start_time=wave_end,
    )

    comparison_df = pd.DataFrame(
        [
            build_result_row(
                nn_evaluation,
                nn_runtime,
            ),
            build_result_row(
                priority_evaluation,
                priority_runtime,
            ),
            build_result_row(
                cw_evaluation,
                cw_runtime,
            ),
            build_result_row(
                q_evaluation,
                rollout_runtime,
            ),
        ]
    )

    print_comparison(
        comparison_df
    )

    training_df = (
        build_training_dataframe(
            training_result
        )
    )

    route_df = (
        build_route_dataframe(
            rollout
        )
    )

    last_100 = (
        training_df
        .tail(
            100
        )
    )

    summary_df = pd.DataFrame(
        [
            {
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
                    config.learning_rate,

                "discount_factor":
                    config.discount_factor,

                "epsilon_start":
                    config.epsilon_start,

                "epsilon_min":
                    config.epsilon_min,

                "epsilon_decay":
                    config.epsilon_decay,

                "final_epsilon":
                    training_result.final_epsilon,

                "q_table_size":
                    training_result.q_table_size,

                "training_runtime_seconds":
                    training_runtime,

                "matrix_preparation_seconds":
                    matrix_runtime,

                "greedy_reward":
                    rollout.total_reward,

                "greedy_distance_km":
                    rollout.total_distance_km,

                "greedy_travel_time_minutes":
                    rollout.total_travel_time_minutes,

                "mean_last_100_reward":
                    last_100[
                        "total_reward"
                    ].mean(),

                "max_last_100_reward":
                    last_100[
                        "total_reward"
                    ].max(),
            }
        ]
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_df.to_csv(
        TRAINING_OUTPUT_PATH,
        index=False,
    )

    route_df.to_csv(
        ROUTE_OUTPUT_PATH,
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
        TRAINING_OUTPUT_PATH
    )

    print(
        ROUTE_OUTPUT_PATH
    )

    print(
        COMPARISON_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()