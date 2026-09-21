from itertools import permutations
from math import factorial
from pathlib import Path
from time import perf_counter

import numpy as np
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
    RoadMatrixBuilder,
    RoadNetworkService,
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

SEED_SUMMARY_PATH = (
    RESULTS_DIR
    / "q_learning_stability_seed_summary.csv"
)

TRAINING_HISTORY_PATH = (
    RESULTS_DIR
    / "q_learning_stability_training_history.csv"
)

EXACT_SEARCH_PATH = (
    RESULTS_DIR
    / "q_learning_exact_route_search.csv"
)

OVERVIEW_PATH = (
    RESULTS_DIR
    / "q_learning_stability_overview.csv"
)


WAVE_MINUTES = 120

MIN_WORKLOAD_SIZE = 5

TRAINING_EPISODES = 5000

RANDOM_SEEDS = [
    1,
    7,
    13,
    21,
    42,
    99,
    123,
    2026,
    31415,
    27182,
]


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

    if result.empty:
        raise ValueError(
            "No deliveries were loaded."
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


def select_same_small_workload(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use exactly the same workload-selection
    rule as Phase 8C:

        1. wave size >= 5
        2. choose smallest eligible wave size
        3. choose earliest wave when tied
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

    candidates = (
        eligible[
            eligible
            == minimum_size
        ]
        .sort_index()
    )

    selected_wave = (
        candidates.index[
            0
        ]
    )

    workload = (
        dataframe[
            dataframe[
                "wave_start"
            ]
            == selected_wave
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


def evaluate_action_order(
    deliveries,
    road_matrix,
    start_time,
    action_order,
):
    """
    Execute one fixed action permutation
    using exactly the same environment and
    reward function as Q-Learning.

    Therefore the result is directly
    comparable with a learned policy.
    """

    environment = (
        RoutingEnvironment(
            deliveries=deliveries,
            road_matrix=road_matrix,
            start_time=start_time,
        )
    )

    environment.reset()

    total_reward = 0.0

    delivery_order = []

    for action in action_order:
        (
            _,
            reward,
            _,
            info,
        ) = environment.step(
            int(
                action
            )
        )

        total_reward += (
            reward
        )

        delivery_order.append(
            str(
                info.delivery_id
            )
        )

    return {
        "total_reward":
            float(
                total_reward
            ),

        "total_distance_km":
            float(
                environment
                .total_distance_km
            ),

        "total_travel_time_minutes":
            float(
                environment
                .total_travel_time_minutes
            ),

        "action_order":
            tuple(
                int(
                    action
                )
                for action
                in action_order
            ),

        "delivery_order":
            tuple(
                delivery_order
            ),
    }


def exhaustive_search(
    deliveries,
    road_matrix,
    start_time,
) -> pd.DataFrame:
    """
    Exhaustively evaluate every possible
    route.

    With 5 deliveries:

        5! = 120 routes

    This gives the exact optimum for the
    current reward function.
    """

    delivery_count = len(
        deliveries
    )

    number_of_routes = factorial(
        delivery_count
    )

    print()
    print(
        "========================================"
    )

    print(
        "Exact Reward Search"
    )

    print(
        "========================================"
    )

    print(
        f"Deliveries: "
        f"{delivery_count}"
    )

    print(
        f"Possible routes: "
        f"{number_of_routes}"
    )

    rows = []

    for action_order in permutations(
        range(
            delivery_count
        )
    ):
        result = (
            evaluate_action_order(
                deliveries=deliveries,
                road_matrix=road_matrix,
                start_time=start_time,
                action_order=action_order,
            )
        )

        rows.append(
            {
                "total_reward":
                    result[
                        "total_reward"
                    ],

                "total_distance_km":
                    result[
                        "total_distance_km"
                    ],

                "total_travel_time_minutes":
                    result[
                        "total_travel_time_minutes"
                    ],

                "action_order":
                    " -> ".join(
                        str(
                            action
                        )
                        for action
                        in result[
                            "action_order"
                        ]
                    ),

                "delivery_order":
                    " -> ".join(
                        result[
                            "delivery_order"
                        ]
                    ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe = (
        dataframe
        .sort_values(
            by=[
                "total_reward",
                "total_distance_km",
                "total_travel_time_minutes",
                "action_order",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    dataframe[
        "exact_rank"
    ] = (
        dataframe.index
        + 1
    )

    return dataframe


def build_training_rows(
    seed: int,
    training_result,
):
    rows = []

    for episode in (
        training_result.episodes
    ):
        rows.append(
            {
                "seed":
                    seed,

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

                "delivery_order":
                    " -> ".join(
                        episode.delivery_order
                    ),
            }
        )

    return rows


def calculate_convergence_metrics(
    training_df: pd.DataFrame,
):
    """
    Measure how training behaves near the
    end of the run.

    Note:
        epsilon remains at epsilon_min,
        therefore episode reward still
        contains some exploration noise.
    """

    rewards = (
        training_df[
            "total_reward"
        ]
        .astype(
            float
        )
    )

    first_100 = (
        rewards.head(
            100
        )
    )

    last_100 = (
        rewards.tail(
            100
        )
    )

    last_500 = (
        rewards.tail(
            500
        )
    )

    rolling_100 = (
        rewards
        .rolling(
            window=100,
            min_periods=100,
        )
        .mean()
    )

    valid_rolling = (
        rolling_100.dropna()
    )

    if valid_rolling.empty:
        best_rolling_100 = np.nan
        final_rolling_100 = np.nan
        gap_percent = np.nan

    else:
        best_rolling_100 = float(
            valid_rolling.max()
        )

        final_rolling_100 = float(
            valid_rolling.iloc[
                -1
            ]
        )

        if abs(
            best_rolling_100
        ) > 1e-12:
            gap_percent = (
                (
                    best_rolling_100
                    - final_rolling_100
                )
                / abs(
                    best_rolling_100
                )
                * 100.0
            )

        else:
            gap_percent = np.nan

    return {
        "mean_first_100_reward":
            float(
                first_100.mean()
            ),

        "mean_last_100_reward":
            float(
                last_100.mean()
            ),

        "std_last_100_reward":
            float(
                last_100.std(
                    ddof=0
                )
            ),

        "mean_last_500_reward":
            float(
                last_500.mean()
            ),

        "std_last_500_reward":
            float(
                last_500.std(
                    ddof=0
                )
            ),

        "best_episode_reward":
            float(
                rewards.max()
            ),

        "best_rolling_100_reward":
            best_rolling_100,

        "final_rolling_100_reward":
            final_rolling_100,

        "rolling_100_gap_percent":
            gap_percent,
    }


def evaluate_delivery_order(
    evaluator,
    deliveries,
    delivery_nodes,
    delivery_order,
    depot_node,
    start_time,
):
    return evaluator.evaluate(
        algorithm="q_learning",
        depot_node=depot_node,
        deliveries=deliveries,
        delivery_nodes=delivery_nodes,
        delivery_order=delivery_order,
        start_time=start_time,
        return_to_depot=True,
    )


def train_seed(
    seed: int,
    deliveries,
    delivery_nodes,
    road_matrix,
    evaluator,
    depot_node: int,
    start_time,
    exact_best_reward: float,
    exact_best_route: str,
):
    environment = (
        RoutingEnvironment(
            deliveries=deliveries,
            road_matrix=road_matrix,
            start_time=start_time,
        )
    )

    config = QLearningConfig(
        learning_rate=0.20,
        discount_factor=0.95,
        epsilon_start=1.00,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        episodes=TRAINING_EPISODES,
        random_seed=seed,
    )

    agent = QLearningAgent(
        config
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

    rollout_timer = (
        perf_counter()
    )

    rollout = (
        agent.greedy_rollout(
            environment
        )
    )

    rollout_runtime = (
        perf_counter()
        - rollout_timer
    )

    evaluation = (
        evaluate_delivery_order(
            evaluator=evaluator,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=(
                rollout.delivery_order
            ),
            depot_node=depot_node,
            start_time=start_time,
        )
    )

    training_rows = (
        build_training_rows(
            seed=seed,
            training_result=(
                training_result
            ),
        )
    )

    training_df = pd.DataFrame(
        training_rows
    )

    convergence = (
        calculate_convergence_metrics(
            training_df
        )
    )

    learned_route = " -> ".join(
        rollout.delivery_order
    )

    reward_gap = (
        exact_best_reward
        - rollout.total_reward
    )

    if abs(
        exact_best_reward
    ) > 1e-12:
        reward_gap_percent = (
            reward_gap
            / abs(
                exact_best_reward
            )
            * 100.0
        )
    else:
        reward_gap_percent = np.nan

    optimal_reward_match = bool(
        np.isclose(
            rollout.total_reward,
            exact_best_reward,
            atol=1e-9,
            rtol=1e-9,
        )
    )

    exact_route_match = (
        learned_route
        == exact_best_route
    )

    summary = {
        "seed":
            seed,

        "training_episodes":
            TRAINING_EPISODES,

        "training_runtime_seconds":
            training_runtime,

        "rollout_runtime_seconds":
            rollout_runtime,

        "final_epsilon":
            training_result.final_epsilon,

        "q_table_size":
            training_result.q_table_size,

        "greedy_reward":
            rollout.total_reward,

        "exact_best_reward":
            exact_best_reward,

        "reward_gap":
            reward_gap,

        "reward_gap_percent":
            reward_gap_percent,

        "optimal_reward_match":
            optimal_reward_match,

        "exact_route_match":
            exact_route_match,

        "greedy_distance_km":
            evaluation.total_distance_km,

        "greedy_travel_time_minutes":
            evaluation.total_travel_time_minutes,

        "on_time_deliveries":
            evaluation.on_time_deliveries,

        "late_deliveries":
            evaluation.late_deliveries,

        "on_time_rate":
            evaluation.on_time_rate,

        "total_lateness_minutes":
            evaluation.total_lateness_minutes,

        "greedy_route":
            learned_route,
    }

    summary.update(
        convergence
    )

    return (
        summary,
        training_rows,
    )


def print_exact_result(
    exact_df: pd.DataFrame,
):
    best = exact_df.iloc[
        0
    ]

    print()
    print(
        "========================================"
    )

    print(
        "Exact Best Route"
    )

    print(
        "========================================"
    )

    print(
        f"Reward: "
        f"{best['total_reward']:.4f}"
    )

    print(
        f"Distance: "
        f"{best['total_distance_km']:.3f} km"
    )

    print(
        f"Travel time: "
        f"{best['total_travel_time_minutes']:.2f} min"
    )

    print(
        f"Route: "
        f"{best['delivery_order']}"
    )


def print_seed_summary(
    summary_df: pd.DataFrame,
):
    print()
    print(
        "========================================"
    )

    print(
        "Q-Learning Multi-seed Stability"
    )

    print(
        "========================================"
    )

    columns = [
        "seed",
        "greedy_reward",
        "reward_gap",
        "reward_gap_percent",
        "optimal_reward_match",
        "exact_route_match",
        "greedy_distance_km",
        "greedy_travel_time_minutes",
        "on_time_deliveries",
        "late_deliveries",
        "q_table_size",
        "training_runtime_seconds",
        "mean_first_100_reward",
        "mean_last_100_reward",
        "std_last_100_reward",
        "rolling_100_gap_percent",
    ]

    print(
        summary_df[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


def build_overview(
    summary_df: pd.DataFrame,
    exact_df: pd.DataFrame,
):
    exact_best = (
        exact_df.iloc[
            0
        ]
    )

    route_counts = (
        summary_df[
            "greedy_route"
        ]
        .value_counts()
    )

    most_common_route = (
        route_counts.index[
            0
        ]
    )

    most_common_count = int(
        route_counts.iloc[
            0
        ]
    )

    return pd.DataFrame(
        [
            {
                "random_seeds":
                    len(
                        summary_df
                    ),

                "training_episodes_per_seed":
                    TRAINING_EPISODES,

                "exact_routes_evaluated":
                    len(
                        exact_df
                    ),

                "exact_best_reward":
                    exact_best[
                        "total_reward"
                    ],

                "exact_best_distance_km":
                    exact_best[
                        "total_distance_km"
                    ],

                "exact_best_travel_time_minutes":
                    exact_best[
                        "total_travel_time_minutes"
                    ],

                "exact_best_route":
                    exact_best[
                        "delivery_order"
                    ],

                "optimal_reward_seed_count":
                    int(
                        summary_df[
                            "optimal_reward_match"
                        ].sum()
                    ),

                "exact_route_seed_count":
                    int(
                        summary_df[
                            "exact_route_match"
                        ].sum()
                    ),

                "unique_greedy_routes":
                    int(
                        summary_df[
                            "greedy_route"
                        ].nunique()
                    ),

                "most_common_greedy_route":
                    most_common_route,

                "most_common_route_seed_count":
                    most_common_count,

                "mean_greedy_reward":
                    float(
                        summary_df[
                            "greedy_reward"
                        ].mean()
                    ),

                "std_greedy_reward":
                    float(
                        summary_df[
                            "greedy_reward"
                        ].std(
                            ddof=0
                        )
                    ),

                "mean_reward_gap":
                    float(
                        summary_df[
                            "reward_gap"
                        ].mean()
                    ),

                "max_reward_gap":
                    float(
                        summary_df[
                            "reward_gap"
                        ].max()
                    ),

                "mean_training_runtime_seconds":
                    float(
                        summary_df[
                            "training_runtime_seconds"
                        ].mean()
                    ),

                "mean_q_table_size":
                    float(
                        summary_df[
                            "q_table_size"
                        ].mean()
                    ),

                "mean_last_100_reward":
                    float(
                        summary_df[
                            "mean_last_100_reward"
                        ].mean()
                    ),

                "mean_last_100_std":
                    float(
                        summary_df[
                            "std_last_100_reward"
                        ].mean()
                    ),
            }
        ]
    )


def print_overview(
    overview_df: pd.DataFrame,
):
    row = overview_df.iloc[
        0
    ]

    print()
    print(
        "========================================"
    )

    print(
        "Stability Overview"
    )

    print(
        "========================================"
    )

    print(
        f"Seeds: "
        f"{int(row['random_seeds'])}"
    )

    print(
        f"Exact routes evaluated: "
        f"{int(row['exact_routes_evaluated'])}"
    )

    print(
        f"Optimal reward seeds: "
        f"{int(row['optimal_reward_seed_count'])}"
        f"/"
        f"{int(row['random_seeds'])}"
    )

    print(
        f"Exact route seeds: "
        f"{int(row['exact_route_seed_count'])}"
        f"/"
        f"{int(row['random_seeds'])}"
    )

    print(
        f"Unique greedy routes: "
        f"{int(row['unique_greedy_routes'])}"
    )

    print(
        f"Mean greedy reward: "
        f"{row['mean_greedy_reward']:.4f}"
    )

    print(
        f"Reward std across seeds: "
        f"{row['std_greedy_reward']:.4f}"
    )

    print(
        f"Mean reward gap: "
        f"{row['mean_reward_gap']:.4f}"
    )

    print(
        f"Max reward gap: "
        f"{row['max_reward_gap']:.4f}"
    )

    print(
        f"Mean training runtime: "
        f"{row['mean_training_runtime_seconds']:.4f} sec"
    )

    print(
        f"Mean Q-table size: "
        f"{row['mean_q_table_size']:.1f}"
    )

    print(
        f"Mean last-100 reward: "
        f"{row['mean_last_100_reward']:.4f}"
    )

    print(
        f"Mean last-100 std: "
        f"{row['mean_last_100_std']:.4f}"
    )

    print()
    print(
        "Most common learned route:"
    )

    print(
        row[
            "most_common_greedy_route"
        ]
    )

    print(
        f"Seeds using this route: "
        f"{int(row['most_common_route_seed_count'])}"
    )


def save_results(
    exact_df,
    seed_summary_df,
    training_history_df,
    overview_df,
):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    exact_df.to_csv(
        EXACT_SEARCH_PATH,
        index=False,
    )

    seed_summary_df.to_csv(
        SEED_SUMMARY_PATH,
        index=False,
    )

    training_history_df.to_csv(
        TRAINING_HISTORY_PATH,
        index=False,
    )

    overview_df.to_csv(
        OVERVIEW_PATH,
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
        EXACT_SEARCH_PATH
    )

    print(
        SEED_SUMMARY_PATH
    )

    print(
        TRAINING_HISTORY_PATH
    )

    print(
        OVERVIEW_PATH
    )


def main():
    print()
    print(
        "========================================"
    )

    print(
        "Q-Learning Convergence & Stability"
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
        select_same_small_workload(
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
        "Preparing road matrix..."
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

    evaluator = (
        RouteEvaluator(
            road_network=service
        )
    )

    evaluator.matrix_builder = (
        matrix_builder
    )

    exact_df = exhaustive_search(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=wave_end,
    )

    print_exact_result(
        exact_df
    )

    exact_best_reward = float(
        exact_df.iloc[
            0
        ][
            "total_reward"
        ]
    )

    exact_best_route = str(
        exact_df.iloc[
            0
        ][
            "delivery_order"
        ]
    )

    print()
    print(
        "========================================"
    )

    print(
        "Multi-seed Training"
    )

    print(
        "========================================"
    )

    seed_summaries = []

    training_rows = []

    for index, seed in enumerate(
        RANDOM_SEEDS,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(RANDOM_SEEDS)}] "
            f"Seed {seed}"
        )

        (
            summary,
            seed_training_rows,
        ) = train_seed(
            seed=seed,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            road_matrix=road_matrix,
            evaluator=evaluator,
            depot_node=depot_node,
            start_time=wave_end,
            exact_best_reward=(
                exact_best_reward
            ),
            exact_best_route=(
                exact_best_route
            ),
        )

        seed_summaries.append(
            summary
        )

        training_rows.extend(
            seed_training_rows
        )

        print(
            f"  Greedy reward: "
            f"{summary['greedy_reward']:.4f}"
        )

        print(
            f"  Reward gap: "
            f"{summary['reward_gap']:.4f}"
        )

        print(
            f"  Optimal reward: "
            f"{summary['optimal_reward_match']}"
        )

        print(
            f"  Exact route: "
            f"{summary['exact_route_match']}"
        )

        print(
            f"  Route: "
            f"{summary['greedy_route']}"
        )

        print(
            f"  Runtime: "
            f"{summary['training_runtime_seconds']:.3f}s"
        )

    seed_summary_df = pd.DataFrame(
        seed_summaries
    )

    training_history_df = pd.DataFrame(
        training_rows
    )

    overview_df = (
        build_overview(
            summary_df=(
                seed_summary_df
            ),
            exact_df=exact_df,
        )
    )

    print_seed_summary(
        seed_summary_df
    )

    print_overview(
        overview_df
    )

    save_results(
        exact_df=exact_df,
        seed_summary_df=(
            seed_summary_df
        ),
        training_history_df=(
            training_history_df
        ),
        overview_df=overview_df,
    )


if __name__ == "__main__":
    main()