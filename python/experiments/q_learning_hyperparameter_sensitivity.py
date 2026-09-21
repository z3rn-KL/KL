from itertools import permutations
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
    RewardConfig,
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

RUN_OUTPUT_PATH = (
    RESULTS_DIR
    / "q_learning_hyperparameter_runs.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "q_learning_hyperparameter_summary.csv"
)

EXACT_OUTPUT_PATH = (
    RESULTS_DIR
    / "q_learning_hyperparameter_exact_routes.csv"
)


WAVE_MINUTES = 120
MIN_WORKLOAD_SIZE = 5

RANDOM_SEEDS = [
    1,
    7,
    13,
    21,
    42,
]


BASELINE = {
    "learning_rate": 0.20,
    "discount_factor": 0.95,
    "epsilon_start": 1.00,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.998,
    "episodes": 5000,
}


CONFIGURATIONS = [
    {
        "configuration": "baseline",
        **BASELINE,
    },

    # Learning-rate sensitivity
    {
        "configuration": "alpha_0.05",
        **BASELINE,
        "learning_rate": 0.05,
    },
    {
        "configuration": "alpha_0.10",
        **BASELINE,
        "learning_rate": 0.10,
    },
    {
        "configuration": "alpha_0.50",
        **BASELINE,
        "learning_rate": 0.50,
    },

    # Discount-factor sensitivity
    {
        "configuration": "gamma_0.80",
        **BASELINE,
        "discount_factor": 0.80,
    },
    {
        "configuration": "gamma_0.90",
        **BASELINE,
        "discount_factor": 0.90,
    },
    {
        "configuration": "gamma_0.99",
        **BASELINE,
        "discount_factor": 0.99,
    },

    # Exploration-decay sensitivity
    {
        "configuration": "epsilon_decay_0.990",
        **BASELINE,
        "epsilon_decay": 0.990,
    },
    {
        "configuration": "epsilon_decay_0.995",
        **BASELINE,
        "epsilon_decay": 0.995,
    },
    {
        "configuration": "epsilon_decay_0.9995",
        **BASELINE,
        "epsilon_decay": 0.9995,
    },

    # Number-of-episodes sensitivity
    {
        "configuration": "episodes_500",
        **BASELINE,
        "episodes": 500,
    },
    {
        "configuration": "episodes_1000",
        **BASELINE,
        "episodes": 1000,
    },
    {
        "configuration": "episodes_2500",
        **BASELINE,
        "episodes": 2500,
    },
]


def load_road_network() -> RoadNetworkService:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing road graph: {GRAPH_PATH}"
        )

    print()
    print("Loading cached road graph...")

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
    rule as Phases 8C and 8D.
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

    return (
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


def route_to_string(
    delivery_order,
) -> str:
    return " -> ".join(
        str(delivery_id)
        for delivery_id
        in delivery_order
    )


def evaluate_fixed_action_order(
    deliveries,
    road_matrix,
    start_time,
    action_order,
):
    environment = RoutingEnvironment(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=start_time,
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
            int(action)
        )

        total_reward += reward

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

        "distance_km":
            float(
                environment
                .total_distance_km
            ),

        "travel_time_minutes":
            float(
                environment
                .total_travel_time_minutes
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
    Because this workload contains only
    five deliveries, all 5! = 120 routes
    can be evaluated exactly.
    """

    rows = []

    for action_order in permutations(
        range(
            len(
                deliveries
            )
        )
    ):
        result = (
            evaluate_fixed_action_order(
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
                        "distance_km"
                    ],

                "total_travel_time_minutes":
                    result[
                        "travel_time_minutes"
                    ],

                "delivery_order":
                    route_to_string(
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
                "delivery_order",
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


def build_exact_lookup(
    exact_df: pd.DataFrame,
) -> dict:
    lookup = {}

    for _, row in (
        exact_df.iterrows()
    ):
        lookup[
            str(
                row[
                    "delivery_order"
                ]
            )
        ] = {
            "exact_rank":
                int(
                    row[
                        "exact_rank"
                    ]
                ),

            "exact_reward":
                float(
                    row[
                        "total_reward"
                    ]
                ),
        }

    return lookup


def evaluate_rollout(
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


def run_training(
    configuration: dict,
    seed: int,
    deliveries,
    delivery_nodes,
    road_matrix,
    evaluator,
    depot_node,
    start_time,
    exact_best_reward: float,
    exact_lookup: dict,
):
    config = QLearningConfig(
        learning_rate=float(
            configuration[
                "learning_rate"
            ]
        ),
        discount_factor=float(
            configuration[
                "discount_factor"
            ]
        ),
        epsilon_start=float(
            configuration[
                "epsilon_start"
            ]
        ),
        epsilon_min=float(
            configuration[
                "epsilon_min"
            ]
        ),
        epsilon_decay=float(
            configuration[
                "epsilon_decay"
            ]
        ),
        episodes=int(
            configuration[
                "episodes"
            ]
        ),
        random_seed=seed,
    )

    environment = RoutingEnvironment(
        deliveries=deliveries,
        road_matrix=road_matrix,
        start_time=start_time,
    )

    agent = QLearningAgent(
        config
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

    evaluation = (
        evaluate_rollout(
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

    learned_route = (
        route_to_string(
            rollout.delivery_order
        )
    )

    exact_data = (
        exact_lookup.get(
            learned_route
        )
    )

    if exact_data is None:
        exact_rank = np.nan

    else:
        exact_rank = (
            exact_data[
                "exact_rank"
            ]
        )

    reward_gap = (
        exact_best_reward
        - rollout.total_reward
    )

    optimal_reward_match = bool(
        np.isclose(
            rollout.total_reward,
            exact_best_reward,
            atol=1e-9,
            rtol=1e-9,
        )
    )

    return {
        "configuration":
            configuration[
                "configuration"
            ],

        "seed":
            seed,

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

        "episodes":
            config.episodes,

        "final_epsilon":
            training_result.final_epsilon,

        "q_table_size":
            training_result.q_table_size,

        "training_runtime_seconds":
            training_runtime,

        "rollout_runtime_seconds":
            rollout_runtime,

        "greedy_reward":
            rollout.total_reward,

        "exact_best_reward":
            exact_best_reward,

        "reward_gap":
            reward_gap,

        "optimal_reward_match":
            optimal_reward_match,

        "exact_rank":
            exact_rank,

        "distance_km":
            evaluation.total_distance_km,

        "travel_time_minutes":
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


def summarize_configuration(
    dataframe: pd.DataFrame,
) -> dict:
    first = dataframe.iloc[
        0
    ]

    run_count = len(
        dataframe
    )

    optimal_hits = int(
        dataframe[
            "optimal_reward_match"
        ].sum()
    )

    return {
        "configuration":
            first[
                "configuration"
            ],

        "learning_rate":
            first[
                "learning_rate"
            ],

        "discount_factor":
            first[
                "discount_factor"
            ],

        "epsilon_start":
            first[
                "epsilon_start"
            ],

        "epsilon_min":
            first[
                "epsilon_min"
            ],

        "epsilon_decay":
            first[
                "epsilon_decay"
            ],

        "episodes":
            int(
                first[
                    "episodes"
                ]
            ),

        "runs":
            run_count,

        "optimal_hit_count":
            optimal_hits,

        "optimal_hit_rate":
            (
                optimal_hits
                / run_count
            ),

        "mean_exact_rank":
            float(
                dataframe[
                    "exact_rank"
                ].mean()
            ),

        "median_exact_rank":
            float(
                dataframe[
                    "exact_rank"
                ].median()
            ),

        "mean_greedy_reward":
            float(
                dataframe[
                    "greedy_reward"
                ].mean()
            ),

        "std_greedy_reward":
            float(
                dataframe[
                    "greedy_reward"
                ].std(
                    ddof=0
                )
            ),

        "mean_reward_gap":
            float(
                dataframe[
                    "reward_gap"
                ].mean()
            ),

        "max_reward_gap":
            float(
                dataframe[
                    "reward_gap"
                ].max()
            ),

        "mean_distance_km":
            float(
                dataframe[
                    "distance_km"
                ].mean()
            ),

        "mean_travel_time_minutes":
            float(
                dataframe[
                    "travel_time_minutes"
                ].mean()
            ),

        "mean_on_time_rate":
            float(
                dataframe[
                    "on_time_rate"
                ].mean()
            ),

        "mean_late_deliveries":
            float(
                dataframe[
                    "late_deliveries"
                ].mean()
            ),

        "mean_total_lateness_minutes":
            float(
                dataframe[
                    "total_lateness_minutes"
                ].mean()
            ),

        "mean_training_runtime_seconds":
            float(
                dataframe[
                    "training_runtime_seconds"
                ].mean()
            ),

        "mean_q_table_size":
            float(
                dataframe[
                    "q_table_size"
                ].mean()
            ),

        "unique_greedy_routes":
            int(
                dataframe[
                    "greedy_route"
                ].nunique()
            ),
    }


def build_summary(
    runs_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for configuration in (
        runs_df[
            "configuration"
        ].unique()
    ):
        dataframe = (
            runs_df[
                runs_df[
                    "configuration"
                ]
                == configuration
            ]
            .copy()
        )

        rows.append(
            summarize_configuration(
                dataframe
            )
        )

    summary = pd.DataFrame(
        rows
    )

    summary = (
        summary
        .sort_values(
            by=[
                "optimal_hit_rate",
                "mean_reward_gap",
                "mean_exact_rank",
                "mean_training_runtime_seconds",
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

    summary[
        "sensitivity_rank"
    ] = (
        summary.index
        + 1
    )

    return summary


def print_reward_configuration():
    reward = RewardConfig()

    print()
    print(
        "========================================"
    )

    print(
        "Fixed Reward Configuration"
    )

    print(
        "========================================"
    )

    print(
        f"Priority bonus: "
        f"{reward.priority_bonus}"
    )

    print(
        f"Travel efficiency bonus: "
        f"{reward.travel_efficiency_bonus}"
    )

    print(
        f"Distance penalty: "
        f"{reward.distance_penalty}"
    )

    print(
        f"Travel-time penalty: "
        f"{reward.travel_time_penalty}"
    )

    print(
        f"Late-delivery penalty: "
        f"{reward.late_delivery_penalty}"
    )

    print(
        f"Lateness penalty: "
        f"{reward.lateness_penalty}"
    )


def print_exact_result(
    exact_df: pd.DataFrame,
):
    best = exact_df.iloc[
        0
    ]

    best_reward = float(
        best[
            "total_reward"
        ]
    )

    optimal_count = int(
        np.isclose(
            exact_df[
                "total_reward"
            ],
            best_reward,
            atol=1e-9,
            rtol=1e-9,
        ).sum()
    )

    print()
    print(
        "========================================"
    )

    print(
        "Exact Reward Reference"
    )

    print(
        "========================================"
    )

    print(
        f"Routes evaluated: "
        f"{len(exact_df)}"
    )

    print(
        f"Best reward: "
        f"{best_reward:.4f}"
    )

    print(
        f"Number of optimal routes: "
        f"{optimal_count}"
    )

    print(
        f"Best distance: "
        f"{best['total_distance_km']:.3f} km"
    )

    print(
        f"Best travel time: "
        f"{best['total_travel_time_minutes']:.2f} min"
    )

    print(
        f"Best route: "
        f"{best['delivery_order']}"
    )


def print_summary(
    summary_df: pd.DataFrame,
):
    print()
    print(
        "========================================"
    )

    print(
        "Hyperparameter Sensitivity Summary"
    )

    print(
        "========================================"
    )

    columns = [
        "sensitivity_rank",
        "configuration",
        "learning_rate",
        "discount_factor",
        "epsilon_decay",
        "episodes",
        "optimal_hit_count",
        "optimal_hit_rate",
        "mean_exact_rank",
        "mean_reward_gap",
        "mean_distance_km",
        "mean_travel_time_minutes",
        "mean_on_time_rate",
        "mean_training_runtime_seconds",
        "mean_q_table_size",
        "unique_greedy_routes",
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


def save_results(
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    exact_df: pd.DataFrame,
):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs_df.to_csv(
        RUN_OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    exact_df.to_csv(
        EXACT_OUTPUT_PATH,
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
        RUN_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        EXACT_OUTPUT_PATH
    )


def main():
    print()
    print(
        "========================================"
    )

    print(
        "Q-Learning Hyperparameter Sensitivity"
    )

    print(
        "========================================"
    )

    print(
        f"Configurations: "
        f"{len(CONFIGURATIONS)}"
    )

    print(
        f"Seeds per configuration: "
        f"{len(RANDOM_SEEDS)}"
    )

    print(
        f"Total training runs: "
        f"{len(CONFIGURATIONS) * len(RANDOM_SEEDS)}"
    )

    print_reward_configuration()

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

    print()
    print(
        "Calculating exact reward reference..."
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

    exact_lookup = (
        build_exact_lookup(
            exact_df
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        "Sensitivity Runs"
    )

    print(
        "========================================"
    )

    run_rows = []

    total_runs = (
        len(
            CONFIGURATIONS
        )
        * len(
            RANDOM_SEEDS
        )
    )

    run_number = 0

    for configuration in CONFIGURATIONS:
        print()
        print(
            "----------------------------------------"
        )

        print(
            f"Configuration: "
            f"{configuration['configuration']}"
        )

        print(
            "----------------------------------------"
        )

        print(
            f"alpha="
            f"{configuration['learning_rate']} "
            f"| gamma="
            f"{configuration['discount_factor']} "
            f"| decay="
            f"{configuration['epsilon_decay']} "
            f"| episodes="
            f"{configuration['episodes']}"
        )

        for seed in RANDOM_SEEDS:
            run_number += 1

            print(
                f"  [{run_number}/{total_runs}] "
                f"seed={seed}"
            )

            result = run_training(
                configuration=configuration,
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
                exact_lookup=exact_lookup,
            )

            run_rows.append(
                result
            )

            print(
                f"    reward="
                f"{result['greedy_reward']:.4f} "
                f"| gap="
                f"{result['reward_gap']:.4f} "
                f"| rank="
                f"{result['exact_rank']} "
                f"| optimal="
                f"{result['optimal_reward_match']} "
                f"| runtime="
                f"{result['training_runtime_seconds']:.3f}s"
            )

    runs_df = pd.DataFrame(
        run_rows
    )

    summary_df = build_summary(
        runs_df
    )

    print_summary(
        summary_df
    )

    save_results(
        runs_df=runs_df,
        summary_df=summary_df,
        exact_df=exact_df,
    )


if __name__ == "__main__":
    main()