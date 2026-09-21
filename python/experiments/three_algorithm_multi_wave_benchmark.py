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

BENCHMARK_OUTPUT_PATH = (
    RESULTS_DIR
    / "three_algorithm_multi_wave_benchmark.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "three_algorithm_multi_wave_summary.csv"
)

COMPARISON_OUTPUT_PATH = (
    RESULTS_DIR
    / "three_algorithm_multi_wave_comparison.csv"
)

DETAIL_OUTPUT_PATH = (
    RESULTS_DIR
    / "three_algorithm_multi_wave_details.csv"
)

SKIPPED_OUTPUT_PATH = (
    RESULTS_DIR
    / "three_algorithm_multi_wave_skipped.csv"
)


WAVE_MINUTES = 120
MIN_WAVE_SIZE = 5


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
            f"Missing SCC snapped-node file: "
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

    print()
    print(
        f"Deliveries with missing road node: "
        f"{missing_nodes}"
    )

    if missing_nodes > 0:
        raise ValueError(
            "SCC road-node file is incomplete."
        )

    dataframe[
        "road_node"
    ] = dataframe[
        "road_node"
    ].astype(
        int
    )

    return dataframe


def get_eligible_waves(
    dataframe: pd.DataFrame,
):
    waves = []

    grouped = dataframe.groupby(
        "wave_start",
        sort=True,
    )

    for wave_start, wave_df in grouped:
        if len(
            wave_df
        ) < MIN_WAVE_SIZE:
            continue

        waves.append(
            (
                wave_start,
                wave_df.copy(),
            )
        )

    return waves


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


def build_wave_workload(
    wave_df: pd.DataFrame,
    delivery_lookup: dict,
):
    deliveries = []
    delivery_nodes = []

    for _, row in wave_df.iterrows():
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


def build_benchmark_row(
    wave_start,
    wave_end,
    algorithm: str,
    evaluation,
    routing_runtime_seconds: float,
    matrix_runtime_seconds: float,
    extra: dict | None = None,
) -> dict:
    row = {
        "wave_start":
            wave_start,

        "wave_end":
            wave_end,

        "algorithm":
            algorithm,

        "number_of_deliveries":
            evaluation.number_of_deliveries,

        "deliveries_with_deadline":
            evaluation.deliveries_with_deadline,

        "deliveries_without_deadline":
            evaluation.deliveries_without_deadline,

        "on_time_deliveries":
            evaluation.on_time_deliveries,

        "late_deliveries":
            evaluation.late_deliveries,

        "on_time_rate":
            (
                np.nan
                if evaluation.on_time_rate
                is None
                else evaluation.on_time_rate
            ),

        "total_lateness_minutes":
            evaluation.total_lateness_minutes,

        "mean_lateness_minutes":
            evaluation.mean_lateness_minutes,

        "max_lateness_minutes":
            evaluation.max_lateness_minutes,

        "total_distance_km":
            evaluation.total_distance_km,

        "total_travel_time_minutes":
            evaluation.total_travel_time_minutes,

        "routing_runtime_seconds":
            routing_runtime_seconds,

        "matrix_preparation_seconds":
            matrix_runtime_seconds,

        "returned_to_depot":
            evaluation.returned_to_depot,
    }

    if extra:
        row.update(
            extra
        )

    return row


def build_detail_rows(
    wave_start,
    wave_end,
    evaluation,
):
    rows = []

    for result in (
        evaluation.delivery_results
    ):
        rows.append(
            {
                "wave_start":
                    wave_start,

                "wave_end":
                    wave_end,

                "algorithm":
                    evaluation.algorithm,

                "sequence":
                    result.sequence,

                "delivery_id":
                    result.delivery_id,

                "arrival_time":
                    result.arrival_time,

                "expected_delivery_time":
                    result.expected_delivery_time,

                "has_deadline":
                    result.has_deadline,

                "on_time":
                    result.on_time,

                "lateness_minutes":
                    result.lateness_minutes,

                "leg_distance_km":
                    result.leg_distance_km,

                "leg_travel_time_minutes":
                    result.leg_travel_time_minutes,
            }
        )

    return rows


def summarize_algorithm(
    benchmark_df: pd.DataFrame,
    algorithm: str,
) -> dict:
    dataframe = benchmark_df[
        benchmark_df[
            "algorithm"
        ]
        == algorithm
    ].copy()

    total_deadline = int(
        dataframe[
            "deliveries_with_deadline"
        ].sum()
    )

    total_on_time = int(
        dataframe[
            "on_time_deliveries"
        ].sum()
    )

    total_late = int(
        dataframe[
            "late_deliveries"
        ].sum()
    )

    total_lateness = float(
        dataframe[
            "total_lateness_minutes"
        ].sum()
    )

    if total_deadline > 0:
        overall_on_time_rate = (
            total_on_time
            / total_deadline
        )
    else:
        overall_on_time_rate = np.nan

    return {
        "algorithm":
            algorithm,

        "evaluated_waves":
            len(
                dataframe
            ),

        "total_deliveries":
            int(
                dataframe[
                    "number_of_deliveries"
                ].sum()
            ),

        "deliveries_with_deadline":
            total_deadline,

        "deliveries_without_deadline":
            int(
                dataframe[
                    "deliveries_without_deadline"
                ].sum()
            ),

        "on_time_deliveries":
            total_on_time,

        "late_deliveries":
            total_late,

        "overall_on_time_rate":
            overall_on_time_rate,

        "waves_with_late_deliveries":
            int(
                (
                    dataframe[
                        "late_deliveries"
                    ]
                    > 0
                ).sum()
            ),

        "total_lateness_minutes":
            total_lateness,

        "max_lateness_minutes":
            float(
                dataframe[
                    "max_lateness_minutes"
                ].max()
            ),

        "mean_distance_km":
            float(
                dataframe[
                    "total_distance_km"
                ].mean()
            ),

        "median_distance_km":
            float(
                dataframe[
                    "total_distance_km"
                ].median()
            ),

        "total_distance_km":
            float(
                dataframe[
                    "total_distance_km"
                ].sum()
            ),

        "mean_travel_time_minutes":
            float(
                dataframe[
                    "total_travel_time_minutes"
                ].mean()
            ),

        "median_travel_time_minutes":
            float(
                dataframe[
                    "total_travel_time_minutes"
                ].median()
            ),

        "mean_routing_runtime_seconds":
            float(
                dataframe[
                    "routing_runtime_seconds"
                ].mean()
            ),

        "median_routing_runtime_seconds":
            float(
                dataframe[
                    "routing_runtime_seconds"
                ].median()
            ),
    }


def build_comparison_row(
    wave_start,
    wave_end,
    nn_row: dict,
    priority_row: dict,
    cw_row: dict,
) -> dict:
    return {
        "wave_start":
            wave_start,

        "wave_end":
            wave_end,

        "number_of_deliveries":
            nn_row[
                "number_of_deliveries"
            ],

        "nn_distance_km":
            nn_row[
                "total_distance_km"
            ],

        "priority_nn_distance_km":
            priority_row[
                "total_distance_km"
            ],

        "clarke_wright_distance_km":
            cw_row[
                "total_distance_km"
            ],

        "cw_vs_nn_distance_change_km":
            (
                cw_row[
                    "total_distance_km"
                ]
                - nn_row[
                    "total_distance_km"
                ]
            ),

        "cw_vs_priority_distance_change_km":
            (
                cw_row[
                    "total_distance_km"
                ]
                - priority_row[
                    "total_distance_km"
                ]
            ),

        "nn_travel_time_minutes":
            nn_row[
                "total_travel_time_minutes"
            ],

        "priority_nn_travel_time_minutes":
            priority_row[
                "total_travel_time_minutes"
            ],

        "clarke_wright_travel_time_minutes":
            cw_row[
                "total_travel_time_minutes"
            ],

        "nn_on_time_deliveries":
            nn_row[
                "on_time_deliveries"
            ],

        "priority_nn_on_time_deliveries":
            priority_row[
                "on_time_deliveries"
            ],

        "clarke_wright_on_time_deliveries":
            cw_row[
                "on_time_deliveries"
            ],

        "nn_late_deliveries":
            nn_row[
                "late_deliveries"
            ],

        "priority_nn_late_deliveries":
            priority_row[
                "late_deliveries"
            ],

        "clarke_wright_late_deliveries":
            cw_row[
                "late_deliveries"
            ],

        "nn_total_lateness_minutes":
            nn_row[
                "total_lateness_minutes"
            ],

        "priority_nn_total_lateness_minutes":
            priority_row[
                "total_lateness_minutes"
            ],

        "clarke_wright_total_lateness_minutes":
            cw_row[
                "total_lateness_minutes"
            ],

        "nn_runtime_seconds":
            nn_row[
                "routing_runtime_seconds"
            ],

        "priority_nn_runtime_seconds":
            priority_row[
                "routing_runtime_seconds"
            ],

        "clarke_wright_runtime_seconds":
            cw_row[
                "routing_runtime_seconds"
            ],

        "clarke_wright_savings_km":
            cw_row.get(
                "clarke_wright_savings_km",
                np.nan,
            ),

        "clarke_wright_merge_count":
            cw_row.get(
                "clarke_wright_merge_count",
                np.nan,
            ),
    }


def compare_pair(
    comparison_df: pd.DataFrame,
    left_column: str,
    right_column: str,
    lower_is_better: bool = True,
):
    tolerance = 1e-9

    difference = (
        comparison_df[
            left_column
        ]
        - comparison_df[
            right_column
        ]
    )

    if lower_is_better:
        left_better = int(
            (
                difference
                < -tolerance
            ).sum()
        )

        right_better = int(
            (
                difference
                > tolerance
            ).sum()
        )
    else:
        left_better = int(
            (
                difference
                > tolerance
            ).sum()
        )

        right_better = int(
            (
                difference
                < -tolerance
            ).sum()
        )

    equal = int(
        (
            difference.abs()
            <= tolerance
        ).sum()
    )

    return (
        left_better,
        right_better,
        equal,
    )


def print_pairwise_summary(
    comparison_df: pd.DataFrame,
):
    print()
    print(
        "========================================"
    )
    print(
        "Three-algorithm Pairwise Comparison"
    )
    print(
        "========================================"
    )

    print(
        f"Completed waves: "
        f"{len(comparison_df)}"
    )

    (
        cw_shorter,
        nn_shorter,
        equal_distance,
    ) = compare_pair(
        comparison_df,
        "clarke_wright_distance_km",
        "nn_distance_km",
        lower_is_better=True,
    )

    print()
    print(
        "Distance: Clarke-Wright vs NN"
    )

    print(
        f"  Clarke-Wright shorter: "
        f"{cw_shorter}"
    )

    print(
        f"  NN shorter: "
        f"{nn_shorter}"
    )

    print(
        f"  Equal: "
        f"{equal_distance}"
    )

    (
        cw_shorter_priority,
        priority_shorter,
        equal_priority_distance,
    ) = compare_pair(
        comparison_df,
        "clarke_wright_distance_km",
        "priority_nn_distance_km",
        lower_is_better=True,
    )

    print()
    print(
        "Distance: Clarke-Wright "
        "vs Priority-NN"
    )

    print(
        f"  Clarke-Wright shorter: "
        f"{cw_shorter_priority}"
    )

    print(
        f"  Priority-NN shorter: "
        f"{priority_shorter}"
    )

    print(
        f"  Equal: "
        f"{equal_priority_distance}"
    )

    (
        cw_more_on_time,
        nn_more_on_time,
        equal_on_time,
    ) = compare_pair(
        comparison_df,
        "clarke_wright_on_time_deliveries",
        "nn_on_time_deliveries",
        lower_is_better=False,
    )

    print()
    print(
        "On-time: Clarke-Wright vs NN"
    )

    print(
        f"  Clarke-Wright higher: "
        f"{cw_more_on_time}"
    )

    print(
        f"  NN higher: "
        f"{nn_more_on_time}"
    )

    print(
        f"  Equal: "
        f"{equal_on_time}"
    )

    (
        priority_more_on_time,
        cw_more_than_priority,
        equal_priority_on_time,
    ) = compare_pair(
        comparison_df,
        "priority_nn_on_time_deliveries",
        "clarke_wright_on_time_deliveries",
        lower_is_better=False,
    )

    print()
    print(
        "On-time: Priority-NN "
        "vs Clarke-Wright"
    )

    print(
        f"  Priority-NN higher: "
        f"{priority_more_on_time}"
    )

    print(
        f"  Clarke-Wright higher: "
        f"{cw_more_than_priority}"
    )

    print(
        f"  Equal: "
        f"{equal_priority_on_time}"
    )

    (
        cw_lower_lateness,
        nn_lower_lateness,
        equal_lateness,
    ) = compare_pair(
        comparison_df,
        "clarke_wright_total_lateness_minutes",
        "nn_total_lateness_minutes",
        lower_is_better=True,
    )

    print()
    print(
        "Lateness: Clarke-Wright vs NN"
    )

    print(
        f"  Clarke-Wright lower: "
        f"{cw_lower_lateness}"
    )

    print(
        f"  NN lower: "
        f"{nn_lower_lateness}"
    )

    print(
        f"  Equal: "
        f"{equal_lateness}"
    )

    (
        priority_lower_lateness,
        cw_lower_than_priority,
        equal_priority_lateness,
    ) = compare_pair(
        comparison_df,
        "priority_nn_total_lateness_minutes",
        "clarke_wright_total_lateness_minutes",
        lower_is_better=True,
    )

    print()
    print(
        "Lateness: Priority-NN "
        "vs Clarke-Wright"
    )

    print(
        f"  Priority-NN lower: "
        f"{priority_lower_lateness}"
    )

    print(
        f"  Clarke-Wright lower: "
        f"{cw_lower_than_priority}"
    )

    print(
        f"  Equal: "
        f"{equal_priority_lateness}"
    )


def print_summary(
    summary_df: pd.DataFrame,
):
    print()
    print(
        "========================================"
    )
    print(
        "Three-algorithm Multi-wave Summary"
    )
    print(
        "========================================"
    )

    columns = [
        "algorithm",
        "evaluated_waves",
        "total_deliveries",
        "on_time_deliveries",
        "late_deliveries",
        "overall_on_time_rate",
        "waves_with_late_deliveries",
        "total_lateness_minutes",
        "max_lateness_minutes",
        "mean_distance_km",
        "mean_travel_time_minutes",
        "mean_routing_runtime_seconds",
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
    benchmark_df,
    summary_df,
    comparison_df,
    details_df,
    skipped_df,
):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_df.to_csv(
        BENCHMARK_OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=False,
    )

    details_df.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )

    skipped_df.to_csv(
        SKIPPED_OUTPUT_PATH,
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
        BENCHMARK_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        COMPARISON_OUTPUT_PATH
    )

    print(
        DETAIL_OUTPUT_PATH
    )

    print(
        SKIPPED_OUTPUT_PATH
    )


def main():
    print()
    print(
        "========================================"
    )
    print(
        "XeDu Three-algorithm "
        "Multi-wave Benchmark"
    )
    print(
        "========================================"
    )

    print(
        f"Wave duration: "
        f"{WAVE_MINUTES} minutes"
    )

    print(
        f"Minimum wave size: "
        f"{MIN_WAVE_SIZE}"
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

    dataframe = prepare_dataframe(
        delivery_df=delivery_df,
        snapped_df=snapped_df,
    )

    waves = get_eligible_waves(
        dataframe
    )

    print()
    print(
        f"Eligible waves: "
        f"{len(waves)}"
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

    benchmark_rows = []
    comparison_rows = []
    detail_rows = []
    skipped_rows = []

    total_waves = len(
        waves
    )

    for wave_number, (
        wave_start,
        wave_df,
    ) in enumerate(
        waves,
        start=1,
    ):
        wave_end = (
            wave_df[
                "wave_end"
            ].iloc[
                0
            ]
        )

        print()
        print(
            f"[{wave_number}/{total_waves}] "
            f"Wave {wave_start} "
            f"| orders={len(wave_df)}"
        )

        shared_builder = None

        try:
            (
                deliveries,
                delivery_nodes,
            ) = build_wave_workload(
                wave_df=wave_df,
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

            all_nodes = (
                depot_node,
                *delivery_nodes,
            )

            shared_builder = (
                RoadMatrixBuilder(
                    service
                )
            )

            matrix_start = (
                perf_counter()
            )

            shared_builder.build(
                all_nodes
            )

            matrix_runtime = (
                perf_counter()
                - matrix_start
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
                shared_builder
            )

            priority_router.matrix_builder = (
                shared_builder
            )

            cw_router.matrix_builder = (
                shared_builder
            )

            evaluator.matrix_builder = (
                shared_builder
            )

            nn_start = perf_counter()

            nn_route = (
                nn_router.build_route(
                    depot_node=depot_node,
                    delivery_ids=(
                        delivery_ids
                    ),
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    return_to_depot=True,
                )
            )

            nn_runtime = (
                perf_counter()
                - nn_start
            )

            priority_start = (
                perf_counter()
            )

            priority_route = (
                priority_router.build_route(
                    depot_node=depot_node,
                    deliveries=deliveries,
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    start_time=wave_end,
                    return_to_depot=True,
                )
            )

            priority_runtime = (
                perf_counter()
                - priority_start
            )

            cw_start = perf_counter()

            cw_route = (
                cw_router.build_route(
                    depot_node=depot_node,
                    delivery_ids=(
                        delivery_ids
                    ),
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    return_to_depot=True,
                )
            )

            cw_runtime = (
                perf_counter()
                - cw_start
            )

            nn_evaluation = (
                evaluator.evaluate(
                    algorithm=(
                        "nearest_neighbor"
                    ),
                    depot_node=depot_node,
                    deliveries=deliveries,
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    delivery_order=(
                        nn_route.delivery_order
                    ),
                    start_time=wave_end,
                    return_to_depot=True,
                )
            )

            priority_evaluation = (
                evaluator.evaluate(
                    algorithm=(
                        "priority_nearest_neighbor"
                    ),
                    depot_node=depot_node,
                    deliveries=deliveries,
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    delivery_order=(
                        priority_route
                        .delivery_order
                    ),
                    start_time=wave_end,
                    return_to_depot=True,
                )
            )

            cw_evaluation = (
                evaluator.evaluate(
                    algorithm=(
                        "clarke_wright"
                    ),
                    depot_node=depot_node,
                    deliveries=deliveries,
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    delivery_order=(
                        cw_route.delivery_order
                    ),
                    start_time=wave_end,
                    return_to_depot=True,
                )
            )

            nn_row = build_benchmark_row(
                wave_start=wave_start,
                wave_end=wave_end,
                algorithm=(
                    "nearest_neighbor"
                ),
                evaluation=(
                    nn_evaluation
                ),
                routing_runtime_seconds=(
                    nn_runtime
                ),
                matrix_runtime_seconds=(
                    matrix_runtime
                ),
            )

            priority_row = (
                build_benchmark_row(
                    wave_start=wave_start,
                    wave_end=wave_end,
                    algorithm=(
                        "priority_nearest_neighbor"
                    ),
                    evaluation=(
                        priority_evaluation
                    ),
                    routing_runtime_seconds=(
                        priority_runtime
                    ),
                    matrix_runtime_seconds=(
                        matrix_runtime
                    ),
                )
            )

            cw_row = build_benchmark_row(
                wave_start=wave_start,
                wave_end=wave_end,
                algorithm=(
                    "clarke_wright"
                ),
                evaluation=(
                    cw_evaluation
                ),
                routing_runtime_seconds=(
                    cw_runtime
                ),
                matrix_runtime_seconds=(
                    matrix_runtime
                ),
                extra={
                    "clarke_wright_initial_distance_km":
                        cw_route.initial_separate_distance_km,

                    "clarke_wright_savings_km":
                        cw_route.total_savings_km,

                    "clarke_wright_merge_count":
                        cw_route.merge_count,
                },
            )

            benchmark_rows.extend(
                [
                    nn_row,
                    priority_row,
                    cw_row,
                ]
            )

            comparison_rows.append(
                build_comparison_row(
                    wave_start=wave_start,
                    wave_end=wave_end,
                    nn_row=nn_row,
                    priority_row=(
                        priority_row
                    ),
                    cw_row=cw_row,
                )
            )

            detail_rows.extend(
                build_detail_rows(
                    wave_start=wave_start,
                    wave_end=wave_end,
                    evaluation=nn_evaluation,
                )
            )

            detail_rows.extend(
                build_detail_rows(
                    wave_start=wave_start,
                    wave_end=wave_end,
                    evaluation=(
                        priority_evaluation
                    ),
                )
            )

            detail_rows.extend(
                build_detail_rows(
                    wave_start=wave_start,
                    wave_end=wave_end,
                    evaluation=cw_evaluation,
                )
            )

            print(
                "  PASS"
            )

            print(
                f"  NN: "
                f"{nn_evaluation.total_distance_km:.2f} km "
                f"| on-time "
                f"{nn_evaluation.on_time_deliveries}/"
                f"{nn_evaluation.deliveries_with_deadline}"
            )

            print(
                f"  Priority-NN: "
                f"{priority_evaluation.total_distance_km:.2f} km "
                f"| on-time "
                f"{priority_evaluation.on_time_deliveries}/"
                f"{priority_evaluation.deliveries_with_deadline}"
            )

            print(
                f"  Clarke-Wright: "
                f"{cw_evaluation.total_distance_km:.2f} km "
                f"| on-time "
                f"{cw_evaluation.on_time_deliveries}/"
                f"{cw_evaluation.deliveries_with_deadline}"
            )

        except Exception as error:
            print(
                f"  SKIPPED: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            skipped_rows.append(
                {
                    "wave_start":
                        wave_start,

                    "wave_end":
                        wave_end,

                    "number_of_deliveries":
                        len(
                            wave_df
                        ),

                    "error_type":
                        type(
                            error
                        ).__name__,

                    "error_message":
                        str(
                            error
                        ),
                }
            )

        finally:
            if shared_builder is not None:
                shared_builder.clear_cache()

    benchmark_df = pd.DataFrame(
        benchmark_rows
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    details_df = pd.DataFrame(
        detail_rows
    )

    skipped_df = pd.DataFrame(
        skipped_rows,
        columns=[
            "wave_start",
            "wave_end",
            "number_of_deliveries",
            "error_type",
            "error_message",
        ],
    )

    if benchmark_df.empty:
        raise ValueError(
            "No waves were successfully evaluated."
        )

    summary_df = pd.DataFrame(
        [
            summarize_algorithm(
                benchmark_df,
                "nearest_neighbor",
            ),
            summarize_algorithm(
                benchmark_df,
                "priority_nearest_neighbor",
            ),
            summarize_algorithm(
                benchmark_df,
                "clarke_wright",
            ),
        ]
    )

    print()
    print(
        "========================================"
    )
    print(
        "Benchmark Completion"
    )
    print(
        "========================================"
    )

    print(
        f"Eligible waves: "
        f"{len(waves)}"
    )

    print(
        f"Completed waves: "
        f"{len(comparison_df)}"
    )

    print(
        f"Skipped waves: "
        f"{len(skipped_df)}"
    )

    print_summary(
        summary_df
    )

    print_pairwise_summary(
        comparison_df
    )

    save_results(
        benchmark_df=benchmark_df,
        summary_df=summary_df,
        comparison_df=comparison_df,
        details_df=details_df,
        skipped_df=skipped_df,
    )


if __name__ == "__main__":
    main()