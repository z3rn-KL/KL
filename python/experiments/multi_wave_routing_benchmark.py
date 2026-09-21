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
    / "multi_wave_route_benchmark.csv"
)

COMPARISON_OUTPUT_PATH = (
    RESULTS_DIR
    / "multi_wave_route_comparison.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "multi_wave_route_summary.csv"
)

DETAIL_OUTPUT_PATH = (
    RESULTS_DIR
    / "multi_wave_route_details.csv"
)

SKIPPED_OUTPUT_PATH = (
    RESULTS_DIR
    / "multi_wave_route_skipped.csv"
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

        rows.append(
            {
                "delivery_id":
                    delivery_id,

                "created_at":
                    delivery.created_at,

                "expected_delivery_time":
                    delivery.expected_delivery_time,

                "service_type":
                    delivery.service_type,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = result.dropna(
        subset=[
            "created_at"
        ]
    ).copy()

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
            f"Missing snapped-node file: "
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

    dataframe = dataframe.dropna(
        subset=[
            "road_node"
        ]
    ).copy()

    dataframe[
        "road_node"
    ] = dataframe[
        "road_node"
    ].astype(
        int
    )

    return dataframe


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
        depot_node,
    )


def get_eligible_waves(
    dataframe: pd.DataFrame,
):
    grouped = dataframe.groupby(
        "wave_start",
        sort=True,
    )

    waves = []

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
) -> dict:
    return {
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


def build_comparison_row(
    nn_row: dict,
    priority_row: dict,
) -> dict:
    nn_distance = float(
        nn_row[
            "total_distance_km"
        ]
    )

    priority_distance = float(
        priority_row[
            "total_distance_km"
        ]
    )

    nn_time = float(
        nn_row[
            "total_travel_time_minutes"
        ]
    )

    priority_time = float(
        priority_row[
            "total_travel_time_minutes"
        ]
    )

    nn_lateness = float(
        nn_row[
            "total_lateness_minutes"
        ]
    )

    priority_lateness = float(
        priority_row[
            "total_lateness_minutes"
        ]
    )

    return {
        "wave_start":
            nn_row[
                "wave_start"
            ],

        "wave_end":
            nn_row[
                "wave_end"
            ],

        "number_of_deliveries":
            nn_row[
                "number_of_deliveries"
            ],

        "nn_distance_km":
            nn_distance,

        "priority_nn_distance_km":
            priority_distance,

        "distance_change_km":
            (
                priority_distance
                - nn_distance
            ),

        "distance_change_percent":
            (
                (
                    priority_distance
                    - nn_distance
                )
                / nn_distance
                * 100.0
                if nn_distance > 0
                else np.nan
            ),

        "nn_travel_time_minutes":
            nn_time,

        "priority_nn_travel_time_minutes":
            priority_time,

        "travel_time_change_minutes":
            (
                priority_time
                - nn_time
            ),

        "travel_time_change_percent":
            (
                (
                    priority_time
                    - nn_time
                )
                / nn_time
                * 100.0
                if nn_time > 0
                else np.nan
            ),

        "nn_on_time_deliveries":
            nn_row[
                "on_time_deliveries"
            ],

        "priority_nn_on_time_deliveries":
            priority_row[
                "on_time_deliveries"
            ],

        "on_time_delivery_change":
            (
                priority_row[
                    "on_time_deliveries"
                ]
                - nn_row[
                    "on_time_deliveries"
                ]
            ),

        "nn_late_deliveries":
            nn_row[
                "late_deliveries"
            ],

        "priority_nn_late_deliveries":
            priority_row[
                "late_deliveries"
            ],

        "nn_on_time_rate":
            nn_row[
                "on_time_rate"
            ],

        "priority_nn_on_time_rate":
            priority_row[
                "on_time_rate"
            ],

        "on_time_rate_change":
            (
                priority_row[
                    "on_time_rate"
                ]
                - nn_row[
                    "on_time_rate"
                ]
            ),

        "nn_total_lateness_minutes":
            nn_lateness,

        "priority_nn_total_lateness_minutes":
            priority_lateness,

        "lateness_change_minutes":
            (
                priority_lateness
                - nn_lateness
            ),

        "nn_routing_runtime_seconds":
            nn_row[
                "routing_runtime_seconds"
            ],

        "priority_nn_routing_runtime_seconds":
            priority_row[
                "routing_runtime_seconds"
            ],

        "matrix_preparation_seconds":
            nn_row[
                "matrix_preparation_seconds"
            ],
    }


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

        mean_lateness_per_deadline = (
            total_lateness
            / total_deadline
        )
    else:
        overall_on_time_rate = np.nan
        mean_lateness_per_deadline = np.nan

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

        "mean_lateness_per_deadline_delivery":
            mean_lateness_per_deadline,

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


def print_pairwise_summary(
    comparison_df: pd.DataFrame,
) -> None:
    tolerance = 1e-9

    priority_more_on_time = int(
        (
            comparison_df[
                "on_time_delivery_change"
            ]
            > 0
        ).sum()
    )

    nn_more_on_time = int(
        (
            comparison_df[
                "on_time_delivery_change"
            ]
            < 0
        ).sum()
    )

    equal_on_time = int(
        (
            comparison_df[
                "on_time_delivery_change"
            ]
            == 0
        ).sum()
    )

    priority_lower_lateness = int(
        (
            comparison_df[
                "lateness_change_minutes"
            ]
            < -tolerance
        ).sum()
    )

    nn_lower_lateness = int(
        (
            comparison_df[
                "lateness_change_minutes"
            ]
            > tolerance
        ).sum()
    )

    equal_lateness = int(
        (
            comparison_df[
                "lateness_change_minutes"
            ].abs()
            <= tolerance
        ).sum()
    )

    priority_shorter = int(
        (
            comparison_df[
                "distance_change_km"
            ]
            < -tolerance
        ).sum()
    )

    nn_shorter = int(
        (
            comparison_df[
                "distance_change_km"
            ]
            > tolerance
        ).sum()
    )

    equal_distance = int(
        (
            comparison_df[
                "distance_change_km"
            ].abs()
            <= tolerance
        ).sum()
    )

    print()
    print(
        "========================================"
    )
    print(
        "Pairwise Wave Comparison"
    )
    print(
        "========================================"
    )

    print(
        f"Completed waves: "
        f"{len(comparison_df)}"
    )

    print()
    print(
        "On-time delivery count:"
    )

    print(
        f"  Priority-NN higher: "
        f"{priority_more_on_time}"
    )

    print(
        f"  NN higher: "
        f"{nn_more_on_time}"
    )

    print(
        f"  Equal: "
        f"{equal_on_time}"
    )

    print()
    print(
        "Total lateness:"
    )

    print(
        f"  Priority-NN lower: "
        f"{priority_lower_lateness}"
    )

    print(
        f"  NN lower: "
        f"{nn_lower_lateness}"
    )

    print(
        f"  Equal: "
        f"{equal_lateness}"
    )

    print()
    print(
        "Distance:"
    )

    print(
        f"  Priority-NN shorter: "
        f"{priority_shorter}"
    )

    print(
        f"  NN shorter: "
        f"{nn_shorter}"
    )

    print(
        f"  Equal: "
        f"{equal_distance}"
    )


def print_summary(
    summary_df: pd.DataFrame,
) -> None:
    print()
    print(
        "========================================"
    )
    print(
        "Multi-wave Benchmark Summary"
    )
    print(
        "========================================"
    )

    columns = [
        "algorithm",
        "evaluated_waves",
        "total_deliveries",
        "deliveries_with_deadline",
        "on_time_deliveries",
        "late_deliveries",
        "overall_on_time_rate",
        "waves_with_late_deliveries",
        "total_lateness_minutes",
        "mean_lateness_per_deadline_delivery",
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
    benchmark_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    details_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_df.to_csv(
        BENCHMARK_OUTPUT_PATH,
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
        COMPARISON_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
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
        "XeDu Multi-wave Routing Benchmark"
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

            all_nodes = (
                int(
                    depot_node
                ),
                *[
                    int(node)
                    for node
                    in delivery_nodes
                ],
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

            evaluator = RouteEvaluator(
                road_network=service
            )

            # Use the same pre-warmed matrix cache
            # for both algorithms and evaluator.
            #
            # Therefore routing runtime below
            # mainly measures routing decisions,
            # not repeated Dijkstra computation.

            nn_router.matrix_builder = (
                shared_builder
            )

            priority_router.matrix_builder = (
                shared_builder
            )

            evaluator.matrix_builder = (
                shared_builder
            )

            delivery_ids = [
                str(
                    delivery.delivery_id
                )
                for delivery
                in deliveries
            ]

            nn_start = (
                perf_counter()
            )

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
                        nn_route
                        .delivery_order
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
                    wave_start=(
                        wave_start
                    ),
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

            benchmark_rows.extend(
                [
                    nn_row,
                    priority_row,
                ]
            )

            comparison_rows.append(
                build_comparison_row(
                    nn_row=nn_row,
                    priority_row=(
                        priority_row
                    ),
                )
            )

            detail_rows.extend(
                build_detail_rows(
                    wave_start=(
                        wave_start
                    ),
                    wave_end=wave_end,
                    evaluation=(
                        nn_evaluation
                    ),
                )
            )

            detail_rows.extend(
                build_detail_rows(
                    wave_start=(
                        wave_start
                    ),
                    wave_end=wave_end,
                    evaluation=(
                        priority_evaluation
                    ),
                )
            )

            print(
                "  PASS"
            )

            print(
                f"  Matrix prep: "
                f"{matrix_runtime:.3f}s"
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
            "No waves were successfully "
            "evaluated."
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
        benchmark_df=(
            benchmark_df
        ),
        comparison_df=(
            comparison_df
        ),
        summary_df=(
            summary_df
        ),
        details_df=(
            details_df
        ),
        skipped_df=(
            skipped_df
        ),
    )


if __name__ == "__main__":
    main()