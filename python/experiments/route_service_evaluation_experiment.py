from pathlib import Path

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
    / "xedu_delivery_road_nodes.csv"
)

NN_ROUTE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nearest_neighbor_route.csv"
)

PRIORITY_NN_ROUTE_PATH = (
    PROJECT_ROOT
    / "results"
    / "priority_nearest_neighbor_route.csv"
)

NN_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "nearest_neighbor_summary.csv"
)

PRIORITY_NN_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "priority_nearest_neighbor_summary.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

COMPARISON_OUTPUT_PATH = (
    RESULTS_DIR
    / "route_service_comparison.csv"
)

DETAIL_OUTPUT_PATH = (
    RESULTS_DIR
    / "route_service_delivery_details.csv"
)


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


def load_route(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing route file: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    dataframe[
        "delivery_id"
    ] = dataframe[
        "delivery_id"
    ].astype(
        str
    )

    dataframe = (
        dataframe
        .sort_values(
            "sequence"
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


def load_routing_runtime(
    path: Path,
) -> float:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing summary file: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    if dataframe.empty:
        raise ValueError(
            f"Summary file is empty: {path}"
        )

    return float(
        dataframe.iloc[
            0
        ][
            "runtime_seconds"
        ]
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


def load_deliveries_by_id() -> dict:
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

    return {
        str(
            delivery.delivery_id
        ): delivery
        for delivery
        in deliveries
    }


def validate_same_workload(
    nn_route_df: pd.DataFrame,
    priority_route_df: pd.DataFrame,
) -> None:
    nn_ids = (
        nn_route_df[
            "delivery_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    priority_ids = (
        priority_route_df[
            "delivery_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if len(
        nn_ids
    ) != len(
        priority_ids
    ):
        raise ValueError(
            "NN and Priority-NN do not "
            "contain the same number of "
            "deliveries."
        )

    if set(
        nn_ids
    ) != set(
        priority_ids
    ):
        raise ValueError(
            "NN and Priority-NN do not "
            "use the same delivery workload."
        )

    if len(
        set(
            nn_ids
        )
    ) != len(
        nn_ids
    ):
        raise ValueError(
            "Duplicate delivery IDs found "
            "in benchmark workload."
        )


def build_fixed_workload(
    nn_route_df: pd.DataFrame,
    deliveries_by_id: dict,
    snapped_df: pd.DataFrame,
):
    road_node_lookup = (
        snapped_df
        .set_index(
            "delivery_id"
        )[
            "road_node"
        ]
        .to_dict()
    )

    deliveries = []
    delivery_nodes = []

    for delivery_id in (
        nn_route_df[
            "delivery_id"
        ].tolist()
    ):
        delivery = (
            deliveries_by_id.get(
                delivery_id
            )
        )

        if delivery is None:
            raise ValueError(
                f"Delivery not found: "
                f"{delivery_id}"
            )

        road_node = (
            road_node_lookup.get(
                delivery_id
            )
        )

        if road_node is None:
            raise ValueError(
                f"Road node not found for "
                f"{delivery_id}"
            )

        deliveries.append(
            delivery
        )

        delivery_nodes.append(
            int(
                road_node
            )
        )

    return (
        deliveries,
        delivery_nodes,
    )


def determine_common_start_time(
    deliveries,
):
    """
    Use the latest created_at among the
    benchmark deliveries.

    Therefore every selected delivery already
    exists at the beginning of the route.

    Both algorithms receive exactly the same
    planning start time.
    """

    created_times = [
        delivery.created_at
        for delivery
        in deliveries
        if delivery.created_at
        is not None
    ]

    if not created_times:
        raise ValueError(
            "No created_at values are "
            "available for workload."
        )

    return max(
        created_times
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
        depot_node,
    )


def build_summary_row(
    evaluation,
    routing_runtime_seconds: float,
) -> dict:
    return {
        "algorithm":
            evaluation.algorithm,

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
            evaluation.on_time_rate,

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

        "start_time":
            evaluation.start_time,

        "end_time":
            evaluation.end_time,

        "returned_to_depot":
            evaluation.returned_to_depot,
    }


def build_detail_rows(
    evaluation,
) -> list[dict]:
    rows = []

    for result in (
        evaluation.delivery_results
    ):
        rows.append(
            {
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


def print_evaluation(
    evaluation,
    runtime_seconds: float,
) -> None:
    print()
    print(
        "----------------------------------------"
    )

    print(
        f"Algorithm: "
        f"{evaluation.algorithm}"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Deliveries: "
        f"{evaluation.number_of_deliveries}"
    )

    print(
        f"With deadline: "
        f"{evaluation.deliveries_with_deadline}"
    )

    print(
        f"Missing deadline: "
        f"{evaluation.deliveries_without_deadline}"
    )

    print(
        f"On-time: "
        f"{evaluation.on_time_deliveries}"
    )

    print(
        f"Late: "
        f"{evaluation.late_deliveries}"
    )

    if evaluation.on_time_rate is None:
        print(
            "On-time rate: N/A"
        )
    else:
        print(
            f"On-time rate: "
            f"{evaluation.on_time_rate:.2%}"
        )

    print(
        f"Total lateness: "
        f"{evaluation.total_lateness_minutes:.2f} min"
    )

    print(
        f"Mean lateness: "
        f"{evaluation.mean_lateness_minutes:.2f} min"
    )

    print(
        f"Max lateness: "
        f"{evaluation.max_lateness_minutes:.2f} min"
    )

    print(
        f"Distance: "
        f"{evaluation.total_distance_km:.3f} km"
    )

    print(
        f"Travel time: "
        f"{evaluation.total_travel_time_minutes:.2f} min"
    )

    print(
        f"Routing runtime: "
        f"{runtime_seconds:.4f} sec"
    )


def print_comparison(
    comparison_df: pd.DataFrame,
) -> None:
    print()
    print(
        "========================================"
    )
    print(
        "Route Service Comparison"
    )
    print(
        "========================================"
    )

    columns = [
        "algorithm",
        "number_of_deliveries",
        "deliveries_with_deadline",
        "deliveries_without_deadline",
        "on_time_deliveries",
        "late_deliveries",
        "on_time_rate",
        "total_lateness_minutes",
        "mean_lateness_minutes",
        "max_lateness_minutes",
        "total_distance_km",
        "total_travel_time_minutes",
        "routing_runtime_seconds",
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


def save_results(
    comparison_df: pd.DataFrame,
    details_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=False,
    )

    details_df.to_csv(
        DETAIL_OUTPUT_PATH,
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
        COMPARISON_OUTPUT_PATH
    )

    print(
        DETAIL_OUTPUT_PATH
    )


def main():
    print()
    print(
        "========================================"
    )
    print(
        "XeDu Route Service Evaluation"
    )
    print(
        "========================================"
    )

    service = (
        load_road_network()
    )

    nn_route_df = (
        load_route(
            NN_ROUTE_PATH
        )
    )

    priority_route_df = (
        load_route(
            PRIORITY_NN_ROUTE_PATH
        )
    )

    validate_same_workload(
        nn_route_df=nn_route_df,
        priority_route_df=(
            priority_route_df
        ),
    )

    print()
    print(
        "Workload validation: PASS"
    )

    print(
        f"Benchmark deliveries: "
        f"{len(nn_route_df)}"
    )

    snapped_df = (
        load_snapped_nodes()
    )

    deliveries_by_id = (
        load_deliveries_by_id()
    )

    (
        deliveries,
        delivery_nodes,
    ) = build_fixed_workload(
        nn_route_df=nn_route_df,
        deliveries_by_id=(
            deliveries_by_id
        ),
        snapped_df=snapped_df,
    )

    common_start_time = (
        determine_common_start_time(
            deliveries
        )
    )

    print(
        f"Common planning start: "
        f"{common_start_time}"
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

    evaluator = RouteEvaluator(
        road_network=service
    )

    nn_order = (
        nn_route_df[
            "delivery_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    priority_order = (
        priority_route_df[
            "delivery_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    print()
    print(
        "Evaluating Nearest Neighbor..."
    )

    nn_evaluation = (
        evaluator.evaluate(
            algorithm=(
                "nearest_neighbor"
            ),
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=nn_order,
            start_time=(
                common_start_time
            ),
            return_to_depot=True,
        )
    )

    print(
        "Evaluating Priority-aware NN..."
    )

    priority_evaluation = (
        evaluator.evaluate(
            algorithm=(
                "priority_nearest_neighbor"
            ),
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=(
                priority_order
            ),
            start_time=(
                common_start_time
            ),
            return_to_depot=True,
        )
    )

    nn_runtime = (
        load_routing_runtime(
            NN_SUMMARY_PATH
        )
    )

    priority_runtime = (
        load_routing_runtime(
            PRIORITY_NN_SUMMARY_PATH
        )
    )

    print_evaluation(
        evaluation=nn_evaluation,
        runtime_seconds=(
            nn_runtime
        ),
    )

    print_evaluation(
        evaluation=(
            priority_evaluation
        ),
        runtime_seconds=(
            priority_runtime
        ),
    )

    comparison_df = pd.DataFrame(
        [
            build_summary_row(
                evaluation=(
                    nn_evaluation
                ),
                routing_runtime_seconds=(
                    nn_runtime
                ),
            ),
            build_summary_row(
                evaluation=(
                    priority_evaluation
                ),
                routing_runtime_seconds=(
                    priority_runtime
                ),
            ),
        ]
    )

    detail_rows = []

    detail_rows.extend(
        build_detail_rows(
            nn_evaluation
        )
    )

    detail_rows.extend(
        build_detail_rows(
            priority_evaluation
        )
    )

    details_df = pd.DataFrame(
        detail_rows
    )

    print_comparison(
        comparison_df
    )

    save_results(
        comparison_df=(
            comparison_df
        ),
        details_df=(
            details_df
        ),
    )


if __name__ == "__main__":
    main()