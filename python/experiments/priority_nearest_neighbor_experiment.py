from pathlib import Path
from time import perf_counter

import pandas as pd

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)

from routing import (
    RoadNetworkService,
)

from routing.priority_nearest_neighbor import (
    PriorityNearestNeighborRouter,
)

from simulation.scenario import ScenarioFactory


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

BASELINE_ROUTE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nearest_neighbor_route.csv"
)

BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "nearest_neighbor_summary.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

ROUTE_OUTPUT_PATH = (
    RESULTS_DIR
    / "priority_nearest_neighbor_route.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "priority_nearest_neighbor_summary.csv"
)

COMPARISON_OUTPUT_PATH = (
    RESULTS_DIR
    / "nn_vs_priority_nn_comparison.csv"
)


def load_road_network() -> RoadNetworkService:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing graph: {GRAPH_PATH}"
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


def load_baseline_route() -> pd.DataFrame:
    if not BASELINE_ROUTE_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline route: "
            f"{BASELINE_ROUTE_PATH}"
        )

    dataframe = pd.read_csv(
        BASELINE_ROUTE_PATH
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


def load_baseline_summary() -> pd.DataFrame:
    if not BASELINE_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline summary: "
            f"{BASELINE_SUMMARY_PATH}"
        )

    return pd.read_csv(
        BASELINE_SUMMARY_PATH
    )


def load_snapped_nodes() -> pd.DataFrame:
    if not SNAPPED_PATH.exists():
        raise FileNotFoundError(
            f"Missing snapped nodes: "
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


def load_deliveries():
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


def build_fixed_workload(
    baseline_route_df: pd.DataFrame,
    deliveries_by_id: dict,
    snapped_df: pd.DataFrame,
):
    """
    Reuse exactly the same delivery IDs
    that were used by the baseline NN.
    """

    road_nodes_by_delivery = (
        snapped_df
        .set_index(
            "delivery_id"
        )[
            "road_node"
        ]
        .to_dict()
    )

    workload_deliveries = []
    workload_nodes = []

    missing_deliveries = []
    missing_nodes = []

    for delivery_id in (
        baseline_route_df[
            "delivery_id"
        ].tolist()
    ):
        delivery = (
            deliveries_by_id.get(
                delivery_id
            )
        )

        if delivery is None:
            missing_deliveries.append(
                delivery_id
            )
            continue

        road_node = (
            road_nodes_by_delivery.get(
                delivery_id
            )
        )

        if road_node is None:
            missing_nodes.append(
                delivery_id
            )
            continue

        workload_deliveries.append(
            delivery
        )

        workload_nodes.append(
            int(
                road_node
            )
        )

    if missing_deliveries:
        raise ValueError(
            "Missing deliveries: "
            f"{missing_deliveries}"
        )

    if missing_nodes:
        raise ValueError(
            "Missing road nodes: "
            f"{missing_nodes}"
        )

    return (
        workload_deliveries,
        workload_nodes,
    )


def determine_start_time(
    deliveries,
):
    """
    Use the latest created_at among the
    fixed workload as the planning start.

    This keeps all selected orders available
    at route-start time and avoids using
    a modern/current timestamp on historical
    data.
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
            "No created_at values available "
            "for selected workload."
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


def build_route_dataframe(
    route,
):
    rows = []

    for step in route.steps:
        rows.append(
            {
                "sequence":
                    step.sequence,

                "delivery_id":
                    step.delivery_id,

                "from_node":
                    step.from_node,

                "to_node":
                    step.to_node,

                "departure_time":
                    step.departure_time,

                "arrival_time":
                    step.arrival_time,

                "leg_distance_km":
                    step.leg_distance_km,

                "leg_travel_time_minutes":
                    step.leg_travel_time_minutes,

                "priority_score":
                    step.priority_score,

                "deadline_urgency":
                    step.deadline_urgency,

                "service_priority":
                    step.service_priority,

                "waiting_time_score":
                    step.waiting_time_score,

                "travel_efficiency_score":
                    step.travel_efficiency_score,

                "information_coverage":
                    step.information_coverage,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_summary(
    route,
    runtime_seconds: float,
    start_time,
    depot_node: int,
):
    return pd.DataFrame(
        [
            {
                "algorithm":
                    "priority_nearest_neighbor",

                "priority_weights":
                    "D55_S20_W15_T10",

                "number_of_deliveries":
                    route.number_of_deliveries(),

                "start_time":
                    start_time,

                "end_time":
                    route.end_time,

                "total_distance_km":
                    route.total_distance_km,

                "total_travel_time_minutes":
                    route.total_travel_time_minutes,

                "runtime_seconds":
                    runtime_seconds,

                "returned_to_depot":
                    route.returned_to_depot,

                "depot_node":
                    depot_node,
            }
        ]
    )


def build_comparison(
    baseline_summary_df: pd.DataFrame,
    priority_summary_df: pd.DataFrame,
):
    baseline = (
        baseline_summary_df.iloc[
            0
        ]
    )

    priority = (
        priority_summary_df.iloc[
            0
        ]
    )

    baseline_distance = float(
        baseline[
            "total_distance_km"
        ]
    )

    priority_distance = float(
        priority[
            "total_distance_km"
        ]
    )

    baseline_time = float(
        baseline[
            "total_travel_time_minutes"
        ]
    )

    priority_time = float(
        priority[
            "total_travel_time_minutes"
        ]
    )

    baseline_runtime = float(
        baseline[
            "runtime_seconds"
        ]
    )

    priority_runtime = float(
        priority[
            "runtime_seconds"
        ]
    )

    distance_change = (
        priority_distance
        - baseline_distance
    )

    time_change = (
        priority_time
        - baseline_time
    )

    runtime_change = (
        priority_runtime
        - baseline_runtime
    )

    return pd.DataFrame(
        [
            {
                "metric":
                    "total_distance_km",

                "nearest_neighbor":
                    baseline_distance,

                "priority_nearest_neighbor":
                    priority_distance,

                "absolute_change":
                    distance_change,

                "percent_change":
                    (
                        distance_change
                        / baseline_distance
                        * 100.0
                    ),
            },
            {
                "metric":
                    "total_travel_time_minutes",

                "nearest_neighbor":
                    baseline_time,

                "priority_nearest_neighbor":
                    priority_time,

                "absolute_change":
                    time_change,

                "percent_change":
                    (
                        time_change
                        / baseline_time
                        * 100.0
                    ),
            },
            {
                "metric":
                    "runtime_seconds",

                "nearest_neighbor":
                    baseline_runtime,

                "priority_nearest_neighbor":
                    priority_runtime,

                "absolute_change":
                    runtime_change,

                "percent_change":
                    (
                        runtime_change
                        / baseline_runtime
                        * 100.0
                    ),
            },
        ]
    )


def print_route(
    route,
):
    print()
    print(
        "Route order:"
    )

    print(
        "Depot"
    )

    for step in route.steps:
        print(
            "  -> "
            f"{step.sequence}. "
            f"{step.delivery_id} "
            f"| priority="
            f"{step.priority_score:.4f} "
            f"| travel="
            f"{step.travel_efficiency_score:.4f}"
        )

    if route.returned_to_depot:
        print(
            "  -> Depot"
        )


def print_summary(
    summary_df: pd.DataFrame,
):
    row = summary_df.iloc[
        0
    ]

    print()
    print(
        "========================================"
    )
    print(
        "Priority-aware NN Result"
    )
    print(
        "========================================"
    )

    print(
        f"Deliveries: "
        f"{int(row['number_of_deliveries'])}"
    )

    print(
        f"Total road distance: "
        f"{row['total_distance_km']:.3f} km"
    )

    print(
        f"Estimated travel time: "
        f"{row['total_travel_time_minutes']:.2f} minutes"
    )

    print(
        f"Runtime: "
        f"{row['runtime_seconds']:.4f} seconds"
    )

    print(
        f"Returned to depot: "
        f"{row['returned_to_depot']}"
    )


def print_comparison(
    comparison_df: pd.DataFrame,
):
    print()
    print(
        "========================================"
    )
    print(
        "NN vs Priority-aware NN"
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


def save_results(
    route_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    route_df.to_csv(
        ROUTE_OUTPUT_PATH,
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

    print()
    print(
        "Saved:"
    )

    print(
        ROUTE_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        COMPARISON_OUTPUT_PATH
    )


def main():
    print()
    print(
        "========================================"
    )
    print(
        "XeDu Priority-aware NN Experiment"
    )
    print(
        "========================================"
    )

    service = (
        load_road_network()
    )

    baseline_route_df = (
        load_baseline_route()
    )

    baseline_summary_df = (
        load_baseline_summary()
    )

    snapped_df = (
        load_snapped_nodes()
    )

    deliveries_by_id = (
        load_deliveries()
    )

    (
        deliveries,
        delivery_nodes,
    ) = build_fixed_workload(
        baseline_route_df=(
            baseline_route_df
        ),
        deliveries_by_id=(
            deliveries_by_id
        ),
        snapped_df=(
            snapped_df
        ),
    )

    print()
    print(
        f"Fixed workload size: "
        f"{len(deliveries)}"
    )

    start_time = (
        determine_start_time(
            deliveries
        )
    )

    print(
        f"Planning start time: "
        f"{start_time}"
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

    router = (
        PriorityNearestNeighborRouter(
            road_network=service
        )
    )

    print()
    print(
        "Building dynamic "
        "Priority-aware NN route..."
    )

    timer_start = (
        perf_counter()
    )

    route = router.build_route(
        depot_node=depot_node,
        deliveries=deliveries,
        delivery_nodes=delivery_nodes,
        start_time=start_time,
        return_to_depot=True,
    )

    runtime_seconds = (
        perf_counter()
        - timer_start
    )

    route_df = (
        build_route_dataframe(
            route
        )
    )

    summary_df = (
        build_summary(
            route=route,
            runtime_seconds=(
                runtime_seconds
            ),
            start_time=(
                start_time
            ),
            depot_node=(
                depot_node
            ),
        )
    )

    comparison_df = (
        build_comparison(
            baseline_summary_df=(
                baseline_summary_df
            ),
            priority_summary_df=(
                summary_df
            ),
        )
    )

    print_route(
        route
    )

    print_summary(
        summary_df
    )

    print_comparison(
        comparison_df
    )

    save_results(
        route_df=route_df,
        summary_df=summary_df,
        comparison_df=comparison_df,
    )


if __name__ == "__main__":
    main()