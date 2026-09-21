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

REFERENCE_ROUTE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nearest_neighbor_route.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

CW_ROUTE_OUTPUT_PATH = (
    RESULTS_DIR
    / "clarke_wright_route.csv"
)

COMPARISON_OUTPUT_PATH = (
    RESULTS_DIR
    / "three_algorithm_single_workload_comparison.csv"
)

DETAIL_OUTPUT_PATH = (
    RESULTS_DIR
    / "three_algorithm_single_workload_details.csv"
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


def load_reference_delivery_ids() -> list[str]:
    if not REFERENCE_ROUTE_PATH.exists():
        raise FileNotFoundError(
            f"Missing reference route: "
            f"{REFERENCE_ROUTE_PATH}"
        )

    dataframe = pd.read_csv(
        REFERENCE_ROUTE_PATH
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

    delivery_ids = (
        dataframe[
            "delivery_id"
        ]
        .tolist()
    )

    if len(
        delivery_ids
    ) != len(
        set(
            delivery_ids
        )
    ):
        raise ValueError(
            "Reference workload contains "
            "duplicate delivery IDs."
        )

    return delivery_ids


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


def load_scc_nodes() -> dict:
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

    return (
        dataframe
        .set_index(
            "delivery_id"
        )[
            "road_node"
        ]
        .to_dict()
    )


def build_fixed_workload(
    reference_ids: list[str],
    deliveries_by_id: dict,
    node_lookup: dict,
):
    deliveries = []
    delivery_nodes = []

    for delivery_id in reference_ids:
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
            node_lookup.get(
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


def determine_start_time(
    deliveries,
):
    created_times = [
        delivery.created_at
        for delivery
        in deliveries
        if delivery.created_at
        is not None
    ]

    if not created_times:
        raise ValueError(
            "No created_at values available."
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
        int(
            depot_node
        ),
    )


def build_summary_row(
    evaluation,
    routing_runtime_seconds: float,
    matrix_runtime_seconds: float,
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

        "matrix_preparation_seconds":
            matrix_runtime_seconds,

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


def build_cw_route_dataframe(
    route,
) -> pd.DataFrame:
    rows = []

    for sequence, (
        delivery_id,
        road_node,
    ) in enumerate(
        zip(
            route.delivery_order,
            route.node_order[
                1:-1
            ],
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

                "road_node":
                    road_node,
            }
        )

    return pd.DataFrame(
        rows
    )


def print_algorithm_result(
    row: dict,
) -> None:
    print()
    print(
        "----------------------------------------"
    )

    print(
        f"Algorithm: "
        f"{row['algorithm']}"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Deliveries: "
        f"{row['number_of_deliveries']}"
    )

    print(
        f"Distance: "
        f"{row['total_distance_km']:.3f} km"
    )

    print(
        f"Travel time: "
        f"{row['total_travel_time_minutes']:.2f} min"
    )

    print(
        f"On-time: "
        f"{row['on_time_deliveries']}/"
        f"{row['deliveries_with_deadline']}"
    )

    print(
        f"Late: "
        f"{row['late_deliveries']}"
    )

    if row[
        "on_time_rate"
    ] is None:
        print(
            "On-time rate: N/A"
        )
    else:
        print(
            f"On-time rate: "
            f"{row['on_time_rate']:.2%}"
        )

    print(
        f"Total lateness: "
        f"{row['total_lateness_minutes']:.2f} min"
    )

    print(
        f"Max lateness: "
        f"{row['max_lateness_minutes']:.2f} min"
    )

    print(
        f"Routing runtime: "
        f"{row['routing_runtime_seconds']:.6f} sec"
    )


def print_comparison(
    comparison_df: pd.DataFrame,
) -> None:
    print()
    print(
        "========================================"
    )

    print(
        "Single-workload Algorithm Comparison"
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


def main():
    print()
    print(
        "========================================"
    )

    print(
        "XeDu Clarke-Wright Experiment"
    )

    print(
        "========================================"
    )

    service = (
        load_road_network()
    )

    reference_ids = (
        load_reference_delivery_ids()
    )

    deliveries_by_id = (
        load_deliveries_by_id()
    )

    node_lookup = (
        load_scc_nodes()
    )

    (
        deliveries,
        delivery_nodes,
    ) = build_fixed_workload(
        reference_ids=(
            reference_ids
        ),
        deliveries_by_id=(
            deliveries_by_id
        ),
        node_lookup=(
            node_lookup
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
        f"Common planning start: "
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

    all_nodes = (
        depot_node,
        *delivery_nodes,
    )

    shared_builder = (
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

    shared_builder.build(
        all_nodes
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

    delivery_ids = [
        str(
            delivery.delivery_id
        )
        for delivery
        in deliveries
    ]

    print()
    print(
        "Running Nearest Neighbor..."
    )

    timer = perf_counter()

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
        - timer
    )

    print(
        "Running Priority-aware NN..."
    )

    timer = perf_counter()

    priority_route = (
        priority_router.build_route(
            depot_node=depot_node,
            deliveries=deliveries,
            delivery_nodes=(
                delivery_nodes
            ),
            start_time=start_time,
            return_to_depot=True,
        )
    )

    priority_runtime = (
        perf_counter()
        - timer
    )

    print(
        "Running Clarke-Wright..."
    )

    timer = perf_counter()

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
        - timer
    )

    print()
    print(
        f"Clarke-Wright initial "
        f"separate distance: "
        f"{cw_route.initial_separate_distance_km:.3f} km"
    )

    print(
        f"Clarke-Wright final distance: "
        f"{cw_route.total_distance_km:.3f} km"
    )

    print(
        f"Clarke-Wright savings: "
        f"{cw_route.total_savings_km:.3f} km"
    )

    print(
        f"Clarke-Wright merges: "
        f"{cw_route.merge_count}"
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
            start_time=start_time,
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
            start_time=start_time,
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
            start_time=start_time,
            return_to_depot=True,
        )
    )

    rows = [
        build_summary_row(
            evaluation=(
                nn_evaluation
            ),
            routing_runtime_seconds=(
                nn_runtime
            ),
            matrix_runtime_seconds=(
                matrix_runtime
            ),
        ),
        build_summary_row(
            evaluation=(
                priority_evaluation
            ),
            routing_runtime_seconds=(
                priority_runtime
            ),
            matrix_runtime_seconds=(
                matrix_runtime
            ),
        ),
        build_summary_row(
            evaluation=(
                cw_evaluation
            ),
            routing_runtime_seconds=(
                cw_runtime
            ),
            matrix_runtime_seconds=(
                matrix_runtime
            ),
        ),
    ]

    comparison_df = pd.DataFrame(
        rows
    )

    for row in rows:
        print_algorithm_result(
            row
        )

    print_comparison(
        comparison_df
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

    detail_rows.extend(
        build_detail_rows(
            cw_evaluation
        )
    )

    details_df = pd.DataFrame(
        detail_rows
    )

    cw_route_df = (
        build_cw_route_dataframe(
            cw_route
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cw_route_df.to_csv(
        CW_ROUTE_OUTPUT_PATH,
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
        CW_ROUTE_OUTPUT_PATH
    )

    print(
        COMPARISON_OUTPUT_PATH
    )

    print(
        DETAIL_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()