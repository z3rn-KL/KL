from pathlib import Path
from time import perf_counter

import pandas as pd

from routing import (
    NearestNeighborRouter,
    RoadNetworkService,
)

from simulation.scenario import ScenarioFactory


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    PROJECT_ROOT
    / "maps"
    / "xedu_drive.graphml"
)

ROAD_NODE_PATH = (
    PROJECT_ROOT
    / "results"
    / "xedu_delivery_road_nodes.csv"
)

PRIORITY_PATH = (
    PROJECT_ROOT
    / "results"
    / "xedu_priority_wave_results.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

ROUTE_RESULT_PATH = (
    RESULTS_DIR
    / "nearest_neighbor_route.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "nearest_neighbor_summary.csv"
)


MIN_WORKLOAD_SIZE = 5


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
        f"Nodes: "
        f"{service.graph.number_of_nodes()}"
    )

    print(
        f"Edges: "
        f"{service.graph.number_of_edges()}"
    )

    return service


def load_snapped_nodes() -> pd.DataFrame:
    if not ROAD_NODE_PATH.exists():
        raise FileNotFoundError(
            f"Missing snapped-node file: "
            f"{ROAD_NODE_PATH}"
        )

    dataframe = pd.read_csv(
        ROAD_NODE_PATH
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
            "latitude",
            "longitude",
            "road_node",
        ]
    ]


def load_priority_results() -> pd.DataFrame:
    if not PRIORITY_PATH.exists():
        raise FileNotFoundError(
            f"Missing priority-wave file: "
            f"{PRIORITY_PATH}"
        )

    dataframe = pd.read_csv(
        PRIORITY_PATH
    )

    dataframe[
        "delivery_id"
    ] = dataframe[
        "delivery_id"
    ].astype(
        str
    )

    dataframe[
        "wave_start"
    ] = pd.to_datetime(
        dataframe[
            "wave_start"
        ]
    )

    dataframe[
        "wave_end"
    ] = pd.to_datetime(
        dataframe[
            "wave_end"
        ]
    )

    return dataframe


def select_real_workload(
    priority_df: pd.DataFrame,
    snapped_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one real XeDu delivery wave.

    Preference:
        - at least MIN_WORKLOAD_SIZE deliveries
        - choose the largest eligible wave
        - deterministic output
    """

    wave_sizes = (
        priority_df
        .groupby(
            "wave_start"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    eligible = wave_sizes[
        wave_sizes
        >= MIN_WORKLOAD_SIZE
    ]

    if eligible.empty:
        raise ValueError(
            "No delivery wave contains enough "
            "orders for NN experiment."
        )

    selected_wave_start = (
        eligible.index[0]
    )

    workload = (
        priority_df[
            priority_df[
                "wave_start"
            ]
            == selected_wave_start
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    workload = workload.merge(
        snapped_df,
        on="delivery_id",
        how="left",
    )

    missing_nodes = int(
        workload[
            "road_node"
        ].isna().sum()
    )

    if missing_nodes > 0:
        raise ValueError(
            f"{missing_nodes} deliveries "
            "do not have road nodes."
        )

    workload[
        "road_node"
    ] = workload[
        "road_node"
    ].astype(
        int
    )

    return workload


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

    depot_node = service.nearest_node(
        latitude=depot.latitude,
        longitude=depot.longitude,
    )

    return (
        depot,
        depot_node,
    )


def build_route_dataframe(
    workload_df: pd.DataFrame,
    route,
) -> pd.DataFrame:
    metadata = (
        workload_df
        .set_index(
            "delivery_id"
        )
    )

    rows = []

    for sequence, delivery_id in enumerate(
        route.delivery_order,
        start=1,
    ):
        row = metadata.loc[
            delivery_id
        ]

        rows.append(
            {
                "sequence":
                    sequence,

                "delivery_id":
                    delivery_id,

                "service_type":
                    row.get(
                        "service_type"
                    ),

                "priority_score":
                    row.get(
                        "priority_score"
                    ),

                "priority_rank_in_wave":
                    row.get(
                        "priority_rank_in_wave"
                    ),

                "latitude":
                    row.get(
                        "latitude"
                    ),

                "longitude":
                    row.get(
                        "longitude"
                    ),

                "road_node":
                    int(
                        row[
                            "road_node"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def run_experiment():
    print()
    print(
        "========================================"
    )
    print(
        "XeDu Nearest Neighbor Experiment"
    )
    print(
        "========================================"
    )

    service = load_road_network()

    snapped_df = (
        load_snapped_nodes()
    )

    priority_df = (
        load_priority_results()
    )

    workload_df = (
        select_real_workload(
            priority_df,
            snapped_df,
        )
    )

    wave_start = workload_df[
        "wave_start"
    ].iloc[
        0
    ]

    wave_end = workload_df[
        "wave_end"
    ].iloc[
        0
    ]

    print()
    print(
        f"Selected wave start: "
        f"{wave_start}"
    )

    print(
        f"Selected wave end: "
        f"{wave_end}"
    )

    print(
        f"Workload size: "
        f"{len(workload_df)}"
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

    delivery_ids = (
        workload_df[
            "delivery_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    delivery_nodes = (
        workload_df[
            "road_node"
        ]
        .astype(
            int
        )
        .tolist()
    )

    router = NearestNeighborRouter(
        road_network=service,
        metric="distance",
    )

    print()
    print(
        "Building NN route..."
    )

    start_time = perf_counter()

    route = router.build_route(
        depot_node=depot_node,
        delivery_ids=delivery_ids,
        delivery_nodes=delivery_nodes,
        return_to_depot=True,
    )

    runtime = (
        perf_counter()
        - start_time
    )

    route_df = (
        build_route_dataframe(
            workload_df,
            route,
        )
    )

    summary_df = pd.DataFrame(
        [
            {
                "algorithm":
                    "nearest_neighbor",

                "selection_metric":
                    "road_distance",

                "wave_start":
                    wave_start,

                "wave_end":
                    wave_end,

                "number_of_deliveries":
                    route.number_of_deliveries(),

                "total_distance_km":
                    route.total_distance_km,

                "total_travel_time_minutes":
                    route.total_travel_time_minutes,

                "runtime_seconds":
                    runtime,

                "returned_to_depot":
                    route.returned_to_depot,

                "depot_node":
                    depot_node,
            }
        ]
    )

    return (
        route,
        route_df,
        summary_df,
    )


def print_results(
    route,
    route_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    summary = summary_df.iloc[
        0
    ]

    print()
    print(
        "========================================"
    )
    print(
        "Nearest Neighbor Result"
    )
    print(
        "========================================"
    )

    print(
        f"Deliveries: "
        f"{int(summary['number_of_deliveries'])}"
    )

    print(
        f"Total road distance: "
        f"{summary['total_distance_km']:.3f} km"
    )

    print(
        f"Estimated travel time: "
        f"{summary['total_travel_time_minutes']:.2f} minutes"
    )

    print(
        f"Runtime: "
        f"{summary['runtime_seconds']:.4f} seconds"
    )

    print(
        f"Returned to depot: "
        f"{summary['returned_to_depot']}"
    )

    print()
    print(
        "Route order:"
    )

    print(
        "Depot"
    )

    for _, row in route_df.iterrows():
        print(
            "  -> "
            f"{int(row['sequence'])}. "
            f"{row['delivery_id']}"
        )

    if route.returned_to_depot:
        print(
            "  -> Depot"
        )

    print()
    print(
        "Route preview:"
    )

    columns = [
        "sequence",
        "delivery_id",
        "service_type",
        "priority_score",
        "priority_rank_in_wave",
        "road_node",
    ]

    print(
        route_df[
            columns
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )


def save_results(
    route_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    route_df.to_csv(
        ROUTE_RESULT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
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
        ROUTE_RESULT_PATH
    )

    print(
        SUMMARY_PATH
    )


def main():
    (
        route,
        route_df,
        summary_df,
    ) = run_experiment()

    print_results(
        route,
        route_df,
        summary_df,
    )

    save_results(
        route_df,
        summary_df,
    )


if __name__ == "__main__":
    main()