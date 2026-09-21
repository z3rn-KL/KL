from pathlib import Path

import networkx as nx
import pandas as pd

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)

from routing import (
    RoadNetworkService,
)

from simulation.scenario import (
    ScenarioFactory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

SKIPPED_PATH = (
    PROJECT_ROOT
    / "results"
    / "multi_wave_route_skipped.csv"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)


WAVE_MINUTES = 120


def load_graph():
    print()
    print(
        "Loading road graph..."
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


def load_skipped_waves() -> pd.DataFrame:
    if not SKIPPED_PATH.exists():
        raise FileNotFoundError(
            f"Missing skipped-wave file: "
            f"{SKIPPED_PATH}"
        )

    dataframe = pd.read_csv(
        SKIPPED_PATH
    )

    dataframe[
        "wave_start"
    ] = pd.to_datetime(
        dataframe[
            "wave_start"
        ],
        utc=True,
    )

    dataframe[
        "wave_end"
    ] = pd.to_datetime(
        dataframe[
            "wave_end"
        ],
        utc=True,
    )

    return dataframe


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


def load_deliveries() -> pd.DataFrame:
    """
    Load XeDu through the project's normal
    preprocessing pipeline instead of assuming
    raw CSV column names.
    """

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

    rows = []

    for delivery in deliveries:
        if delivery.created_at is None:
            continue

        created_at = pd.Timestamp(
            delivery.created_at
        )

        wave_start = (
            created_at.floor(
                f"{WAVE_MINUTES}min"
            )
        )

        rows.append(
            {
                "delivery_id":
                    str(
                        delivery.delivery_id
                    ),

                "created_at":
                    created_at,

                "wave_start":
                    wave_start,

                "latitude":
                    delivery.latitude,

                "longitude":
                    delivery.longitude,

                "service_type":
                    delivery.service_type,

                "expected_delivery_time":
                    delivery.expected_delivery_time,
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
    ] = pd.to_datetime(
        result[
            "wave_start"
        ],
        utc=True,
    )

    return result


def find_depot_scc(
    graph,
    depot_node: int,
) -> set:
    print()
    print(
        "Finding strongly connected "
        "component containing depot..."
    )

    for component in (
        nx.strongly_connected_components(
            graph
        )
    ):
        if depot_node in component:
            return set(
                component
            )

    raise ValueError(
        "Depot node is not present "
        "in any strongly connected component."
    )


def merge_delivery_nodes(
    deliveries_df: pd.DataFrame,
    snapped_df: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = deliveries_df.merge(
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
        f"Deliveries loaded: "
        f"{len(dataframe)}"
    )

    print(
        f"Missing snapped nodes: "
        f"{missing_nodes}"
    )

    return dataframe


def analyze_wave(
    wave_start,
    wave_df: pd.DataFrame,
    graph,
    depot_scc: set,
):
    outside_scc = []

    missing_nodes = []

    missing_from_graph = []

    inside_scc = []

    for _, row in wave_df.iterrows():
        delivery_id = str(
            row[
                "delivery_id"
            ]
        )

        road_node = row[
            "road_node"
        ]

        if pd.isna(
            road_node
        ):
            missing_nodes.append(
                delivery_id
            )

            continue

        road_node = int(
            road_node
        )

        if road_node not in graph:
            missing_from_graph.append(
                (
                    delivery_id,
                    road_node,
                )
            )

            continue

        if road_node not in depot_scc:
            outside_scc.append(
                (
                    delivery_id,
                    road_node,
                )
            )

        else:
            inside_scc.append(
                (
                    delivery_id,
                    road_node,
                )
            )

    print()
    print(
        f"Wave: "
        f"{wave_start}"
    )

    print(
        f"Orders: "
        f"{len(wave_df)}"
    )

    print(
        f"Inside depot SCC: "
        f"{len(inside_scc)}"
    )

    print(
        f"Outside depot SCC: "
        f"{len(outside_scc)}"
    )

    print(
        f"Missing snapped node: "
        f"{len(missing_nodes)}"
    )

    print(
        f"Road node missing from graph: "
        f"{len(missing_from_graph)}"
    )

    if outside_scc:
        print()
        print(
            "Outside depot SCC:"
        )

        for (
            delivery_id,
            node,
        ) in outside_scc:
            print(
                f"  {delivery_id} "
                f"| node={node}"
            )

    if missing_nodes:
        print()
        print(
            "Missing snapped nodes:"
        )

        for delivery_id in (
            missing_nodes
        ):
            print(
                f"  {delivery_id}"
            )

    if missing_from_graph:
        print()
        print(
            "Nodes absent from graph:"
        )

        for (
            delivery_id,
            node,
        ) in missing_from_graph:
            print(
                f"  {delivery_id} "
                f"| node={node}"
            )

    return {
        "wave_start":
            wave_start,

        "number_of_deliveries":
            len(
                wave_df
            ),

        "inside_depot_scc":
            len(
                inside_scc
            ),

        "outside_depot_scc":
            len(
                outside_scc
            ),

        "missing_snapped_node":
            len(
                missing_nodes
            ),

        "node_missing_from_graph":
            len(
                missing_from_graph
            ),

        "problem_nodes":
            (
                len(
                    outside_scc
                )
                + len(
                    missing_nodes
                )
                + len(
                    missing_from_graph
                )
            ),
    }


def main():
    print()
    print(
        "========================================"
    )

    print(
        "Road Connectivity Diagnostic"
    )

    print(
        "========================================"
    )

    service = load_graph()

    graph = (
        service.graph
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
        f"  Node: "
        f"{depot_node}"
    )

    depot_scc = (
        find_depot_scc(
            graph=graph,
            depot_node=depot_node,
        )
    )

    print(
        f"Depot SCC nodes: "
        f"{len(depot_scc)}"
    )

    graph_node_count = (
        graph.number_of_nodes()
    )

    scc_ratio = (
        len(
            depot_scc
        )
        / graph_node_count
        if graph_node_count > 0
        else 0.0
    )

    print(
        f"Depot SCC coverage: "
        f"{scc_ratio:.2%}"
    )

    skipped_df = (
        load_skipped_waves()
    )

    snapped_df = (
        load_snapped_nodes()
    )

    deliveries_df = (
        load_deliveries()
    )

    dataframe = (
        merge_delivery_nodes(
            deliveries_df=(
                deliveries_df
            ),
            snapped_df=(
                snapped_df
            ),
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        "Skipped Wave Connectivity"
    )

    print(
        "========================================"
    )

    summary_rows = []

    for _, skipped in (
        skipped_df.iterrows()
    ):
        wave_start = (
            skipped[
                "wave_start"
            ]
        )

        wave_df = (
            dataframe[
                dataframe[
                    "wave_start"
                ]
                == wave_start
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        summary = analyze_wave(
            wave_start=wave_start,
            wave_df=wave_df,
            graph=graph,
            depot_scc=depot_scc,
        )

        summary_rows.append(
            summary
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    print()
    print(
        "========================================"
    )

    print(
        "Diagnostic Summary"
    )

    print(
        "========================================"
    )

    print(
        f"Skipped waves: "
        f"{len(skipped_df)}"
    )

    print(
        f"Deliveries in skipped waves: "
        f"{int(summary_df['number_of_deliveries'].sum())}"
    )

    print(
        f"Inside depot SCC: "
        f"{int(summary_df['inside_depot_scc'].sum())}"
    )

    print(
        f"Outside depot SCC: "
        f"{int(summary_df['outside_depot_scc'].sum())}"
    )

    print(
        f"Missing snapped nodes: "
        f"{int(summary_df['missing_snapped_node'].sum())}"
    )

    print(
        f"Nodes missing from graph: "
        f"{int(summary_df['node_missing_from_graph'].sum())}"
    )

    print(
        f"Total problem nodes: "
        f"{int(summary_df['problem_nodes'].sum())}"
    )

    print()
    print(
        summary_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()