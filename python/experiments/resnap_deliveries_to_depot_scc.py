from pathlib import Path

import networkx as nx
import osmnx as ox
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

OLD_SNAPPED_PATH = (
    PROJECT_ROOT
    / "results"
    / "xedu_delivery_road_nodes.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "xedu_delivery_road_nodes_scc.csv"
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
        "Could not find strongly connected "
        "component containing depot."
    )


def load_deliveries() -> pd.DataFrame:
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
        if (
            delivery.latitude is None
            or delivery.longitude is None
        ):
            raise ValueError(
                f"Delivery "
                f"{delivery.delivery_id} "
                f"has missing coordinates."
            )

        rows.append(
            {
                "delivery_id":
                    str(
                        delivery.delivery_id
                    ),

                "latitude":
                    float(
                        delivery.latitude
                    ),

                "longitude":
                    float(
                        delivery.longitude
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise ValueError(
            "No deliveries were loaded."
        )

    return result


def build_depot_scc_graph(
    graph,
    depot_scc: set,
):
    print()
    print(
        "Building depot-SCC subgraph..."
    )

    scc_graph = (
        graph.subgraph(
            depot_scc
        )
        .copy()
    )

    print(
        f"SCC nodes: "
        f"{scc_graph.number_of_nodes()}"
    )

    print(
        f"SCC edges: "
        f"{scc_graph.number_of_edges()}"
    )

    if (
        "crs"
        not in scc_graph.graph
    ):
        raise ValueError(
            "Depot SCC graph does not "
            "contain CRS metadata."
        )

    return scc_graph


def snap_deliveries(
    deliveries_df: pd.DataFrame,
    scc_graph,
) -> pd.DataFrame:
    """
    Snap every XeDu delivery to the nearest
    road node that belongs to the depot's
    strongly connected component.

    This guarantees that every snapped node
    is mutually reachable with the depot
    inside the directed road graph.
    """

    print()
    print(
        f"Snapping "
        f"{len(deliveries_df)} deliveries "
        f"to depot SCC..."
    )

    nearest_nodes = (
        ox.distance.nearest_nodes(
            scc_graph,
            X=(
                deliveries_df[
                    "longitude"
                ].to_numpy()
            ),
            Y=(
                deliveries_df[
                    "latitude"
                ].to_numpy()
            ),
        )
    )

    result = (
        deliveries_df.copy()
    )

    result[
        "road_node"
    ] = [
        int(
            node
        )
        for node in nearest_nodes
    ]

    return result


def validate_result(
    dataframe: pd.DataFrame,
    depot_scc: set,
) -> None:
    outside = dataframe[
        ~dataframe[
            "road_node"
        ].isin(
            depot_scc
        )
    ]

    duplicate_ids = int(
        dataframe[
            "delivery_id"
        ].duplicated().sum()
    )

    missing_nodes = int(
        dataframe[
            "road_node"
        ].isna().sum()
    )

    print()
    print(
        "========================================"
    )

    print(
        "Validation"
    )

    print(
        "========================================"
    )

    print(
        f"Deliveries: "
        f"{len(dataframe)}"
    )

    print(
        f"Missing road nodes: "
        f"{missing_nodes}"
    )

    print(
        f"Duplicate delivery IDs: "
        f"{duplicate_ids}"
    )

    print(
        f"Outside depot SCC: "
        f"{len(outside)}"
    )

    if missing_nodes > 0:
        raise ValueError(
            "SCC snapping produced "
            "missing road nodes."
        )

    if duplicate_ids > 0:
        raise ValueError(
            "Duplicate delivery IDs found."
        )

    if not outside.empty:
        raise ValueError(
            "Some deliveries were still "
            "snapped outside depot SCC."
        )


def compare_with_old_snapping(
    new_df: pd.DataFrame,
) -> None:
    if not OLD_SNAPPED_PATH.exists():
        print()
        print(
            "Old snapped-node file not found; "
            "comparison skipped."
        )

        return

    old_df = pd.read_csv(
        OLD_SNAPPED_PATH
    )

    old_df[
        "delivery_id"
    ] = old_df[
        "delivery_id"
    ].astype(
        str
    )

    old_df[
        "road_node"
    ] = old_df[
        "road_node"
    ].astype(
        int
    )

    comparison = new_df[
        [
            "delivery_id",
            "road_node",
        ]
    ].merge(
        old_df[
            [
                "delivery_id",
                "road_node",
            ]
        ],
        on="delivery_id",
        how="left",
        suffixes=(
            "_scc",
            "_old",
        ),
    )

    comparison[
        "changed"
    ] = (
        comparison[
            "road_node_scc"
        ]
        != comparison[
            "road_node_old"
        ]
    )

    changed = comparison[
        comparison[
            "changed"
        ]
    ].copy()

    print()
    print(
        "========================================"
    )

    print(
        "Old vs SCC Snapping"
    )

    print(
        "========================================"
    )

    print(
        f"Unchanged deliveries: "
        f"{len(comparison) - len(changed)}"
    )

    print(
        f"Changed deliveries: "
        f"{len(changed)}"
    )

    if not changed.empty:
        print()
        print(
            "Changed road nodes:"
        )

        print(
            changed.to_string(
                index=False
            )
        )


def save_result(
    dataframe: pd.DataFrame,
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "========================================"
    )

    print(
        "Saved"
    )

    print(
        "========================================"
    )

    print(
        OUTPUT_PATH
    )


def main():
    print()
    print(
        "========================================"
    )

    print(
        "XeDu Depot-SCC Road Node Snapping"
    )

    print(
        "========================================"
    )

    service = (
        load_road_network()
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
            graph=service.graph,
            depot_node=depot_node,
        )
    )

    print(
        f"Depot SCC nodes: "
        f"{len(depot_scc)}"
    )

    graph_nodes = (
        service.graph.number_of_nodes()
    )

    print(
        f"Depot SCC coverage: "
        f"{len(depot_scc) / graph_nodes:.2%}"
    )

    scc_graph = (
        build_depot_scc_graph(
            graph=service.graph,
            depot_scc=depot_scc,
        )
    )

    deliveries_df = (
        load_deliveries()
    )

    snapped_df = (
        snap_deliveries(
            deliveries_df=(
                deliveries_df
            ),
            scc_graph=scc_graph,
        )
    )

    validate_result(
        dataframe=snapped_df,
        depot_scc=depot_scc,
    )

    compare_with_old_snapping(
        snapped_df
    )

    save_result(
        snapped_df
    )


if __name__ == "__main__":
    main()