"""
Live OSMnx smoke test.

Purpose
-------
Validate the actual cached XeDu GraphML using OSMnx and NetworkX.

This is stronger than offline_graph_audit.py because it executes:

- osmnx.io.load_graphml
- real shortest path
- real shortest road distance
- real estimated travel time
- road-node route coordinates

It does NOT download a new graph from the Internet.
"""

from pathlib import Path

import pandas as pd

from routing import (
    RoadNetworkService,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
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


def find_node_column(
    dataframe: pd.DataFrame,
) -> str:
    candidates = [
        "road_node",
        "node",
        "road_node_id",
        "snapped_node",
    ]

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        "Cannot find snapped road-node column. "
        f"Available columns: "
        f"{dataframe.columns.tolist()}"
    )


def main() -> None:
    print()
    print(
        "=" * 72
    )

    print(
        "LIVE OSMNX ROAD GRAPH SMOKE TEST"
    )

    print(
        "=" * 72
    )

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing graph: "
            f"{GRAPH_PATH}"
        )

    if not SNAPPED_PATH.exists():
        raise FileNotFoundError(
            f"Missing snapped nodes: "
            f"{SNAPPED_PATH}"
        )

    print(
        f"Graph: "
        f"{GRAPH_PATH}"
    )

    print(
        "Loading GraphML with OSMnx..."
    )

    service = (
        RoadNetworkService
        .load_graphml(
            GRAPH_PATH
        )
    )

    print(
        "Graph loaded."
    )

    print(
        f"Nodes: "
        f"{service.graph.number_of_nodes():,}"
    )

    print(
        f"Edges: "
        f"{service.graph.number_of_edges():,}"
    )

    snapped = pd.read_csv(
        SNAPPED_PATH
    )

    node_column = (
        find_node_column(
            snapped
        )
    )

    nodes = (
        pd.to_numeric(
            snapped[
                node_column
            ],
            errors="raise",
        )
        .astype(
            int
        )
        .drop_duplicates()
        .tolist()
    )

    if (
        len(nodes)
        < 2
    ):
        raise ValueError(
            "Need at least two snapped road nodes."
        )

    origin_node = None
    destination_node = None
    path_nodes = None

    print(
        "Searching for a reachable snapped-node pair..."
    )

    max_candidates = min(
        40,
        len(nodes),
    )

    for i in range(
        max_candidates
    ):
        for j in range(
            i + 1,
            max_candidates,
        ):
            candidate_origin = (
                nodes[
                    i
                ]
            )

            candidate_destination = (
                nodes[
                    j
                ]
            )

            try:
                candidate_path = (
                    service
                    .shortest_path_nodes(
                        candidate_origin,
                        candidate_destination,
                        weight="length",
                    )
                )

            except Exception:
                continue

            if (
                len(
                    candidate_path
                )
                >= 2
            ):
                origin_node = (
                    candidate_origin
                )

                destination_node = (
                    candidate_destination
                )

                path_nodes = (
                    candidate_path
                )

                break

        if (
            path_nodes
            is not None
        ):
            break

    if (
        path_nodes
        is None
    ):
        raise RuntimeError(
            "Cannot find reachable snapped-node pair."
        )

    print()
    print(
        f"Origin node      : "
        f"{origin_node}"
    )

    print(
        f"Destination node : "
        f"{destination_node}"
    )

    distance_km = (
        service
        .shortest_distance_km(
            origin_node,
            destination_node,
        )
    )

    travel_time_minutes = (
        service
        .shortest_travel_time_minutes(
            origin_node,
            destination_node,
        )
    )

    coordinates = (
        service
        .path_coordinates(
            path_nodes
        )
    )

    print(
        f"Path nodes       : "
        f"{len(path_nodes)}"
    )

    print(
        f"Coordinates      : "
        f"{len(coordinates)}"
    )

    print(
        f"Distance         : "
        f"{distance_km:.4f} km"
    )

    print(
        f"Travel time      : "
        f"{travel_time_minutes:.4f} min"
    )

    print()
    print(
        "First coordinates:"
    )

    for coordinate in (
        coordinates[
            :5
        ]
    ):
        print(
            "  ",
            coordinate,
        )

    if (
        len(
            coordinates
        )
        != len(
            path_nodes
        )
    ):
        raise RuntimeError(
            "Path coordinate count mismatch."
        )

    for (
        latitude,
        longitude,
    ) in coordinates:
        if not (
            -90.0
            <= latitude
            <= 90.0
        ):
            raise RuntimeError(
                "Invalid latitude detected."
            )

        if not (
            -180.0
            <= longitude
            <= 180.0
        ):
            raise RuntimeError(
                "Invalid longitude detected."
            )

    if (
        distance_km
        <= 0
    ):
        raise RuntimeError(
            "Shortest distance must be positive."
        )

    if (
        travel_time_minutes
        <= 0
    ):
        raise RuntimeError(
            "Travel time must be positive."
        )

    print()
    print(
        "=" * 72
    )

    print(
        "LIVE OSMNX SMOKE TEST: PASS"
    )

    print(
        "=" * 72
    )

    print(
        "Validated:"
    )

    print(
        "- OSMnx GraphML loading"
    )

    print(
        "- cached snapped road nodes"
    )

    print(
        "- shortest road path"
    )

    print(
        "- shortest road distance"
    )

    print(
        "- estimated travel time"
    )

    print(
        "- route coordinates for frontend map"
    )

    print()


if __name__ == "__main__":
    main()