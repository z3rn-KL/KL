from pathlib import Path

import pandas as pd

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)

from routing import RoadNetworkService


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


def load_deliveries():
    dataframe = DataLoader.load_csv(
        DATASET_PATH
    )

    dataframe = (
        DataPreprocessor.prepare_deliveries(
            dataframe
        )
    )

    deliveries = (
        DeliveryMapper.from_dataframe(
            dataframe
        )
    )

    return deliveries


def load_road_network():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Road graph not found: {GRAPH_PATH}"
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


def snap_deliveries_to_nodes(
    service: RoadNetworkService,
    deliveries,
) -> pd.DataFrame:
    rows = []

    print()
    print(
        "Snapping delivery coordinates "
        "to nearest road nodes..."
    )

    for index, delivery in enumerate(
        deliveries,
        start=1,
    ):
        node_id = service.nearest_node(
            latitude=delivery.latitude,
            longitude=delivery.longitude,
        )

        rows.append(
            {
                "delivery_id":
                    delivery.delivery_id,

                "latitude":
                    delivery.latitude,

                "longitude":
                    delivery.longitude,

                "road_node":
                    node_id,
            }
        )

        if (
            index == 1
            or index % 500 == 0
            or index == len(deliveries)
        ):
            print(
                f"Processed "
                f"{index}/{len(deliveries)}"
            )

    return pd.DataFrame(
        rows
    )


def test_sample_route(
    service: RoadNetworkService,
    deliveries,
) -> None:
    if len(deliveries) < 2:
        raise ValueError(
            "Need at least 2 deliveries."
        )

    origin = deliveries[0]
    destination = deliveries[-1]

    origin_node = service.nearest_node(
        latitude=origin.latitude,
        longitude=origin.longitude,
    )

    destination_node = service.nearest_node(
        latitude=destination.latitude,
        longitude=destination.longitude,
    )

    print()
    print(
        "========================================"
    )
    print(
        "Sample Road Route"
    )
    print(
        "========================================"
    )

    print(
        f"Origin delivery:"
    )

    print(
        origin.delivery_id
    )

    print(
        f"Origin node:"
    )

    print(
        origin_node
    )

    print()
    print(
        f"Destination delivery:"
    )

    print(
        destination.delivery_id
    )

    print(
        f"Destination node:"
    )

    print(
        destination_node
    )

    distance_km = (
        service.shortest_distance_km(
            origin_node,
            destination_node,
        )
    )

    print()
    print(
        f"Road distance: "
        f"{distance_km:.3f} km"
    )

    try:
        travel_minutes = (
            service
            .shortest_travel_time_minutes(
                origin_node,
                destination_node,
            )
        )

        print(
            f"Estimated travel time: "
            f"{travel_minutes:.2f} minutes"
        )

    except ValueError:
        print(
            "Travel-time data is not "
            "available in graph."
        )

    path = (
        service.shortest_path_nodes(
            origin_node,
            destination_node,
            weight="length",
        )
    )

    print(
        f"Shortest-path nodes: "
        f"{len(path)}"
    )


def save_snap_results(
    dataframe: pd.DataFrame,
) -> None:
    output_path = (
        PROJECT_ROOT
        / "results"
        / "xedu_delivery_road_nodes.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "Saved snapped deliveries:"
    )

    print(
        output_path
    )


def main():
    deliveries = load_deliveries()

    print(
        f"Deliveries: "
        f"{len(deliveries)}"
    )

    service = load_road_network()

    snapped_df = (
        snap_deliveries_to_nodes(
            service,
            deliveries,
        )
    )

    save_snap_results(
        snapped_df
    )

    test_sample_route(
        service,
        deliveries,
    )


if __name__ == "__main__":
    main()