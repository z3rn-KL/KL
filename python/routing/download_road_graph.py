from pathlib import Path
from time import sleep

import osmnx as ox

from data import DataLoader, DataPreprocessor
from routing.road_network import RoadNetworkService


PROJECT_ROOT = Path(__file__).resolve().parents[2]

XEDU_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)

MAPS_DIR = (
    PROJECT_ROOT
    / "maps"
)

GRAPH_PATH = (
    MAPS_DIR
    / "xedu_drive.graphml"
)


BUFFER_DEGREES = 0.03

REQUEST_TIMEOUT_SECONDS = 300

RETRIES_PER_SERVER = 2

RETRY_WAIT_SECONDS = 10


OVERPASS_SERVERS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]


def load_delivery_bbox() -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Load XeDu receiver coordinates and build
    a buffered bounding box.

    Returns:
        west,
        south,
        east,
        north
    """

    dataframe = DataLoader.load_csv(
        XEDU_PATH
    )

    dataframe = (
        DataPreprocessor.prepare_deliveries(
            dataframe
        )
    )

    latitudes = dataframe[
        "receiverLat"
    ]

    longitudes = dataframe[
        "receiverLng"
    ]

    south = (
        float(
            latitudes.min()
        )
        - BUFFER_DEGREES
    )

    north = (
        float(
            latitudes.max()
        )
        + BUFFER_DEGREES
    )

    west = (
        float(
            longitudes.min()
        )
        - BUFFER_DEGREES
    )

    east = (
        float(
            longitudes.max()
        )
        + BUFFER_DEGREES
    )

    return (
        west,
        south,
        east,
        north,
    )


def configure_osmnx(
    overpass_url: str,
) -> None:
    """
    Configure OSMnx for reliable downloading.
    """

    ox.settings.use_cache = True

    ox.settings.log_console = True

    ox.settings.requests_timeout = (
        REQUEST_TIMEOUT_SECONDS
    )

    ox.settings.overpass_url = (
        overpass_url
    )


def download_from_server(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
    overpass_url: str,
):
    """
    Try downloading a drive graph
    from one Overpass server.
    """

    configure_osmnx(
        overpass_url
    )

    west, south, east, north = bbox

    print()
    print(
        "========================================"
    )

    print(
        f"Trying Overpass server:"
    )

    print(
        overpass_url
    )

    print(
        "========================================"
    )

    graph = ox.graph.graph_from_bbox(
        bbox=(
            west,
            south,
            east,
            north,
        ),
        network_type="drive",
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )

    if graph.number_of_nodes() == 0:
        raise ValueError(
            "Downloaded graph has no nodes."
        )

    if graph.number_of_edges() == 0:
        raise ValueError(
            "Downloaded graph has no edges."
        )

    return graph


def download_graph(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
):
    """
    Try multiple Overpass servers
    with retries.
    """

    last_error = None

    for server in OVERPASS_SERVERS:

        for attempt in range(
            1,
            RETRIES_PER_SERVER + 1,
        ):
            try:
                print()
                print(
                    f"Server: {server}"
                )

                print(
                    f"Attempt: "
                    f"{attempt}/"
                    f"{RETRIES_PER_SERVER}"
                )

                graph = (
                    download_from_server(
                        bbox=bbox,
                        overpass_url=server,
                    )
                )

                return graph

            except Exception as error:
                last_error = error

                print()
                print(
                    "Download attempt failed:"
                )

                print(
                    repr(
                        error
                    )
                )

                if (
                    attempt
                    < RETRIES_PER_SERVER
                ):
                    print(
                        f"Waiting "
                        f"{RETRY_WAIT_SECONDS} "
                        f"seconds..."
                    )

                    sleep(
                        RETRY_WAIT_SECONDS
                    )

        print()
        print(
            "Moving to next "
            "Overpass server..."
        )

    raise RuntimeError(
        "Unable to download road graph "
        "from all configured "
        "Overpass servers."
    ) from last_error


def print_bbox(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> None:
    west, south, east, north = bbox

    print()
    print(
        "========================================"
    )

    print(
        "XeDu Road-Network Bounding Box"
    )

    print(
        "========================================"
    )

    print(
        f"West:  {west:.6f}"
    )

    print(
        f"South: {south:.6f}"
    )

    print(
        f"East:  {east:.6f}"
    )

    print(
        f"North: {north:.6f}"
    )


def print_graph_summary(
    service: RoadNetworkService,
) -> None:
    graph = service.graph

    print()
    print(
        "========================================"
    )

    print(
        "Road Graph Summary"
    )

    print(
        "========================================"
    )

    print(
        f"Nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Edges: "
        f"{graph.number_of_edges()}"
    )

    print(
        f"Saved graph:"
    )

    print(
        GRAPH_PATH
    )


def build_and_save_graph(
    force: bool = False,
) -> None:
    """
    Download and cache the XeDu road graph.

    If the graph already exists,
    skip downloading unless force=True.
    """

    MAPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        GRAPH_PATH.exists()
        and not force
    ):
        print()
        print(
            "========================================"
        )

        print(
            "Cached road graph already exists."
        )

        print(
            "========================================"
        )

        print(
            GRAPH_PATH
        )

        print()
        print(
            "Skipping download."
        )

        print(
            "Use --force to download again."
        )

        return

    bbox = load_delivery_bbox()

    print_bbox(
        bbox
    )

    graph = download_graph(
        bbox
    )

    service = RoadNetworkService(
        graph
    )

    print()
    print(
        "Adding estimated speeds "
        "and travel times..."
    )

    service.add_estimated_travel_times(
        fallback_kph=30
    )

    print()
    print(
        "Saving GraphML..."
    )

    service.save_graphml(
        GRAPH_PATH
    )

    print_graph_summary(
        service
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download and cache "
            "XeDu road network."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Download again even if "
            "cached graph already exists."
        ),
    )

    args = parser.parse_args()

    build_and_save_graph(
        force=args.force
    )


if __name__ == "__main__":
    main()