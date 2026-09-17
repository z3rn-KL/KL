from __future__ import annotations

import argparse
from pathlib import Path
from time import sleep

import networkx as nx
import osmnx as ox

from data import DataLoader, DataPreprocessor
from routing import RoadNetworkService


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

XEDU_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)

MAPS_DIR = PROJECT_ROOT / "maps"

GRAPH_PATH = (
    MAPS_DIR
    / "xedu_drive.graphml"
)


# ============================================================
# DOWNLOAD SETTINGS
# ============================================================

# ~3 km vùng đệm quanh phạm vi delivery.
# Mục đích:
# - tránh cắt road network quá sát delivery;
# - cho phép shortest path vòng qua cầu / đường lớn / vật cản.
BUFFER_DEGREES = 0.03

REQUEST_TIMEOUT_SECONDS = 300

RETRIES_PER_SERVER = 2

RETRY_WAIT_SECONDS = 10


# Nếu một Overpass server gặp SSL/timeout,
# chương trình sẽ tự thử server tiếp theo.
OVERPASS_SERVERS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]


# ============================================================
# DATASET BOUNDING BOX
# ============================================================

def load_delivery_bbox() -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Đọc dataset XeDu và xác định vùng bao quanh
    tất cả điểm giao hàng.

    OSMnx 2.x yêu cầu bbox:

        (
            west,
            south,
            east,
            north,
        )

    tương đương:

        (
            left,
            bottom,
            right,
            top,
        )
    """

    print("Đang đọc dataset XeDu...")

    dataframe = DataLoader.load_csv(
        XEDU_PATH
    )

    dataframe = (
        DataPreprocessor
        .prepare_deliveries(
            dataframe
        )
    )

    if dataframe.empty:
        raise ValueError(
            "Không có delivery hợp lệ "
            "sau preprocessing."
        )

    west = float(
        dataframe["receiverLng"].min()
    )

    east = float(
        dataframe["receiverLng"].max()
    )

    south = float(
        dataframe["receiverLat"].min()
    )

    north = float(
        dataframe["receiverLat"].max()
    )

    # Thêm vùng đệm.
    west -= BUFFER_DEGREES
    east += BUFFER_DEGREES
    south -= BUFFER_DEGREES
    north += BUFFER_DEGREES

    return (
        west,
        south,
        east,
        north,
    )


# ============================================================
# OSMNX CONFIGURATION
# ============================================================

def configure_osmnx(
    overpass_url: str,
) -> None:
    """
    Cấu hình OSMnx trước khi gọi Overpass API.
    """

    ox.settings.use_cache = True

    ox.settings.log_console = True

    ox.settings.requests_timeout = (
        REQUEST_TIMEOUT_SECONDS
    )

    ox.settings.overpass_url = (
        overpass_url
    )


# ============================================================
# GRAPH DOWNLOAD
# ============================================================

def download_from_server(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
    server_url: str,
) -> nx.MultiDiGraph:
    """
    Thử tải graph từ một Overpass server.
    """

    configure_osmnx(
        server_url
    )

    for attempt in range(
        1,
        RETRIES_PER_SERVER + 1,
    ):
        print()
        print(
            "----------------------------------------"
        )

        print(
            f"Overpass server: {server_url}"
        )

        print(
            "Attempt:",
            f"{attempt}/{RETRIES_PER_SERVER}",
        )

        print(
            "----------------------------------------"
        )

        try:
            graph = (
                ox.graph.graph_from_bbox(
                    bbox=bbox,
                    network_type="drive",
                    simplify=True,
                    retain_all=False,
                    truncate_by_edge=True,
                )
            )

            if graph.number_of_nodes() == 0:
                raise ValueError(
                    "Graph tải về không có node."
                )

            if graph.number_of_edges() == 0:
                raise ValueError(
                    "Graph tải về không có edge."
                )

            return graph

        except Exception as error:
            print()
            print(
                "Download thất bại:"
            )

            print(
                type(error).__name__,
            )

            print(
                error
            )

            if (
                attempt
                < RETRIES_PER_SERVER
            ):
                wait_seconds = (
                    RETRY_WAIT_SECONDS
                    * attempt
                )

                print()
                print(
                    f"Chờ {wait_seconds} giây "
                    "rồi thử lại..."
                )

                sleep(
                    wait_seconds
                )

    raise RuntimeError(
        f"Không thể tải graph từ "
        f"{server_url}"
    )


def download_graph(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> tuple[
    nx.MultiDiGraph,
    str,
]:
    """
    Thử lần lượt nhiều Overpass server.

    Returns
    -------
    graph
        Road graph tải được.

    server_url
        Server đã download thành công.
    """

    last_error: Exception | None = None

    for server_url in OVERPASS_SERVERS:
        try:
            graph = download_from_server(
                bbox=bbox,
                server_url=server_url,
            )

            return (
                graph,
                server_url,
            )

        except Exception as error:
            last_error = error

            print()
            print(
                "Server không sử dụng được:"
            )

            print(
                server_url
            )

            print()
            print(
                "Chuyển sang Overpass "
                "server tiếp theo..."
            )

    raise RuntimeError(
        "Không thể tải road graph từ "
        "bất kỳ Overpass server nào."
    ) from last_error


# ============================================================
# GRAPH VALIDATION
# ============================================================

def print_graph_summary(
    service: RoadNetworkService,
) -> None:
    """
    In thông tin cơ bản của road graph.
    """

    graph = service.graph

    print()
    print(
        "========================================"
    )

    print(
        "           ROAD GRAPH SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "Nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Edges:",
        graph.number_of_edges(),
    )

    print(
        "CRS:",
        graph.graph.get(
            "crs",
            "UNKNOWN",
        ),
    )


# ============================================================
# MAIN DOWNLOAD PIPELINE
# ============================================================

def build_and_save_graph(
    force: bool = False,
) -> None:
    """
    Pipeline:

    XeDu
      ↓
    delivery bbox
      ↓
    OpenStreetMap / Overpass
      ↓
    drive graph
      ↓
    speed_kph
      ↓
    travel_time
      ↓
    GraphML
    """

    MAPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Nếu graph đã tồn tại thì không tải lại
    # --------------------------------------------------------

    if (
        GRAPH_PATH.exists()
        and not force
    ):
        print(
            "Road graph đã tồn tại:"
        )

        print(
            GRAPH_PATH
        )

        print()
        print(
            "Không tải lại."
        )

        print(
            "Nếu muốn tải mới, chạy:"
        )

        print(
            "python -m "
            "routing.download_road_graph "
            "--force"
        )

        return

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bbox = load_delivery_bbox()

    west, south, east, north = (
        bbox
    )

    print()
    print(
        "========================================"
    )

    print(
        "             XEDU BOUNDING BOX"
    )

    print(
        "========================================"
    )

    print(
        f"West : {west:.6f}"
    )

    print(
        f"South: {south:.6f}"
    )

    print(
        f"East : {east:.6f}"
    )

    print(
        f"North: {north:.6f}"
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    print()
    print(
        "Đang tải road network..."
    )

    graph, server_url = (
        download_graph(
            bbox
        )
    )

    print()
    print(
        "Download thành công từ:"
    )

    print(
        server_url
    )

    # --------------------------------------------------------
    # Service
    # --------------------------------------------------------

    service = RoadNetworkService(
        graph
    )

    print_graph_summary(
        service
    )

    # --------------------------------------------------------
    # Estimated travel time
    # --------------------------------------------------------

    print()
    print(
        "Đang thêm estimated "
        "road speed..."
    )

    print(
        "Fallback speed: 30 km/h"
    )

    service.add_estimated_travel_times(
        fallback_kph=30.0
    )

    # --------------------------------------------------------
    # Save GraphML
    # --------------------------------------------------------

    print()
    print(
        "Đang lưu GraphML..."
    )

    service.save_graphml(
        GRAPH_PATH
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if not GRAPH_PATH.exists():
        raise RuntimeError(
            "GraphML không được tạo."
        )

    file_size_mb = (
        GRAPH_PATH.stat().st_size
        / 1024
        / 1024
    )

    print()
    print(
        "========================================"
    )

    print(
        "              HOÀN TẤT"
    )

    print(
        "========================================"
    )

    print(
        "Graph path:"
    )

    print(
        GRAPH_PATH
    )

    print(
        "File size:",
        f"{file_size_mb:.2f} MB",
    )

    print()
    print(
        "Từ bước này app có thể "
        "load graph local."
    )

    print(
        "Không cần gọi OpenStreetMap "
        "mỗi lần chạy."
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download road network cho "
            "dataset XeDu."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Tải lại graph ngay cả khi "
            "GraphML đã tồn tại."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    build_and_save_graph(
        force=arguments.force
    )


if __name__ == "__main__":
    main()