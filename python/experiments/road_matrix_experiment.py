from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from routing import (
    RoadMatrixBuilder,
    RoadNetworkService,
)


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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

DISTANCE_MATRIX_PATH = (
    RESULTS_DIR
    / "xedu_sample_distance_matrix_km.csv"
)

TRAVEL_TIME_MATRIX_PATH = (
    RESULTS_DIR
    / "xedu_sample_travel_time_matrix_min.csv"
)

WORKLOAD_PATH = (
    RESULTS_DIR
    / "xedu_sample_road_workload.csv"
)


SAMPLE_SIZE = 15


def load_snapped_deliveries() -> pd.DataFrame:
    if not ROAD_NODE_PATH.exists():
        raise FileNotFoundError(
            f"Missing snapped-delivery file: "
            f"{ROAD_NODE_PATH}"
        )

    dataframe = pd.read_csv(
        ROAD_NODE_PATH
    )

    required_columns = {
        "delivery_id",
        "latitude",
        "longitude",
        "road_node",
    }

    missing_columns = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.dropna(
        subset=[
            "delivery_id",
            "road_node",
        ]
    ).reset_index(
        drop=True
    )

    dataframe[
        "road_node"
    ] = dataframe[
        "road_node"
    ].astype(
        int
    )

    return dataframe


def select_sample_workload(
    dataframe: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Select a deterministic sample workload.

    We prefer unique road nodes so the matrix
    is easier to inspect.
    """

    unique_nodes = dataframe.drop_duplicates(
        subset=[
            "road_node",
        ]
    )

    if len(
        unique_nodes
    ) < sample_size:
        raise ValueError(
            "Not enough unique road nodes "
            "for requested sample size."
        )

    sample = (
        unique_nodes
        .head(
            sample_size
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    return sample


def load_road_network() -> RoadNetworkService:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing road graph: "
            f"{GRAPH_PATH}"
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


def build_matrix_dataframe(
    matrix: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )


def print_matrix_preview(
    title: str,
    dataframe: pd.DataFrame,
) -> None:
    print()
    print(
        "========================================"
    )

    print(
        title
    )

    print(
        "========================================"
    )

    preview_size = min(
        8,
        len(
            dataframe
        ),
    )

    preview = dataframe.iloc[
        :preview_size,
        :preview_size,
    ]

    print(
        preview.to_string(
            float_format=lambda value: (
                f"{value:.3f}"
            )
        )
    )


def print_matrix_statistics(
    distance_matrix: np.ndarray,
    travel_time_matrix: np.ndarray,
) -> None:
    distance_values = (
        distance_matrix[
            np.isfinite(
                distance_matrix
            )
            & (
                distance_matrix
                > 0
            )
        ]
    )

    travel_values = (
        travel_time_matrix[
            np.isfinite(
                travel_time_matrix
            )
            & (
                travel_time_matrix
                > 0
            )
        ]
    )

    print()
    print(
        "========================================"
    )

    print(
        "Road Matrix Statistics"
    )

    print(
        "========================================"
    )

    print(
        f"Reachable distance pairs: "
        f"{len(distance_values)}"
    )

    if len(
        distance_values
    ) > 0:
        print(
            f"Distance min: "
            f"{distance_values.min():.3f} km"
        )

        print(
            f"Distance mean: "
            f"{distance_values.mean():.3f} km"
        )

        print(
            f"Distance max: "
            f"{distance_values.max():.3f} km"
        )

    print()

    print(
        f"Reachable travel-time pairs: "
        f"{len(travel_values)}"
    )

    if len(
        travel_values
    ) > 0:
        print(
            f"Travel time min: "
            f"{travel_values.min():.2f} min"
        )

        print(
            f"Travel time mean: "
            f"{travel_values.mean():.2f} min"
        )

        print(
            f"Travel time max: "
            f"{travel_values.max():.2f} min"
        )

    unreachable_distance = int(
        np.isinf(
            distance_matrix
        ).sum()
    )

    unreachable_travel = int(
        np.isinf(
            travel_time_matrix
        ).sum()
    )

    print()

    print(
        f"Unreachable distance cells: "
        f"{unreachable_distance}"
    )

    print(
        f"Unreachable travel-time cells: "
        f"{unreachable_travel}"
    )


def save_results(
    workload_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    travel_time_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    workload_df.to_csv(
        WORKLOAD_PATH,
        index=False,
    )

    distance_df.to_csv(
        DISTANCE_MATRIX_PATH
    )

    travel_time_df.to_csv(
        TRAVEL_TIME_MATRIX_PATH
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
        WORKLOAD_PATH
    )

    print(
        DISTANCE_MATRIX_PATH
    )

    print(
        TRAVEL_TIME_MATRIX_PATH
    )


def main() -> None:
    print()
    print(
        "========================================"
    )

    print(
        "XeDu Road Matrix Experiment"
    )

    print(
        "========================================"
    )

    snapped_df = (
        load_snapped_deliveries()
    )

    print(
        f"Snapped deliveries: "
        f"{len(snapped_df)}"
    )

    workload_df = (
        select_sample_workload(
            snapped_df
        )
    )

    print(
        f"Sample workload size: "
        f"{len(workload_df)}"
    )

    print(
        f"Unique road nodes: "
        f"{workload_df['road_node'].nunique()}"
    )

    service = load_road_network()

    builder = RoadMatrixBuilder(
        service
    )

    node_ids = (
        workload_df[
            "road_node"
        ]
        .astype(
            int
        )
        .tolist()
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

    print()
    print(
        "Building road matrices..."
    )

    start_time = perf_counter()

    result = builder.build(
        node_ids
    )

    runtime = (
        perf_counter()
        - start_time
    )

    print(
        f"Matrix build runtime: "
        f"{runtime:.3f} seconds"
    )

    distance_df = (
        build_matrix_dataframe(
            result.distance_km,
            delivery_ids,
        )
    )

    travel_time_df = (
        build_matrix_dataframe(
            result.travel_time_minutes,
            delivery_ids,
        )
    )

    print_matrix_preview(
        "Distance Matrix Preview (km)",
        distance_df,
    )

    print_matrix_preview(
        "Travel Time Matrix Preview (minutes)",
        travel_time_df,
    )

    print_matrix_statistics(
        result.distance_km,
        result.travel_time_minutes,
    )

    save_results(
        workload_df,
        distance_df,
        travel_time_df,
    )


if __name__ == "__main__":
    main()