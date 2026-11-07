from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import experiments.final_kmeans_guided_dqn_validation as validation
import experiments.kmeans_guided_dqn_sensitivity as guided

from experiments.proposed_rl_dataset_preflight import (
    load_manifest,
    load_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

CACHE_DIR = (
    RESULTS_DIR
    / "proposed_rl_road_cache"
)

MANIFEST_PATH = (
    RESULTS_DIR
    / "proposed_rl_road_cache_manifest.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "proposed_rl_road_cache_summary.txt"
)


# Chỉ development data.
ALLOWED_SPLITS = {
    "train",
    "validation",
}


def build_workload_dataframe(
    *,
    workload_id: int,
    manifest: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    matches = manifest[
        manifest["workload_id"]
        == workload_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one manifest row for "
            f"workload {workload_id}, "
            f"found {len(matches)}."
        )

    workload = (
        validation
        .build_workload_dataframe_from_manifest(
            prepared_dataframe,
            matches.iloc[0],
        )
        .reset_index(
            drop=True
        )
    )

    if workload.empty:
        raise ValueError(
            f"Workload {workload_id} is empty."
        )

    return workload


def prepare_deadlines(
    workload: pd.DataFrame,
) -> tuple[
    pd.Timestamp,
    np.ndarray,
    np.ndarray,
]:
    """
    Convert absolute expected delivery times into
    minutes relative to wave_end.

    Returns:
        start_time
        deadlines_minutes
        deadline_valid_mask
    """

    wave_end = pd.to_datetime(
        workload["wave_end"],
        utc=True,
        errors="raise",
    )

    unique_wave_end = (
        wave_end
        .dropna()
        .unique()
    )

    if len(unique_wave_end) != 1:
        raise ValueError(
            "Expected exactly one wave_end."
        )

    start_time = pd.Timestamp(
        unique_wave_end[0]
    )

    expected = pd.to_datetime(
        workload[
            "expected_delivery_time"
        ],
        utc=True,
        errors="coerce",
    )

    deadline_valid = (
        expected
        .notna()
        .to_numpy(
            dtype=bool
        )
    )

    relative = (
        (
            expected
            - start_time
        )
        .dt
        .total_seconds()
        / 60.0
    )

    # NaN is intentionally retained for missing deadlines.
    deadlines_minutes = (
        relative
        .to_numpy(
            dtype=float
        )
    )

    finite_deadlines = (
        deadlines_minutes[
            deadline_valid
        ]
    )

    if (
        finite_deadlines.size > 0
        and np.any(
            finite_deadlines < 0.0
        )
    ):
        raise ValueError(
            "Negative relative deadline encountered."
        )

    return (
        start_time,
        deadlines_minutes,
        deadline_valid,
    )


def cache_one_workload(
    *,
    workload_id: int,
    split_name: str,
    expected_size: int,
    manifest: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
    depot_node: int,
    matrix_builder,
) -> dict[str, object]:
    workload = build_workload_dataframe(
        workload_id=workload_id,
        manifest=manifest,
        prepared_dataframe=prepared_dataframe,
    )

    if len(workload) != expected_size:
        raise ValueError(
            f"Workload {workload_id}: "
            f"expected N={expected_size}, "
            f"rebuilt N={len(workload)}."
        )

    (
        start_time,
        deadlines_minutes,
        deadline_valid,
    ) = prepare_deadlines(
        workload
    )

    delivery_nodes = np.asarray(
        workload[
            "road_node"
        ]
        .astype(int)
        .tolist(),
        dtype=np.int64,
    )

    cluster_ids = np.asarray(
        workload[
            "cluster_id"
        ]
        .astype(int)
        .tolist(),
        dtype=np.int64,
    )

    delivery_ids = np.asarray(
        workload[
            "delivery_id"
        ]
        .astype(str)
        .tolist(),
        dtype=str,
    )

    node_ids = [
        int(depot_node),
        *delivery_nodes.tolist(),
    ]

    cache_path = (
        CACHE_DIR
        / f"workload_{workload_id}.npz"
    )

    matrix_start = perf_counter()

    road_matrix = (
        matrix_builder.build(
            node_ids
        )
    )

    matrix_runtime = (
        perf_counter()
        - matrix_start
    )

    distance_km = np.asarray(
        road_matrix.distance_km,
        dtype=np.float64,
    )

    travel_time_minutes = np.asarray(
        road_matrix.travel_time_minutes,
        dtype=np.float64,
    )

    expected_shape = (
        expected_size + 1,
        expected_size + 1,
    )

    if distance_km.shape != expected_shape:
        raise ValueError(
            f"Unexpected distance matrix shape "
            f"for workload {workload_id}: "
            f"{distance_km.shape}"
        )

    if (
        travel_time_minutes.shape
        != expected_shape
    ):
        raise ValueError(
            f"Unexpected travel-time matrix shape "
            f"for workload {workload_id}: "
            f"{travel_time_minutes.shape}"
        )

    if not np.isfinite(
        distance_km
    ).all():
        raise ValueError(
            f"Non-finite distance found "
            f"in workload {workload_id}."
        )

    if not np.isfinite(
        travel_time_minutes
    ).all():
        raise ValueError(
            f"Non-finite travel time found "
            f"in workload {workload_id}."
        )

    np.savez_compressed(
        cache_path,

        workload_id=np.asarray(
            [workload_id],
            dtype=np.int64,
        ),

        split=np.asarray(
            [split_name],
            dtype=str,
        ),

        number_of_deliveries=np.asarray(
            [expected_size],
            dtype=np.int64,
        ),

        depot_node=np.asarray(
            [depot_node],
            dtype=np.int64,
        ),

        node_ids=np.asarray(
            node_ids,
            dtype=np.int64,
        ),

        delivery_nodes=(
            delivery_nodes
        ),

        delivery_ids=(
            delivery_ids
        ),

        cluster_ids=(
            cluster_ids
        ),

        deadlines_minutes=(
            deadlines_minutes
        ),

        deadline_valid=(
            deadline_valid
        ),

        distance_km=(
            distance_km
        ),

        travel_time_minutes=(
            travel_time_minutes
        ),

        route_start_time=np.asarray(
            [start_time.isoformat()],
            dtype=str,
        ),
    )

    # Critical for memory safety:
    # do not accumulate shortest-path caches across workloads.
    matrix_builder.clear_cache()

    return {
        "workload_id": (
            workload_id
        ),

        "split": (
            split_name
        ),

        "number_of_deliveries": (
            expected_size
        ),

        "valid_deadlines": int(
            deadline_valid.sum()
        ),

        "matrix_size": int(
            expected_size + 1
        ),

        "matrix_runtime_seconds": (
            matrix_runtime
        ),

        "cache_file": str(
            cache_path
        ),

        "cache_bytes": int(
            cache_path.stat().st_size
        ),
    }


def validate_cache_file(
    path: Path,
) -> None:
    """
    Small read-back validation after writing.
    """

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        required = {
            "workload_id",
            "split",
            "number_of_deliveries",
            "depot_node",
            "node_ids",
            "delivery_nodes",
            "delivery_ids",
            "cluster_ids",
            "deadlines_minutes",
            "deadline_valid",
            "distance_km",
            "travel_time_minutes",
            "route_start_time",
        }

        missing = (
            required
            - set(data.files)
        )

        if missing:
            raise ValueError(
                f"Cache file {path} is missing: "
                f"{sorted(missing)}"
            )

        n = int(
            data[
                "number_of_deliveries"
            ][0]
        )

        expected_shape = (
            n + 1,
            n + 1,
        )

        if (
            data[
                "distance_km"
            ].shape
            != expected_shape
        ):
            raise ValueError(
                f"Invalid distance shape in {path}."
            )

        if (
            data[
                "travel_time_minutes"
            ].shape
            != expected_shape
        ):
            raise ValueError(
                f"Invalid time shape in {path}."
            )


def main() -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "PROPOSED RL - "
        "TRAIN/VALIDATION ROAD CACHE"
    )
    print("=" * 78)

    split = load_split()

    selected = (
        split[
            split[
                "split"
            ].isin(
                ALLOWED_SPLITS
            )
        ]
        .copy()
        .sort_values(
            [
                "split",
                "number_of_deliveries",
                "workload_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    train_count = int(
        (
            selected["split"]
            == "train"
        )
        .sum()
    )

    validation_count = int(
        (
            selected["split"]
            == "validation"
        )
        .sum()
    )

    if train_count != 76:
        raise ValueError(
            f"Expected 76 train workloads, "
            f"found {train_count}."
        )

    if validation_count != 16:
        raise ValueError(
            f"Expected 16 validation workloads, "
            f"found {validation_count}."
        )

    print(
        f"Train workloads: "
        f"{train_count}"
    )

    print(
        f"Validation workloads: "
        f"{validation_count}"
    )

    print(
        "Test workloads are intentionally "
        "not cached at this stage."
    )

    manifest = load_manifest()

    print()
    print(
        "Preparing canonical experiment dataframe..."
    )

    prepared_dataframe, _ = (
        guided.prepare_experiment_dataframe()
    )

    print()
    print(
        "Loading road graph..."
    )

    road_network = (
        validation.load_road_network()
    )

    depot, depot_node = (
        validation.get_depot_node(
            road_network
        )
    )

    depot_node = int(
        depot_node
    )

    print(
        f"Depot: {depot.name}"
    )

    print(
        f"Depot road node: "
        f"{depot_node}"
    )

    matrix_builder = (
        validation.RoadMatrixBuilder(
            road_network
        )
    )

    rows: list[
        dict[str, object]
    ] = []

    total_start = perf_counter()

    total = len(
        selected
    )

    for index, row in selected.iterrows():
        workload_id = int(
            row[
                "workload_id"
            ]
        )

        split_name = str(
            row[
                "split"
            ]
        )

        n = int(
            row[
                "number_of_deliveries"
            ]
        )

        print()
        print(
            f"[{index + 1:2d}/{total}] "
            f"workload={workload_id:3d} | "
            f"split={split_name:10s} | "
            f"N={n:2d}"
        )

        result = cache_one_workload(
            workload_id=(
                workload_id
            ),

            split_name=(
                split_name
            ),

            expected_size=(
                n
            ),

            manifest=(
                manifest
            ),

            prepared_dataframe=(
                prepared_dataframe
            ),

            depot_node=(
                depot_node
            ),

            matrix_builder=(
                matrix_builder
            ),
        )

        validate_cache_file(
            Path(
                result[
                    "cache_file"
                ]
            )
        )

        rows.append(
            result
        )

        # Autosave manifest after every workload.
        pd.DataFrame(
            rows
        ).to_csv(
            MANIFEST_PATH,
            index=False,
        )

        print(
            "  matrix runtime: "
            f"{result['matrix_runtime_seconds']:.3f} s"
        )

        print(
            "  cache size: "
            f"{result['cache_bytes'] / 1024.0:.2f} KiB"
        )

        print(
            "  status: PASS"
        )

        gc.collect()

    total_runtime = (
        perf_counter()
        - total_start
    )

    matrix_builder.clear_cache()

    cache_manifest = pd.DataFrame(
        rows
    )

    cache_manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    total_cache_bytes = int(
        cache_manifest[
            "cache_bytes"
        ].sum()
    )

    mean_matrix_runtime = float(
        cache_manifest[
            "matrix_runtime_seconds"
        ].mean()
    )

    total_matrix_runtime = float(
        cache_manifest[
            "matrix_runtime_seconds"
        ].sum()
    )

    print()
    print("=" * 78)
    print(
        "ROAD CACHE SUMMARY"
    )
    print("=" * 78)

    print(
        f"Cached workloads: "
        f"{len(cache_manifest)}"
    )

    print(
        f"Train: "
        f"{train_count}"
    )

    print(
        f"Validation: "
        f"{validation_count}"
    )

    print(
        f"Mean matrix runtime: "
        f"{mean_matrix_runtime:.3f} s"
    )

    print(
        f"Total matrix runtime: "
        f"{total_matrix_runtime:.2f} s"
    )

    print(
        f"Total experiment runtime: "
        f"{total_runtime / 60.0:.2f} min"
    )

    print(
        f"Total cache size: "
        f"{total_cache_bytes / 1024.0 / 1024.0:.3f} MiB"
    )

    summary_lines = [
        (
            "PROPOSED RL "
            "TRAIN/VALIDATION ROAD CACHE"
        ),

        "",

        (
            f"cached_workloads = "
            f"{len(cache_manifest)}"
        ),

        (
            f"train_workloads = "
            f"{train_count}"
        ),

        (
            f"validation_workloads = "
            f"{validation_count}"
        ),

        (
            "test_workloads_cached = 0"
        ),

        (
            f"depot_node = "
            f"{depot_node}"
        ),

        (
            f"mean_matrix_runtime_seconds = "
            f"{mean_matrix_runtime:.6f}"
        ),

        (
            f"total_matrix_runtime_seconds = "
            f"{total_matrix_runtime:.6f}"
        ),

        (
            f"total_runtime_seconds = "
            f"{total_runtime:.6f}"
        ),

        (
            f"total_cache_bytes = "
            f"{total_cache_bytes}"
        ),

        "",

        (
            "Test road matrices are intentionally "
            "deferred until the final objective "
            "and model configuration are locked."
        ),
    ]

    SUMMARY_PATH.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Cache directory: "
        f"{CACHE_DIR}"
    )

    print(
        f"Manifest saved to: "
        f"{MANIFEST_PATH}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()