"""
Expanded validation for adverse-delivery screening.

Scope
-----
All 108 routing workloads from the canonical dataset-scale benchmark.

The experiment validates road-network-based adverse-delivery signals:

1. unreachable_from_all_peers
2. road_isolation_candidate
3. insertion_detour_km

DBSCAN labels are intentionally not injected here because there is no
single canonical DBSCAN labeling artifact aligned with the 108 benchmark
workloads. DBSCAN spatial-noise analysis remains a separate clustering
analysis.

The detour threshold is not arbitrarily hard-coded. The script first
computes insertion detour values using the existing screening function
with detour_threshold_km=None, then uses the global 95th percentile as
an exploratory validation threshold.

This script does NOT modify routing algorithms or benchmark results.
"""

from __future__ import annotations

import gc
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import api.dependencies as deps

from clustering.adverse_deliveries import (
    screen_adverse_deliveries,
)

from routing import (
    RoadMatrixBuilder,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

DETAIL_PATH = (
    RESULTS_DIR
    / "adverse_delivery_validation_details.csv"
)

WORKLOAD_PATH = (
    RESULTS_DIR
    / "adverse_delivery_validation_workloads.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "adverse_delivery_validation_summary.csv"
)

FINDINGS_PATH = (
    RESULTS_DIR
    / "adverse_delivery_validation_findings.txt"
)

TOP_ISOLATION_PATH = (
    RESULTS_DIR
    / "adverse_delivery_top_isolation_candidates.csv"
)

TOP_DETOUR_PATH = (
    RESULTS_DIR
    / "adverse_delivery_top_detour_candidates.csv"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "chapter4"
    / "figures"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "figure_adverse_delivery_reason_counts.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

ISOLATION_MULTIPLIER = 3.0

DETOUR_QUANTILE = 0.95

TOP_N = 20


# ============================================================
# ROAD MATRIX EXTRACTION
# ============================================================

def extract_distance_matrix(
    road_matrix,
) -> np.ndarray:
    """
    Extract the distance matrix from RoadMatrixResult.

    The helper is intentionally defensive so this experiment
    stays compatible with the current routing implementation.
    """

    possible_attributes = [
        "distance_km",
        "distance_matrix_km",
        "distance_matrix",
        "distances_km",
    ]

    for attribute in (
        possible_attributes
    ):
        value = getattr(
            road_matrix,
            attribute,
            None,
        )

        if value is None:
            continue

        matrix = np.asarray(
            value,
            dtype=float,
        )

        if matrix.ndim == 2:
            return matrix

    if isinstance(
        road_matrix,
        np.ndarray,
    ):
        matrix = np.asarray(
            road_matrix,
            dtype=float,
        )

        if matrix.ndim == 2:
            return matrix

    raise RuntimeError(
        "Could not extract distance matrix "
        "from RoadMatrixResult."
    )


# ============================================================
# SAFE SAVE
# ============================================================

def safe_save_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    dataframe.to_csv(
        temp_path,
        index=False,
    )

    temp_path.replace(
        path
    )


# ============================================================
# EXISTING RESULTS / RESUME
# ============================================================

def load_existing_detail(
) -> pd.DataFrame:
    if not DETAIL_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        DETAIL_PATH
    )


def load_existing_workloads(
) -> pd.DataFrame:
    if not WORKLOAD_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        WORKLOAD_PATH
    )


def completed_workload_ids(
    workload_results: pd.DataFrame,
) -> set[int]:
    if workload_results.empty:
        return set()

    return set(
        pd.to_numeric(
            workload_results[
                "workload_id"
            ],
            errors="raise",
        )
        .astype(
            int
        )
        .tolist()
    )


# ============================================================
# WORKLOAD PROCESSING
# ============================================================

def process_workload(
    *,
    workload_id: int,
    delivery_ids: list[str],
    number_of_deliveries: int,
    wave_start,
    wave_end,
    matrix_builder: RoadMatrixBuilder,
    depot_node: int,
) -> tuple[
    list[dict],
    dict,
]:
    (
        deliveries,
        delivery_nodes,
    ) = (
        deps.resolve_deliveries(
            delivery_ids
        )
    )

    if (
        len(
            deliveries
        )
        != number_of_deliveries
    ):
        raise ValueError(
            f"Workload {workload_id}: "
            "delivery count mismatch."
        )

    matrix_start = (
        time.perf_counter()
    )

    road_matrix = (
        matrix_builder.build(
            [
                depot_node,
                *delivery_nodes,
            ]
        )
    )

    matrix_runtime = (
        time.perf_counter()
        - matrix_start
    )

    distance_matrix = (
        extract_distance_matrix(
            road_matrix
        )
    )

    expected_shape = (
        number_of_deliveries
        + 1
    )

    if (
        distance_matrix.shape
        != (
            expected_shape,
            expected_shape,
        )
    ):
        raise ValueError(
            f"Workload {workload_id}: "
            f"unexpected distance matrix shape "
            f"{distance_matrix.shape}; "
            f"expected "
            f"({expected_shape}, {expected_shape})."
        )

    report = (
        screen_adverse_deliveries(
            deliveries=deliveries,
            distance_km=(
                distance_matrix
            ),

            # DBSCAN is deliberately excluded from this
            # road-network validation.
            dbscan_labels=None,

            isolation_multiplier=(
                ISOLATION_MULTIPLIER
            ),

            # Pass 1 gathers detour values without
            # imposing an arbitrary threshold.
            detour_threshold_km=None,

            depot_included=True,
        )
    )

    detail_rows = []

    for entry in report.entries:
        reasons = set(
            entry.reasons
        )

        detail_rows.append(
            {
                "workload_id":
                    workload_id,

                "number_of_deliveries":
                    number_of_deliveries,

                "wave_start":
                    wave_start,

                "wave_end":
                    wave_end,

                "delivery_id":
                    str(
                        entry.delivery_id
                    ),

                "dbscan_noise":
                    bool(
                        entry.dbscan_noise
                    ),

                "nearest_peer_distance_km":
                    entry
                    .nearest_peer_distance_km,

                "isolation_ratio":
                    entry
                    .isolation_ratio,

                "insertion_detour_km":
                    entry
                    .insertion_detour_km,

                "unreachable_from_all_peers":
                    (
                        "unreachable_from_all_peers"
                        in reasons
                    ),

                "road_isolation_candidate":
                    (
                        "road_isolation_candidate"
                        in reasons
                    ),

                "base_reasons":
                    "|".join(
                        entry.reasons
                    ),
            }
        )

    workload_row = {
        "workload_id":
            workload_id,

        "number_of_deliveries":
            number_of_deliveries,

        "wave_start":
            wave_start,

        "wave_end":
            wave_end,

        "road_matrix_runtime_seconds":
            float(
                matrix_runtime
            ),

        "road_isolation_candidates":
            int(
                sum(
                    row[
                        "road_isolation_candidate"
                    ]
                    for row
                    in detail_rows
                )
            ),

        "unreachable_candidates":
            int(
                sum(
                    row[
                        "unreachable_from_all_peers"
                    ]
                    for row
                    in detail_rows
                )
            ),
    }

    return (
        detail_rows,
        workload_row,
    )


# ============================================================
# FINAL DETOUR THRESHOLD
# ============================================================

def apply_detour_threshold(
    detail: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    float,
]:
    result = (
        detail.copy()
    )

    finite_detours = (
        pd.to_numeric(
            result[
                "insertion_detour_km"
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if finite_detours.empty:
        raise ValueError(
            "No finite insertion-detour values "
            "were available."
        )

    threshold = float(
        finite_detours.quantile(
            DETOUR_QUANTILE
        )
    )

    result[
        "high_insertion_detour_candidate"
    ] = (
        pd.to_numeric(
            result[
                "insertion_detour_km"
            ],
            errors="coerce",
        )
        >= threshold
    )

    result[
        "final_flagged"
    ] = (
        result[
            "unreachable_from_all_peers"
        ].astype(
            bool
        )
        |
        result[
            "road_isolation_candidate"
        ].astype(
            bool
        )
        |
        result[
            "high_insertion_detour_candidate"
        ].astype(
            bool
        )
    )

    final_reasons = []

    for row in (
        result.itertuples(
            index=False
        )
    ):
        reasons = []

        if (
            row
            .unreachable_from_all_peers
        ):
            reasons.append(
                "unreachable_from_all_peers"
            )

        if (
            row
            .road_isolation_candidate
        ):
            reasons.append(
                "road_isolation_candidate"
            )

        if (
            row
            .high_insertion_detour_candidate
        ):
            reasons.append(
                "high_insertion_detour_candidate"
            )

        final_reasons.append(
            "|".join(
                reasons
            )
        )

    result[
        "final_reasons"
    ] = final_reasons

    return (
        result,
        threshold,
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    detail: pd.DataFrame,
    workload_results: pd.DataFrame,
    detour_threshold_km: float,
) -> pd.DataFrame:
    deliveries = int(
        len(
            detail
        )
    )

    workloads = int(
        detail[
            "workload_id"
        ]
        .nunique()
    )

    road_isolation = int(
        detail[
            "road_isolation_candidate"
        ]
        .sum()
    )

    unreachable = int(
        detail[
            "unreachable_from_all_peers"
        ]
        .sum()
    )

    high_detour = int(
        detail[
            "high_insertion_detour_candidate"
        ]
        .sum()
    )

    flagged = int(
        detail[
            "final_flagged"
        ]
        .sum()
    )

    finite_peer = (
        pd.to_numeric(
            detail[
                "nearest_peer_distance_km"
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    finite_isolation = (
        pd.to_numeric(
            detail[
                "isolation_ratio"
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    finite_detour = (
        pd.to_numeric(
            detail[
                "insertion_detour_km"
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    return pd.DataFrame(
        [
            {
                "workloads":
                    workloads,

                "deliveries":
                    deliveries,

                "isolation_multiplier":
                    ISOLATION_MULTIPLIER,

                "detour_quantile":
                    DETOUR_QUANTILE,

                "derived_detour_threshold_km":
                    detour_threshold_km,

                "road_isolation_candidates":
                    road_isolation,

                "road_isolation_rate":
                    (
                        road_isolation
                        / deliveries
                    ),

                "unreachable_candidates":
                    unreachable,

                "unreachable_rate":
                    (
                        unreachable
                        / deliveries
                    ),

                "high_detour_candidates":
                    high_detour,

                "high_detour_rate":
                    (
                        high_detour
                        / deliveries
                    ),

                "unique_flagged_deliveries":
                    flagged,

                "flagged_rate":
                    (
                        flagged
                        / deliveries
                    ),

                "median_nearest_peer_distance_km":
                    float(
                        finite_peer.median()
                    ),

                "mean_nearest_peer_distance_km":
                    float(
                        finite_peer.mean()
                    ),

                "max_nearest_peer_distance_km":
                    float(
                        finite_peer.max()
                    ),

                "median_isolation_ratio":
                    float(
                        finite_isolation.median()
                    ),

                "max_isolation_ratio":
                    float(
                        finite_isolation.max()
                    ),

                "median_insertion_detour_km":
                    float(
                        finite_detour.median()
                    ),

                "mean_insertion_detour_km":
                    float(
                        finite_detour.mean()
                    ),

                "max_insertion_detour_km":
                    float(
                        finite_detour.max()
                    ),

                "mean_road_matrix_runtime_seconds":
                    float(
                        workload_results[
                            "road_matrix_runtime_seconds"
                        ]
                        .mean()
                    ),
            }
        ]
    )


# ============================================================
# TOP CANDIDATES
# ============================================================

def save_top_candidates(
    detail: pd.DataFrame,
) -> None:
    isolation = (
        detail[
            detail[
                "isolation_ratio"
            ]
            .notna()
        ]
        .sort_values(
            [
                "isolation_ratio",
                "nearest_peer_distance_km",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            TOP_N
        )
    )

    detour = (
        detail[
            detail[
                "insertion_detour_km"
            ]
            .notna()
        ]
        .sort_values(
            "insertion_detour_km",
            ascending=False,
        )
        .head(
            TOP_N
        )
    )

    isolation.to_csv(
        TOP_ISOLATION_PATH,
        index=False,
    )

    detour.to_csv(
        TOP_DETOUR_PATH,
        index=False,
    )


# ============================================================
# FIGURE
# ============================================================

def create_figure(
    detail: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = [
        "Road isolation",
        "Unreachable",
        "High detour",
        "Unique flagged",
    ]

    values = [
        int(
            detail[
                "road_isolation_candidate"
            ]
            .sum()
        ),
        int(
            detail[
                "unreachable_from_all_peers"
            ]
            .sum()
        ),
        int(
            detail[
                "high_insertion_detour_candidate"
            ]
            .sum()
        ),
        int(
            detail[
                "final_flagged"
            ]
            .sum()
        ),
    ]

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        labels,
        values,
    )

    plt.xlabel(
        "Adverse-delivery signal"
    )

    plt.ylabel(
        "Number of deliveries"
    )

    plt.title(
        "Expanded Adverse-Delivery Validation"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FINDINGS
# ============================================================

def write_findings(
    summary: pd.DataFrame,
) -> None:
    row = (
        summary.iloc[
            0
        ]
    )

    lines = [
        "EXPANDED ADVERSE-DELIVERY VALIDATION",
        "=" * 72,
        "",
        "Scope:",
        (
            f"- Workloads: "
            f"{int(row['workloads'])}"
        ),
        (
            f"- Deliveries: "
            f"{int(row['deliveries'])}"
        ),
        (
            f"- Isolation multiplier: "
            f"{row['isolation_multiplier']:.2f}"
        ),
        (
            f"- Detour validation percentile: "
            f"{row['detour_quantile']:.2f}"
        ),
        (
            "- Derived detour threshold: "
            f"{row['derived_detour_threshold_km']:.4f} km"
        ),
        "",
        "Results:",
        (
            "- Road-isolation candidates: "
            f"{int(row['road_isolation_candidates'])} "
            f"({row['road_isolation_rate'] * 100:.2f}%)"
        ),
        (
            "- Unreachable-from-all-peers candidates: "
            f"{int(row['unreachable_candidates'])} "
            f"({row['unreachable_rate'] * 100:.2f}%)"
        ),
        (
            "- High-insertion-detour candidates: "
            f"{int(row['high_detour_candidates'])} "
            f"({row['high_detour_rate'] * 100:.2f}%)"
        ),
        (
            "- Unique flagged deliveries: "
            f"{int(row['unique_flagged_deliveries'])} "
            f"({row['flagged_rate'] * 100:.2f}%)"
        ),
        "",
        "Distance / isolation statistics:",
        (
            "- Median nearest-peer distance: "
            f"{row['median_nearest_peer_distance_km']:.4f} km"
        ),
        (
            "- Maximum nearest-peer distance: "
            f"{row['max_nearest_peer_distance_km']:.4f} km"
        ),
        (
            "- Median isolation ratio: "
            f"{row['median_isolation_ratio']:.4f}"
        ),
        (
            "- Maximum isolation ratio: "
            f"{row['max_isolation_ratio']:.4f}"
        ),
        (
            "- Median insertion detour: "
            f"{row['median_insertion_detour_km']:.4f} km"
        ),
        (
            "- Maximum insertion detour: "
            f"{row['max_insertion_detour_km']:.4f} km"
        ),
        "",
        "Interpretation:",
        (
            "- This experiment expands road-based adverse-delivery "
            "validation from a small audit sample to all benchmark "
            "workloads."
        ),
        (
            "- The 95th-percentile detour threshold is a "
            "data-derived exploratory validation threshold, "
            "not an authoritative XeDu business rule."
        ),
        (
            "- DBSCAN spatial-noise labels are not mixed into this "
            "experiment because no single canonical DBSCAN labeling "
            "artifact is aligned with these 108 routing workloads."
        ),
        (
            "- A flagged delivery is an investigation candidate, "
            "not automatically an invalid or unserviceable order."
        ),
        "",
    ]

    FINDINGS_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("=" * 72)
    print(
        "EXPANDED ADVERSE-DELIVERY VALIDATION"
    )
    print("=" * 72)

    manifest = (
        deps.get_workload_manifest()
        .copy()
    )

    manifest = (
        manifest
        .sort_values(
            "workload_id"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Workloads: "
        f"{len(manifest)}"
    )

    print(
        "Isolation multiplier: "
        f"{ISOLATION_MULTIPLIER}"
    )

    print(
        "Detour threshold: "
        "derived from global 95th percentile"
    )

    detail = (
        load_existing_detail()
    )

    workload_results = (
        load_existing_workloads()
    )

    completed = (
        completed_workload_ids(
            workload_results
        )
    )

    road_network = (
        deps.get_road_network()
    )

    depot = (
        deps.get_depot_context()
    )

    matrix_builder = (
        RoadMatrixBuilder(
            road_network
        )
    )

    total_workloads = (
        len(
            manifest
        )
    )

    for position, row in (
        manifest.iterrows()
    ):
        workload_id = int(
            row[
                "workload_id"
            ]
        )

        if (
            workload_id
            in completed
        ):
            print(
                f"SKIP "
                f"workload={workload_id}"
            )

            continue

        delivery_ids = (
            deps
            .parse_manifest_delivery_ids(
                row[
                    "delivery_ids"
                ]
            )
        )

        number_of_deliveries = int(
            row[
                "number_of_deliveries"
            ]
        )

        print()
        print(
            "-" * 72
        )

        print(
            f"Workload "
            f"{position + 1}/"
            f"{total_workloads}: "
            f"id={workload_id}, "
            f"N={number_of_deliveries}"
        )

        (
            detail_rows,
            workload_row,
        ) = (
            process_workload(
                workload_id=(
                    workload_id
                ),
                delivery_ids=(
                    delivery_ids
                ),
                number_of_deliveries=(
                    number_of_deliveries
                ),
                wave_start=(
                    row[
                        "wave_start"
                    ]
                ),
                wave_end=(
                    row[
                        "wave_end"
                    ]
                ),
                matrix_builder=(
                    matrix_builder
                ),
                depot_node=(
                    depot.road_node
                ),
            )
        )

        detail = pd.concat(
            [
                detail,
                pd.DataFrame(
                    detail_rows
                ),
            ],
            ignore_index=True,
        )

        workload_results = (
            pd.concat(
                [
                    workload_results,
                    pd.DataFrame(
                        [
                            workload_row
                        ]
                    ),
                ],
                ignore_index=True,
            )
        )

        safe_save_csv(
            detail,
            DETAIL_PATH,
        )

        safe_save_csv(
            workload_results,
            WORKLOAD_PATH,
        )

        completed.add(
            workload_id
        )

        print(
            "Road isolation candidates: "
            f"{workload_row['road_isolation_candidates']}"
        )

        print(
            "Unreachable candidates: "
            f"{workload_row['unreachable_candidates']}"
        )

        print(
            "Matrix runtime: "
            f"{workload_row['road_matrix_runtime_seconds']:.3f}s"
        )

        # Prevent the directional Dijkstra cache from
        # accumulating across all 108 workloads.
        matrix_builder.clear_cache()

        gc.collect()

    expected_deliveries = int(
        manifest[
            "number_of_deliveries"
        ]
        .sum()
    )

    if (
        len(
            detail
        )
        != expected_deliveries
    ):
        raise ValueError(
            "Expanded validation detail count "
            "does not match workload manifest: "
            f"{len(detail)} != "
            f"{expected_deliveries}"
        )

    if (
        detail[
            "workload_id"
        ]
        .nunique()
        != len(
            manifest
        )
    ):
        raise ValueError(
            "Not all workloads were validated."
        )

    (
        detail,
        detour_threshold,
    ) = (
        apply_detour_threshold(
            detail
        )
    )

    safe_save_csv(
        detail,
        DETAIL_PATH,
    )

    summary = (
        build_summary(
            detail=detail,
            workload_results=(
                workload_results
            ),
            detour_threshold_km=(
                detour_threshold
            ),
        )
    )

    safe_save_csv(
        summary,
        SUMMARY_PATH,
    )

    save_top_candidates(
        detail
    )

    create_figure(
        detail
    )

    write_findings(
        summary
    )

    print()
    print("=" * 72)
    print(
        "ADVERSE-DELIVERY SUMMARY"
    )
    print("=" * 72)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Created:"
    )

    for path in [
        DETAIL_PATH,
        WORKLOAD_PATH,
        SUMMARY_PATH,
        FINDINGS_PATH,
        TOP_ISOLATION_PATH,
        TOP_DETOUR_PATH,
        FIGURE_PATH,
    ]:
        print(
            f"- {path}"
        )

    print()
    print(
        "Expanded adverse-delivery "
        "validation complete."
    )
    print()


if __name__ == "__main__":
    main()