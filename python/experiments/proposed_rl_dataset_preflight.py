from __future__ import annotations

from pathlib import Path

import pandas as pd

import experiments.final_kmeans_guided_dqn_validation as validation
import experiments.kmeans_guided_dqn_sensitivity as guided


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

SPLIT_PATH = (
    RESULTS_DIR
    / "proposed_rl_workload_split.csv"
)

DETAIL_PATH = (
    RESULTS_DIR
    / "proposed_rl_dataset_preflight_details.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "proposed_rl_dataset_preflight_summary.csv"
)

FINDINGS_PATH = (
    RESULTS_DIR
    / "proposed_rl_dataset_preflight_findings.txt"
)


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split() -> pd.DataFrame:
    """
    Load fixed Train / Validation / Test split.

    Expected:
        108 workloads
        76 train
        16 validation
        16 test
    """

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_PATH}"
        )

    split = pd.read_csv(
        SPLIT_PATH
    )

    required_columns = {
        "workload_id",
        "number_of_deliveries",
        "split",
    }

    missing_columns = (
        required_columns
        - set(split.columns)
    )

    if missing_columns:
        raise ValueError(
            "Split file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    split["workload_id"] = (
        split["workload_id"]
        .astype(int)
    )

    split["number_of_deliveries"] = (
        split["number_of_deliveries"]
        .astype(int)
    )

    split["split"] = (
        split["split"]
        .astype(str)
    )

    if len(split) != 108:
        raise ValueError(
            "Expected exactly 108 split workloads, "
            f"found {len(split)}."
        )

    if (
        split["workload_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate workload_id found "
            "in split file."
        )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        split["split"]
    )

    if actual_splits != expected_splits:
        raise ValueError(
            "Expected train/validation/test splits, "
            f"found {sorted(actual_splits)}."
        )

    return split


# ============================================================
# LOAD CANONICAL WORKLOAD MANIFEST
# ============================================================

def load_manifest() -> pd.DataFrame:
    """
    Load canonical manifest containing all 108 workloads.

    Raw manifest contains:
        delivery_ids

    build_workload_dataframe_from_manifest()
    expects:
        manifest_delivery_ids

    Therefore an alias is created without modifying
    the underlying delivery membership.
    """

    manifest = (
        validation.load_workload_manifest()
        .copy()
    )

    required_columns = {
        "workload_id",
        "wave_start",
        "wave_end",
        "number_of_deliveries",
        "delivery_ids",
        "wave_start_key",
        "wave_end_key",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Workload manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    manifest["workload_id"] = (
        manifest["workload_id"]
        .astype(int)
    )

    manifest["number_of_deliveries"] = (
        manifest["number_of_deliveries"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Compatibility alias
    # --------------------------------------------------------
    #
    # Raw manifest:
    #
    #     delivery_ids
    #
    # Existing final-validation helper expects:
    #
    #     manifest_delivery_ids
    #
    # Đây chỉ là alias, không thay đổi membership
    # của bất kỳ workload nào.
    # --------------------------------------------------------

    manifest[
        "manifest_delivery_ids"
    ] = manifest[
        "delivery_ids"
    ]

    if len(manifest) != 108:
        raise ValueError(
            "Expected exactly 108 workloads "
            "in canonical manifest, "
            f"found {len(manifest)}."
        )

    if (
        manifest["workload_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate workload_id found "
            "in workload manifest."
        )

    return manifest


# ============================================================
# SPLIT / MANIFEST CONSISTENCY
# ============================================================

def validate_split_against_manifest(
    *,
    split: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    """
    Ensure the frozen split refers to exactly the same
    108 canonical workloads.
    """

    split_ids = set(
        split["workload_id"]
        .astype(int)
    )

    manifest_ids = set(
        manifest["workload_id"]
        .astype(int)
    )

    missing_from_manifest = (
        split_ids
        - manifest_ids
    )

    missing_from_split = (
        manifest_ids
        - split_ids
    )

    if missing_from_manifest:
        raise ValueError(
            "Split contains workload IDs "
            "not found in manifest: "
            f"{sorted(missing_from_manifest)}"
        )

    if missing_from_split:
        raise ValueError(
            "Manifest contains workload IDs "
            "not found in split: "
            f"{sorted(missing_from_split)}"
        )

    merged = split.merge(
        manifest[
            [
                "workload_id",
                "number_of_deliveries",
            ]
        ],
        on="workload_id",
        how="left",
        suffixes=(
            "_split",
            "_manifest",
        ),
    )

    mismatched_sizes = merged[
        merged[
            "number_of_deliveries_split"
        ]
        != merged[
            "number_of_deliveries_manifest"
        ]
    ]

    if not mismatched_sizes.empty:
        raise ValueError(
            "Workload size mismatch between "
            "split and canonical manifest."
        )


# ============================================================
# INSPECT ONE WORKLOAD
# ============================================================

def inspect_workload(
    *,
    workload_id: int,
    expected_size: int,
    manifest: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
    split_name: str,
) -> dict[str, object]:
    """
    Rebuild one workload using the existing canonical
    final-validation helper and inspect its integrity.
    """

    matches = manifest[
        manifest["workload_id"]
        == workload_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one manifest row for "
            f"workload {workload_id}, "
            f"found {len(matches)}."
        )

    manifest_row = (
        matches.iloc[0]
    )

    workload = (
        validation
        .build_workload_dataframe_from_manifest(
            prepared_dataframe,
            manifest_row,
        )
        .reset_index(
            drop=True
        )
    )

    actual_size = len(
        workload
    )

    if actual_size != expected_size:
        raise ValueError(
            f"Workload {workload_id}: "
            f"split says N={expected_size}, "
            f"rebuilt workload has N={actual_size}."
        )

    required_columns = {
        "delivery_id",
        "expected_delivery_time",
        "wave_start",
        "wave_end",
        "road_node",
        "cluster_id",
    }

    missing_columns = (
        required_columns
        - set(workload.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Workload {workload_id} "
            "is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ========================================================
    # BASIC STRUCTURAL INTEGRITY
    # ========================================================

    duplicate_delivery_ids = int(
        workload[
            "delivery_id"
        ]
        .astype(str)
        .duplicated()
        .sum()
    )

    missing_road_nodes = int(
        workload[
            "road_node"
        ]
        .isna()
        .sum()
    )

    missing_cluster_ids = int(
        workload[
            "cluster_id"
        ]
        .isna()
        .sum()
    )

    # ========================================================
    # ROUTE START TIME
    # ========================================================
    #
    # Existing baseline convention:
    #
    #     start_time = wave_end
    #
    # Proposed RL keeps the same convention.
    # ========================================================

    wave_end = pd.to_datetime(
        workload[
            "wave_end"
        ],
        utc=True,
        errors="coerce",
    )

    invalid_wave_end_count = int(
        wave_end
        .isna()
        .sum()
    )

    valid_wave_end_values = (
        wave_end
        .dropna()
        .unique()
    )

    unique_wave_end_count = int(
        len(
            valid_wave_end_values
        )
    )

    if (
        invalid_wave_end_count == 0
        and unique_wave_end_count == 1
    ):
        start_time = pd.Timestamp(
            valid_wave_end_values[0]
        )
    else:
        start_time = None

    # ========================================================
    # DEADLINE DIAGNOSTICS
    # ========================================================

    deadlines = pd.to_datetime(
        workload[
            "expected_delivery_time"
        ],
        utc=True,
        errors="coerce",
    )

    missing_deadlines = int(
        deadlines
        .isna()
        .sum()
    )

    valid_deadlines = int(
        deadlines
        .notna()
        .sum()
    )

    negative_deadlines = 0

    zero_deadlines = 0

    min_relative_deadline_minutes = None

    max_relative_deadline_minutes = None

    if start_time is not None:
        relative_minutes = (
            (
                deadlines
                - start_time
            )
            .dt
            .total_seconds()
            / 60.0
        )

        valid_relative = (
            relative_minutes
            .dropna()
        )

        negative_deadlines = int(
            (
                valid_relative
                < 0.0
            )
            .sum()
        )

        zero_deadlines = int(
            (
                valid_relative
                == 0.0
            )
            .sum()
        )

        if not valid_relative.empty:
            min_relative_deadline_minutes = float(
                valid_relative.min()
            )

            max_relative_deadline_minutes = float(
                valid_relative.max()
            )

    # ========================================================
    # CLUSTER DIAGNOSTICS
    # ========================================================

    if missing_cluster_ids == 0:
        cluster_count = int(
            workload[
                "cluster_id"
            ]
            .astype(int)
            .nunique()
        )
    else:
        cluster_count = 0

    # ========================================================
    # PREFLIGHT STATUS
    # ========================================================

    structural_ok = (
        actual_size == expected_size
        and duplicate_delivery_ids == 0
        and missing_road_nodes == 0
        and missing_cluster_ids == 0
        and invalid_wave_end_count == 0
        and unique_wave_end_count == 1
    )

    current_environment_compatible = (
        structural_ok
        and negative_deadlines == 0
    )

    return {
        "workload_id": (
            workload_id
        ),

        "split": (
            split_name
        ),

        "number_of_deliveries": (
            actual_size
        ),

        "duplicate_delivery_ids": (
            duplicate_delivery_ids
        ),

        "missing_road_nodes": (
            missing_road_nodes
        ),

        "missing_cluster_ids": (
            missing_cluster_ids
        ),

        "cluster_count": (
            cluster_count
        ),

        "valid_deadlines": (
            valid_deadlines
        ),

        "missing_deadlines": (
            missing_deadlines
        ),

        "negative_relative_deadlines": (
            negative_deadlines
        ),

        "zero_relative_deadlines": (
            zero_deadlines
        ),

        "min_relative_deadline_minutes": (
            min_relative_deadline_minutes
        ),

        "max_relative_deadline_minutes": (
            max_relative_deadline_minutes
        ),

        "unique_wave_end_count": (
            unique_wave_end_count
        ),

        "invalid_wave_end_count": (
            invalid_wave_end_count
        ),

        "structural_ok": (
            structural_ok
        ),

        "current_environment_compatible": (
            current_environment_compatible
        ),
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    details: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate diagnostics by Train / Validation / Test.
    """

    rows: list[
        dict[str, object]
    ] = []

    split_order = (
        "train",
        "validation",
        "test",
    )

    for split_name in split_order:
        group = details[
            details["split"]
            == split_name
        ]

        if group.empty:
            continue

        rows.append(
            {
                "split": (
                    split_name
                ),

                "workloads": (
                    len(group)
                ),

                "deliveries": int(
                    group[
                        "number_of_deliveries"
                    ].sum()
                ),

                "valid_deadlines": int(
                    group[
                        "valid_deadlines"
                    ].sum()
                ),

                "missing_deadlines": int(
                    group[
                        "missing_deadlines"
                    ].sum()
                ),

                "negative_relative_deadlines": int(
                    group[
                        "negative_relative_deadlines"
                    ].sum()
                ),

                "workloads_with_negative_deadlines": int(
                    (
                        group[
                            "negative_relative_deadlines"
                        ]
                        > 0
                    )
                    .sum()
                ),

                "missing_road_nodes": int(
                    group[
                        "missing_road_nodes"
                    ].sum()
                ),

                "missing_cluster_ids": int(
                    group[
                        "missing_cluster_ids"
                    ].sum()
                ),

                "duplicate_delivery_ids": int(
                    group[
                        "duplicate_delivery_ids"
                    ].sum()
                ),

                "structurally_valid_workloads": int(
                    group[
                        "structural_ok"
                    ].sum()
                ),

                "environment_compatible_workloads": int(
                    group[
                        "current_environment_compatible"
                    ].sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)

    print(
        "PROPOSED RL - DATASET PREFLIGHT"
    )

    print("=" * 78)

    # --------------------------------------------------------
    # 1. Load frozen split
    # --------------------------------------------------------

    split = load_split()

    # --------------------------------------------------------
    # 2. Load canonical 108-workload manifest
    # --------------------------------------------------------

    manifest = load_manifest()

    # --------------------------------------------------------
    # 3. Verify split and manifest refer to exactly
    #    the same workloads
    # --------------------------------------------------------

    validate_split_against_manifest(
        split=split,
        manifest=manifest,
    )

    print()
    print(
        "Split / manifest consistency: PASS"
    )

    # --------------------------------------------------------
    # 4. Prepare canonical delivery dataframe
    # --------------------------------------------------------

    print()
    print(
        "Preparing canonical experiment dataframe..."
    )

    prepared_dataframe, _ = (
        guided.prepare_experiment_dataframe()
    )

    # --------------------------------------------------------
    # 5. Inspect all 108 workloads
    # --------------------------------------------------------

    rows: list[
        dict[str, object]
    ] = []

    split_order = pd.CategoricalDtype(
        categories=[
            "train",
            "validation",
            "test",
        ],
        ordered=True,
    )

    ordered_split = (
        split
        .copy()
    )

    ordered_split[
        "_split_order"
    ] = ordered_split[
        "split"
    ].astype(
        split_order
    )

    ordered_split = (
        ordered_split
        .sort_values(
            [
                "_split_order",
                "number_of_deliveries",
                "workload_id",
            ]
        )
        .drop(
            columns=[
                "_split_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    total_workloads = len(
        ordered_split
    )

    for index, row in ordered_split.iterrows():
        workload_id = int(
            row[
                "workload_id"
            ]
        )

        expected_size = int(
            row[
                "number_of_deliveries"
            ]
        )

        split_name = str(
            row[
                "split"
            ]
        )

        result = inspect_workload(
            workload_id=(
                workload_id
            ),

            expected_size=(
                expected_size
            ),

            manifest=(
                manifest
            ),

            prepared_dataframe=(
                prepared_dataframe
            ),

            split_name=(
                split_name
            ),
        )

        rows.append(
            result
        )

        status_text = (
            "OK"
            if result[
                "structural_ok"
            ]
            else "ERROR"
        )

        print(
            f"[{index + 1:3d}/{total_workloads}] "
            f"workload={workload_id:3d} | "
            f"split={split_name:10s} | "
            f"N={expected_size:2d} | "
            f"missing_deadline="
            f"{result['missing_deadlines']:2d} | "
            f"negative_deadline="
            f"{result['negative_relative_deadlines']:2d} | "
            f"status={status_text}"
        )

    # --------------------------------------------------------
    # 6. Build result tables
    # --------------------------------------------------------

    details = pd.DataFrame(
        rows
    )

    summary = build_summary(
        details
    )

    # --------------------------------------------------------
    # 7. Save CSV artifacts
    # --------------------------------------------------------

    details.to_csv(
        DETAIL_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 8. Global diagnostics
    # --------------------------------------------------------

    total_negative = int(
        details[
            "negative_relative_deadlines"
        ].sum()
    )

    negative_workloads = int(
        (
            details[
                "negative_relative_deadlines"
            ]
            > 0
        )
        .sum()
    )

    structural_failures = int(
        (
            ~details[
                "structural_ok"
            ]
        )
        .sum()
    )

    compatible_workloads = int(
        details[
            "current_environment_compatible"
        ].sum()
    )

    total_missing_deadlines = int(
        details[
            "missing_deadlines"
        ].sum()
    )

    total_missing_road_nodes = int(
        details[
            "missing_road_nodes"
        ].sum()
    )

    total_missing_cluster_ids = int(
        details[
            "missing_cluster_ids"
        ].sum()
    )

    total_duplicate_delivery_ids = int(
        details[
            "duplicate_delivery_ids"
        ].sum()
    )

    total_deliveries = int(
        details[
            "number_of_deliveries"
        ].sum()
    )

    # --------------------------------------------------------
    # 9. Print summary
    # --------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "PREFLIGHT SUMMARY"
    )

    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "GLOBAL CHECK"
    )

    print(
        "-" * 78
    )

    print(
        f"Total workloads: "
        f"{len(details)}"
    )

    print(
        f"Total deliveries: "
        f"{total_deliveries}"
    )

    print(
        f"Structural failures: "
        f"{structural_failures}"
    )

    print(
        f"Missing road nodes: "
        f"{total_missing_road_nodes}"
    )

    print(
        f"Missing cluster IDs: "
        f"{total_missing_cluster_ids}"
    )

    print(
        f"Duplicate delivery IDs: "
        f"{total_duplicate_delivery_ids}"
    )

    print(
        f"Missing deadlines: "
        f"{total_missing_deadlines}"
    )

    print(
        f"Negative relative deadlines: "
        f"{total_negative}"
    )

    print(
        "Workloads containing "
        f"negative deadlines: "
        f"{negative_workloads}"
    )

    print(
        "Compatible with current environment: "
        f"{compatible_workloads}/108"
    )

    # --------------------------------------------------------
    # 10. Overall status
    # --------------------------------------------------------

    all_structurally_valid = (
        structural_failures == 0
    )

    all_environment_compatible = (
        compatible_workloads
        == len(details)
    )

    if all_environment_compatible:
        preflight_status = (
            "PASS"
        )

    elif all_structurally_valid:
        preflight_status = (
            "STRUCTURE_PASS_DEADLINE_ACTION_REQUIRED"
        )

    else:
        preflight_status = (
            "FAIL"
        )

    print()
    print(
        f"PREFLIGHT STATUS: "
        f"{preflight_status}"
    )

    # --------------------------------------------------------
    # 11. Findings text
    # --------------------------------------------------------

    findings = [
        (
            "PROPOSED RL DATASET PREFLIGHT"
        ),

        "",

        (
            f"total_workloads = "
            f"{len(details)}"
        ),

        (
            f"total_deliveries = "
            f"{total_deliveries}"
        ),

        (
            f"structural_failures = "
            f"{structural_failures}"
        ),

        (
            f"missing_road_nodes = "
            f"{total_missing_road_nodes}"
        ),

        (
            f"missing_cluster_ids = "
            f"{total_missing_cluster_ids}"
        ),

        (
            f"duplicate_delivery_ids = "
            f"{total_duplicate_delivery_ids}"
        ),

        (
            f"missing_deadlines = "
            f"{total_missing_deadlines}"
        ),

        (
            f"negative_relative_deadlines = "
            f"{total_negative}"
        ),

        (
            "workloads_with_negative_deadlines = "
            f"{negative_workloads}"
        ),

        (
            "environment_compatible_workloads = "
            f"{compatible_workloads}"
        ),

        (
            f"preflight_status = "
            f"{preflight_status}"
        ),

        "",

        (
            "Route start-time convention:"
        ),

        (
            "  start_time = wave_end"
        ),

        "",

        (
            "Relative deadline definition:"
        ),

        (
            "  relative_deadline_minutes = "
            "expected_delivery_time - wave_end"
        ),

        "",

        (
            "Missing expected_delivery_time values "
            "remain missing and are excluded from "
            "deadline/SLA denominators."
        ),

        "",

        (
            "Negative relative deadlines are not "
            "silently clamped to zero because doing "
            "so would alter the original delivery "
            "constraint."
        ),

        "",

        (
            "The frozen Train/Validation/Test split "
            "is checked against the canonical "
            "108-workload manifest before inspection."
        ),
    ]

    FINDINGS_PATH.write_text(
        "\n".join(
            findings
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 12. Output paths
    # --------------------------------------------------------

    print()
    print(
        f"Details saved to: "
        f"{DETAIL_PATH}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Findings saved to: "
        f"{FINDINGS_PATH}"
    )


if __name__ == "__main__":
    main()