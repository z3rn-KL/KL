from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

SPLIT_PATH = (
    RESULTS_DIR
    / "proposed_rl_workload_split.csv"
)

BASELINE_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_all_results.csv"
)

PROPOSED_DETAILS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_test_details.csv"
)

COMBINED_DETAILS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_baseline_test_details.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_baseline_test_summary.csv"
)

PAIRWISE_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_baseline_pairwise.csv"
)

FINDINGS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_baseline_findings.txt"
)


EXPECTED_TEST_WORKLOADS = 16
EXPECTED_TEST_DELIVERIES = 131

BASELINE_ALGORITHMS = (
    "NN",
    "CW",
    "Q-Learning",
    "SARSA",
    "DQN",
)

PROPOSED_NAME = "Proposed RL"


# ============================================================
# COLUMN HELPERS
# ============================================================

def resolve_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    if required:
        raise ValueError(
            "Could not find required column. "
            f"Tried: {candidates}. "
            f"Available: {list(dataframe.columns)}"
        )

    return None


def normalize_algorithm_name(
    value: object,
) -> str:
    raw = str(value).strip()

    key = (
        raw.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    aliases = {
        "nn": "NN",
        "nearestneighbor": "NN",
        "nearestneighbour": "NN",

        "cw": "CW",
        "clarkewright": "CW",
        "clarkewrightsavings": "CW",

        "qlearning": "Q-Learning",
        "qlearn": "Q-Learning",

        "sarsa": "SARSA",

        "dqn": "DQN",
        "deepqnetwork": "DQN",
    }

    if key not in aliases:
        raise ValueError(
            f"Unknown algorithm label: {raw!r}"
        )

    return aliases[key]


# ============================================================
# LOAD FROZEN TEST SPLIT
# ============================================================

def load_test_split() -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_PATH}"
        )

    split = pd.read_csv(
        SPLIT_PATH
    )

    required = {
        "workload_id",
        "number_of_deliveries",
        "split",
    }

    missing = (
        required
        - set(split.columns)
    )

    if missing:
        raise ValueError(
            "Split file is missing columns: "
            f"{sorted(missing)}"
        )

    split["workload_id"] = (
        split["workload_id"]
        .astype(int)
    )

    split["number_of_deliveries"] = (
        split["number_of_deliveries"]
        .astype(int)
    )

    test = (
        split[
            split["split"]
            .astype(str)
            == "test"
        ]
        .copy()
        .sort_values(
            [
                "number_of_deliveries",
                "workload_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(test) != EXPECTED_TEST_WORKLOADS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_WORKLOADS} "
            f"Test workloads, found {len(test)}."
        )

    deliveries = int(
        test[
            "number_of_deliveries"
        ].sum()
    )

    if deliveries != EXPECTED_TEST_DELIVERIES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_DELIVERIES} "
            f"Test deliveries, found {deliveries}."
        )

    return test


# ============================================================
# BASELINE RESULTS
# ============================================================

def load_baseline_results(
    test_split: pd.DataFrame,
) -> pd.DataFrame:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline benchmark: "
            f"{BASELINE_PATH}"
        )

    raw = pd.read_csv(
        BASELINE_PATH
    )

    workload_col = resolve_column(
        raw,
        (
            "workload_id",
            "workload",
        ),
    )

    algorithm_col = resolve_column(
        raw,
        (
            "algorithm",
            "algorithm_name",
            "method",
        ),
    )

    size_col = resolve_column(
        raw,
        (
            "number_of_deliveries",
            "delivery_count",
            "n_deliveries",
            "size",
        ),
    )

    distance_col = resolve_column(
        raw,
        (
            "total_distance_km",
            "distance_km",
        ),
    )

    time_col = resolve_column(
        raw,
        (
            "total_travel_time_minutes",
            "travel_time_minutes",
        ),
    )

    late_col = resolve_column(
        raw,
        (
            "late_deliveries",
            "number_of_late_deliveries",
        ),
    )

    lateness_col = resolve_column(
        raw,
        (
            "total_lateness_minutes",
            "lateness_minutes",
        ),
    )

    baseline = pd.DataFrame(
        {
            "workload_id": (
                raw[
                    workload_col
                ].astype(int)
            ),

            "algorithm": (
                raw[
                    algorithm_col
                ].map(
                    normalize_algorithm_name
                )
            ),

            "number_of_deliveries": (
                raw[
                    size_col
                ].astype(int)
            ),

            "distance_km": (
                raw[
                    distance_col
                ].astype(float)
            ),

            "travel_time_minutes": (
                raw[
                    time_col
                ].astype(float)
            ),

            "late_deliveries": (
                raw[
                    late_col
                ].astype(int)
            ),

            "lateness_minutes": (
                raw[
                    lateness_col
                ].astype(float)
            ),
        }
    )

    test_ids = set(
        test_split[
            "workload_id"
        ].astype(int)
    )

    baseline = (
        baseline[
            baseline[
                "workload_id"
            ].isin(
                test_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    expected_rows = (
        EXPECTED_TEST_WORKLOADS
        * len(BASELINE_ALGORITHMS)
    )

    if len(baseline) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} baseline rows "
            f"(16 workloads x 5 algorithms), "
            f"found {len(baseline)}."
        )

    actual_algorithms = set(
        baseline[
            "algorithm"
        ]
    )

    expected_algorithms = set(
        BASELINE_ALGORITHMS
    )

    if actual_algorithms != expected_algorithms:
        raise ValueError(
            "Baseline algorithm set mismatch. "
            f"Found: {sorted(actual_algorithms)}"
        )

    duplicate_pairs = (
        baseline[
            [
                "workload_id",
                "algorithm",
            ]
        ]
        .duplicated()
        .any()
    )

    if duplicate_pairs:
        raise ValueError(
            "Duplicate workload/algorithm rows "
            "found in baseline benchmark."
        )

    # --------------------------------------------------------
    # Ensure workload sizes match frozen Test split
    # --------------------------------------------------------

    size_reference = (
        test_split[
            [
                "workload_id",
                "number_of_deliveries",
            ]
        ]
        .rename(
            columns={
                "number_of_deliveries":
                "split_number_of_deliveries"
            }
        )
    )

    checked = baseline.merge(
        size_reference,
        on="workload_id",
        how="left",
    )

    mismatch = checked[
        checked[
            "number_of_deliveries"
        ]
        != checked[
            "split_number_of_deliveries"
        ]
    ]

    if not mismatch.empty:
        raise ValueError(
            "Baseline workload size does not "
            "match frozen Test split."
        )

    # Preflight established that all 131 Test deliveries
    # have valid deadlines.
    baseline[
        "valid_deadlines"
    ] = baseline[
        "number_of_deliveries"
    ]

    baseline[
        "on_time_deliveries"
    ] = (
        baseline[
            "valid_deadlines"
        ]
        - baseline[
            "late_deliveries"
        ]
    )

    baseline[
        "on_time_rate"
    ] = (
        baseline[
            "on_time_deliveries"
        ]
        / baseline[
            "valid_deadlines"
        ]
    )

    return baseline


# ============================================================
# PROPOSED RL RESULTS
# ============================================================

def load_proposed_results(
    test_split: pd.DataFrame,
) -> pd.DataFrame:
    if not PROPOSED_DETAILS_PATH.exists():
        raise FileNotFoundError(
            "Missing Proposed RL final Test details: "
            f"{PROPOSED_DETAILS_PATH}"
        )

    raw = pd.read_csv(
        PROPOSED_DETAILS_PATH
    )

    required = {
        "workload_id",
        "number_of_deliveries",
        "distance_km",
        "travel_time_minutes",
        "late_deliveries",
        "valid_deadlines",
        "on_time_deliveries",
        "on_time_rate",
        "lateness_minutes",
    }

    missing = (
        required
        - set(raw.columns)
    )

    if missing:
        raise ValueError(
            "Proposed RL Test details missing: "
            f"{sorted(missing)}"
        )

    proposed = raw.copy()

    proposed["workload_id"] = (
        proposed[
            "workload_id"
        ].astype(int)
    )

    proposed[
        "number_of_deliveries"
    ] = proposed[
        "number_of_deliveries"
    ].astype(int)

    proposed["algorithm"] = (
        PROPOSED_NAME
    )

    if len(proposed) != EXPECTED_TEST_WORKLOADS:
        raise ValueError(
            "Expected exactly 16 Proposed RL "
            f"Test rows, found {len(proposed)}."
        )

    if (
        proposed[
            "workload_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate Proposed RL workload row."
        )

    expected_ids = set(
        test_split[
            "workload_id"
        ]
    )

    actual_ids = set(
        proposed[
            "workload_id"
        ]
    )

    if actual_ids != expected_ids:
        raise ValueError(
            "Proposed RL Test workload IDs "
            "do not match frozen Test split."
        )

    return proposed[
        [
            "workload_id",
            "algorithm",
            "number_of_deliveries",
            "distance_km",
            "travel_time_minutes",
            "late_deliveries",
            "valid_deadlines",
            "on_time_deliveries",
            "on_time_rate",
            "lateness_minutes",
        ]
    ].copy()


# ============================================================
# SUMMARY
# ============================================================

def summarize_algorithm(
    group: pd.DataFrame,
) -> dict[str, object]:
    total_deliveries = int(
        group[
            "number_of_deliveries"
        ].sum()
    )

    total_valid = int(
        group[
            "valid_deadlines"
        ].sum()
    )

    total_late = int(
        group[
            "late_deliveries"
        ].sum()
    )

    total_on_time = int(
        group[
            "on_time_deliveries"
        ].sum()
    )

    if total_valid == 0:
        weighted_sla = 1.0
    else:
        weighted_sla = (
            total_on_time
            / total_valid
        )

    return {
        "algorithm": (
            str(
                group[
                    "algorithm"
                ].iloc[0]
            )
        ),

        "workloads": (
            len(group)
        ),

        "deliveries": (
            total_deliveries
        ),

        "mean_distance_km": float(
            group[
                "distance_km"
            ].mean()
        ),

        "mean_travel_time_minutes": float(
            group[
                "travel_time_minutes"
            ].mean()
        ),

        "late_deliveries": (
            total_late
        ),

        "valid_deadlines": (
            total_valid
        ),

        "weighted_sla": (
            weighted_sla
        ),

        "total_lateness_minutes": float(
            group[
                "lateness_minutes"
            ].sum()
        ),

        "mean_lateness_minutes": float(
            group[
                "lateness_minutes"
            ].mean()
        ),
    }


def build_summary(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    preferred_order = [
        "NN",
        "CW",
        "Q-Learning",
        "SARSA",
        "DQN",
        PROPOSED_NAME,
    ]

    rows = []

    for algorithm in preferred_order:
        group = combined[
            combined[
                "algorithm"
            ]
            == algorithm
        ]

        if len(group) != EXPECTED_TEST_WORKLOADS:
            raise ValueError(
                f"{algorithm}: expected 16 rows, "
                f"found {len(group)}."
            )

        rows.append(
            summarize_algorithm(
                group
            )
        )

    summary = pd.DataFrame(
        rows
    )

    return summary


# ============================================================
# PAIRWISE COMPARISON AGAINST PROPOSED RL
# ============================================================

def build_pairwise(
    combined: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    proposed_summary = (
        summary[
            summary[
                "algorithm"
            ]
            == PROPOSED_NAME
        ]
        .iloc[0]
    )

    proposed_details = (
        combined[
            combined[
                "algorithm"
            ]
            == PROPOSED_NAME
        ][
            [
                "workload_id",
                "distance_km",
                "travel_time_minutes",
                "lateness_minutes",
            ]
        ]
        .rename(
            columns={
                "distance_km":
                    "proposed_distance_km",

                "travel_time_minutes":
                    "proposed_travel_time_minutes",

                "lateness_minutes":
                    "proposed_lateness_minutes",
            }
        )
    )

    rows = []

    for baseline_name in BASELINE_ALGORITHMS:
        baseline_summary = (
            summary[
                summary[
                    "algorithm"
                ]
                == baseline_name
            ]
            .iloc[0]
        )

        baseline_details = (
            combined[
                combined[
                    "algorithm"
                ]
                == baseline_name
            ][
                [
                    "workload_id",
                    "distance_km",
                    "travel_time_minutes",
                    "lateness_minutes",
                ]
            ]
            .rename(
                columns={
                    "distance_km":
                        "baseline_distance_km",

                    "travel_time_minutes":
                        "baseline_travel_time_minutes",

                    "lateness_minutes":
                        "baseline_lateness_minutes",
                }
            )
        )

        paired = baseline_details.merge(
            proposed_details,
            on="workload_id",
            how="inner",
            validate="one_to_one",
        )

        if len(paired) != EXPECTED_TEST_WORKLOADS:
            raise ValueError(
                f"Pairwise merge failed for "
                f"{baseline_name}."
            )

        # ----------------------------------------------------
        # Ratio-of-means improvement:
        #
        # positive = Proposed RL is better
        # for distance/time/lateness.
        # ----------------------------------------------------

        distance_improvement = (
            (
                float(
                    baseline_summary[
                        "mean_distance_km"
                    ]
                )
                - float(
                    proposed_summary[
                        "mean_distance_km"
                    ]
                )
            )
            / float(
                baseline_summary[
                    "mean_distance_km"
                ]
            )
            * 100.0
        )

        time_improvement = (
            (
                float(
                    baseline_summary[
                        "mean_travel_time_minutes"
                    ]
                )
                - float(
                    proposed_summary[
                        "mean_travel_time_minutes"
                    ]
                )
            )
            / float(
                baseline_summary[
                    "mean_travel_time_minutes"
                ]
            )
            * 100.0
        )

        baseline_total_lateness = float(
            baseline_summary[
                "total_lateness_minutes"
            ]
        )

        proposed_total_lateness = float(
            proposed_summary[
                "total_lateness_minutes"
            ]
        )

        if baseline_total_lateness > 0.0:
            lateness_improvement = (
                (
                    baseline_total_lateness
                    - proposed_total_lateness
                )
                / baseline_total_lateness
                * 100.0
            )
        else:
            lateness_improvement = np.nan

        # ----------------------------------------------------
        # Mean of paired per-workload improvements.
        #
        # Keep separate from ratio-of-means.
        # ----------------------------------------------------

        distance_denominator = (
            paired[
                "baseline_distance_km"
            ]
            .replace(
                0.0,
                np.nan,
            )
        )

        paired[
            "distance_improvement_pct"
        ] = (
            (
                paired[
                    "baseline_distance_km"
                ]
                - paired[
                    "proposed_distance_km"
                ]
            )
            / distance_denominator
            * 100.0
        )

        time_denominator = (
            paired[
                "baseline_travel_time_minutes"
            ]
            .replace(
                0.0,
                np.nan,
            )
        )

        paired[
            "time_improvement_pct"
        ] = (
            (
                paired[
                    "baseline_travel_time_minutes"
                ]
                - paired[
                    "proposed_travel_time_minutes"
                ]
            )
            / time_denominator
            * 100.0
        )

        rows.append(
            {
                "baseline_algorithm": (
                    baseline_name
                ),

                "test_workloads": (
                    EXPECTED_TEST_WORKLOADS
                ),

                "test_deliveries": (
                    EXPECTED_TEST_DELIVERIES
                ),

                "distance_improvement_ratio_of_means_pct": (
                    distance_improvement
                ),

                "travel_time_improvement_ratio_of_means_pct": (
                    time_improvement
                ),

                "mean_paired_distance_improvement_pct": float(
                    paired[
                        "distance_improvement_pct"
                    ].mean()
                ),

                "mean_paired_travel_time_improvement_pct": float(
                    paired[
                        "time_improvement_pct"
                    ].mean()
                ),

                "sla_change_percentage_points": (
                    (
                        float(
                            proposed_summary[
                                "weighted_sla"
                            ]
                        )
                        - float(
                            baseline_summary[
                                "weighted_sla"
                            ]
                        )
                    )
                    * 100.0
                ),

                "late_delivery_change": (
                    int(
                        proposed_summary[
                            "late_deliveries"
                        ]
                    )
                    - int(
                        baseline_summary[
                            "late_deliveries"
                        ]
                    )
                ),

                "lateness_improvement_pct": (
                    lateness_improvement
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FINDINGS
# ============================================================

def write_findings(
    *,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    lines = [
        (
            "PROPOSED RL - FINAL HELD-OUT "
            "BASELINE COMPARISON"
        ),

        "",

        (
            "All algorithms are compared on exactly "
            "the same 16 held-out Test workloads "
            "containing 131 deliveries."
        ),

        "",

        (
            "Proposed RL objective/model configuration "
            "was locked before Test evaluation."
        ),

        "",

        (
            "No Proposed RL training or tuning "
            "was performed on Test."
        ),

        "",

        (
            "Important interpretation:"
        ),

        (
            "ratio-of-means improvements and mean "
            "paired per-workload improvements are "
            "reported separately and must not be "
            "treated as the same statistic."
        ),

        "",
        "SUMMARY",
        "",
    ]

    for row in summary.itertuples(
        index=False
    ):
        lines.extend(
            [
                f"[{row.algorithm}]",
                (
                    f"mean_distance_km = "
                    f"{row.mean_distance_km:.9f}"
                ),
                (
                    "mean_travel_time_minutes = "
                    f"{row.mean_travel_time_minutes:.9f}"
                ),
                (
                    f"late_deliveries = "
                    f"{row.late_deliveries}"
                ),
                (
                    f"weighted_sla = "
                    f"{row.weighted_sla:.9f}"
                ),
                (
                    "total_lateness_minutes = "
                    f"{row.total_lateness_minutes:.9f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "PROPOSED RL VS BASELINES",
            "",
        ]
    )

    for row in pairwise.itertuples(
        index=False
    ):
        lines.extend(
            [
                (
                    f"[vs {row.baseline_algorithm}]"
                ),
                (
                    "distance improvement "
                    "(ratio of means) = "
                    f"{row.distance_improvement_ratio_of_means_pct:.6f}%"
                ),
                (
                    "travel time improvement "
                    "(ratio of means) = "
                    f"{row.travel_time_improvement_ratio_of_means_pct:.6f}%"
                ),
                (
                    "mean paired distance improvement = "
                    f"{row.mean_paired_distance_improvement_pct:.6f}%"
                ),
                (
                    "mean paired travel-time improvement = "
                    f"{row.mean_paired_travel_time_improvement_pct:.6f}%"
                ),
                (
                    "SLA change = "
                    f"{row.sla_change_percentage_points:.6f} "
                    "percentage points"
                ),
                (
                    "late-delivery change = "
                    f"{row.late_delivery_change}"
                ),
                "",
            ]
        )

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
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)

    print(
        "PROPOSED RL - FINAL TEST "
        "BASELINE COMPARISON"
    )

    print("=" * 88)

    print()
    print(
        "This script performs NO training."
    )

    print(
        "It compares all algorithms on the "
        "same frozen held-out Test split."
    )

    # --------------------------------------------------------
    # 1. Frozen Test split
    # --------------------------------------------------------

    test_split = load_test_split()

    print()
    print(
        f"Test workloads: "
        f"{len(test_split)}"
    )

    print(
        "Test deliveries: "
        f"{test_split['number_of_deliveries'].sum()}"
    )

    # --------------------------------------------------------
    # 2. Existing five-algorithm benchmark
    # --------------------------------------------------------

    baseline = load_baseline_results(
        test_split
    )

    print(
        f"Baseline rows loaded: "
        f"{len(baseline)}"
    )

    # --------------------------------------------------------
    # 3. Locked Proposed RL Test results
    # --------------------------------------------------------

    proposed = load_proposed_results(
        test_split
    )

    print(
        f"Proposed RL rows loaded: "
        f"{len(proposed)}"
    )

    # --------------------------------------------------------
    # 4. Combine
    # --------------------------------------------------------

    combined = pd.concat(
        [
            baseline,
            proposed,
        ],
        ignore_index=True,
    )

    expected_combined_rows = (
        EXPECTED_TEST_WORKLOADS
        * 6
    )

    if (
        len(combined)
        != expected_combined_rows
    ):
        raise ValueError(
            f"Expected {expected_combined_rows} "
            f"combined rows, found {len(combined)}."
        )

    combined.to_csv(
        COMBINED_DETAILS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    summary = build_summary(
        combined
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Proposed-vs-baseline paired comparison
    # --------------------------------------------------------

    pairwise = build_pairwise(
        combined,
        summary,
    )

    pairwise.to_csv(
        PAIRWISE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 7. Findings
    # --------------------------------------------------------

    write_findings(
        summary=summary,
        pairwise=pairwise,
    )

    # --------------------------------------------------------
    # 8. Display
    # --------------------------------------------------------

    print()
    print("=" * 88)

    print(
        "FINAL TEST ALGORITHM SUMMARY"
    )

    print("=" * 88)

    display_summary = summary[
        [
            "algorithm",
            "workloads",
            "deliveries",
            "mean_distance_km",
            "mean_travel_time_minutes",
            "late_deliveries",
            "weighted_sla",
            "total_lateness_minutes",
        ]
    ].copy()

    display_summary[
        "weighted_sla"
    ] = (
        display_summary[
            "weighted_sla"
        ]
        * 100.0
    )

    print(
        display_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 88)

    print(
        "PROPOSED RL VS BASELINES"
    )

    print("=" * 88)

    print(
        pairwise.to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretation convention:"
    )

    print(
        "  improvement > 0 "
        "means Proposed RL is better "
        "for distance/time/lateness."
    )

    print(
        "  SLA change > 0 means Proposed RL "
        "has higher weighted SLA."
    )

    print(
        "  late_delivery_change < 0 means "
        "Proposed RL has fewer late deliveries."
    )

    print()
    print(
        f"Combined details: "
        f"{COMBINED_DETAILS_PATH}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Pairwise: "
        f"{PAIRWISE_PATH}"
    )

    print(
        f"Findings: "
        f"{FINDINGS_PATH}"
    )


if __name__ == "__main__":
    main()