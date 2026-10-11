"""
Final four-method comparison for the thesis.

Compare on the SAME 12 final-validation workloads:

1. Nearest Neighbor
2. Clarke-Wright
3. Vanilla DQN
4. K-Means-guided DQN

The script does NOT retrain any algorithm.

Sources
-------
- Canonical dataset-scale benchmark:
    results/dataset_algorithm_all_results.csv

- Final K-Means-guided DQN validation:
    automatically located from results/

Outputs
-------
- final_guided_dqn_four_method_workloads.csv
- final_guided_dqn_four_method_summary.csv
- final_guided_dqn_four_method_pairwise.csv
- final_guided_dqn_comparison_findings.txt
- Chapter 4 figures

The operating-cost calculation uses the same thesis proxy scenario:

    distance_cost_per_km = 4000
    travel_time_cost_per_minute = 40000 / 60
    late_delivery_penalty = 0
    lateness_cost_per_minute = 0

These are scenario-based proxy costs, NOT verified XeDu accounting costs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.operating_cost import (
    OperatingCostConfig,
    OperatingCostEvaluator,
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

BENCHMARK_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_all_results.csv"
)

OUTPUT_WORKLOADS_PATH = (
    RESULTS_DIR
    / "final_guided_dqn_four_method_workloads.csv"
)

OUTPUT_SUMMARY_PATH = (
    RESULTS_DIR
    / "final_guided_dqn_four_method_summary.csv"
)

OUTPUT_PAIRWISE_PATH = (
    RESULTS_DIR
    / "final_guided_dqn_four_method_pairwise.csv"
)

OUTPUT_FINDINGS_PATH = (
    RESULTS_DIR
    / "final_guided_dqn_comparison_findings.txt"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "chapter4"
    / "figures"
)

DISTANCE_FIGURE_PATH = (
    FIGURE_DIR
    / "figure_guided_dqn_four_method_distance.png"
)

TIME_FIGURE_PATH = (
    FIGURE_DIR
    / "figure_guided_dqn_four_method_time.png"
)

COST_FIGURE_PATH = (
    FIGURE_DIR
    / "figure_guided_dqn_four_method_cost.png"
)


# ============================================================
# CONSTANTS
# ============================================================

METHOD_ORDER = [
    "nearest_neighbor",
    "clarke_wright",
    "vanilla_dqn",
    "guided_dqn",
]

METHOD_LABELS = {
    "nearest_neighbor":
        "Nearest Neighbor",

    "clarke_wright":
        "Clarke-Wright",

    "vanilla_dqn":
        "Vanilla DQN",

    "guided_dqn":
        "K-Means-guided DQN",
}

EXPECTED_WORKLOADS = 12


# Same scenario used by the operating-cost benchmark.
COST_CONFIG = OperatingCostConfig(
    distance_cost_per_km=4000.0,
    travel_time_cost_per_minute=(
        40000.0
        / 60.0
    ),
    late_delivery_penalty=0.0,
    lateness_cost_per_minute=0.0,
    cost_unit="VND_proxy_scenario",
)


# ============================================================
# COLUMN HELPERS
# ============================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    """
    Return the first matching column name.
    """

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    if required:
        raise ValueError(
            "Could not find any of these columns: "
            f"{candidates}\n"
            f"Available columns: "
            f"{list(dataframe.columns)}"
        )

    return None


def normalize_algorithm(
    value: object,
) -> str:
    text = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "nearest_neighbor":
            "nearest_neighbor",

        "nearestneighbour":
            "nearest_neighbor",

        "nearest_neighbour":
            "nearest_neighbor",

        "nn":
            "nearest_neighbor",

        "clarke_wright":
            "clarke_wright",

        "clarke_wright_savings":
            "clarke_wright",

        "cw":
            "clarke_wright",

        "dqn":
            "dqn",

        "deep_q_network":
            "dqn",
    }

    return aliases.get(
        text,
        text,
    )


# ============================================================
# FINAL VALIDATION FILE DISCOVERY
# ============================================================

def locate_validation_file(
) -> Path:
    """
    Automatically locate the detailed 12-workload
    K-Means-guided DQN final-validation CSV.
    """

    exact_candidates = [
        RESULTS_DIR
        / "final_kmeans_guided_dqn_validation_results.csv",

        RESULTS_DIR
        / "kmeans_guided_dqn_final_validation_results.csv",

        RESULTS_DIR
        / "final_guided_dqn_validation_results.csv",
    ]

    for path in exact_candidates:
        if not path.exists():
            continue

        dataframe = pd.read_csv(
            path,
            nrows=5,
        )

        if (
            "workload_id"
            in dataframe.columns
            and any(
                column
                in dataframe.columns
                for column
                in [
                    "guided_distance_km",
                    "guided_total_distance_km",
                ]
            )
        ):
            return path

    glob_patterns = [
        "*kmeans*guided*dqn*validation*.csv",
        "*guided*dqn*validation*.csv",
    ]

    checked = set()

    for pattern in glob_patterns:
        for path in sorted(
            RESULTS_DIR.glob(
                pattern
            )
        ):
            if path in checked:
                continue

            checked.add(
                path
            )

            try:
                dataframe = pd.read_csv(
                    path,
                    nrows=5,
                )
            except Exception:
                continue

            if (
                "workload_id"
                not in dataframe.columns
            ):
                continue

            has_guided_distance = any(
                column
                in dataframe.columns
                for column
                in [
                    "guided_distance_km",
                    "guided_total_distance_km",
                ]
            )

            has_guided_time = any(
                column
                in dataframe.columns
                for column
                in [
                    "guided_travel_time_minutes",
                    "guided_total_travel_time_minutes",
                ]
            )

            if (
                has_guided_distance
                and has_guided_time
            ):
                return path

    raise FileNotFoundError(
        "Could not locate the detailed "
        "K-Means-guided DQN final-validation CSV "
        f"inside {RESULTS_DIR}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_benchmark(
) -> pd.DataFrame:
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Missing benchmark: "
            f"{BENCHMARK_PATH}"
        )

    dataframe = pd.read_csv(
        BENCHMARK_PATH
    )

    required = [
        "workload_id",
        "algorithm",
        "number_of_deliveries",
        "total_distance_km",
        "total_travel_time_minutes",
        "on_time_rate",
        "on_time_deliveries",
        "late_deliveries",
        "total_lateness_minutes",
    ]

    missing = [
        column
        for column
        in required
        if column
        not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Canonical benchmark is missing: "
            f"{missing}"
        )

    dataframe = (
        dataframe.copy()
    )

    dataframe[
        "workload_id"
    ] = pd.to_numeric(
        dataframe[
            "workload_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dataframe[
        "algorithm_normalized"
    ] = (
        dataframe[
            "algorithm"
        ]
        .map(
            normalize_algorithm
        )
    )

    return dataframe


def load_validation(
) -> tuple[
    Path,
    pd.DataFrame,
]:
    path = (
        locate_validation_file()
    )

    dataframe = pd.read_csv(
        path
    )

    dataframe[
        "workload_id"
    ] = pd.to_numeric(
        dataframe[
            "workload_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    if (
        dataframe[
            "workload_id"
        ]
        .duplicated()
        .any()
    ):
        duplicated = (
            dataframe.loc[
                dataframe[
                    "workload_id"
                ]
                .duplicated(
                    keep=False
                ),
                "workload_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Final validation contains "
            f"duplicated workload IDs: "
            f"{duplicated}"
        )

    return (
        path,
        dataframe,
    )


# ============================================================
# EXTRACT GUIDED-DQN COLUMNS
# ============================================================

def prepare_guided_validation(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    size_column = (
        find_column(
            dataframe,
            [
                "number_of_deliveries",
                "delivery_count",
                "workload_size",
            ],
        )
    )

    guided_distance_column = (
        find_column(
            dataframe,
            [
                "guided_distance_km",
                "guided_total_distance_km",
            ],
        )
    )

    guided_time_column = (
        find_column(
            dataframe,
            [
                "guided_travel_time_minutes",
                "guided_total_travel_time_minutes",
            ],
        )
    )

    guided_sla_column = (
        find_column(
            dataframe,
            [
                "guided_on_time_rate",
                "guided_sla",
                "guided_on_time_ratio",
            ],
        )
    )

    baseline_distance_column = (
        find_column(
            dataframe,
            [
                "baseline_distance_km",
                "vanilla_distance_km",
                "baseline_total_distance_km",
            ],
            required=False,
        )
    )

    baseline_time_column = (
        find_column(
            dataframe,
            [
                "baseline_travel_time_minutes",
                "vanilla_travel_time_minutes",
                "baseline_total_travel_time_minutes",
            ],
            required=False,
        )
    )

    baseline_sla_column = (
        find_column(
            dataframe,
            [
                "baseline_on_time_rate",
                "vanilla_on_time_rate",
                "baseline_sla",
            ],
            required=False,
        )
    )

    guided_switch_column = (
        find_column(
            dataframe,
            [
                "guided_cluster_switch_rate",
                "guided_switch_rate",
            ],
            required=False,
        )
    )

    baseline_switch_column = (
        find_column(
            dataframe,
            [
                "baseline_cluster_switch_rate",
                "vanilla_cluster_switch_rate",
            ],
            required=False,
        )
    )

    selection_role_column = (
        find_column(
            dataframe,
            [
                "selection_role",
                "role",
            ],
            required=False,
        )
    )

    result = pd.DataFrame(
        {
            "workload_id":
                dataframe[
                    "workload_id"
                ].astype(
                    int
                ),

            "number_of_deliveries":
                pd.to_numeric(
                    dataframe[
                        size_column
                    ],
                    errors="raise",
                ).astype(
                    int
                ),

            "guided_distance_km":
                pd.to_numeric(
                    dataframe[
                        guided_distance_column
                    ],
                    errors="raise",
                ),

            "guided_travel_time_minutes":
                pd.to_numeric(
                    dataframe[
                        guided_time_column
                    ],
                    errors="raise",
                ),

            "guided_on_time_rate":
                pd.to_numeric(
                    dataframe[
                        guided_sla_column
                    ],
                    errors="raise",
                ),
        }
    )

    optional_mapping = [
        (
            baseline_distance_column,
            "validation_baseline_distance_km",
        ),
        (
            baseline_time_column,
            "validation_baseline_travel_time_minutes",
        ),
        (
            baseline_sla_column,
            "validation_baseline_on_time_rate",
        ),
        (
            guided_switch_column,
            "guided_cluster_switch_rate",
        ),
        (
            baseline_switch_column,
            "baseline_cluster_switch_rate",
        ),
        (
            selection_role_column,
            "selection_role",
        ),
    ]

    for (
        source,
        target,
    ) in optional_mapping:
        if source is None:
            continue

        result[
            target
        ] = dataframe[
            source
        ].values

    return result


# ============================================================
# CANONICAL METHOD EXTRACTION
# ============================================================

def extract_canonical_method(
    benchmark: pd.DataFrame,
    workload_ids: set[int],
    *,
    source_algorithm: str,
    method: str,
) -> pd.DataFrame:
    subset = (
        benchmark[
            (
                benchmark[
                    "algorithm_normalized"
                ]
                == source_algorithm
            )
            &
            (
                benchmark[
                    "workload_id"
                ]
                .isin(
                    workload_ids
                )
            )
        ]
        .copy()
    )

    if (
        len(
            subset
        )
        != len(
            workload_ids
        )
    ):
        raise ValueError(
            f"Expected "
            f"{len(workload_ids)} "
            f"{source_algorithm} rows, "
            f"found {len(subset)}."
        )

    if (
        subset[
            "workload_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            f"Duplicate {source_algorithm} "
            "benchmark rows detected."
        )

    result = subset[
        [
            "workload_id",
            "number_of_deliveries",
            "total_distance_km",
            "total_travel_time_minutes",
            "on_time_rate",
            "on_time_deliveries",
            "late_deliveries",
            "total_lateness_minutes",
        ]
    ].copy()

    result[
        "method"
    ] = method

    result = result.rename(
        columns={
            "total_distance_km":
                "distance_km",

            "total_travel_time_minutes":
                "travel_time_minutes",
        }
    )

    return result


# ============================================================
# GUIDED METHOD TABLE
# ============================================================

def build_guided_method(
    validation: pd.DataFrame,
    vanilla_dqn: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reuse the valid-deadline count from Vanilla DQN.

    Vanilla and Guided DQN use the exact same delivery
    sets for each selected workload, so the number of
    deliveries with valid deadlines is shared.
    """

    deadline_reference = (
        vanilla_dqn[
            [
                "workload_id",
                "on_time_deliveries",
                "late_deliveries",
            ]
        ]
        .copy()
    )

    deadline_reference[
        "valid_deadline_deliveries"
    ] = (
        deadline_reference[
            "on_time_deliveries"
        ]
        +
        deadline_reference[
            "late_deliveries"
        ]
    )

    guided = validation.merge(
        deadline_reference[
            [
                "workload_id",
                "valid_deadline_deliveries",
            ]
        ],
        on="workload_id",
        how="left",
        validate="one_to_one",
    )

    if (
        guided[
            "valid_deadline_deliveries"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Could not determine valid deadline "
            "counts for Guided DQN."
        )

    guided[
        "on_time_deliveries"
    ] = (
        guided[
            "guided_on_time_rate"
        ]
        *
        guided[
            "valid_deadline_deliveries"
        ]
    )

    guided[
        "late_deliveries"
    ] = (
        guided[
            "valid_deadline_deliveries"
        ]
        -
        guided[
            "on_time_deliveries"
        ]
    )

    # Financial lateness penalty is zero in the current
    # thesis scenario, so a missing guided lateness value
    # does not change operating-cost results.
    guided[
        "total_lateness_minutes"
    ] = 0.0

    result = pd.DataFrame(
        {
            "workload_id":
                guided[
                    "workload_id"
                ],

            "number_of_deliveries":
                guided[
                    "number_of_deliveries"
                ],

            "distance_km":
                guided[
                    "guided_distance_km"
                ],

            "travel_time_minutes":
                guided[
                    "guided_travel_time_minutes"
                ],

            "on_time_rate":
                guided[
                    "guided_on_time_rate"
                ],

            "on_time_deliveries":
                guided[
                    "on_time_deliveries"
                ],

            "late_deliveries":
                guided[
                    "late_deliveries"
                ],

            "total_lateness_minutes":
                guided[
                    "total_lateness_minutes"
                ],

            "method":
                "guided_dqn",
        }
    )

    return result


# ============================================================
# COST
# ============================================================

def add_estimated_cost(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        dataframe.copy()
    )

    evaluator = (
        OperatingCostEvaluator(
            COST_CONFIG
        )
    )

    costs = []

    for row in (
        result.itertuples(
            index=False
        )
    ):
        evaluation = (
            evaluator.evaluate(
                total_distance_km=float(
                    row.distance_km
                ),
                total_travel_time_minutes=float(
                    row
                    .travel_time_minutes
                ),
                late_deliveries=int(
                    round(
                        float(
                            row
                            .late_deliveries
                        )
                    )
                ),
                total_lateness_minutes=float(
                    row
                    .total_lateness_minutes
                ),
            )
        )

        costs.append(
            float(
                evaluation
                .total_estimated_cost
            )
        )

    result[
        "estimated_operating_cost"
    ] = costs

    result[
        "cost_unit"
    ] = (
        COST_CONFIG.cost_unit
    )

    return result


# ============================================================
# CROSS-CHECK FINAL-VALIDATION BASELINE
# ============================================================

def audit_validation_baseline(
    validation: pd.DataFrame,
    vanilla_dqn: pd.DataFrame,
) -> dict[str, float]:
    merged = validation.merge(
        vanilla_dqn[
            [
                "workload_id",
                "distance_km",
                "travel_time_minutes",
                "on_time_rate",
            ]
        ],
        on="workload_id",
        how="left",
        validate="one_to_one",
    )

    audit = {}

    if (
        "validation_baseline_distance_km"
        in merged.columns
    ):
        audit[
            "max_baseline_distance_difference_km"
        ] = float(
            (
                pd.to_numeric(
                    merged[
                        "validation_baseline_distance_km"
                    ],
                    errors="coerce",
                )
                -
                merged[
                    "distance_km"
                ]
            )
            .abs()
            .max()
        )

    if (
        "validation_baseline_travel_time_minutes"
        in merged.columns
    ):
        audit[
            "max_baseline_time_difference_minutes"
        ] = float(
            (
                pd.to_numeric(
                    merged[
                        "validation_baseline_travel_time_minutes"
                    ],
                    errors="coerce",
                )
                -
                merged[
                    "travel_time_minutes"
                ]
            )
            .abs()
            .max()
        )

    if (
        "validation_baseline_on_time_rate"
        in merged.columns
    ):
        audit[
            "max_baseline_sla_difference"
        ] = float(
            (
                pd.to_numeric(
                    merged[
                        "validation_baseline_on_time_rate"
                    ],
                    errors="coerce",
                )
                -
                merged[
                    "on_time_rate"
                ]
            )
            .abs()
            .max()
        )

    return audit


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    long_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for method in (
        METHOD_ORDER
    ):
        subset = (
            long_results[
                long_results[
                    "method"
                ]
                == method
            ]
        )

        if (
            len(
                subset
            )
            != EXPECTED_WORKLOADS
        ):
            raise ValueError(
                f"{method}: expected "
                f"{EXPECTED_WORKLOADS} rows, "
                f"found {len(subset)}."
            )

        valid_deadlines = (
            subset[
                "on_time_deliveries"
            ]
            +
            subset[
                "late_deliveries"
            ]
        )

        total_valid = float(
            valid_deadlines.sum()
        )

        total_on_time = float(
            subset[
                "on_time_deliveries"
            ]
            .sum()
        )

        weighted_sla = (
            total_on_time
            / total_valid
            if total_valid > 0
            else float(
                "nan"
            )
        )

        rows.append(
            {
                "method":
                    method,

                "method_label":
                    METHOD_LABELS[
                        method
                    ],

                "workloads":
                    int(
                        len(
                            subset
                        )
                    ),

                "deliveries":
                    int(
                        subset[
                            "number_of_deliveries"
                        ]
                        .sum()
                    ),

                "mean_distance_km":
                    float(
                        subset[
                            "distance_km"
                        ]
                        .mean()
                    ),

                "mean_travel_time_minutes":
                    float(
                        subset[
                            "travel_time_minutes"
                        ]
                        .mean()
                    ),

                "weighted_on_time_rate":
                    weighted_sla,

                "mean_estimated_operating_cost":
                    float(
                        subset[
                            "estimated_operating_cost"
                        ]
                        .mean()
                    ),

                "total_estimated_operating_cost":
                    float(
                        subset[
                            "estimated_operating_cost"
                        ]
                        .sum()
                    ),

                "estimated_cost_per_delivery":
                    float(
                        subset[
                            "estimated_operating_cost"
                        ]
                        .sum()
                        /
                        subset[
                            "number_of_deliveries"
                        ]
                        .sum()
                    ),

                "cost_unit":
                    COST_CONFIG.cost_unit,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PAIRWISE VS GUIDED DQN
# ============================================================

def improvement_pct(
    baseline: float,
    candidate: float,
) -> float:
    """
    Positive = candidate is better / lower.
    """

    if np.isclose(
        baseline,
        0.0,
    ):
        return float(
            "nan"
        )

    return float(
        (
            baseline
            - candidate
        )
        / baseline
        * 100.0
    )


def build_pairwise(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    guided = (
        summary[
            summary[
                "method"
            ]
            == "guided_dqn"
        ]
        .iloc[
            0
        ]
    )

    rows = []

    for comparator in [
        "nearest_neighbor",
        "clarke_wright",
        "vanilla_dqn",
    ]:
        baseline = (
            summary[
                summary[
                    "method"
                ]
                == comparator
            ]
            .iloc[
                0
            ]
        )

        rows.append(
            {
                "comparison":
                    (
                        "Guided DQN vs "
                        f"{METHOD_LABELS[comparator]}"
                    ),

                "comparator":
                    comparator,

                "distance_improvement_pct":
                    improvement_pct(
                        float(
                            baseline[
                                "mean_distance_km"
                            ]
                        ),
                        float(
                            guided[
                                "mean_distance_km"
                            ]
                        ),
                    ),

                "travel_time_improvement_pct":
                    improvement_pct(
                        float(
                            baseline[
                                "mean_travel_time_minutes"
                            ]
                        ),
                        float(
                            guided[
                                "mean_travel_time_minutes"
                            ]
                        ),
                    ),

                "sla_change_percentage_points":
                    float(
                        (
                            guided[
                                "weighted_on_time_rate"
                            ]
                            -
                            baseline[
                                "weighted_on_time_rate"
                            ]
                        )
                        * 100.0
                    ),

                "estimated_cost_improvement_pct":
                    improvement_pct(
                        float(
                            baseline[
                                "mean_estimated_operating_cost"
                            ]
                        ),
                        float(
                            guided[
                                "mean_estimated_operating_cost"
                            ]
                        ),
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# WIDE WORKLOAD TABLE
# ============================================================

def build_wide_workload_table(
    long_results: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "distance_km",
        "travel_time_minutes",
        "on_time_rate",
        "estimated_operating_cost",
    ]

    wide = (
        long_results.pivot(
            index=[
                "workload_id",
                "number_of_deliveries",
            ],
            columns="method",
            values=metrics,
        )
    )

    wide.columns = [
        f"{method}_{metric}"
        for metric, method
        in wide.columns
    ]

    wide = (
        wide
        .reset_index()
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

    optional_columns = [
        "selection_role",
        "baseline_cluster_switch_rate",
        "guided_cluster_switch_rate",
    ]

    available = [
        column
        for column
        in optional_columns
        if column
        in validation.columns
    ]

    if available:
        extra = validation[
            [
                "workload_id",
                *available,
            ]
        ]

        wide = wide.merge(
            extra,
            on="workload_id",
            how="left",
            validate="one_to_one",
        )

    return wide


# ============================================================
# FIGURES
# ============================================================

def save_bar_figure(
    summary: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    labels = (
        summary[
            "method_label"
        ]
        .tolist()
    )

    values = (
        summary[
            column
        ]
        .tolist()
    )

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
        "Method"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def create_figures(
    summary: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_bar_figure(
        summary,
        column=(
            "mean_distance_km"
        ),
        ylabel=(
            "Mean distance (km)"
        ),
        title=(
            "Four-Method Distance Comparison "
            "on Final Validation Workloads"
        ),
        path=(
            DISTANCE_FIGURE_PATH
        ),
    )

    save_bar_figure(
        summary,
        column=(
            "mean_travel_time_minutes"
        ),
        ylabel=(
            "Mean travel time (minutes)"
        ),
        title=(
            "Four-Method Travel-Time Comparison "
            "on Final Validation Workloads"
        ),
        path=(
            TIME_FIGURE_PATH
        ),
    )

    save_bar_figure(
        summary,
        column=(
            "mean_estimated_operating_cost"
        ),
        ylabel=(
            "Mean estimated operating cost "
            "(VND proxy / workload)"
        ),
        title=(
            "Four-Method Estimated Cost Comparison "
            "on Final Validation Workloads"
        ),
        path=(
            COST_FIGURE_PATH
        ),
    )


# ============================================================
# FINDINGS
# ============================================================

def write_findings(
    validation_path: Path,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    validation: pd.DataFrame,
    baseline_audit: dict[str, float],
) -> None:
    guided = (
        summary[
            summary[
                "method"
            ]
            == "guided_dqn"
        ]
        .iloc[
            0
        ]
    )

    lines = [
        "FINAL FOUR-METHOD GUIDED-DQN COMPARISON",
        "=" * 72,
        "",
        "Scope:",
        (
            f"- Final-validation workloads: "
            f"{EXPECTED_WORKLOADS}"
        ),
        (
            f"- Deliveries: "
            f"{int(guided['deliveries'])}"
        ),
        (
            "- Methods: Nearest Neighbor, "
            "Clarke-Wright, Vanilla DQN, "
            "K-Means-guided DQN"
        ),
        (
            "- Guided validation source: "
            f"{validation_path.name}"
        ),
        "",
        "Operating-cost scenario:",
        (
            "- distance_cost_per_km = "
            f"{COST_CONFIG.distance_cost_per_km:.2f}"
        ),
        (
            "- travel_time_cost_per_minute = "
            f"{COST_CONFIG.travel_time_cost_per_minute:.4f}"
        ),
        (
            "- late_delivery_penalty = 0"
        ),
        (
            "- lateness_cost_per_minute = 0"
        ),
        (
            "- cost_unit = "
            f"{COST_CONFIG.cost_unit}"
        ),
        "",
        "Method summary:",
    ]

    for row in (
        summary.itertuples(
            index=False
        )
    ):
        lines.append(
            (
                f"- {row.method_label}: "
                f"distance={row.mean_distance_km:.4f} km; "
                f"time={row.mean_travel_time_minutes:.4f} min; "
                f"SLA={row.weighted_on_time_rate * 100:.2f}%; "
                f"estimated cost="
                f"{row.mean_estimated_operating_cost:.2f} "
                f"{row.cost_unit}/workload."
            )
        )

    lines.extend(
        [
            "",
            "Guided-DQN pairwise results:",
        ]
    )

    for row in (
        pairwise.itertuples(
            index=False
        )
    ):
        lines.append(
            (
                f"- {row.comparison}: "
                f"distance improvement="
                f"{row.distance_improvement_pct:+.2f}%; "
                f"time improvement="
                f"{row.travel_time_improvement_pct:+.2f}%; "
                f"SLA change="
                f"{row.sla_change_percentage_points:+.2f} pp; "
                f"estimated-cost improvement="
                f"{row.estimated_cost_improvement_pct:+.2f}%."
            )
        )

    if baseline_audit:
        lines.extend(
            [
                "",
                "Vanilla-DQN baseline audit:",
            ]
        )

        for key, value in (
            baseline_audit.items()
        ):
            lines.append(
                f"- {key} = {value:.12g}"
            )

    if (
        "baseline_cluster_switch_rate"
        in validation.columns
        and
        "guided_cluster_switch_rate"
        in validation.columns
    ):
        switch_subset = validation[
            [
                "baseline_cluster_switch_rate",
                "guided_cluster_switch_rate",
            ]
        ].dropna()

        if not switch_subset.empty:
            baseline_switch = float(
                switch_subset[
                    "baseline_cluster_switch_rate"
                ]
                .mean()
            )

            guided_switch = float(
                switch_subset[
                    "guided_cluster_switch_rate"
                ]
                .mean()
            )

            reduction = (
                (
                    baseline_switch
                    - guided_switch
                )
                / baseline_switch
                * 100.0
                if baseline_switch > 0
                else float(
                    "nan"
                )
            )

            lines.extend(
                [
                    "",
                    "Cluster-switch comparison "
                    "where baseline data are available:",
                    (
                        f"- Workloads: "
                        f"{len(switch_subset)}"
                    ),
                    (
                        f"- Vanilla mean switch rate: "
                        f"{baseline_switch:.4f}"
                    ),
                    (
                        f"- Guided mean switch rate: "
                        f"{guided_switch:.4f}"
                    ),
                    (
                        f"- Relative reduction: "
                        f"{reduction:.2f}%"
                    ),
                ]
            )

    lines.extend(
        [
            "",
            "Interpretation:",
            (
                "- Guided DQN must be interpreted as a "
                "trade-off method: cluster guidance can improve "
                "spatial continuity without necessarily minimizing "
                "distance or travel time."
            ),
            (
                "- Comparisons in this report use the exact same "
                "12 final-validation workloads for all four methods."
            ),
            (
                "- Estimated operating cost is a scenario-based "
                "proxy, not verified XeDu accounting data."
            ),
            "",
        ]
    )

    OUTPUT_FINDINGS_PATH.write_text(
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
        "FINAL GUIDED-DQN FOUR-METHOD COMPARISON"
    )
    print("=" * 72)

    benchmark = (
        load_benchmark()
    )

    (
        validation_path,
        raw_validation,
    ) = (
        load_validation()
    )

    print(
        "Validation file:"
    )

    print(
        validation_path
    )

    validation = (
        prepare_guided_validation(
            raw_validation
        )
    )

    if (
        len(
            validation
        )
        != EXPECTED_WORKLOADS
    ):
        raise ValueError(
            "Expected exactly "
            f"{EXPECTED_WORKLOADS} final-validation workloads, "
            f"found {len(validation)}."
        )

    workload_ids = set(
        validation[
            "workload_id"
        ]
        .tolist()
    )

    print()
    print(
        f"Workloads: "
        f"{len(workload_ids)}"
    )

    print(
        "Workload IDs:"
    )

    print(
        sorted(
            workload_ids
        )
    )

    nn = (
        extract_canonical_method(
            benchmark,
            workload_ids,
            source_algorithm=(
                "nearest_neighbor"
            ),
            method=(
                "nearest_neighbor"
            ),
        )
    )

    cw = (
        extract_canonical_method(
            benchmark,
            workload_ids,
            source_algorithm=(
                "clarke_wright"
            ),
            method=(
                "clarke_wright"
            ),
        )
    )

    vanilla = (
        extract_canonical_method(
            benchmark,
            workload_ids,
            source_algorithm=(
                "dqn"
            ),
            method=(
                "vanilla_dqn"
            ),
        )
    )

    baseline_audit = (
        audit_validation_baseline(
            validation,
            vanilla,
        )
    )

    guided = (
        build_guided_method(
            validation,
            vanilla,
        )
    )

    long_results = pd.concat(
        [
            nn,
            cw,
            vanilla,
            guided,
        ],
        ignore_index=True,
    )

    long_results = (
        add_estimated_cost(
            long_results
        )
    )

    summary = (
        build_summary(
            long_results
        )
    )

    pairwise = (
        build_pairwise(
            summary
        )
    )

    wide_workloads = (
        build_wide_workload_table(
            long_results,
            validation,
        )
    )

    wide_workloads.to_csv(
        OUTPUT_WORKLOADS_PATH,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY_PATH,
        index=False,
    )

    pairwise.to_csv(
        OUTPUT_PAIRWISE_PATH,
        index=False,
    )

    create_figures(
        summary
    )

    write_findings(
        validation_path=(
            validation_path
        ),
        summary=(
            summary
        ),
        pairwise=(
            pairwise
        ),
        validation=(
            validation
        ),
        baseline_audit=(
            baseline_audit
        ),
    )

    print()
    print("=" * 72)
    print(
        "FINAL FOUR-METHOD SUMMARY"
    )
    print("=" * 72)

    print(
        summary[
            [
                "method_label",
                "workloads",
                "deliveries",
                "mean_distance_km",
                "mean_travel_time_minutes",
                "weighted_on_time_rate",
                "mean_estimated_operating_cost",
                "estimated_cost_per_delivery",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print(
        "PAIRWISE: GUIDED DQN VS OTHER METHODS"
    )
    print("=" * 72)

    print(
        pairwise.to_string(
            index=False
        )
    )

    if baseline_audit:
        print()
        print(
            "Vanilla-DQN baseline audit:"
        )

        for key, value in (
            baseline_audit.items()
        ):
            print(
                f"{key}: "
                f"{value:.12g}"
            )

    print()
    print(
        "Created:"
    )

    for path in [
        OUTPUT_WORKLOADS_PATH,
        OUTPUT_SUMMARY_PATH,
        OUTPUT_PAIRWISE_PATH,
        OUTPUT_FINDINGS_PATH,
        DISTANCE_FIGURE_PATH,
        TIME_FIGURE_PATH,
        COST_FIGURE_PATH,
    ]:
        print(
            f"- {path}"
        )

    print()
    print(
        "Final four-method comparison complete."
    )
    print()


if __name__ == "__main__":
    main()