from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
    / "dataset_benchmark"
)

MASTER_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_all_results.csv"
)

WORKLOAD_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_workloads.csv"
)


OVERALL_OUTPUT_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_overall_analysis.csv"
)

BY_SIZE_OUTPUT_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_by_workload_size.csv"
)

PAIRWISE_OVERALL_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_pairwise_overall.csv"
)

PAIRWISE_BY_SIZE_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_pairwise_by_size.csv"
)

SIZE_COUNTS_PATH = (
    RESULTS_DIR
    / "dataset_workload_size_counts.csv"
)

KEY_FINDINGS_PATH = (
    RESULTS_DIR
    / "dataset_benchmark_key_findings.txt"
)


# ============================================================
# CONSTANTS
# ============================================================

ALGORITHM_ORDER = [
    "nearest_neighbor",
    "clarke_wright",
    "q_learning",
    "sarsa",
    "dqn",
]

RL_ALGORITHMS = [
    "q_learning",
    "sarsa",
    "dqn",
]

REFERENCE_ALGORITHMS = [
    "nearest_neighbor",
    "clarke_wright",
]


# ============================================================
# LOAD
# ============================================================

def normalize_workload_id(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )


def load_results() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {MASTER_PATH}"
        )

    dataframe = pd.read_csv(
        MASTER_PATH
    )

    required_columns = [
        "workload_id",
        "algorithm",
        "number_of_deliveries",
        "total_distance_km",
        "total_travel_time_minutes",
        "on_time_deliveries",
        "late_deliveries",
        "on_time_rate",
        "total_lateness_minutes",
        "max_lateness_minutes",
        "route_runtime_seconds",
        "matrix_preparation_seconds",
        "training_runtime_seconds",
        "greedy_reward",
        "model_size",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column
        not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Master CSV thiếu các cột: "
            + ", ".join(
                missing_columns
            )
        )

    dataframe[
        "workload_id"
    ] = normalize_workload_id(
        dataframe[
            "workload_id"
        ]
    )

    dataframe[
        "algorithm"
    ] = (
        dataframe[
            "algorithm"
        ]
        .astype(str)
        .str.strip()
    )

    # number_of_deliveries là nguồn chính xác
    # cho workload size.
    dataframe[
        "workload_size"
    ] = pd.to_numeric(
        dataframe[
            "number_of_deliveries"
        ],
        errors="raise",
    ).astype(int)

    # Alias ngắn để phần phân tích dễ đọc.
    dataframe[
        "distance_km"
    ] = dataframe[
        "total_distance_km"
    ]

    dataframe[
        "travel_time_minutes"
    ] = dataframe[
        "total_travel_time_minutes"
    ]

    return dataframe


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    dataframe: pd.DataFrame,
) -> None:
    print()
    print(
        "=" * 64
    )
    print(
        "DATASET BENCHMARK VALIDATION"
    )
    print(
        "=" * 64
    )

    number_of_rows = len(
        dataframe
    )

    number_of_workloads = (
        dataframe[
            "workload_id"
        ]
        .nunique()
    )

    algorithms = sorted(
        dataframe[
            "algorithm"
        ]
        .unique()
        .tolist()
    )

    min_size = int(
        dataframe[
            "workload_size"
        ].min()
    )

    max_size = int(
        dataframe[
            "workload_size"
        ].max()
    )

    print(
        "Rows:",
        number_of_rows,
    )

    print(
        "Workloads:",
        number_of_workloads,
    )

    print(
        "Algorithms:",
        algorithms,
    )

    print(
        "Workload size:",
        min_size,
        "->",
        max_size,
    )

    duplicates = (
        dataframe
        .duplicated(
            subset=[
                "workload_id",
                "algorithm",
            ]
        )
        .sum()
    )

    print(
        "Duplicate workload-algorithm rows:",
        duplicates,
    )

    if duplicates != 0:
        raise ValueError(
            "Có duplicate workload_id + algorithm."
        )

    if number_of_rows != 540:
        raise ValueError(
            f"Expected 540 runs, "
            f"nhưng có {number_of_rows}."
        )

    if number_of_workloads != 108:
        raise ValueError(
            f"Expected 108 workloads, "
            f"nhưng có {number_of_workloads}."
        )

    if set(
        algorithms
    ) != set(
        ALGORITHM_ORDER
    ):
        raise ValueError(
            "Danh sách algorithm không đúng expected."
        )

    if (
        min_size != 5
        or max_size != 16
    ):
        raise ValueError(
            "Workload size phải nằm trong "
            "khoảng 5–16."
        )

    runs_per_workload = (
        dataframe
        .groupby(
            "workload_id"
        )[
            "algorithm"
        ]
        .nunique()
    )

    incomplete = (
        runs_per_workload[
            runs_per_workload
            != len(
                ALGORITHM_ORDER
            )
        ]
    )

    if not incomplete.empty:
        raise ValueError(
            "Có workload chưa đủ 5 thuật toán:\n"
            + incomplete.to_string()
        )

    print(
        "Validation: PASSED"
    )


# ============================================================
# WORKLOAD DISTRIBUTION
# ============================================================

def build_size_counts(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    unique_workloads = (
        dataframe[
            [
                "workload_id",
                "workload_size",
            ]
        ]
        .drop_duplicates(
            subset=[
                "workload_id"
            ]
        )
    )

    result = (
        unique_workloads
        .groupby(
            "workload_size",
            as_index=False,
        )
        .agg(
            workloads=(
                "workload_id",
                "nunique",
            )
        )
        .sort_values(
            "workload_size"
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "deliveries"
    ] = (
        result[
            "workload_size"
        ]
        * result[
            "workloads"
        ]
    )

    return result


# ============================================================
# OVERALL SUMMARY
# ============================================================

def build_overall_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        dataframe
        .groupby(
            "algorithm",
            as_index=False,
        )
        .agg(
            workloads=(
                "workload_id",
                "nunique",
            ),

            deliveries=(
                "workload_size",
                "sum",
            ),

            mean_distance_km=(
                "distance_km",
                "mean",
            ),

            std_distance_km=(
                "distance_km",
                "std",
            ),

            median_distance_km=(
                "distance_km",
                "median",
            ),

            mean_travel_time_minutes=(
                "travel_time_minutes",
                "mean",
            ),

            std_travel_time_minutes=(
                "travel_time_minutes",
                "std",
            ),

            mean_on_time_rate=(
                "on_time_rate",
                "mean",
            ),

            min_on_time_rate=(
                "on_time_rate",
                "min",
            ),

            total_late_deliveries=(
                "late_deliveries",
                "sum",
            ),

            total_lateness_minutes=(
                "total_lateness_minutes",
                "sum",
            ),

            max_lateness_minutes=(
                "max_lateness_minutes",
                "max",
            ),

            mean_route_runtime_seconds=(
                "route_runtime_seconds",
                "mean",
            ),

            mean_matrix_preparation_seconds=(
                "matrix_preparation_seconds",
                "mean",
            ),

            mean_training_runtime_seconds=(
                "training_runtime_seconds",
                "mean",
            ),

            total_training_runtime_seconds=(
                "training_runtime_seconds",
                "sum",
            ),

            mean_greedy_reward=(
                "greedy_reward",
                "mean",
            ),

            std_greedy_reward=(
                "greedy_reward",
                "std",
            ),

            mean_model_size=(
                "model_size",
                "mean",
            ),
        )
    )

    algorithm_order = {
        algorithm: index
        for index, algorithm
        in enumerate(
            ALGORITHM_ORDER
        )
    }

    summary[
        "_order"
    ] = (
        summary[
            "algorithm"
        ]
        .map(
            algorithm_order
        )
    )

    summary = (
        summary
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# ============================================================
# SUMMARY BY WORKLOAD SIZE
# ============================================================

def build_by_size_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        dataframe
        .groupby(
            [
                "workload_size",
                "algorithm",
            ],
            as_index=False,
        )
        .agg(
            workloads=(
                "workload_id",
                "nunique",
            ),

            mean_distance_km=(
                "distance_km",
                "mean",
            ),

            std_distance_km=(
                "distance_km",
                "std",
            ),

            median_distance_km=(
                "distance_km",
                "median",
            ),

            mean_travel_time_minutes=(
                "travel_time_minutes",
                "mean",
            ),

            std_travel_time_minutes=(
                "travel_time_minutes",
                "std",
            ),

            mean_on_time_rate=(
                "on_time_rate",
                "mean",
            ),

            min_on_time_rate=(
                "on_time_rate",
                "min",
            ),

            total_late_deliveries=(
                "late_deliveries",
                "sum",
            ),

            total_lateness_minutes=(
                "total_lateness_minutes",
                "sum",
            ),

            max_lateness_minutes=(
                "max_lateness_minutes",
                "max",
            ),

            mean_route_runtime_seconds=(
                "route_runtime_seconds",
                "mean",
            ),

            mean_matrix_preparation_seconds=(
                "matrix_preparation_seconds",
                "mean",
            ),

            mean_training_runtime_seconds=(
                "training_runtime_seconds",
                "mean",
            ),

            mean_greedy_reward=(
                "greedy_reward",
                "mean",
            ),

            std_greedy_reward=(
                "greedy_reward",
                "std",
            ),

            mean_model_size=(
                "model_size",
                "mean",
            ),
        )
    )

    algorithm_order = {
        algorithm: index
        for index, algorithm
        in enumerate(
            ALGORITHM_ORDER
        )
    }

    summary[
        "_algorithm_order"
    ] = (
        summary[
            "algorithm"
        ]
        .map(
            algorithm_order
        )
    )

    summary = (
        summary
        .sort_values(
            [
                "workload_size",
                "_algorithm_order",
            ]
        )
        .drop(
            columns=[
                "_algorithm_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# ============================================================
# PAIRWISE ANALYSIS
# ============================================================

def build_pairwise_analysis(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    distance_pivot = (
        dataframe
        .pivot_table(
            index=[
                "workload_id",
                "workload_size",
            ],
            columns="algorithm",
            values="distance_km",
            aggfunc="first",
        )
    )

    time_pivot = (
        dataframe
        .pivot_table(
            index=[
                "workload_id",
                "workload_size",
            ],
            columns="algorithm",
            values="travel_time_minutes",
            aggfunc="first",
        )
    )

    on_time_pivot = (
        dataframe
        .pivot_table(
            index=[
                "workload_id",
                "workload_size",
            ],
            columns="algorithm",
            values="on_time_rate",
            aggfunc="first",
        )
    )

    overall_rows = []
    by_size_rows = []

    for algorithm in RL_ALGORITHMS:
        for reference in (
            REFERENCE_ALGORITHMS
        ):
            comparison = pd.DataFrame(
                {
                    "algorithm_distance":
                        distance_pivot[
                            algorithm
                        ],

                    "reference_distance":
                        distance_pivot[
                            reference
                        ],

                    "algorithm_time":
                        time_pivot[
                            algorithm
                        ],

                    "reference_time":
                        time_pivot[
                            reference
                        ],

                    "algorithm_on_time":
                        on_time_pivot[
                            algorithm
                        ],

                    "reference_on_time":
                        on_time_pivot[
                            reference
                        ],
                }
            ).dropna()

            comparison[
                "distance_improvement_pct"
            ] = (
                (
                    comparison[
                        "reference_distance"
                    ]
                    - comparison[
                        "algorithm_distance"
                    ]
                )
                / comparison[
                    "reference_distance"
                ]
                * 100.0
            )

            comparison[
                "time_improvement_pct"
            ] = (
                (
                    comparison[
                        "reference_time"
                    ]
                    - comparison[
                        "algorithm_time"
                    ]
                )
                / comparison[
                    "reference_time"
                ]
                * 100.0
            )

            comparison[
                "distance_win"
            ] = (
                comparison[
                    "algorithm_distance"
                ]
                < comparison[
                    "reference_distance"
                ]
            )

            comparison[
                "distance_tie"
            ] = np.isclose(
                comparison[
                    "algorithm_distance"
                ],
                comparison[
                    "reference_distance"
                ],
            )

            comparison[
                "time_win"
            ] = (
                comparison[
                    "algorithm_time"
                ]
                < comparison[
                    "reference_time"
                ]
            )

            comparison[
                "time_tie"
            ] = np.isclose(
                comparison[
                    "algorithm_time"
                ],
                comparison[
                    "reference_time"
                ],
            )

            comparison[
                "on_time_win"
            ] = (
                comparison[
                    "algorithm_on_time"
                ]
                > comparison[
                    "reference_on_time"
                ]
            )

            overall_rows.append(
                {
                    "algorithm":
                        algorithm,

                    "reference":
                        reference,

                    "workloads":
                        len(
                            comparison
                        ),

                    "mean_distance_improvement_pct":
                        comparison[
                            "distance_improvement_pct"
                        ].mean(),

                    "median_distance_improvement_pct":
                        comparison[
                            "distance_improvement_pct"
                        ].median(),

                    "distance_win_rate":
                        comparison[
                            "distance_win"
                        ].mean(),

                    "distance_tie_rate":
                        comparison[
                            "distance_tie"
                        ].mean(),

                    "mean_time_improvement_pct":
                        comparison[
                            "time_improvement_pct"
                        ].mean(),

                    "median_time_improvement_pct":
                        comparison[
                            "time_improvement_pct"
                        ].median(),

                    "time_win_rate":
                        comparison[
                            "time_win"
                        ].mean(),

                    "time_tie_rate":
                        comparison[
                            "time_tie"
                        ].mean(),

                    "on_time_win_rate":
                        comparison[
                            "on_time_win"
                        ].mean(),
                }
            )

            comparison = (
                comparison
                .reset_index()
            )

            for (
                workload_size,
                group,
            ) in comparison.groupby(
                "workload_size"
            ):
                by_size_rows.append(
                    {
                        "algorithm":
                            algorithm,

                        "reference":
                            reference,

                        "workload_size":
                            int(
                                workload_size
                            ),

                        "workloads":
                            len(
                                group
                            ),

                        "mean_distance_improvement_pct":
                            group[
                                "distance_improvement_pct"
                            ].mean(),

                        "median_distance_improvement_pct":
                            group[
                                "distance_improvement_pct"
                            ].median(),

                        "distance_win_rate":
                            group[
                                "distance_win"
                            ].mean(),

                        "distance_tie_rate":
                            group[
                                "distance_tie"
                            ].mean(),

                        "mean_time_improvement_pct":
                            group[
                                "time_improvement_pct"
                            ].mean(),

                        "median_time_improvement_pct":
                            group[
                                "time_improvement_pct"
                            ].median(),

                        "time_win_rate":
                            group[
                                "time_win"
                            ].mean(),

                        "time_tie_rate":
                            group[
                                "time_tie"
                            ].mean(),

                        "on_time_win_rate":
                            group[
                                "on_time_win"
                            ].mean(),
                    }
                )

    overall = pd.DataFrame(
        overall_rows
    )

    by_size = (
        pd.DataFrame(
            by_size_rows
        )
        .sort_values(
            [
                "reference",
                "algorithm",
                "workload_size",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        overall,
        by_size,
    )


# ============================================================
# PLOTS
# ============================================================

def save_figure(
    filename: str,
) -> None:
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def plot_workload_distribution(
    size_counts: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        size_counts[
            "workload_size"
        ],
        size_counts[
            "workloads"
        ],
    )

    plt.title(
        "Dataset-Scale Workload Distribution"
    )

    plt.xlabel(
        "Number of deliveries in workload"
    )

    plt.ylabel(
        "Number of workloads"
    )

    plt.xticks(
        size_counts[
            "workload_size"
        ]
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure(
        "workload_size_distribution.png"
    )


def plot_metric_by_size(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    algorithms: list[str],
    log_scale: bool = False,
) -> None:
    plt.figure(
        figsize=(10, 6)
    )

    plotted = False

    for algorithm in algorithms:
        subset = summary[
            summary[
                "algorithm"
            ]
            == algorithm
        ].copy()

        subset = subset.dropna(
            subset=[
                metric
            ]
        )

        if subset.empty:
            continue

        plt.plot(
            subset[
                "workload_size"
            ],
            subset[
                metric
            ],
            marker="o",
            label=algorithm,
        )

        plotted = True

    if not plotted:
        plt.close()
        return

    if log_scale:
        plt.yscale(
            "log"
        )

    plt.title(
        title
    )

    plt.xlabel(
        "Workload size"
    )

    plt.ylabel(
        ylabel
    )

    plt.xticks(
        sorted(
            summary[
                "workload_size"
            ].unique()
        )
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    save_figure(
        filename
    )


def plot_pairwise_improvement(
    pairwise_by_size: pd.DataFrame,
    reference: str,
    metric: str,
    filename: str,
) -> None:
    subset = pairwise_by_size[
        pairwise_by_size[
            "reference"
        ]
        == reference
    ].copy()

    if subset.empty:
        return

    if metric == "distance":
        column = (
            "mean_distance_improvement_pct"
        )

        title = (
            "RL Distance Improvement vs "
            + reference
        )

        ylabel = (
            "Mean distance improvement (%)"
        )

    else:
        column = (
            "mean_time_improvement_pct"
        )

        title = (
            "RL Travel-Time Improvement vs "
            + reference
        )

        ylabel = (
            "Mean travel-time improvement (%)"
        )

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in RL_ALGORITHMS:
        algorithm_data = subset[
            subset[
                "algorithm"
            ]
            == algorithm
        ]

        plt.plot(
            algorithm_data[
                "workload_size"
            ],
            algorithm_data[
                column
            ],
            marker="o",
            label=algorithm,
        )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.title(
        title
    )

    plt.xlabel(
        "Workload size"
    )

    plt.ylabel(
        ylabel
    )

    plt.xticks(
        sorted(
            subset[
                "workload_size"
            ].unique()
        )
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    save_figure(
        filename
    )


# ============================================================
# FINDINGS
# ============================================================

def build_key_findings(
    dataframe: pd.DataFrame,
    size_counts: pd.DataFrame,
    overall: pd.DataFrame,
    pairwise_overall: pd.DataFrame,
) -> str:
    lines = []

    lines.append(
        "DATASET-SCALE BENCHMARK KEY FINDINGS"
    )

    lines.append(
        "=" * 64
    )

    lines.append(
        f"Workloads: "
        f"{dataframe['workload_id'].nunique()}"
    )

    lines.append(
        f"Algorithm runs: "
        f"{len(dataframe)}"
    )

    lines.append(
        "Workload size range: "
        f"{dataframe['workload_size'].min()}"
        "-"
        f"{dataframe['workload_size'].max()}"
    )

    lines.append(
        "Total benchmark deliveries: "
        f"{size_counts['deliveries'].sum()}"
    )

    lines.append("")

    lines.append(
        "OVERALL RESULTS"
    )

    lines.append(
        "-" * 64
    )

    for _, row in (
        overall
        .iterrows()
    ):
        text = (
            f"{row['algorithm']}: "
            f"distance="
            f"{row['mean_distance_km']:.4f} km, "
            f"time="
            f"{row['mean_travel_time_minutes']:.4f} min, "
            f"on_time="
            f"{row['mean_on_time_rate']:.4f}, "
            f"late="
            f"{row['total_late_deliveries']:.0f}"
        )

        if not pd.isna(
            row[
                "mean_training_runtime_seconds"
            ]
        ):
            text += (
                ", mean_training="
                f"{row['mean_training_runtime_seconds']:.4f} s"
            )

        lines.append(
            text
        )

    lines.append("")
    lines.append(
        "PAIRWISE WORKLOAD-LEVEL COMPARISONS"
    )

    lines.append(
        "-" * 64
    )

    for _, row in (
        pairwise_overall
        .iterrows()
    ):
        lines.append(
            (
                f"{row['algorithm']} "
                f"vs {row['reference']}: "
                f"mean distance improvement="
                f"{row['mean_distance_improvement_pct']:.2f}%, "
                f"distance win="
                f"{row['distance_win_rate'] * 100:.1f}%, "
                f"distance tie="
                f"{row['distance_tie_rate'] * 100:.1f}%, "
                f"mean time improvement="
                f"{row['mean_time_improvement_pct']:.2f}%, "
                f"time win="
                f"{row['time_win_rate'] * 100:.1f}%"
            )
        )

    lines.append("")
    lines.append(
        "WORKLOAD DISTRIBUTION"
    )

    lines.append(
        "-" * 64
    )

    for _, row in (
        size_counts
        .iterrows()
    ):
        lines.append(
            (
                f"N={int(row['workload_size'])}: "
                f"{int(row['workloads'])} workloads, "
                f"{int(row['deliveries'])} deliveries"
            )
        )

    lines.append("")
    lines.append(
        "INTERPRETATION CAUTION"
    )

    lines.append(
        "-" * 64
    )

    lines.append(
        "Workload sizes 14, 15 and 16 each contain "
        "only one workload. Their values should be "
        "treated as individual case evidence rather "
        "than statistically stable trends."
    )

    lines.append(
        "Workload sizes with larger sample counts "
        "(especially 5, 6, 8 and 10) are more useful "
        "for assessing general trends."
    )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_results()

    validate_results(
        dataframe
    )

    size_counts = (
        build_size_counts(
            dataframe
        )
    )

    overall = (
        build_overall_summary(
            dataframe
        )
    )

    by_size = (
        build_by_size_summary(
            dataframe
        )
    )

    (
        pairwise_overall,
        pairwise_by_size,
    ) = build_pairwise_analysis(
        dataframe
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    size_counts.to_csv(
        SIZE_COUNTS_PATH,
        index=False,
    )

    overall.to_csv(
        OVERALL_OUTPUT_PATH,
        index=False,
    )

    by_size.to_csv(
        BY_SIZE_OUTPUT_PATH,
        index=False,
    )

    pairwise_overall.to_csv(
        PAIRWISE_OVERALL_PATH,
        index=False,
    )

    pairwise_by_size.to_csv(
        PAIRWISE_BY_SIZE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    plot_workload_distribution(
        size_counts
    )

    plot_metric_by_size(
        summary=by_size,
        metric="mean_distance_km",
        title=(
            "Mean Route Distance by Workload Size"
        ),
        ylabel="Mean distance (km)",
        filename=(
            "distance_by_workload_size.png"
        ),
        algorithms=ALGORITHM_ORDER,
    )

    plot_metric_by_size(
        summary=by_size,
        metric="mean_travel_time_minutes",
        title=(
            "Mean Travel Time by Workload Size"
        ),
        ylabel="Mean travel time (minutes)",
        filename=(
            "travel_time_by_workload_size.png"
        ),
        algorithms=ALGORITHM_ORDER,
    )

    plot_metric_by_size(
        summary=by_size,
        metric="mean_on_time_rate",
        title=(
            "Mean On-Time Rate by Workload Size"
        ),
        ylabel="Mean on-time rate",
        filename=(
            "on_time_rate_by_workload_size.png"
        ),
        algorithms=ALGORITHM_ORDER,
    )

    plot_metric_by_size(
        summary=by_size,
        metric=(
            "mean_training_runtime_seconds"
        ),
        title=(
            "RL Training Runtime by Workload Size"
        ),
        ylabel=(
            "Mean training runtime (seconds)"
        ),
        filename=(
            "rl_training_runtime_by_workload_size.png"
        ),
        algorithms=RL_ALGORITHMS,
        log_scale=True,
    )

    plot_metric_by_size(
        summary=by_size,
        metric="mean_model_size",
        title=(
            "RL Model Size by Workload Size"
        ),
        ylabel="Mean Q-table / model size",
        filename=(
            "rl_model_size_by_workload_size.png"
        ),
        algorithms=RL_ALGORITHMS,
        log_scale=True,
    )

    plot_metric_by_size(
        summary=by_size,
        metric="mean_greedy_reward",
        title=(
            "RL Greedy Reward by Workload Size"
        ),
        ylabel="Mean greedy reward",
        filename=(
            "rl_reward_by_workload_size.png"
        ),
        algorithms=RL_ALGORITHMS,
    )

    plot_pairwise_improvement(
        pairwise_by_size=(
            pairwise_by_size
        ),
        reference="nearest_neighbor",
        metric="distance",
        filename=(
            "rl_distance_improvement_vs_nn.png"
        ),
    )

    plot_pairwise_improvement(
        pairwise_by_size=(
            pairwise_by_size
        ),
        reference="nearest_neighbor",
        metric="time",
        filename=(
            "rl_time_improvement_vs_nn.png"
        ),
    )

    plot_pairwise_improvement(
        pairwise_by_size=(
            pairwise_by_size
        ),
        reference="clarke_wright",
        metric="distance",
        filename=(
            "rl_distance_improvement_vs_cw.png"
        ),
    )

    plot_pairwise_improvement(
        pairwise_by_size=(
            pairwise_by_size
        ),
        reference="clarke_wright",
        metric="time",
        filename=(
            "rl_time_improvement_vs_cw.png"
        ),
    )

    # --------------------------------------------------------
    # KEY FINDINGS
    # --------------------------------------------------------

    findings = build_key_findings(
        dataframe=dataframe,
        size_counts=size_counts,
        overall=overall,
        pairwise_overall=(
            pairwise_overall
        ),
    )

    KEY_FINDINGS_PATH.write_text(
        findings,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "=" * 64
    )
    print(
        "WORKLOAD DISTRIBUTION"
    )
    print(
        "=" * 64
    )

    print(
        size_counts.to_string(
            index=False
        )
    )

    print()
    print(
        "Total deliveries:",
        int(
            size_counts[
                "deliveries"
            ].sum()
        ),
    )

    print()
    print(
        "=" * 64
    )
    print(
        "OVERALL RESULTS"
    )
    print(
        "=" * 64
    )

    columns = [
        "algorithm",
        "workloads",
        "deliveries",
        "mean_distance_km",
        "mean_travel_time_minutes",
        "mean_on_time_rate",
        "total_late_deliveries",
        "mean_training_runtime_seconds",
        "mean_model_size",
    ]

    print(
        overall[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 64
    )
    print(
        "PAIRWISE OVERALL"
    )
    print(
        "=" * 64
    )

    print(
        pairwise_overall.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 64
    )
    print(
        "KEY FINDINGS"
    )
    print(
        "=" * 64
    )

    print(
        findings
    )

    print()
    print(
        "=" * 64
    )
    print(
        "CREATED FILES"
    )
    print(
        "=" * 64
    )

    output_paths = [
        OVERALL_OUTPUT_PATH,
        BY_SIZE_OUTPUT_PATH,
        PAIRWISE_OVERALL_PATH,
        PAIRWISE_BY_SIZE_PATH,
        SIZE_COUNTS_PATH,
        KEY_FINDINGS_PATH,
        FIGURES_DIR,
    ]

    for path in output_paths:
        print(
            "-",
            path,
        )

    print()
    print(
        "Analysis complete."
    )


if __name__ == "__main__":
    main()