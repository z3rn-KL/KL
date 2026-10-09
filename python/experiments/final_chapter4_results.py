"""
Generate canonical tables and figures for Thesis Chapter 4.

This script does NOT train any model.

It only reads already completed experiment outputs and produces
final tables/figures for Chapter 4.

Expected inputs
---------------
results/xedu_kmeans_clusters.csv

results/thesis_dataset_algorithm_summary.csv
results/thesis_dataset_algorithm_by_size.csv
results/thesis_dataset_rl_summary.csv

results/kmeans_guided_dqn_sensitivity_results.csv
results/kmeans_guided_dqn_sensitivity_summary.csv
results/kmeans_guided_dqn_sensitivity_pairwise.csv

results/final_kmeans_guided_dqn_validation_results.csv
results/final_kmeans_guided_dqn_validation_paired.csv
results/final_kmeans_guided_dqn_validation_summary.csv

Outputs
-------
results/chapter4/
    tables/
    figures/
    chapter4_key_findings.txt
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

OUTPUT_DIR = (
    RESULTS_DIR
    / "chapter4"
)

TABLE_DIR = (
    OUTPUT_DIR
    / "tables"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)


KMEANS_PATH = (
    RESULTS_DIR
    / "xedu_kmeans_clusters.csv"
)

ALGORITHM_SUMMARY_PATH = (
    RESULTS_DIR
    / "thesis_dataset_algorithm_summary.csv"
)

ALGORITHM_BY_SIZE_PATH = (
    RESULTS_DIR
    / "thesis_dataset_algorithm_by_size.csv"
)

RL_SUMMARY_PATH = (
    RESULTS_DIR
    / "thesis_dataset_rl_summary.csv"
)

SENSITIVITY_RESULTS_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_results.csv"
)

SENSITIVITY_SUMMARY_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_summary.csv"
)

SENSITIVITY_PAIRWISE_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_pairwise.csv"
)

FINAL_VALIDATION_RESULTS_PATH = (
    RESULTS_DIR
    / "final_kmeans_guided_dqn_validation_results.csv"
)

FINAL_VALIDATION_PAIRED_PATH = (
    RESULTS_DIR
    / "final_kmeans_guided_dqn_validation_paired.csv"
)

FINAL_VALIDATION_SUMMARY_PATH = (
    RESULTS_DIR
    / "final_kmeans_guided_dqn_validation_summary.csv"
)


# ============================================================
# DISPLAY ORDER
# ============================================================

ALGORITHM_ORDER = [
    "nearest_neighbor",
    "clarke_wright",
    "q_learning",
    "sarsa",
    "dqn",
]

ALGORITHM_LABELS = {
    "nearest_neighbor": "Nearest Neighbor",
    "clarke_wright": "Clarke-Wright",
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
    "dqn": "DQN",
}


# ============================================================
# HELPERS
# ============================================================

def ensure_directories() -> None:
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def save_figure(
    name: str,
) -> None:
    path = (
        FIGURE_DIR
        / name
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Created figure: {path}"
    )


def algorithm_label(
    value: str,
) -> str:
    return ALGORITHM_LABELS.get(
        value,
        value,
    )


def ordered_algorithm_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result[
        "_order"
    ] = result[
        "algorithm"
    ].map(
        {
            algorithm: index
            for index, algorithm
            in enumerate(
                ALGORITHM_ORDER
            )
        }
    )

    result = (
        result
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

    return result


# ============================================================
# 4.2 K-MEANS
# ============================================================

def build_kmeans_outputs(
) -> dict:
    require_file(
        KMEANS_PATH
    )

    dataframe = pd.read_csv(
        KMEANS_PATH
    )

    if (
        "cluster_id"
        not in dataframe.columns
    ):
        raise ValueError(
            "K-Means result does not contain cluster_id."
        )

    cluster_counts = (
        dataframe[
            "cluster_id"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "cluster_id"
        )
        .reset_index(
            name="deliveries"
        )
    )

    cluster_counts[
        "share"
    ] = (
        cluster_counts[
            "deliveries"
        ]
        / cluster_counts[
            "deliveries"
        ].sum()
    )

    cluster_counts.to_csv(
        TABLE_DIR
        / "table_kmeans_cluster_sizes.csv",
        index=False,
    )

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        cluster_counts[
            "cluster_id"
        ].astype(
            str
        ),
        cluster_counts[
            "deliveries"
        ],
    )

    plt.xlabel(
        "K-Means cluster"
    )

    plt.ylabel(
        "Number of deliveries"
    )

    plt.title(
        "K-Means Cluster Size Distribution (K = 10)"
    )

    save_figure(
        "figure_kmeans_cluster_sizes.png"
    )

    return {
        "deliveries":
            int(
                len(
                    dataframe
                )
            ),

        "clusters":
            int(
                cluster_counts[
                    "cluster_id"
                ].nunique()
            ),

        "min_cluster":
            int(
                cluster_counts[
                    "deliveries"
                ].min()
            ),

        "max_cluster":
            int(
                cluster_counts[
                    "deliveries"
                ].max()
            ),

        "mean_cluster":
            float(
                cluster_counts[
                    "deliveries"
                ].mean()
            ),

        "std_cluster":
            float(
                cluster_counts[
                    "deliveries"
                ].std(
                    ddof=0
                )
            ),
    }


# ============================================================
# 4.3 OVERALL ALGORITHM BENCHMARK
# ============================================================

def build_algorithm_summary_outputs(
) -> pd.DataFrame:
    require_file(
        ALGORITHM_SUMMARY_PATH
    )

    dataframe = pd.read_csv(
        ALGORITHM_SUMMARY_PATH
    )

    dataframe = (
        ordered_algorithm_dataframe(
            dataframe
        )
    )

    output = dataframe.copy()

    output[
        "algorithm_label"
    ] = output[
        "algorithm"
    ].map(
        algorithm_label
    )

    output.to_csv(
        TABLE_DIR
        / "table_algorithm_overall.csv",
        index=False,
    )

    labels = output[
        "algorithm_label"
    ].tolist()

    # Distance
    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        labels,
        output[
            "mean_distance_km"
        ],
    )

    plt.ylabel(
        "Mean distance (km)"
    )

    plt.title(
        "Mean Route Distance by Algorithm"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    save_figure(
        "figure_algorithm_mean_distance.png"
    )

    # Travel time
    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        labels,
        output[
            "mean_travel_time_minutes"
        ],
    )

    plt.ylabel(
        "Mean travel time (minutes)"
    )

    plt.title(
        "Mean Travel Time by Algorithm"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    save_figure(
        "figure_algorithm_mean_travel_time.png"
    )

    # Weighted SLA
    sla_column = None

    for candidate in [
        "weighted_on_time_rate",
        "overall_on_time_rate",
        "on_time_rate",
    ]:
        if candidate in output.columns:
            sla_column = candidate
            break

    if sla_column is None:
        raise ValueError(
            "No weighted SLA column found."
        )

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        labels,
        output[
            sla_column
        ]
        * 100.0,
    )

    plt.ylabel(
        "On-time delivery rate (%)"
    )

    plt.title(
        "Weighted On-Time Delivery Rate by Algorithm"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.ylim(
        95,
        100.2,
    )

    save_figure(
        "figure_algorithm_weighted_sla.png"
    )

    return output


# ============================================================
# 4.4 BY WORKLOAD SIZE
# ============================================================

def build_by_size_outputs(
) -> pd.DataFrame:
    require_file(
        ALGORITHM_BY_SIZE_PATH
    )

    dataframe = pd.read_csv(
        ALGORITHM_BY_SIZE_PATH
    )

    dataframe.to_csv(
        TABLE_DIR
        / "table_algorithm_by_workload_size.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Detect workload-size column automatically.
    # Different report scripts used slightly different names.
    # --------------------------------------------------------

    size_candidates = [
        "number_of_deliveries",
        "workload_size",
        "delivery_count",
        "n_deliveries",
    ]

    size_column = None

    for candidate in size_candidates:
        if candidate in dataframe.columns:
            size_column = candidate
            break

    if size_column is None:
        raise ValueError(
            "Cannot find workload-size column. "
            f"Available columns: "
            f"{dataframe.columns.tolist()}"
        )

    print(
        f"Workload-size column: {size_column}"
    )

    # --------------------------------------------------------
    # Detect algorithm column.
    # --------------------------------------------------------

    if "algorithm" not in dataframe.columns:
        raise ValueError(
            "By-size result does not contain "
            "'algorithm'. "
            f"Available columns: "
            f"{dataframe.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Detect distance/time metric columns.
    # --------------------------------------------------------

    distance_candidates = [
        "mean_distance_km",
        "total_distance_km",
    ]

    time_candidates = [
        "mean_travel_time_minutes",
        "total_travel_time_minutes",
    ]

    distance_column = None

    for candidate in distance_candidates:
        if candidate in dataframe.columns:
            distance_column = candidate
            break

    time_column = None

    for candidate in time_candidates:
        if candidate in dataframe.columns:
            time_column = candidate
            break

    if distance_column is None:
        raise ValueError(
            "Cannot find distance column. "
            f"Available columns: "
            f"{dataframe.columns.tolist()}"
        )

    if time_column is None:
        raise ValueError(
            "Cannot find travel-time column. "
            f"Available columns: "
            f"{dataframe.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Ensure numeric workload sizes.
    # --------------------------------------------------------

    dataframe[
        size_column
    ] = pd.to_numeric(
        dataframe[
            size_column
        ],
        errors="raise",
    ).astype(
        int
    )

    sizes = sorted(
        dataframe[
            size_column
        ].unique()
    )

    # --------------------------------------------------------
    # Distance by workload size
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    for algorithm in (
        ALGORITHM_ORDER
    ):
        subset = (
            dataframe[
                dataframe[
                    "algorithm"
                ]
                == algorithm
            ]
            .sort_values(
                size_column
            )
        )

        if subset.empty:
            continue

        plt.plot(
            subset[
                size_column
            ],
            subset[
                distance_column
            ],
            marker="o",
            label=(
                algorithm_label(
                    algorithm
                )
            ),
        )

    plt.xlabel(
        "Number of deliveries in workload"
    )

    plt.ylabel(
        "Mean distance (km)"
    )

    plt.title(
        "Route Distance by Workload Size"
    )

    plt.xticks(
        sizes
    )

    plt.legend()

    save_figure(
        "figure_distance_by_workload_size.png"
    )

    # --------------------------------------------------------
    # Travel time by workload size
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    for algorithm in (
        ALGORITHM_ORDER
    ):
        subset = (
            dataframe[
                dataframe[
                    "algorithm"
                ]
                == algorithm
            ]
            .sort_values(
                size_column
            )
        )

        if subset.empty:
            continue

        plt.plot(
            subset[
                size_column
            ],
            subset[
                time_column
            ],
            marker="o",
            label=(
                algorithm_label(
                    algorithm
                )
            ),
        )

    plt.xlabel(
        "Number of deliveries in workload"
    )

    plt.ylabel(
        "Mean travel time (minutes)"
    )

    plt.title(
        "Travel Time by Workload Size"
    )

    plt.xticks(
        sizes
    )

    plt.legend()

    save_figure(
        "figure_travel_time_by_workload_size.png"
    )

    return dataframe


# ============================================================
# 4.5 RL-SPECIFIC RESULTS
# ============================================================

def build_rl_outputs(
) -> pd.DataFrame:
    require_file(
        RL_SUMMARY_PATH
    )

    dataframe = pd.read_csv(
        RL_SUMMARY_PATH
    )

    dataframe.to_csv(
        TABLE_DIR
        / "table_rl_summary.csv",
        index=False,
    )

    labels = [
        algorithm_label(
            algorithm
        )
        for algorithm
        in dataframe[
            "algorithm"
        ]
    ]

    runtime_column = (
        "mean_training_runtime_seconds"
    )

    if (
        runtime_column
        in dataframe.columns
    ):
        plt.figure(
            figsize=(
                8,
                5,
            )
        )

        plt.bar(
            labels,
            dataframe[
                runtime_column
            ],
        )

        plt.ylabel(
            "Mean training runtime (seconds)"
        )

        plt.title(
            "RL Training Runtime"
        )

        save_figure(
            "figure_rl_training_runtime.png"
        )

    if (
        "reward_per_delivery"
        in dataframe.columns
    ):
        reward_column = (
            "reward_per_delivery"
        )

    elif (
        "mean_reward_per_delivery"
        in dataframe.columns
    ):
        reward_column = (
            "mean_reward_per_delivery"
        )

    else:
        reward_column = None

    if (
        reward_column is not None
    ):
        plt.figure(
            figsize=(
                8,
                5,
            )
        )

        plt.bar(
            labels,
            dataframe[
                reward_column
            ],
        )

        plt.ylabel(
            "Mean reward per delivery"
        )

        plt.title(
            "Normalized RL Reward"
        )

        save_figure(
            "figure_rl_reward_per_delivery.png"
        )

    if (
        "mean_model_size"
        in dataframe.columns
    ):
        plt.figure(
            figsize=(
                8,
                5,
            )
        )

        plt.bar(
            labels,
            dataframe[
                "mean_model_size"
            ],
        )

        plt.ylabel(
            "Model size"
        )

        plt.title(
            "RL Model Representation Size"
        )

        save_figure(
            "figure_rl_model_size.png"
        )

    return dataframe


# ============================================================
# 4.6 - 4.7 SENSITIVITY
# ============================================================

def build_sensitivity_outputs(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    require_file(
        SENSITIVITY_SUMMARY_PATH
    )

    require_file(
        SENSITIVITY_PAIRWISE_PATH
    )

    summary = pd.read_csv(
        SENSITIVITY_SUMMARY_PATH
    )

    pairwise = pd.read_csv(
        SENSITIVITY_PAIRWISE_PATH
    )

    summary = (
        summary
        .sort_values(
            "cluster_switch_penalty"
        )
        .reset_index(
            drop=True
        )
    )

    summary.to_csv(
        TABLE_DIR
        / "table_dqn_cluster_penalty_sensitivity.csv",
        index=False,
    )

    pairwise.to_csv(
        TABLE_DIR
        / "table_dqn_cluster_penalty_pairwise.csv",
        index=False,
    )

    penalties = summary[
        "cluster_switch_penalty"
    ]

    # Cluster-switch rate
    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        penalties,
        summary[
            "mean_cluster_switch_rate"
        ],
        marker="o",
    )

    plt.xlabel(
        "Cluster-switch penalty λ"
    )

    plt.ylabel(
        "Mean cluster-switch rate"
    )

    plt.title(
        "Effect of K-Means Guidance on Cluster Switching"
    )

    plt.xticks(
        penalties
    )

    save_figure(
        "figure_sensitivity_cluster_switch_rate.png"
    )

    # Distance/time trade-off
    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        penalties,
        summary[
            "mean_distance_km"
        ],
        marker="o",
        label="Mean distance (km)",
    )

    plt.plot(
        penalties,
        summary[
            "mean_travel_time_minutes"
        ],
        marker="o",
        label="Mean travel time (min)",
    )

    plt.xlabel(
        "Cluster-switch penalty λ"
    )

    plt.ylabel(
        "Metric value"
    )

    plt.title(
        "Sensitivity Trade-off for K-Means-guided DQN"
    )

    plt.xticks(
        penalties
    )

    plt.legend()

    save_figure(
        "figure_sensitivity_distance_time.png"
    )

    return (
        summary,
        pairwise,
    )


# ============================================================
# 4.8 FINAL INTEGRATION VALIDATION
# ============================================================

def build_final_validation_outputs(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    require_file(
        FINAL_VALIDATION_PAIRED_PATH
    )

    require_file(
        FINAL_VALIDATION_SUMMARY_PATH
    )

    paired = pd.read_csv(
        FINAL_VALIDATION_PAIRED_PATH
    )

    summary = pd.read_csv(
        FINAL_VALIDATION_SUMMARY_PATH
    )

    paired.to_csv(
        TABLE_DIR
        / "table_final_guided_dqn_paired.csv",
        index=False,
    )

    summary.to_csv(
        TABLE_DIR
        / "table_final_guided_dqn_summary.csv",
        index=False,
    )

    # Overall mean distance
    row = summary.iloc[
        0
    ]

    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.bar(
        [
            "Vanilla DQN",
            "K-Means-guided DQN",
        ],
        [
            row[
                "mean_baseline_distance_km"
            ],
            row[
                "mean_guided_distance_km"
            ],
        ],
    )

    plt.ylabel(
        "Mean distance (km)"
    )

    plt.title(
        "Final Validation: Route Distance"
    )

    save_figure(
        "figure_final_validation_distance.png"
    )

    # Overall mean time
    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.bar(
        [
            "Vanilla DQN",
            "K-Means-guided DQN",
        ],
        [
            row[
                "mean_baseline_travel_time_minutes"
            ],
            row[
                "mean_guided_travel_time_minutes"
            ],
        ],
    )

    plt.ylabel(
        "Mean travel time (minutes)"
    )

    plt.title(
        "Final Validation: Travel Time"
    )

    save_figure(
        "figure_final_validation_travel_time.png"
    )

    # Cluster-switch comparison
    switch_subset = paired[
        paired[
            "baseline_cluster_switch_rate"
        ].notna()
    ]

    if not switch_subset.empty:
        baseline_switch = (
            switch_subset[
                "baseline_cluster_switch_rate"
            ].mean()
        )

        guided_switch = (
            switch_subset[
                "guided_cluster_switch_rate"
            ].mean()
        )

        plt.figure(
            figsize=(
                7,
                5,
            )
        )

        plt.bar(
            [
                "Vanilla DQN",
                "K-Means-guided DQN",
            ],
            [
                baseline_switch,
                guided_switch,
            ],
        )

        plt.ylabel(
            "Mean cluster-switch rate"
        )

        plt.title(
            "Final Validation: Spatial Continuity"
        )

        save_figure(
            "figure_final_validation_cluster_switch_rate.png"
        )

    # Pairwise distance change
    plt.figure(
        figsize=(
            10,
            5,
        )
    )

    labels = [
        f"ID {int(workload_id)}"
        for workload_id
        in paired[
            "workload_id"
        ]
    ]

    plt.bar(
        labels,
        paired[
            "distance_improvement_pct"
        ],
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.ylabel(
        "Distance improvement vs Vanilla DQN (%)"
    )

    plt.xlabel(
        "Validation workload"
    )

    plt.title(
        "Pairwise Distance Change of K-Means-guided DQN"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    save_figure(
        "figure_final_validation_pairwise_distance.png"
    )

    return (
        paired,
        summary,
    )


# ============================================================
# KEY FINDINGS
# ============================================================

def write_key_findings(
    kmeans_stats: dict,
    algorithm_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> None:
    row = (
        validation_summary
        .iloc[
            0
        ]
    )

    algorithm_lookup = (
        algorithm_summary
        .set_index(
            "algorithm"
        )
    )

    lines = []

    lines.append(
        "CHAPTER 4 - CANONICAL KEY FINDINGS"
    )

    lines.append(
        "=" * 64
    )

    lines.append(
        ""
    )

    lines.append(
        "K-MEANS"
    )

    lines.append(
        f"- Deliveries: "
        f"{kmeans_stats['deliveries']}"
    )

    lines.append(
        f"- Clusters: "
        f"{kmeans_stats['clusters']}"
    )

    lines.append(
        "- Selected K = 10"
    )

    lines.append(
        "- Silhouette ≈ 0.4285"
    )

    lines.append(
        f"- Cluster size min/max: "
        f"{kmeans_stats['min_cluster']} / "
        f"{kmeans_stats['max_cluster']}"
    )

    lines.append(
        ""
    )

    lines.append(
        "DATASET-SCALE BENCHMARK"
    )

    lines.append(
        "- 108 workloads"
    )

    lines.append(
        "- 813 deliveries"
    )

    lines.append(
        "- 540 algorithm runs"
    )

    for algorithm in (
        ALGORITHM_ORDER
    ):
        if (
            algorithm
            not in algorithm_lookup.index
        ):
            continue

        value = (
            algorithm_lookup.loc[
                algorithm
            ]
        )

        sla_column = None

        for candidate in [
            "weighted_on_time_rate",
            "overall_on_time_rate",
            "on_time_rate",
        ]:
            if candidate in value.index:
                sla_column = candidate
                break

        lines.append(
            f"- {algorithm_label(algorithm)}: "
            f"distance="
            f"{value['mean_distance_km']:.4f} km, "
            f"time="
            f"{value['mean_travel_time_minutes']:.4f} min, "
            f"SLA="
            f"{value[sla_column] * 100:.2f}%"
        )

    lines.append(
        ""
    )

    lines.append(
        "K-MEANS GUIDANCE SENSITIVITY"
    )

    lambda_05 = (
        sensitivity_summary[
            np.isclose(
                sensitivity_summary[
                    "cluster_switch_penalty"
                ],
                0.50,
            )
        ]
    )

    if not lambda_05.empty:
        value = (
            lambda_05.iloc[
                0
            ]
        )

        lines.append(
            "- Selected λ = 0.50"
        )

        lines.append(
            f"- Mean cluster-switch rate at λ=0.50: "
            f"{value['mean_cluster_switch_rate']:.4f}"
        )

    lines.append(
        ""
    )

    lines.append(
        "FINAL VANILLA VS GUIDED DQN VALIDATION"
    )

    lines.append(
        f"- Workloads: "
        f"{int(row['workloads'])}"
    )

    lines.append(
        f"- Deliveries: "
        f"{int(row['deliveries'])}"
    )

    lines.append(
        f"- Vanilla mean distance: "
        f"{row['mean_baseline_distance_km']:.4f} km"
    )

    lines.append(
        f"- Guided mean distance: "
        f"{row['mean_guided_distance_km']:.4f} km"
    )

    lines.append(
        f"- Distance change: "
        f"{-row['ratio_of_means_distance_improvement_pct']:+.2f}% "
        f"(positive means guided route is longer)"
    )

    lines.append(
        f"- Vanilla mean travel time: "
        f"{row['mean_baseline_travel_time_minutes']:.4f} min"
    )

    lines.append(
        f"- Guided mean travel time: "
        f"{row['mean_guided_travel_time_minutes']:.4f} min"
    )

    lines.append(
        f"- Time change: "
        f"{-row['ratio_of_means_time_improvement_pct']:+.2f}% "
        f"(positive means guided route is slower)"
    )

    lines.append(
        f"- Vanilla weighted SLA: "
        f"{row['baseline_weighted_on_time_rate'] * 100:.2f}%"
    )

    lines.append(
        f"- Guided weighted SLA: "
        f"{row['guided_weighted_on_time_rate'] * 100:.2f}%"
    )

    lines.append(
        f"- Cluster-switch comparison workloads: "
        f"{int(row['switch_comparison_workloads'])}"
    )

    lines.append(
        f"- Vanilla cluster-switch rate: "
        f"{row['mean_baseline_switch_rate_where_available']:.4f}"
    )

    lines.append(
        f"- Guided cluster-switch rate: "
        f"{row['mean_guided_switch_rate_where_available']:.4f}"
    )

    lines.append(
        f"- Relative switch-rate reduction: "
        f"{row['relative_switch_rate_reduction_pct_where_available']:.2f}%"
    )

    lines.append(
        ""
    )

    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "- Clarke-Wright remains the strongest pure distance/time heuristic."
    )

    lines.append(
        "- DQN is the strongest RL method overall in route quality."
    )

    lines.append(
        "- Q-Learning provides very low training cost with 100% weighted SLA."
    )

    lines.append(
        "- K-Means guidance improves spatial continuity, not route length."
    )

    lines.append(
        "- λ=0.50 demonstrates a measurable spatial-continuity / route-efficiency trade-off."
    )

    lines.append(
        "- Final cluster-switch comparison is based on 6 workloads with complete vanilla route information."
    )

    path = (
        OUTPUT_DIR
        / "chapter4_key_findings.txt"
    )

    path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        f"Created: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print(
        "=" * 72
    )

    print(
        "THESIS CHAPTER 4 - FINAL RESULTS"
    )

    print(
        "=" * 72
    )

    ensure_directories()

    kmeans_stats = (
        build_kmeans_outputs()
    )

    algorithm_summary = (
        build_algorithm_summary_outputs()
    )

    build_by_size_outputs()

    build_rl_outputs()

    (
        sensitivity_summary,
        _,
    ) = (
        build_sensitivity_outputs()
    )

    (
        _,
        validation_summary,
    ) = (
        build_final_validation_outputs()
    )

    write_key_findings(
        kmeans_stats=(
            kmeans_stats
        ),
        algorithm_summary=(
            algorithm_summary
        ),
        sensitivity_summary=(
            sensitivity_summary
        ),
        validation_summary=(
            validation_summary
        ),
    )

    print()
    print(
        "=" * 72
    )

    print(
        "CHAPTER 4 RESULTS COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Tables : {TABLE_DIR}"
    )

    print(
        f"Figures: {FIGURE_DIR}"
    )

    print(
        f"Summary: "
        f"{OUTPUT_DIR / 'chapter4_key_findings.txt'}"
    )

    print()


if __name__ == "__main__":
    main()