from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


KMEANS_INPUT_PATH = (
    RESULTS_DIR
    / "thesis_kmeans_elbow_summary.csv"
)

CORE_INPUT_PATH = (
    RESULTS_DIR
    / "thesis_core_algorithm_summary.csv"
)

RL_INPUT_PATH = (
    RESULTS_DIR
    / "thesis_rl_algorithm_summary.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "thesis_experiment_summary.csv"
)


ELBOW_REFERENCE_K = 12

OPERATIONAL_K_RANGE = "10-12"


def load_inputs():
    kmeans_df = pd.read_csv(
        KMEANS_INPUT_PATH
    )

    core_df = pd.read_csv(
        CORE_INPUT_PATH
    )

    rl_df = pd.read_csv(
        RL_INPUT_PATH
    )

    return (
        kmeans_df,
        core_df,
        rl_df,
    )


def add_numeric_row(
    rows: list[dict],
    section: str,
    experiment_group: str,
    algorithm: str,
    configuration: str,
    metric: str,
    value,
    unit: str,
    note: str,
    source_file: str,
) -> None:
    if pd.isna(
        value
    ):
        return

    rows.append(
        {
            "section":
                section,

            "experiment_group":
                experiment_group,

            "algorithm":
                algorithm,

            "configuration":
                configuration,

            "metric":
                metric,

            "value_numeric":
                float(
                    value
                ),

            "value_text":
                "",

            "unit":
                unit,

            "note":
                note,

            "source_file":
                source_file,
        }
    )


def add_text_row(
    rows: list[dict],
    section: str,
    experiment_group: str,
    algorithm: str,
    configuration: str,
    metric: str,
    value: str,
    note: str,
    source_file: str,
) -> None:
    rows.append(
        {
            "section":
                section,

            "experiment_group":
                experiment_group,

            "algorithm":
                algorithm,

            "configuration":
                configuration,

            "metric":
                metric,

            "value_numeric":
                None,

            "value_text":
                value,

            "unit":
                "",

            "note":
                note,

            "source_file":
                source_file,
        }
    )


def build_kmeans_rows(
    dataframe: pd.DataFrame,
    rows: list[dict],
) -> None:
    source_file = (
        KMEANS_INPUT_PATH.name
    )

    numeric_metrics = {
        "inertia":
            (
                "Inertia / WCSS",
                "",
            ),

        "silhouette_score":
            (
                "Silhouette score",
                "",
            ),

        "min_cluster_size":
            (
                "Minimum cluster size",
                "deliveries",
            ),

        "max_cluster_size":
            (
                "Maximum cluster size",
                "deliveries",
            ),

        "mean_cluster_size":
            (
                "Mean cluster size",
                "deliveries",
            ),

        "std_cluster_size":
            (
                "Cluster-size standard deviation",
                "deliveries",
            ),

        "imbalance_ratio":
            (
                "Cluster imbalance ratio",
                "",
            ),

        "silhouette_rank":
            (
                "Silhouette rank",
                "",
            ),
    }

    for row in dataframe.itertuples():
        k = int(
            row.k
        )

        configuration = (
            f"K={k}"
        )

        for (
            column,
            (
                metric_name,
                unit,
            ),
        ) in numeric_metrics.items():
            if not hasattr(
                row,
                column
            ):
                continue

            value = getattr(
                row,
                column
            )

            add_numeric_row(
                rows=rows,
                section="clustering",
                experiment_group=(
                    "kmeans_grid"
                ),
                algorithm="K-Means",
                configuration=(
                    configuration
                ),
                metric=metric_name,
                value=value,
                unit=unit,
                note="",
                source_file=(
                    source_file
                ),
            )

    best_row = (
        dataframe
        .sort_values(
            "silhouette_score",
            ascending=False,
        )
        .iloc[
            0
        ]
    )

    add_numeric_row(
        rows=rows,
        section="clustering",
        experiment_group=(
            "kmeans_key_findings"
        ),
        algorithm="K-Means",
        configuration="selected",
        metric="Best silhouette K",
        value=best_row[
            "k"
        ],
        unit="clusters",
        note=(
            "K with the highest "
            "Silhouette Score."
        ),
        source_file=source_file,
    )

    add_numeric_row(
        rows=rows,
        section="clustering",
        experiment_group=(
            "kmeans_key_findings"
        ),
        algorithm="K-Means",
        configuration="selected",
        metric="Best silhouette score",
        value=best_row[
            "silhouette_score"
        ],
        unit="",
        note=(
            "Highest observed "
            "Silhouette Score."
        ),
        source_file=source_file,
    )

    add_numeric_row(
        rows=rows,
        section="clustering",
        experiment_group=(
            "kmeans_key_findings"
        ),
        algorithm="K-Means",
        configuration="selected",
        metric="Elbow reference K",
        value=ELBOW_REFERENCE_K,
        unit="clusters",
        note=(
            "Elbow region used in the "
            "thesis interpretation."
        ),
        source_file=source_file,
    )

    add_text_row(
        rows=rows,
        section="clustering",
        experiment_group=(
            "kmeans_key_findings"
        ),
        algorithm="K-Means",
        configuration="selected",
        metric="Operational K range",
        value=(
            OPERATIONAL_K_RANGE
        ),
        note=(
            "Range retained as a practical "
            "balance between separation, "
            "inertia reduction and cluster "
            "fragmentation."
        ),
        source_file=source_file,
    )


def build_core_algorithm_rows(
    dataframe: pd.DataFrame,
    rows: list[dict],
) -> None:
    source_file = (
        CORE_INPUT_PATH.name
    )

    metrics = {
        "workloads":
            (
                "Benchmark workloads",
                "workloads",
            ),

        "deliveries":
            (
                "Benchmark deliveries",
                "deliveries",
            ),

        "mean_distance_km":
            (
                "Mean route distance",
                "km",
            ),

        "std_distance_km":
            (
                "Route distance standard deviation",
                "km",
            ),

        "median_distance_km":
            (
                "Median route distance",
                "km",
            ),

        "mean_travel_time_minutes":
            (
                "Mean travel time",
                "minutes",
            ),

        "std_travel_time_minutes":
            (
                "Travel-time standard deviation",
                "minutes",
            ),

        "mean_on_time_rate":
            (
                "Mean on-time rate",
                "ratio",
            ),

        "min_on_time_rate":
            (
                "Minimum on-time rate",
                "ratio",
            ),

        "total_late_deliveries":
            (
                "Total late deliveries",
                "deliveries",
            ),

        "total_lateness_minutes":
            (
                "Total lateness",
                "minutes",
            ),

        "max_lateness_minutes":
            (
                "Maximum lateness",
                "minutes",
            ),

        "distance_reduction_vs_nn_pct":
            (
                "Distance reduction vs NN",
                "percent",
            ),

        "travel_time_reduction_vs_nn_pct":
            (
                "Travel-time reduction vs NN",
                "percent",
            ),

        "mean_route_runtime_seconds":
            (
                "Mean route runtime",
                "seconds",
            ),

        "mean_matrix_preparation_seconds":
            (
                "Mean road-matrix preparation time",
                "seconds",
            ),

        "mean_training_runtime_seconds":
            (
                "Mean training runtime",
                "seconds",
            ),

        "total_training_runtime_seconds":
            (
                "Total training runtime",
                "seconds",
            ),

        "mean_greedy_reward":
            (
                "Mean greedy reward",
                "",
            ),

        "std_greedy_reward":
            (
                "Greedy reward standard deviation",
                "",
            ),

        "mean_model_size":
            (
                "Mean model size",
                "entries_or_parameters",
            ),
    }

    for row in dataframe.itertuples():
        algorithm = (
            row.display_name
            if hasattr(
                row,
                "display_name"
            )
            else row.algorithm
        )

        for (
            column,
            (
                metric_name,
                unit,
            ),
        ) in metrics.items():
            if not hasattr(
                row,
                column
            ):
                continue

            value = getattr(
                row,
                column
            )

            note = ""

            if column == (
                "mean_route_runtime_seconds"
            ):
                note = (
                    "Runtime scope differs "
                    "between heuristics and RL; "
                    "do not compare directly "
                    "without considering road "
                    "matrix preparation and "
                    "training."
                )

            add_numeric_row(
                rows=rows,
                section="routing",
                experiment_group=(
                    "core_algorithm_benchmark"
                ),
                algorithm=str(
                    algorithm
                ),
                configuration=(
                    "31 exact-5-delivery waves"
                ),
                metric=metric_name,
                value=value,
                unit=unit,
                note=note,
                source_file=(
                    source_file
                ),
            )


def build_rl_rows(
    dataframe: pd.DataFrame,
    rows: list[dict],
) -> None:
    source_file = (
        RL_INPUT_PATH.name
    )

    metrics = {
        "mean_distance_km":
            (
                "Mean route distance",
                "km",
            ),

        "mean_travel_time_minutes":
            (
                "Mean travel time",
                "minutes",
            ),

        "mean_greedy_reward":
            (
                "Mean greedy reward",
                "",
            ),

        "mean_training_runtime_seconds":
            (
                "Mean training runtime",
                "seconds",
            ),

        "training_runtime_vs_q_learning":
            (
                "Training runtime relative to Q-Learning",
                "ratio",
            ),

        "mean_model_size":
            (
                "Mean model size",
                "entries_or_parameters",
            ),
    }

    for row in dataframe.itertuples():
        algorithm = (
            row.display_name
            if hasattr(
                row,
                "display_name"
            )
            else row.algorithm
        )

        for (
            column,
            (
                metric_name,
                unit,
            ),
        ) in metrics.items():
            if not hasattr(
                row,
                column
            ):
                continue

            value = getattr(
                row,
                column
            )

            note = ""

            if column == (
                "mean_model_size"
            ):
                note = (
                    "Q-Learning/SARSA model size "
                    "is Q-table entries; DQN model "
                    "size is neural-network "
                    "parameters. Values are not "
                    "directly equivalent."
                )

            add_numeric_row(
                rows=rows,
                section="reinforcement_learning",
                experiment_group=(
                    "rl_algorithm_comparison"
                ),
                algorithm=str(
                    algorithm
                ),
                configuration=(
                    "5000 episodes, 31 workloads"
                ),
                metric=metric_name,
                value=value,
                unit=unit,
                note=note,
                source_file=(
                    source_file
                ),
            )


def main():
    print()
    print(
        "========================================"
    )

    print(
        "Building Thesis Experiment Summary"
    )

    print(
        "========================================"
    )

    (
        kmeans_df,
        core_df,
        rl_df,
    ) = load_inputs()

    rows = []

    build_kmeans_rows(
        dataframe=kmeans_df,
        rows=rows,
    )

    build_core_algorithm_rows(
        dataframe=core_df,
        rows=rows,
    )

    build_rl_rows(
        dataframe=rl_df,
        rows=rows,
    )

    summary_df = pd.DataFrame(
        rows
    )

    section_order = {
        "clustering": 0,
        "routing": 1,
        "reinforcement_learning": 2,
    }

    summary_df[
        "_section_order"
    ] = (
        summary_df[
            "section"
        ]
        .map(
            section_order
        )
    )

    summary_df = (
        summary_df
        .sort_values(
            by=[
                "_section_order",
                "experiment_group",
                "algorithm",
                "configuration",
                "metric",
            ]
        )
        .drop(
            columns=[
                "_section_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Rows created: "
        f"{len(summary_df)}"
    )

    print()

    print(
        "Sections:"
    )

    print(
        summary_df[
            "section"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Algorithms:"
    )

    print(
        summary_df[
            "algorithm"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "========================================"
    )

    print(
        "KEY THESIS VALUES"
    )

    print(
        "========================================"
    )

    key_metrics = [
        "Best silhouette K",
        "Best silhouette score",
        "Elbow reference K",
        "Operational K range",
        "Mean route distance",
        "Mean travel time",
        "Mean training runtime",
        "Mean greedy reward",
    ]

    key_df = (
        summary_df[
            summary_df[
                "metric"
            ]
            .isin(
                key_metrics
            )
        ]
        .copy()
    )

    print(
        key_df[
            [
                "section",
                "algorithm",
                "configuration",
                "metric",
                "value_numeric",
                "value_text",
                "unit",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        "FILE CREATED"
    )

    print(
        "========================================"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()