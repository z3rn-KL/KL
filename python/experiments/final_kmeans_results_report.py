from pathlib import Path

import matplotlib.pyplot as plt
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

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "thesis_kmeans_elbow_summary.csv"
)

INERTIA_FIGURE_PATH = (
    FIGURES_DIR
    / "kmeans_final_inertia.png"
)

SILHOUETTE_FIGURE_PATH = (
    FIGURES_DIR
    / "kmeans_final_silhouette.png"
)

BALANCE_FIGURE_PATH = (
    FIGURES_DIR
    / "kmeans_final_cluster_balance.png"
)


EXPECTED_K_VALUES = [
    5,
    8,
    10,
    12,
    15,
    20,
    25,
    30,
]


COLUMN_ALIASES = {
    "k": [
        "k",
        "n_clusters",
        "clusters",
        "number_of_clusters",
    ],

    "inertia": [
        "inertia",
        "wcss",
    ],

    "silhouette_score": [
        "silhouette_score",
        "silhouette",
    ],

    "min_cluster_size": [
        "min_cluster_size",
        "cluster_size_min",
        "min_size",
    ],

    "max_cluster_size": [
        "max_cluster_size",
        "cluster_size_max",
        "max_size",
    ],

    "mean_cluster_size": [
        "mean_cluster_size",
        "cluster_size_mean",
        "mean_size",
    ],

    "std_cluster_size": [
        "std_cluster_size",
        "cluster_size_std",
        "std_size",
    ],

    "imbalance_ratio": [
        "imbalance_ratio",
        "imbalance",
        "cluster_imbalance",
    ],

    "elbow_score": [
        "elbow_score",
        "elbow",
    ],
}


def find_kmeans_result_file() -> Path:
    """
    Find the existing K-Means experiment CSV.

    The project already contains K-Means
    results from previous experiments, so
    this report should reuse them instead
    of rerunning clustering.
    """

    candidates = []

    patterns = [
        "kmeans*results*.csv",
        "kmeans*.csv",
    ]

    for pattern in patterns:
        candidates.extend(
            RESULTS_DIR.glob(
                pattern
            )
        )

    candidates = [
        path
        for path in candidates
        if path.name
        != OUTPUT_PATH.name
    ]

    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy CSV kết quả "
            "K-Means trong results/."
        )

    candidates = sorted(
        set(
            candidates
        ),
        key=lambda path: (
            0
            if "experiment" in path.name.lower()
            else 1,
            path.name,
        ),
    )

    print()
    print(
        "K-Means result candidates:"
    )

    for candidate in candidates:
        print(
            f"  {candidate.name}"
        )

    selected = candidates[
        0
    ]

    print()
    print(
        f"Using: "
        f"{selected}"
    )

    return selected


def resolve_column(
    dataframe: pd.DataFrame,
    logical_name: str,
    required: bool = True,
):
    aliases = (
        COLUMN_ALIASES[
            logical_name
        ]
    )

    lookup = {
        str(column).lower():
            column
        for column
        in dataframe.columns
    }

    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[
                alias.lower()
            ]

    if required:
        raise ValueError(
            f"Không tìm thấy cột cho "
            f"'{logical_name}'. "
            f"Columns hiện có: "
            f"{list(dataframe.columns)}"
        )

    return None


def normalize_results(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    k_column = resolve_column(
        raw_df,
        "k",
    )

    inertia_column = resolve_column(
        raw_df,
        "inertia",
    )

    silhouette_column = resolve_column(
        raw_df,
        "silhouette_score",
    )

    min_column = resolve_column(
        raw_df,
        "min_cluster_size",
        required=False,
    )

    max_column = resolve_column(
        raw_df,
        "max_cluster_size",
        required=False,
    )

    mean_column = resolve_column(
        raw_df,
        "mean_cluster_size",
        required=False,
    )

    std_column = resolve_column(
        raw_df,
        "std_cluster_size",
        required=False,
    )

    imbalance_column = resolve_column(
        raw_df,
        "imbalance_ratio",
        required=False,
    )

    elbow_column = resolve_column(
        raw_df,
        "elbow_score",
        required=False,
    )

    dataframe = pd.DataFrame(
        {
            "k":
                raw_df[
                    k_column
                ].astype(
                    int
                ),

            "inertia":
                raw_df[
                    inertia_column
                ].astype(
                    float
                ),

            "silhouette_score":
                raw_df[
                    silhouette_column
                ].astype(
                    float
                ),
        }
    )

    optional_columns = [
        (
            "min_cluster_size",
            min_column,
        ),
        (
            "max_cluster_size",
            max_column,
        ),
        (
            "mean_cluster_size",
            mean_column,
        ),
        (
            "std_cluster_size",
            std_column,
        ),
        (
            "imbalance_ratio",
            imbalance_column,
        ),
        (
            "elbow_score",
            elbow_column,
        ),
    ]

    for (
        output_name,
        source_column,
    ) in optional_columns:
        if source_column is None:
            dataframe[
                output_name
            ] = None
        else:
            dataframe[
                output_name
            ] = raw_df[
                source_column
            ]

    dataframe = (
        dataframe[
            dataframe[
                "k"
            ].isin(
                EXPECTED_K_VALUES
            )
        ]
        .sort_values(
            "k"
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


def add_interpretation_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        dataframe.copy()
    )

    best_silhouette = (
        result[
            "silhouette_score"
        ]
        .max()
    )

    result[
        "is_best_silhouette"
    ] = (
        result[
            "silhouette_score"
        ]
        == best_silhouette
    )

    result[
        "silhouette_rank"
    ] = (
        result[
            "silhouette_score"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(
            int
        )
    )

    result[
        "inertia_reduction_pct"
    ] = (
        result[
            "inertia"
        ]
        .pct_change()
        .mul(
            -100.0
        )
    )

    result[
        "thesis_role"
    ] = ""

    result.loc[
        result[
            "k"
        ]
        == 10,
        "thesis_role",
    ] = (
        "Best silhouette candidate"
    )

    result.loc[
        result[
            "k"
        ]
        == 12,
        "thesis_role",
    ] = (
        "Elbow / balance candidate"
    )

    result.loc[
        result[
            "k"
        ]
        >= 25,
        "thesis_role",
    ] = (
        "High-K fragmentation analysis"
    )

    return result


def plot_inertia(
    dataframe: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    axis.plot(
        dataframe[
            "k"
        ],
        dataframe[
            "inertia"
        ],
        marker="o",
    )

    axis.axvline(
        12,
        linestyle="--",
        alpha=0.6,
    )

    axis.set_title(
        "K-Means Elbow Method"
    )

    axis.set_xlabel(
        "Number of clusters (K)"
    )

    axis.set_ylabel(
        "Inertia / WCSS"
    )

    axis.grid(
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        INERTIA_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_silhouette(
    dataframe: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    axis.plot(
        dataframe[
            "k"
        ],
        dataframe[
            "silhouette_score"
        ],
        marker="o",
    )

    axis.axvline(
        10,
        linestyle="--",
        alpha=0.6,
    )

    axis.set_title(
        "K-Means Silhouette Score"
    )

    axis.set_xlabel(
        "Number of clusters (K)"
    )

    axis.set_ylabel(
        "Silhouette score"
    )

    axis.grid(
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        SILHOUETTE_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_cluster_balance(
    dataframe: pd.DataFrame,
) -> None:
    usable = (
        dataframe[
            dataframe[
                "min_cluster_size"
            ]
            .notna()
            &
            dataframe[
                "max_cluster_size"
            ]
            .notna()
        ]
        .copy()
    )

    if usable.empty:
        print()
        print(
            "Cluster-size columns not found; "
            "balance figure skipped."
        )

        return

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    axis.plot(
        usable[
            "k"
        ],
        usable[
            "max_cluster_size"
        ],
        marker="o",
        label="Maximum cluster size",
    )

    axis.plot(
        usable[
            "k"
        ],
        usable[
            "min_cluster_size"
        ],
        marker="o",
        label="Minimum cluster size",
    )

    if (
        usable[
            "mean_cluster_size"
        ]
        .notna()
        .any()
    ):
        axis.plot(
            usable[
                "k"
            ],
            usable[
                "mean_cluster_size"
            ],
            marker="o",
            label="Mean cluster size",
        )

    axis.set_title(
        "K-Means Cluster Size Balance"
    )

    axis.set_xlabel(
        "Number of clusters (K)"
    )

    axis.set_ylabel(
        "Deliveries per cluster"
    )

    axis.legend()

    axis.grid(
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        BALANCE_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def print_summary(
    dataframe: pd.DataFrame,
) -> None:
    columns = [
        "k",
        "inertia",
        "silhouette_score",
        "min_cluster_size",
        "max_cluster_size",
        "mean_cluster_size",
        "std_cluster_size",
        "imbalance_ratio",
        "silhouette_rank",
        "thesis_role",
    ]

    existing_columns = [
        column
        for column
        in columns
        if column
        in dataframe.columns
    ]

    print()
    print(
        "========================================"
    )

    print(
        "THESIS K-MEANS / ELBOW SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        dataframe[
            existing_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
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

    print()
    print(
        "========================================"
    )

    print(
        "KEY FINDINGS"
    )

    print(
        "========================================"
    )

    print(
        f"Best silhouette K: "
        f"{int(best_row['k'])}"
    )

    print(
        f"Best silhouette score: "
        f"{best_row['silhouette_score']:.4f}"
    )

    if (
        12
        in dataframe[
            "k"
        ].values
    ):
        k12 = (
            dataframe[
                dataframe[
                    "k"
                ]
                == 12
            ]
            .iloc[
                0
            ]
        )

        print(
            f"K=12 silhouette: "
            f"{k12['silhouette_score']:.4f}"
        )

    print(
        "Elbow interpretation: "
        "around K=12"
    )

    print(
        "High-K interpretation: "
        "K=25/30 used to inspect "
        "cluster fragmentation."
    )


def main():
    print()
    print(
        "========================================"
    )

    print(
        "Final K-Means / Elbow Thesis Report"
    )

    print(
        "========================================"
    )

    source_path = (
        find_kmeans_result_file()
    )

    raw_df = pd.read_csv(
        source_path
    )

    print(
        f"Rows loaded: "
        f"{len(raw_df)}"
    )

    print(
        f"Columns: "
        f"{list(raw_df.columns)}"
    )

    dataframe = (
        normalize_results(
            raw_df
        )
    )

    dataframe = (
        add_interpretation_columns(
            dataframe
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    plot_inertia(
        dataframe
    )

    plot_silhouette(
        dataframe
    )

    plot_cluster_balance(
        dataframe
    )

    print_summary(
        dataframe
    )

    print()
    print(
        "========================================"
    )

    print(
        "FILES CREATED"
    )

    print(
        "========================================"
    )

    print(
        OUTPUT_PATH
    )

    print(
        INERTIA_FIGURE_PATH
    )

    print(
        SILHOUETTE_FIGURE_PATH
    )

    if BALANCE_FIGURE_PATH.exists():
        print(
            BALANCE_FIGURE_PATH
        )


if __name__ == "__main__":
    main()