from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

WORKLOAD_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_workloads.csv"
)

CLUSTER_PATH = (
    RESULTS_DIR
    / "xedu_kmeans_clusters.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "routing_workload_cluster_analysis.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "routing_workload_cluster_summary.csv"
)


def load_workloads() -> pd.DataFrame:
    dataframe = pd.read_csv(
        WORKLOAD_PATH
    )

    required = [
        "workload_id",
        "number_of_deliveries",
        "delivery_ids",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Workload CSV thiếu: "
            + ", ".join(missing)
        )

    return dataframe


def load_clusters() -> pd.DataFrame:
    dataframe = pd.read_csv(
        CLUSTER_PATH
    )

    if "id" not in dataframe.columns:
        raise ValueError(
            "K-Means CSV không có cột id."
        )

    if "cluster_id" not in dataframe.columns:
        raise ValueError(
            "K-Means CSV không có cluster_id."
        )

    result = (
        dataframe[
            [
                "id",
                "cluster_id",
            ]
        ]
        .rename(
            columns={
                "id": "delivery_id"
            }
        )
        .copy()
    )

    result[
        "delivery_id"
    ] = (
        result[
            "delivery_id"
        ]
        .astype(str)
        .str.strip()
    )

    result[
        "cluster_id"
    ] = pd.to_numeric(
        result[
            "cluster_id"
        ],
        errors="raise",
    ).astype(int)

    return result


def parse_delivery_ids(
    value: str,
) -> list[str]:
    return [
        delivery_id.strip()
        for delivery_id
        in str(value).split("->")
        if delivery_id.strip()
    ]


def analyze_workloads(
    workloads: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    cluster_lookup = dict(
        zip(
            clusters[
                "delivery_id"
            ],
            clusters[
                "cluster_id"
            ],
        )
    )

    rows = []

    for _, workload in (
        workloads.iterrows()
    ):
        delivery_ids = (
            parse_delivery_ids(
                workload[
                    "delivery_ids"
                ]
            )
        )

        cluster_ids = []

        missing_ids = []

        for delivery_id in delivery_ids:
            if delivery_id not in cluster_lookup:
                missing_ids.append(
                    delivery_id
                )
                continue

            cluster_ids.append(
                cluster_lookup[
                    delivery_id
                ]
            )

        if missing_ids:
            raise ValueError(
                "Không tìm thấy cluster cho "
                f"workload {workload['workload_id']}: "
                f"{missing_ids[:5]}"
            )

        counts = (
            pd.Series(
                cluster_ids
            )
            .value_counts()
            .sort_index()
        )

        number_of_clusters = len(
            counts
        )

        dominant_cluster_size = int(
            counts.max()
        )

        number_of_deliveries = len(
            delivery_ids
        )

        dominant_cluster_share = (
            dominant_cluster_size
            / number_of_deliveries
        )

        rows.append(
            {
                "workload_id":
                    workload[
                        "workload_id"
                    ],

                "number_of_deliveries":
                    number_of_deliveries,

                "number_of_clusters":
                    number_of_clusters,

                "dominant_cluster_id":
                    int(
                        counts.idxmax()
                    ),

                "dominant_cluster_size":
                    dominant_cluster_size,

                "dominant_cluster_share":
                    dominant_cluster_share,

                "cluster_distribution":
                    "; ".join(
                        f"{int(cluster_id)}:{int(count)}"
                        for (
                            cluster_id,
                            count,
                        )
                        in counts.items()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_summary(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    return (
        analysis
        .groupby(
            "number_of_clusters",
            as_index=False,
        )
        .agg(
            workloads=(
                "workload_id",
                "count",
            ),

            mean_workload_size=(
                "number_of_deliveries",
                "mean",
            ),

            mean_dominant_cluster_share=(
                "dominant_cluster_share",
                "mean",
            ),
        )
        .sort_values(
            "number_of_clusters"
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    workloads = (
        load_workloads()
    )

    clusters = (
        load_clusters()
    )

    analysis = (
        analyze_workloads(
            workloads,
            clusters,
        )
    )

    summary = (
        build_summary(
            analysis
        )
    )

    analysis.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    total = len(
        analysis
    )

    single_cluster = int(
        (
            analysis[
                "number_of_clusters"
            ]
            == 1
        ).sum()
    )

    multi_cluster = int(
        (
            analysis[
                "number_of_clusters"
            ]
            > 1
        ).sum()
    )

    print()
    print(
        "=" * 68
    )

    print(
        "ROUTING WORKLOAD + K-MEANS ANALYSIS"
    )

    print(
        "=" * 68
    )

    print(
        "Routing workloads:",
        total,
    )

    print(
        "Deliveries:",
        int(
            analysis[
                "number_of_deliveries"
            ].sum()
        ),
    )

    print(
        "Single-cluster workloads:",
        single_cluster,
    )

    print(
        "Multi-cluster workloads:",
        multi_cluster,
    )

    print(
        "Multi-cluster rate:",
        f"{multi_cluster / total * 100:.2f}%",
    )

    print()

    print(
        "Clusters per routing workload:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Dominant cluster share:"
    )

    print(
        analysis[
            "dominant_cluster_share"
        ]
        .describe()
        .to_string()
    )

    print()

    print(
        "Cluster count by workload size:"
    )

    cross = pd.crosstab(
        analysis[
            "number_of_deliveries"
        ],
        analysis[
            "number_of_clusters"
        ],
    )

    print(
        cross.to_string()
    )

    print()

    print(
        "Created:"
    )

    print(
        "-",
        OUTPUT_PATH,
    )

    print(
        "-",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()