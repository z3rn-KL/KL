from pathlib import Path

import pandas as pd

from experiments.q_learning_single_workload_experiment import (
    load_delivery_dataframe,
    load_snapped_nodes,
    prepare_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

CLUSTER_PATH = (
    RESULTS_DIR
    / "xedu_kmeans_clusters.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "cluster_aware_workloads.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "cluster_aware_workload_summary.csv"
)

WAVE_SUMMARY_PATH = (
    RESULTS_DIR
    / "cluster_aware_wave_summary.csv"
)


MIN_ROUTING_SIZE = 5


def load_cluster_assignments() -> pd.DataFrame:
    if not CLUSTER_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy K-Means result: "
            f"{CLUSTER_PATH}"
        )

    dataframe = pd.read_csv(
        CLUSTER_PATH
    )

    # K-Means dataset sử dụng "id",
    # trong khi routing pipeline sử dụng "delivery_id".
    if "delivery_id" not in dataframe.columns:
        if "id" in dataframe.columns:
            dataframe = dataframe.rename(
                columns={
                    "id": "delivery_id"
                }
            )
        else:
            raise ValueError(
                "K-Means CSV không có "
                "'id' hoặc 'delivery_id'."
            )

    required = [
        "delivery_id",
        "cluster_id",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "K-Means CSV thiếu cột: "
            + ", ".join(missing)
        )

    result = (
        dataframe[
            [
                "delivery_id",
                "cluster_id",
            ]
        ]
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

    if result[
        "delivery_id"
    ].duplicated().any():
        duplicate_count = int(
            result[
                "delivery_id"
            ].duplicated().sum()
        )

        raise ValueError(
            f"K-Means output có "
            f"{duplicate_count} delivery_id trùng."
        )

    return result


def load_prepared_deliveries() -> pd.DataFrame:
    dataframe, _ = (
        load_delivery_dataframe()
    )

    snapped_df = (
        load_snapped_nodes()
    )

    dataframe = (
        prepare_dataframe(
            dataframe,
            snapped_df,
        )
    )

    dataframe[
        "delivery_id"
    ] = (
        dataframe[
            "delivery_id"
        ]
        .astype(str)
    )

    if "wave_start" not in dataframe.columns:
        raise ValueError(
            "Prepared dataframe không có wave_start."
        )

    return dataframe


def merge_cluster_and_wave() -> pd.DataFrame:
    deliveries = (
        load_prepared_deliveries()
    )

    clusters = (
        load_cluster_assignments()
    )

    dataframe = deliveries.merge(
        clusters,
        on="delivery_id",
        how="left",
        validate="one_to_one",
    )

    missing_cluster = dataframe[
        "cluster_id"
    ].isna().sum()

    if missing_cluster != 0:
        raise ValueError(
            f"Có {missing_cluster} delivery "
            "không có cluster_id."
        )

    dataframe[
        "cluster_id"
    ] = dataframe[
        "cluster_id"
    ].astype(int)

    return dataframe


def build_cluster_aware_workloads(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    groups = dataframe.groupby(
        [
            "wave_start",
            "cluster_id",
        ],
        sort=True,
    )

    workload_id = 1

    for (
        (
            wave_start,
            cluster_id,
        ),
        group,
    ) in groups:

        group = (
            group
            .sort_values(
                "delivery_id"
            )
            .copy()
        )

        delivery_ids = (
            group[
                "delivery_id"
            ]
            .astype(str)
            .tolist()
        )

        rows.append(
            {
                "cluster_workload_id":
                    workload_id,

                "wave_start":
                    wave_start,

                "cluster_id":
                    int(
                        cluster_id
                    ),

                "number_of_deliveries":
                    len(
                        group
                    ),

                "routing_eligible":
                    len(
                        group
                    )
                    >= MIN_ROUTING_SIZE,

                "delivery_ids":
                    " -> ".join(
                        delivery_ids
                    ),
            }
        )

        workload_id += 1

    return pd.DataFrame(
        rows
    )


def build_size_summary(
    workloads: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        workloads
        .groupby(
            "number_of_deliveries",
            as_index=False,
        )
        .agg(
            workloads=(
                "cluster_workload_id",
                "count",
            )
        )
        .sort_values(
            "number_of_deliveries"
        )
        .reset_index(
            drop=True
        )
    )

    summary[
        "deliveries"
    ] = (
        summary[
            "number_of_deliveries"
        ]
        * summary[
            "workloads"
        ]
    )

    return summary


def build_wave_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        wave_start,
        group,
    ) in dataframe.groupby(
        "wave_start"
    ):
        cluster_counts = (
            group[
                "cluster_id"
            ]
            .value_counts()
        )

        total_deliveries = len(
            group
        )

        number_of_clusters = len(
            cluster_counts
        )

        dominant_cluster_size = int(
            cluster_counts.max()
        )

        dominant_cluster_share = (
            dominant_cluster_size
            / total_deliveries
        )

        rows.append(
            {
                "wave_start":
                    wave_start,

                "number_of_deliveries":
                    total_deliveries,

                "number_of_clusters":
                    number_of_clusters,

                "dominant_cluster_size":
                    dominant_cluster_size,

                "dominant_cluster_share":
                    dominant_cluster_share,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "wave_start"
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    dataframe = (
        merge_cluster_and_wave()
    )

    workloads = (
        build_cluster_aware_workloads(
            dataframe
        )
    )

    size_summary = (
        build_size_summary(
            workloads
        )
    )

    wave_summary = (
        build_wave_summary(
            dataframe
        )
    )

    workloads.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    size_summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    wave_summary.to_csv(
        WAVE_SUMMARY_PATH,
        index=False,
    )

    total_deliveries = len(
        dataframe
    )

    total_waves = dataframe[
        "wave_start"
    ].nunique()

    total_cluster_workloads = len(
        workloads
    )

    eligible = workloads[
        workloads[
            "routing_eligible"
        ]
    ]

    eligible_workloads = len(
        eligible
    )

    eligible_deliveries = int(
        eligible[
            "number_of_deliveries"
        ].sum()
    )

    multi_cluster_waves = int(
        (
            wave_summary[
                "number_of_clusters"
            ]
            > 1
        ).sum()
    )

    single_cluster_waves = int(
        (
            wave_summary[
                "number_of_clusters"
            ]
            == 1
        ).sum()
    )

    print()
    print(
        "=" * 68
    )
    print(
        "K-MEANS + TEMPORAL WAVE INTEGRATION"
    )
    print(
        "=" * 68
    )

    print(
        "K-Means clusters:",
        dataframe[
            "cluster_id"
        ].nunique(),
    )

    print(
        "Deliveries:",
        total_deliveries,
    )

    print(
        "Temporal waves:",
        total_waves,
    )

    print(
        "Cluster-aware workloads:",
        total_cluster_workloads,
    )

    print(
        "Single-cluster waves:",
        single_cluster_waves,
    )

    print(
        "Multi-cluster waves:",
        multi_cluster_waves,
    )

    print()
    print(
        "Routing threshold:",
        MIN_ROUTING_SIZE,
    )

    print(
        "Eligible cluster workloads:",
        eligible_workloads,
    )

    print(
        "Deliveries in eligible workloads:",
        eligible_deliveries,
    )

    print()
    print(
        "Cluster-aware workload size distribution:"
    )

    print(
        size_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Wave cluster-count distribution:"
    )

    print(
        wave_summary[
            "number_of_clusters"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "clusters_per_wave"
        )
        .reset_index(
            name="waves"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Dominant-cluster share:"
    )

    print(
        wave_summary[
            "dominant_cluster_share"
        ].describe().to_string()
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

    print(
        "-",
        WAVE_SUMMARY_PATH,
    )

    print()
    print(
        "Analysis complete."
    )


if __name__ == "__main__":
    main()