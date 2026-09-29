from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
)

from clustering.dbscan_clusterer import (
    DBSCANDeliveryClusterer,
)

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

GRID_OUTPUT_PATH = (
    RESULTS_DIR
    / "dbscan_grid_results.csv"
)

LABEL_OUTPUT_PATH = (
    RESULTS_DIR
    / "dbscan_best_labels.csv"
)


EPS_VALUES_KM = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
]

MIN_SAMPLES_VALUES = [
    3,
    5,
    8,
    10,
]


def load_deliveries():
    dataframe = (
        DataLoader.load_csv(
            DATASET_PATH
        )
    )

    dataframe = (
        DataPreprocessor
        .prepare_deliveries(
            dataframe
        )
    )

    deliveries = (
        DeliveryMapper
        .from_dataframe(
            dataframe
        )
    )

    return (
        dataframe,
        deliveries,
    )


def calculate_silhouette(
    projected_points: np.ndarray,
    labels: np.ndarray,
) -> float | None:
    """
    Silhouette is calculated only on
    non-noise DBSCAN points.

    Noise label = -1.
    """

    valid_mask = (
        labels != -1
    )

    valid_points = (
        projected_points[
            valid_mask
        ]
    )

    valid_labels = (
        labels[
            valid_mask
        ]
    )

    unique_labels = np.unique(
        valid_labels
    )

    if len(
        unique_labels
    ) < 2:
        return None

    if len(
        valid_points
    ) <= len(
        unique_labels
    ):
        return None

    return float(
        silhouette_score(
            valid_points,
            valid_labels,
            metric="euclidean",
        )
    )


def calculate_cluster_sizes(
    labels: np.ndarray,
) -> list[int]:
    cluster_labels = [
        int(label)
        for label
        in np.unique(
            labels
        )
        if int(label) != -1
    ]

    sizes = []

    for cluster_id in (
        cluster_labels
    ):
        size = int(
            np.sum(
                labels
                == cluster_id
            )
        )

        sizes.append(
            size
        )

    return sizes


def run_configuration(
    deliveries,
    eps_km: float,
    min_samples: int,
) -> dict:
    clusterer = (
        DBSCANDeliveryClusterer(
            eps_km=eps_km,
            min_samples=min_samples,
        )
    )

    timer = perf_counter()

    result = clusterer.fit(
        deliveries
    )

    runtime = (
        perf_counter()
        - timer
    )

    silhouette = (
        calculate_silhouette(
            projected_points=(
                result.projected_points
            ),
            labels=result.labels,
        )
    )

    cluster_sizes = (
        calculate_cluster_sizes(
            result.labels
        )
    )

    if cluster_sizes:
        min_cluster_size = min(
            cluster_sizes
        )

        max_cluster_size = max(
            cluster_sizes
        )

        mean_cluster_size = float(
            np.mean(
                cluster_sizes
            )
        )

        median_cluster_size = float(
            np.median(
                cluster_sizes
            )
        )

        std_cluster_size = float(
            np.std(
                cluster_sizes
            )
        )

    else:
        min_cluster_size = 0
        max_cluster_size = 0
        mean_cluster_size = 0.0
        median_cluster_size = 0.0
        std_cluster_size = 0.0

    clustered_count = (
        len(
            deliveries
        )
        - result.n_noise
    )

    return {
        "eps_km":
            eps_km,

        "eps_meters":
            eps_km
            * 1000.0,

        "min_samples":
            min_samples,

        "number_of_deliveries":
            len(
                deliveries
            ),

        "number_of_clusters":
            result.n_clusters,

        "number_of_noise":
            result.n_noise,

        "noise_ratio":
            result.noise_ratio,

        "clustered_deliveries":
            clustered_count,

        "clustered_ratio":
            clustered_count
            / len(
                deliveries
            ),

        "silhouette_score":
            silhouette,

        "min_cluster_size":
            min_cluster_size,

        "max_cluster_size":
            max_cluster_size,

        "mean_cluster_size":
            mean_cluster_size,

        "median_cluster_size":
            median_cluster_size,

        "std_cluster_size":
            std_cluster_size,

        "runtime_seconds":
            runtime,
    }


def select_reference_configuration(
    results_df: pd.DataFrame,
):
    """
    Select a reference configuration for
    further analysis.

    We first keep configurations with:
    - at least 2 clusters;
    - silhouette available;
    - noise ratio <= 20%.

    Then select the highest silhouette.

    This is only a reference configuration,
    not an assertion that it is universally
    optimal.
    """

    candidates = (
        results_df[
            (
                results_df[
                    "number_of_clusters"
                ]
                >= 2
            )
            &
            (
                results_df[
                    "silhouette_score"
                ]
                .notna()
            )
            &
            (
                results_df[
                    "noise_ratio"
                ]
                <= 0.20
            )
        ]
        .copy()
    )

    if candidates.empty:
        candidates = (
            results_df[
                (
                    results_df[
                        "number_of_clusters"
                    ]
                    >= 2
                )
                &
                (
                    results_df[
                        "silhouette_score"
                    ]
                    .notna()
                )
            ]
            .copy()
        )

    if candidates.empty:
        return None

    candidates = (
        candidates
        .sort_values(
            by=[
                "silhouette_score",
                "noise_ratio",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    return candidates.iloc[
        0
    ]


def save_reference_labels(
    deliveries,
    eps_km: float,
    min_samples: int,
):
    clusterer = (
        DBSCANDeliveryClusterer(
            eps_km=eps_km,
            min_samples=min_samples,
        )
    )

    result = clusterer.fit(
        deliveries
    )

    rows = []

    for delivery, label in zip(
        deliveries,
        result.labels,
        strict=True,
    ):
        rows.append(
            {
                "delivery_id":
                    str(
                        delivery.delivery_id
                    ),

                "latitude":
                    delivery.latitude,

                "longitude":
                    delivery.longitude,

                "dbscan_label":
                    int(
                        label
                    ),

                "is_noise":
                    int(
                        label
                    )
                    == -1,
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        LABEL_OUTPUT_PATH,
        index=False,
    )


def main():
    print()
    print(
        "========================================"
    )
    print(
        "XeDu DBSCAN Parameter Experiment"
    )
    print(
        "========================================"
    )

    (
        _,
        deliveries,
    ) = load_deliveries()

    print()
    print(
        f"Deliveries: "
        f"{len(deliveries)}"
    )

    rows = []

    total_runs = (
        len(
            EPS_VALUES_KM
        )
        * len(
            MIN_SAMPLES_VALUES
        )
    )

    run_number = 0

    for eps_km in (
        EPS_VALUES_KM
    ):
        for min_samples in (
            MIN_SAMPLES_VALUES
        ):
            run_number += 1

            print()
            print(
                f"[{run_number}/{total_runs}] "
                f"eps={eps_km:.2f} km, "
                f"min_samples={min_samples}"
            )

            row = (
                run_configuration(
                    deliveries=deliveries,
                    eps_km=eps_km,
                    min_samples=min_samples,
                )
            )

            rows.append(
                row
            )

            silhouette = row[
                "silhouette_score"
            ]

            silhouette_text = (
                "N/A"
                if silhouette is None
                else f"{silhouette:.4f}"
            )

            print(
                f"  clusters: "
                f"{row['number_of_clusters']}"
            )

            print(
                f"  noise: "
                f"{row['number_of_noise']} "
                f"({row['noise_ratio']:.2%})"
            )

            print(
                f"  silhouette: "
                f"{silhouette_text}"
            )

            print(
                f"  cluster size: "
                f"{row['min_cluster_size']} "
                f"- "
                f"{row['max_cluster_size']}"
            )

    results_df = pd.DataFrame(
        rows
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "silhouette_score",
                "noise_ratio",
            ],
            ascending=[
                False,
                True,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        GRID_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "========================================"
    )
    print(
        "DBSCAN Grid Summary"
    )
    print(
        "========================================"
    )

    columns = [
        "eps_km",
        "min_samples",
        "number_of_clusters",
        "number_of_noise",
        "noise_ratio",
        "silhouette_score",
        "min_cluster_size",
        "max_cluster_size",
        "mean_cluster_size",
    ]

    print(
        results_df[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    reference = (
        select_reference_configuration(
            results_df
        )
    )

    if reference is not None:
        eps_km = float(
            reference[
                "eps_km"
            ]
        )

        min_samples = int(
            reference[
                "min_samples"
            ]
        )

        print()
        print(
            "========================================"
        )
        print(
            "Reference DBSCAN Configuration"
        )
        print(
            "========================================"
        )

        print(
            f"eps: "
            f"{eps_km:.2f} km "
            f"({eps_km * 1000:.0f} m)"
        )

        print(
            f"min_samples: "
            f"{min_samples}"
        )

        print(
            f"clusters: "
            f"{int(reference['number_of_clusters'])}"
        )

        print(
            f"noise: "
            f"{int(reference['number_of_noise'])} "
            f"({reference['noise_ratio']:.2%})"
        )

        print(
            f"silhouette: "
            f"{reference['silhouette_score']:.4f}"
        )

        save_reference_labels(
            deliveries=deliveries,
            eps_km=eps_km,
            min_samples=min_samples,
        )

    else:
        print()
        print(
            "No valid DBSCAN reference "
            "configuration found."
        )

    print()
    print(
        "========================================"
    )
    print(
        "Results Saved"
    )
    print(
        "========================================"
    )

    print(
        GRID_OUTPUT_PATH
    )

    if reference is not None:
        print(
            LABEL_OUTPUT_PATH
        )


if __name__ == "__main__":
    main()