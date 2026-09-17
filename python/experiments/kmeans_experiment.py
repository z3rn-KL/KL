from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data import DataLoader, DataPreprocessor
from clustering.geo_projection import LocalGeoProjection


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_PATH = (
    RESULTS_DIR
    / "kmeans_experiment_results.csv"
)

K_VALUES = [
    5,
    8,
    10,
    12,
    15,
    20,
    25,
    30,
]


def load_xedu_coordinates() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load and preprocess XeDu delivery data.

    Returns:
        dataframe:
            Cleaned XeDu dataframe.

        coordinates:
            Nx2 numpy array in:
            [latitude, longitude]
    """

    dataframe = DataLoader.load_csv(
        DATASET_PATH
    )

    dataframe = (
        DataPreprocessor.prepare_deliveries(
            dataframe
        )
    )

    coordinates = dataframe[
        [
            "receiverLat",
            "receiverLng",
        ]
    ].to_numpy(
        dtype=float
    )

    return dataframe, coordinates


def calculate_cluster_statistics(
    labels: np.ndarray,
) -> dict:
    """
    Calculate cluster-size statistics.
    """

    _, counts = np.unique(
        labels,
        return_counts=True,
    )

    minimum = int(
        counts.min()
    )

    maximum = int(
        counts.max()
    )

    mean = float(
        counts.mean()
    )

    std = float(
        counts.std()
    )

    imbalance_ratio = (
        float(maximum / minimum)
        if minimum > 0
        else float("inf")
    )

    return {
        "min_cluster_size": minimum,
        "max_cluster_size": maximum,
        "mean_cluster_size": mean,
        "std_cluster_size": std,
        "imbalance_ratio": imbalance_ratio,
    }


def normalize_series(
    values: np.ndarray,
) -> np.ndarray:
    """
    Min-max normalization to [0, 1].
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    minimum = values.min()
    maximum = values.max()

    if np.isclose(
        maximum,
        minimum,
    ):
        return np.zeros_like(
            values,
            dtype=float,
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def calculate_elbow_strength(
    k_values: np.ndarray,
    inertias: np.ndarray,
) -> np.ndarray:
    """
    Estimate the strength of the elbow.

    Method
    ------
    1. Normalize K to [0, 1].
    2. Normalize inertia to [0, 1].
    3. Reverse normalized inertia so the curve
       grows from low to high.
    4. Measure vertical deviation from the
       straight diagonal line.

    Larger value = stronger elbow candidate.

    This avoids requiring an additional
    dependency such as kneed.
    """

    normalized_k = normalize_series(
        k_values
    )

    normalized_inertia = normalize_series(
        inertias
    )

    reversed_inertia = (
        1.0 - normalized_inertia
    )

    elbow_strength = (
        reversed_inertia
        - normalized_k
    )

    return elbow_strength


def run_kmeans_experiment() -> pd.DataFrame:
    """
    Run K-Means for multiple values of K.

    Metrics:
        - inertia
        - silhouette score
        - cluster-size statistics
        - imbalance ratio
        - runtime
        - elbow strength
    """

    print(
        "\n"
        "========================================"
    )
    print(
        "XeDu K-Means Experiment"
    )
    print(
        "========================================"
    )

    dataframe, coordinates = (
        load_xedu_coordinates()
    )

    print(
        f"Deliveries: {len(dataframe)}"
    )

    print(
        f"K values: {K_VALUES}"
    )

    print(
        "\nProjecting GPS coordinates..."
    )

    projection = (
        LocalGeoProjection
        .from_coordinates(
            coordinates
        )
    )

    projected = projection.transform(
        coordinates
    )

    results = []

    for k in K_VALUES:
        print(
            f"\nRunning K-Means with K={k}..."
        )

        start_time = perf_counter()

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(
            projected
        )

        runtime = (
            perf_counter()
            - start_time
        )

        inertia = float(
            model.inertia_
        )

        silhouette = float(
            silhouette_score(
                projected,
                labels,
            )
        )

        stats = (
            calculate_cluster_statistics(
                labels
            )
        )

        result = {
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette,
            **stats,
            "runtime_seconds": runtime,
        }

        results.append(
            result
        )

        print(
            f"  Inertia: "
            f"{inertia:.4f}"
        )

        print(
            f"  Silhouette: "
            f"{silhouette:.4f}"
        )

        print(
            f"  Cluster size: "
            f"{stats['min_cluster_size']} "
            f"-> "
            f"{stats['max_cluster_size']}"
        )

        print(
            f"  Std size: "
            f"{stats['std_cluster_size']:.2f}"
        )

        print(
            f"  Imbalance ratio: "
            f"{stats['imbalance_ratio']:.2f}"
        )

        print(
            f"  Runtime: "
            f"{runtime:.4f}s"
        )

    results_df = pd.DataFrame(
        results
    )

    elbow_strength = (
        calculate_elbow_strength(
            results_df[
                "k"
            ].to_numpy(),
            results_df[
                "inertia"
            ].to_numpy(),
        )
    )

    results_df[
        "elbow_strength"
    ] = elbow_strength

    return results_df


def print_summary(
    results_df: pd.DataFrame,
) -> None:
    """
    Print important experiment candidates.
    """

    print(
        "\n"
        "========================================"
    )
    print(
        "Experiment Summary"
    )
    print(
        "========================================"
    )

    display_columns = [
        "k",
        "inertia",
        "silhouette",
        "min_cluster_size",
        "max_cluster_size",
        "std_cluster_size",
        "imbalance_ratio",
        "elbow_strength",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    silhouette_index = (
        results_df[
            "silhouette"
        ].idxmax()
    )

    elbow_index = (
        results_df[
            "elbow_strength"
        ].idxmax()
    )

    balance_index = (
        results_df[
            "std_cluster_size"
        ].idxmin()
    )

    best_silhouette = (
        results_df.loc[
            silhouette_index
        ]
    )

    best_elbow = (
        results_df.loc[
            elbow_index
        ]
    )

    best_balance = (
        results_df.loc[
            balance_index
        ]
    )

    print(
        "\n----------------------------------------"
    )

    print(
        "Best Silhouette candidate:"
    )

    print(
        f"  K = "
        f"{int(best_silhouette['k'])}"
    )

    print(
        f"  Silhouette = "
        f"{best_silhouette['silhouette']:.4f}"
    )

    print(
        "\nStrongest Elbow candidate:"
    )

    print(
        f"  K = "
        f"{int(best_elbow['k'])}"
    )

    print(
        f"  Elbow strength = "
        f"{best_elbow['elbow_strength']:.4f}"
    )

    print(
        "\nBest cluster-size balance candidate:"
    )

    print(
        f"  K = "
        f"{int(best_balance['k'])}"
    )

    print(
        f"  Std = "
        f"{best_balance['std_cluster_size']:.4f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not automatically select K using "
        "only one metric."
    )

    print(
        "Final K should consider:"
    )

    print(
        "  1. Silhouette quality"
    )

    print(
        "  2. Elbow behaviour"
    )

    print(
        "  3. Cluster balance"
    )

    print(
        "  4. Operational feasibility"
    )


def save_results(
    results_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_PATH
    )


def main() -> None:
    results_df = (
        run_kmeans_experiment()
    )

    print_summary(
        results_df
    )

    save_results(
        results_df
    )


if __name__ == "__main__":
    main()