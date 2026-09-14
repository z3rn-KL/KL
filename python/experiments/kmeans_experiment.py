from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from clustering.experiment import (
    load_xedu,
    build_coordinates,
    project_coordinates,
    run_kmeans,
    calculate_cluster_sizes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_PATH = (
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


def evaluate_k(
    projected_coordinates: np.ndarray,
    k: int,
) -> dict:
    """
    Chạy K-Means với một giá trị K và trả về các metrics.
    """

    start_time = perf_counter()

    model = run_kmeans(
        projected_coordinates=projected_coordinates,
        n_clusters=k,
    )

    runtime_seconds = (
        perf_counter()
        - start_time
    )

    labels = model.labels_

    cluster_sizes = calculate_cluster_sizes(
        labels
    )

    sizes = np.array(
        list(cluster_sizes.values()),
        dtype=np.float64,
    )

    silhouette = silhouette_score(
        projected_coordinates,
        labels,
    )

    smallest_cluster = int(
        sizes.min()
    )

    largest_cluster = int(
        sizes.max()
    )

    mean_cluster_size = float(
        sizes.mean()
    )

    std_cluster_size = float(
        sizes.std()
    )

    imbalance_ratio = (
        largest_cluster
        / smallest_cluster
    )

    return {
        "k": k,
        "inertia": float(
            model.inertia_
        ),
        "silhouette_score": float(
            silhouette
        ),
        "smallest_cluster": smallest_cluster,
        "largest_cluster": largest_cluster,
        "mean_cluster_size": mean_cluster_size,
        "std_cluster_size": std_cluster_size,
        "imbalance_ratio": float(
            imbalance_ratio
        ),
        "runtime_seconds": float(
            runtime_seconds
        ),
    }


def print_result(
    result: dict,
) -> None:
    print(
        f"K={result['k']:>2} | "
        f"Silhouette={result['silhouette_score']:.4f} | "
        f"Inertia={result['inertia']:.2f} | "
        f"Min={result['smallest_cluster']:>4} | "
        f"Max={result['largest_cluster']:>4} | "
        f"Std={result['std_cluster_size']:.2f} | "
        f"Ratio={result['imbalance_ratio']:.2f} | "
        f"Runtime={result['runtime_seconds']:.4f}s"
    )


def run_experiment() -> pd.DataFrame:
    print(
        "=============================================="
    )

    print(
        "       K-MEANS MULTI-K EXPERIMENT"
    )

    print(
        "=============================================="
    )

    print(
        "\nĐang đọc dataset XeDu..."
    )

    dataframe = load_xedu()

    print(
        "Số delivery:",
        len(dataframe),
    )

    coordinates = build_coordinates(
        dataframe
    )

    _, projected = project_coordinates(
        coordinates
    )

    print(
        "\nBắt đầu thử các giá trị K:\n"
    )

    results = []

    for k in K_VALUES:
        result = evaluate_k(
            projected_coordinates=projected,
            k=k,
        )

        results.append(
            result
        )

        print_result(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n=============================================="
    )

    print(
        "Đã lưu kết quả tại:"
    )

    print(
        OUTPUT_PATH
    )

    return results_df


def print_best_candidates(
    results_df: pd.DataFrame,
) -> None:
    best_silhouette = (
        results_df
        .sort_values(
            "silhouette_score",
            ascending=False,
        )
        .iloc[0]
    )

    best_balance = (
        results_df
        .sort_values(
            "std_cluster_size",
            ascending=True,
        )
        .iloc[0]
    )

    print(
        "\n===== BEST SILHOUETTE ====="
    )

    print(
        f"K = {int(best_silhouette['k'])}"
    )

    print(
        "Silhouette =",
        round(
            float(
                best_silhouette[
                    "silhouette_score"
                ]
            ),
            4,
        ),
    )

    print(
        "\n===== BEST SIZE BALANCE ====="
    )

    print(
        f"K = {int(best_balance['k'])}"
    )

    print(
        "Std cluster size =",
        round(
            float(
                best_balance[
                    "std_cluster_size"
                ]
            ),
            2,
        ),
    )


def main() -> None:
    results_df = run_experiment()

    print_best_candidates(
        results_df
    )


if __name__ == "__main__":
    main()