from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data import (
    DataLoader,
    DataPreprocessor,
)

from clustering.geo_projection import LocalGeoProjection


PROJECT_ROOT = Path(__file__).resolve().parents[2]

XEDU_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

DEFAULT_N_CLUSTERS = 10
RANDOM_STATE = 42


def load_xedu() -> pd.DataFrame:
    """
    Đọc và tiền xử lý dataset XeDu.
    """

    dataframe = DataLoader.load_csv(
        XEDU_PATH
    )

    cleaned = DataPreprocessor.prepare_deliveries(
        dataframe
    )

    return cleaned


def build_coordinates(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """
    Chuyển receiverLat / receiverLng
    thành NumPy array dạng:

    [
        [lat1, lng1],
        [lat2, lng2],
        ...
    ]
    """

    coordinates = dataframe[
        [
            "receiverLat",
            "receiverLng",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    return coordinates


def project_coordinates(
    coordinates: np.ndarray,
) -> tuple[
    LocalGeoProjection,
    np.ndarray,
]:
    """
    Chuyển lat/lon sang tọa độ cục bộ x/y theo km.
    """

    projection = (
        LocalGeoProjection.from_coordinates(
            coordinates
        )
    )

    projected = projection.transform(
        coordinates
    )

    return projection, projected


def run_kmeans(
    projected_coordinates: np.ndarray,
    n_clusters: int,
) -> KMeans:
    """
    Huấn luyện K-Means.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    model.fit(
        projected_coordinates
    )

    return model


def calculate_cluster_sizes(
    labels: np.ndarray,
) -> dict[int, int]:
    """
    Đếm số điểm thuộc mỗi cluster.
    """

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    return {
        int(label): int(count)
        for label, count in zip(
            unique_labels,
            counts,
        )
    }


def print_cluster_report(
    model: KMeans,
    projected_coordinates: np.ndarray,
) -> None:
    """
    In thống kê clustering.
    """

    labels = model.labels_

    cluster_sizes = calculate_cluster_sizes(
        labels
    )

    silhouette = silhouette_score(
        projected_coordinates,
        labels,
    )

    sizes = np.array(
        list(cluster_sizes.values()),
        dtype=np.float64,
    )

    print(
        "\n======================================"
    )

    print(
        "       XEDU K-MEANS EXPERIMENT"
    )

    print(
        "======================================"
    )

    print(
        "Tổng số delivery:",
        len(projected_coordinates),
    )

    print(
        "Số cluster:",
        model.n_clusters,
    )

    print(
        "Inertia:",
        round(
            float(model.inertia_),
            4,
        ),
    )

    print(
        "Silhouette score:",
        round(
            float(silhouette),
            4,
        ),
    )

    print(
        "Cluster nhỏ nhất:",
        int(sizes.min()),
    )

    print(
        "Cluster lớn nhất:",
        int(sizes.max()),
    )

    print(
        "Kích thước cluster trung bình:",
        round(
            float(sizes.mean()),
            2,
        ),
    )

    print(
        "Độ lệch chuẩn kích thước cluster:",
        round(
            float(sizes.std()),
            2,
        ),
    )

    print(
        "\n===== CLUSTER SIZE ====="
    )

    for cluster_id in sorted(
        cluster_sizes
    ):
        print(
            f"Cluster {cluster_id:02d}: "
            f"{cluster_sizes[cluster_id]} delivery"
        )

    print(
        "\n===== CENTROIDS (x/y km) ====="
    )

    for cluster_id, centroid in enumerate(
        model.cluster_centers_
    ):
        x_km = float(
            centroid[0]
        )

        y_km = float(
            centroid[1]
        )

        print(
            f"Cluster {cluster_id:02d}: "
            f"x={x_km:.3f} km, "
            f"y={y_km:.3f} km"
        )


def save_results(
    dataframe: pd.DataFrame,
    model: KMeans,
) -> Path:
    """
    Gắn cluster_id vào dataset và lưu CSV.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = dataframe.copy()

    result["cluster_id"] = (
        model.labels_
    )

    output_path = (
        RESULTS_DIR
        / "xedu_kmeans_clusters.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return output_path


def run_experiment(
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> None:
    """
    Chạy toàn bộ pipeline clustering.
    """

    print(
        "Đang đọc XeDu..."
    )

    dataframe = load_xedu()

    print(
        "Số delivery hợp lệ:",
        len(dataframe),
    )

    coordinates = build_coordinates(
        dataframe
    )

    print(
        "Đang chuyển lat/lon sang x/y..."
    )

    _, projected = project_coordinates(
        coordinates
    )

    print(
        "Đang chạy K-Means..."
    )

    model = run_kmeans(
        projected_coordinates=projected,
        n_clusters=n_clusters,
    )

    print_cluster_report(
        model=model,
        projected_coordinates=projected,
    )

    output_path = save_results(
        dataframe=dataframe,
        model=model,
    )

    print(
        "\nĐã lưu kết quả:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    run_experiment()