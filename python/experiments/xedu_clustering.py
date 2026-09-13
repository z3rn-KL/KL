from pathlib import Path

from clustering import (
    KMeansDeliveryClusterer,
    evaluate_clustering,
)

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)


PROJECT_ROOT = Path(
    "/home/deez/Khóa Luận"
)

XEDU_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)


def test_xedu_kmeans():
    dataframe = DataLoader.load_csv(
        XEDU_PATH
    )

    cleaned = (
        DataPreprocessor
        .prepare_deliveries(
            dataframe
        )
    )

    deliveries = (
        DeliveryMapper
        .from_dataframe(
            cleaned
        )
    )

    # Đây chỉ là smoke test.
    # Chưa khẳng định K=10 là tối ưu.
    number_of_clusters = 10

    clusterer = (
        KMeansDeliveryClusterer(
            n_clusters=number_of_clusters,
            random_state=42,
        )
    )

    result = clusterer.fit(
        deliveries
    )

    metrics = evaluate_clustering(
        result
    )

    print(
        "\n===== XEDU K-MEANS ====="
    )

    print(
        "Số Delivery:",
        len(deliveries),
    )

    print(
        "K:",
        metrics.n_clusters,
    )

    print(
        "Inertia:",
        round(
            metrics.inertia,
            4,
        ),
    )

    print(
        "Silhouette:",
        round(
            metrics.silhouette,
            4,
        )
        if metrics.silhouette
        is not None
        else None,
    )

    print(
        "Cluster size min:",
        metrics.min_cluster_size,
    )

    print(
        "Cluster size max:",
        metrics.max_cluster_size,
    )

    print(
        "Cluster size mean:",
        round(
            metrics.mean_cluster_size,
            2,
        ),
    )

    print(
        "Cluster size CV:",
        round(
            metrics.cluster_size_cv,
            4,
        ),
    )

    print(
        "Total weight:",
        round(
            metrics.total_weight_kg,
            2,
        ),
        "kg",
    )

    print(
        "Cluster weight min:",
        round(
            metrics.min_cluster_weight_kg,
            2,
        ),
    )

    print(
        "Cluster weight max:",
        round(
            metrics.max_cluster_weight_kg,
            2,
        ),
    )

    print(
        "Cluster weight CV:",
        round(
            metrics.cluster_weight_cv,
            4,
        ),
    )

    print(
        "\n===== CLUSTER DETAIL ====="
    )

    for cluster in result.clusters:
        print(
            f"Cluster {cluster.cluster_id:02d}"
            f" | orders={cluster.size():4d}"
            f" | weight={cluster.total_weight():8.2f} kg"
            f" | centroid=("
            f"{cluster.centroid_latitude:.6f}, "
            f"{cluster.centroid_longitude:.6f})"
        )

    assert len(
        result.clusters
    ) == number_of_clusters

    assert sum(
        cluster.size()
        for cluster
        in result.clusters
    ) == len(deliveries)