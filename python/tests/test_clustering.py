import numpy as np

from clustering import (
    KMeansDeliveryClusterer,
    LocalGeoProjection,
    evaluate_clustering,
)

from domain import Delivery


def test_projection_round_trip():
    coordinates = np.array(
        [
            [10.77, 106.70],
            [10.78, 106.71],
            [10.79, 106.72],
        ],
        dtype=np.float64,
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

    restored = (
        projection.inverse_transform(
            projected
        )
    )

    np.testing.assert_allclose(
        restored,
        coordinates,
        atol=1e-9,
    )


def test_kmeans_two_clusters():
    deliveries = [
        Delivery(
            delivery_id="A1",
            latitude=10.7500,
            longitude=106.6500,
            weight=5.0,
        ),
        Delivery(
            delivery_id="A2",
            latitude=10.7510,
            longitude=106.6510,
            weight=4.0,
        ),
        Delivery(
            delivery_id="A3",
            latitude=10.7520,
            longitude=106.6490,
            weight=6.0,
        ),

        Delivery(
            delivery_id="B1",
            latitude=10.8500,
            longitude=106.8000,
            weight=7.0,
        ),
        Delivery(
            delivery_id="B2",
            latitude=10.8510,
            longitude=106.8010,
            weight=8.0,
        ),
        Delivery(
            delivery_id="B3",
            latitude=10.8490,
            longitude=106.7990,
            weight=5.0,
        ),
    ]

    clusterer = (
        KMeansDeliveryClusterer(
            n_clusters=2,
            random_state=42,
        )
    )

    result = clusterer.fit(
        deliveries
    )

    assert len(result.clusters) == 2

    cluster_sizes = sorted(
        cluster.size()
        for cluster
        in result.clusters
    )

    assert cluster_sizes == [
        3,
        3,
    ]

    assert len(result.labels) == 6

    assert result.inertia >= 0


def test_clustering_metrics():
    deliveries = [
        Delivery(
            "A1",
            10.750,
            106.650,
            5.0,
        ),
        Delivery(
            "A2",
            10.751,
            106.651,
            5.0,
        ),
        Delivery(
            "B1",
            10.850,
            106.800,
            5.0,
        ),
        Delivery(
            "B2",
            10.851,
            106.801,
            5.0,
        ),
    ]

    clusterer = (
        KMeansDeliveryClusterer(
            n_clusters=2
        )
    )

    result = clusterer.fit(
        deliveries
    )

    metrics = evaluate_clustering(
        result
    )

    assert metrics.n_clusters == 2

    assert (
        metrics.min_cluster_size
        == 2
    )

    assert (
        metrics.max_cluster_size
        == 2
    )

    assert metrics.silhouette is not None

    assert (
        metrics.total_weight_kg
        == 20.0
    )s