from fastapi.testclient import (
    TestClient,
)

from api.main import (
    app,
)


client = TestClient(
    app
)


SAMPLE_DELIVERIES = [
    {
        "delivery_id": "A1",
        "latitude": 10.7500,
        "longitude": 106.6500,
        "weight": 5.0,
    },
    {
        "delivery_id": "A2",
        "latitude": 10.7510,
        "longitude": 106.6510,
        "weight": 4.0,
    },
    {
        "delivery_id": "A3",
        "latitude": 10.7520,
        "longitude": 106.6490,
        "weight": 6.0,
    },
    {
        "delivery_id": "B1",
        "latitude": 10.8500,
        "longitude": 106.8000,
        "weight": 7.0,
    },
    {
        "delivery_id": "B2",
        "latitude": 10.8510,
        "longitude": 106.8010,
        "weight": 8.0,
    },
    {
        "delivery_id": "B3",
        "latitude": 10.8490,
        "longitude": 106.7990,
        "weight": 5.0,
    },
]


def test_precomputed_kmeans():
    response = client.get(
        "/api/clustering/kmeans",
        params={
            "limit": 5,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "algorithm"
        ]
        == "kmeans"
    )

    assert (
        payload[
            "total_deliveries"
        ]
        == 2394
    )

    assert (
        payload[
            "n_clusters"
        ]
        == 10
    )

    assert (
        payload[
            "returned"
        ]
        == 5
    )


def test_live_kmeans():
    response = client.post(
        "/api/clustering/kmeans/run",
        json={
            "deliveries":
                SAMPLE_DELIVERIES,

            "n_clusters":
                2,

            "random_state":
                42,

            "n_init":
                20,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "algorithm"
        ]
        == "kmeans"
    )

    assert (
        payload[
            "metrics"
        ][
            "n_clusters"
        ]
        == 2
    )

    assert (
        len(
            payload[
                "assignments"
            ]
        )
        == 6
    )

    sizes = sorted(
        cluster[
            "size"
        ]
        for cluster
        in payload[
            "clusters"
        ]
    )

    assert sizes == [
        3,
        3,
    ]


def test_live_kmeans_invalid_cluster_count():
    response = client.post(
        "/api/clustering/kmeans/run",
        json={
            "deliveries":
                SAMPLE_DELIVERIES,

            "n_clusters":
                20,
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_live_kmeans_duplicate_ids():
    deliveries = [
        SAMPLE_DELIVERIES[
            0
        ],
        {
            **SAMPLE_DELIVERIES[
                1
            ],
            "delivery_id":
                "A1",
        },
    ]

    response = client.post(
        "/api/clustering/kmeans/run",
        json={
            "deliveries":
                deliveries,

            "n_clusters":
                1,
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_live_dbscan():
    response = client.post(
        "/api/clustering/dbscan/run",
        json={
            "deliveries":
                SAMPLE_DELIVERIES,

            "eps_km":
                1.0,

            "min_samples":
                2,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "algorithm"
        ]
        == "dbscan"
    )

    assert (
        payload[
            "metrics"
        ][
            "n_clusters"
        ]
        == 2
    )

    assert (
        payload[
            "metrics"
        ][
            "n_noise"
        ]
        == 0
    )

    assert (
        len(
            payload[
                "assignments"
            ]
        )
        == 6
    )


def test_dbscan_noise():
    deliveries = (
        SAMPLE_DELIVERIES
        + [
            {
                "delivery_id":
                    "OUTLIER",

                "latitude":
                    11.3000,

                "longitude":
                    107.3000,

                "weight":
                    1.0,
            }
        ]
    )

    response = client.post(
        "/api/clustering/dbscan/run",
        json={
            "deliveries":
                deliveries,

            "eps_km":
                1.0,

            "min_samples":
                2,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        "OUTLIER"
        in payload[
            "noise_delivery_ids"
        ]
    )

    assignments = {
        item[
            "delivery_id"
        ]:
            item[
                "cluster_id"
            ]
        for item
        in payload[
            "assignments"
        ]
    }

    assert (
        assignments[
            "OUTLIER"
        ]
        == -1
    )


def test_invalid_coordinates_rejected():
    response = client.post(
        "/api/clustering/kmeans/run",
        json={
            "deliveries": [
                {
                    "delivery_id":
                        "BAD",

                    "latitude":
                        120.0,

                    "longitude":
                        106.7,
                }
            ],

            "n_clusters":
                1,
        },
    )

    assert (
        response.status_code
        == 422
    )