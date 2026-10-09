from fastapi.testclient import (
    TestClient,
)

from api.main import (
    app,
)


client = TestClient(
    app
)


DELIVERIES = [
    {
        "delivery_id": "A",
        "latitude": 10.75,
        "longitude": 106.65,
    },
    {
        "delivery_id": "B",
        "latitude": 10.76,
        "longitude": 106.66,
    },
    {
        "delivery_id": "C",
        "latitude": 10.90,
        "longitude": 106.90,
    },
]


# Matrix:
#
# index 0 = depot
# index 1 = A
# index 2 = B
# index 3 = C
#
# A and B are near each other.
# C is far from both A and B.
#
DISTANCE_MATRIX = [
    [
        0.0,
        2.0,
        2.5,
        12.0,
    ],
    [
        2.0,
        0.0,
        1.0,
        10.0,
    ],
    [
        2.5,
        1.0,
        0.0,
        10.0,
    ],
    [
        12.0,
        10.0,
        10.0,
        0.0,
    ],
]


def test_adverse_screening():
    response = client.post(
        "/api/adverse-deliveries/screen",
        json={
            "deliveries":
                DELIVERIES,

            "distance_km":
                DISTANCE_MATRIX,

            "dbscan_labels": [
                0,
                0,
                -1,
            ],

            "isolation_multiplier":
                3.0,

            "detour_threshold_km":
                None,

            "depot_included":
                True,
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
        == (
            "adverse_delivery_screening"
        )
    )

    assert (
        payload[
            "total_deliveries"
        ]
        == 3
    )

    flagged = {
        entry[
            "delivery_id"
        ]:
            entry
        for entry
        in payload[
            "flagged"
        ]
    }

    assert (
        "C"
        in flagged
    )

    assert (
        "dbscan_spatial_noise"
        in flagged[
            "C"
        ][
            "reasons"
        ]
    )

    assert (
        "road_isolation_candidate"
        in flagged[
            "C"
        ][
            "reasons"
        ]
    )


def test_adverse_screening_isolation_ratio():
    response = client.post(
        "/api/adverse-deliveries/screen",
        json={
            "deliveries":
                DELIVERIES,

            "distance_km":
                DISTANCE_MATRIX,

            "isolation_multiplier":
                3.0,

            "depot_included":
                True,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    entries = {
        entry[
            "delivery_id"
        ]:
            entry
        for entry
        in payload[
            "entries"
        ]
    }

    # nearest peers:
    # A = 1
    # B = 1
    # C = 10
    #
    # median baseline = 1
    # C ratio = 10

    assert (
        entries[
            "C"
        ][
            "nearest_peer_distance_km"
        ]
        == 10.0
    )

    assert (
        entries[
            "C"
        ][
            "isolation_ratio"
        ]
        == 10.0
    )


def test_adverse_screening_without_depot():
    matrix = [
        [
            0.0,
            1.0,
            10.0,
        ],
        [
            1.0,
            0.0,
            10.0,
        ],
        [
            10.0,
            10.0,
            0.0,
        ],
    ]

    response = client.post(
        "/api/adverse-deliveries/screen",
        json={
            "deliveries":
                DELIVERIES,

            "distance_km":
                matrix,

            "isolation_multiplier":
                3.0,

            "depot_included":
                False,
        },
    )

    assert (
        response.status_code
        == 200
    )


def test_invalid_matrix_shape():
    response = client.post(
        "/api/adverse-deliveries/screen",
        json={
            "deliveries":
                DELIVERIES,

            "distance_km": [
                [
                    0.0,
                    1.0,
                ],
                [
                    1.0,
                    0.0,
                ],
            ],

            "depot_included":
                True,
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_dbscan_label_count_mismatch():
    response = client.post(
        "/api/adverse-deliveries/screen",
        json={
            "deliveries":
                DELIVERIES,

            "distance_km":
                DISTANCE_MATRIX,

            "dbscan_labels": [
                0,
                -1,
            ],

            "depot_included":
                True,
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_invalid_isolation_multiplier():
    response = client.post(
        "/api/adverse-deliveries/screen",
        json={
            "deliveries":
                DELIVERIES,

            "distance_km":
                DISTANCE_MATRIX,

            "isolation_multiplier":
                1.0,

            "depot_included":
                True,
        },
    )

    # Pydantic catches gt=1 before
    # the domain function is called.

    assert (
        response.status_code
        == 422
    )


def test_detour_requires_depot():
    response = client.post(
        "/api/adverse-deliveries/screen",
        json={
            "deliveries":
                DELIVERIES,

            "distance_km": [
                [
                    0.0,
                    1.0,
                    10.0,
                ],
                [
                    1.0,
                    0.0,
                    10.0,
                ],
                [
                    10.0,
                    10.0,
                    0.0,
                ],
            ],

            "detour_threshold_km":
                2.0,

            "depot_included":
                False,
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_duplicate_delivery_ids_rejected():
    deliveries = [
        DELIVERIES[
            0
        ],
        {
            **DELIVERIES[
                1
            ],
            "delivery_id":
                "A",
        },
    ]

    response = client.post(
        "/api/adverse-deliveries/screen",
        json={
            "deliveries":
                deliveries,

            "distance_km": [
                [
                    0.0,
                    2.0,
                    2.0,
                ],
                [
                    2.0,
                    0.0,
                    1.0,
                ],
                [
                    2.0,
                    1.0,
                    0.0,
                ],
            ],

            "depot_included":
                True,
        },
    )

    assert (
        response.status_code
        == 422
    )