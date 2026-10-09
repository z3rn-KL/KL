from fastapi.testclient import (
    TestClient,
)

import api.dependencies as deps

from api.main import (
    app,
)


client = TestClient(
    app
)


def first_two_real_delivery_ids(
) -> list[str]:
    row = (
        deps.get_workload_row(
            1
        )
    )

    return (
        deps
        .parse_manifest_delivery_ids(
            row[
                "delivery_ids"
            ]
        )[
            :2
        ]
    )


def test_precomputed_catalog():
    response = client.get(
        "/api/routing/precomputed"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        1
        in payload[
            "dqn"
        ][
            "available_workload_ids"
        ]
    )


def test_q_learning_precomputed_without_geometry():
    response = client.get(
        "/api/routing/precomputed/q_learning",
        params={
            "include_geometry":
                False,
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
            "workload_id"
        ]
        == 1
    )

    assert (
        payload[
            "online_training"
        ]
        is False
    )

    assert (
        len(
            payload[
                "delivery_order"
            ]
        )
        == 5
    )


def test_guided_dqn_precomputed_without_geometry():
    response = client.get(
        "/api/routing/precomputed/kmeans_guided_dqn",
        params={
            "workload_id":
                29,

            "include_geometry":
                False,
        },
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "workload_id"
        ]
        == 29
    )


def test_guided_dqn_requires_workload_id():
    response = client.get(
        "/api/routing/precomputed/kmeans_guided_dqn",
        params={
            "include_geometry":
                False,
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_clarke_wright_rejects_travel_time_metric():
    ids = (
        first_two_real_delivery_ids()
    )

    response = client.post(
        "/api/routing/optimize",
        json={
            "algorithm":
                "clarke_wright",

            "delivery_ids":
                ids,

            "metric":
                "travel_time",
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_live_nearest_neighbor_real_graph():
    ids = (
        first_two_real_delivery_ids()
    )

    response = client.post(
        "/api/routing/optimize",
        json={
            "algorithm":
                "nearest_neighbor",

            "delivery_ids":
                ids,

            "metric":
                "distance",
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
        == "nearest_neighbor"
    )

    assert (
        payload[
            "evaluation"
        ][
            "number_of_deliveries"
        ]
        == 2
    )

    assert (
        payload[
            "evaluation"
        ][
            "total_distance_km"
        ]
        > 0
    )

    assert (
        len(
            payload[
                "evaluation"
            ][
                "route_geometry"
            ]
        )
        > 1
    )


def test_live_clarke_wright_real_graph():
    ids = (
        first_two_real_delivery_ids()
    )

    response = client.post(
        "/api/routing/optimize",
        json={
            "algorithm":
                "clarke_wright",

            "delivery_ids":
                ids,

            "metric":
                "distance",
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
        == "clarke_wright"
    )

    assert (
        payload[
            "evaluation"
        ][
            "number_of_deliveries"
        ]
        == 2
    )