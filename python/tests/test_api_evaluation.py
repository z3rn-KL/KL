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


def test_custom_cost_evaluation():
    response = client.post(
        "/api/evaluation/cost",
        json={
            "total_distance_km":
                10.0,

            "total_travel_time_minutes":
                20.0,

            "late_deliveries":
                1,

            "total_lateness_minutes":
                4.0,

            "cost_config": {
                "distance_cost_per_km":
                    2.0,

                "travel_time_cost_per_minute":
                    0.5,

                "late_delivery_penalty":
                    10.0,

                "lateness_cost_per_minute":
                    0.25,

                "cost_unit":
                    "test_unit",
            },
        },
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "result"
        ][
            "total_estimated_cost"
        ]
        == 41.0
    )


def test_default_cost_evaluation():
    response = client.post(
        "/api/evaluation/cost",
        json={
            "total_distance_km":
                1.0,

            "total_travel_time_minutes":
                1.0,
        },
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "result"
        ][
            "cost_unit"
        ]
        == "VND_proxy_scenario"
    )


def test_route_evaluation_real_graph():
    ids = (
        first_two_real_delivery_ids()
    )

    response = client.post(
        "/api/evaluation/route",
        json={
            "delivery_ids":
                ids,

            "delivery_order":
                ids,

            "algorithm_label":
                "api_test_route",
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
            "evaluation"
        ][
            "number_of_deliveries"
        ]
        == 2
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


def test_route_order_mismatch_rejected():
    ids = (
        first_two_real_delivery_ids()
    )

    response = client.post(
        "/api/evaluation/route",
        json={
            "delivery_ids":
                ids,

            "delivery_order": [
                ids[
                    0
                ],
                "__unknown__",
            ],
        },
    )

    assert (
        response.status_code
        == 422
    )