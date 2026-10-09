from fastapi.testclient import (
    TestClient,
)

from api.main import (
    app,
)

from api.routers.deliveries import (
    load_deliveries,
)


client = TestClient(
    app
)


def test_list_deliveries():
    response = client.get(
        "/api/deliveries"
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
            "total"
        ]
        == 2394
    )

    assert (
        payload[
            "offset"
        ]
        == 0
    )

    assert (
        payload[
            "limit"
        ]
        == 50
    )

    assert (
        payload[
            "returned"
        ]
        == 50
    )

    assert (
        len(
            payload[
                "items"
            ]
        )
        == 50
    )


def test_delivery_pagination():
    response = client.get(
        "/api/deliveries",
        params={
            "offset": 10,
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
            "offset"
        ]
        == 10
    )

    assert (
        payload[
            "limit"
        ]
        == 5
    )

    assert (
        payload[
            "returned"
        ]
        == 5
    )


def test_delivery_limit_validation():
    response = client.get(
        "/api/deliveries",
        params={
            "limit": 1000,
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_delivery_stats():
    response = client.get(
        "/api/deliveries/stats"
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
            "deliveries"
        ]
        == 2394
    )

    assert (
        "column_names"
        in payload
    )

    assert (
        payload[
            "delivery_id_column"
        ]
        in {
            "id",
            "delivery_id",
            "deliveryId",
        }
    )


def test_get_existing_delivery():
    dataframe = (
        load_deliveries()
    )

    if (
        "id"
        in dataframe.columns
    ):
        id_column = "id"

    elif (
        "delivery_id"
        in dataframe.columns
    ):
        id_column = (
            "delivery_id"
        )

    else:
        id_column = (
            "deliveryId"
        )

    delivery_id = str(
        dataframe.iloc[
            0
        ][
            id_column
        ]
    )

    response = client.get(
        f"/api/deliveries/"
        f"{delivery_id}"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        str(
            payload[
                id_column
            ]
        )
        == delivery_id
    )


def test_get_unknown_delivery():
    response = client.get(
        "/api/deliveries/"
        "__delivery_does_not_exist__"
    )

    assert (
        response.status_code
        == 404
    )