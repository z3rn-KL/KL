from fastapi.testclient import (
    TestClient,
)

from api.main import (
    app,
)


client = TestClient(
    app
)


def test_root_endpoint():
    response = client.get(
        "/"
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
            "status"
        ]
        == "running"
    )

    assert (
        payload[
            "docs"
        ]
        == "/docs"
    )

    assert (
        payload[
            "health"
        ]
        == "/api/health"
    )


def test_health_endpoint():
    response = client.get(
        "/api/health"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        "status"
        in payload
    )

    assert (
        "artifacts"
        in payload
    )

    assert (
        "ready_for_frontend"
        in payload
    )


def test_health_contains_required_artifacts():
    response = client.get(
        "/api/health"
    )

    payload = (
        response.json()
    )

    artifacts = (
        payload[
            "artifacts"
        ]
    )

    expected = {
        "dataset",
        "road_graph",
        "snapped_nodes",
        "kmeans_results",
        "dataset_benchmark",
        "final_validation",
    }

    assert (
        set(
            artifacts
        )
        == expected
    )


def test_openapi_available():
    response = client.get(
        "/openapi.json"
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
            "info"
        ][
            "title"
        ]
        == (
            "Delivery Routing Optimization API"
        )
    )