from fastapi.testclient import (
    TestClient,
)

from api.main import (
    app,
)


client = TestClient(
    app
)


def test_results_overview():
    response = client.get(
        "/api/results/overview"
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
            "benchmark"
        ][
            "workloads"
        ]
        == 108
    )

    assert (
        payload[
            "benchmark"
        ][
            "deliveries"
        ]
        == 813
    )

    assert (
        payload[
            "benchmark"
        ][
            "algorithm_runs"
        ]
        == 540
    )


def test_benchmark_summary():
    response = client.get(
        "/api/results/benchmark/summary"
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "rows"
        ]
        == 5
    )


def test_benchmark_runs_filter():
    response = client.get(
        "/api/results/benchmark/runs",
        params={
            "algorithm":
                "dqn",

            "limit":
                2,
        },
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "returned"
        ]
        == 2
    )


def test_workload_manifest_api():
    response = client.get(
        "/api/results/workloads",
        params={
            "limit":
                3,
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
            "total"
        ]
        == 108
    )

    assert (
        len(
            payload[
                "items"
            ][
                0
            ][
                "delivery_ids"
            ]
        )
        >= 5
    )


def test_sensitivity_results():
    response = client.get(
        "/api/results/sensitivity"
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "selected_lambda"
        ]
        == 0.5
    )


def test_final_validation_results():
    response = client.get(
        "/api/results/final-validation"
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
            "summary"
        ][
            "workloads"
        ]
        == 12
    )

    assert (
        payload[
            "summary"
        ][
            "deliveries"
        ]
        == 108
    )