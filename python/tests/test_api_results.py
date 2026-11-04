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

def test_daily_planning_summary_filters():
    response = client.get(
        "/api/results/daily-planning/summary",
        params={
            "load_multiplier": 12,
            "overtime_policy": "hybrid",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert len(payload["rows"]) == 2
    assert all(
        row["overtime_policy"] == "hybrid"
        for row in payload["rows"]
    )


def test_daily_planning_best_within_cap_respects_cap():
    payload = client.get(
        "/api/results/daily-planning/summary"
    ).json()

    assert len(payload["best_within_cap"]) == 6
    assert all(
        row["over_cap_shipper_days"] == 0
        for row in payload["best_within_cap"]
    )


def test_daily_planning_daily_requires_valid_scenario():
    ok = client.get(
        "/api/results/daily-planning/daily",
        params={
            "load_multiplier": 4,
            "allocation": "dedicated",
            "overtime_policy": "defer",
        },
    )

    assert ok.status_code == 200
    assert len(ok.json()["days"]) == 30

    bad = client.get(
        "/api/results/daily-planning/daily",
        params={
            "load_multiplier": 4,
            "allocation": "nope",
            "overtime_policy": "defer",
        },
    )

    assert bad.status_code == 422


def test_daily_planning_sensitivity_unknown_parameter():
    response = client.get(
        "/api/results/daily-planning/sensitivity",
        params={"parameter": "nope"},
    )

    assert response.status_code == 404


def test_proposed_rl_final_test():
    payload = client.get(
        "/api/results/proposed-rl/final-test"
    ).json()

    assert payload["summary"]
    assert payload["pairwise_vs_baselines"]


def test_real_data_summary():
    payload = client.get("/api/results/real-data").json()

    assert 0.0 < payload["observed"]["late_rate"] < 1.0
    assert payload["time_model"]["pairs"] > 0
    assert payload["time_model"]["real_median_minutes"] > (
        payload["time_model"]["model_median_minutes"]
    )

    count = payload["clean_shifts"]["count"]

    for unit in ("time", "km"):
        methods = {row["method"]: row for row in payload["replay"][unit]}

        assert set(methods) == {
            "real",
            "nearest",
            "nearest_search",
            "optimal",
        }
        assert methods["optimal"]["at_optimum"] == count
        assert methods["optimal"]["mean_excess_pct"] == 0.0
        assert methods["real"]["mean"] >= methods["optimal"]["mean"]


def test_shift_rl_summary():
    payload = client.get("/api/results/shift-rl").json()

    assert payload["shifts"] > 0
    assert payload["groups"][0]["group"] == "tất cả"

    first = [m["key"] for m in payload["groups"][0]["methods"]]

    assert all(
        [m["key"] for m in g["methods"]] == first for g in payload["groups"]
    )
    assert "nn_all_ls" in first
    assert payload["paired_vs"]["nn_all_ls"]["wins"] >= 0



def test_rl_decision_trace_explorer():
    response = client.get("/api/results/rl-decision-trace?shipper_id=shipper_08&day=1&seed=42")
    assert response.status_code == 200
    payload = response.json()

    assert payload["shipper_id"] == "shipper_08"
    assert payload["day"] == 1
    assert payload["assigned_orders_count"] > 0
    assert payload["depot"]["lat"] == 10.7744
    assert len(payload["feature_names"]) == 8
    assert payload["model_architecture"]["name"] == "CandidateScoringDQN"
    assert payload["model_architecture"]["input_dim"] == 8

    # 3 algorithm comparison routes
    assert "proposed_rl" in payload["routes"]
    assert "nearest_neighbor" in payload["routes"]
    assert "clarke_wright" in payload["routes"]
    assert len(payload["routes"]["proposed_rl"]) == payload["assigned_orders_count"]
    assert len(payload["routes"]["nearest_neighbor"]) == payload["assigned_orders_count"]
    assert len(payload["routes"]["clarke_wright"]) == payload["assigned_orders_count"]

    # Metrics
    for algo in ("proposed_rl", "nearest_neighbor", "clarke_wright"):
        m = payload["metrics"][algo]
        assert m["distance_km"] > 0
        assert m["total_time_minutes"] > 0
        assert 0.0 <= m["on_time_rate"] <= 100.0

    # Divergence
    assert "diverges_from_nn" in payload["divergence"]
    assert "diverges_from_cw" in payload["divergence"]


def test_training_history_all_algorithms():
    for algo in ("proposed_rl", "dqn", "sarsa", "q_learning"):
        response = client.get(f"/api/results/training-history?algorithm={algo}&sample_rate=50")
        assert response.status_code == 200
        payload = response.json()

        assert payload["algorithm"] == algo
        assert payload["name"]
        assert payload["total_episodes"] > 0
        assert len(payload["datapoints"]) > 0
        assert payload["hyperparameters"]
        assert payload["training_vs_inference"]
        assert payload["scientific_interpretation"]

        if algo == "proposed_rl":
            assert payload["total_epochs"] == 40
            # Epoch 30 is best checkpoint
            ep30 = next(p for p in payload["datapoints"] if p["epoch"] == 30)
            assert ep30["is_best_checkpoint"] is True
            assert ep30["is_global_best"] is True
            assert ep30["validation_ran"] is True
            assert ep30["validation_weighted_sla_pct"] == 98.55
            assert ep30["validation_late_deliveries"] == 2
        else:
            # Baseline single-workload
            assert payload["total_episodes"] == 5000
            last_pt = payload["datapoints"][-1]
            assert last_pt["episode"] == 5000 or last_pt["episode"] >= 4950
            assert "distance_ma50" in last_pt
            assert "reward_ma50" in last_pt
