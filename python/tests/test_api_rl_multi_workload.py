from fastapi.testclient import TestClient
import pandas as pd
import pytest

from api.main import app
import api.dependencies as deps

client = TestClient(app)

TARGET_WORKLOADS = [1, 29, 42, 62, 71, 90]
TARGET_ALGOS = ["q_learning", "sarsa"]


def test_precomputed_catalog_contains_expanded_workloads():
    response = client.get("/api/routing/precomputed")
    assert response.status_code == 200
    data = response.json()

    for algo in TARGET_ALGOS:
        available_ids = data[algo]["available_workload_ids"]
        for wid in TARGET_WORKLOADS:
            assert wid in available_ids, f"{algo} catalog missing WL #{wid}"
        assert "workload_provenance" in data[algo]


@pytest.mark.parametrize("algo", TARGET_ALGOS)
@pytest.mark.parametrize("wid", TARGET_WORKLOADS)
def test_rl_precomputed_all_target_workloads(algo, wid):
    response = client.get(
        f"/api/routing/precomputed/{algo}",
        params={"workload_id": wid, "include_geometry": False},
    )
    assert response.status_code == 200, f"Failed for {algo} on WL #{wid}: {response.text}"
    payload = response.json()

    assert payload["algorithm"] == algo
    assert payload["workload_id"] == wid
    assert payload["online_training"] is False

    delivery_order = payload["delivery_order"]
    assert len(delivery_order) > 0

    # Verify no duplicates
    assert len(set(delivery_order)) == len(delivery_order)

    # Cross-check with workload definition
    wl_row = deps.get_workload_row(wid)
    expected_ids = set(deps.parse_manifest_delivery_ids(wl_row["delivery_ids"]))
    assert set(delivery_order) == expected_ids
    assert len(delivery_order) == int(wl_row["number_of_deliveries"])

    # Verify route_type and provenance
    assert "route_type" in payload
    assert "provenance" in payload
    assert "data_status" in payload

    if wid == 1:
        assert payload["route_type"] == "original_precomputed"
    else:
        assert payload["route_type"] == "validated_recovered"

    # Verify reference benchmark metrics match
    bench_df = pd.read_csv("results/dataset_algorithm_all_results.csv")
    matched = bench_df[(bench_df["workload_id"] == wid) & (bench_df["algorithm"] == algo)]
    assert not matched.empty
    ref_row = matched.iloc[0]
    ref_metrics = payload["reference_metrics"]
    assert abs(ref_metrics["total_distance_km"] - float(ref_row["total_distance_km"])) < 1e-4
    assert abs(ref_metrics["total_travel_time_minutes"] - float(ref_row["total_travel_time_minutes"])) < 1e-4


@pytest.mark.parametrize("algo", TARGET_ALGOS)
def test_unsupported_workload_returns_404(algo):
    # WL 999 does not exist
    res = client.get(
        f"/api/routing/precomputed/{algo}",
        params={"workload_id": 999, "include_geometry": False},
    )
    assert res.status_code == 404
    assert "Supported workloads" in res.json()["detail"]

    # WL 5 is in benchmark but not in verified Q-Learning/SARSA expansion targets
    res5 = client.get(
        f"/api/routing/precomputed/{algo}",
        params={"workload_id": 5, "include_geometry": False},
    )
    assert res5.status_code == 404
    assert "Supported workloads" in res5.json()["detail"]


def test_unsupported_algorithm_returns_422():
    res = client.get(
        "/api/routing/precomputed/unsupported_algo",
        params={"workload_id": 1, "include_geometry": False},
    )
    assert res.status_code == 422


def test_wl29_cross_source_provenance_explicit():
    for algo in TARGET_ALGOS:
        res = client.get(
            f"/api/routing/precomputed/{algo}",
            params={"workload_id": 29, "include_geometry": False},
        )
        assert res.status_code == 200
        payload = res.json()
        assert "Route Recovery" in payload["provenance"]
        assert "Reproducibility" in payload["provenance"]
        assert payload["source"] == "proposed_rl_baseline_route_recovery"
        assert payload["data_status"] == "verified_exact"


def test_rl_route_with_road_geometry_for_wl42():
    # Test full road network geometry resolution for Q-Learning on WL #42
    res = client.get(
        "/api/routing/precomputed/q_learning",
        params={"workload_id": 42, "include_geometry": True},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["evaluation"] is not None
    ev = payload["evaluation"]
    assert len(ev["route_geometry"]) > 10
    assert ev["total_distance_km"] > 0
    assert ev["returned_to_depot"] is True
    assert ev["number_of_deliveries"] == 6

