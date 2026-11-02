from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_dispatch_capabilities():
    response = client.get("/api/dispatch/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "live_dispatch_algorithms" in data
    assert "research_replay_algorithms" in data
    algo_ids = [a["id"] for a in data["live_dispatch_algorithms"]]
    assert "proposed_rl" in algo_ids
    assert "nearest_neighbor" in algo_ids
    assert "clarke_wright" in algo_ids


def test_api_dispatch_scenarios():
    response = client.get("/api/dispatch/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    assert any("SIMULATED SCENARIO" in s["data_label"] for s in data)


def test_api_dispatch_lifecycle():
    # 1. Create session
    create_res = client.post(
        "/api/dispatch/sessions",
        json={
            "scenario_id": "simulated_stress_day1_to_3",
            "routing_algorithm": "proposed_rl",
            "priority_policy": "sla_aware",
            "total_days": 3,
            "seed": 42,
        },
    )
    assert create_res.status_code == 200
    session_data = create_res.json()
    session_id = session_data["session_id"]
    assert session_data["current_day"] == 1
    assert len(session_data["shippers"]) == 16

    # 2. Query shippers
    shippers_res = client.get(f"/api/dispatch/sessions/{session_id}/shippers")
    assert shippers_res.status_code == 200
    assert len(shippers_res.json()) == 16

    # 3. Query nearby orders for Shipper #01
    nearby_res = client.get(f"/api/dispatch/sessions/{session_id}/shippers/shipper_01/nearby-orders?limit=5")
    assert nearby_res.status_code == 200
    nearby = nearby_res.json()
    assert len(nearby) > 0
    test_order_id = nearby[0]["order_id"]

    # 4. Assign order to Shipper #01
    assign_res = client.post(
        f"/api/dispatch/sessions/{session_id}/assign",
        json={"shipper_id": "shipper_01", "order_ids": [test_order_id]},
    )
    assert assign_res.status_code == 200
    assert assign_res.json()["assigned_count"] == 1

    # 5. Attempt duplicate assignment to Shipper #02 -> MUST FAIL with 409
    dup_res = client.post(
        f"/api/dispatch/sessions/{session_id}/assign",
        json={"shipper_id": "shipper_02", "order_ids": [test_order_id]},
    )
    assert dup_res.status_code == 409

    # 6. Step simulation clock
    step_res = client.post(
        f"/api/dispatch/sessions/{session_id}/clock/step",
        json={"delta_minutes": 30.0},
    )
    assert step_res.status_code == 200
    assert step_res.json()["simulation_clock"] == 30.0

    # 7. End day
    end_res = client.post(f"/api/dispatch/sessions/{session_id}/end-day")
    assert end_res.status_code == 200
    assert end_res.json()["success"] is True

    # 8. Start next day
    next_res = client.post(f"/api/dispatch/sessions/{session_id}/next-day")
    assert next_res.status_code == 200
    assert next_res.json()["day"] == 2
    assert next_res.json()["simulation_clock"] == 0.0

    # 9. Get metrics
    metrics_res = client.get(f"/api/dispatch/sessions/{session_id}/metrics")
    assert metrics_res.status_code == 200
    assert len(metrics_res.json()["daily_metrics"]) == 1


def test_api_dispatch_preview_and_confirm_workflow():
    # Create session
    create_res = client.post(
        "/api/dispatch/sessions",
        json={
            "scenario_id": "simulated_stress_day1_to_3",
            "routing_algorithm": "proposed_rl",
            "priority_policy": "sla_aware",
            "total_days": 3,
            "seed": 42,
        },
    )
    assert create_res.status_code == 200
    session_id = create_res.json()["session_id"]

    # Preview
    preview_res = client.post(
        f"/api/dispatch/sessions/{session_id}/preview",
        json={"strategy": "cluster_aware_rl", "algorithm": "proposed_rl"},
    )
    assert preview_res.status_code == 200
    p_data = preview_res.json()
    assert p_data["success"] is True
    assert p_data["preview_id"] is not None
    assert p_data["candidate_kpi"]["dispatched_orders"] == 85
    assert len(p_data["shipper_previews"]) == 16

    # Verify session is still unassigned
    state_res = client.get(f"/api/dispatch/sessions/{session_id}")
    assert state_res.status_code == 200
    assert state_res.json()["order_summary"]["assigned"] == 0

    # Confirm preview
    confirm_res = client.post(
        f"/api/dispatch/sessions/{session_id}/confirm-preview",
        json={"preview_id": p_data["preview_id"]},
    )
    assert confirm_res.status_code == 200
    assert confirm_res.json()["dispatched_orders"] == 85

    # Verify session now has assigned orders
    state_after = client.get(f"/api/dispatch/sessions/{session_id}")
    assert state_after.json()["order_summary"]["assigned"] == 85


