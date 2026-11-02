import pytest
from fastapi.testclient import TestClient
from api.main import app
from simulation.dispatch_engine import (
    DispatchSession,
    LOGISTICS_STAGING_BASES,
    haversine_distance_km,
)

client = TestClient(app)


def test_session_depot_model():
    session = DispatchSession(
        session_id="test_depot_01",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Depot Test",
        total_days=3,
        seed=42,
    )
    assert session.depot["depot_id"] == "depot_main"
    assert session.depot["depot_type"] == "CENTRAL_HUB"
    assert session.depot["status"] == "ACTIVE"
    assert len(session.depot["served_clusters"]) == 10
    assert abs(session.depot["lat"] - 10.7744) < 1e-4
    assert abs(session.depot["lng"] - 106.6575) < 1e-4


def test_staging_bases_and_shipper_profiles():
    session = DispatchSession(
        session_id="test_depot_02",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Staging Bases Test",
        total_days=3,
        seed=42,
    )
    assert len(LOGISTICS_STAGING_BASES) >= 5
    assert len(session.shippers) == 16

    # Verify Shipper #01
    s1 = session.shippers["shipper_01"]
    assert s1.shift_start_mode == "START_AT_DEPOT"  # Default Mode A
    assert s1.home_cluster_id == 0
    assert 4 in s1.preferred_cluster_ids
    assert s1.base_id in LOGISTICS_STAGING_BASES

    # Verify Shipper #06 (Cluster 4, Tan Binh Base)
    s6 = session.shippers["shipper_06"]
    assert s6.home_cluster_id == 4
    assert s6.base_id == "base_tb"


def test_update_shipper_service_area_and_mode_b_pickup():
    session = DispatchSession(
        session_id="test_depot_03",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Mode B Test",
        total_days=3,
        seed=42,
    )

    # Switch shipper_03 to START_AT_BASE
    updated = session.update_shipper_service_area(
        shipper_id="shipper_03",
        shift_start_mode="START_AT_BASE",
        base_id="base_q1",
        home_cluster_id=1,
        preferred_cluster_ids=[6, 0],
    )
    assert updated["shift_start_mode"] == "START_AT_BASE"
    assert updated["base_id"] == "base_q1"
    assert session.shippers["shipper_03"].start_lat == LOGISTICS_STAGING_BASES["base_q1"]["lat"]

    # Assign an order to shipper_03 and verify route has DEPOT_PICKUP
    eligible = session.get_eligible_orders_for_day()
    assert len(eligible) > 0
    order = eligible[0]
    session.assign_orders("shipper_03", [order.order_id])

    s3 = session.shippers["shipper_03"]
    assert len(s3.assigned_order_ids) == 1
    assert len(s3.timeline_stops) >= 2  # Depot pickup + delivery stop + return

    # First timeline stop must be DEPOT_PICKUP!
    first_stop = s3.timeline_stops[0]
    assert first_stop.get("stop_type") == "DEPOT_PICKUP"
    assert first_stop["order_code"] == "PICKUP-DEPOT"
    assert abs(first_stop["lat"] - session.depot["lat"]) < 1e-4
    assert abs(first_stop["lng"] - session.depot["lng"]) < 1e-4

    # Route geometry must start at Base then Depot
    assert len(s3.route_geometry) >= 3
    base_info = LOGISTICS_STAGING_BASES["base_q1"]
    assert abs(s3.route_geometry[0][0] - base_info["lng"]) < 1e-4
    assert abs(s3.route_geometry[0][1] - base_info["lat"]) < 1e-4
    assert abs(s3.route_geometry[1][0] - session.depot["lng"]) < 1e-4
    assert abs(s3.route_geometry[1][1] - session.depot["lat"]) < 1e-4


def test_depot_zone_capacity_aware_dispatch():
    session = DispatchSession(
        session_id="test_depot_04",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Depot-Aware Dispatch Test",
        total_days=3,
        seed=42,
    )

    res = session.auto_dispatch_day(strategy="depot_zone_capacity_aware", algorithm="proposed_rl")
    assert res["success"] is True
    assert res["strategy_used"] == "depot_zone_capacity_aware"
    assert res["dispatched_orders"] > 0

    # Verify no shipper exceeded 100kg capacity
    for s in session.shippers.values():
        assert s.current_load_kg <= s.capacity_kg

    # Verify assigned orders have explicit reasons
    assigned = [o for o in session.orders.values() if o.delivery_status == "ASSIGNED"]
    assert len(assigned) > 0
    for o in assigned:
        assert o.dispatch_strategy == "depot_zone_capacity_aware"
        assert "cụm" in o.dispatch_reason or "kho" in o.dispatch_reason


def test_depot_and_service_area_api_endpoints():
    # 1. Create Session
    create_res = client.post("/api/dispatch/sessions", json={
        "scenario_id": "simulated_stress_day1_to_3",
        "routing_algorithm": "proposed_rl",
        "seed": 42,
    })
    assert create_res.status_code == 200
    session_id = create_res.json()["session_id"]

    # 2. GET /depot
    depot_res = client.get(f"/api/dispatch/sessions/{session_id}/depot")
    assert depot_res.status_code == 200
    d_data = depot_res.json()
    assert d_data["linked_shippers_count"] == 16
    assert d_data["depot"]["depot_id"] == "depot_main"

    # 3. GET /bases
    bases_res = client.get(f"/api/dispatch/sessions/{session_id}/bases")
    assert bases_res.status_code == 200
    bases = bases_res.json()
    assert len(bases) >= 5

    # 4. GET /service-areas
    areas_res = client.get(f"/api/dispatch/sessions/{session_id}/service-areas")
    assert areas_res.status_code == 200
    areas = areas_res.json()
    assert len(areas) == 16
    assert areas[0]["shipper_id"] == "shipper_01"
    assert "preferred_cluster_ids" in areas[0]

    # 5. PATCH /shippers/{id}/service-area
    patch_res = client.patch(
        f"/api/dispatch/sessions/{session_id}/shippers/shipper_01/service-area",
        json={
            "shift_start_mode": "START_AT_BASE",
            "base_id": "base_tb",
            "home_cluster_id": 0,
            "preferred_cluster_ids": [4, 2],
        }
    )
    assert patch_res.status_code == 200
    updated_shipper = patch_res.json()["shipper"]
    assert updated_shipper["shift_start_mode"] == "START_AT_BASE"
    assert updated_shipper["preferred_cluster_ids"] == [4, 2]

    # 6. Preview with depot_zone_capacity_aware
    preview_res = client.post(
        f"/api/dispatch/sessions/{session_id}/preview",
        json={
            "strategy": "depot_zone_capacity_aware",
            "algorithm": "proposed_rl",
        }
    )
    assert preview_res.status_code == 200
    preview = preview_res.json()
    assert preview["strategy_used"] == "depot_zone_capacity_aware"
    assert preview["candidate_kpi"]["dispatched_orders"] > 0
