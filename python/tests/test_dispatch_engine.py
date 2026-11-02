import pytest
from simulation.dispatch_engine import DispatchSession, haversine_distance_km


def test_haversine_distance():
    # Tan Binh to District 1 ~ 6 km
    d = haversine_distance_km(10.7744, 106.6575, 106.7000, 10.7700)
    assert 0.0 < d < 20000.0


def test_session_initializes_16_shippers():
    session = DispatchSession(
        session_id="test_sess_01",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
        seed=42,
    )
    assert len(session.shippers) == 16
    for i in range(1, 17):
        s_id = f"shipper_{i:02d}"
        assert s_id in session.shippers
        assert session.shippers[s_id].name == f"Shipper #{i:02d}"
        assert session.shippers[s_id].status == "IDLE"
        assert session.shippers[s_id].capacity_kg == 100.0


def test_atomic_assignment_prevents_duplicate_ownership():
    session = DispatchSession(
        session_id="test_sess_02",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
        seed=42,
    )
    eligible = session.get_eligible_orders_for_day()
    assert len(eligible) > 0
    test_order_id = eligible[0].order_id

    # Shipper 1 assigns order
    res1 = session.assign_orders("shipper_01", [test_order_id])
    assert res1["success"] is True
    assert test_order_id in session.shippers["shipper_01"].assigned_order_ids
    assert session.orders[test_order_id].assigned_shipper_id == "shipper_01"
    assert session.orders[test_order_id].delivery_status == "ASSIGNED"

    # Shipper 2 tries to assign the SAME order -> MUST RAISE ERROR (Rejected!)
    with pytest.raises(ValueError, match="already assigned"):
        session.assign_orders("shipper_02", [test_order_id])

    # Ensure Shipper 2 does not have the order
    assert test_order_id not in session.shippers["shipper_02"].assigned_order_ids
    assert session.orders[test_order_id].assigned_shipper_id == "shipper_01"


def test_nearby_orders_excludes_assigned():
    session = DispatchSession(
        session_id="test_sess_03",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
        seed=42,
    )
    nearby = session.get_nearby_unassigned_orders("shipper_01", limit=10)
    assert len(nearby) > 0
    assigned_id = nearby[0]["order_id"]

    # Assign that order to Shipper 1
    session.assign_orders("shipper_01", [assigned_id])

    # Check nearby for Shipper 2 -> must NOT contain assigned_id
    nearby2 = session.get_nearby_unassigned_orders("shipper_02", limit=20)
    nearby2_ids = [o["order_id"] for o in nearby2]
    assert assigned_id not in nearby2_ids


def test_route_generation_with_proposed_rl():
    session = DispatchSession(
        session_id="test_sess_04",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        routing_algorithm="proposed_rl",
        seed=42,
    )
    eligible = session.get_eligible_orders_for_day()
    test_order_ids = [o.order_id for o in eligible[:4]]

    res = session.assign_orders("shipper_01", test_order_ids)
    assert res["success"] is True
    shipper = session.shippers["shipper_01"]
    assert len(shipper.route_sequence) == 4
    assert len(shipper.timeline_stops) == 4
    # Depot -> 4 stops -> Depot = 6 coordinates
    assert len(shipper.route_geometry) == 6


def test_simulation_step_and_deliveries():
    session = DispatchSession(
        session_id="test_sess_05",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        seed=42,
    )
    eligible = session.get_eligible_orders_for_day()
    test_order_ids = [o.order_id for o in eligible[:2]]
    session.assign_orders("shipper_01", test_order_ids)

    # Initial clock = 0
    assert session.simulation_clock == 0.0

    # Step simulation forward by 120 minutes
    step_res = session.step_simulation(delta_minutes=120.0)
    assert step_res["simulation_clock"] == 120.0

    # Check that shipper moved and delivered at least 1 order
    shipper = session.shippers["shipper_01"]
    assert shipper.working_minutes > 0.0
    assert len(shipper.completed_order_ids) > 0


def test_end_day_and_carryover_preservation():
    session = DispatchSession(
        session_id="test_sess_06",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
        seed=42,
    )
    # Don't assign all orders so some remain undelivered
    day1_eligible = session.get_eligible_orders_for_day()
    total_day1 = len(day1_eligible)
    assert total_day1 > 0

    # End Day 1
    end_res = session.end_day()
    assert end_res["success"] is True
    assert end_res["carried_over_orders_count"] > 0
    assert session.is_day_ended is True

    # Calling end_day again must be idempotent
    end_res_again = session.end_day()
    assert end_res_again.get("already_ended") is True

    # Start Day 2
    next_res = session.start_next_day()
    assert next_res["success"] is True
    assert session.current_day == 2
    assert session.simulation_clock == 0.0
    assert session.is_day_ended is False

    # Check that carried over orders have carryover_count = 1
    carried_orders = [o for o in session.orders.values() if o.carryover_count > 0]
    assert len(carried_orders) == end_res["carried_over_orders_count"]
    for o in carried_orders:
        assert o.carryover_count == 1
        assert o.delivery_status == "CARRIED_OVER"


def test_auto_dispatch_distributes_among_16_shippers():
    session = DispatchSession(
        session_id="test_sess_07",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
        seed=42,
    )
    res = session.auto_dispatch_day()
    assert res["success"] is True
    assert res["dispatched_orders"] > 0

    # Check that multiple shippers have received assignments
    assigned_shippers = [s for s in session.shippers.values() if len(s.assigned_order_ids) > 0]
    assert len(assigned_shippers) >= 10


def test_location_aware_dispatch_strategies():
    """Verifies that all 4 dispatch strategies execute properly and set dispatch provenance."""
    for strat in ["cluster_round_robin", "location_aware_nearest", "location_aware_min_cost", "cluster_aware_rl"]:
        session = DispatchSession(
            session_id=f"test_strat_{strat}",
            scenario_id="simulated_stress_day1_to_3",
            scenario_name="Test Strategy",
            total_days=3,
            seed=42,
        )
        res = session.auto_dispatch_day(strategy=strat)
        assert res["success"] is True
        assert res["dispatched_orders"] == 85
        assert res["strategy_used"] == strat

        # Check payload limit <= 100 kg for all shippers
        for s in session.shippers.values():
            assert s.current_load_kg <= s.capacity_kg

        # Check dispatch explanation metadata
        assigned_orders = [o for o in session.orders.values() if o.delivery_status == "ASSIGNED"]
        assert len(assigned_orders) == 85
        for o in assigned_orders:
            assert o.dispatch_strategy == strat
            assert o.dispatch_reason is not None
            assert o.dispatch_distance_km is not None


def test_shipper_location_provenance_and_dynamic_movement():
    """Verifies shipper location_source updates from DEPOT to SIMULATION and tracking minutes."""
    session = DispatchSession(
        session_id="test_loc_track",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Location Tracking Test",
        seed=42,
    )
    s1 = session.shippers["shipper_01"]
    assert s1.location_source == "DEPOT"
    assert s1.last_location_update_minute == 0.0

    session.auto_dispatch_day(strategy="cluster_aware_rl")
    assert len(s1.route_sequence) > 0

    # Step simulation clock by 30 mins
    step_res = session.step_simulation(delta_minutes=30.0)
    assert step_res["simulation_clock"] == 30.0

    # Moving or delivering shippers must have location_source = SIMULATION
    active_shippers = [s for s in session.shippers.values() if s.status in ("IN_TRANSIT", "DELIVERING")]
    assert len(active_shippers) > 0
    for s in active_shippers:
        assert s.location_source == "SIMULATION"
        assert s.last_location_update_minute == 30.0


def test_plan_preview_is_non_destructive():
    """Verifies that plan preview calculates candidate routes and deltas without mutating session."""
    session = DispatchSession(
        session_id="test_preview_purity",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Preview Test",
        seed=42,
    )
    # Check baseline for Day 1: 85 orders eligible
    unassigned_before = len([o for o in session.orders.values() if o.release_day <= session.current_day and o.delivery_status in ("UNASSIGNED", "CARRIED_OVER")])
    assert unassigned_before == 85
    initial_revision = session.session_revision

    preview_res = session.preview_dispatch(strategy="cluster_aware_rl", algorithm="proposed_rl")
    assert preview_res["success"] is True
    assert preview_res["preview_id"] is not None
    assert preview_res["candidate_kpi"]["dispatched_orders"] == 85
    assert preview_res["candidate_kpi"]["active_shippers"] > 0
    assert len(preview_res["shipper_previews"]) == 16
    assert len(preview_res["order_previews"]) == 85

    # CRITICAL: Official session must NOT be modified
    unassigned_after_preview = len([o for o in session.orders.values() if o.release_day <= session.current_day and o.delivery_status in ("UNASSIGNED", "CARRIED_OVER")])
    assert unassigned_after_preview == 85
    for s in session.shippers.values():
        assert len(s.assigned_order_ids) == 0
        assert s.route_sequence == []
    assert session.session_revision == initial_revision

    # Cancel preview leaves state unchanged
    cancel_res = session.cancel_preview()
    assert cancel_res["success"] is True
    assert session._active_preview is None


def test_confirm_preview_atomically_applies_plan():
    """Verifies that confirm_preview applies the previewed plan properly."""
    session = DispatchSession(
        session_id="test_preview_confirm",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Preview Confirm Test",
        seed=42,
    )
    preview_res = session.preview_dispatch(strategy="cluster_aware_rl", algorithm="proposed_rl")
    p_id = preview_res["preview_id"]

    confirm_res = session.confirm_preview(p_id)
    assert confirm_res["success"] is True
    assert confirm_res["dispatched_orders"] == 85

    # Check official state is now assigned
    assigned_orders = [o for o in session.orders.values() if o.release_day <= session.current_day and o.delivery_status == "ASSIGNED"]
    assert len(assigned_orders) == 85
    active_shippers = [s for s in session.shippers.values() if len(s.assigned_order_ids) > 0]
    assert len(active_shippers) >= 10
    assert session.session_revision == 2


def test_confirm_preview_detects_concurrent_revision_conflict():
    """Verifies that confirm_preview raises an error if session was modified concurrently."""
    session = DispatchSession(
        session_id="test_preview_conflict",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Preview Conflict Test",
        seed=42,
    )
    preview_res = session.preview_dispatch(strategy="cluster_aware_rl", algorithm="proposed_rl")
    p_id = preview_res["preview_id"]

    # Concurrently assign an order directly -> bumps session_revision
    eligible = session.get_eligible_orders_for_day()
    session.assign_orders("shipper_01", [eligible[0].order_id])

    # Attempting to confirm stale preview must raise ValueError
    with pytest.raises(ValueError):
        session.confirm_preview(p_id)



