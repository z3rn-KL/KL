"""
Regression test suite for NEXUS Dispatch Engine 8-Issue Correction.
Verifies:
- Issue 1: Deadlock in start_next_day()
- Issue 2: Deadlock in DispatchManager.get_session()
- Issue 3: KeyError in _dispatch_location_aware_min_cost()
- Issue 4: Multi-day deadline & SLA preservation
- Issue 5: Haversine distance model labeling
- Issue 6: Preview baseline and SLA integrity (no hardcoded 1.0 SLA)
- Issue 7: Capacity constraints and overloaded shipper fallback prevention
- Issue 8: Defined logger and RL fallback traceability
"""

import threading
import time
import pytest
from unittest.mock import patch

from simulation.dispatch_engine import (
    DispatchSession,
    DispatchManager,
    OrderItem,
    haversine_distance_km,
)


def run_with_timeout(func, timeout_seconds=4.0):
    """Runs a function in a background thread and asserts it completes before timeout."""
    result = {"completed": False, "res": None, "err": None}

    def worker():
        try:
            result["res"] = func()
            result["completed"] = True
        except Exception as e:
            result["err"] = e
            result["completed"] = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if not result["completed"]:
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s (Deadlock detected!)")
    if result["err"]:
        raise result["err"]
    return result["res"]


def test_issue1_no_deadlock_start_next_day():
    """Issue 1: Calling start_next_day() before end_day() must not deadlock."""
    session = DispatchSession(
        session_id="test_i1",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
    )
    assert session.current_day == 1
    assert not session.is_day_ended

    # In original code, start_next_day() acquires self._lock then calls end_day()
    # which acquires self._lock, causing an immediate deadlock!
    def call_start_next_day():
        return session.start_next_day()

    res = run_with_timeout(call_start_next_day, timeout_seconds=3.0)
    assert res["success"] is True
    assert session.current_day == 2
    assert not session.is_day_ended
    assert session.simulation_clock == 0.0


def test_issue2_no_deadlock_get_session_nonexistent():
    """Issue 2: get_session() with nonexistent ID must not deadlock or silently create a session."""
    dm = DispatchManager()

    def call_get_nonexistent():
        return dm.get_session("totally_bogus_session_id")

    # In original code, get_session() acquires self._lock then calls create_session()
    # which acquires self._lock, causing deadlock!
    res = run_with_timeout(call_get_nonexistent, timeout_seconds=3.0)
    assert res is None, "Nonexistent session ID must return None, not create a new one silently"


def test_issue3_no_keyerror_with_preassigned_orders():
    """Issue 3: _dispatch_location_aware_min_cost must not raise KeyError when shippers have pre-assigned orders."""
    session = DispatchSession(
        session_id="test_i3",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
    )
    # Manually pre-assign order 0 to shipper_01
    day1_unassigned = [o for o in session.orders.values() if o.release_day == 1 and o.delivery_status == "UNASSIGNED"]
    assert len(day1_unassigned) >= 5

    first_order = day1_unassigned[0]
    session.assign_orders(shipper_id="shipper_01", order_ids=[first_order.order_id])

    # Now auto-dispatch remaining orders using location_aware_min_cost
    # In original code, coords calculation accesses order_dict[oid] for oid in cur_assigned,
    # raising KeyError because first_order is not in the remaining unassigned orders!
    res = session.auto_dispatch_day(algorithm="nearest_neighbor", strategy="location_aware_min_cost")
    assert res["success"] is True
    assert res["dispatched_orders"] > 0


def test_issue4_multiday_deadline_and_sla_preservation():
    """Issue 4: Multi-day deadline must be preserved and carried-over orders must not reset to ON_TIME."""
    session = DispatchSession(
        session_id="test_i4",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
    )
    # Pick an order on day 1 with deadline = 180 min
    o = OrderItem(
        order_id="test_ord_sla_1",
        latitude=10.77,
        longitude=106.65,
        weight_kg=2.0,
        cluster_id=0,
        release_day=1,
        created_at="Day 1 08:00:00",
        deadline_minutes=180.0,
        delivery_status="UNASSIGNED",
    )
    session.orders["test_ord_sla_1"] = o

    # Clock advances to 200m on Day 1 -> Order is overdue
    session.step_simulation(200.0)
    o.update_sla_status(session.simulation_clock, session.current_day)
    assert o.sla_status == "OVERDUE"

    # End Day 1 and start Day 2
    session.end_day()
    assert o.delivery_status == "CARRIED_OVER"
    assert o.carryover_count == 1

    session.start_next_day()
    assert session.current_day == 2
    assert session.simulation_clock == 0.0

    # On Day 2 at minute 10, this carried-over order must STILL be OVERDUE!
    # It must NOT reset to ON_TIME because its deadline was on Day 1!
    o.update_sla_status(session.simulation_clock, session.current_day)
    assert o.sla_status == "OVERDUE", "Carried-over order past original deadline must remain OVERDUE on Day 2"


def test_issue5_haversine_labeling():
    """Issue 5: Dispatch session must explicitly state Haversine distance model and speed."""
    session = DispatchSession(
        session_id="test_i5",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
    )
    state = session.get_state()
    assert state.get("distance_model") == "haversine"
    assert "25" in str(state.get("speed_assumption", ""))


def test_issue6_preview_baseline_and_sla_integrity():
    """Issue 6: Preview baseline SLA must not be hardcoded 1.0 when no baseline exists."""
    session = DispatchSession(
        session_id="test_i6",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
    )
    # No orders assigned yet
    preview = session.preview_dispatch(strategy="cluster_aware_rl", algorithm="nearest_neighbor")
    assert preview["success"] is True

    # When no baseline orders exist, baseline_kpi should be None or have sla_on_time_rate as None
    # and deltas should not claim an arbitrary SLA baseline of 1.0!
    baseline = preview.get("baseline_kpi")
    if baseline is not None:
        assert baseline.get("sla_on_time_rate") is None, "Baseline SLA rate must not be hardcoded to 1.0"


def test_issue7_capacity_constraints_strictly_enforced():
    """Issue 7: Orders must not be assigned to shippers exceeding payload capacity."""
    session = DispatchSession(
        session_id="test_i7",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
    )
    # Fill all 16 shippers to full capacity
    for s in session.shippers.values():
        s.current_load_kg = s.capacity_kg

    # Add a new heavy order of 10kg
    heavy_order = OrderItem(
        order_id="heavy_overflow_01",
        latitude=10.77,
        longitude=106.65,
        weight_kg=10.0,
        cluster_id=0,
        release_day=1,
        created_at="Day 1 08:00:00",
        deadline_minutes=300.0,
        delivery_status="UNASSIGNED",
    )
    session.orders["heavy_overflow_01"] = heavy_order

    res = session.auto_dispatch_day(algorithm="nearest_neighbor", strategy="location_aware_min_cost")
    # heavy order must NOT be assigned because all shippers are full!
    assert session.orders["heavy_overflow_01"].delivery_status == "UNASSIGNED"
    assert "tải trọng" in (session.orders["heavy_overflow_01"].dispatch_reason or "").lower()


def test_issue8_logger_and_rl_fallback():
    """Issue 8: When Proposed RL throws an exception, logger must not fail with NameError and fallback must be logged."""
    session = DispatchSession(
        session_id="test_i8",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Scenario",
        total_days=3,
    )
    day1_unassigned = [o for o in session.orders.values() if o.release_day == 1 and o.delivery_status == "UNASSIGNED"]
    assert len(day1_unassigned) >= 2

    # Mock solve_proposed_rl_route to raise an unexpected exception
    with patch("simulation.dispatch_engine.solve_proposed_rl_route", side_effect=RuntimeError("Simulated RL inference CUDA failure")):
        # Must gracefully fall back to Nearest Neighbor without raising NameError for undefined logger!
        res = session._solve_route_for_orders(
            assigned_orders=day1_unassigned[:3],
            algo="proposed_rl",
            start_lat=10.7744,
            start_lng=106.6575,
        )
        assert res["algorithm_used"] == "nearest_neighbor_fallback"
        assert len(res["route_sequence"]) == 3
