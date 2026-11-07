import pytest
from simulation.dispatch_engine import DispatchSession

def test_preview_delta_orders_dispatched_calculation():
    """
    Verifies that Preview Delta for orders dispatched computes:
    Candidate Total Assigned - Baseline Total Assigned (e.g. 85 - 2 = 83),
    resolving the double-subtraction defect where Delta was reported as +81.
    """
    session = DispatchSession(
        session_id="test_delta_fix",
        scenario_id="simulated_stress_day1_to_3",
        scenario_name="Test Delta Fix",
        total_days=3,
        seed=42,
    )
    
    # 85 unassigned orders initially
    unassigned_ids = [o.order_id for o in session.orders.values() if o.release_day <= 1]
    assert len(unassigned_ids) == 85
    
    # Assign 2 orders manually to shipper_01
    two_orders = unassigned_ids[:2]
    res_assign = session.assign_orders("shipper_01", two_orders)
    assert res_assign["assigned_count"] == 2
    assert session.shippers["shipper_01"].assigned_order_ids == two_orders
    
    # Run preview
    preview = session.preview_dispatch(strategy="cluster_aware_rl", algorithm="proposed_rl")
    
    assert preview["success"] is True
    baseline_kpi = preview["baseline_kpi"]
    deltas = preview["deltas"]
    candidate_kpi = preview["candidate_kpi"]
    
    assert baseline_kpi is not None
    assert baseline_kpi["assigned_orders"] == 2
    assert candidate_kpi["dispatched_orders"] == 83 # 83 newly dispatched from unassigned
    assert candidate_kpi["total_assigned_orders"] == 85 # 85 total candidate assigned
    
    # CRITICAL CHECK: delta_orders_dispatched must be 83 (85 - 2), NOT 81!
    assert deltas["delta_orders_dispatched"] == 83

