import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_benchmark_comparison_cases_api():
    res = client.get("/api/dispatch/benchmarks/comparison-cases")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "cases" in data
    cases = data["cases"]
    assert len(cases) > 0
    # Check workload 28 (RL wins)
    w28 = next(c for c in cases if c["workload_id"] == 28)
    assert w28["verdict"] == "RL_OUTPERFORMS_NN"
    assert w28["rl_vs_nn_diff_km"] < 0
    
    # Check workload 17 (NN wins)
    w17 = next(c for c in cases if c["workload_id"] == 17)
    assert w17["verdict"] == "NN_OUTPERFORMS_RL"
    assert w17["rl_vs_nn_diff_km"] > 0

def test_rl_decision_trace_api():
    # 1. Create a session
    sess_res = client.post("/api/dispatch/sessions", json={
        "scenario_id": "simulated_stress_day1_to_3",
        "routing_algorithm": "proposed_rl",
        "priority_policy": "sla_aware",
        "total_days": 3,
        "seed": 42
    })
    assert sess_res.status_code == 200
    session_id = sess_res.json()["session_id"]
    
    # 2. Call RL decision trace endpoint
    trace_res = client.post(f"/api/dispatch/sessions/{session_id}/rl-decision-trace", json={})
    assert trace_res.status_code == 200
    trace_data = trace_res.json()
    assert trace_data["success"] is True
    assert trace_data["algorithm"] == "proposed_rl"
    assert "decision_trace" in trace_data
    assert len(trace_data["decision_trace"]) > 0
    
    # Check step structure
    first_step = trace_data["decision_trace"][0]
    assert "current_node" in first_step
    assert "chosen_delivery_id" in first_step
    assert "chosen_q_score" in first_step
    assert "candidates_evaluated" in first_step
    assert len(first_step["candidates_evaluated"]) > 0
    first_cand = first_step["candidates_evaluated"][0]
    assert "features" in first_cand
    assert len(first_cand["features"]) == 8
    assert "feature_dict" in first_cand

