"""
FastAPI Router for NEXUS 16-Shipper Dispatch & Multi-Day Simulation.
"""

from typing import Any, Literal
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from simulation.dispatch_engine import dispatch_manager


def _get_session_or_404(session_id: str):
    sess = dispatch_manager.get_session(session_id)
    if sess is None:
        raise HTTPException(
            status_code=404,
            detail=f"Phiên điều phối '{session_id}' không tồn tại hoặc đã hết hạn.",
        )
    return sess



router = APIRouter(
    prefix="/api/dispatch",
    tags=["Dispatch Intelligence"],
)


class CreateSessionRequest(BaseModel):
    scenario_id: str = "simulated_stress_day1_to_3"
    routing_algorithm: Literal["nearest_neighbor", "clarke_wright", "proposed_rl"] = "proposed_rl"
    priority_policy: Literal["overdue_first", "earliest_deadline", "nearest_available", "sla_aware"] = "sla_aware"
    total_days: int = Field(default=3, ge=1, le=14)
    seed: int = 42


class AssignOrdersRequest(BaseModel):
    shipper_id: str
    order_ids: list[str] = Field(min_length=1)


class UnassignOrdersRequest(BaseModel):
    order_ids: list[str] = Field(min_length=1)


class StepClockRequest(BaseModel):
    delta_minutes: float = Field(default=15.0, ge=1.0, le=120.0)


class AutoDispatchRequest(BaseModel):
    algorithm: Literal["nearest_neighbor", "clarke_wright", "proposed_rl"] | None = None
    strategy: Literal[
        "cluster_round_robin",
        "location_aware_nearest",
        "location_aware_min_cost",
        "cluster_aware_rl",
        "depot_zone_capacity_aware",
    ] | None = None


class PreviewDispatchRequest(BaseModel):
    strategy: Literal[
        "cluster_round_robin",
        "location_aware_nearest",
        "location_aware_min_cost",
        "cluster_aware_rl",
        "depot_zone_capacity_aware",
    ] | None = "cluster_aware_rl"
    algorithm: Literal["nearest_neighbor", "clarke_wright", "proposed_rl"] | None = "proposed_rl"


class UpdateDepotRequest(BaseModel):
    name: str | None = None
    lat: float | None = None
    lng: float | None = None
    description: str | None = None


class UpdateShipperServiceAreaRequest(BaseModel):
    shift_start_mode: Literal["START_AT_DEPOT", "START_AT_BASE"] | None = None
    base_id: str | None = None
    home_cluster_id: int | None = None
    preferred_cluster_ids: list[int] | None = None


class ConfirmPreviewRequest(BaseModel):
    preview_id: str


@router.get("/capabilities")
def get_dispatch_capabilities() -> dict[str, Any]:
    """Returns supported live dispatch algorithms vs research replay algorithms."""
    return {
        "live_dispatch_algorithms": [
            {
                "id": "depot_zone_capacity_aware",
                "name": "Depot, Zone & Capacity-Aware Heuristic",
                "description": "Phân công theo kho trung tâm, trạm trực ca, cụm ưu tiên và tải trọng phương tiện",
                "type": "Logistics Dispatch Heuristic",
                "is_live_capable": True,
            },
            {
                "id": "proposed_rl",
                "name": "Proposed RL (Candidate-Scoring Deep Q-Agent)",
                "description": "Mạng nơ-ron sâu học tăng cường với 8 đặc trưng bất biến, suy luận trực tiếp từ checkpoint D_moderate_sla_best.pt",
                "type": "Reinforcement Learning",
                "is_live_capable": True,
                "has_checkpoint": True,
            },
            {
                "id": "nearest_neighbor",
                "name": "Nearest Neighbor (NN Heuristic)",
                "description": "Thuật toán tham lam cổ điển tìm điểm giao gần nhất kế tiếp",
                "type": "Classical Heuristic",
                "is_live_capable": True,
                "has_checkpoint": False,
            },
            {
                "id": "clarke_wright",
                "name": "Clarke–Wright Savings",
                "description": "Thuật toán ghép cặp tiết kiệm chi phí di chuyển từ depot",
                "type": "Classical Heuristic",
                "is_live_capable": True,
                "has_checkpoint": False,
            },
        ],
        "research_replay_algorithms": [
            {
                "id": "q_learning",
                "name": "Tabular Q-Learning",
                "reason": "Mô hình dạng bảng (Q-table) với không gian trạng thái cố định theo từng workload; không hỗ trợ suy luận đơn hàng mới.",
                "type": "Reinforcement Learning",
                "is_live_capable": False,
                "supported_workloads": [1, 29, 42, 62, 71, 90],
            },
            {
                "id": "sarsa",
                "name": "Tabular SARSA",
                "reason": "Mô hình dạng bảng on-policy với không gian trạng thái cố định theo từng workload; không hỗ trợ suy luận đơn hàng mới.",
                "type": "Reinforcement Learning",
                "is_live_capable": False,
                "supported_workloads": [1, 29, 42, 62, 71, 90],
            },
            {
                "id": "dqn",
                "name": "Vanilla DQN",
                "reason": "Không lưu checkpoint trọng số (.pt) trong repository; kết quả được đối chiếu từ dữ liệu thực nghiệm đã xuất.",
                "type": "Reinforcement Learning",
                "is_live_capable": False,
                "supported_workloads": [1],
            },
            {
                "id": "guided_dqn",
                "name": "K-Means-guided DQN",
                "reason": "Thực nghiệm kiểm định 12 workloads; không có file trọng số cho suy luận đơn động.",
                "type": "Reinforcement Learning",
                "is_live_capable": False,
                "supported_workloads": [12],
            },
        ],
        "priority_policies": [
            {"id": "sla_aware", "name": "SLA-Aware & Cluster Priority (Khuyên dùng)"},
            {"id": "overdue_first", "name": "Overdue & Carry-over First (Ưu tiên đơn tồn)"},
            {"id": "earliest_deadline", "name": "Earliest Deadline First (EDF)"},
            {"id": "nearest_available", "name": "Nearest Available Order (Khoảng cách gần nhất)"},
        ],
        "dispatch_assignment_strategies": [
            {
                "id": "cluster_aware_rl",
                "name": "Cluster-Aware Soft Affinity + Proposed RL (Khuyên dùng)",
                "description": "Kết hợp phân cụm mềm K-Means, chi phí chèn thêm tối thiểu và mạng nơ-ron Proposed RL định tuyến đơn xe.",
                "is_default": True,
            },
            {
                "id": "location_aware_min_cost",
                "name": "Location-Aware Minimum Incremental Cost",
                "description": "Tối ưu hóa vị trí thời gian thực và chi phí chèn chặng (detour cost) không áp đặt biên giới cụm cứng.",
                "is_default": False,
            },
            {
                "id": "location_aware_nearest",
                "name": "Location-Aware Nearest Feasible",
                "description": "Thuật toán tham lam gán đơn gần vị trí shipper nhất kèm ràng buộc tải trọng và cân bằng đội xe.",
                "is_default": False,
            },
            {
                "id": "cluster_round_robin",
                "name": "Cluster-First Round-Robin (Legacy Baseline)",
                "description": "Phương pháp kế thừa chia xoay vòng theo cụm cứng K-Means; dễ gây đói đơn cho cụm thưa.",
                "is_default": False,
            },
        ],
        "default_fleet_size": 16,
        "default_shift_hours": 10.0,
    }


@router.get("/scenarios")
def get_available_scenarios() -> list[dict[str, Any]]:
    """Returns available simulation scenarios with scientific data provenance."""
    return [
        {
            "scenario_id": "simulated_stress_day1_to_3",
            "name": "Kịch bản Mô phỏng 16 Shipper — 3 Ngày Vận hành",
            "description": "Mô phỏng áp lực điều phối 16 shipper trên tập đơn XeDu được phân bổ theo 3 ngày với hạn chót và đơn dồn tồn đọng.",
            "data_label": "SIMULATED SCENARIO — NOT ORIGINAL EXPERIMENTAL DATA",
            "total_days": 3,
            "orders_per_day_approx": 85,
            "total_shippers": 16,
            "reproducible_seed": 42,
        },
        {
            "scenario_id": "real_busiest_window",
            "name": "Kịch bản Chuỗi Ngày Bận rộn Nhất XeDu",
            "description": "Các đơn hàng từ cửa sổ ngày có mật độ giao cao nhất trong lịch sử XeDu TP.HCM.",
            "data_label": "REAL XEDU HISTORICAL WINDOW — 16 SHIPPERS DISPATCH",
            "total_days": 3,
            "orders_per_day_approx": 60,
            "total_shippers": 16,
            "reproducible_seed": 42,
        },
    ]


@router.post("/sessions")
def create_dispatch_session(request: CreateSessionRequest) -> dict[str, Any]:
    """Initializes a new isolated simulation session."""
    session = dispatch_manager.create_session(
        scenario_id=request.scenario_id,
        routing_algorithm=request.routing_algorithm,
        priority_policy=request.priority_policy,
        total_days=request.total_days,
        seed=request.seed,
    )
    return session.get_state()


@router.get("/sessions/{session_id}/depot")
def get_session_depot(session_id: str) -> dict[str, Any]:
    """Retrieves central depot details, connected fleet summary, and served clusters."""
    session = _get_session_or_404(session_id)
    return {
        "depot": session.depot,
        "linked_shippers_count": len(session.shippers),
        "served_clusters": getattr(session, "depot", {}).get("served_clusters", list(range(10))),
        "total_fleet_capacity_kg": sum(s.capacity_kg for s in session.shippers.values()),
        "active_routes_count": sum(1 for s in session.shippers.values() if len(s.assigned_order_ids) > 0),
    }


@router.patch("/sessions/{session_id}/depot")
def update_session_depot(session_id: str, req: UpdateDepotRequest) -> dict[str, Any]:
    """Updates central depot parameters for the current simulation session."""
    session = _get_session_or_404(session_id)
    updated = session.update_depot(
        name=req.name,
        lat=req.lat,
        lng=req.lng,
        description=req.description,
    )
    return {"success": True, "depot": updated}


@router.get("/sessions/{session_id}/bases")
def get_staging_bases(session_id: str) -> list[dict[str, Any]]:
    """Returns logistics staging operating bases for shippers."""
    from simulation.dispatch_engine import LOGISTICS_STAGING_BASES
    return list(LOGISTICS_STAGING_BASES.values())


@router.get("/sessions/{session_id}/service-areas")
def get_service_areas(session_id: str) -> list[dict[str, Any]]:
    """Returns geographic service area mapping for all shippers."""
    session = _get_session_or_404(session_id)
    result = []
    for s in session.shippers.values():
        result.append({
            "shipper_id": s.shipper_id,
            "name": s.name,
            "depot_id": s.depot_id,
            "shift_start_mode": getattr(s, "shift_start_mode", "START_AT_DEPOT"),
            "start_lat": getattr(s, "start_lat", session.depot["lat"]),
            "start_lng": getattr(s, "start_lng", session.depot["lng"]),
            "base_id": getattr(s, "base_id", "base_tb"),
            "base_name": getattr(s, "base_name", "Trạm điều phối Tân Bình"),
            "base_lat": getattr(s, "base_lat", 10.7980),
            "base_lng": getattr(s, "base_lng", 106.6530),
            "home_cluster_id": s.home_cluster_id,
            "preferred_cluster_ids": getattr(s, "preferred_cluster_ids", []),
            "capacity_kg": s.capacity_kg,
            "current_load_kg": s.current_load_kg,
            "assigned_order_count": len(s.assigned_order_ids),
            "status": s.status,
        })
    return result


@router.patch("/sessions/{session_id}/shippers/{shipper_id}/service-area")
def update_shipper_service_area(
    session_id: str,
    shipper_id: str,
    req: UpdateShipperServiceAreaRequest,
) -> dict[str, Any]:
    """Updates a shipper's operating base, start mode, and priority clusters within active session."""
    session = _get_session_or_404(session_id)
    try:
        updated_shipper = session.update_shipper_service_area(
            shipper_id=shipper_id,
            shift_start_mode=req.shift_start_mode,
            base_id=req.base_id,
            home_cluster_id=req.home_cluster_id,
            preferred_cluster_ids=req.preferred_cluster_ids,
        )
        return {"success": True, "shipper": updated_shipper}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sessions/{session_id}")
def get_session_state(session_id: str) -> dict[str, Any]:
    """Retrieves current session state."""
    session = _get_session_or_404(session_id)
    return session.get_state()


@router.get("/sessions/{session_id}/shippers")
def get_session_shippers(session_id: str) -> list[dict[str, Any]]:
    """Returns 16 shippers for this session."""
    session = _get_session_or_404(session_id)
    state = session.get_state()
    return state["shippers"]


@router.get("/sessions/{session_id}/shippers/{shipper_id}/nearby-orders")
def get_nearby_orders(
    session_id: str,
    shipper_id: str,
    limit: int = Query(default=15, ge=1, le=50),
) -> list[dict[str, Any]]:
    """Returns nearby eligible unassigned orders for a specific shipper."""
    session = _get_session_or_404(session_id)
    return session.get_nearby_unassigned_orders(shipper_id=shipper_id, limit=limit)


@router.post("/sessions/{session_id}/assign")
def assign_orders(
    session_id: str,
    request: AssignOrdersRequest,
) -> dict[str, Any]:
    """
    Atomically assigns orders to a shipper.
    Rejects with 409 Conflict if any order is already assigned.
    """
    session = _get_session_or_404(session_id)
    try:
        return session.assign_orders(
            shipper_id=request.shipper_id,
            order_ids=request.order_ids,
        )
    except ValueError as err:
        raise HTTPException(status_code=409, detail=str(err))


@router.post("/sessions/{session_id}/unassign")
def unassign_orders(
    session_id: str,
    request: UnassignOrdersRequest,
) -> dict[str, Any]:
    """Unassigns orders from a shipper."""
    session = _get_session_or_404(session_id)
    try:
        return session.unassign_orders(order_ids=request.order_ids)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


@router.post("/sessions/{session_id}/auto-dispatch")
def auto_dispatch(
    session_id: str,
    request: AutoDispatchRequest | None = None,
) -> dict[str, Any]:
    """Dispatches all unassigned orders today across 16 shippers."""
    session = _get_session_or_404(session_id)
    algo = request.algorithm if request else None
    strat = request.strategy if request else None
    return session.auto_dispatch_day(algorithm=algo, strategy=strat)


@router.post("/sessions/{session_id}/preview")
@router.post("/sessions/{session_id}/preview-dispatch")
def preview_dispatch(
    session_id: str,
    request: PreviewDispatchRequest | None = None,
) -> dict[str, Any]:
    """
    Computes non-destructive plan preview and comparison without mutating session state.
    """
    session = _get_session_or_404(session_id)
    strat = request.strategy if request else "cluster_aware_rl"
    algo = request.algorithm if request else "proposed_rl"
    return session.preview_dispatch(strategy=strat, algorithm=algo)


@router.post("/sessions/{session_id}/confirm-preview")
def confirm_preview(
    session_id: str,
    request: ConfirmPreviewRequest,
) -> dict[str, Any]:
    """
    Atomically applies an active preview plan to the official session.
    Fails with 409 Conflict if preview expired or session was modified concurrently.
    """
    session = _get_session_or_404(session_id)
    try:
        return session.confirm_preview(preview_id=request.preview_id)
    except ValueError as err:
        raise HTTPException(status_code=409, detail=str(err))


@router.post("/sessions/{session_id}/cancel-preview")
def cancel_preview(session_id: str) -> dict[str, Any]:
    """Cancels active preview and releases candidate plan."""
    session = _get_session_or_404(session_id)
    return session.cancel_preview()


@router.post("/sessions/{session_id}/clock/step")
def step_clock(
    session_id: str,
    request: StepClockRequest,
) -> dict[str, Any]:
    """Advances the simulation clock by delta_minutes."""
    session = _get_session_or_404(session_id)
    return session.step_simulation(delta_minutes=request.delta_minutes)


@router.post("/sessions/{session_id}/end-day")
def end_day(session_id: str) -> dict[str, Any]:
    """Closes workday, transitions undelivered orders to CARRIED_OVER, and records metrics."""
    session = _get_session_or_404(session_id)
    return session.end_day()


@router.post("/sessions/{session_id}/next-day")
def next_day(session_id: str) -> dict[str, Any]:
    """Advances to next day, merges carry-over backlog with new orders, resets shippers to depot."""
    session = _get_session_or_404(session_id)
    return session.start_next_day()


@router.get("/sessions/{session_id}/metrics")
def get_metrics(session_id: str) -> dict[str, Any]:
    """Returns operational and scientific simulation metrics."""
    session = _get_session_or_404(session_id)
    state = session.get_state()
    return {
        "session_id": session_id,
        "current_day": state["current_day"],
        "total_days": state["total_days"],
        "daily_metrics": state["daily_metrics"],
        "fleet_summary": state["fleet_summary"],
        "order_summary": state["order_summary"],
    }

class RlDecisionTraceRequest(BaseModel):
    shipper_id: str | None = None
    order_ids: list[str] | None = None


@router.post("/sessions/{session_id}/rl-decision-trace")
def get_rl_decision_trace(session_id: str, req: RlDecisionTraceRequest | None = None) -> dict[str, Any]:
    """
    Explainable RL Decision Explorer:
    Generates step-by-step decision traces with candidate Q-scores and 8 invariant state features
    directly inferred from the locked PyTorch model checkpoint.
    """
    session = _get_session_or_404(session_id)
    target_order_ids: list[str] = []
    resolved_shipper_id = None

    if req and req.order_ids:
        target_order_ids = req.order_ids
    elif req and req.shipper_id:
        shipper = session.shippers.get(req.shipper_id)
        if not shipper:
            raise HTTPException(status_code=404, detail=f"Shipper '{req.shipper_id}' không tồn tại.")
        resolved_shipper_id = shipper.shipper_id
        target_order_ids = list(shipper.assigned_order_ids)
        if not target_order_ids and shipper.completed_order_ids:
            target_order_ids = list(shipper.completed_order_ids)
        if not target_order_ids and shipper.timeline_stops:
            target_order_ids = [
                st["order_id"] for st in shipper.timeline_stops
                if st.get("order_id") and st.get("order_id") not in ("depot_pickup", "DEPOT_PICKUP")
            ]
        # If requested shipper still has no orders, fallback to first active shipper with routes
        if not target_order_ids:
            for s in session.shippers.values():
                candidates = s.assigned_order_ids or s.completed_order_ids or [
                    st["order_id"] for st in s.timeline_stops
                    if st.get("order_id") and st.get("order_id") not in ("depot_pickup", "DEPOT_PICKUP")
                ]
                if candidates:
                    target_order_ids = list(candidates)
                    resolved_shipper_id = s.shipper_id
                    break
    else:
        active_shippers = [s for s in session.shippers.values() if s.assigned_order_ids or s.completed_order_ids]
        if active_shippers:
            target_order_ids = list(active_shippers[0].assigned_order_ids or active_shippers[0].completed_order_ids)
            resolved_shipper_id = active_shippers[0].shipper_id
        else:
            unassigned = [o.order_id for o in session.orders.values() if o.release_day <= session.current_day]
            target_order_ids = unassigned[:5]

    active_shippers_summary = [
        {
            "shipper_id": s.shipper_id,
            "name": s.name,
            "order_count": len(s.assigned_order_ids or s.completed_order_ids),
        }
        for s in session.shippers.values()
    ]

    if not target_order_ids:
        return {
            "success": False,
            "message": "Chưa có đơn hàng nào được phân công để giải thích tuyến RL. Hãy xác nhận kế hoạch tại Bàn lập kế hoạch.",
            "ordered_ids": [],
            "decision_trace": [],
            "active_shippers": active_shippers_summary,
        }

    orders = [session.orders[oid] for oid in target_order_ids if oid in session.orders]
    if not orders:
        return {
            "success": False,
            "message": "Không có thông tin chi tiết của đơn hàng.",
            "ordered_ids": [],
            "decision_trace": [],
        }

    n = len(orders)
    coords = [(session.depot["lat"], session.depot["lng"])] + [(o.latitude, o.longitude) for o in orders]
    import numpy as np
    from simulation.dispatch_engine import haversine_distance_km
    from routing.proposed_rl_solver import solve_proposed_rl_route_with_trace, LOCKED_OBJECTIVE

    dist_matrix = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                dist_matrix[i, j] = haversine_distance_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
    time_matrix = dist_matrix / session.average_speed_kmh * 60.0
    deadlines = [o.deadline_minutes for o in orders]
    clusters = [o.cluster_id for o in orders]
    order_ids = [o.order_id for o in orders]

    ordered_ids, trace = solve_proposed_rl_route_with_trace(
        delivery_ids=order_ids,
        distance_matrix_km=dist_matrix,
        travel_time_matrix_minutes=time_matrix,
        deadlines_minutes=deadlines,
        cluster_ids=clusters,
    )

    order_map = {o.order_id: o for o in orders}
    for step in trace:
        for c in step["candidates_evaluated"]:
            cand_o = order_map.get(c["delivery_id"])
            if cand_o:
                c["package_name"] = cand_o.package_name
                c["order_code"] = cand_o.order_code
                c["cluster_id"] = cand_o.cluster_id
                c["weight_kg"] = cand_o.weight_kg
                c["lat"] = cand_o.latitude
                c["lng"] = cand_o.longitude

    return {
        "success": True,
        "session_id": session_id,
        "shipper_id": resolved_shipper_id or (req.shipper_id if req else None),
        "active_shippers": active_shippers_summary,
        "total_stops": len(ordered_ids),
        "ordered_ids": ordered_ids,
        "decision_trace": trace,
        "objective_weights": {
            "distance": LOCKED_OBJECTIVE.distance_weight,
            "travel_time": LOCKED_OBJECTIVE.travel_time_weight,
            "late_delivery": LOCKED_OBJECTIVE.late_delivery_weight,
            "lateness": LOCKED_OBJECTIVE.lateness_weight,
            "cluster_switch": LOCKED_OBJECTIVE.cluster_switch_weight,
        },
        "algorithm": "proposed_rl",
        "model_type": "CandidateScoringDQN (8 invariant features)",
        "checkpoint": "results/proposed_rl_multiworkload_D_moderate_sla_best.pt",
    }


@router.get("/benchmarks/comparison-cases")
def get_benchmark_comparison_cases() -> dict[str, Any]:
    """
    Returns authentic empirical benchmark comparison cases from core_algorithm_all_results.csv,
    highlighting both where RL outperforms Nearest Neighbor and where Nearest Neighbor wins.
    """
    import pandas as pd
    import os

    results_path = os.path.join("results", "core_algorithm_all_results.csv")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Benchmark CSV file not found on disk.")
    df = pd.read_csv(results_path)

    target_wids = [5, 12, 17, 28, 30]
    sub = df[df["workload_id"].isin(target_wids)]

    cases = []
    for wid in target_wids:
        w_rows = sub[sub["workload_id"] == wid]
        row_dict = {}
        for _, r in w_rows.iterrows():
            algo = r["algorithm"]
            row_dict[algo] = {
                "distance_km": round(float(r["total_distance_km"]), 2),
                "time_minutes": round(float(r["total_travel_time_minutes"]), 1),
                "on_time_rate": float(r["on_time_rate"]),
                "deliveries_count": int(r["number_of_deliveries"]),
            }
        diff_vs_nn = round(row_dict.get("dqn", {}).get("distance_km", 0) - row_dict.get("nearest_neighbor", {}).get("distance_km", 0), 2)
        pct_vs_nn = round((diff_vs_nn / max(0.01, row_dict.get("nearest_neighbor", {}).get("distance_km", 1.0))) * 100, 2)
        cases.append({
            "workload_id": wid,
            "deliveries_count": int(w_rows.iloc[0]["number_of_deliveries"]) if not w_rows.empty else 0,
            "results": row_dict,
            "rl_vs_nn_diff_km": diff_vs_nn,
            "rl_vs_nn_diff_pct": pct_vs_nn,
            "verdict": "RL_OUTPERFORMS_NN" if diff_vs_nn < -0.05 else ("NN_OUTPERFORMS_RL" if diff_vs_nn > 0.05 else "NEAR_EQUIVALENT"),
        })

    return {
        "success": True,
        "source_file": "results/core_algorithm_all_results.csv",
        "cohort": "31 Core Workloads from Thesis Benchmark",
        "cases": cases,
    }
