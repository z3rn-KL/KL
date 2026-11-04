"""
NEXUS Delivery Intelligence - Dispatch & Simulation Engine.

Manages 16 shippers across multi-day simulation scenarios with:
- Spatial clustering (K-Means K=10)
- Atomic order assignment (zero duplicate assignments)
- Multi-algorithm route generation (Nearest Neighbor, Clarke-Wright, Proposed RL)
- Unified simulation clock and concurrent shipper state machine
- Workday shift limit, overtime policy, and end-of-day carry-over backlog
- Strict scientific traceability and reproducible seeds
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from clustering.geo_projection import LocalGeoProjection
from routing.nearest_neighbor import NearestNeighborRouter
from routing.clarke_wright import ClarkeWrightRouter
from routing.proposed_rl_solver import solve_proposed_rl_route

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "dataset" / "xedu_cleaned.csv"
CLUSTERS_PATH = PROJECT_ROOT / "results" / "xedu_kmeans_clusters.csv"
DEPOT_COORDS = {"lat": 10.7744, "lng": 106.6575}  # Tan Binh Central Depot
DEFAULT_AVERAGE_SPEED_KMH = 25.0  # Urban delivery speed in HCMC


def solve_matrix_nearest_neighbor(delivery_ids: Sequence[str], dist_matrix: np.ndarray) -> list[str]:
    """Greedy Nearest Neighbor sequencing using distance matrix."""
    n = len(delivery_ids)
    if n <= 1:
        return list(delivery_ids)
    unvisited = set(range(1, n + 1))
    current = 0
    order = []
    while unvisited:
        next_node = min(unvisited, key=lambda j: dist_matrix[current, j])
        order.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return [delivery_ids[i - 1] for i in order]


def solve_matrix_clarke_wright(delivery_ids: Sequence[str], dist_matrix: np.ndarray) -> list[str]:
    """Clarke-Wright savings heuristic sequencing using distance matrix."""
    n = len(delivery_ids)
    if n <= 1:
        return list(delivery_ids)
    savings = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
                savings.append((s, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)

    routes = {i: [i] for i in range(1, n + 1)}
    route_of = {i: i for i in range(1, n + 1)}

    for _, i, j in savings:
        r_i = route_of[i]
        r_j = route_of[j]
        if r_i != r_j:
            if routes[r_i][-1] == i and routes[r_j][0] == j:
                new_route = routes[r_i] + routes[r_j]
                routes[r_i] = new_route
                for node in routes[r_j]:
                    route_of[node] = r_i
                del routes[r_j]

    remaining_routes = list(routes.values())
    remaining_routes.sort(key=lambda r: dist_matrix[0, r[0]])
    final_nodes = [node for r in remaining_routes for node in r]
    return [delivery_ids[i - 1] for i in final_nodes]


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two geographic coordinates in kilometers."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return float(2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a))))


# Shipper home cluster assignment (16 shippers covering 10 clusters)
# Clusters with high density (4, 6, 0, 9, 7) have 2-3 shippers assigned
# Logistics Staging Operating Bases in Ho Chi Minh City (Trạm giao nhận / điểm trực ca)
LOGISTICS_STAGING_BASES = {
    "base_tb": {
        "base_id": "base_tb",
        "name": "Trạm điều phối Tân Bình (Cụm 0, 4)",
        "lat": 10.7980,
        "lng": 106.6530,
        "district": "Tân Bình",
        "primary_clusters": [0, 4],
    },
    "base_q1": {
        "base_id": "base_q1",
        "name": "Trạm điều phối Bến Thành (Cụm 1, 6)",
        "lat": 10.7725,
        "lng": 106.6980,
        "district": "Quận 1",
        "primary_clusters": [1, 6],
    },
    "base_bt": {
        "base_id": "base_bt",
        "name": "Trạm điều phối Thị Nghè (Cụm 2, 7)",
        "lat": 10.8010,
        "lng": 106.7110,
        "district": "Bình Thạnh",
        "primary_clusters": [2, 7],
    },
    "base_q10": {
        "base_id": "base_q10",
        "name": "Trạm điều phối Quận 10 (Cụm 3, 5)",
        "lat": 10.7670,
        "lng": 106.6620,
        "district": "Quận 10",
        "primary_clusters": [3, 5],
    },
    "base_q5": {
        "base_id": "base_q5",
        "name": "Trạm điều phối Chợ Lớn (Cụm 8, 9)",
        "lat": 10.7550,
        "lng": 106.6600,
        "district": "Quận 5",
        "primary_clusters": [8, 9],
    },
    "base_td": {
        "base_id": "base_td",
        "name": "Trạm điều phối TP Thủ Đức (Cụm 7)",
        "lat": 10.8490,
        "lng": 106.7720,
        "district": "TP Thủ Đức",
        "primary_clusters": [7],
    },
}

# Shipper profiles with staging base and secondary preferred clusters
SHIPPER_SERVICE_PROFILES = {
    "shipper_01": {"home_cluster": 0, "preferred_clusters": [4, 3], "base_id": "base_tb"},
    "shipper_02": {"home_cluster": 0, "preferred_clusters": [4, 1], "base_id": "base_tb"},
    "shipper_03": {"home_cluster": 1, "preferred_clusters": [6, 0], "base_id": "base_q1"},
    "shipper_04": {"home_cluster": 2, "preferred_clusters": [7, 1], "base_id": "base_bt"},
    "shipper_05": {"home_cluster": 3, "preferred_clusters": [5, 0], "base_id": "base_q10"},
    "shipper_06": {"home_cluster": 4, "preferred_clusters": [0, 8], "base_id": "base_tb"},
    "shipper_07": {"home_cluster": 4, "preferred_clusters": [0, 2], "base_id": "base_tb"},
    "shipper_08": {"home_cluster": 4, "preferred_clusters": [0, 9], "base_id": "base_tb"},
    "shipper_09": {"home_cluster": 5, "preferred_clusters": [3, 8], "base_id": "base_q10"},
    "shipper_10": {"home_cluster": 6, "preferred_clusters": [1, 2], "base_id": "base_q1"},
    "shipper_11": {"home_cluster": 6, "preferred_clusters": [1, 7], "base_id": "base_q1"},
    "shipper_12": {"home_cluster": 7, "preferred_clusters": [2, 6], "base_id": "base_td"},
    "shipper_13": {"home_cluster": 7, "preferred_clusters": [2, 1], "base_id": "base_bt"},
    "shipper_14": {"home_cluster": 8, "preferred_clusters": [9, 5], "base_id": "base_q5"},
    "shipper_15": {"home_cluster": 9, "preferred_clusters": [8, 4], "base_id": "base_q5"},
    "shipper_16": {"home_cluster": 9, "preferred_clusters": [8, 0], "base_id": "base_q5"},
}

SHIPPER_HOME_CLUSTERS = {
    "shipper_01": 0,
    "shipper_02": 0,
    "shipper_03": 1,
    "shipper_04": 2,
    "shipper_05": 3,
    "shipper_06": 4,
    "shipper_07": 4,
    "shipper_08": 4,
    "shipper_09": 5,
    "shipper_10": 6,
    "shipper_11": 6,
    "shipper_12": 7,
    "shipper_13": 7,
    "shipper_14": 8,
    "shipper_15": 9,
    "shipper_16": 9,
}


@dataclass
class OrderItem:
    order_id: str
    latitude: float
    longitude: float
    weight_kg: float
    cluster_id: int
    release_day: int
    created_at: str
    deadline_minutes: float | None
    service_duration_minutes: float = 10.0
    assigned_shipper_id: str | None = None
    delivery_status: Literal[
        "UNASSIGNED",
        "ASSIGNED",
        "IN_TRANSIT",
        "DELIVERED",
        "CARRIED_OVER",
        "FAILED",
    ] = "UNASSIGNED"
    sla_status: Literal["ON_TIME", "AT_RISK", "OVERDUE"] = "ON_TIME"
    carryover_count: int = 0
    assigned_at_minute: float | None = None
    delivered_at_minute: float | None = None
    delivered_day: int | None = None
    is_simulated_scenario: bool = True
    order_code: str = ""
    package_name: str = ""
    dispatch_reason: str | None = None
    dispatch_distance_km: float | None = None
    dispatch_strategy: str | None = None

    def update_sla_status(self, clock_minutes: float, current_day: int = 1) -> None:
        if self.deadline_minutes is None:
            self.sla_status = "ON_TIME"
            return

        if self.delivered_at_minute is not None:
            deliv_day = self.delivered_day if self.delivered_day is not None else self.release_day
            if deliv_day > self.release_day:
                self.sla_status = "OVERDUE"
            elif self.delivered_at_minute <= self.deadline_minutes:
                self.sla_status = "ON_TIME"
            else:
                self.sla_status = "OVERDUE"
            return

        if current_day > self.release_day:
            self.sla_status = "OVERDUE"
            return

        slack = self.deadline_minutes - clock_minutes
        if slack < 0.0:
            self.sla_status = "OVERDUE"
        elif slack <= 45.0:
            self.sla_status = "AT_RISK"
        else:
            self.sla_status = "ON_TIME"


@dataclass
class ShipperStop:
    order_id: str
    lat: float
    lng: float
    arrival_minute: float
    departure_minute: float
    weight_kg: float
    is_depot: bool = False


@dataclass
class ShipperItem:
    shipper_id: str
    name: str
    depot_id: str
    home_cluster_id: int
    current_lat: float
    current_lng: float
    status: Literal[
        "IDLE",
        "IN_TRANSIT",
        "DELIVERING",
        "RETURNING",
        "SHIFT_ENDED",
    ] = "IDLE"
    assigned_order_ids: list[str] = field(default_factory=list)
    completed_order_ids: list[str] = field(default_factory=list)
    current_order_id: str | None = None
    capacity_kg: float = 100.0
    current_load_kg: float = 0.0
    distance_travelled_km: float = 0.0
    total_distance_km: float = 0.0
    working_minutes: float = 0.0
    overtime_minutes: float = 0.0
    route_sequence: list[str] = field(default_factory=list)
    route_geometry: list[list[float]] = field(default_factory=list)  # [[lng, lat], ...]
    timeline_stops: list[dict[str, Any]] = field(default_factory=list)
    location_source: Literal["DEPOT", "BASE", "SIMULATION", "REAL_GPS", "ESTIMATED"] = "DEPOT"
    last_location_update_minute: float = 0.0

    # Enhanced geographic location and service base
    shift_start_mode: Literal["START_AT_DEPOT", "START_AT_BASE"] = "START_AT_DEPOT"
    start_lat: float = 10.7744
    start_lng: float = 106.6575
    base_id: str = "base_tb"
    base_name: str = "Trạm điều phối Tân Bình"
    base_lat: float = 10.7980
    base_lng: float = 106.6530
    preferred_cluster_ids: list[int] = field(default_factory=list)

    @property
    def remaining_capacity_kg(self) -> float:
        return max(0.0, self.capacity_kg - self.current_load_kg)

    def can_accept_order(self, order: "OrderItem") -> bool:
        return (self.current_load_kg + order.weight_kg) <= self.capacity_kg


@dataclass
class SimulationEvent:
    event_id: str
    timestamp_minute: float
    day: int
    event_type: str  # "ASSIGN", "UNASSIGN", "START_DELIVERY", "DELIVERED", "END_DAY", "NEXT_DAY"
    shipper_id: str | None
    order_id: str | None
    description: str


class DispatchSession:
    """
    Isolated multi-shipper multi-day simulation session.
    Guarantees thread-safe atomic assignment and simulation progression.
    """

    def __init__(
        self,
        session_id: str,
        scenario_id: str,
        scenario_name: str,
        total_days: int = 3,
        routing_algorithm: Literal["nearest_neighbor", "clarke_wright", "proposed_rl"] = "proposed_rl",
        priority_policy: Literal["overdue_first", "earliest_deadline", "nearest_available", "sla_aware"] = "sla_aware",
        seed: int = 42,
    ) -> None:
        self.session_id = session_id
        self.scenario_id = scenario_id
        self.scenario_name = scenario_name
        self.total_days = total_days
        self.current_day = 1
        self.routing_algorithm = routing_algorithm
        self.priority_policy = priority_policy
        self.seed = seed

        self.shift_minutes = 600.0  # 10 hours
        self.max_overtime_minutes = 120.0  # 2 hours
        self.service_minutes = 10.0
        self.average_speed_kmh = DEFAULT_AVERAGE_SPEED_KMH

        self.simulation_clock = 0.0  # Minutes from 0..600
        self.is_running = False
        self.speed_multiplier = 1.0
        self.is_day_ended = False

        self.depot = {
            "depot_id": "depot_main",
            "name": "Kho trung tâm Tân Bình (Depot)",
            "lat": DEPOT_COORDS["lat"],
            "lng": DEPOT_COORDS["lng"],
            "depot_type": "CENTRAL_HUB",
            "status": "ACTIVE",
            "description": "Kho điều phối hàng hóa trung tâm Tân Bình, TP.HCM (Mô phỏng)",
            "served_clusters": list(range(10)),
        }

        self.shippers: dict[str, ShipperItem] = {}
        self.orders: dict[str, OrderItem] = {}
        self.daily_metrics: list[dict[str, Any]] = []
        self.event_history: list[SimulationEvent] = []
        self.session_revision: int = 1
        self._active_preview: dict[str, Any] | None = None

        self._lock = threading.Lock()
        self._initialize_shippers()
        self._load_scenario_orders()

    def update_depot(
        self,
        name: str | None = None,
        lat: float | None = None,
        lng: float | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Update session depot configuration in simulation mode."""
        with self._lock:
            if name is not None:
                self.depot["name"] = name
            if lat is not None:
                self.depot["lat"] = lat
            if lng is not None:
                self.depot["lng"] = lng
            if description is not None:
                self.depot["description"] = description
            self.session_revision += 1
            return dict(self.depot)

    def update_shipper_service_area(
        self,
        shipper_id: str,
        shift_start_mode: Literal["START_AT_DEPOT", "START_AT_BASE"] | None = None,
        base_id: str | None = None,
        home_cluster_id: int | None = None,
        preferred_cluster_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Updates shipper service area, base, and start mode within this session."""
        with self._lock:
            if shipper_id not in self.shippers:
                raise ValueError(f"Shipper '{shipper_id}' không tồn tại trong phiên.")
            s = self.shippers[shipper_id]
            if base_id is not None and base_id in LOGISTICS_STAGING_BASES:
                base_info = LOGISTICS_STAGING_BASES[base_id]
                s.base_id = base_info["base_id"]
                s.base_name = base_info["name"]
                s.base_lat = base_info["lat"]
                s.base_lng = base_info["lng"]
            if home_cluster_id is not None:
                s.home_cluster_id = home_cluster_id
            if preferred_cluster_ids is not None:
                s.preferred_cluster_ids = list(preferred_cluster_ids)
            if shift_start_mode is not None:
                s.shift_start_mode = shift_start_mode
                if shift_start_mode == "START_AT_BASE":
                    s.start_lat = s.base_lat
                    s.start_lng = s.base_lng
                    if s.status == "IDLE":
                        s.current_lat = s.base_lat
                        s.current_lng = s.base_lng
                        s.location_source = "BASE"
                else:  # START_AT_DEPOT
                    s.start_lat = self.depot["lat"]
                    s.start_lng = self.depot["lng"]
                    if s.status == "IDLE":
                        s.current_lat = self.depot["lat"]
                        s.current_lng = self.depot["lng"]
                        s.location_source = "DEPOT"

            if s.assigned_order_ids:
                self._recalculate_shipper_route(shipper_id)

            self.session_revision += 1
            return asdict(s)

    def _initialize_shippers(self) -> None:
        """Create 16 deterministic shippers with stable IDs #01 to #16, staging bases, and preferred service areas."""
        for i in range(1, 17):
            s_id = f"shipper_{i:02d}"
            name = f"Shipper #{i:02d}"
            prof = SHIPPER_SERVICE_PROFILES.get(
                s_id,
                {"home_cluster": (i - 1) % 10, "preferred_clusters": [], "base_id": "base_tb"}
            )
            base_info = LOGISTICS_STAGING_BASES.get(prof["base_id"], LOGISTICS_STAGING_BASES["base_tb"])
            self.shippers[s_id] = ShipperItem(
                shipper_id=s_id,
                name=name,
                depot_id=self.depot["depot_id"],
                home_cluster_id=prof["home_cluster"],
                current_lat=self.depot["lat"],
                current_lng=self.depot["lng"],
                status="IDLE",
                capacity_kg=100.0,
                shift_start_mode="START_AT_DEPOT",
                start_lat=self.depot["lat"],
                start_lng=self.depot["lng"],
                base_id=base_info["base_id"],
                base_name=base_info["name"],
                base_lat=base_info["lat"],
                base_lng=base_info["lng"],
                preferred_cluster_ids=list(prof["preferred_clusters"]),
                location_source="DEPOT",
            )

    def _load_scenario_orders(self) -> None:
        """Loads and constructs scenario orders from dataset with fixed random seed."""
        cluster_df = pd.read_csv(CLUSTERS_PATH)
        rng = np.random.RandomState(self.seed)

        # Distribute orders across days (e.g. 80-100 orders per day)
        # Sample representative orders across all 10 clusters
        all_indices = np.arange(len(cluster_df))
        rng.shuffle(all_indices)

        orders_per_day = 85
        total_needed = min(len(cluster_df), orders_per_day * self.total_days)

        for day in range(1, self.total_days + 1):
            day_indices = all_indices[(day - 1) * orders_per_day : day * orders_per_day]
            for idx in day_indices:
                row = cluster_df.iloc[idx]
                order_id = str(row["id"])
                order_code = str(row.get("mdh", "")) if pd.notna(row.get("mdh")) else order_id[-6:]
                package_name = str(row.get("package_name", "Kiện hàng tiêu chuẩn")) if pd.notna(row.get("package_name")) else "Kiện hàng"
                lat = float(row["receiverLat"])
                lng = float(row["receiverLng"])
                weight = float(row.get("weight", 2.5))
                if weight <= 0 or math.isnan(weight):
                    weight = 2.5
                cluster_id = int(row["cluster_id"])

                # Realistic delivery deadline: between 120 and 540 minutes from workday start
                raw_dl = rng.choice([180.0, 240.0, 300.0, 360.0, 480.0, 540.0, None], p=[0.15, 0.2, 0.2, 0.2, 0.15, 0.05, 0.05])
                deadline = float(raw_dl) if raw_dl is not None else None

                self.orders[order_id] = OrderItem(
                    order_id=order_id,
                    latitude=lat,
                    longitude=lng,
                    weight_kg=weight,
                    cluster_id=cluster_id,
                    release_day=day,
                    created_at=f"Day {day} 08:00:00",
                    deadline_minutes=deadline,
                    service_duration_minutes=self.service_minutes,
                    delivery_status="UNASSIGNED",
                    sla_status="ON_TIME",
                    carryover_count=0,
                    is_simulated_scenario=True,
                    order_code=order_code,
                    package_name=package_name,
                )

    def get_eligible_orders_for_day(self) -> list[OrderItem]:
        """Returns orders available for delivery today (released today or carried over)."""
        return [
            o for o in self.orders.values()
            if o.release_day <= self.current_day
            and o.delivery_status in ("UNASSIGNED", "CARRIED_OVER", "ASSIGNED", "IN_TRANSIT")
        ]

    def get_nearby_unassigned_orders(
        self,
        shipper_id: str,
        limit: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Returns unassigned orders eligible for current day sorted by proximity
        and policy score to the shipper's location.
        """
        with self._lock:
            shipper = self.shippers.get(shipper_id)
            if not shipper:
                return []

            unassigned = [
                o for o in self.orders.values()
                if o.release_day <= self.current_day
                and o.delivery_status in ("UNASSIGNED", "CARRIED_OVER")
            ]

            results = []
            for o in unassigned:
                dist = haversine_distance_km(
                    shipper.current_lat, shipper.current_lng,
                    o.latitude, o.longitude
                )
                travel_time_min = (dist / self.average_speed_kmh) * 60.0
                est_arrival = self.simulation_clock + travel_time_min

                is_overdue = False
                slack = None
                if o.deadline_minutes is not None:
                    slack = o.deadline_minutes - est_arrival
                    if slack < 0:
                        is_overdue = True

                # Priority score (higher = should be assigned first)
                priority_score = 0.0
                reasons: list[str] = []

                if o.carryover_count > 0:
                    pts = o.carryover_count * 100.0
                    priority_score += pts
                    reasons.append(f"Tồn đọng {o.carryover_count} ngày (+{pts:.0f}đ)")

                if is_overdue:
                    priority_score += 200.0
                    reasons.append("Quá hạn SLA (+200đ)")
                elif slack is not None and slack < 45.0:
                    pts = round(100.0 - slack, 1)
                    priority_score += pts
                    reasons.append(f"Hạn gấp còn {slack:.0f}p (+{pts:.0f}đ)")

                # Cluster affinity bonus
                if o.cluster_id == shipper.home_cluster_id:
                    priority_score += 30.0
                    reasons.append(f"Cùng cụm #{o.cluster_id} (+30đ)")

                # Distance penalty
                dist_penalty = round(dist * 2.0, 1)
                priority_score -= dist_penalty
                reasons.append(f"Đường chim bay {dist:.2f}km (-{dist_penalty}đ)")

                # Recommendation summary label
                if o.carryover_count > 0:
                    rec_reason = f"Ưu tiên đơn tồn {o.carryover_count} ngày • " + ("Cùng cụm" if o.cluster_id == shipper.home_cluster_id else f"{dist:.1f}km")
                elif is_overdue or (slack is not None and slack < 45.0):
                    rec_reason = "Ưu tiên gấp SLA • " + ("Cùng cụm" if o.cluster_id == shipper.home_cluster_id else f"{dist:.1f}km")
                elif o.cluster_id == shipper.home_cluster_id:
                    rec_reason = f"Cùng cụm phụ trách #{o.cluster_id} • {dist:.2f}km chim bay"
                else:
                    rec_reason = f"Đơn khả dụng • {dist:.2f}km chim bay"

                results.append({
                    "order_id": o.order_id,
                    "order_code": o.order_code or o.order_id[-6:],
                    "package_name": o.package_name or "Kiện hàng",
                    "latitude": o.latitude,
                    "longitude": o.longitude,
                    "weight_kg": o.weight_kg,
                    "cluster_id": o.cluster_id,
                    "cluster_match": o.cluster_id == shipper.home_cluster_id,
                    "distance_km": round(dist, 2),
                    "distance_type": "haversine",
                    "travel_time_minutes": round(travel_time_min, 1),
                    "travel_time_type": "estimated_speed_25kmh",
                    "deadline_minutes": o.deadline_minutes,
                    "slack_minutes": round(slack, 1) if slack is not None else None,
                    "is_overdue": is_overdue,
                    "carryover_count": o.carryover_count,
                    "priority_score": round(priority_score, 2),
                    "delivery_status": o.delivery_status,
                    "sla_status": o.sla_status,
                    "capacity_feasible": (shipper.current_load_kg + o.weight_kg) <= shipper.capacity_kg,
                    "recommendation_reason": rec_reason,
                    "recommendation_breakdown": reasons,
                    "policy_name": "SLA-Aware & Cluster Spatial Priority",
                })

            # Sort descending by priority score
            results.sort(key=lambda x: x["priority_score"], reverse=True)
            return results[:limit]

    def assign_orders(
        self,
        shipper_id: str,
        order_ids: Sequence[str],
    ) -> dict[str, Any]:
        """
        Atomic assignment: Assigns orders to shipper.
        Guarantees NO double-assignment. If ANY order is already assigned, rejects.
        """
        with self._lock:
            shipper = self.shippers.get(shipper_id)
            if not shipper:
                raise ValueError(f"Shipper {shipper_id} not found.")

            # Validate atomicity
            for o_id in order_ids:
                order = self.orders.get(o_id)
                if not order:
                    raise ValueError(f"Order {o_id} not found.")
                if order.delivery_status in ("ASSIGNED", "IN_TRANSIT", "DELIVERED"):
                    if order.assigned_shipper_id != shipper_id:
                        raise ValueError(
                            f"Order {o_id} is already assigned to {order.assigned_shipper_id}. Assignment rejected."
                        )

            # Apply assignment
            assigned_now = []
            for o_id in order_ids:
                order = self.orders[o_id]
                if order.assigned_shipper_id != shipper_id:
                    order.assigned_shipper_id = shipper_id
                    order.delivery_status = "ASSIGNED"
                    order.assigned_at_minute = self.simulation_clock
                    if o_id not in shipper.assigned_order_ids:
                        shipper.assigned_order_ids.append(o_id)
                    assigned_now.append(o_id)

                    self.event_history.append(
                        SimulationEvent(
                            event_id=str(uuid.uuid4())[:8],
                            timestamp_minute=self.simulation_clock,
                            day=self.current_day,
                            event_type="ASSIGN",
                            shipper_id=shipper_id,
                            order_id=o_id,
                            description=f"{shipper.name} assigned Order #{o_id[:6]} (Cluster {order.cluster_id})",
                        )
                    )

            # Recalculate route sequence and geometry for shipper
            self._recalculate_shipper_route(shipper_id)
            self.session_revision += 1
            self._active_preview = None
            return {
                "success": True,
                "shipper_id": shipper_id,
                "assigned_count": len(assigned_now),
                "total_assigned": len(shipper.assigned_order_ids),
                "route_sequence": shipper.route_sequence,
            }

    def unassign_orders(
        self,
        order_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Unassigns orders if they have not yet been completed."""
        with self._lock:
            affected_shippers = set()
            for o_id in order_ids:
                order = self.orders.get(o_id)
                if not order:
                    continue
                if order.delivery_status in ("IN_TRANSIT", "DELIVERED"):
                    raise ValueError(f"Cannot unassign order {o_id} because it is {order.delivery_status}.")

                s_id = order.assigned_shipper_id
                if s_id and s_id in self.shippers:
                    affected_shippers.add(s_id)
                    if o_id in self.shippers[s_id].assigned_order_ids:
                        self.shippers[s_id].assigned_order_ids.remove(o_id)

                order.assigned_shipper_id = None
                order.delivery_status = "CARRIED_OVER" if order.carryover_count > 0 else "UNASSIGNED"
                order.assigned_at_minute = None

                self.event_history.append(
                    SimulationEvent(
                        event_id=str(uuid.uuid4())[:8],
                        timestamp_minute=self.simulation_clock,
                        day=self.current_day,
                        event_type="UNASSIGN",
                        shipper_id=s_id,
                        order_id=o_id,
                        description=f"Order #{o_id[:6]} unassigned",
                    )
                )

            for s_id in affected_shippers:
                self._recalculate_shipper_route(s_id)

            self.session_revision += 1
            self._active_preview = None
            return {"success": True, "unassigned_count": len(order_ids)}

    def auto_dispatch_day(
        self,
        algorithm: Literal["nearest_neighbor", "clarke_wright", "proposed_rl"] | None = None,
        strategy: Literal[
            "cluster_round_robin",
            "location_aware_nearest",
            "location_aware_min_cost",
            "cluster_aware_rl",
            "depot_zone_capacity_aware",
        ] | None = None,
    ) -> dict[str, Any]:
        """
        Dispatches all eligible unassigned orders for today across the 16 shippers
        using configurable assignment strategies and route optimization algorithms.
        """
        with self._lock:
            algo = algorithm or self.routing_algorithm
            strat = strategy or "cluster_aware_rl"
            eligible_unassigned = [
                o for o in self.orders.values()
                if o.release_day <= self.current_day
                and o.delivery_status in ("UNASSIGNED", "CARRIED_OVER")
            ]

            if not eligible_unassigned:
                return {
                    "success": True,
                    "dispatched_orders": 0,
                    "message": "No unassigned orders to dispatch.",
                    "strategy_used": strat,
                    "algorithm_used": algo,
                }

            if strat == "cluster_round_robin":
                assigned_count = self._dispatch_cluster_round_robin(eligible_unassigned)
            elif strat == "location_aware_nearest":
                assigned_count = self._dispatch_location_aware_nearest(eligible_unassigned)
            elif strat == "location_aware_min_cost":
                assigned_count = self._dispatch_location_aware_min_cost(eligible_unassigned, soft_cluster=False)
            elif strat == "depot_zone_capacity_aware":
                assigned_count = self._dispatch_depot_zone_capacity_aware(eligible_unassigned)
            else:  # "cluster_aware_rl"
                assigned_count = self._dispatch_location_aware_min_cost(eligible_unassigned, soft_cluster=True)

            # Recalculate routes for all shippers
            for s_id in self.shippers:
                self._recalculate_shipper_route(s_id, algorithm=algo)

            self.session_revision += 1
            self._active_preview = None

            self.event_history.append(
                SimulationEvent(
                    event_id=str(uuid.uuid4())[:8],
                    timestamp_minute=self.simulation_clock,
                    day=self.current_day,
                    event_type="AUTO_DISPATCH",
                    shipper_id=None,
                    order_id=None,
                    description=f"Auto-dispatched {assigned_count} orders across 16 shippers using {strat} + {algo}.",
                )
            )

            return {
                "success": True,
                "dispatched_orders": assigned_count,
                "strategy_used": strat,
                "algorithm_used": algo,
            }

    def _dispatch_depot_zone_capacity_aware(
        self,
        orders: list[OrderItem],
        shippers_pool: dict[str, ShipperItem] | None = None,
    ) -> int:
        """
        Geographic, depot-aware and capacity-aware dispatch heuristic:
        - Evaluates distance from central depot, operating base, and current route.
        - Strict payload capacity check (remaining_capacity >= weight).
        - Home cluster and preferred secondary cluster affinity.
        - Workload balancing across fleet.
        - Deadline urgency awareness.
        """
        pool = shippers_pool or self.shippers
        shippers = list(pool.values())
        assigned_count = 0
        order_dict = {o.order_id: o for o in orders}

        def _get_coords(oid: str) -> tuple[float, float] | None:
            if oid in order_dict:
                return (order_dict[oid].latitude, order_dict[oid].longitude)
            if oid in self.orders:
                return (self.orders[oid].latitude, self.orders[oid].longitude)
            return None

        # Prioritize carry-overs and earlier deadlines
        sorted_orders = sorted(
            orders,
            key=lambda o: (
                -o.carryover_count,
                o.deadline_minutes if o.deadline_minutes is not None else 9999.0,
            ),
        )

        depot_lat, depot_lng = self.depot["lat"], self.depot["lng"]

        for o in sorted_orders:
            best_shipper = None
            best_cost = float("inf")
            best_detour = 0.0

            for s in shippers:
                if not s.can_accept_order(o):
                    continue

                cur_assigned = s.assigned_order_ids
                s_lat, s_lng = s.current_lat, s.current_lng

                # Incremental detour calculation considering depot
                if not cur_assigned:
                    if s.shift_start_mode == "START_AT_BASE":
                        # Base -> Depot -> Order -> Depot
                        detour = (
                            haversine_distance_km(s.base_lat, s.base_lng, depot_lat, depot_lng)
                            + haversine_distance_km(depot_lat, depot_lng, o.latitude, o.longitude)
                            + haversine_distance_km(o.latitude, o.longitude, depot_lat, depot_lng)
                        )
                    else:
                        # Depot -> Order -> Depot
                        detour = haversine_distance_km(depot_lat, depot_lng, o.latitude, o.longitude) * 2.0
                else:
                    assigned_coords = [_get_coords(oid) for oid in cur_assigned]
                    valid_coords = [c for c in assigned_coords if c is not None]
                    start_node = (s.start_lat, s.start_lng) if s.status == "IDLE" else (s_lat, s_lng)
                    coords = [start_node] + valid_coords + [(depot_lat, depot_lng)]
                    min_detour = float("inf")
                    for k in range(len(coords) - 1):
                        p1, p2 = coords[k], coords[k + 1]
                        d_orig = haversine_distance_km(p1[0], p1[1], p2[0], p2[1])
                        d_new = (
                            haversine_distance_km(p1[0], p1[1], o.latitude, o.longitude)
                            + haversine_distance_km(o.latitude, o.longitude, p2[0], p2[1])
                        )
                        diff = d_new - d_orig
                        if diff < min_detour:
                            min_detour = diff
                    detour = min_detour

                cost = detour

                # Zone affinity: home cluster vs preferred vs foreign
                preferred = getattr(s, "preferred_cluster_ids", [])
                if o.cluster_id == s.home_cluster_id:
                    cost -= 2.50  # Strong home cluster affinity bonus
                elif o.cluster_id in preferred:
                    cost -= 1.00  # Secondary cluster affinity bonus
                else:
                    cost += 3.00  # Cross-zone detour penalty

                # Proximity to depot bonus (encourages consolidating near-depot orders)
                dist_to_depot = haversine_distance_km(depot_lat, depot_lng, o.latitude, o.longitude)
                cost += dist_to_depot * 0.10

                # Workload balancing & capacity pressure
                load_ratio = s.current_load_kg / s.capacity_kg
                cost += load_ratio * 2.0
                cost += len(cur_assigned) * 1.0
                if len(cur_assigned) >= 8:
                    cost += 40.0  # soft limit to avoid overloading single shipper

                # Deadline sensitivity
                if o.deadline_minutes is not None and o.deadline_minutes < 300.0:
                    cost += len(cur_assigned) * 1.5  # assign urgent orders to shippers with less queue

                if cost < best_cost:
                    best_cost = cost
                    best_shipper = s
                    best_detour = detour

            if best_shipper is None:
                o.dispatch_reason = "Không thể phân công: Vượt quá tải trọng tối đa (capacity_kg) của toàn bộ shipper khả dụng"
                continue

            # Determine reason description
            preferred = getattr(best_shipper, "preferred_cluster_ids", [])
            if o.cluster_id == best_shipper.home_cluster_id:
                zone_desc = f"cụm chính #{o.cluster_id}"
            elif o.cluster_id in preferred:
                zone_desc = f"cụm phụ trách mở rộng #{o.cluster_id}"
            else:
                zone_desc = f"hỗ trợ liên cụm #{o.cluster_id}"

            start_mode_desc = "trạm trực ca" if best_shipper.shift_start_mode == "START_AT_BASE" else "kho trung tâm"

            o.assigned_shipper_id = best_shipper.shipper_id
            o.delivery_status = "ASSIGNED"
            o.assigned_at_minute = self.simulation_clock
            o.dispatch_strategy = "depot_zone_capacity_aware"
            o.dispatch_distance_km = round(best_detour, 2)
            o.dispatch_reason = (
                f"Phân công theo {start_mode_desc} & {zone_desc} "
                f"(chi phí chèn +{best_detour:.2f} km, tải: {best_shipper.current_load_kg + o.weight_kg:.1f}/{best_shipper.capacity_kg:.0f} kg) "
                f"cho {best_shipper.name}"
            )

            best_shipper.assigned_order_ids.append(o.order_id)
            best_shipper.current_load_kg += o.weight_kg
            assigned_count += 1

        return assigned_count

    def _dispatch_cluster_round_robin(self, orders: list[OrderItem], shippers_pool: dict[str, ShipperItem] | None = None) -> int:
        """Legacy cluster-first round-robin assignment."""
        pool = shippers_pool or self.shippers
        cluster_groups: dict[int, list[OrderItem]] = {}
        for o in orders:
            cluster_groups.setdefault(o.cluster_id, []).append(o)

        cluster_to_shippers: dict[int, list[str]] = {}
        for s_id, c_id in SHIPPER_HOME_CLUSTERS.items():
            cluster_to_shippers.setdefault(c_id, []).append(s_id)

        assigned_count = 0
        for c_id, cl_orders in cluster_groups.items():
            target_ids = cluster_to_shippers.get(c_id, list(pool.keys()))
            target_shippers = [pool[sid] for sid in target_ids if sid in pool]
            for o in cl_orders:
                feasible = [s for s in target_shippers if s.can_accept_order(o)]
                if not feasible:
                    feasible = [s for s in pool.values() if s.can_accept_order(o)]
                if not feasible:
                    o.dispatch_reason = "Không thể phân công: Toàn bộ đội xe đã đạt giới hạn tải trọng (capacity_kg)"
                    continue

                best_shipper = min(feasible, key=lambda s: s.current_load_kg)
                o.assigned_shipper_id = best_shipper.shipper_id
                o.delivery_status = "ASSIGNED"
                o.assigned_at_minute = self.simulation_clock
                o.dispatch_strategy = "cluster_round_robin"
                d = haversine_distance_km(best_shipper.current_lat, best_shipper.current_lng, o.latitude, o.longitude)
                o.dispatch_distance_km = round(d, 2)
                o.dispatch_reason = f"Phân công cụm K-Means #{c_id} cho {best_shipper.name}"
                best_shipper.assigned_order_ids.append(o.order_id)
                best_shipper.current_load_kg += o.weight_kg
                assigned_count += 1
        return assigned_count

    def _dispatch_location_aware_nearest(self, orders: list[OrderItem], shippers_pool: dict[str, ShipperItem] | None = None) -> int:
        """Greedy nearest feasible assignment respecting payload capacity and balance."""
        pool = shippers_pool or self.shippers
        shippers = list(pool.values())
        assigned_count = 0
        max_orders_per_shipper = int(math.ceil(len(orders) / len(shippers))) + 2

        sorted_orders = sorted(
            orders,
            key=lambda o: (
                -o.carryover_count,
                o.deadline_minutes if o.deadline_minutes is not None else 9999.0,
            ),
        )

        eff_coords = {s.shipper_id: (s.current_lat, s.current_lng) for s in shippers}

        for o in sorted_orders:
            best_shipper = None
            best_dist = float("inf")

            for s in shippers:
                if not s.can_accept_order(o):
                    continue
                if len(s.assigned_order_ids) >= max_orders_per_shipper:
                    continue

                lat, lng = eff_coords[s.shipper_id]
                d = haversine_distance_km(lat, lng, o.latitude, o.longitude)
                if d < best_dist:
                    best_dist = d
                    best_shipper = s

            if best_shipper is None:
                feasible = [s for s in shippers if s.can_accept_order(o)]
                if feasible:
                    best_shipper = min(
                        feasible,
                        key=lambda s: haversine_distance_km(
                            eff_coords[s.shipper_id][0], eff_coords[s.shipper_id][1], o.latitude, o.longitude
                        ),
                    )
                    best_dist = haversine_distance_km(
                        eff_coords[best_shipper.shipper_id][0], eff_coords[best_shipper.shipper_id][1], o.latitude, o.longitude
                    )
                else:
                    o.dispatch_reason = "Không thể phân công: Không có shipper nào còn đủ tải trọng (capacity_kg)"
                    continue

            o.assigned_shipper_id = best_shipper.shipper_id
            o.delivery_status = "ASSIGNED"
            o.assigned_at_minute = self.simulation_clock
            o.dispatch_strategy = "location_aware_nearest"
            o.dispatch_distance_km = round(best_dist, 2)
            o.dispatch_reason = f"Gán theo khoảng cách gần nhất ({best_dist:.2f} km) đến vị trí {best_shipper.name}"
            best_shipper.assigned_order_ids.append(o.order_id)
            best_shipper.current_load_kg += o.weight_kg
            eff_coords[best_shipper.shipper_id] = (o.latitude, o.longitude)
            assigned_count += 1

        return assigned_count

    def _dispatch_location_aware_min_cost(self, orders: list[OrderItem], soft_cluster: bool = False, shippers_pool: dict[str, ShipperItem] | None = None) -> int:
        """Minimum incremental insertion cost dispatch with workload balancing and optional soft cluster affinity."""
        pool = shippers_pool or self.shippers
        shippers = list(pool.values())
        assigned_count = 0
        order_dict = {o.order_id: o for o in orders}

        def _get_coords(oid: str) -> tuple[float, float] | None:
            if oid in order_dict:
                return (order_dict[oid].latitude, order_dict[oid].longitude)
            if oid in self.orders:
                return (self.orders[oid].latitude, self.orders[oid].longitude)
            return None

        sorted_orders = sorted(
            orders,
            key=lambda o: (
                -o.carryover_count,
                o.deadline_minutes if o.deadline_minutes is not None else 9999.0,
            ),
        )

        for o in sorted_orders:
            best_shipper = None
            best_cost = float("inf")
            best_detour = 0.0

            for s in shippers:
                if not s.can_accept_order(o):
                    continue

                cur_assigned = s.assigned_order_ids
                s_lat, s_lng = s.current_lat, s.current_lng

                if not cur_assigned:
                    detour = (
                        haversine_distance_km(s_lat, s_lng, o.latitude, o.longitude)
                        + haversine_distance_km(o.latitude, o.longitude, self.depot["lat"], self.depot["lng"])
                    )
                else:
                    assigned_coords = [_get_coords(oid) for oid in cur_assigned]
                    valid_coords = [c for c in assigned_coords if c is not None]
                    coords = (
                        [(s_lat, s_lng)]
                        + valid_coords
                        + [(self.depot["lat"], self.depot["lng"])]
                    )
                    min_detour = float("inf")
                    for k in range(len(coords) - 1):
                        p1, p2 = coords[k], coords[k + 1]
                        d_orig = haversine_distance_km(p1[0], p1[1], p2[0], p2[1])
                        d_new = (
                            haversine_distance_km(p1[0], p1[1], o.latitude, o.longitude)
                            + haversine_distance_km(o.latitude, o.longitude, p2[0], p2[1])
                        )
                        diff = d_new - d_orig
                        if diff < min_detour:
                            min_detour = diff
                    detour = min_detour

                cost = detour
                if soft_cluster and o.cluster_id == s.home_cluster_id:
                    cost -= 1.50

                # Workload balance penalty
                if len(cur_assigned) >= 7:
                    cost += 50.0  # soft cap
                cost += len(cur_assigned) * 1.50
                cost += (s.current_load_kg / s.capacity_kg) * 1.50

                if cost < best_cost:
                    best_cost = cost
                    best_shipper = s
                    best_detour = detour

            if best_shipper is None:
                o.dispatch_reason = "Không thể phân công: Toàn bộ đội xe đã đạt giới hạn tải trọng (capacity_kg)"
                continue

            strategy_name = "cluster_aware_rl" if soft_cluster else "location_aware_min_cost"
            o.assigned_shipper_id = best_shipper.shipper_id
            o.delivery_status = "ASSIGNED"
            o.assigned_at_minute = self.simulation_clock
            o.dispatch_strategy = strategy_name
            o.dispatch_distance_km = round(best_detour, 2)
            if soft_cluster:
                match_txt = f"cùng cụm #{o.cluster_id}" if o.cluster_id == best_shipper.home_cluster_id else f"cụm lân cận #{o.cluster_id}"
                o.dispatch_reason = f"Gán kết hợp {match_txt} và chi phí chèn (+{best_detour:.2f} km) cho {best_shipper.name}"
            else:
                o.dispatch_reason = f"Gán theo chi phí chèn tối thiểu (+{best_detour:.2f} km) cho {best_shipper.name}"

            best_shipper.assigned_order_ids.append(o.order_id)
            best_shipper.current_load_kg += o.weight_kg
            assigned_count += 1

        return assigned_count

    def _solve_route_for_orders(
        self,
        assigned_orders: list[OrderItem],
        algo: str,
        start_lat: float,
        start_lng: float,
        shift_start_mode: str = "START_AT_DEPOT",
        base_coords: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Core tour sequencing and timeline calculation engine."""
        if not assigned_orders:
            return {
                "route_sequence": [],
                "route_geometry": [[self.depot["lng"], self.depot["lat"]]],
                "timeline_stops": [],
                "current_load_kg": 0.0,
                "total_distance_km": 0.0,
                "estimated_duration_minutes": 0.0,
                "sla_on_time_count": 0,
                "sla_overdue_count": 0,
                "algorithm_used": algo,
            }

        order_ids = [o.order_id for o in assigned_orders]
        n = len(order_ids)

        coords = [(start_lat, start_lng)] + [
            (o.latitude, o.longitude) for o in assigned_orders
        ]
        num_nodes = n + 1
        dist_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
        time_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    d = haversine_distance_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                    dist_matrix[i, j] = d
                    time_matrix[i, j] = (d / self.average_speed_kmh) * 60.0

        deadlines = [o.deadline_minutes for o in assigned_orders]
        clusters = [o.cluster_id for o in assigned_orders]

        actual_algo = algo
        if algo == "proposed_rl":
            try:
                ordered_ids = solve_proposed_rl_route(
                    delivery_ids=order_ids,
                    distance_matrix_km=dist_matrix,
                    travel_time_matrix_minutes=time_matrix,
                    deadlines_minutes=deadlines,
                    cluster_ids=clusters,
                )
            except Exception as e:
                logger.warning("Proposed RL solver failed (%s); falling back to Nearest Neighbor", e)
                ordered_ids = solve_matrix_nearest_neighbor(order_ids, dist_matrix)
                actual_algo = "nearest_neighbor_fallback"
        elif algo == "clarke_wright":
            ordered_ids = solve_matrix_clarke_wright(order_ids, dist_matrix)
        else:  # nearest_neighbor
            ordered_ids = solve_matrix_nearest_neighbor(order_ids, dist_matrix)

        order_map = {o.order_id: o for o in assigned_orders}
        total_load = sum(order_map[o_id].weight_kg for o_id in ordered_ids if o_id in order_map)

        timeline: list[dict[str, Any]] = []
        geom: list[list[float]] = [[start_lng, start_lat]]
        curr_time = self.simulation_clock
        prev_lat, prev_lng = start_lat, start_lng
        total_dist = 0.0
        sla_on_time = 0
        sla_overdue = 0

        # Mode B: If starting at base and goods are at central depot, insert depot pickup leg
        is_mode_base = shift_start_mode == "START_AT_BASE" and (
            abs(start_lat - self.depot["lat"]) > 1e-4 or abs(start_lng - self.depot["lng"]) > 1e-4
        )
        if is_mode_base and assigned_orders:
            pickup_dist = haversine_distance_km(start_lat, start_lng, self.depot["lat"], self.depot["lng"])
            pickup_time = (pickup_dist / self.average_speed_kmh) * 60.0
            depot_arrival = curr_time + pickup_time
            depot_departure = depot_arrival + 10.0  # 10 min pickup loading
            timeline.append({
                "order_id": "depot_pickup",
                "order_code": "PICKUP-DEPOT",
                "package_name": f"Nhận {len(assigned_orders)} kiện tại Kho trung tâm",
                "weight_kg": round(total_load, 1),
                "lat": self.depot["lat"],
                "lng": self.depot["lng"],
                "cluster_id": -1,
                "arrival_minute": round(depot_arrival, 1),
                "departure_minute": round(depot_departure, 1),
                "leg_distance_km": round(pickup_dist, 2),
                "deadline_minutes": None,
                "is_overdue": False,
                "stop_type": "DEPOT_PICKUP",
            })
            geom.append([self.depot["lng"], self.depot["lat"]])
            total_dist += pickup_dist
            curr_time = depot_departure
            prev_lat, prev_lng = self.depot["lat"], self.depot["lng"]

        for o_id in ordered_ids:
            order = order_map.get(o_id) or self.orders.get(o_id)
            if not order:
                continue
            leg_dist = haversine_distance_km(prev_lat, prev_lng, order.latitude, order.longitude)
            total_dist += leg_dist
            leg_time = (leg_dist / self.average_speed_kmh) * 60.0
            arrival = curr_time + leg_time
            departure = arrival + self.service_minutes

            is_prior_day = order.release_day < self.current_day
            is_overdue = False
            if order.deadline_minutes is not None:
                if is_prior_day:
                    is_overdue = True
                else:
                    is_overdue = arrival > order.deadline_minutes
            if is_overdue:
                sla_overdue += 1
            else:
                sla_on_time += 1

            timeline.append({
                "order_id": o_id,
                "order_code": order.order_code or o_id[-6:],
                "package_name": order.package_name or "Kiện hàng",
                "weight_kg": order.weight_kg,
                "lat": order.latitude,
                "lng": order.longitude,
                "cluster_id": order.cluster_id,
                "arrival_minute": round(arrival, 1),
                "departure_minute": round(departure, 1),
                "leg_distance_km": round(leg_dist, 2),
                "deadline_minutes": order.deadline_minutes,
                "is_overdue": is_overdue,
            })

            geom.append([order.longitude, order.latitude])
            curr_time = departure
            prev_lat, prev_lng = order.latitude, order.longitude

        # Return leg to depot
        return_dist = haversine_distance_km(prev_lat, prev_lng, self.depot["lat"], self.depot["lng"])
        total_dist += return_dist
        return_time = (return_dist / self.average_speed_kmh) * 60.0
        return_arrival = curr_time + return_time
        geom.append([self.depot["lng"], self.depot["lat"]])

        curr_time = return_arrival

        return {
            "route_sequence": ordered_ids,
            "route_geometry": geom,
            "timeline_stops": timeline,
            "current_load_kg": round(total_load, 1),
            "total_distance_km": round(total_dist, 2),
            "estimated_duration_minutes": round(curr_time - self.simulation_clock, 1),
            "sla_on_time_count": sla_on_time,
            "sla_overdue_count": sla_overdue,
            "algorithm_used": actual_algo,
        }

    def _recalculate_shipper_route(
        self,
        shipper_id: str,
        algorithm: Literal["nearest_neighbor", "clarke_wright", "proposed_rl"] | None = None,
    ) -> None:
        """
        Sequences assigned orders and generates route coordinates and timeline.
        Uses selected algorithm (Nearest Neighbor, Clarke-Wright, or Proposed RL).
        """
        shipper = self.shippers[shipper_id]
        algo = algorithm or self.routing_algorithm

        assigned_orders = [
            self.orders[o_id]
            for o_id in shipper.assigned_order_ids
            if o_id in self.orders and self.orders[o_id].delivery_status != "DELIVERED"
        ]

        start_lat = shipper.current_lat if shipper.status != "IDLE" else shipper.start_lat
        start_lng = shipper.current_lng if shipper.status != "IDLE" else shipper.start_lng
        res = self._solve_route_for_orders(
            assigned_orders,
            algo,
            start_lat,
            start_lng,
            shift_start_mode=shipper.shift_start_mode,
            base_coords=(shipper.base_lat, shipper.base_lng),
        )

        shipper.route_sequence = res["route_sequence"]
        shipper.route_geometry = res["route_geometry"]
        shipper.timeline_stops = res["timeline_stops"]
        shipper.current_load_kg = res["current_load_kg"]
        shipper.total_distance_km = res["total_distance_km"]

    def preview_dispatch(
        self,
        strategy: Literal[
            "cluster_round_robin",
            "location_aware_nearest",
            "location_aware_min_cost",
            "cluster_aware_rl",
            "depot_zone_capacity_aware",
        ] | None = None,
        algorithm: Literal["nearest_neighbor", "clarke_wright", "proposed_rl"] | None = None,
    ) -> dict[str, Any]:
        """
        True non-destructive Plan Preview & Compare.
        Simulates assignment and route sequencing on an isolated snapshot.
        Guarantees zero mutation to official session state.
        """
        with self._lock:
            strat = strategy or "cluster_aware_rl"
            algo = algorithm or self.routing_algorithm

            # Snapshot of eligible unassigned orders
            eligible_unassigned = [
                copy.deepcopy(o) for o in self.orders.values()
                if o.release_day <= self.current_day
                and o.delivery_status in ("UNASSIGNED", "CARRIED_OVER")
            ]

            if not eligible_unassigned:
                return {
                    "success": True,
                    "preview_id": None,
                    "message": "Không có đơn hàng nào cần phân công hôm nay.",
                    "candidate_kpi": None,
                    "baseline_kpi": None,
                    "deltas": None,
                    "shipper_previews": [],
                }

            # Cloned shippers pool
            cloned_shippers = {s_id: copy.deepcopy(s) for s_id, s in self.shippers.items()}

            # Run chosen assignment strategy on the clone
            if strat == "cluster_round_robin":
                assigned_count = self._dispatch_cluster_round_robin(eligible_unassigned, shippers_pool=cloned_shippers)
            elif strat == "location_aware_nearest":
                assigned_count = self._dispatch_location_aware_nearest(eligible_unassigned, shippers_pool=cloned_shippers)
            elif strat == "location_aware_min_cost":
                assigned_count = self._dispatch_location_aware_min_cost(eligible_unassigned, soft_cluster=False, shippers_pool=cloned_shippers)
            elif strat == "depot_zone_capacity_aware":
                assigned_count = self._dispatch_depot_zone_capacity_aware(eligible_unassigned, shippers_pool=cloned_shippers)
            else:  # "cluster_aware_rl"
                assigned_count = self._dispatch_location_aware_min_cost(eligible_unassigned, soft_cluster=True, shippers_pool=cloned_shippers)

            # Solve candidate route plans for all 16 shippers
            order_clone_map = {o.order_id: o for o in eligible_unassigned}
            shipper_previews: list[dict[str, Any]] = []
            candidate_total_distance = 0.0
            candidate_total_load = 0.0
            candidate_sla_on_time = 0
            candidate_sla_overdue = 0
            candidate_max_duration = 0.0
            candidate_total_overtime = 0.0
            active_count = 0

            for s_id, s in cloned_shippers.items():
                s_orders = [
                    order_clone_map[o_id] if o_id in order_clone_map else self.orders[o_id]
                    for o_id in s.assigned_order_ids
                    if o_id in order_clone_map or (o_id in self.orders and self.orders[o_id].delivery_status != "DELIVERED")
                ]
                start_lat = s.current_lat if s.status != "IDLE" else s.start_lat
                start_lng = s.current_lng if s.status != "IDLE" else s.start_lng
                route_res = self._solve_route_for_orders(
                    s_orders,
                    algo,
                    start_lat,
                    start_lng,
                    shift_start_mode=s.shift_start_mode,
                    base_coords=(s.base_lat, s.base_lng),
                )

                has_orders = len(s.assigned_order_ids) > 0
                if has_orders:
                    active_count += 1

                dist_km = route_res["total_distance_km"]
                dur_min = route_res["estimated_duration_minutes"]
                overtime_min = max(0.0, dur_min - self.shift_minutes)
                candidate_total_distance += dist_km
                candidate_total_load += route_res["current_load_kg"]
                candidate_sla_on_time += route_res["sla_on_time_count"]
                candidate_sla_overdue += route_res["sla_overdue_count"]
                candidate_max_duration = max(candidate_max_duration, dur_min)
                candidate_total_overtime += overtime_min

                shipper_previews.append({
                    "shipper_id": s_id,
                    "name": s.name,
                    "home_cluster_id": s.home_cluster_id,
                    "status": s.status,
                    "current_lat": s.current_lat,
                    "current_lng": s.current_lng,
                    "location_source": s.location_source,
                    "assigned_order_ids": s.assigned_order_ids,
                    "order_count": len(s.assigned_order_ids),
                    "current_load_kg": route_res["current_load_kg"],
                    "capacity_kg": s.capacity_kg,
                    "is_overweight": route_res["current_load_kg"] > s.capacity_kg,
                    "total_distance_km": dist_km,
                    "estimated_duration_minutes": dur_min,
                    "is_overtime": overtime_min > 0,
                    "route_sequence": route_res["route_sequence"],
                    "route_geometry": route_res["route_geometry"],
                    "timeline_stops": route_res["timeline_stops"],
                    "sla_on_time_count": route_res["sla_on_time_count"],
                    "sla_overdue_count": route_res["sla_overdue_count"],
                    "algorithm_used": route_res["algorithm_used"],
                })

            total_evaluated_orders = candidate_sla_on_time + candidate_sla_overdue
            candidate_sla_rate = (candidate_sla_on_time / total_evaluated_orders) if total_evaluated_orders > 0 else 1.0
            candidate_cost = candidate_total_distance * 4000.0 + candidate_total_overtime * 1500.0

            # Current Baseline KPIs: evaluated honestly on currently assigned orders
            current_active = sum(1 for s in self.shippers.values() if len(s.assigned_order_ids) > 0)
            current_assigned = sum(len(s.assigned_order_ids) for s in self.shippers.values())

            has_valid_baseline = current_assigned > 0
            baseline_distance = 0.0
            baseline_overtime = 0.0
            baseline_on_time_count = 0
            baseline_overdue_count = 0

            if has_valid_baseline:
                for s in self.shippers.values():
                    if not s.assigned_order_ids:
                        continue
                    dur = sum(st.get("leg_distance_km", 0.0) / self.average_speed_kmh * 60.0 + self.service_minutes for st in s.timeline_stops)
                    dist = sum(st.get("leg_distance_km", 0.0) for st in s.timeline_stops)
                    baseline_distance += dist
                    baseline_overtime += max(0.0, dur - self.shift_minutes)
                    for st in s.timeline_stops:
                        if st.get("is_overdue"):
                            baseline_overdue_count += 1
                        else:
                            baseline_on_time_count += 1

                baseline_total_eval = baseline_on_time_count + baseline_overdue_count
                baseline_sla_rate = (baseline_on_time_count / baseline_total_eval) if baseline_total_eval > 0 else None
                baseline_cost = baseline_distance * 4000.0 + baseline_overtime * 1500.0

                baseline_kpi = {
                    "assigned_orders": current_assigned,
                    "active_shippers": current_active,
                    "total_distance_km": round(baseline_distance, 2),
                    "sla_on_time_rate": round(baseline_sla_rate, 4) if baseline_sla_rate is not None else None,
                    "estimated_cost": round(baseline_cost, 0),
                    "is_comparable": True,
                }
                deltas = {
                    "delta_distance_km": round(candidate_total_distance - baseline_distance, 2),
                    "delta_active_shippers": active_count - current_active,
                    "delta_orders_dispatched": sum(len(s.assigned_order_ids) for s in cloned_shippers.values()) - current_assigned,
                    "delta_cost": round(candidate_cost - baseline_cost, 0),
                    "delta_sla_rate": round(candidate_sla_rate - baseline_sla_rate, 4) if baseline_sla_rate is not None else None,
                }
            else:
                baseline_kpi = None
                deltas = None

            preview_id = f"prev_{uuid.uuid4().hex[:8]}"

            # Order preview details
            order_previews = []
            for o in eligible_unassigned:
                order_previews.append({
                    "order_id": o.order_id,
                    "order_code": o.order_code,
                    "package_name": o.package_name,
                    "cluster_id": o.cluster_id,
                    "weight_kg": o.weight_kg,
                    "deadline_minutes": o.deadline_minutes,
                    "candidate_shipper_id": o.assigned_shipper_id,
                    "dispatch_strategy": o.dispatch_strategy,
                    "dispatch_reason": o.dispatch_reason,
                    "dispatch_distance_km": o.dispatch_distance_km,
                })

            # Save in memory for atomic confirmation
            self._active_preview = {
                "preview_id": preview_id,
                "session_revision": self.session_revision,
                "strategy": strat,
                "algorithm": algo,
                "cloned_assignments": {s_id: list(s.assigned_order_ids) for s_id, s in cloned_shippers.items()},
                "assigned_orders_meta": {
                    o.order_id: {
                        "shipper_id": o.assigned_shipper_id,
                        "strategy": o.dispatch_strategy,
                        "reason": o.dispatch_reason,
                        "distance_km": o.dispatch_distance_km,
                    } for o in eligible_unassigned if o.assigned_shipper_id
                },
            }

            used_algos = list({s["algorithm_used"] for s in shipper_previews if s["order_count"] > 0})
            is_fallback = any("fallback" in s["algorithm_used"] for s in shipper_previews)
            actual_algo = algo
            fallback_reason = None
            if is_fallback:
                actual_algo = "nearest_neighbor_fallback"
                fallback_reason = "Proposed RL solver tự động chuyển dự phòng (fallback) sang Nearest Neighbor cho các tour phức tạp."
            elif used_algos:
                actual_algo = used_algos[0] if len(used_algos) == 1 else "hybrid"

            return {
                "success": True,
                "preview_id": preview_id,
                "session_id": self.session_id,
                "session_revision": self.session_revision,
                "requested_strategy": strat,
                "requested_algorithm": algo,
                "strategy_used": strat,
                "algorithm_used": actual_algo,
                "is_fallback": is_fallback,
                "fallback_reason": fallback_reason,
                "distance_model": "haversine",
                "speed_assumption": f"{self.average_speed_kmh} km/h (Mô phỏng đường chim bay)",
                "candidate_kpi": {
                    "total_eligible_orders": len(eligible_unassigned),
                    "dispatched_orders": assigned_count,
                    "total_assigned_orders": sum(len(s.assigned_order_ids) for s in cloned_shippers.values()),
                    "unassigned_orders": len(eligible_unassigned) - assigned_count,
                    "active_shippers": active_count,
                    "total_distance_km": round(candidate_total_distance, 2),
                    "mean_distance_km": round(candidate_total_distance / max(1, active_count), 2),
                    "total_load_kg": round(candidate_total_load, 1),
                    "sla_on_time_rate": round(candidate_sla_rate, 4),
                    "max_duration_minutes": round(candidate_max_duration, 1),
                    "total_overtime_minutes": round(candidate_total_overtime, 1),
                    "estimated_cost": round(candidate_cost, 0),
                },
                "baseline_kpi": baseline_kpi,
                "deltas": deltas,
                "shipper_previews": shipper_previews,
                "order_previews": order_previews,
            }

    def confirm_preview(self, preview_id: str) -> dict[str, Any]:
        """Atomically applies an active preview plan to the official session."""
        with self._lock:
            if not self._active_preview or self._active_preview.get("preview_id") != preview_id:
                raise ValueError("Bản xem trước (preview) không tồn tại hoặc đã hết hạn.")
            if self._active_preview.get("session_revision") != self.session_revision:
                raise ValueError("Trạng thái ca giao đã thay đổi trong khi xem trước. Vui lòng tạo bản xem trước mới.")

            algo = self._active_preview["algorithm"]
            strat = self._active_preview["strategy"]
            assigned_meta = self._active_preview["assigned_orders_meta"]
            cloned_assigns = self._active_preview["cloned_assignments"]

            assigned_count = 0
            for o_id, meta in assigned_meta.items():
                if o_id in self.orders:
                    o = self.orders[o_id]
                    o.assigned_shipper_id = meta["shipper_id"]
                    o.delivery_status = "ASSIGNED"
                    o.assigned_at_minute = self.simulation_clock
                    o.dispatch_strategy = meta["strategy"]
                    o.dispatch_reason = meta["reason"]
                    o.dispatch_distance_km = meta["distance_km"]
                    assigned_count += 1

            for s_id, o_ids in cloned_assigns.items():
                if s_id in self.shippers:
                    self.shippers[s_id].assigned_order_ids = list(o_ids)
                    self._recalculate_shipper_route(s_id, algorithm=algo)

            self.session_revision += 1
            self._active_preview = None

            self.event_history.append(
                SimulationEvent(
                    event_id=str(uuid.uuid4())[:8],
                    timestamp_minute=self.simulation_clock,
                    day=self.current_day,
                    event_type="CONFIRM_PREVIEW",
                    shipper_id=None,
                    order_id=None,
                    description=f"Confirmed plan preview ({preview_id}) applying {strat} + {algo} across {assigned_count} orders.",
                )
            )

            return {
                "success": True,
                "dispatched_orders": assigned_count,
                "algorithm_used": algo,
                "strategy_used": strat,
                "preview_id": preview_id,
            }

    def cancel_preview(self) -> dict[str, Any]:
        """Cancels any active preview without modifying session state."""
        with self._lock:
            self._active_preview = None
            return {"success": True, "message": "Bản xem trước đã bị hủy mà không thay đổi ca giao."}

    def step_simulation(self, delta_minutes: float = 15.0) -> dict[str, Any]:
        """
        Advances the simulation clock by delta_minutes.
        Updates 16 shippers' concurrent movement, handles deliveries, and respects workday limits.
        """
        with self._lock:
            if self.is_day_ended:
                return {"message": "Current workday has ended. Please advance to Next Day.", "is_day_ended": True}

            self.simulation_clock += delta_minutes

            # Check for shift end
            if self.simulation_clock >= self.shift_minutes + self.max_overtime_minutes:
                self.simulation_clock = self.shift_minutes + self.max_overtime_minutes
                self.is_running = False

            deliveries_made = 0

            for shipper in self.shippers.values():
                if not shipper.timeline_stops:
                    shipper.status = "IDLE"
                    continue

                shipper.working_minutes = min(self.shift_minutes, self.simulation_clock)
                if self.simulation_clock > self.shift_minutes:
                    shipper.overtime_minutes = self.simulation_clock - self.shift_minutes

                # Process stops in timeline with accurate leg tracking and Mode B pickup support
                prev_lat = shipper.start_lat if shipper.shift_start_mode == "START_AT_BASE" else self.depot["lat"]
                prev_lng = shipper.start_lng if shipper.shift_start_mode == "START_AT_BASE" else self.depot["lng"]
                prev_dep_time = 0.0

                active_stop = None
                for stop in shipper.timeline_stops:
                    is_pickup = (stop.get("stop_type") == "DEPOT_PICKUP" or stop.get("order_id") == "depot_pickup")
                    is_return = (stop.get("stop_type") == "RETURN_DEPOT" or stop.get("order_id") == "return_to_depot")

                    if is_pickup:
                        if self.simulation_clock >= stop["departure_minute"]:
                            prev_lat, prev_lng = stop["lat"], stop["lng"]
                            prev_dep_time = stop["departure_minute"]
                            continue
                        elif self.simulation_clock >= stop["arrival_minute"]:
                            active_stop = stop
                            shipper.current_lat = stop["lat"]
                            shipper.current_lng = stop["lng"]
                            shipper.status = "PICKING_UP"
                            shipper.current_order_id = "depot_pickup"
                            shipper.location_source = "DEPOT"
                            shipper.last_location_update_minute = self.simulation_clock
                            break
                        else:
                            active_stop = stop
                            shipper.status = "IN_TRANSIT"
                            shipper.current_order_id = "depot_pickup"
                            shipper.location_source = "SIMULATION"
                            shipper.last_location_update_minute = self.simulation_clock
                            leg_duration = max(0.1, stop["arrival_minute"] - prev_dep_time)
                            progress = max(0.0, min(1.0, (self.simulation_clock - prev_dep_time) / leg_duration))
                            shipper.current_lat = prev_lat + progress * (stop["lat"] - prev_lat)
                            shipper.current_lng = prev_lng + progress * (stop["lng"] - prev_lng)
                            break

                    elif is_return:
                        if self.simulation_clock >= stop["arrival_minute"]:
                            prev_lat, prev_lng = stop["lat"], stop["lng"]
                            prev_dep_time = stop["arrival_minute"]
                            continue
                        else:
                            active_stop = stop
                            shipper.status = "RETURNING"
                            shipper.current_order_id = "return_to_depot"
                            shipper.location_source = "SIMULATION"
                            shipper.last_location_update_minute = self.simulation_clock
                            leg_duration = max(0.1, stop["arrival_minute"] - prev_dep_time)
                            progress = max(0.0, min(1.0, (self.simulation_clock - prev_dep_time) / leg_duration))
                            shipper.current_lat = prev_lat + progress * (stop["lat"] - prev_lat)
                            shipper.current_lng = prev_lng + progress * (stop["lng"] - prev_lng)
                            break

                    o_id = stop["order_id"]
                    order = self.orders.get(o_id)
                    if not order or order.delivery_status == "DELIVERED":
                        prev_lat, prev_lng = stop["lat"], stop["lng"]
                        prev_dep_time = stop["departure_minute"]
                        continue

                    # If clock has passed arrival + service time, mark delivered
                    if self.simulation_clock >= stop["departure_minute"]:
                        order.delivery_status = "DELIVERED"
                        order.delivered_at_minute = stop["departure_minute"]
                        order.delivered_day = self.current_day
                        order.update_sla_status(self.simulation_clock, self.current_day)

                        if o_id not in shipper.completed_order_ids:
                            shipper.completed_order_ids.append(o_id)
                            shipper.distance_travelled_km += stop["leg_distance_km"]
                            deliveries_made += 1

                            self.event_history.append(
                                SimulationEvent(
                                    event_id=str(uuid.uuid4())[:8],
                                    timestamp_minute=stop["departure_minute"],
                                    day=self.current_day,
                                    event_type="DELIVERED",
                                    shipper_id=shipper.shipper_id,
                                    order_id=o_id,
                                    description=(
                                        f"{shipper.name} delivered Order #{o_id[:6]} "
                                        f"({order.sla_status}) at t={stop['departure_minute']:.0f}m"
                                    ),
                                )
                            )
                        prev_lat, prev_lng = stop["lat"], stop["lng"]
                        prev_dep_time = stop["departure_minute"]
                    elif self.simulation_clock >= stop["arrival_minute"]:
                        # Delivering now
                        active_stop = stop
                        order.delivery_status = "IN_TRANSIT"
                        shipper.current_lat = stop["lat"]
                        shipper.current_lng = stop["lng"]
                        shipper.status = "DELIVERING"
                        shipper.current_order_id = o_id
                        shipper.location_source = "SIMULATION"
                        shipper.last_location_update_minute = self.simulation_clock
                        break
                    else:
                        # Moving towards this stop from prev_stop
                        active_stop = stop
                        order.delivery_status = "IN_TRANSIT"
                        shipper.status = "IN_TRANSIT"
                        shipper.current_order_id = o_id
                        shipper.location_source = "SIMULATION"
                        shipper.last_location_update_minute = self.simulation_clock
                        leg_duration = max(0.1, stop["arrival_minute"] - prev_dep_time)
                        progress = max(0.0, min(1.0, (self.simulation_clock - prev_dep_time) / leg_duration))
                        shipper.current_lat = prev_lat + progress * (stop["lat"] - prev_lat)
                        shipper.current_lng = prev_lng + progress * (stop["lng"] - prev_lng)
                        break

                if not active_stop and shipper.assigned_order_ids and all(
                    self.orders[oid].delivery_status == "DELIVERED"
                    for oid in shipper.assigned_order_ids if oid in self.orders
                ):
                    return_dist = haversine_distance_km(prev_lat, prev_lng, self.depot["lat"], self.depot["lng"])
                    return_arrival = prev_dep_time + (return_dist / self.average_speed_kmh) * 60.0
                    if self.simulation_clock < return_arrival:
                        shipper.status = "RETURNING"
                        shipper.current_order_id = "return_to_depot"
                        shipper.location_source = "SIMULATION"
                        shipper.last_location_update_minute = self.simulation_clock
                        leg_duration = max(0.1, return_arrival - prev_dep_time)
                        progress = max(0.0, min(1.0, (self.simulation_clock - prev_dep_time) / leg_duration))
                        shipper.current_lat = prev_lat + progress * (self.depot["lat"] - prev_lat)
                        shipper.current_lng = prev_lng + progress * (self.depot["lng"] - prev_lng)
                    else:
                        shipper.status = "IDLE"
                        shipper.current_order_id = None
                        shipper.current_lat = self.depot["lat"]
                        shipper.current_lng = self.depot["lng"]
                        shipper.location_source = "DEPOT"
                        shipper.last_location_update_minute = self.simulation_clock

            # Update SLA for all remaining uncompleted orders
            for o in self.orders.values():
                if o.release_day <= self.current_day and o.delivery_status != "DELIVERED":
                    o.update_sla_status(self.simulation_clock, self.current_day)

            return {
                "simulation_clock": round(self.simulation_clock, 1),
                "current_day": self.current_day,
                "is_running": self.is_running,
                "deliveries_made": deliveries_made,
                "is_overtime": self.simulation_clock > self.shift_minutes,
            }

    def _end_day_locked(self) -> dict[str, Any]:
        """Internal end_day implementation executed while holding self._lock."""
        if self.is_day_ended:
            return {
                "success": True,
                "message": f"Day {self.current_day} is already ended.",
                "already_ended": True,
            }

        self.is_running = False
        self.is_day_ended = True

        today_orders = [
            o for o in self.orders.values()
            if o.release_day <= self.current_day
        ]

        delivered = [o for o in today_orders if o.delivery_status == "DELIVERED"]
        on_time = [o for o in delivered if o.sla_status == "ON_TIME"]
        overdue = [o for o in delivered if o.sla_status == "OVERDUE"]

        # Orders not delivered become carry-over
        undelivered = [o for o in today_orders if o.delivery_status != "DELIVERED"]
        new_orders_today = [o for o in today_orders if o.release_day == self.current_day]
        carried_in = [o for o in today_orders if o.release_day < self.current_day]

        # Daily delivered orders for this shift
        day_deliv = [o for o in self.orders.values() if o.delivered_day == self.current_day]
        day_ontime = [o for o in day_deliv if o.sla_status == "ON_TIME"]
        day_overdue = [o for o in day_deliv if o.sla_status == "OVERDUE"]
        day_sla_pct = round(len(day_ontime) / max(1, len(day_deliv)), 3) if day_deliv else 0.0

        # Cumulative delivered metrics up to current_day
        cum_released = [o for o in self.orders.values() if o.release_day <= self.current_day]
        cum_delivered = [o for o in cum_released if o.delivery_status == "DELIVERED"]
        cum_ontime = [o for o in cum_delivered if o.sla_status == "ON_TIME"]
        cum_overdue = [o for o in cum_delivered if o.sla_status == "OVERDUE"]
        cum_sla_pct = round(len(cum_ontime) / max(1, len(cum_delivered)), 3) if cum_delivered else 0.0

        # Capture snapshot of shipper performance for this specific shift BEFORE resetting shipper assignments
        shipper_perf = []
        for s in self.shippers.values():
            # Orders completed today by this shipper
            s_completed = [
                o.order_id for o in day_deliv
                if o.assigned_shipper_id == s.shipper_id or o.order_id in s.completed_order_ids
            ]
            # Orders assigned to this shipper today that were not completed (pending carryover)
            s_pending = [
                o.order_id for o in undelivered
                if o.assigned_shipper_id == s.shipper_id or (o.order_id in s.assigned_order_ids and o.delivery_status != "DELIVERED")
            ]
            s_all_today = list(dict.fromkeys(s_completed + s_pending))
            s_weight = sum(self.orders[o_id].weight_kg for o_id in s_all_today if o_id in self.orders)

            shipper_perf.append({
                "shipper_id": s.shipper_id,
                "name": s.name,
                "assigned_count": len(s_all_today),
                "completed_count": len(s_completed),
                "pending_count": len(s_pending),
                "weight_kg": round(s_weight, 1),
                "distance_km": round(s.distance_travelled_km, 2),
            })

        for o in undelivered:
            o.delivery_status = "CARRIED_OVER"
            o.carryover_count += 1
            o.assigned_shipper_id = None
            o.assigned_at_minute = None

        # Reset shipper active states for shift end
        for s in self.shippers.values():
            s.status = "SHIFT_ENDED"
            s.current_order_id = None
            s.assigned_order_ids.clear()
            s.timeline_stops.clear()

        metrics = {
            "day": self.current_day,
            # Daily breakdown
            "new_orders_today": len(new_orders_today),
            "carried_over_in": len(carried_in),
            "total_orders_to_process": len(today_orders),
            "day_delivered": len(day_deliv),
            "day_delivered_on_time": len(day_ontime),
            "day_delivered_overdue": len(day_overdue),
            "day_on_time_rate": day_sla_pct,
            "day_sla_rate": day_sla_pct,
            "day_remaining": len(undelivered),
            "day_carried_over_out": len(undelivered),
            "carried_over_out": len(undelivered),
            # Cumulative breakdown
            "total_orders": len(cum_released),
            "cumulative_total_released": len(cum_released),
            "delivered": len(cum_delivered),
            "cumulative_delivered": len(cum_delivered),
            "delivered_on_time": len(cum_ontime),
            "delivered_overdue": len(cum_overdue),
            "on_time_rate": cum_sla_pct,
            "cumulative_sla_rate": cum_sla_pct,
            "delivery_completion_rate": round(len(cum_delivered) / max(1, len(cum_released)), 3) if cum_released else 0.0,
            # Fleet stats
            "shippers_active": 16,
            "total_distance_km": round(sum(s.distance_travelled_km for s in self.shippers.values()), 2),
            "total_work_minutes": round(sum(s.working_minutes for s in self.shippers.values()), 1),
            "total_overtime_minutes": round(sum(s.overtime_minutes for s in self.shippers.values()), 1),
            "shift_status": "EARLY_CUT" if self.simulation_clock < self.shift_minutes else ("FULL_SHIFT" if self.simulation_clock <= self.shift_minutes else "OVERTIME"),
            "simulation_clock_at_end": round(self.simulation_clock, 1),
            "shift_duration_minutes": self.shift_minutes,
            "shipper_performance": shipper_perf,
        }
        self.daily_metrics.append(metrics)

        self.event_history.append(
            SimulationEvent(
                event_id=str(uuid.uuid4())[:8],
                timestamp_minute=self.simulation_clock,
                day=self.current_day,
                event_type="END_DAY",
                shipper_id=None,
                order_id=None,
                description=(
                    f"Day {self.current_day} ended. Delivered: {len(delivered)}, "
                    f"Carried over: {len(undelivered)}."
                ),
            )
        )

        return {
            "success": True,
            "day": self.current_day,
            "metrics": metrics,
            "carried_over_orders_count": len(undelivered),
        }

    def end_day(self) -> dict[str, Any]:
        """
        Ends the current workday.
        - Stops all ongoing in-transit movements.
        - Identifies all undelivered orders (ASSIGNED, IN_TRANSIT, UNASSIGNED).
        - Transitions them to CARRIED_OVER, preserves deadlines, and increments carryover_count.
        - Computes daily operational metrics.
        - Prevents duplicate call errors.
        """
        with self._lock:
            return self._end_day_locked()

    def start_next_day(self) -> dict[str, Any]:
        """
        Transitions to next day.
        Order Pool = Carry-over Orders (with carryover_count incremented & SLA preserved) + New Orders.
        Resets clock to 0.0 and resets shippers to Depot.
        """
        with self._lock:
            if not self.is_day_ended:
                self._end_day_locked()

            if self.current_day >= self.total_days:
                return {
                    "success": False,
                    "message": f"Simulation reached total duration of {self.total_days} days.",
                    "simulation_completed": True,
                }

            self.current_day += 1
            self.simulation_clock = 0.0
            self.is_day_ended = False
            self.is_running = False

            # Reposition shippers to depot and reset daily counters
            for s in self.shippers.values():
                s.current_lat = self.depot["lat"]
                s.current_lng = self.depot["lng"]
                s.location_source = "DEPOT"
                s.last_location_update_minute = 0.0
                s.status = "IDLE"
                s.current_load_kg = 0.0
                s.working_minutes = 0.0
                s.overtime_minutes = 0.0
                s.distance_travelled_km = 0.0
                s.completed_order_ids.clear()
                s.assigned_order_ids.clear()
                s.route_sequence.clear()
                s.route_geometry = [[self.depot["lng"], self.depot["lat"]]]
                s.timeline_stops.clear()

            # Identify today's pool
            carried_over = [
                o for o in self.orders.values()
                if o.delivery_status == "CARRIED_OVER"
            ]
            new_incoming = [
                o for o in self.orders.values()
                if o.release_day == self.current_day and o.delivery_status == "UNASSIGNED"
            ]

            self.event_history.append(
                SimulationEvent(
                    event_id=str(uuid.uuid4())[:8],
                    timestamp_minute=0.0,
                    day=self.current_day,
                    event_type="NEXT_DAY",
                    shipper_id=None,
                    order_id=None,
                    description=(
                        f"Day {self.current_day} started. "
                        f"Pool: {len(carried_over)} carry-over + {len(new_incoming)} new orders."
                    ),
                )
            )

            return {
                "success": True,
                "day": self.current_day,
                "simulation_clock": 0.0,
                "carried_over_count": len(carried_over),
                "new_incoming_count": len(new_incoming),
                "total_available_orders": len(carried_over) + len(new_incoming),
            }

    def get_state(self) -> dict[str, Any]:
        """Serializes current session state for API responses."""
        with self._lock:
            shippers_data = [asdict(s) for s in self.shippers.values()]
            orders_data = [asdict(o) for o in self.orders.values() if o.release_day <= self.current_day]

            # Overall metrics
            delivered = sum(1 for o in orders_data if o["delivery_status"] == "DELIVERED")
            carried_over = sum(1 for o in orders_data if o["delivery_status"] == "CARRIED_OVER")
            unassigned = sum(1 for o in orders_data if o["delivery_status"] == "UNASSIGNED")
            assigned = sum(1 for o in orders_data if o["delivery_status"] in ("ASSIGNED", "IN_TRANSIT"))

            return {
                "session_id": self.session_id,
                "scenario_id": self.scenario_id,
                "scenario_name": self.scenario_name,
                "current_day": self.current_day,
                "total_days": self.total_days,
                "simulation_clock": round(self.simulation_clock, 1),
                "shift_minutes": self.shift_minutes,
                "max_overtime_minutes": self.max_overtime_minutes,
                "is_running": self.is_running,
                "is_day_ended": self.is_day_ended,
                "routing_algorithm": self.routing_algorithm,
                "distance_model": "haversine",
                "speed_assumption": f"{self.average_speed_kmh} km/h (Mô phỏng đường chim bay)",
                "priority_policy": self.priority_policy,
                "depot": self.depot,
                "staging_bases": list(LOGISTICS_STAGING_BASES.values()),
                "session_revision": self.session_revision,
                "has_active_preview": self._active_preview is not None,
                "active_preview_id": self._active_preview.get("preview_id") if self._active_preview else None,
                "fleet_summary": {
                    "total_shippers": len(self.shippers),
                    "active_shippers": sum(1 for s in self.shippers.values() if s.status != "IDLE"),
                    "total_distance_km": round(sum(s.distance_travelled_km for s in self.shippers.values()), 2),
                },
                "order_summary": {
                    "total": len(orders_data),
                    "delivered": delivered,
                    "assigned": assigned,
                    "unassigned": unassigned,
                    "carried_over": carried_over,
                },
                "shippers": shippers_data,
                "orders": orders_data,
                "daily_metrics": self.daily_metrics,
                "recent_events": [asdict(e) for e in self.event_history[-20:]],
            }


class DispatchManager:
    """Manages active simulation sessions."""

    def __init__(self) -> None:
        self.sessions: dict[str, DispatchSession] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        scenario_id: str = "simulated_stress_day1_to_3",
        routing_algorithm: Literal["nearest_neighbor", "clarke_wright", "proposed_rl"] = "proposed_rl",
        priority_policy: Literal["overdue_first", "earliest_deadline", "nearest_available", "sla_aware"] = "sla_aware",
        total_days: int = 3,
        seed: int = 42,
    ) -> DispatchSession:
        with self._lock:
            s_id = str(uuid.uuid4())[:8]
            name = (
                "Kịch bản Mô phỏng 16 Shipper Đa ngày (SIMULATED SCENARIO)"
                if scenario_id == "simulated_stress_day1_to_3"
                else "Kịch bản Dữ liệu Thực tế Busiest Window"
            )
            session = DispatchSession(
                session_id=s_id,
                scenario_id=scenario_id,
                scenario_name=name,
                total_days=total_days,
                routing_algorithm=routing_algorithm,
                priority_policy=priority_policy,
                seed=seed,
            )
            self.sessions[s_id] = session
            return session

    def get_session(self, session_id: str) -> DispatchSession | None:
        with self._lock:
            return self.sessions.get(session_id)


dispatch_manager = DispatchManager()
