from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans

from clustering.geo_projection import LocalGeoProjection


ALLOCATION_POLICIES = (
    "dedicated",
    "merge_small",
)

OVERTIME_POLICIES = (
    "overtime",
    "defer",
    "hybrid",
)

MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True)
class Order:
    """
    An order released at the start of `release_day`.

    node:
        row/column index in the road matrices
        (0 = depot, 1..N = delivery nodes).

    deadline_minutes:
        deadline measured from the start of the release day,
        or None when the order has no valid deadline.
    """

    order_id: str
    node: int
    latitude: float
    longitude: float
    weight: float
    release_day: int
    deadline_minutes: float | None


@dataclass(frozen=True)
class PlanningConfig:
    """
    Cost values use the same units as config/settings.py (VND).
    They are scenario parameters, not real XeDu costs.
    """

    n_shippers: int = 16

    shift_minutes: float = 600.0
    max_overtime_minutes: float = 120.0

    service_minutes: float = 10.0
    capacity_kg: float = 100.0

    wage_per_hour: float = 40_000.0
    overtime_multiplier: float = 1.5
    cost_per_km: float = 4_000.0

    defer_penalty_per_order_day: float = 60_000.0
    late_delivery_penalty: float = 30_000.0

    allocation: str = "dedicated"
    overtime_policy: str = "hybrid"

    cluster_target_size: int = 4
    small_cluster_threshold: int = 3

    seed: int = 42

    def __post_init__(self) -> None:
        if self.allocation not in ALLOCATION_POLICIES:
            raise ValueError(
                f"Unknown allocation policy: {self.allocation}"
            )

        if self.overtime_policy not in OVERTIME_POLICIES:
            raise ValueError(
                f"Unknown overtime policy: {self.overtime_policy}"
            )

        if self.n_shippers <= 0:
            raise ValueError("n_shippers must be > 0.")

        if self.shift_minutes <= 0.0:
            raise ValueError("shift_minutes must be > 0.")

        if self.max_overtime_minutes < 0.0:
            raise ValueError("max_overtime_minutes must be >= 0.")

        if self.capacity_kg <= 0.0:
            raise ValueError("capacity_kg must be > 0.")

        if self.cluster_target_size <= 0:
            raise ValueError("cluster_target_size must be > 0.")

    @property
    def wage_per_minute(self) -> float:
        return self.wage_per_hour / 60.0


@dataclass(frozen=True)
class Stop:
    order: Order
    arrival_minutes: float


@dataclass(frozen=True)
class RouteTimeline:
    stops: tuple[Stop, ...]
    finish_minutes: float
    distance_km: float
    trips: int


@dataclass(frozen=True)
class ShipperDayResult:
    shipper_index: int
    delivered: tuple[Stop, ...]
    deferred: tuple[Order, ...]
    finish_minutes: float
    distance_km: float
    overtime_minutes: float
    trips: int


@dataclass(frozen=True)
class DayRecord:
    day: int

    incoming_orders: int
    backlog_in: int

    delivered: int
    delivered_on_time: int
    deferred_out: int

    shippers_used: int
    work_minutes: float
    overtime_minutes: float
    distance_km: float

    wage_cost: float
    overtime_cost: float
    distance_cost: float
    defer_cost: float
    late_cost: float

    # Shipper-days whose overtime exceeds max_overtime_minutes.
    over_cap_shipper_days: int = 0

    @property
    def total_cost(self) -> float:
        return (
            self.wage_cost
            + self.overtime_cost
            + self.distance_cost
            + self.defer_cost
            + self.late_cost
        )


@dataclass(frozen=True)
class SimulationResult:
    config: PlanningConfig
    days: tuple[DayRecord, ...]
    final_backlog: int
    total_orders: int

    @property
    def delivered(self) -> int:
        return sum(day.delivered for day in self.days)

    @property
    def delivered_on_time(self) -> int:
        return sum(day.delivered_on_time for day in self.days)

    @property
    def total_cost(self) -> float:
        return sum(day.total_cost for day in self.days)


# ============================================================
# ROUTE CONSTRUCTION
# ============================================================

def _path_time(
    nodes: Sequence[int],
    travel_time: np.ndarray,
) -> float:
    total = 0.0
    previous = 0

    for node in nodes:
        total += travel_time[previous, node]
        previous = node

    return total + travel_time[previous, 0]


def _two_opt(
    nodes: list[int],
    travel_time: np.ndarray,
) -> list[int]:
    """
    2-opt on one depot-to-depot trip (asymmetric travel time).
    """

    if len(nodes) < 3:
        return nodes

    best = list(nodes)
    best_time = _path_time(best, travel_time)

    improved = True

    while improved:
        improved = False

        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = (
                    best[:i]
                    + best[i : j + 1][::-1]
                    + best[j + 1 :]
                )

                candidate_time = _path_time(
                    candidate,
                    travel_time,
                )

                if candidate_time < best_time - 1e-9:
                    best = candidate
                    best_time = candidate_time
                    improved = True

    return best


def _split_trips(
    orders: Sequence[Order],
    travel_time: np.ndarray,
    capacity_kg: float,
) -> list[list[Order]]:
    """
    Nearest-neighbour construction with capacity reloads.
    """

    remaining = list(orders)
    trips: list[list[Order]] = []

    while remaining:
        trip: list[Order] = []
        load = 0.0
        current = 0

        while remaining:
            feasible = [
                order
                for order in remaining
                if not trip
                or load + order.weight <= capacity_kg
            ]

            if not feasible:
                break

            chosen = min(
                feasible,
                key=lambda order: (
                    travel_time[current, order.node],
                    order.order_id,
                ),
            )

            trip.append(chosen)
            remaining.remove(chosen)
            load += chosen.weight
            current = chosen.node

        trips.append(trip)

    return trips


def build_route(
    orders: Sequence[Order],
    *,
    distance_km: np.ndarray,
    travel_time: np.ndarray,
    config: PlanningConfig,
    improve: bool = True,
) -> RouteTimeline:
    """
    Build the day's timeline for one shipper starting at t = 0.

    Each stop costs `service_minutes`. When the vehicle is full it
    returns to the depot and reloads (no reload time is charged).
    """

    if not orders:
        return RouteTimeline((), 0.0, 0.0, 0)

    trips = _split_trips(
        orders,
        travel_time,
        config.capacity_kg,
    )

    clock = 0.0
    distance = 0.0
    stops: list[Stop] = []

    by_node: dict[int, list[Order]] = {}

    for trip in trips:
        for order in trip:
            by_node.setdefault(order.node, []).append(order)

    for trip in trips:
        nodes = [order.node for order in trip]

        if improve:
            nodes = _two_opt(nodes, travel_time)

        previous = 0

        for node in nodes:
            order = by_node[node].pop(0)

            clock += travel_time[previous, node]
            distance += distance_km[previous, node]

            stops.append(Stop(order, clock))

            clock += config.service_minutes
            previous = node

        clock += travel_time[previous, 0]
        distance += distance_km[previous, 0]

    return RouteTimeline(
        stops=tuple(stops),
        finish_minutes=clock,
        distance_km=distance,
        trips=len(trips),
    )


# ============================================================
# OVERTIME / DEFERRAL POLICY
# ============================================================

def _priority(
    order: Order,
    day: int,
) -> tuple[float, float]:
    """
    Higher tuple = more important to keep in today's route.
    """

    carried_days = day - order.release_day

    deadline = (
        order.deadline_minutes
        if order.deadline_minutes is not None
        else 1e9
    )

    return (float(carried_days), -deadline)


def _overtime(
    finish_minutes: float,
    config: PlanningConfig,
) -> float:
    return max(0.0, finish_minutes - config.shift_minutes)


def plan_shipper_day(
    orders: Sequence[Order],
    *,
    day: int,
    shipper_index: int,
    distance_km: np.ndarray,
    travel_time: np.ndarray,
    config: PlanningConfig,
) -> ShipperDayResult:
    """
    Decide which of the shipper's orders are delivered today.

    overtime:
        deliver everything, however long it takes.

    defer:
        the route must be back at the depot within the shift;
        lowest-priority orders are carried to the next day.

    hybrid:
        up to `max_overtime_minutes` is allowed, and an order is
        deferred only when the overtime it would cost is higher
        than its defer penalty.
    """

    kept = sorted(
        orders,
        key=lambda order: _priority(order, day),
        reverse=True,
    )

    deferred: list[Order] = []

    def route(
        subset: Sequence[Order],
        improve: bool,
    ) -> RouteTimeline:
        return build_route(
            subset,
            distance_km=distance_km,
            travel_time=travel_time,
            config=config,
            improve=improve,
        )

    if config.overtime_policy == "overtime":
        limit = float("inf")
    elif config.overtime_policy == "defer":
        limit = config.shift_minutes
    else:
        limit = (
            config.shift_minutes
            + config.max_overtime_minutes
        )

    # Every stop costs at least `service_minutes`, so more orders than
    # this can never fit; defer them without building a route.
    if limit != float("inf"):
        max_stops = max(1, int(limit // config.service_minutes))

        if len(kept) > max_stops:
            deferred.extend(kept[max_stops:])
            kept = kept[:max_stops]

    timeline = route(kept, improve=False)

    while kept and timeline.finish_minutes > limit:
        # Drop in batches while far over the limit, then one by one.
        per_order = timeline.finish_minutes / len(kept)

        batch = max(
            1,
            int(
                0.5
                * (timeline.finish_minutes - limit)
                / per_order
            ),
        )

        for _ in range(min(batch, len(kept))):
            deferred.append(kept.pop())

        timeline = route(kept, improve=False)

    timeline = route(kept, improve=True)

    # 2-opt can only shorten the route; re-check anyway.
    while kept and timeline.finish_minutes > limit:
        deferred.append(kept.pop())
        timeline = route(kept, improve=True)

    if config.overtime_policy == "hybrid":
        # Drop an order when the overtime it costs is higher
        # than the penalty for carrying it one more day.
        while kept:
            overtime_now = _overtime(
                timeline.finish_minutes,
                config,
            )

            if overtime_now <= 0.0:
                break

            candidate = kept[-1]

            reduced = route(kept[:-1], improve=True)

            overtime_saved_minutes = (
                overtime_now
                - _overtime(reduced.finish_minutes, config)
            )

            overtime_saving = (
                overtime_saved_minutes
                * config.wage_per_minute
                * config.overtime_multiplier
            )

            defer_cost = (
                config.defer_penalty_per_order_day
                * (1.0 + day - candidate.release_day)
            )

            if overtime_saving <= defer_cost:
                break

            deferred.append(kept.pop())
            timeline = reduced

    return ShipperDayResult(
        shipper_index=shipper_index,
        delivered=timeline.stops,
        deferred=tuple(deferred),
        finish_minutes=timeline.finish_minutes,
        distance_km=timeline.distance_km,
        overtime_minutes=_overtime(
            timeline.finish_minutes,
            config,
        ),
        trips=timeline.trips,
    )


# ============================================================
# CLUSTER -> SHIPPER ALLOCATION
# ============================================================

def cluster_orders(
    orders: Sequence[Order],
    config: PlanningConfig,
) -> list[list[Order]]:
    """
    K-Means on projected coordinates.

    K = ceil(n / target size), capped by number of shippers.
    """

    if not orders:
        return []

    k = min(
        config.n_shippers,
        len(orders),
        max(
            1,
            -(-len(orders) // config.cluster_target_size),
        ),
    )

    if k == 1:
        return [list(orders)]

    coordinates = np.array(
        [
            [order.latitude, order.longitude]
            for order in orders
        ],
        dtype=np.float64,
    )

    projection = LocalGeoProjection.from_coordinates(
        coordinates
    )

    projected = projection.transform(coordinates)

    # Duplicate points would make K-Means warn about
    # fewer distinct clusters than requested.
    distinct = len(np.unique(projected, axis=0))
    k = min(k, distinct)

    labels = KMeans(
        n_clusters=k,
        random_state=config.seed,
        n_init=10,
    ).fit_predict(projected)

    groups: dict[int, list[Order]] = {}

    for order, label in zip(orders, labels):
        groups.setdefault(int(label), []).append(order)

    return [
        groups[label]
        for label in sorted(groups)
    ]


def _cluster_gap(
    first: Sequence[Order],
    second: Sequence[Order],
    travel_time: np.ndarray,
) -> float:
    a = [order.node for order in first]
    b = [order.node for order in second]

    forward = travel_time[np.ix_(a, b)]
    backward = travel_time[np.ix_(b, a)]

    return float(min(forward.min(), backward.min()))


def merge_small_clusters(
    clusters: list[list[Order]],
    *,
    distance_km: np.ndarray,
    travel_time: np.ndarray,
    config: PlanningConfig,
) -> list[list[Order]]:
    """
    A small cluster is handed to the nearest other cluster's shipper,
    provided that shipper still finishes within the regular shift.
    """

    merged = [list(cluster) for cluster in clusters]

    small_indices = sorted(
        (
            index
            for index, cluster in enumerate(merged)
            if len(cluster) < config.small_cluster_threshold
        ),
        key=lambda index: len(merged[index]),
    )

    for index in small_indices:
        cluster = merged[index]

        if not cluster:
            continue

        targets = sorted(
            (
                other
                for other, members in enumerate(merged)
                if other != index and members
            ),
            key=lambda other: _cluster_gap(
                cluster,
                merged[other],
                travel_time,
            ),
        )

        for other in targets:
            combined = merged[other] + cluster

            timeline = build_route(
                combined,
                distance_km=distance_km,
                travel_time=travel_time,
                config=config,
                improve=False,
            )

            if timeline.finish_minutes <= config.shift_minutes:
                merged[other] = combined
                merged[index] = []
                break

    return [cluster for cluster in merged if cluster]


def allocate_orders(
    orders: Sequence[Order],
    *,
    distance_km: np.ndarray,
    travel_time: np.ndarray,
    config: PlanningConfig,
) -> list[list[Order]]:
    clusters = cluster_orders(orders, config)

    if config.allocation == "merge_small":
        clusters = merge_small_clusters(
            clusters,
            distance_km=distance_km,
            travel_time=travel_time,
            config=config,
        )

    return clusters


# ============================================================
# MULTI-DAY SIMULATION
# ============================================================

def simulate_horizon(
    orders_by_day: Mapping[int, Sequence[Order]],
    *,
    distance_km: np.ndarray,
    travel_time: np.ndarray,
    config: PlanningConfig,
    n_days: int,
) -> SimulationResult:
    """
    Simulate `n_days` consecutive days. Orders that are not delivered
    stay in the backlog and are prioritised on the next day.
    """

    backlog: list[Order] = []
    records: list[DayRecord] = []
    total_orders = 0

    for day in range(n_days):
        incoming = list(orders_by_day.get(day, ()))
        total_orders += len(incoming)

        pending = backlog + incoming

        clusters = allocate_orders(
            pending,
            distance_km=distance_km,
            travel_time=travel_time,
            config=config,
        )

        results = [
            plan_shipper_day(
                cluster,
                day=day,
                shipper_index=index,
                distance_km=distance_km,
                travel_time=travel_time,
                config=config,
            )
            for index, cluster in enumerate(clusters)
        ]

        backlog = [
            order
            for result in results
            for order in result.deferred
        ]

        delivered = 0
        on_time = 0
        late = 0

        for result in results:
            for stop in result.delivered:
                delivered += 1

                carried = day - stop.order.release_day

                is_on_time = carried == 0 and (
                    stop.order.deadline_minutes is None
                    or stop.arrival_minutes
                    <= stop.order.deadline_minutes
                )

                if is_on_time:
                    on_time += 1
                else:
                    late += 1

        used = [
            result
            for result in results
            if result.delivered
        ]

        regular_minutes = sum(
            config.shift_minutes for _ in used
        )

        overtime_minutes = sum(
            result.overtime_minutes for result in used
        )

        distance = sum(
            result.distance_km for result in results
        )

        defer_cost = sum(
            config.defer_penalty_per_order_day
            * (1.0 + day - order.release_day)
            for order in backlog
        )

        records.append(
            DayRecord(
                day=day,
                incoming_orders=len(incoming),
                backlog_in=len(pending) - len(incoming),
                delivered=delivered,
                delivered_on_time=on_time,
                deferred_out=len(backlog),
                shippers_used=len(used),
                work_minutes=regular_minutes + overtime_minutes,
                overtime_minutes=overtime_minutes,
                distance_km=distance,
                wage_cost=regular_minutes * config.wage_per_minute,
                overtime_cost=(
                    overtime_minutes
                    * config.wage_per_minute
                    * config.overtime_multiplier
                ),
                distance_cost=distance * config.cost_per_km,
                defer_cost=defer_cost,
                late_cost=late * config.late_delivery_penalty,
                over_cap_shipper_days=sum(
                    result.overtime_minutes
                    > config.max_overtime_minutes + 1e-9
                    for result in used
                ),
            )
        )

    return SimulationResult(
        config=config,
        days=tuple(records),
        final_backlog=len(backlog),
        total_orders=total_orders,
    )


__all__ = [
    "ALLOCATION_POLICIES",
    "OVERTIME_POLICIES",
    "DayRecord",
    "Order",
    "PlanningConfig",
    "RouteTimeline",
    "ShipperDayResult",
    "SimulationResult",
    "Stop",
    "allocate_orders",
    "build_route",
    "cluster_orders",
    "merge_small_clusters",
    "plan_shipper_day",
    "simulate_horizon",
]
