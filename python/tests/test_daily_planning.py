import numpy as np
import pytest

from simulation.daily_planning import (
    Order,
    PlanningConfig,
    allocate_orders,
    build_route,
    plan_shipper_day,
    simulate_horizon,
)


def make_matrices(n: int, leg_minutes: float = 30.0):
    """
    Nodes 0..n; every pair is `leg_minutes` apart (1 km per 2 min).
    """

    time = np.full((n + 1, n + 1), leg_minutes)
    np.fill_diagonal(time, 0.0)

    return time / 2.0, time


def make_orders(n, release_day=0, weight=1.0, deadline=None):
    return [
        Order(
            order_id=f"o{release_day}_{i}",
            node=i + 1,
            latitude=10.77 + 0.01 * i,
            longitude=106.70 + 0.01 * (i % 3),
            weight=weight,
            release_day=release_day,
            deadline_minutes=deadline,
        )
        for i in range(n)
    ]


def test_build_route_timeline_includes_service_time():
    distance, time = make_matrices(2, leg_minutes=30.0)
    config = PlanningConfig(service_minutes=10.0)

    route = build_route(
        make_orders(2),
        distance_km=distance,
        travel_time=time,
        config=config,
    )

    # 30 + 10 + 30 + 10 + 30 (return)
    assert route.finish_minutes == pytest.approx(110.0)
    assert [stop.arrival_minutes for stop in route.stops] == [30.0, 70.0]
    assert route.trips == 1


def test_build_route_reloads_when_capacity_exceeded():
    distance, time = make_matrices(3)
    config = PlanningConfig(capacity_kg=10.0)

    route = build_route(
        make_orders(3, weight=6.0),
        distance_km=distance,
        travel_time=time,
        config=config,
    )

    assert route.trips == 3
    assert len(route.stops) == 3


def test_overtime_policy_delivers_everything():
    distance, time = make_matrices(20)
    config = PlanningConfig(overtime_policy="overtime")

    result = plan_shipper_day(
        make_orders(20),
        day=0,
        shipper_index=0,
        distance_km=distance,
        travel_time=time,
        config=config,
    )

    assert not result.deferred
    assert result.overtime_minutes > 0.0


def test_defer_policy_respects_shift_limit():
    distance, time = make_matrices(20)
    config = PlanningConfig(overtime_policy="defer")

    result = plan_shipper_day(
        make_orders(20),
        day=0,
        shipper_index=0,
        distance_km=distance,
        travel_time=time,
        config=config,
    )

    assert result.deferred
    assert result.finish_minutes <= config.shift_minutes + 1e-9
    assert result.overtime_minutes == 0.0
    assert len(result.delivered) + len(result.deferred) == 20


def test_defer_policy_keeps_older_orders_first():
    distance, time = make_matrices(20)
    config = PlanningConfig(overtime_policy="defer")

    old = make_orders(10, release_day=0)
    new = [
        Order(f"n{i}", 11 + i, 10.8, 106.7, 1.0, 3, None)
        for i in range(10)
    ]

    result = plan_shipper_day(
        old + new,
        day=3,
        shipper_index=0,
        distance_km=distance,
        travel_time=time,
        config=config,
    )

    delivered_ids = {stop.order.order_id for stop in result.delivered}

    assert all(order.order_id in delivered_ids for order in old)
    assert all(order.release_day == 3 for order in result.deferred)


def test_hybrid_overtime_is_capped():
    distance, time = make_matrices(30)
    config = PlanningConfig(
        overtime_policy="hybrid",
        max_overtime_minutes=60.0,
        defer_penalty_per_order_day=1e9,
    )

    result = plan_shipper_day(
        make_orders(30),
        day=0,
        shipper_index=0,
        distance_km=distance,
        travel_time=time,
        config=config,
    )

    assert result.overtime_minutes <= 60.0 + 1e-9
    assert result.deferred


def test_hybrid_defers_when_penalty_is_cheap():
    distance, time = make_matrices(20)

    cheap = PlanningConfig(
        overtime_policy="hybrid",
        defer_penalty_per_order_day=1.0,
    )
    dear = PlanningConfig(
        overtime_policy="hybrid",
        defer_penalty_per_order_day=1e9,
    )

    def run(config):
        return plan_shipper_day(
            make_orders(20),
            day=0,
            shipper_index=0,
            distance_km=distance,
            travel_time=time,
            config=config,
        )

    assert run(cheap).overtime_minutes == 0.0
    assert len(run(cheap).deferred) > len(run(dear).deferred)


def test_merge_small_uses_fewer_shippers():
    distance, time = make_matrices(8)

    orders = make_orders(8)

    dedicated = allocate_orders(
        orders,
        distance_km=distance,
        travel_time=time,
        config=PlanningConfig(
            allocation="dedicated",
            cluster_target_size=2,
            small_cluster_threshold=3,
        ),
    )

    merged = allocate_orders(
        orders,
        distance_km=distance,
        travel_time=time,
        config=PlanningConfig(
            allocation="merge_small",
            cluster_target_size=2,
            small_cluster_threshold=3,
        ),
    )

    assert sum(len(c) for c in dedicated) == 8
    assert sum(len(c) for c in merged) == 8
    assert len(merged) < len(dedicated)


def test_merge_refused_when_target_cannot_finish_shift():
    distance, time = make_matrices(8)

    def clusters(allocation):
        return allocate_orders(
            make_orders(8),
            distance_km=distance,
            travel_time=time,
            config=PlanningConfig(
                allocation=allocation,
                cluster_target_size=2,
                small_cluster_threshold=3,
                # Too short for any cluster to take another's orders.
                shift_minutes=80.0,
            ),
        )

    assert len(clusters("merge_small")) == len(clusters("dedicated"))


def test_backlog_carries_over_and_clears():
    distance, time = make_matrices(40)
    config = PlanningConfig(overtime_policy="defer", n_shippers=1)

    orders_by_day = {0: make_orders(40, release_day=0)}

    result = simulate_horizon(
        orders_by_day,
        distance_km=distance,
        travel_time=time,
        config=config,
        n_days=6,
    )

    assert result.days[0].deferred_out > 0
    assert result.days[1].backlog_in == result.days[0].deferred_out
    assert result.final_backlog == 0
    assert result.delivered == 40
    # Only orders delivered on release day can be on time.
    assert result.delivered_on_time < 40


def test_simulation_conserves_orders():
    distance, time = make_matrices(30)

    orders_by_day = {
        0: make_orders(15, release_day=0),
        1: [
            Order(f"b{i}", 16 + i, 10.8, 106.7, 1.0, 1, None)
            for i in range(15)
        ],
    }

    for policy in ("overtime", "defer", "hybrid"):
        result = simulate_horizon(
            orders_by_day,
            distance_km=distance,
            travel_time=time,
            config=PlanningConfig(overtime_policy=policy),
            n_days=3,
        )

        assert result.delivered + result.final_backlog == 30
        assert result.total_orders == 30
        assert result.total_cost > 0.0


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        PlanningConfig(overtime_policy="nope")

    with pytest.raises(ValueError):
        PlanningConfig(allocation="nope")
