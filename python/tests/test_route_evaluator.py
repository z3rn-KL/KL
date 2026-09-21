from datetime import datetime, timedelta

import networkx as nx
import pytest

from domain import Delivery

from evaluation import (
    RouteEvaluator,
)

from routing import (
    RoadNetworkService,
)


def create_network():
    graph = nx.MultiDiGraph()

    for node in [
        1,
        2,
        3,
    ]:
        graph.add_node(
            node
        )

    graph.add_edge(
        1,
        2,
        length=1000.0,
        travel_time=600.0,
    )

    graph.add_edge(
        2,
        3,
        length=1000.0,
        travel_time=600.0,
    )

    graph.add_edge(
        3,
        1,
        length=2000.0,
        travel_time=1200.0,
    )

    graph.add_edge(
        2,
        1,
        length=1000.0,
        travel_time=600.0,
    )

    graph.add_edge(
        3,
        2,
        length=1000.0,
        travel_time=600.0,
    )

    graph.add_edge(
        1,
        3,
        length=2000.0,
        travel_time=1200.0,
    )

    return RoadNetworkService(
        graph
    )


def create_delivery(
    delivery_id: str,
    created_at: datetime,
    deadline: datetime | None,
) -> Delivery:
    return Delivery(
        delivery_id=delivery_id,
        latitude=10.0,
        longitude=106.0,
        created_at=created_at,
        expected_delivery_time=deadline,
        service_type="3h",
    )


def test_on_time_delivery_is_counted():
    start = datetime(
        2026,
        9,
        21,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="A",
        created_at=(
            start
            - timedelta(
                minutes=30
            )
        ),
        deadline=(
            start
            + timedelta(
                minutes=20
            )
        ),
    )

    evaluator = RouteEvaluator(
        create_network()
    )

    result = evaluator.evaluate(
        algorithm="test",
        depot_node=1,
        deliveries=[
            delivery
        ],
        delivery_nodes=[
            2
        ],
        delivery_order=[
            "A"
        ],
        start_time=start,
    )

    assert (
        result.on_time_deliveries
        == 1
    )

    assert (
        result.late_deliveries
        == 0
    )

    assert (
        result.on_time_rate
        == pytest.approx(1.0)
    )


def test_late_delivery_is_counted():
    start = datetime(
        2026,
        9,
        21,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="A",
        created_at=(
            start
            - timedelta(
                minutes=30
            )
        ),
        deadline=(
            start
            + timedelta(
                minutes=5
            )
        ),
    )

    evaluator = RouteEvaluator(
        create_network()
    )

    result = evaluator.evaluate(
        algorithm="test",
        depot_node=1,
        deliveries=[
            delivery
        ],
        delivery_nodes=[
            2
        ],
        delivery_order=[
            "A"
        ],
        start_time=start,
    )

    assert (
        result.on_time_deliveries
        == 0
    )

    assert (
        result.late_deliveries
        == 1
    )

    assert (
        result.max_lateness_minutes
        == pytest.approx(
            5.0
        )
    )


def test_missing_deadline_is_excluded():
    start = datetime(
        2026,
        9,
        21,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="A",
        created_at=start,
        deadline=None,
    )

    evaluator = RouteEvaluator(
        create_network()
    )

    result = evaluator.evaluate(
        algorithm="test",
        depot_node=1,
        deliveries=[
            delivery
        ],
        delivery_nodes=[
            2
        ],
        delivery_order=[
            "A"
        ],
        start_time=start,
    )

    assert (
        result.deliveries_with_deadline
        == 0
    )

    assert (
        result.deliveries_without_deadline
        == 1
    )

    assert (
        result.on_time_rate
        is None
    )


def test_route_distance_and_time_are_correct():
    start = datetime(
        2026,
        9,
        21,
        10,
        0,
    )

    delivery_a = create_delivery(
        delivery_id="A",
        created_at=start,
        deadline=(
            start
            + timedelta(
                hours=2
            )
        ),
    )

    delivery_b = create_delivery(
        delivery_id="B",
        created_at=start,
        deadline=(
            start
            + timedelta(
                hours=2
            )
        ),
    )

    evaluator = RouteEvaluator(
        create_network()
    )

    result = evaluator.evaluate(
        algorithm="test",
        depot_node=1,
        deliveries=[
            delivery_a,
            delivery_b,
        ],
        delivery_nodes=[
            2,
            3,
        ],
        delivery_order=[
            "A",
            "B",
        ],
        start_time=start,
        return_to_depot=True,
    )

    # 1 -> 2 = 1 km
    # 2 -> 3 = 1 km
    # 3 -> 1 = 2 km

    assert (
        result.total_distance_km
        == pytest.approx(
            4.0
        )
    )

    # 10 + 10 + 20 minutes

    assert (
        result.total_travel_time_minutes
        == pytest.approx(
            40.0
        )
    )


def test_arrival_times_follow_route_order():
    start = datetime(
        2026,
        9,
        21,
        10,
        0,
    )

    delivery_a = create_delivery(
        delivery_id="A",
        created_at=start,
        deadline=None,
    )

    delivery_b = create_delivery(
        delivery_id="B",
        created_at=start,
        deadline=None,
    )

    evaluator = RouteEvaluator(
        create_network()
    )

    result = evaluator.evaluate(
        algorithm="test",
        depot_node=1,
        deliveries=[
            delivery_a,
            delivery_b,
        ],
        delivery_nodes=[
            2,
            3,
        ],
        delivery_order=[
            "A",
            "B",
        ],
        start_time=start,
        return_to_depot=False,
    )

    assert (
        result.delivery_results[
            0
        ].arrival_time
        == (
            start
            + timedelta(
                minutes=10
            )
        )
    )

    assert (
        result.delivery_results[
            1
        ].arrival_time
        == (
            start
            + timedelta(
                minutes=20
            )
        )
    )


def test_invalid_delivery_order_is_rejected():
    start = datetime(
        2026,
        9,
        21,
        10,
        0,
    )

    delivery_a = create_delivery(
        delivery_id="A",
        created_at=start,
        deadline=None,
    )

    delivery_b = create_delivery(
        delivery_id="B",
        created_at=start,
        deadline=None,
    )

    evaluator = RouteEvaluator(
        create_network()
    )

    with pytest.raises(
        ValueError
    ):
        evaluator.evaluate(
            algorithm="test",
            depot_node=1,
            deliveries=[
                delivery_a,
                delivery_b,
            ],
            delivery_nodes=[
                2,
                3,
            ],
            delivery_order=[
                "A",
                "A",
            ],
            start_time=start,
        )