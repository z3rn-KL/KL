from datetime import datetime, timedelta

import networkx as nx
import pytest

from domain import Delivery

from routing import (
    RoadNetworkService,
)

from routing.priority_nearest_neighbor import (
    PriorityNearestNeighborRouter,
)


def create_test_network():
    graph = nx.MultiDiGraph()

    for node in [
        1,
        2,
        3,
        4,
    ]:
        graph.add_node(
            node
        )

    edges = [
        (
            1,
            2,
            1000.0,
            120.0,
        ),
        (
            1,
            3,
            2000.0,
            240.0,
        ),
        (
            1,
            4,
            3000.0,
            360.0,
        ),
        (
            2,
            1,
            1000.0,
            120.0,
        ),
        (
            3,
            1,
            2000.0,
            240.0,
        ),
        (
            4,
            1,
            3000.0,
            360.0,
        ),
        (
            2,
            3,
            1000.0,
            120.0,
        ),
        (
            2,
            4,
            2000.0,
            240.0,
        ),
        (
            3,
            2,
            1000.0,
            120.0,
        ),
        (
            3,
            4,
            1000.0,
            120.0,
        ),
        (
            4,
            2,
            2000.0,
            240.0,
        ),
        (
            4,
            3,
            1000.0,
            120.0,
        ),
    ]

    for (
        origin,
        destination,
        length,
        travel_time,
    ) in edges:
        graph.add_edge(
            origin,
            destination,
            length=length,
            travel_time=travel_time,
        )

    return RoadNetworkService(
        graph
    )


def create_delivery(
    delivery_id: str,
    now: datetime,
    waiting_minutes: int,
    remaining_deadline_minutes: int,
    service_type: str,
) -> Delivery:
    created_at = (
        now
        - timedelta(
            minutes=waiting_minutes
        )
    )

    expected_delivery_time = (
        now
        + timedelta(
            minutes=(
                remaining_deadline_minutes
            )
        )
    )

    return Delivery(
        delivery_id=delivery_id,
        latitude=10.0,
        longitude=106.0,
        created_at=created_at,
        expected_delivery_time=(
            expected_delivery_time
        ),
        service_type=service_type,
    )


def test_more_urgent_delivery_can_override_nearest():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    service = create_test_network()

    near_delivery = create_delivery(
        delivery_id="NEAR",
        now=now,
        waiting_minutes=10,
        remaining_deadline_minutes=300,
        service_type="5h",
    )

    urgent_delivery = create_delivery(
        delivery_id="URGENT",
        now=now,
        waiting_minutes=170,
        remaining_deadline_minutes=10,
        service_type="3h",
    )

    router = (
        PriorityNearestNeighborRouter(
            service
        )
    )

    route = router.build_route(
        depot_node=1,
        deliveries=[
            near_delivery,
            urgent_delivery,
        ],
        delivery_nodes=[
            2,
            3,
        ],
        start_time=now,
    )

    assert (
        route.delivery_order[0]
        == "URGENT"
    )


def test_all_deliveries_are_visited():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    service = create_test_network()

    deliveries = [
        create_delivery(
            delivery_id="A",
            now=now,
            waiting_minutes=60,
            remaining_deadline_minutes=120,
            service_type="3h",
        ),
        create_delivery(
            delivery_id="B",
            now=now,
            waiting_minutes=30,
            remaining_deadline_minutes=240,
            service_type="5h",
        ),
        create_delivery(
            delivery_id="C",
            now=now,
            waiting_minutes=90,
            remaining_deadline_minutes=180,
            service_type="3h",
        ),
    ]

    router = (
        PriorityNearestNeighborRouter(
            service
        )
    )

    route = router.build_route(
        depot_node=1,
        deliveries=deliveries,
        delivery_nodes=[
            2,
            3,
            4,
        ],
        start_time=now,
    )

    assert set(
        route.delivery_order
    ) == {
        "A",
        "B",
        "C",
    }

    assert (
        route.number_of_deliveries()
        == 3
    )


def test_route_returns_to_depot():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    service = create_test_network()

    delivery = create_delivery(
        delivery_id="A",
        now=now,
        waiting_minutes=60,
        remaining_deadline_minutes=120,
        service_type="3h",
    )

    router = (
        PriorityNearestNeighborRouter(
            service
        )
    )

    route = router.build_route(
        depot_node=1,
        deliveries=[
            delivery
        ],
        delivery_nodes=[
            2
        ],
        start_time=now,
        return_to_depot=True,
    )

    assert (
        route.node_order
        == (
            1,
            2,
            1,
        )
    )

    assert (
        route.returned_to_depot
        is True
    )


def test_route_can_skip_return_to_depot():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    service = create_test_network()

    delivery = create_delivery(
        delivery_id="A",
        now=now,
        waiting_minutes=60,
        remaining_deadline_minutes=120,
        service_type="3h",
    )

    router = (
        PriorityNearestNeighborRouter(
            service
        )
    )

    route = router.build_route(
        depot_node=1,
        deliveries=[
            delivery
        ],
        delivery_nodes=[
            2
        ],
        start_time=now,
        return_to_depot=False,
    )

    assert (
        route.node_order
        == (
            1,
            2,
        )
    )

    assert (
        route.returned_to_depot
        is False
    )


def test_current_time_advances_after_each_leg():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    service = create_test_network()

    delivery = create_delivery(
        delivery_id="A",
        now=now,
        waiting_minutes=60,
        remaining_deadline_minutes=120,
        service_type="3h",
    )

    router = (
        PriorityNearestNeighborRouter(
            service
        )
    )

    route = router.build_route(
        depot_node=1,
        deliveries=[
            delivery
        ],
        delivery_nodes=[
            2
        ],
        start_time=now,
        return_to_depot=False,
    )

    step = route.steps[
        0
    ]

    assert (
        step.departure_time
        == now
    )

    assert (
        step.arrival_time
        == (
            now
            + timedelta(
                minutes=2
            )
        )
    )

    assert (
        route.end_time
        == step.arrival_time
    )


def test_single_delivery_has_full_travel_efficiency():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    service = create_test_network()

    delivery = create_delivery(
        delivery_id="A",
        now=now,
        waiting_minutes=60,
        remaining_deadline_minutes=120,
        service_type="3h",
    )

    router = (
        PriorityNearestNeighborRouter(
            service
        )
    )

    route = router.build_route(
        depot_node=1,
        deliveries=[
            delivery
        ],
        delivery_nodes=[
            2
        ],
        start_time=now,
    )

    assert (
        route.steps[
            0
        ].travel_efficiency_score
        == pytest.approx(
            1.0
        )
    )


def test_step_contains_priority_information():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    service = create_test_network()

    delivery = create_delivery(
        delivery_id="A",
        now=now,
        waiting_minutes=60,
        remaining_deadline_minutes=120,
        service_type="3h",
    )

    router = (
        PriorityNearestNeighborRouter(
            service
        )
    )

    route = router.build_route(
        depot_node=1,
        deliveries=[
            delivery
        ],
        delivery_nodes=[
            2
        ],
        start_time=now,
    )

    step = route.steps[
        0
    ]

    assert (
        0.0
        <= step.priority_score
        <= 1.0
    )

    assert (
        step.deadline_urgency
        is not None
    )

    assert (
        step.service_priority
        is not None
    )

    assert (
        step.waiting_time_score
        is not None
    )

    assert (
        step.travel_efficiency_score
        is not None
    )

    assert (
        step.information_coverage
        == pytest.approx(
            1.0
        )
    )


def test_mismatched_inputs_are_rejected():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    service = create_test_network()

    delivery = create_delivery(
        delivery_id="A",
        now=now,
        waiting_minutes=60,
        remaining_deadline_minutes=120,
        service_type="3h",
    )

    router = (
        PriorityNearestNeighborRouter(
            service
        )
    )

    with pytest.raises(
        ValueError
    ):
        router.build_route(
            depot_node=1,
            deliveries=[
                delivery
            ],
            delivery_nodes=[
                2,
                3,
            ],
            start_time=now,
        )


def test_duplicate_delivery_ids_are_rejected():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    service = create_test_network()

    delivery_a = create_delivery(
        delivery_id="A",
        now=now,
        waiting_minutes=60,
        remaining_deadline_minutes=120,
        service_type="3h",
    )

    delivery_b = create_delivery(
        delivery_id="A",
        now=now,
        waiting_minutes=30,
        remaining_deadline_minutes=180,
        service_type="5h",
    )

    router = (
        PriorityNearestNeighborRouter(
            service
        )
    )

    with pytest.raises(
        ValueError
    ):
        router.build_route(
            depot_node=1,
            deliveries=[
                delivery_a,
                delivery_b,
            ],
            delivery_nodes=[
                2,
                3,
            ],
            start_time=now,
        )