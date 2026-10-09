from datetime import (
    datetime,
    timezone,
)
from types import SimpleNamespace

import networkx as nx
import pytest

from evaluation.operating_cost import (
    OperatingCostConfig,
)

from evaluation.route_evaluation_service import (
    RouteEvaluationService,
)

from routing import (
    RoadNetworkService,
)


class FakeRouteEvaluator:
    """
    Minimal stand-in for the existing RouteEvaluator.

    The goal is to test the new integration layer
    without changing or duplicating RouteEvaluator logic.
    """

    def evaluate(
        self,
        algorithm,
        depot_node,
        deliveries,
        delivery_nodes,
        delivery_order,
        start_time,
        return_to_depot,
    ):
        return SimpleNamespace(
            algorithm=algorithm,
            number_of_deliveries=(
                len(
                    deliveries
                )
            ),
            total_distance_km=12.0,
            total_travel_time_minutes=20.0,
            on_time_deliveries=1,
            late_deliveries=1,
            on_time_rate=0.5,
            total_lateness_minutes=5.0,
            max_lateness_minutes=5.0,
        )


def make_graph(
) -> nx.MultiDiGraph:
    graph = (
        nx.MultiDiGraph()
    )

    graph.graph[
        "crs"
    ] = "EPSG:4326"

    graph.add_node(
        0,
        x=106.7000,
        y=10.7700,
    )

    graph.add_node(
        1,
        x=106.7010,
        y=10.7710,
    )

    graph.add_node(
        2,
        x=106.7020,
        y=10.7720,
    )

    graph.add_edge(
        0,
        1,
        length=1000.0,
        travel_time=60.0,
    )

    graph.add_edge(
        1,
        2,
        length=1000.0,
        travel_time=60.0,
    )

    graph.add_edge(
        2,
        0,
        length=1000.0,
        travel_time=60.0,
    )

    graph.add_edge(
        1,
        0,
        length=1000.0,
        travel_time=60.0,
    )

    graph.add_edge(
        2,
        1,
        length=1000.0,
        travel_time=60.0,
    )

    graph.add_edge(
        0,
        2,
        length=1000.0,
        travel_time=60.0,
    )

    return graph


def make_deliveries():
    return [
        SimpleNamespace(
            delivery_id="A"
        ),
        SimpleNamespace(
            delivery_id="B"
        ),
    ]


def make_service(
) -> RouteEvaluationService:
    road_network = (
        RoadNetworkService(
            make_graph()
        )
    )

    cost_config = (
        OperatingCostConfig(
            distance_cost_per_km=2.0,
            travel_time_cost_per_minute=0.5,
            late_delivery_penalty=10.0,
            lateness_cost_per_minute=0.2,
            cost_unit="test_unit",
        )
    )

    return RouteEvaluationService(
        route_evaluator=(
            FakeRouteEvaluator()
        ),
        road_network=(
            road_network
        ),
        cost_config=(
            cost_config
        ),
    )


def test_combined_route_evaluation():
    service = (
        make_service()
    )

    result = service.evaluate(
        algorithm="test_algorithm",
        depot_node=0,
        deliveries=(
            make_deliveries()
        ),
        delivery_nodes=[
            1,
            2,
        ],
        delivery_order=[
            "A",
            "B",
        ],
        start_time=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        return_to_depot=True,
    )

    assert (
        result.algorithm
        == "test_algorithm"
    )

    assert (
        result.number_of_deliveries
        == 2
    )

    assert (
        result.delivery_order
        == (
            "A",
            "B",
        )
    )

    assert (
        result.total_distance_km
        == 12.0
    )

    assert (
        result.total_travel_time_minutes
        == 20.0
    )

    assert (
        result.on_time_deliveries
        == 1
    )

    assert (
        result.late_deliveries
        == 1
    )

    assert (
        result.on_time_rate
        == 0.5
    )


def test_estimated_operating_cost():
    service = (
        make_service()
    )

    result = service.evaluate(
        algorithm="test",
        depot_node=0,
        deliveries=(
            make_deliveries()
        ),
        delivery_nodes=[
            1,
            2,
        ],
        delivery_order=[
            "A",
            "B",
        ],
        start_time=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    # 12 km * 2.0
    # + 20 min * 0.5
    # + 1 late * 10
    # + 5 late min * 0.2
    #
    # = 24 + 10 + 10 + 1
    # = 45

    assert (
        result.estimated_operating_cost
        == 45.0
    )

    assert (
        result.operating_cost_breakdown
        .distance_cost
        == 24.0
    )

    assert (
        result.operating_cost_breakdown
        .travel_time_cost
        == 10.0
    )

    assert (
        result.operating_cost_breakdown
        .late_delivery_cost
        == 10.0
    )

    assert (
        result.operating_cost_breakdown
        .lateness_cost
        == 1.0
    )


def test_route_geometry_closed_route():
    service = (
        make_service()
    )

    result = service.evaluate(
        algorithm="test",
        depot_node=0,
        deliveries=(
            make_deliveries()
        ),
        delivery_nodes=[
            1,
            2,
        ],
        delivery_order=[
            "A",
            "B",
        ],
        start_time=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        return_to_depot=True,
    )

    assert (
        result.route_geometry
        == (
            (
                10.7700,
                106.7000,
            ),
            (
                10.7710,
                106.7010,
            ),
            (
                10.7720,
                106.7020,
            ),
            (
                10.7700,
                106.7000,
            ),
        )
    )


def test_route_geometry_open_route():
    service = (
        make_service()
    )

    result = service.evaluate(
        algorithm="test",
        depot_node=0,
        deliveries=(
            make_deliveries()
        ),
        delivery_nodes=[
            1,
            2,
        ],
        delivery_order=[
            "A",
            "B",
        ],
        start_time=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        return_to_depot=False,
    )

    assert (
        result.route_geometry[
            -1
        ]
        == (
            10.7720,
            106.7020,
        )
    )


def test_to_dict():
    service = (
        make_service()
    )

    result = service.evaluate(
        algorithm="test",
        depot_node=0,
        deliveries=(
            make_deliveries()
        ),
        delivery_nodes=[
            1,
            2,
        ],
        delivery_order=[
            "A",
            "B",
        ],
        start_time=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    payload = (
        result.to_dict()
    )

    assert (
        payload[
            "estimated_operating_cost"
        ]
        == 45.0
    )

    assert (
        payload[
            "operating_cost_breakdown"
        ][
            "cost_unit"
        ]
        == "test_unit"
    )


def test_delivery_node_count_mismatch():
    service = (
        make_service()
    )

    with pytest.raises(
        ValueError
    ):
        service.evaluate(
            algorithm="test",
            depot_node=0,
            deliveries=(
                make_deliveries()
            ),
            delivery_nodes=[
                1,
            ],
            delivery_order=[
                "A",
                "B",
            ],
            start_time=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )


def test_duplicate_delivery_order_rejected():
    service = (
        make_service()
    )

    with pytest.raises(
        ValueError
    ):
        service.evaluate(
            algorithm="test",
            depot_node=0,
            deliveries=(
                make_deliveries()
            ),
            delivery_nodes=[
                1,
                2,
            ],
            delivery_order=[
                "A",
                "A",
            ],
            start_time=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )


def test_unknown_delivery_order_rejected():
    service = (
        make_service()
    )

    with pytest.raises(
        ValueError
    ):
        service.evaluate(
            algorithm="test",
            depot_node=0,
            deliveries=(
                make_deliveries()
            ),
            delivery_nodes=[
                1,
                2,
            ],
            delivery_order=[
                "A",
                "C",
            ],
            start_time=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )