import networkx as nx
import pytest

from routing import (
    NearestNeighborRouter,
    RoadNetworkService,
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

    # Depot = node 1
    #
    # Delivery A = node 2
    # Delivery B = node 3
    # Delivery C = node 4

    graph.add_edge(
        1,
        2,
        length=1000.0,
        travel_time=120.0,
    )

    graph.add_edge(
        1,
        3,
        length=3000.0,
        travel_time=300.0,
    )

    graph.add_edge(
        1,
        4,
        length=5000.0,
        travel_time=600.0,
    )

    graph.add_edge(
        2,
        3,
        length=1000.0,
        travel_time=120.0,
    )

    graph.add_edge(
        2,
        4,
        length=4000.0,
        travel_time=480.0,
    )

    graph.add_edge(
        3,
        4,
        length=1000.0,
        travel_time=120.0,
    )

    graph.add_edge(
        2,
        1,
        length=1000.0,
        travel_time=120.0,
    )

    graph.add_edge(
        3,
        1,
        length=2000.0,
        travel_time=240.0,
    )

    graph.add_edge(
        4,
        1,
        length=3000.0,
        travel_time=360.0,
    )

    graph.add_edge(
        3,
        2,
        length=1000.0,
        travel_time=120.0,
    )

    graph.add_edge(
        4,
        3,
        length=1000.0,
        travel_time=120.0,
    )

    graph.add_edge(
        4,
        2,
        length=2000.0,
        travel_time=240.0,
    )

    return RoadNetworkService(
        graph
    )


def test_nearest_neighbor_delivery_order():
    service = create_test_network()

    router = NearestNeighborRouter(
        service,
        metric="distance",
    )

    route = router.build_route(
        depot_node=1,
        delivery_ids=[
            "A",
            "B",
            "C",
        ],
        delivery_nodes=[
            2,
            3,
            4,
        ],
    )

    assert route.delivery_order == (
        "A",
        "B",
        "C",
    )


def test_nearest_neighbor_returns_to_depot():
    service = create_test_network()

    router = NearestNeighborRouter(
        service
    )

    route = router.build_route(
        depot_node=1,
        delivery_ids=[
            "A",
            "B",
            "C",
        ],
        delivery_nodes=[
            2,
            3,
            4,
        ],
        return_to_depot=True,
    )

    assert route.node_order == (
        1,
        2,
        3,
        4,
        1,
    )

    assert route.returned_to_depot is True


def test_total_distance_is_correct():
    service = create_test_network()

    router = NearestNeighborRouter(
        service
    )

    route = router.build_route(
        depot_node=1,
        delivery_ids=[
            "A",
            "B",
            "C",
        ],
        delivery_nodes=[
            2,
            3,
            4,
        ],
    )

    # 1 -> 2 = 1 km
    # 2 -> 3 = 1 km
    # 3 -> 4 = 1 km
    # 4 -> 1 = 3 km
    #
    # total = 6 km

    assert (
        route.total_distance_km
        == pytest.approx(
            6.0
        )
    )


def test_total_travel_time_is_correct():
    service = create_test_network()

    router = NearestNeighborRouter(
        service
    )

    route = router.build_route(
        depot_node=1,
        delivery_ids=[
            "A",
            "B",
            "C",
        ],
        delivery_nodes=[
            2,
            3,
            4,
        ],
    )

    # 120 + 120 + 120 + 360
    # = 720 sec
    # = 12 minutes

    assert (
        route.total_travel_time_minutes
        == pytest.approx(
            12.0
        )
    )


def test_route_without_return_to_depot():
    service = create_test_network()

    router = NearestNeighborRouter(
        service
    )

    route = router.build_route(
        depot_node=1,
        delivery_ids=[
            "A",
            "B",
            "C",
        ],
        delivery_nodes=[
            2,
            3,
            4,
        ],
        return_to_depot=False,
    )

    assert route.node_order == (
        1,
        2,
        3,
        4,
    )

    assert (
        route.total_distance_km
        == pytest.approx(
            3.0
        )
    )

    assert (
        route.returned_to_depot
        is False
    )


def test_number_of_deliveries():
    service = create_test_network()

    router = NearestNeighborRouter(
        service
    )

    route = router.build_route(
        depot_node=1,
        delivery_ids=[
            "A",
            "B",
            "C",
        ],
        delivery_nodes=[
            2,
            3,
            4,
        ],
    )

    assert (
        route.number_of_deliveries()
        == 3
    )


def test_invalid_metric_is_rejected():
    service = create_test_network()

    with pytest.raises(
        ValueError
    ):
        NearestNeighborRouter(
            service,
            metric="invalid",
        )


def test_mismatched_delivery_inputs_are_rejected():
    service = create_test_network()

    router = NearestNeighborRouter(
        service
    )

    with pytest.raises(
        ValueError
    ):
        router.build_route(
            depot_node=1,
            delivery_ids=[
                "A",
                "B",
            ],
            delivery_nodes=[
                2,
            ],
        )


def test_duplicate_delivery_ids_are_rejected():
    service = create_test_network()

    router = NearestNeighborRouter(
        service
    )

    with pytest.raises(
        ValueError
    ):
        router.build_route(
            depot_node=1,
            delivery_ids=[
                "A",
                "A",
            ],
            delivery_nodes=[
                2,
                3,
            ],
        )


def test_travel_time_metric_is_supported():
    service = create_test_network()

    router = NearestNeighborRouter(
        service,
        metric="travel_time",
    )

    route = router.build_route(
        depot_node=1,
        delivery_ids=[
            "A",
            "B",
            "C",
        ],
        delivery_nodes=[
            2,
            3,
            4,
        ],
    )

    assert (
        route.number_of_deliveries()
        == 3
    )