import networkx as nx
import pytest

from routing import (
    RoadNetworkService,
)

from routing.clarke_wright import (
    ClarkeWrightRouter,
)


def create_line_network():
    """
    Directed but symmetric test network.

    Depot = 1 at position 0
    A     = 2 at position 1
    B     = 3 at position 2
    C     = 4 at position 3

    Distance between adjacent positions:
        1 km

    Travel time:
        2 minutes / km
    """

    graph = nx.MultiDiGraph()

    positions = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
    }

    for node in positions:
        graph.add_node(
            node
        )

    for origin in positions:
        for destination in positions:
            if (
                origin
                == destination
            ):
                continue

            difference = abs(
                positions[
                    origin
                ]
                - positions[
                    destination
                ]
            )

            graph.add_edge(
                origin,
                destination,
                length=(
                    difference
                    * 1000.0
                ),
                travel_time=(
                    difference
                    * 120.0
                ),
            )

    return RoadNetworkService(
        graph
    )


def create_unreachable_network():
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
        travel_time=120.0,
    )

    graph.add_edge(
        2,
        1,
        length=1000.0,
        travel_time=120.0,
    )

    # Node 3 is disconnected.

    return RoadNetworkService(
        graph
    )


def test_clarke_wright_builds_expected_order():
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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
        route.delivery_order
        == (
            "A",
            "B",
            "C",
        )
    )


def test_route_starts_and_ends_at_depot():
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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
        route.node_order
        == (
            1,
            2,
            3,
            4,
            1,
        )
    )

    assert (
        route.returned_to_depot
        is True
    )


def test_total_distance_is_correct():
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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

    # D -> A = 1 km
    # A -> B = 1 km
    # B -> C = 1 km
    # C -> D = 3 km
    #
    # total = 6 km

    assert (
        route.total_distance_km
        == pytest.approx(
            6.0
        )
    )


def test_total_travel_time_is_correct():
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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

    # 2 + 2 + 2 + 6 minutes

    assert (
        route.total_travel_time_minutes
        == pytest.approx(
            12.0
        )
    )


def test_initial_separate_distance_is_correct():
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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

    # D -> A -> D = 2 km
    # D -> B -> D = 4 km
    # D -> C -> D = 6 km
    #
    # total = 12 km

    assert (
        route.initial_separate_distance_km
        == pytest.approx(
            12.0
        )
    )


def test_total_savings_is_correct():
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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

    # Initial:
    # 12 km
    #
    # Final:
    # 6 km
    #
    # Savings:
    # 6 km

    assert (
        route.total_savings_km
        == pytest.approx(
            6.0
        )
    )


def test_three_deliveries_require_two_merges():
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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
        route.merge_count
        == 2
    )


def test_number_of_deliveries():
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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


def test_route_without_return_to_depot():
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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

    assert (
        route.node_order
        == (
            1,
            2,
            3,
            4,
        )
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


def test_mismatched_inputs_are_rejected():
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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
    service = (
        create_line_network()
    )

    router = ClarkeWrightRouter(
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


def test_unreachable_delivery_is_rejected():
    service = (
        create_unreachable_network()
    )

    router = ClarkeWrightRouter(
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
                3,
            ],
        )