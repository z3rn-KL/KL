import networkx as nx
import numpy as np
import pytest

from routing import (
    RoadMatrixBuilder,
    RoadNetworkService,
)


def create_test_network():
    graph = nx.MultiDiGraph()

    graph.add_node(
        1,
        x=106.70,
        y=10.77,
    )

    graph.add_node(
        2,
        x=106.71,
        y=10.78,
    )

    graph.add_node(
        3,
        x=106.72,
        y=10.79,
    )

    graph.add_edge(
        1,
        2,
        length=1000.0,
        travel_time=120.0,
    )

    graph.add_edge(
        2,
        3,
        length=500.0,
        travel_time=60.0,
    )

    graph.add_edge(
        1,
        3,
        length=3000.0,
        travel_time=500.0,
    )

    graph.add_edge(
        3,
        1,
        length=2000.0,
        travel_time=240.0,
    )

    service = RoadNetworkService(
        graph
    )

    return service


def test_matrix_shape():
    service = create_test_network()

    builder = RoadMatrixBuilder(
        service
    )

    result = builder.build(
        [
            1,
            2,
            3,
        ]
    )

    assert (
        result.distance_km.shape
        == (
            3,
            3,
        )
    )

    assert (
        result.travel_time_minutes.shape
        == (
            3,
            3,
        )
    )


def test_diagonal_is_zero():
    service = create_test_network()

    builder = RoadMatrixBuilder(
        service
    )

    result = builder.build(
        [
            1,
            2,
            3,
        ]
    )

    assert np.allclose(
        np.diag(
            result.distance_km
        ),
        0.0,
    )

    assert np.allclose(
        np.diag(
            result.travel_time_minutes
        ),
        0.0,
    )


def test_shortest_distance_matrix():
    service = create_test_network()

    builder = RoadMatrixBuilder(
        service
    )

    result = builder.build(
        [
            1,
            2,
            3,
        ]
    )

    # 1 -> 3 should use:
    #
    # 1 -> 2 = 1.0 km
    # 2 -> 3 = 0.5 km
    #
    # total = 1.5 km
    #
    # instead of direct 3.0 km.

    assert result.distance_km[
        0,
        2,
    ] == pytest.approx(
        1.5
    )


def test_shortest_travel_time_matrix():
    service = create_test_network()

    builder = RoadMatrixBuilder(
        service
    )

    result = builder.build(
        [
            1,
            2,
            3,
        ]
    )

    # 1 -> 2 = 120 sec
    # 2 -> 3 = 60 sec
    #
    # total = 180 sec = 3 minutes.

    assert result.travel_time_minutes[
        0,
        2,
    ] == pytest.approx(
        3.0
    )


def test_matrix_is_directional():
    service = create_test_network()

    builder = RoadMatrixBuilder(
        service
    )

    result = builder.build(
        [
            1,
            3,
        ]
    )

    distance_1_to_3 = (
        result.distance_km[
            0,
            1,
        ]
    )

    distance_3_to_1 = (
        result.distance_km[
            1,
            0,
        ]
    )

    assert distance_1_to_3 == pytest.approx(
        1.5
    )

    assert distance_3_to_1 == pytest.approx(
        2.0
    )

    assert (
        distance_1_to_3
        != distance_3_to_1
    )


def test_unknown_node_is_rejected():
    service = create_test_network()

    builder = RoadMatrixBuilder(
        service
    )

    with pytest.raises(
        ValueError
    ):
        builder.build(
            [
                1,
                999,
            ]
        )


def test_missing_travel_time_is_rejected():
    graph = nx.MultiDiGraph()

    graph.add_node(
        1
    )

    graph.add_node(
        2
    )

    graph.add_edge(
        1,
        2,
        length=1000.0,
    )

    service = RoadNetworkService(
        graph
    )

    builder = RoadMatrixBuilder(
        service
    )

    with pytest.raises(
        ValueError
    ):
        builder.build(
            [
                1,
                2,
            ]
        )


def test_duplicate_nodes_are_supported():
    service = create_test_network()

    builder = RoadMatrixBuilder(
        service
    )

    result = builder.build(
        [
            1,
            1,
            3,
        ]
    )

    assert result.size() == 3

    assert result.distance_km[
        0,
        1,
    ] == pytest.approx(
        0.0
    )

    assert result.distance_km[
        1,
        2,
    ] == pytest.approx(
        1.5
    )