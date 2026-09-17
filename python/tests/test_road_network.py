import networkx as nx
import pytest

from routing import RoadNetworkService


def create_test_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    graph.add_node(
        1,
        x=106.7000,
        y=10.7700,
    )

    graph.add_node(
        2,
        x=106.7010,
        y=10.7710,
    )

    graph.add_node(
        3,
        x=106.7020,
        y=10.7720,
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
        length=2000.0,
        travel_time=300.0,
    )

    graph.graph["crs"] = "EPSG:4326"

    return graph


def test_empty_graph_rejected():
    graph = nx.MultiDiGraph()

    with pytest.raises(ValueError):
        RoadNetworkService(graph)


def test_shortest_distance():
    service = RoadNetworkService(
        create_test_graph()
    )

    distance_m = service.shortest_distance_m(
        1,
        3,
    )

    assert distance_m == 1500.0


def test_shortest_distance_km():
    service = RoadNetworkService(
        create_test_graph()
    )

    distance_km = service.shortest_distance_km(
        1,
        3,
    )

    assert distance_km == 1.5


def test_shortest_travel_time():
    service = RoadNetworkService(
        create_test_graph()
    )

    travel_time_minutes = (
        service.shortest_travel_time_minutes(
            1,
            3,
        )
    )

    assert travel_time_minutes == 3.0


def test_shortest_path_nodes():
    service = RoadNetworkService(
        create_test_graph()
    )

    path = service.shortest_path_nodes(
        1,
        3,
        weight="length",
    )

    assert path == [
        1,
        2,
        3,
    ]