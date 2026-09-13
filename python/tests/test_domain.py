from domain import (
    Delivery,
    Depot,
    Vehicle,
    Shipper,
    Cluster,
    Route,
    Assignment,
    RLState,
)


def test_create_delivery():
    delivery = Delivery(
        delivery_id="D001",
        latitude=10.77,
        longitude=106.70,
        weight=5.0,
    )

    assert delivery.delivery_id == "D001"
    assert delivery.weight == 5.0


def test_create_depot():
    depot = Depot(
        depot_id="DEPOT_01",
        name="Kho 1",
        latitude=10.77,
        longitude=106.70,
    )

    assert depot.depot_id == "DEPOT_01"


def test_vehicle():
    vehicle = Vehicle(
        vehicle_id="V01",
        depot_id="DEPOT_01",
        capacity_kg=100.0,
    )

    assert vehicle.capacity_kg == 100.0


def test_rl_terminal():
    state = RLState(
        current_location_id="D003",
        unvisited_delivery_ids=(),
    )

    assert state.is_terminal()