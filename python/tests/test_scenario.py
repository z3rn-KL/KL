from domain import (
    Depot,
    Vehicle,
    Shipper,
)

from simulation import OperationalScenario


def test_operational_scenario():
    depot = Depot(
        depot_id="DEPOT_01",
        name="Kho mô phỏng 1",
        latitude=10.77,
        longitude=106.70,
    )

    vehicles = [
        Vehicle(
            vehicle_id="V01",
            depot_id="DEPOT_01",
            capacity_kg=100.0,
            cost_per_km=4000.0,
            cost_per_hour=40000.0,
        ),
        Vehicle(
            vehicle_id="V02",
            depot_id="DEPOT_01",
            capacity_kg=100.0,
            cost_per_km=4000.0,
            cost_per_hour=40000.0,
        ),
    ]

    shippers = [
        Shipper(
            shipper_id="S01",
            depot_id="DEPOT_01",
            max_orders=30,
            max_work_hours=8.0,
        ),
        Shipper(
            shipper_id="S02",
            depot_id="DEPOT_01",
            max_orders=30,
            max_work_hours=8.0,
        ),
    ]

    scenario = OperationalScenario(
        name="TEST_SCENARIO",
        depots=[depot],
        vehicles=vehicles,
        shippers=shippers,
    )

    assert scenario.number_of_depots() == 1
    assert scenario.number_of_vehicles() == 2
    assert scenario.number_of_shippers() == 2

    assert (
        scenario.total_vehicle_capacity_kg()
        == 200.0
    )

    assert (
        len(
            scenario.vehicles_for_depot(
                "DEPOT_01"
            )
        )
        == 2
    )