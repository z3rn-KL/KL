from simulation import (
    OperationalScenario,
    ScenarioFactory,
)


def test_create_operational_scenario():
    scenario = (
        ScenarioFactory
        .create_single_depot_scenario()
    )

    assert isinstance(
        scenario,
        OperationalScenario,
    )


def test_scenario_resource_counts():
    scenario = (
        ScenarioFactory
        .create_single_depot_scenario()
    )

    assert scenario.number_of_depots() == 1
    assert scenario.number_of_vehicles() == 8
    assert scenario.number_of_shippers() == 16


def test_total_vehicle_capacity():
    scenario = (
        ScenarioFactory
        .create_single_depot_scenario()
    )

    assert scenario.total_vehicle_capacity_kg == 800.0


def test_vehicle_assignment_to_depot():
    scenario = (
        ScenarioFactory
        .create_single_depot_scenario()
    )

    depot_id = scenario.depots[0].depot_id

    vehicles = scenario.vehicles_for_depot(
        depot_id
    )

    assert len(vehicles) == 8

    for vehicle in vehicles:
        assert vehicle.depot_id == depot_id


def test_shipper_assignment_to_depot():
    scenario = (
        ScenarioFactory
        .create_single_depot_scenario()
    )

    depot_id = scenario.depots[0].depot_id

    shippers = scenario.shippers_for_depot(
        depot_id
    )

    assert len(shippers) == 16

    for shipper in shippers:
        assert shipper.depot_id == depot_id