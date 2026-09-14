from dataclasses import dataclass, field

from domain import Depot, Vehicle, Shipper


@dataclass
class OperationalScenario:
    name: str

    depots: list[Depot] = field(
        default_factory=list
    )

    vehicles: list[Vehicle] = field(
        default_factory=list
    )

    shippers: list[Shipper] = field(
        default_factory=list
    )

    def number_of_depots(self) -> int:
        return len(self.depots)

    def number_of_vehicles(self) -> int:
        return len(self.vehicles)

    def number_of_shippers(self) -> int:
        return len(self.shippers)

    @property
    def total_vehicle_capacity_kg(self) -> float:
        return sum(
            vehicle.capacity_kg
            for vehicle in self.vehicles
        )

    def vehicles_for_depot(
        self,
        depot_id: str,
    ) -> list[Vehicle]:
        return [
            vehicle
            for vehicle in self.vehicles
            if vehicle.depot_id == depot_id
        ]

    def shippers_for_depot(
        self,
        depot_id: str,
    ) -> list[Shipper]:
        return [
            shipper
            for shipper in self.shippers
            if shipper.depot_id == depot_id
        ]


class ScenarioFactory:

    @staticmethod
    def create_single_depot_scenario() -> OperationalScenario:
        depot = Depot(
            depot_id="DEPOT_01",
            name="Depot 01",
            latitude=10.7769,
            longitude=106.7009,
        )

        vehicles = [
            Vehicle(
                vehicle_id=f"V{i:02d}",
                depot_id=depot.depot_id,
                capacity_kg=100.0,
                cost_per_km=4000.0,
                cost_per_hour=40000.0,
                max_work_hours=8.0,
            )
            for i in range(1, 9)
        ]

        shippers = [
            Shipper(
                shipper_id=f"S{i:02d}",
                depot_id=depot.depot_id,
                max_orders=None,
                max_work_hours=8.0,
            )
            for i in range(1, 17)
        ]

        return OperationalScenario(
            name="single_depot_8_vehicles_16_shippers",
            depots=[depot],
            vehicles=vehicles,
            shippers=shippers,
        )