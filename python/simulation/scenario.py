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