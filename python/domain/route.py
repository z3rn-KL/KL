from dataclasses import dataclass, field

from .delivery import Delivery


@dataclass
class Route:
    route_id: str

    depot_id: str | None = None
    vehicle_id: str | None = None
    shipper_id: str | None = None

    deliveries: list[Delivery] = field(
        default_factory=list
    )

    total_distance_km: float = 0.0
    total_time_hours: float = 0.0
    total_cost: float = 0.0

    def add_delivery(
        self,
        delivery: Delivery,
    ) -> None:
        self.deliveries.append(delivery)

    def size(self) -> int:
        return len(self.deliveries)

    def delivery_ids(self) -> list[str]:
        return [
            delivery.delivery_id
            for delivery in self.deliveries
        ]