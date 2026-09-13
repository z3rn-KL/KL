from dataclasses import dataclass, field

from .delivery import Delivery


@dataclass
class Cluster:
    cluster_id: int

    deliveries: list[Delivery] = field(
        default_factory=list
    )

    centroid_latitude: float | None = None
    centroid_longitude: float | None = None

    def size(self) -> int:
        return len(self.deliveries)

    def total_weight(self) -> float:
        return sum(
            delivery.weight
            for delivery in self.deliveries
            if delivery.weight is not None
        )

    def add_delivery(
        self,
        delivery: Delivery,
    ) -> None:
        self.deliveries.append(delivery)