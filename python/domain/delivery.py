from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Delivery:
    delivery_id: str

    latitude: float
    longitude: float

    weight: float | None = None

    created_at: datetime | None = None
    expected_delivery_time: datetime | None = None
    delivered_at: datetime | None = None

    service_type: str | None = None

    def location(self) -> tuple[float, float]:
        return self.latitude, self.longitude

    def has_deadline(self) -> bool:
        return self.expected_delivery_time is not None