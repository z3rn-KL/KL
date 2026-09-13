from dataclasses import dataclass


@dataclass
class Shipper:
    shipper_id: str

    depot_id: str | None = None

    service_area: str | None = None

    max_orders: int | None = None
    max_work_hours: float | None = None