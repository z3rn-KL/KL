from dataclasses import dataclass


@dataclass(frozen=True)
class Assignment:
    delivery_id: str

    cluster_id: int

    depot_id: str | None = None
    vehicle_id: str | None = None
    shipper_id: str | None = None