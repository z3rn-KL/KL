from dataclasses import dataclass


@dataclass
class Vehicle:
    vehicle_id: str

    depot_id: str

    capacity_kg: float

    cost_per_km: float = 0.0
    cost_per_hour: float = 0.0

    max_work_hours: float | None = None