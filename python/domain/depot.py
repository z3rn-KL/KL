from dataclasses import dataclass


@dataclass(frozen=True)
class Depot:
    depot_id: str
    name: str

    latitude: float
    longitude: float

    def location(self) -> tuple[float, float]:
        return self.latitude, self.longitude