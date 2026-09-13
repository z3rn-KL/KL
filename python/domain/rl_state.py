from dataclasses import dataclass


@dataclass(frozen=True)
class RLState:
    current_location_id: str

    unvisited_delivery_ids: tuple[str, ...]

    current_time_minutes: float = 0.0

    remaining_capacity_kg: float | None = None

    def is_terminal(self) -> bool:
        return len(
            self.unvisited_delivery_ids
        ) == 0

    def number_of_unvisited(self) -> int:
        return len(
            self.unvisited_delivery_ids
        )