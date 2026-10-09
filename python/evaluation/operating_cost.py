from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingCostConfig:
    """
    Configuration for estimated operating cost.

    Important
    ---------
    The coefficients are NOT assumed to be actual XeDu financial costs.

    They represent a configurable operational-cost scenario.

    Formula
    -------
    cost =
        distance_cost_per_km * distance_km
        + travel_time_cost_per_minute * travel_time_minutes
        + late_delivery_penalty * late_deliveries
        + lateness_cost_per_minute * total_lateness_minutes

    The monetary/unit meaning depends entirely on the coefficients
    supplied by the experiment or API caller.
    """

    distance_cost_per_km: float

    travel_time_cost_per_minute: float

    late_delivery_penalty: float = 0.0

    lateness_cost_per_minute: float = 0.0

    cost_unit: str = "cost_unit"

    def validate(self) -> None:
        values = [
            self.distance_cost_per_km,
            self.travel_time_cost_per_minute,
            self.late_delivery_penalty,
            self.lateness_cost_per_minute,
        ]

        if any(
            value < 0
            for value in values
        ):
            raise ValueError(
                "Operating-cost coefficients "
                "cannot be negative."
            )

        if not self.cost_unit.strip():
            raise ValueError(
                "cost_unit cannot be empty."
            )


@dataclass(frozen=True)
class OperatingCostBreakdown:
    """
    Detailed estimated operating-cost result.
    """

    total_distance_km: float

    total_travel_time_minutes: float

    late_deliveries: int

    total_lateness_minutes: float

    distance_cost: float

    travel_time_cost: float

    late_delivery_cost: float

    lateness_cost: float

    total_estimated_cost: float

    cost_unit: str


class OperatingCostEvaluator:
    """
    Calculate a configurable estimated operating cost.

    This evaluator deliberately stays independent from
    route construction algorithms.

    It can therefore be used with:
    - Nearest Neighbor
    - Clarke-Wright
    - Q-Learning
    - SARSA
    - DQN
    - K-Means-guided DQN
    """

    def __init__(
        self,
        config: OperatingCostConfig,
    ) -> None:
        config.validate()

        self.config = config

    def evaluate(
        self,
        total_distance_km: float,
        total_travel_time_minutes: float,
        late_deliveries: int = 0,
        total_lateness_minutes: float = 0.0,
    ) -> OperatingCostBreakdown:
        self._validate_inputs(
            total_distance_km=(
                total_distance_km
            ),
            total_travel_time_minutes=(
                total_travel_time_minutes
            ),
            late_deliveries=(
                late_deliveries
            ),
            total_lateness_minutes=(
                total_lateness_minutes
            ),
        )

        distance_cost = (
            self.config
            .distance_cost_per_km
            * total_distance_km
        )

        travel_time_cost = (
            self.config
            .travel_time_cost_per_minute
            * total_travel_time_minutes
        )

        late_delivery_cost = (
            self.config
            .late_delivery_penalty
            * late_deliveries
        )

        lateness_cost = (
            self.config
            .lateness_cost_per_minute
            * total_lateness_minutes
        )

        total_estimated_cost = (
            distance_cost
            + travel_time_cost
            + late_delivery_cost
            + lateness_cost
        )

        return OperatingCostBreakdown(
            total_distance_km=float(
                total_distance_km
            ),
            total_travel_time_minutes=float(
                total_travel_time_minutes
            ),
            late_deliveries=int(
                late_deliveries
            ),
            total_lateness_minutes=float(
                total_lateness_minutes
            ),
            distance_cost=float(
                distance_cost
            ),
            travel_time_cost=float(
                travel_time_cost
            ),
            late_delivery_cost=float(
                late_delivery_cost
            ),
            lateness_cost=float(
                lateness_cost
            ),
            total_estimated_cost=float(
                total_estimated_cost
            ),
            cost_unit=(
                self.config.cost_unit
            ),
        )

    @staticmethod
    def _validate_inputs(
        total_distance_km: float,
        total_travel_time_minutes: float,
        late_deliveries: int,
        total_lateness_minutes: float,
    ) -> None:
        if total_distance_km < 0:
            raise ValueError(
                "total_distance_km "
                "cannot be negative."
            )

        if (
            total_travel_time_minutes
            < 0
        ):
            raise ValueError(
                "total_travel_time_minutes "
                "cannot be negative."
            )

        if late_deliveries < 0:
            raise ValueError(
                "late_deliveries "
                "cannot be negative."
            )

        if (
            total_lateness_minutes
            < 0
        ):
            raise ValueError(
                "total_lateness_minutes "
                "cannot be negative."
            )

        if not isinstance(
            late_deliveries,
            int,
        ):
            raise ValueError(
                "late_deliveries "
                "must be an integer."
            )