from pydantic import (
    BaseModel,
    Field,
)

from evaluation.operating_cost import (
    OperatingCostConfig,
)


class OperatingCostConfigRequest(
    BaseModel
):
    """
    Optional caller-supplied operating-cost scenario.

    These values are scenario coefficients,
    not automatically verified XeDu accounting costs.
    """

    distance_cost_per_km: float = Field(
        ge=0.0
    )

    travel_time_cost_per_minute: float = Field(
        ge=0.0
    )

    late_delivery_penalty: float = Field(
        default=0.0,
        ge=0.0,
    )

    lateness_cost_per_minute: float = Field(
        default=0.0,
        ge=0.0,
    )

    cost_unit: str = Field(
        default="custom_cost_unit",
        min_length=1,
    )

    def to_domain(
        self,
    ) -> OperatingCostConfig:
        return OperatingCostConfig(
            distance_cost_per_km=float(
                self.distance_cost_per_km
            ),
            travel_time_cost_per_minute=float(
                self.travel_time_cost_per_minute
            ),
            late_delivery_penalty=float(
                self.late_delivery_penalty
            ),
            lateness_cost_per_minute=float(
                self.lateness_cost_per_minute
            ),
            cost_unit=(
                self.cost_unit.strip()
            ),
        )