from dataclasses import dataclass
from datetime import datetime
from math import exp

from domain import Delivery


SERVICE_PRIORITY = {
    "3h": 1.00,
    "ban_tai_nhanh": 1.00,
    "ban_tai_4h": 0.80,
    "5h": 0.70,
    "luu_kho": 0.30,
}


def clamp_01(
    value: float,
) -> float:
    """
    Clamp a numeric value to [0, 1].
    """
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


@dataclass(frozen=True)
class PriorityWeights:
    """
    Priority Model V3 weights.

    Selected from the four-factor grid experiment:

        Deadline urgency   : 55%
        Service priority   : 20%
        Waiting time       : 15%
        Travel efficiency  : 10%

    These are experimental thesis parameters,
    not official XeDu business rules.
    """

    deadline_urgency: float = 0.55
    service_priority: float = 0.20
    waiting_time: float = 0.15
    travel_efficiency: float = 0.10

    def validate(self) -> None:
        weights = [
            self.deadline_urgency,
            self.service_priority,
            self.waiting_time,
            self.travel_efficiency,
        ]

        if any(
            weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Priority weights cannot be negative."
            )

        total = sum(
            weights
        )

        if total <= 0:
            raise ValueError(
                "At least one priority weight "
                "must be greater than 0."
            )

        if abs(
            total - 1.0
        ) > 1e-9:
            raise ValueError(
                "Priority weights must sum to 1.0."
            )


@dataclass(frozen=True)
class PriorityResult:
    """
    Detailed priority calculation result.
    """

    delivery_id: str

    priority_score: float
    base_priority_score: float
    information_coverage: float

    deadline_urgency: float | None
    sla_progress: float | None

    service_priority: float | None
    waiting_time_score: float | None
    travel_efficiency_score: float | None


class DeliveryPriorityModel:
    """
    Delivery Priority Model V3.

    Priority is calculated from:

        1. Deadline urgency
        2. Service priority
        3. Waiting time
        4. Road-based travel efficiency

    Missing values are not imputed.

    When one component is unavailable,
    information coverage decreases.
    """

    def __init__(
        self,
        weights: PriorityWeights | None = None,
    ) -> None:
        self.weights = (
            weights
            or PriorityWeights()
        )

        self.weights.validate()

    def calculate(
        self,
        delivery: Delivery,
        current_time: datetime,
        waiting_scale_minutes: float = 180.0,
        travel_efficiency_score: float | None = None,
    ) -> PriorityResult:
        """
        Calculate priority for one delivery.
        """

        sla_progress = (
            self.calculate_sla_progress(
                delivery=delivery,
                current_time=current_time,
            )
        )

        deadline_score = (
            self.deadline_urgency(
                delivery=delivery,
                current_time=current_time,
            )
        )

        service_score = (
            self.service_priority(
                delivery
            )
        )

        waiting_score = (
            self.waiting_time_score(
                delivery=delivery,
                current_time=current_time,
                waiting_scale_minutes=(
                    waiting_scale_minutes
                ),
            )
        )

        if travel_efficiency_score is None:
            travel_score = None
        else:
            travel_score = clamp_01(
                travel_efficiency_score
            )

        (
            base_score,
            coverage,
            final_score,
        ) = self._weighted_score(
            deadline_urgency=(
                deadline_score
            ),
            service_priority=(
                service_score
            ),
            waiting_time=(
                waiting_score
            ),
            travel_efficiency=(
                travel_score
            ),
        )

        return PriorityResult(
            delivery_id=(
                delivery.delivery_id
            ),
            priority_score=(
                final_score
            ),
            base_priority_score=(
                base_score
            ),
            information_coverage=(
                coverage
            ),
            deadline_urgency=(
                deadline_score
            ),
            sla_progress=(
                sla_progress
            ),
            service_priority=(
                service_score
            ),
            waiting_time_score=(
                waiting_score
            ),
            travel_efficiency_score=(
                travel_score
            ),
        )

    def _weighted_score(
        self,
        deadline_urgency: float | None,
        service_priority: float | None,
        waiting_time: float | None,
        travel_efficiency: float | None,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        """
        Calculate weighted priority.

        Returns:
            base_score
            information_coverage
            final_score
        """

        components = [
            (
                deadline_urgency,
                self.weights.deadline_urgency,
            ),
            (
                service_priority,
                self.weights.service_priority,
            ),
            (
                waiting_time,
                self.weights.waiting_time,
            ),
            (
                travel_efficiency,
                self.weights.travel_efficiency,
            ),
        ]

        total_configured_weight = sum(
            weight
            for _, weight in components
            if weight > 0
        )

        active_weight = 0.0
        weighted_sum = 0.0

        for score, weight in components:
            if weight <= 0:
                continue

            if score is None:
                continue

            active_weight += weight

            weighted_sum += (
                score
                * weight
            )

        if active_weight <= 0:
            return (
                0.0,
                0.0,
                0.0,
            )

        base_score = (
            weighted_sum
            / active_weight
        )

        coverage = (
            active_weight
            / total_configured_weight
        )

        final_score = (
            base_score
            * coverage
        )

        return (
            clamp_01(
                base_score
            ),
            clamp_01(
                coverage
            ),
            clamp_01(
                final_score
            ),
        )

    @staticmethod
    def calculate_sla_progress(
        delivery: Delivery,
        current_time: datetime,
    ) -> float | None:
        """
        Calculate percentage of SLA consumed.

        Examples:

            0.0
                order has just been created

            0.5
                50% SLA consumed

            1.0
                deadline reached

            > 1.0
                overdue
        """

        created_at = (
            delivery.created_at
        )

        deadline = (
            delivery.expected_delivery_time
        )

        if (
            created_at is None
            or deadline is None
        ):
            return None

        sla_seconds = (
            deadline
            - created_at
        ).total_seconds()

        if sla_seconds <= 0:
            return None

        elapsed_seconds = (
            current_time
            - created_at
        ).total_seconds()

        elapsed_seconds = max(
            0.0,
            elapsed_seconds,
        )

        return (
            elapsed_seconds
            / sla_seconds
        )

    @classmethod
    def deadline_urgency(
        cls,
        delivery: Delivery,
        current_time: datetime,
    ) -> float | None:
        """
        Convert SLA progress to urgency.

        Logistic function:

            1 / (
                1 + exp(
                    -6 * (progress - 0.8)
                )
            )

        Urgency rises rapidly when about
        80% of SLA has been consumed.
        """

        progress = (
            cls.calculate_sla_progress(
                delivery=delivery,
                current_time=current_time,
            )
        )

        if progress is None:
            return None

        urgency = (
            1.0
            / (
                1.0
                + exp(
                    -6.0
                    * (
                        progress
                        - 0.8
                    )
                )
            )
        )

        return clamp_01(
            urgency
        )

    @staticmethod
    def service_priority(
        delivery: Delivery,
    ) -> float | None:
        """
        Convert service type to priority score.

        Values are experimental thesis parameters.
        """

        service_type = (
            delivery.service_type
        )

        if service_type is None:
            return None

        normalized_type = (
            str(
                service_type
            )
            .strip()
            .lower()
        )

        if not normalized_type:
            return None

        return SERVICE_PRIORITY.get(
            normalized_type,
            0.50,
        )

    @staticmethod
    def waiting_time_score(
        delivery: Delivery,
        current_time: datetime,
        waiting_scale_minutes: float = 180.0,
    ) -> float | None:
        """
        Calculate smooth waiting-time priority.

        Formula:

            1 - exp(
                -waiting_minutes
                / waiting_scale_minutes
            )

        This prevents abrupt score saturation.
        """

        if (
            waiting_scale_minutes
            <= 0
        ):
            raise ValueError(
                "waiting_scale_minutes "
                "must be greater than 0."
            )

        created_at = (
            delivery.created_at
        )

        if created_at is None:
            return None

        waiting_minutes = (
            current_time
            - created_at
        ).total_seconds() / 60.0

        waiting_minutes = max(
            0.0,
            waiting_minutes,
        )

        score = (
            1.0
            - exp(
                -waiting_minutes
                / waiting_scale_minutes
            )
        )

        return clamp_01(
            score
        )