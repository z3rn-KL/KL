from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObjectiveConfig:
    """
    Cấu hình hàm mục tiêu của Proposed RL.

    J(pi) =
        w_d * D_bar(pi)
        + w_t * T_bar(pi)
        + w_m * M_bar(pi)
        + w_l * L_bar(pi)
        + w_c * S_bar(pi)

    Trong đó:

        D_bar:
            quãng đường chuẩn hóa

        T_bar:
            travel time chuẩn hóa

        M_bar:
            số delivery giao trễ chuẩn hóa

        L_bar:
            tổng số phút trễ chuẩn hóa

        S_bar:
            số lần chuyển cluster chuẩn hóa

    Lưu ý:
        late_delivery_weight mặc định = 0.0
        để giữ tương thích với objective trước đây.

    Bộ trọng số cuối cùng sẽ được lựa chọn bằng
    sensitivity analysis, không chốt tùy ý.
    """

    distance_weight: float = 0.40
    travel_time_weight: float = 0.40

    late_delivery_weight: float = 0.00
    lateness_weight: float = 0.15

    cluster_switch_weight: float = 0.05

    distance_ref_km: float = 10.0
    travel_time_ref_minutes: float = 15.0
    lateness_ref_minutes: float = 30.0

    def __post_init__(self) -> None:
        weights = (
            self.distance_weight,
            self.travel_time_weight,
            self.late_delivery_weight,
            self.lateness_weight,
            self.cluster_switch_weight,
        )

        if any(
            weight < 0.0
            for weight in weights
        ):
            raise ValueError(
                "Objective weights must be non-negative."
            )

        total_weight = sum(weights)

        if not math.isclose(
            total_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Objective weights must sum to 1.0. "
                f"Current sum = {total_weight:.12f}"
            )

        if self.distance_ref_km <= 0.0:
            raise ValueError(
                "distance_ref_km must be > 0."
            )

        if self.travel_time_ref_minutes <= 0.0:
            raise ValueError(
                "travel_time_ref_minutes must be > 0."
            )

        if self.lateness_ref_minutes <= 0.0:
            raise ValueError(
                "lateness_ref_minutes must be > 0."
            )


@dataclass(frozen=True)
class ObjectiveContext:
    """
    Thông tin cố định của một workload / episode.
    """

    number_of_deliveries: int
    number_of_valid_deadlines: int

    def __post_init__(self) -> None:
        if self.number_of_deliveries <= 0:
            raise ValueError(
                "number_of_deliveries must be > 0."
            )

        if self.number_of_valid_deadlines < 0:
            raise ValueError(
                "number_of_valid_deadlines must be >= 0."
            )

        if (
            self.number_of_valid_deadlines
            > self.number_of_deliveries
        ):
            raise ValueError(
                "number_of_valid_deadlines cannot exceed "
                "number_of_deliveries."
            )

    @property
    def route_leg_count(self) -> int:
        """
        Một closed route gồm N+1 road legs:

            depot -> delivery 1
            delivery 1 -> delivery 2
            ...
            delivery N -> depot
        """

        return self.number_of_deliveries + 1

    @property
    def valid_deadline_denominator(self) -> int:
        """
        Tránh chia 0 khi workload không có deadline hợp lệ.
        """

        return max(
            1,
            self.number_of_valid_deadlines,
        )

    @property
    def possible_cluster_switches(self) -> int:
        """
        Chỉ tính delivery -> delivery.

        Không tính:
            depot -> delivery
            delivery -> depot
        """

        return max(
            1,
            self.number_of_deliveries - 1,
        )


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """
    Phân rã đầy đủ hàm mục tiêu.
    """

    normalized_distance: float
    normalized_travel_time: float

    normalized_late_deliveries: float
    normalized_lateness: float

    normalized_cluster_switches: float

    distance_cost: float
    travel_time_cost: float

    late_delivery_cost: float
    lateness_cost: float

    cluster_switch_cost: float

    total_cost: float


def _validate_non_negative(
    *,
    distance_km: float,
    travel_time_minutes: float,
    lateness_minutes: float,
) -> None:
    if distance_km < 0.0:
        raise ValueError(
            "distance_km must be >= 0."
        )

    if travel_time_minutes < 0.0:
        raise ValueError(
            "travel_time_minutes must be >= 0."
        )

    if lateness_minutes < 0.0:
        raise ValueError(
            "lateness_minutes must be >= 0."
        )


def compute_step_cost(
    *,
    distance_km: float,
    travel_time_minutes: float,
    lateness_minutes: float,
    cluster_switched: bool,
    context: ObjectiveContext,
    config: ObjectiveConfig | None = None,
) -> ObjectiveBreakdown:
    """
    Tính Delta J_t cho một action.

    Delivery được xem là late khi:

        lateness_minutes > 0

    Reward của Proposed RL:

        r_t = - Delta J_t
    """

    if config is None:
        config = ObjectiveConfig()

    _validate_non_negative(
        distance_km=distance_km,
        travel_time_minutes=travel_time_minutes,
        lateness_minutes=lateness_minutes,
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    normalized_distance = (
        distance_km
        / (
            context.route_leg_count
            * config.distance_ref_km
        )
    )

    normalized_travel_time = (
        travel_time_minutes
        / (
            context.route_leg_count
            * config.travel_time_ref_minutes
        )
    )

    is_late = (
        lateness_minutes > 0.0
    )

    normalized_late_deliveries = (
        float(is_late)
        / context.valid_deadline_denominator
    )

    normalized_lateness = (
        lateness_minutes
        / (
            context.valid_deadline_denominator
            * config.lateness_ref_minutes
        )
    )

    normalized_cluster_switches = (
        float(cluster_switched)
        / context.possible_cluster_switches
    )

    # ========================================================
    # WEIGHTED COST COMPONENTS
    # ========================================================

    distance_cost = (
        config.distance_weight
        * normalized_distance
    )

    travel_time_cost = (
        config.travel_time_weight
        * normalized_travel_time
    )

    late_delivery_cost = (
        config.late_delivery_weight
        * normalized_late_deliveries
    )

    lateness_cost = (
        config.lateness_weight
        * normalized_lateness
    )

    cluster_switch_cost = (
        config.cluster_switch_weight
        * normalized_cluster_switches
    )

    total_cost = (
        distance_cost
        + travel_time_cost
        + late_delivery_cost
        + lateness_cost
        + cluster_switch_cost
    )

    return ObjectiveBreakdown(
        normalized_distance=(
            normalized_distance
        ),
        normalized_travel_time=(
            normalized_travel_time
        ),
        normalized_late_deliveries=(
            normalized_late_deliveries
        ),
        normalized_lateness=(
            normalized_lateness
        ),
        normalized_cluster_switches=(
            normalized_cluster_switches
        ),
        distance_cost=(
            distance_cost
        ),
        travel_time_cost=(
            travel_time_cost
        ),
        late_delivery_cost=(
            late_delivery_cost
        ),
        lateness_cost=(
            lateness_cost
        ),
        cluster_switch_cost=(
            cluster_switch_cost
        ),
        total_cost=(
            total_cost
        ),
    )


def compute_route_objective(
    *,
    total_distance_km: float,
    total_travel_time_minutes: float,
    total_lateness_minutes: float,
    cluster_switches: int,
    context: ObjectiveContext,
    late_deliveries: int = 0,
    config: ObjectiveConfig | None = None,
) -> ObjectiveBreakdown:
    """
    Tính J(pi) cho toàn bộ route.

    J(pi) =
        w_d * D_bar
        + w_t * T_bar
        + w_m * M_bar
        + w_l * L_bar
        + w_c * S_bar
    """

    if config is None:
        config = ObjectiveConfig()

    _validate_non_negative(
        distance_km=total_distance_km,
        travel_time_minutes=(
            total_travel_time_minutes
        ),
        lateness_minutes=(
            total_lateness_minutes
        ),
    )

    if late_deliveries < 0:
        raise ValueError(
            "late_deliveries must be >= 0."
        )

    if (
        late_deliveries
        > context.number_of_valid_deadlines
    ):
        raise ValueError(
            "late_deliveries cannot exceed "
            "number_of_valid_deadlines."
        )

    if cluster_switches < 0:
        raise ValueError(
            "cluster_switches must be >= 0."
        )

    max_cluster_switches = max(
        0,
        context.number_of_deliveries - 1,
    )

    if (
        cluster_switches
        > max_cluster_switches
    ):
        raise ValueError(
            "cluster_switches cannot exceed N - 1."
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    normalized_distance = (
        total_distance_km
        / (
            context.route_leg_count
            * config.distance_ref_km
        )
    )

    normalized_travel_time = (
        total_travel_time_minutes
        / (
            context.route_leg_count
            * config.travel_time_ref_minutes
        )
    )

    normalized_late_deliveries = (
        late_deliveries
        / context.valid_deadline_denominator
    )

    normalized_lateness = (
        total_lateness_minutes
        / (
            context.valid_deadline_denominator
            * config.lateness_ref_minutes
        )
    )

    normalized_cluster_switches = (
        cluster_switches
        / context.possible_cluster_switches
    )

    # ========================================================
    # WEIGHTED COST COMPONENTS
    # ========================================================

    distance_cost = (
        config.distance_weight
        * normalized_distance
    )

    travel_time_cost = (
        config.travel_time_weight
        * normalized_travel_time
    )

    late_delivery_cost = (
        config.late_delivery_weight
        * normalized_late_deliveries
    )

    lateness_cost = (
        config.lateness_weight
        * normalized_lateness
    )

    cluster_switch_cost = (
        config.cluster_switch_weight
        * normalized_cluster_switches
    )

    total_cost = (
        distance_cost
        + travel_time_cost
        + late_delivery_cost
        + lateness_cost
        + cluster_switch_cost
    )

    return ObjectiveBreakdown(
        normalized_distance=(
            normalized_distance
        ),
        normalized_travel_time=(
            normalized_travel_time
        ),
        normalized_late_deliveries=(
            normalized_late_deliveries
        ),
        normalized_lateness=(
            normalized_lateness
        ),
        normalized_cluster_switches=(
            normalized_cluster_switches
        ),
        distance_cost=(
            distance_cost
        ),
        travel_time_cost=(
            travel_time_cost
        ),
        late_delivery_cost=(
            late_delivery_cost
        ),
        lateness_cost=(
            lateness_cost
        ),
        cluster_switch_cost=(
            cluster_switch_cost
        ),
        total_cost=(
            total_cost
        ),
    )


def reward_from_step_cost(
    breakdown: ObjectiveBreakdown,
) -> float:
    """
    Quan hệ trung tâm:

        r_t = -Delta J_t
    """

    return -breakdown.total_cost