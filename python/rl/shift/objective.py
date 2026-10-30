from __future__ import annotations

from dataclasses import dataclass


# Assumed delivery-fee tariff. XeDu data has no order value or price, so
# this is a scenario parameter, not a real XeDu figure.
SERVICE_FEE_MULTIPLIER = {
    "3h": 1.0,
    "5h": 0.8,
    "ban_tai_4h": 1.0,
    "ban_tai_nhanh": 1.3,
    "luu_kho": 0.8,
}


@dataclass(frozen=True)
class FeeTariff:
    base_fee: float = 15_000.0
    fee_per_km: float = 3_000.0

    def fee(self, distance_km: float, service_type: str) -> float:
        return (
            self.base_fee + self.fee_per_km * distance_km
        ) * SERVICE_FEE_MULTIPLIER.get(service_type, 1.0)


@dataclass(frozen=True)
class ShiftObjectiveConfig:
    """
    Score of one shipper shift, in VND-equivalent points (higher is better).

    Rewards (all small next to the penalties, by construction):
        commission:   commission_rate of the order value for every order
                      delivered on time, plus up to early_commission_rate
                      more when it arrives early_slack_minutes ahead.
        early finish: wage of the minutes (up to early_finish_minutes) by
                      which the shift ends before shift_minutes, paid only
                      when no order was deferred.

    Penalties:
        time and distance spent (wage and cost per km), overtime at
        overtime_multiplier times the wage, late orders (a share of the
        order value plus a per-minute charge), orders whose own margin
        falls below required_margin, and every order deferred.
    """

    wage_per_minute: float = 40_000.0 / 60.0
    cost_per_km: float = 4_000.0
    service_minutes: float = 10.0
    shift_minutes: float = 600.0

    commission_rate: float = 0.03
    early_commission_rate: float = 0.07
    early_slack_minutes: float = 60.0
    early_finish_minutes: float = 30.0

    late_fixed_fraction: float = 0.50
    late_penalty_per_minute: float = 1_000.0
    overtime_multiplier: float = 3.0
    required_margin: float = 0.30
    margin_shortfall_weight: float = 2.0
    defer_penalty: float = 60_000.0

    def __post_init__(self) -> None:
        for name in (
            "wage_per_minute",
            "cost_per_km",
            "service_minutes",
            "shift_minutes",
            "commission_rate",
            "early_commission_rate",
            "early_slack_minutes",
            "early_finish_minutes",
            "late_fixed_fraction",
            "late_penalty_per_minute",
            "margin_shortfall_weight",
            "defer_penalty",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0.")

        if self.early_slack_minutes == 0.0:
            raise ValueError("early_slack_minutes must be > 0.")

        if not 0.0 <= self.required_margin < 1.0:
            raise ValueError("required_margin must be in [0, 1).")

        if self.overtime_multiplier < 1.0:
            raise ValueError("overtime_multiplier must be >= 1.")

        # The rule that penalties outweigh rewards: the smallest penalty
        # for a late order must exceed the largest commission it could earn.
        if self.late_fixed_fraction <= (
            self.commission_rate + self.early_commission_rate
        ):
            raise ValueError(
                "late_fixed_fraction must exceed the maximum commission rate."
            )

    @property
    def early_finish_bonus(self) -> float:
        return self.wage_per_minute * self.early_finish_minutes


def order_terms(
    cfg: ShiftObjectiveConfig,
    *,
    value: float,
    deadline: float | None,
    arrival: float,
    travel_minutes: float,
    distance_km: float,
) -> tuple[float, float, float, float]:
    """
    Score terms of serving one order, as
    (commission, move_cost, late_penalty, margin_penalty); the step reward is
    commission - move_cost - late_penalty - margin_penalty.
    """

    move_cost = (
        cfg.wage_per_minute * (travel_minutes + cfg.service_minutes)
        + cfg.cost_per_km * distance_km
    )

    commission = 0.0
    late_penalty = 0.0

    if deadline is None:
        commission = cfg.commission_rate * value
    elif arrival <= deadline:
        early = min(1.0, (deadline - arrival) / cfg.early_slack_minutes)
        commission = (
            cfg.commission_rate + cfg.early_commission_rate * early
        ) * value
    else:
        late_penalty = (
            cfg.late_fixed_fraction * value
            + cfg.late_penalty_per_minute * (arrival - deadline)
        )

    # Margin of this order alone: fee minus what serving it cost.
    margin_shortfall = max(
        0.0,
        move_cost - (1.0 - cfg.required_margin) * value,
    )

    return (
        commission,
        move_cost,
        late_penalty,
        cfg.margin_shortfall_weight * margin_shortfall,
    )


def terminal_terms(
    cfg: ShiftObjectiveConfig,
    *,
    finish_minutes: float,
    return_minutes: float,
    return_km: float,
    deferred: int,
) -> tuple[float, float, float, float]:
    """
    Score terms of ending the shift, as
    (early_bonus, return_cost, overtime_penalty, defer_penalty); the reward is
    early_bonus - return_cost - overtime_penalty - defer_penalty.
    """

    end = finish_minutes + return_minutes

    return_cost = (
        cfg.wage_per_minute * return_minutes + cfg.cost_per_km * return_km
    )

    overtime_penalty = (
        cfg.overtime_multiplier
        * cfg.wage_per_minute
        * max(0.0, end - cfg.shift_minutes)
    )

    early_bonus = 0.0

    if deferred == 0:
        early_bonus = cfg.wage_per_minute * min(
            cfg.early_finish_minutes,
            max(0.0, cfg.shift_minutes - end),
        )

    return (
        early_bonus,
        return_cost,
        overtime_penalty,
        cfg.defer_penalty * deferred,
    )
