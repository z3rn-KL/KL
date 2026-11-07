import pytest

from rl.shift.objective import (
    ShiftObjectiveConfig,
    order_terms,
    terminal_terms,
)


def test_on_time_early_order_terms_by_hand():
    cfg = ShiftObjectiveConfig()

    commission, move, late, margin = order_terms(
        cfg,
        value=30_000.0,
        deadline=100.0,
        arrival=40.0,
        travel_minutes=10.0,
        distance_km=4.0,
    )

    # Arrives 60 min early: full 3% + 7% commission.
    assert commission == pytest.approx(3_000.0)
    # 40,000 / 60 per min * (10 travel + 10 service) + 4,000 * 4 km.
    assert move == pytest.approx(29_333.3333, rel=1e-6)
    assert late == 0.0
    # Cost exceeds (1 - 30%) of the value by 8,333.33; weight 2.
    assert margin == pytest.approx(16_666.6667, rel=1e-6)


def test_late_order_loses_commission_and_pays_penalty():
    cfg = ShiftObjectiveConfig()

    commission, _, late, _ = order_terms(
        cfg,
        value=30_000.0,
        deadline=100.0,
        arrival=130.0,
        travel_minutes=5.0,
        distance_km=1.0,
    )

    assert commission == 0.0
    # 50% of the value plus 30 minutes late at 1,000 per minute.
    assert late == pytest.approx(15_000.0 + 30_000.0)


def test_order_without_deadline_earns_base_commission_only():
    commission, _, late, _ = order_terms(
        ShiftObjectiveConfig(),
        value=20_000.0,
        deadline=None,
        arrival=500.0,
        travel_minutes=5.0,
        distance_km=1.0,
    )

    assert commission == pytest.approx(600.0)
    assert late == 0.0


def test_penalty_exceeds_reward_for_every_order_value():
    cfg = ShiftObjectiveConfig()

    for value in (5_000.0, 30_000.0, 200_000.0):
        best, _, _, _ = order_terms(
            cfg,
            value=value,
            deadline=1_000.0,
            arrival=0.0,
            travel_minutes=0.0,
            distance_km=0.0,
        )
        _, _, worst, _ = order_terms(
            cfg,
            value=value,
            deadline=0.0,
            arrival=0.001,
            travel_minutes=0.0,
            distance_km=0.0,
        )

        assert worst > best


def test_config_rejects_rewards_that_outweigh_penalties():
    with pytest.raises(ValueError):
        ShiftObjectiveConfig(late_fixed_fraction=0.10)


def test_early_finish_bonus_needs_every_order_served_and_is_capped():
    cfg = ShiftObjectiveConfig()

    early, _, _, _ = terminal_terms(
        cfg, finish_minutes=400.0, return_minutes=10.0, return_km=1.0, deferred=0
    )
    assert early == pytest.approx(cfg.early_finish_bonus)

    partial, _, _, _ = terminal_terms(
        cfg, finish_minutes=595.0, return_minutes=0.0, return_km=0.0, deferred=0
    )
    assert partial == pytest.approx(cfg.wage_per_minute * 5.0)

    skipped, _, _, _ = terminal_terms(
        cfg, finish_minutes=100.0, return_minutes=0.0, return_km=0.0, deferred=1
    )
    assert skipped == 0.0


def test_overtime_and_deferral_penalties():
    cfg = ShiftObjectiveConfig()

    _, _, overtime, defer = terminal_terms(
        cfg, finish_minutes=620.0, return_minutes=10.0, return_km=0.0, deferred=2
    )

    # 30 minutes past the shift at 3x the wage.
    assert overtime == pytest.approx(3.0 * cfg.wage_per_minute * 30.0)
    assert defer == pytest.approx(2 * cfg.defer_penalty)
