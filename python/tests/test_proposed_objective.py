import math

import pytest

from rl.proposed.objective import (
    ObjectiveConfig,
    ObjectiveContext,
    compute_route_objective,
    compute_step_cost,
    reward_from_step_cost,
)


def test_default_weights_sum_to_one():
    config = ObjectiveConfig()

    total_weight = (
        config.distance_weight
        + config.travel_time_weight
        + config.lateness_weight
        + config.cluster_switch_weight
    )

    assert math.isclose(
        total_weight,
        1.0,
    )


def test_shorter_and_faster_is_better():
    context = ObjectiveContext(
        number_of_deliveries=5,
        number_of_valid_deadlines=5,
    )

    good = compute_step_cost(
        distance_km=2.0,
        travel_time_minutes=4.0,
        lateness_minutes=0.0,
        cluster_switched=False,
        context=context,
    )

    bad = compute_step_cost(
        distance_km=8.0,
        travel_time_minutes=15.0,
        lateness_minutes=0.0,
        cluster_switched=False,
        context=context,
    )

    assert good.total_cost < bad.total_cost


def test_lateness_increases_cost():
    context = ObjectiveContext(
        number_of_deliveries=5,
        number_of_valid_deadlines=5,
    )

    on_time = compute_step_cost(
        distance_km=4.0,
        travel_time_minutes=8.0,
        lateness_minutes=0.0,
        cluster_switched=False,
        context=context,
    )

    late = compute_step_cost(
        distance_km=4.0,
        travel_time_minutes=8.0,
        lateness_minutes=20.0,
        cluster_switched=False,
        context=context,
    )

    assert late.total_cost > on_time.total_cost


def test_cluster_switch_increases_cost():
    context = ObjectiveContext(
        number_of_deliveries=5,
        number_of_valid_deadlines=5,
    )

    same_cluster = compute_step_cost(
        distance_km=4.0,
        travel_time_minutes=8.0,
        lateness_minutes=0.0,
        cluster_switched=False,
        context=context,
    )

    switched_cluster = compute_step_cost(
        distance_km=4.0,
        travel_time_minutes=8.0,
        lateness_minutes=0.0,
        cluster_switched=True,
        context=context,
    )

    assert (
        switched_cluster.total_cost
        > same_cluster.total_cost
    )


def test_reward_is_negative_step_cost():
    context = ObjectiveContext(
        number_of_deliveries=5,
        number_of_valid_deadlines=5,
    )

    breakdown = compute_step_cost(
        distance_km=3.0,
        travel_time_minutes=6.0,
        lateness_minutes=0.0,
        cluster_switched=False,
        context=context,
    )

    reward = reward_from_step_cost(
        breakdown
    )

    assert reward == pytest.approx(
        -breakdown.total_cost
    )


def test_sum_step_cost_equals_route_objective():
    context = ObjectiveContext(
        number_of_deliveries=3,
        number_of_valid_deadlines=3,
    )

    # depot -> d1
    step_1 = compute_step_cost(
        distance_km=3.0,
        travel_time_minutes=5.0,
        lateness_minutes=0.0,
        cluster_switched=False,
        context=context,
    )

    # d1 -> d2
    step_2 = compute_step_cost(
        distance_km=4.0,
        travel_time_minutes=7.0,
        lateness_minutes=5.0,
        cluster_switched=True,
        context=context,
    )

    # d2 -> d3
    step_3 = compute_step_cost(
        distance_km=2.0,
        travel_time_minutes=4.0,
        lateness_minutes=0.0,
        cluster_switched=False,
        context=context,
    )

    # d3 -> depot
    step_4 = compute_step_cost(
        distance_km=5.0,
        travel_time_minutes=8.0,
        lateness_minutes=0.0,
        cluster_switched=False,
        context=context,
    )

    total_step_cost = sum(
        step.total_cost
        for step in (
            step_1,
            step_2,
            step_3,
            step_4,
        )
    )

    route = compute_route_objective(
        total_distance_km=14.0,
        total_travel_time_minutes=24.0,
        total_lateness_minutes=5.0,
        cluster_switches=1,
        context=context,
    )

    assert total_step_cost == pytest.approx(
        route.total_cost
    )


def test_invalid_weight_sum_is_rejected():
    with pytest.raises(ValueError):
        ObjectiveConfig(
            distance_weight=0.50,
            travel_time_weight=0.50,
            lateness_weight=0.20,
            cluster_switch_weight=0.10,
        )


def test_cluster_switch_cannot_exceed_n_minus_one():
    context = ObjectiveContext(
        number_of_deliveries=5,
        number_of_valid_deadlines=5,
    )

    with pytest.raises(ValueError):
        compute_route_objective(
            total_distance_km=20.0,
            total_travel_time_minutes=30.0,
            total_lateness_minutes=0.0,
            cluster_switches=5,
            context=context,
        )