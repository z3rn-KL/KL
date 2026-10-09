import pytest

from evaluation.operating_cost import (
    OperatingCostConfig,
    OperatingCostEvaluator,
)


def make_evaluator(
) -> OperatingCostEvaluator:
    config = OperatingCostConfig(
        distance_cost_per_km=2.0,
        travel_time_cost_per_minute=0.5,
        late_delivery_penalty=10.0,
        lateness_cost_per_minute=0.25,
        cost_unit="test_unit",
    )

    return OperatingCostEvaluator(
        config
    )


def test_operating_cost_calculation():
    evaluator = (
        make_evaluator()
    )

    result = evaluator.evaluate(
        total_distance_km=10.0,
        total_travel_time_minutes=20.0,
        late_deliveries=2,
        total_lateness_minutes=8.0,
    )

    assert (
        result.distance_cost
        == 20.0
    )

    assert (
        result.travel_time_cost
        == 10.0
    )

    assert (
        result.late_delivery_cost
        == 20.0
    )

    assert (
        result.lateness_cost
        == 2.0
    )

    assert (
        result.total_estimated_cost
        == 52.0
    )

    assert (
        result.cost_unit
        == "test_unit"
    )


def test_zero_route_has_zero_cost():
    evaluator = (
        make_evaluator()
    )

    result = evaluator.evaluate(
        total_distance_km=0.0,
        total_travel_time_minutes=0.0,
        late_deliveries=0,
        total_lateness_minutes=0.0,
    )

    assert (
        result.total_estimated_cost
        == 0.0
    )


def test_distance_only_cost():
    config = OperatingCostConfig(
        distance_cost_per_km=3.0,
        travel_time_cost_per_minute=0.0,
    )

    evaluator = (
        OperatingCostEvaluator(
            config
        )
    )

    result = evaluator.evaluate(
        total_distance_km=5.0,
        total_travel_time_minutes=100.0,
    )

    assert (
        result.total_estimated_cost
        == 15.0
    )


def test_negative_coefficient_rejected():
    config = OperatingCostConfig(
        distance_cost_per_km=-1.0,
        travel_time_cost_per_minute=1.0,
    )

    with pytest.raises(
        ValueError
    ):
        OperatingCostEvaluator(
            config
        )


def test_empty_cost_unit_rejected():
    config = OperatingCostConfig(
        distance_cost_per_km=1.0,
        travel_time_cost_per_minute=1.0,
        cost_unit="",
    )

    with pytest.raises(
        ValueError
    ):
        OperatingCostEvaluator(
            config
        )


@pytest.mark.parametrize(
    (
        "distance",
        "travel_time",
        "late_deliveries",
        "lateness",
    ),
    [
        (
            -1.0,
            10.0,
            0,
            0.0,
        ),
        (
            1.0,
            -10.0,
            0,
            0.0,
        ),
        (
            1.0,
            10.0,
            -1,
            0.0,
        ),
        (
            1.0,
            10.0,
            0,
            -1.0,
        ),
    ],
)
def test_negative_route_metrics_rejected(
    distance,
    travel_time,
    late_deliveries,
    lateness,
):
    evaluator = (
        make_evaluator()
    )

    with pytest.raises(
        ValueError
    ):
        evaluator.evaluate(
            total_distance_km=(
                distance
            ),
            total_travel_time_minutes=(
                travel_time
            ),
            late_deliveries=(
                late_deliveries
            ),
            total_lateness_minutes=(
                lateness
            ),
        )


def test_late_deliveries_must_be_integer():
    evaluator = (
        make_evaluator()
    )

    with pytest.raises(
        ValueError
    ):
        evaluator.evaluate(
            total_distance_km=1.0,
            total_travel_time_minutes=1.0,
            late_deliveries=1.5,
            total_lateness_minutes=0.0,
        )