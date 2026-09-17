from datetime import datetime, timedelta

import pytest

from domain import Delivery
from prioritization import (
    DeliveryPriorityModel,
    PriorityWeights,
)


def create_delivery(
    delivery_id: str,
    now: datetime,
    deadline_minutes: int | None = None,
    waiting_minutes: int | None = None,
    service_type: str | None = None,
) -> Delivery:
    expected_delivery_time = None
    created_at = None

    if deadline_minutes is not None:
        expected_delivery_time = (
            now
            + timedelta(
                minutes=deadline_minutes
            )
        )

    if waiting_minutes is not None:
        created_at = (
            now
            - timedelta(
                minutes=waiting_minutes
            )
        )

    return Delivery(
        delivery_id=delivery_id,
        latitude=10.7769,
        longitude=106.7009,
        created_at=created_at,
        expected_delivery_time=expected_delivery_time,
        service_type=service_type,
    )


def test_urgent_delivery_scores_higher():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    urgent = create_delivery(
        delivery_id="D001",
        now=now,
        deadline_minutes=20,
        waiting_minutes=160,
        service_type="3h",
    )

    normal = create_delivery(
        delivery_id="D002",
        now=now,
        deadline_minutes=240,
        waiting_minutes=60,
        service_type="5h",
    )

    model = DeliveryPriorityModel()

    urgent_result = model.calculate(
        delivery=urgent,
        current_time=now,
        travel_efficiency_score=0.50,
    )

    normal_result = model.calculate(
        delivery=normal,
        current_time=now,
        travel_efficiency_score=0.50,
    )

    assert (
        urgent_result.priority_score
        >
        normal_result.priority_score
    )


def test_overdue_delivery_has_high_urgency():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="D001",
        now=now,
        deadline_minutes=-10,
        waiting_minutes=100,
        service_type="3h",
    )

    urgency = (
        DeliveryPriorityModel
        .deadline_urgency(
            delivery=delivery,
            current_time=now,
        )
    )

    assert urgency is not None
    assert urgency > 0.80
    assert urgency < 1.0


def test_more_overdue_delivery_has_higher_urgency():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    slightly_overdue = create_delivery(
        delivery_id="D001",
        now=now,
        deadline_minutes=-5,
        waiting_minutes=100,
        service_type="3h",
    )

    heavily_overdue = create_delivery(
        delivery_id="D002",
        now=now,
        deadline_minutes=-40,
        waiting_minutes=100,
        service_type="3h",
    )

    slightly_overdue_urgency = (
        DeliveryPriorityModel
        .deadline_urgency(
            delivery=slightly_overdue,
            current_time=now,
        )
    )

    heavily_overdue_urgency = (
        DeliveryPriorityModel
        .deadline_urgency(
            delivery=heavily_overdue,
            current_time=now,
        )
    )

    assert slightly_overdue_urgency is not None
    assert heavily_overdue_urgency is not None

    assert (
        heavily_overdue_urgency
        >
        slightly_overdue_urgency
    )


def test_missing_deadline_is_not_imputed():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="D001",
        now=now,
        deadline_minutes=None,
        waiting_minutes=100,
        service_type="3h",
    )

    urgency = (
        DeliveryPriorityModel
        .deadline_urgency(
            delivery=delivery,
            current_time=now,
        )
    )

    assert urgency is None


def test_longer_waiting_order_has_higher_score():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    older = create_delivery(
        delivery_id="D001",
        now=now,
        waiting_minutes=240,
    )

    newer = create_delivery(
        delivery_id="D002",
        now=now,
        waiting_minutes=30,
    )

    older_score = (
        DeliveryPriorityModel
        .waiting_time_score(
            delivery=older,
            current_time=now,
        )
    )

    newer_score = (
        DeliveryPriorityModel
        .waiting_time_score(
            delivery=newer,
            current_time=now,
        )
    )

    assert older_score is not None
    assert newer_score is not None
    assert older_score > newer_score


def test_waiting_score_does_not_saturate_abruptly():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="D001",
        now=now,
        waiting_minutes=300,
    )

    score = (
        DeliveryPriorityModel
        .waiting_time_score(
            delivery=delivery,
            current_time=now,
        )
    )

    assert score is not None
    assert score > 0.0
    assert score < 1.0


def test_express_service_has_higher_priority():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    express = create_delivery(
        delivery_id="D001",
        now=now,
        service_type="3h",
    )

    storage = create_delivery(
        delivery_id="D002",
        now=now,
        service_type="luu_kho",
    )

    express_score = (
        DeliveryPriorityModel
        .service_priority(
            express
        )
    )

    storage_score = (
        DeliveryPriorityModel
        .service_priority(
            storage
        )
    )

    assert express_score is not None
    assert storage_score is not None

    assert express_score > storage_score


def test_missing_deadline_reduces_information_coverage():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="D001",
        now=now,
        deadline_minutes=None,
        waiting_minutes=240,
        service_type="3h",
    )

    model = DeliveryPriorityModel()

    result = model.calculate(
        delivery=delivery,
        current_time=now,
        travel_efficiency_score=0.80,
    )

    assert result.deadline_urgency is None

    assert (
        result.information_coverage
        == pytest.approx(0.45)
    )

    assert result.priority_score > 0.0


def test_complete_delivery_has_full_information_coverage():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="D001",
        now=now,
        deadline_minutes=60,
        waiting_minutes=100,
        service_type="3h",
    )

    model = DeliveryPriorityModel()

    result = model.calculate(
        delivery=delivery,
        current_time=now,
        travel_efficiency_score=0.75,
    )

    assert (
        result.information_coverage
        == pytest.approx(1.0)
    )

    assert (
        result.travel_efficiency_score
        == pytest.approx(0.75)
    )


def test_missing_travel_efficiency_reduces_information_coverage():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="D001",
        now=now,
        deadline_minutes=60,
        waiting_minutes=100,
        service_type="3h",
    )

    model = DeliveryPriorityModel()

    result = model.calculate(
        delivery=delivery,
        current_time=now,
        travel_efficiency_score=None,
    )

    assert (
        result.information_coverage
        == pytest.approx(0.90)
    )

    assert (
        result.travel_efficiency_score
        is None
    )


def test_invalid_negative_weight_is_rejected():
    weights = PriorityWeights(
        deadline_urgency=-0.50,
        service_priority=0.30,
        waiting_time=0.20,
        travel_efficiency=1.00,
    )

    with pytest.raises(
        ValueError
    ):
        DeliveryPriorityModel(
            weights=weights
        )


def test_all_zero_weights_are_rejected():
    weights = PriorityWeights(
        deadline_urgency=0.00,
        service_priority=0.00,
        waiting_time=0.00,
        travel_efficiency=0.00,
    )

    with pytest.raises(
        ValueError
    ):
        DeliveryPriorityModel(
            weights=weights
        )


def test_weights_must_sum_to_one():
    weights = PriorityWeights(
        deadline_urgency=0.50,
        service_priority=0.30,
        waiting_time=0.20,
        travel_efficiency=0.20,
    )

    with pytest.raises(
        ValueError
    ):
        DeliveryPriorityModel(
            weights=weights
        )


def test_travel_component_affects_priority():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="D001",
        now=now,
        deadline_minutes=100,
        waiting_minutes=100,
        service_type="3h",
    )

    model = DeliveryPriorityModel()

    low_travel_result = model.calculate(
        delivery=delivery,
        current_time=now,
        travel_efficiency_score=0.10,
    )

    high_travel_result = model.calculate(
        delivery=delivery,
        current_time=now,
        travel_efficiency_score=0.90,
    )

    assert (
        high_travel_result.priority_score
        >
        low_travel_result.priority_score
    )


def test_travel_score_is_clamped_to_valid_range():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    delivery = create_delivery(
        delivery_id="D001",
        now=now,
        deadline_minutes=100,
        waiting_minutes=100,
        service_type="3h",
    )

    model = DeliveryPriorityModel()

    result = model.calculate(
        delivery=delivery,
        current_time=now,
        travel_efficiency_score=5.0,
    )

    assert (
        result.travel_efficiency_score
        == pytest.approx(1.0)
    )


def test_sla_progress_is_calculated_correctly():
    now = datetime(
        2026,
        9,
        17,
        10,
        0,
    )

    delivery = Delivery(
        delivery_id="D001",
        latitude=10.7769,
        longitude=106.7009,
        created_at=now - timedelta(
            minutes=90
        ),
        expected_delivery_time=now + timedelta(
            minutes=90
        ),
        service_type="3h",
    )

    progress = (
        DeliveryPriorityModel
        .calculate_sla_progress(
            delivery=delivery,
            current_time=now,
        )
    )

    assert progress is not None

    assert progress == pytest.approx(
        0.50
    )