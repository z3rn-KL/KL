from datetime import datetime, timedelta

import numpy as np
import pytest

from domain import Delivery

from rl import (
    RewardConfig,
    RoutingEnvironment,
    RoutingState,
)

from routing import (
    RoadMatrixResult,
)


def create_delivery(
    delivery_id: str,
    start_time: datetime,
    deadline_minutes: int | None = 60,
) -> Delivery:
    if deadline_minutes is None:
        deadline = None
    else:
        deadline = (
            start_time
            + timedelta(
                minutes=deadline_minutes
            )
        )

    return Delivery(
        delivery_id=delivery_id,
        latitude=10.0,
        longitude=106.0,
        created_at=(
            start_time
            - timedelta(
                minutes=30
            )
        ),
        expected_delivery_time=(
            deadline
        ),
        service_type="3h",
    )


def create_two_delivery_matrix():
    """
    Matrix order:

        0 = depot
        1 = A
        2 = B
    """

    distance = np.array(
        [
            [
                0.0,
                1.0,
                2.0,
            ],
            [
                1.0,
                0.0,
                1.0,
            ],
            [
                2.0,
                1.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    travel_time = np.array(
        [
            [
                0.0,
                10.0,
                20.0,
            ],
            [
                10.0,
                0.0,
                10.0,
            ],
            [
                20.0,
                10.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    return RoadMatrixResult(
        node_ids=(
            100,
            101,
            102,
        ),
        distance_km=distance,
        travel_time_minutes=(
            travel_time
        ),
    )


def create_single_delivery_matrix():
    distance = np.array(
        [
            [
                0.0,
                1.0,
            ],
            [
                1.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    travel_time = np.array(
        [
            [
                0.0,
                10.0,
            ],
            [
                10.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    return RoadMatrixResult(
        node_ids=(
            100,
            101,
        ),
        distance_km=distance,
        travel_time_minutes=(
            travel_time
        ),
    )


def test_reset_returns_initial_state():
    start = datetime(
        2026,
        9,
        21,
        8,
        0,
    )

    deliveries = [
        create_delivery(
            "A",
            start,
        ),
        create_delivery(
            "B",
            start,
        ),
    ]

    environment = (
        RoutingEnvironment(
            deliveries=deliveries,
            road_matrix=(
                create_two_delivery_matrix()
            ),
            start_time=start,
        )
    )

    state = environment.reset()

    assert state == RoutingState(
        current_index=0,
        visited_mask=0,
        elapsed_time_bucket=0,
    )


def test_all_deliveries_are_initially_available():
    start = datetime(
        2026,
        9,
        21,
        8,
        0,
    )

    environment = (
        RoutingEnvironment(
            deliveries=[
                create_delivery(
                    "A",
                    start,
                ),
                create_delivery(
                    "B",
                    start,
                ),
            ],
            road_matrix=(
                create_two_delivery_matrix()
            ),
            start_time=start,
        )
    )

    assert (
        environment.available_actions()
        == (
            0,
            1,
        )
    )


def test_step_marks_delivery_as_visited():
    start = datetime(
        2026,
        9,
        21,
        8,
        0,
    )

    environment = (
        RoutingEnvironment(
            deliveries=[
                create_delivery(
                    "A",
                    start,
                ),
                create_delivery(
                    "B",
                    start,
                ),
            ],
            road_matrix=(
                create_two_delivery_matrix()
            ),
            start_time=start,
        )
    )

    state, _, done, _ = (
        environment.step(
            0
        )
    )

    assert done is False

    assert (
        state.current_index
        == 1
    )

    assert (
        state.visited_mask
        == 1
    )

    assert (
        environment.available_actions()
        == (
            1,
        )
    )


def test_time_advances_after_step():
    start = datetime(
        2026,
        9,
        21,
        8,
        0,
    )

    environment = (
        RoutingEnvironment(
            deliveries=[
                create_delivery(
                    "A",
                    start,
                ),
                create_delivery(
                    "B",
                    start,
                ),
            ],
            road_matrix=(
                create_two_delivery_matrix()
            ),
            start_time=start,
        )
    )

    state, _, _, info = (
        environment.step(
            0
        )
    )

    assert (
        info.arrival_time
        == (
            start
            + timedelta(
                minutes=10
            )
        )
    )

    assert (
        state.elapsed_time_bucket
        == 1
    )


def test_repeated_action_is_rejected():
    start = datetime(
        2026,
        9,
        21,
        8,
        0,
    )

    environment = (
        RoutingEnvironment(
            deliveries=[
                create_delivery(
                    "A",
                    start,
                ),
                create_delivery(
                    "B",
                    start,
                ),
            ],
            road_matrix=(
                create_two_delivery_matrix()
            ),
            start_time=start,
        )
    )

    environment.step(
        0
    )

    with pytest.raises(
        ValueError
    ):
        environment.step(
            0
        )


def test_final_delivery_returns_to_depot():
    start = datetime(
        2026,
        9,
        21,
        8,
        0,
    )

    environment = (
        RoutingEnvironment(
            deliveries=[
                create_delivery(
                    "A",
                    start,
                ),
            ],
            road_matrix=(
                create_single_delivery_matrix()
            ),
            start_time=start,
        )
    )

    state, _, done, info = (
        environment.step(
            0
        )
    )

    assert done is True

    assert (
        state.current_index
        == 0
    )

    assert (
        environment.total_distance_km
        == pytest.approx(
            2.0
        )
    )

    assert (
        environment
        .total_travel_time_minutes
        == pytest.approx(
            20.0
        )
    )

    assert (
        info.return_distance_km
        == pytest.approx(
            1.0
        )
    )

    assert (
        info.return_travel_time_minutes
        == pytest.approx(
            10.0
        )
    )


def test_late_delivery_receives_lower_reward():
    start = datetime(
        2026,
        9,
        21,
        8,
        0,
    )

    matrix = (
        create_single_delivery_matrix()
    )

    on_time_environment = (
        RoutingEnvironment(
            deliveries=[
                create_delivery(
                    "A",
                    start,
                    deadline_minutes=30,
                )
            ],
            road_matrix=matrix,
            start_time=start,
        )
    )

    late_environment = (
        RoutingEnvironment(
            deliveries=[
                create_delivery(
                    "A",
                    start,
                    deadline_minutes=5,
                )
            ],
            road_matrix=matrix,
            start_time=start,
        )
    )

    (
        _,
        on_time_reward,
        _,
        on_time_info,
    ) = on_time_environment.step(
        0
    )

    (
        _,
        late_reward,
        _,
        late_info,
    ) = late_environment.step(
        0
    )

    assert (
        on_time_info.on_time
        is True
    )

    assert (
        late_info.on_time
        is False
    )

    assert (
        late_info.lateness_minutes
        == pytest.approx(
            5.0
        )
    )

    assert (
        on_time_reward
        >
        late_reward
    )


def test_missing_deadline_is_supported():
    start = datetime(
        2026,
        9,
        21,
        8,
        0,
    )

    environment = (
        RoutingEnvironment(
            deliveries=[
                create_delivery(
                    "A",
                    start,
                    deadline_minutes=None,
                )
            ],
            road_matrix=(
                create_single_delivery_matrix()
            ),
            start_time=start,
        )
    )

    _, reward, done, info = (
        environment.step(
            0
        )
    )

    assert done is True

    assert (
        info.on_time
        is None
    )

    assert (
        info.lateness_minutes
        is None
    )

    assert np.isfinite(
        reward
    )


def test_invalid_action_is_rejected():
    start = datetime(
        2026,
        9,
        21,
        8,
        0,
    )

    environment = (
        RoutingEnvironment(
            deliveries=[
                create_delivery(
                    "A",
                    start,
                )
            ],
            road_matrix=(
                create_single_delivery_matrix()
            ),
            start_time=start,
        )
    )

    with pytest.raises(
        ValueError
    ):
        environment.step(
            5
        )


def test_reward_config_rejects_invalid_scale():
    config = RewardConfig(
        distance_scale_km=0.0
    )

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_state_is_hashable_for_q_table():
    state = RoutingState(
        current_index=2,
        visited_mask=5,
        elapsed_time_bucket=3,
    )

    q_table = {
        (
            state,
            1,
        ): 42.0
    }

    assert (
        q_table[
            (
                state,
                1,
            )
        ]
        == pytest.approx(
            42.0
        )
    )

def test_reward_rejects_nonfinite_weights():
    with pytest.raises(ValueError):
        RewardConfig(cluster_switch_penalty=float('nan')).validate()
    with pytest.raises(ValueError):
        RewardConfig(distance_scale_km=float('inf')).validate()


def test_cluster_ids_reject_fractional_values():
    start = datetime(2026, 9, 21)
    with pytest.raises(ValueError, match='integer'):
        RoutingEnvironment(
            deliveries=[create_delivery('a', start), create_delivery('b', start)],
            road_matrix=create_two_delivery_matrix(),
            start_time=start,
            delivery_cluster_ids=[0, 1.5],
        )
