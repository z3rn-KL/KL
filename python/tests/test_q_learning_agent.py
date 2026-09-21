from datetime import datetime, timedelta

import numpy as np
import pytest

from domain import Delivery

from rl import (
    QLearningAgent,
    QLearningConfig,
    RoutingEnvironment,
    RoutingState,
)

from routing import (
    RoadMatrixResult,
)


def create_delivery(
    delivery_id: str,
    start_time: datetime,
    deadline_minutes: int = 60,
) -> Delivery:
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
            start_time
            + timedelta(
                minutes=deadline_minutes
            )
        ),
        service_type="3h",
    )


def create_environment():
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
            deadline_minutes=60,
        ),
        create_delivery(
            "B",
            start,
            deadline_minutes=60,
        ),
    ]

    distance = np.array(
        [
            [
                0.0,
                1.0,
                3.0,
            ],
            [
                1.0,
                0.0,
                1.0,
            ],
            [
                3.0,
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
                5.0,
                15.0,
            ],
            [
                5.0,
                0.0,
                5.0,
            ],
            [
                15.0,
                5.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    matrix = RoadMatrixResult(
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

    return RoutingEnvironment(
        deliveries=deliveries,
        road_matrix=matrix,
        start_time=start,
    )


def test_unseen_q_value_is_zero():
    agent = QLearningAgent()

    state = RoutingState(
        current_index=0,
        visited_mask=0,
        elapsed_time_bucket=0,
    )

    assert (
        agent.get_q_value(
            state,
            0,
        )
        == pytest.approx(
            0.0
        )
    )


def test_q_value_can_be_set():
    agent = QLearningAgent()

    state = RoutingState(
        current_index=0,
        visited_mask=0,
        elapsed_time_bucket=0,
    )

    agent.set_q_value(
        state=state,
        action=1,
        value=3.5,
    )

    assert (
        agent.get_q_value(
            state,
            1,
        )
        == pytest.approx(
            3.5
        )
    )


def test_best_action_uses_highest_q_value():
    agent = QLearningAgent()

    state = RoutingState(
        current_index=0,
        visited_mask=0,
        elapsed_time_bucket=0,
    )

    agent.set_q_value(
        state,
        0,
        1.0,
    )

    agent.set_q_value(
        state,
        1,
        5.0,
    )

    assert (
        agent.best_action(
            state,
            (
                0,
                1,
            ),
        )
        == 1
    )


def test_best_action_tie_breaks_by_lower_action():
    agent = QLearningAgent()

    state = RoutingState(
        current_index=0,
        visited_mask=0,
        elapsed_time_bucket=0,
    )

    assert (
        agent.best_action(
            state,
            (
                1,
                0,
            ),
        )
        == 0
    )


def test_terminal_q_update_is_correct():
    config = QLearningConfig(
        learning_rate=0.50,
        discount_factor=0.90,
    )

    agent = QLearningAgent(
        config
    )

    state = RoutingState(
        current_index=0,
        visited_mask=0,
        elapsed_time_bucket=0,
    )

    next_state = RoutingState(
        current_index=0,
        visited_mask=1,
        elapsed_time_bucket=1,
    )

    new_q = agent.update(
        state=state,
        action=0,
        reward=10.0,
        next_state=next_state,
        next_actions=tuple(),
        done=True,
    )

    # old Q = 0
    # target = 10
    # alpha = 0.5
    #
    # new Q = 5

    assert (
        new_q
        == pytest.approx(
            5.0
        )
    )


def test_non_terminal_q_update_uses_future_value():
    config = QLearningConfig(
        learning_rate=0.50,
        discount_factor=0.90,
    )

    agent = QLearningAgent(
        config
    )

    state = RoutingState(
        current_index=0,
        visited_mask=0,
        elapsed_time_bucket=0,
    )

    next_state = RoutingState(
        current_index=1,
        visited_mask=1,
        elapsed_time_bucket=1,
    )

    agent.set_q_value(
        next_state,
        1,
        4.0,
    )

    new_q = agent.update(
        state=state,
        action=0,
        reward=2.0,
        next_state=next_state,
        next_actions=(
            1,
        ),
        done=False,
    )

    # target =
    # 2 + 0.9 * 4
    # = 5.6
    #
    # new Q =
    # 0 + 0.5 * 5.6
    # = 2.8

    assert (
        new_q
        == pytest.approx(
            2.8
        )
    )


def test_training_episode_visits_all_deliveries():
    environment = (
        create_environment()
    )

    agent = QLearningAgent(
        QLearningConfig(
            episodes=1,
            epsilon_start=0.0,
            epsilon_min=0.0,
        )
    )

    result = (
        agent.train_episode(
            environment=environment,
            episode=1,
            epsilon=0.0,
        )
    )

    assert (
        result.steps
        == 2
    )

    assert set(
        result.delivery_order
    ) == {
        "A",
        "B",
    }

    assert environment.done is True


def test_training_populates_q_table():
    environment = (
        create_environment()
    )

    agent = QLearningAgent(
        QLearningConfig(
            episodes=10,
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9,
        )
    )

    result = agent.train(
        environment
    )

    assert (
        len(
            result.episodes
        )
        == 10
    )

    assert (
        result.q_table_size
        > 0
    )

    assert (
        len(
            agent.q_table
        )
        > 0
    )


def test_epsilon_never_decays_below_minimum():
    environment = (
        create_environment()
    )

    config = QLearningConfig(
        episodes=100,
        epsilon_start=0.20,
        epsilon_min=0.10,
        epsilon_decay=0.50,
    )

    agent = QLearningAgent(
        config
    )

    result = agent.train(
        environment
    )

    assert (
        result.final_epsilon
        == pytest.approx(
            0.10
        )
    )


def test_greedy_rollout_visits_all_deliveries():
    environment = (
        create_environment()
    )

    agent = QLearningAgent(
        QLearningConfig(
            episodes=50,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.95,
        )
    )

    agent.train(
        environment
    )

    result = agent.greedy_rollout(
        environment
    )

    assert (
        result.steps
        == 2
    )

    assert set(
        result.delivery_order
    ) == {
        "A",
        "B",
    }

    assert (
        result.total_distance_km
        > 0.0
    )

    assert (
        result.total_travel_time_minutes
        > 0.0
    )


def test_clear_q_table():
    agent = QLearningAgent()

    state = RoutingState(
        current_index=0,
        visited_mask=0,
        elapsed_time_bucket=0,
    )

    agent.set_q_value(
        state,
        0,
        5.0,
    )

    agent.clear_q_table()

    assert (
        len(
            agent.q_table
        )
        == 0
    )


def test_invalid_learning_rate_is_rejected():
    config = QLearningConfig(
        learning_rate=0.0
    )

    with pytest.raises(
        ValueError
    ):
        QLearningAgent(
            config
        )