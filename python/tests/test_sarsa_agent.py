from types import SimpleNamespace

import pytest

from rl.routing_environment import RoutingState
from rl.sarsa_agent import (
    SarsaAgent,
    SarsaConfig,
)


class FakeRoutingEnvironment:
    """
    Small deterministic environment used
    only for testing the SARSA agent.

    It mimics the public API required by
    SarsaAgent without depending on the
    real road network.
    """

    def __init__(
        self,
        delivery_count: int = 2,
    ) -> None:
        self.delivery_count = (
            delivery_count
        )

        self.reset()

    def reset(
        self,
    ) -> RoutingState:
        self.visited_mask = 0
        self.current_index = 0
        self.done = False

        self.total_distance_km = 0.0
        self.total_travel_time_minutes = 0.0

        self.step_count = 0

        return self.state

    @property
    def state(
        self,
    ) -> RoutingState:
        return RoutingState(
            current_index=(
                self.current_index
            ),
            visited_mask=(
                self.visited_mask
            ),
            elapsed_time_bucket=(
                self.step_count
            ),
        )

    def available_actions(
        self,
    ) -> tuple[int, ...]:
        if self.done:
            return tuple()

        actions = []

        for action in range(
            self.delivery_count
        ):
            visited = (
                self.visited_mask
                & (
                    1 << action
                )
            )

            if not visited:
                actions.append(
                    action
                )

        return tuple(
            actions
        )

    def step(
        self,
        action: int,
    ):
        if action not in (
            self.available_actions()
        ):
            raise ValueError(
                "Invalid action."
            )

        self.visited_mask |= (
            1 << action
        )

        self.current_index = (
            action + 1
        )

        self.step_count += 1

        distance = float(
            action + 1
        )

        travel_time = float(
            (
                action + 1
            )
            * 2
        )

        self.total_distance_km += (
            distance
        )

        self.total_travel_time_minutes += (
            travel_time
        )

        reward = 1.0

        if (
            len(
                self.available_actions()
            )
            == 0
        ):
            self.done = True

        info = SimpleNamespace(
            delivery_id=(
                f"delivery_{action}"
            )
        )

        return (
            self.state,
            reward,
            self.done,
            info,
        )


def make_state(
    current_index: int = 0,
    visited_mask: int = 0,
    elapsed_time_bucket: int = 0,
) -> RoutingState:
    return RoutingState(
        current_index=current_index,
        visited_mask=visited_mask,
        elapsed_time_bucket=(
            elapsed_time_bucket
        ),
    )


def test_default_config_is_valid():
    config = SarsaConfig()

    config.validate()


def test_invalid_learning_rate():
    config = SarsaConfig(
        learning_rate=0.0
    )

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_invalid_discount_factor():
    config = SarsaConfig(
        discount_factor=1.1
    )

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_invalid_epsilon_range():
    config = SarsaConfig(
        epsilon_start=1.1
    )

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_epsilon_min_cannot_exceed_start():
    config = SarsaConfig(
        epsilon_start=0.20,
        epsilon_min=0.30,
    )

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_unseen_q_value_is_zero():
    agent = SarsaAgent()

    state = make_state()

    assert (
        agent.get_q_value(
            state,
            0,
        )
        == 0.0
    )


def test_set_and_get_q_value():
    agent = SarsaAgent()

    state = make_state()

    agent.set_q_value(
        state=state,
        action=1,
        value=2.5,
    )

    assert (
        agent.get_q_value(
            state,
            1,
        )
        == pytest.approx(
            2.5
        )
    )


def test_best_action_selects_highest_q():
    agent = SarsaAgent()

    state = make_state()

    agent.set_q_value(
        state,
        0,
        1.0,
    )

    agent.set_q_value(
        state,
        1,
        3.0,
    )

    action = (
        agent.best_action(
            state=state,
            available_actions=(
                0,
                1,
            ),
        )
    )

    assert action == 1


def test_best_action_tie_breaks_by_smallest_action():
    agent = SarsaAgent()

    state = make_state()

    agent.set_q_value(
        state,
        0,
        2.0,
    )

    agent.set_q_value(
        state,
        1,
        2.0,
    )

    action = (
        agent.best_action(
            state=state,
            available_actions=(
                1,
                0,
            ),
        )
    )

    assert action == 0


def test_choose_action_with_zero_epsilon_is_greedy():
    agent = SarsaAgent()

    state = make_state()

    agent.set_q_value(
        state,
        0,
        1.0,
    )

    agent.set_q_value(
        state,
        1,
        4.0,
    )

    action = (
        agent.choose_action(
            state=state,
            available_actions=(
                0,
                1,
            ),
            epsilon=0.0,
        )
    )

    assert action == 1


def test_terminal_update_uses_reward_only():
    config = SarsaConfig(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    agent = SarsaAgent(
        config=config
    )

    state = make_state()

    next_state = make_state(
        current_index=1,
        visited_mask=1,
    )

    agent.set_q_value(
        state,
        0,
        2.0,
    )

    new_q = agent.update(
        state=state,
        action=0,
        reward=4.0,
        next_state=next_state,
        next_action=None,
        done=True,
    )

    expected = (
        2.0
        + 0.5
        * (
            4.0
            - 2.0
        )
    )

    assert new_q == pytest.approx(
        expected
    )


def test_non_terminal_update_uses_selected_next_action():
    config = SarsaConfig(
        learning_rate=0.5,
        discount_factor=0.9,
    )

    agent = SarsaAgent(
        config=config
    )

    state = make_state()

    next_state = make_state(
        current_index=1,
        visited_mask=1,
    )

    agent.set_q_value(
        state,
        0,
        1.0,
    )

    agent.set_q_value(
        next_state,
        1,
        2.0,
    )

    agent.set_q_value(
        next_state,
        2,
        10.0,
    )

    new_q = agent.update(
        state=state,
        action=0,
        reward=3.0,
        next_state=next_state,
        next_action=1,
        done=False,
    )

    target = (
        3.0
        + 0.9
        * 2.0
    )

    expected = (
        1.0
        + 0.5
        * (
            target
            - 1.0
        )
    )

    assert new_q == pytest.approx(
        expected
    )


def test_non_terminal_update_requires_next_action():
    agent = SarsaAgent()

    state = make_state()

    next_state = make_state(
        current_index=1
    )

    with pytest.raises(
        ValueError
    ):
        agent.update(
            state=state,
            action=0,
            reward=1.0,
            next_state=next_state,
            next_action=None,
            done=False,
        )


def test_train_episode_completes_all_deliveries():
    config = SarsaConfig(
        epsilon_start=0.0,
        epsilon_min=0.0,
        episodes=1,
    )

    agent = SarsaAgent(
        config=config
    )

    environment = (
        FakeRoutingEnvironment(
            delivery_count=2
        )
    )

    result = (
        agent.train_episode(
            environment=environment,
            episode=1,
            epsilon=0.0,
        )
    )

    assert result.steps == 2

    assert len(
        result.delivery_order
    ) == 2

    assert len(
        result.action_order
    ) == 2

    assert result.total_reward == (
        pytest.approx(
            2.0
        )
    )


def test_train_returns_training_summary():
    config = SarsaConfig(
        episodes=3,
        epsilon_start=0.5,
        epsilon_min=0.1,
        epsilon_decay=0.5,
        random_seed=42,
    )

    agent = SarsaAgent(
        config=config
    )

    environment = (
        FakeRoutingEnvironment(
            delivery_count=2
        )
    )

    result = agent.train(
        environment
    )

    assert len(
        result.episodes
    ) == 3

    assert result.final_epsilon == (
        pytest.approx(
            0.1
        )
    )

    assert result.q_table_size > 0


def test_greedy_rollout_does_not_modify_q_table():
    agent = SarsaAgent()

    environment = (
        FakeRoutingEnvironment(
            delivery_count=2
        )
    )

    initial_state = (
        environment.reset()
    )

    agent.set_q_value(
        initial_state,
        0,
        5.0,
    )

    table_before = dict(
        agent.q_table
    )

    result = (
        agent.greedy_rollout(
            environment
        )
    )

    assert result.steps == 2

    assert agent.q_table == (
        table_before
    )


def test_clear_q_table():
    agent = SarsaAgent()

    state = make_state()

    agent.set_q_value(
        state,
        0,
        1.0,
    )

    assert len(
        agent.q_table
    ) == 1

    agent.clear_q_table()

    assert len(
        agent.q_table
    ) == 0