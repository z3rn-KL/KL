from types import SimpleNamespace

import pytest
import torch

from rl.dqn_agent import (
    DQNAgent,
    DQNConfig,
    DQNTransition,
    ReplayBuffer,
)
from rl.routing_environment import (
    RoutingState,
)


class FakeRoutingEnvironment:
    """
    Small deterministic environment used
    only for testing DQN.

    It implements the same public API
    required by DQNAgent without using the
    real OSM road network.
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
        self.step_count = 0

        self.total_distance_km = 0.0
        self.total_travel_time_minutes = 0.0

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

        if not self.available_actions():
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


def make_test_config(
    **overrides,
) -> DQNConfig:
    values = {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.9,
        "episodes": 3,
        "batch_size": 2,
        "replay_capacity": 20,
        "min_replay_size": 2,
        "target_update_interval": 2,
        "hidden_dim": 8,
        "gradient_clip_norm": 5.0,
        "elapsed_time_bucket_scale": 36.0,
        "random_seed": 42,
        "device": "cpu",
    }

    values.update(
        overrides
    )

    return DQNConfig(
        **values
    )


def test_default_config_is_valid():
    config = DQNConfig()

    config.validate()


def test_invalid_learning_rate():
    config = DQNConfig(
        learning_rate=0.0
    )

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_invalid_discount_factor():
    config = DQNConfig(
        discount_factor=1.1
    )

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_invalid_epsilon_configuration():
    config = DQNConfig(
        epsilon_start=0.2,
        epsilon_min=0.3,
    )

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_invalid_replay_configuration():
    config = DQNConfig(
        replay_capacity=10,
        min_replay_size=20,
    )

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_agent_dimensions():
    agent = DQNAgent(
        delivery_count=5,
        config=make_test_config(),
    )

    assert agent.input_dim == 12
    assert agent.action_dim == 5
    assert agent.parameter_count > 0


def test_agent_uses_requested_cpu():
    agent = DQNAgent(
        delivery_count=2,
        config=make_test_config(
            device="cpu"
        ),
    )

    assert str(
        agent.device
    ) == "cpu"


def test_encode_state():
    agent = DQNAgent(
        delivery_count=3,
        config=make_test_config(),
    )

    state = make_state(
        current_index=2,
        visited_mask=(
            (1 << 0)
            | (1 << 2)
        ),
        elapsed_time_bucket=18,
    )

    encoded = (
        agent.encode_state(
            state
        )
        .detach()
        .cpu()
    )

    expected = torch.tensor(
        [
            # Current-position one-hot:
            0.0,
            0.0,
            1.0,
            0.0,

            # Visited mask:
            1.0,
            0.0,
            1.0,

            # 18 / 36:
            0.5,
        ],
        dtype=torch.float32,
    )

    assert encoded.shape == (
        8,
    )

    assert torch.allclose(
        encoded,
        expected,
    )


def test_encode_state_rejects_invalid_current_index():
    agent = DQNAgent(
        delivery_count=2,
        config=make_test_config(),
    )

    state = make_state(
        current_index=3
    )

    with pytest.raises(
        ValueError
    ):
        agent.encode_state(
            state
        )


def test_best_action_respects_action_mask():
    agent = DQNAgent(
        delivery_count=3,
        config=make_test_config(),
    )

    final_layer = (
        agent.online_network
        .network[
            -1
        ]
    )

    with torch.no_grad():
        final_layer.weight.zero_()

        final_layer.bias.copy_(
            torch.tensor(
                [
                    1.0,
                    100.0,
                    2.0,
                ],
                dtype=torch.float32,
            )
        )

    state = make_state()

    action = (
        agent.best_action(
            state=state,
            available_actions=(
                0,
                2,
            ),
        )
    )

    # Action 1 has the largest Q-value,
    # but it is unavailable and therefore
    # must not be selected.
    assert action == 2


def test_best_action_tie_breaks_by_smallest_action():
    agent = DQNAgent(
        delivery_count=2,
        config=make_test_config(),
    )

    final_layer = (
        agent.online_network
        .network[
            -1
        ]
    )

    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.zero_()

    state = make_state()

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


def test_choose_action_zero_epsilon_is_greedy():
    agent = DQNAgent(
        delivery_count=2,
        config=make_test_config(),
    )

    final_layer = (
        agent.online_network
        .network[
            -1
        ]
    )

    with torch.no_grad():
        final_layer.weight.zero_()

        final_layer.bias.copy_(
            torch.tensor(
                [
                    1.0,
                    5.0,
                ]
            )
        )

    state = make_state()

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


def test_replay_buffer_capacity():
    buffer = ReplayBuffer(
        capacity=2,
        random_seed=42,
    )

    for index in range(
        3
    ):
        transition = DQNTransition(
            state=make_state(
                elapsed_time_bucket=index
            ),
            action=0,
            reward=1.0,
            next_state=make_state(
                elapsed_time_bucket=(
                    index + 1
                )
            ),
            next_actions=(
                0,
            ),
            done=False,
        )

        buffer.append(
            transition
        )

    assert len(
        buffer
    ) == 2


def test_replay_buffer_sample():
    buffer = ReplayBuffer(
        capacity=10,
        random_seed=42,
    )

    for action in range(
        3
    ):
        buffer.append(
            DQNTransition(
                state=make_state(),
                action=action,
                reward=float(
                    action
                ),
                next_state=make_state(
                    current_index=1
                ),
                next_actions=(
                    0,
                ),
                done=False,
            )
        )

    sample = buffer.sample(
        2
    )

    assert len(
        sample
    ) == 2

    assert all(
        isinstance(
            transition,
            DQNTransition,
        )
        for transition
        in sample
    )


def test_remember_adds_transition():
    agent = DQNAgent(
        delivery_count=2,
        config=make_test_config(),
    )

    state = make_state()

    next_state = make_state(
        current_index=1,
        visited_mask=1,
    )

    agent.remember(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        next_actions=(
            1,
        ),
        done=False,
    )

    assert len(
        agent.replay_buffer
    ) == 1


def test_optimize_waits_for_minimum_replay_size():
    agent = DQNAgent(
        delivery_count=2,
        config=make_test_config(
            batch_size=2,
            min_replay_size=2,
        ),
    )

    agent.remember(
        state=make_state(),
        action=0,
        reward=1.0,
        next_state=make_state(
            current_index=1,
            visited_mask=1,
        ),
        next_actions=(
            1,
        ),
        done=False,
    )

    loss = agent.optimize()

    assert loss is None
    assert (
        agent.optimization_steps
        == 0
    )


def test_optimize_performs_gradient_update():
    agent = DQNAgent(
        delivery_count=2,
        config=make_test_config(
            batch_size=2,
            min_replay_size=2,
            target_update_interval=10,
        ),
    )

    first_state = make_state()

    second_state = make_state(
        current_index=1,
        visited_mask=1,
        elapsed_time_bucket=1,
    )

    terminal_state = make_state(
        current_index=2,
        visited_mask=3,
        elapsed_time_bucket=2,
    )

    agent.remember(
        state=first_state,
        action=0,
        reward=1.0,
        next_state=second_state,
        next_actions=(
            1,
        ),
        done=False,
    )

    agent.remember(
        state=second_state,
        action=1,
        reward=2.0,
        next_state=terminal_state,
        next_actions=tuple(),
        done=True,
    )

    parameters_before = [
        parameter
        .detach()
        .clone()
        for parameter
        in agent.online_network.parameters()
    ]

    loss = agent.optimize()

    parameters_after = list(
        agent.online_network.parameters()
    )

    assert loss is not None
    assert loss >= 0.0

    assert (
        agent.optimization_steps
        == 1
    )

    changed = any(
        not torch.equal(
            before,
            after.detach(),
        )
        for (
            before,
            after,
        ) in zip(
            parameters_before,
            parameters_after,
            strict=True,
        )
    )

    assert changed


def test_sync_target_network():
    agent = DQNAgent(
        delivery_count=2,
        config=make_test_config(),
    )

    with torch.no_grad():
        first_parameter = next(
            agent.online_network
            .parameters()
        )

        first_parameter.add_(
            1.0
        )

    different_before = any(
        not torch.equal(
            online.detach(),
            target.detach(),
        )
        for (
            online,
            target,
        ) in zip(
            agent.online_network.parameters(),
            agent.target_network.parameters(),
            strict=True,
        )
    )

    assert different_before

    agent.sync_target_network()

    for (
        online,
        target,
    ) in zip(
        agent.online_network.parameters(),
        agent.target_network.parameters(),
        strict=True,
    ):
        assert torch.equal(
            online.detach(),
            target.detach(),
        )


def test_train_episode_completes_all_deliveries():
    config = make_test_config(
        epsilon_start=0.0,
        epsilon_min=0.0,
        episodes=1,
        batch_size=1,
        min_replay_size=1,
    )

    agent = DQNAgent(
        delivery_count=2,
        config=config,
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

    assert (
        agent.optimization_steps
        > 0
    )


def test_train_returns_training_summary():
    config = make_test_config(
        episodes=3,
        epsilon_start=0.5,
        epsilon_min=0.1,
        epsilon_decay=0.5,
        batch_size=1,
        min_replay_size=1,
    )

    agent = DQNAgent(
        delivery_count=2,
        config=config,
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

    assert result.replay_size > 0

    assert (
        result.optimization_steps
        > 0
    )

    assert result.parameter_count == (
        agent.parameter_count
    )

    assert result.device == "cpu"


def test_greedy_rollout_does_not_train():
    config = make_test_config(
        batch_size=1,
        min_replay_size=1,
    )

    agent = DQNAgent(
        delivery_count=2,
        config=config,
    )

    environment = (
        FakeRoutingEnvironment(
            delivery_count=2
        )
    )

    replay_before = len(
        agent.replay_buffer
    )

    optimization_before = (
        agent.optimization_steps
    )

    parameters_before = [
        parameter
        .detach()
        .clone()
        for parameter
        in agent.online_network.parameters()
    ]

    result = (
        agent.greedy_rollout(
            environment
        )
    )

    assert result.steps == 2

    assert len(
        agent.replay_buffer
    ) == replay_before

    assert (
        agent.optimization_steps
        == optimization_before
    )

    for (
        before,
        after,
    ) in zip(
        parameters_before,
        agent.online_network.parameters(),
        strict=True,
    ):
        assert torch.equal(
            before,
            after.detach(),
        )


def test_clear_replay_buffer():
    agent = DQNAgent(
        delivery_count=2,
        config=make_test_config(),
    )

    agent.remember(
        state=make_state(),
        action=0,
        reward=1.0,
        next_state=make_state(
            current_index=1,
        ),
        next_actions=(
            1,
        ),
        done=False,
    )

    assert len(
        agent.replay_buffer
    ) == 1

    agent.clear_replay_buffer()

    assert len(
        agent.replay_buffer
    ) == 0