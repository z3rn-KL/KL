import torch
import pytest

from rl.proposed.agent import (
    CandidateQNetwork,
    ProposedAgentConfig,
    ProposedRoutingAgent,
    ReplayBuffer,
    ReplayTransition,
)
from rl.proposed.candidate_encoder import (
    CandidateEncoder,
)
from rl.proposed.environment import (
    ProposedRoutingEnvironment,
)


def build_environment():
    return ProposedRoutingEnvironment(
        distance_matrix_km=[
            [0.0, 2.0, 4.0, 6.0],
            [2.0, 0.0, 2.0, 4.0],
            [4.0, 2.0, 0.0, 2.0],
            [6.0, 4.0, 2.0, 0.0],
        ],
        travel_time_matrix_minutes=[
            [0.0, 4.0, 8.0, 12.0],
            [4.0, 0.0, 4.0, 8.0],
            [8.0, 4.0, 0.0, 4.0],
            [12.0, 8.0, 4.0, 0.0],
        ],
        deadlines_minutes=[
            10.0,
            20.0,
            30.0,
        ],
        cluster_ids=[
            0,
            0,
            1,
        ],
    )


def build_agent(
    *,
    epsilon_start=0.0,
    batch_size=2,
    min_replay_size=2,
):
    return ProposedRoutingAgent(
        config=ProposedAgentConfig(
            epsilon_start=epsilon_start,
            epsilon_min=0.0,
            epsilon_decay=0.9,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            replay_capacity=100,
            target_update_interval=2,
            hidden_dimension=16,
            device="cpu",
            seed=42,
        )
    )


def test_q_network_outputs_one_value_per_candidate():
    network = CandidateQNetwork(
        input_dimension=8,
        hidden_dimension=16,
    )

    features = torch.zeros(
        (5, 8),
        dtype=torch.float32,
    )

    output = network(features)

    assert output.shape == (5,)


def test_agent_uses_eight_features():
    agent = build_agent()

    assert agent.feature_dimension == 8


def test_select_action_returns_available_action():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent()

    encoded = encoder.encode_available_actions(
        env
    )

    action = agent.select_action(
        encoded,
        explore=False,
    )

    assert action in env.available_actions()


def test_agent_does_not_select_visited_action():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent()

    env.step(0)

    encoded = encoder.encode_available_actions(
        env
    )

    action = agent.select_action(
        encoded,
        explore=False,
    )

    assert action in (1, 2)
    assert action != 0


def test_zero_network_selects_first_candidate():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent()

    for parameter in (
        agent.online_network.parameters()
    ):
        parameter.data.zero_()

    encoded = encoder.encode_available_actions(
        env
    )

    action = agent.select_action(
        encoded,
        explore=False,
    )

    assert action == 0


def test_replay_buffer_capacity_is_respected():
    buffer = ReplayBuffer(
        capacity=2,
        seed=42,
    )

    transition = ReplayTransition(
        chosen_features=(0.0,) * 8,
        reward=-1.0,
        next_candidate_features=(),
        done=True,
    )

    buffer.append(transition)
    buffer.append(transition)
    buffer.append(transition)

    assert len(buffer) == 2


def test_agent_cannot_train_before_minimum_replay():
    agent = build_agent(
        batch_size=2,
        min_replay_size=2,
    )

    assert agent.can_train() is False
    assert agent.train_step() is None


def test_terminal_transition_rejects_next_actions():
    agent = build_agent()

    with pytest.raises(ValueError):
        agent.remember(
            chosen_features=(0.0,) * 8,
            reward=-1.0,
            next_candidate_features=[
                (0.0,) * 8
            ],
            done=True,
        )


def test_epsilon_decays():
    agent = build_agent(
        epsilon_start=1.0,
    )

    new_epsilon = agent.decay_epsilon()

    assert new_epsilon == pytest.approx(
        0.9
    )


def test_training_step_runs():
    agent = build_agent(
        batch_size=2,
        min_replay_size=2,
    )

    next_features = (
        (0.1,) * 8,
        (0.2,) * 8,
    )

    agent.remember(
        chosen_features=(0.3,) * 8,
        reward=-0.2,
        next_candidate_features=next_features,
        done=False,
    )

    agent.remember(
        chosen_features=(0.4,) * 8,
        reward=-0.5,
        next_candidate_features=(),
        done=True,
    )

    result = agent.train_step()

    assert result is not None
    assert result.loss >= 0.0
    assert result.replay_size == 2
    assert result.gradient_step == 1


def test_target_network_syncs():
    agent = build_agent(
        batch_size=2,
        min_replay_size=2,
    )

    for parameter in (
        agent.online_network.parameters()
    ):
        parameter.data.fill_(1.0)

    agent.sync_target_network()

    online_parameters = list(
        agent.online_network.parameters()
    )

    target_parameters = list(
        agent.target_network.parameters()
    )

    for online, target in zip(
        online_parameters,
        target_parameters,
    ):
        assert torch.equal(
            online,
            target,
        )