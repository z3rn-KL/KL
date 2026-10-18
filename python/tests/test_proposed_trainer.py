import pytest

from rl.proposed.agent import (
    ProposedAgentConfig,
    ProposedRoutingAgent,
)
from rl.proposed.candidate_encoder import (
    CandidateEncoder,
)
from rl.proposed.environment import (
    ProposedRoutingEnvironment,
)
from rl.proposed.trainer import (
    evaluate_agent,
    run_training_episode,
    train_agent,
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
    epsilon_start=1.0,
    epsilon_decay=0.9,
    batch_size=2,
    min_replay_size=2,
):
    return ProposedRoutingAgent(
        config=ProposedAgentConfig(
            epsilon_start=epsilon_start,
            epsilon_min=0.0,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            replay_capacity=100,
            target_update_interval=2,
            hidden_dimension=16,
            device="cpu",
            seed=42,
        )
    )


def test_training_episode_completes_route():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent()

    result = run_training_episode(
        env=env,
        encoder=encoder,
        agent=agent,
        episode_index=1,
    )

    assert len(result.route_actions) == 3

    assert set(result.route_actions) == {
        0,
        1,
        2,
    }

    assert env.done is True


def test_episode_reward_matches_negative_objective():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent()

    result = run_training_episode(
        env=env,
        encoder=encoder,
        agent=agent,
        episode_index=1,
    )

    assert result.total_reward == pytest.approx(
        -result.objective_cost
    )


def test_training_episode_adds_one_transition_per_delivery():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent()

    run_training_episode(
        env=env,
        encoder=encoder,
        agent=agent,
        episode_index=1,
    )

    assert len(agent.replay_buffer) == 3


def test_epsilon_decays_once_per_episode():
    env = build_environment()
    encoder = CandidateEncoder()

    agent = build_agent(
        epsilon_start=1.0,
        epsilon_decay=0.9,
    )

    result = run_training_episode(
        env=env,
        encoder=encoder,
        agent=agent,
        episode_index=1,
    )

    assert result.epsilon_before == pytest.approx(
        1.0
    )

    assert result.epsilon_after == pytest.approx(
        0.9
    )


def test_training_produces_gradient_updates():
    env = build_environment()
    encoder = CandidateEncoder()

    agent = build_agent(
        batch_size=2,
        min_replay_size=2,
    )

    result = run_training_episode(
        env=env,
        encoder=encoder,
        agent=agent,
        episode_index=1,
    )

    assert result.gradient_updates > 0
    assert result.mean_loss is not None
    assert result.mean_loss >= 0.0


def test_train_agent_runs_multiple_episodes():
    env = build_environment()
    encoder = CandidateEncoder()

    agent = build_agent()

    run = train_agent(
        env=env,
        encoder=encoder,
        agent=agent,
        episodes=5,
    )

    assert run.episode_count == 5

    assert len(run.episodes) == 5

    for episode in run.episodes:
        assert len(
            episode.route_actions
        ) == 3


def test_replay_buffer_grows_across_episodes():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent()

    train_agent(
        env=env,
        encoder=encoder,
        agent=agent,
        episodes=4,
    )

    # 3 deliveries x 4 episodes
    assert len(agent.replay_buffer) == 12


def test_evaluation_does_not_change_replay_size():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent()

    train_agent(
        env=env,
        encoder=encoder,
        agent=agent,
        episodes=2,
    )

    replay_before = len(
        agent.replay_buffer
    )

    evaluate_agent(
        env=env,
        encoder=encoder,
        agent=agent,
    )

    replay_after = len(
        agent.replay_buffer
    )

    assert replay_before == replay_after


def test_evaluation_does_not_decay_epsilon():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent()

    epsilon_before = agent.epsilon

    evaluate_agent(
        env=env,
        encoder=encoder,
        agent=agent,
    )

    assert agent.epsilon == pytest.approx(
        epsilon_before
    )


def test_evaluation_completes_valid_route():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent(
        epsilon_start=0.0,
    )

    result = evaluate_agent(
        env=env,
        encoder=encoder,
        agent=agent,
    )

    assert len(result.route_actions) == 3

    assert set(result.route_actions) == {
        0,
        1,
        2,
    }

    assert result.total_reward == pytest.approx(
        -result.objective_cost
    )


def test_invalid_episode_count_is_rejected():
    env = build_environment()
    encoder = CandidateEncoder()
    agent = build_agent()

    with pytest.raises(ValueError):
        train_agent(
            env=env,
            encoder=encoder,
            agent=agent,
            episodes=0,
        )