import numpy as np
import pytest

from rl.shift.agent import ShiftAgent, ShiftAgentConfig
from rl.shift.encoder import FEATURE_DIMENSION


def small_agent(**overrides):
    config = ShiftAgentConfig(
        batch_size=32,
        min_replay_size=32,
        replay_capacity=2_000,
        hidden_dimension=32,
        learning_rate=3e-3,
        **overrides,
    )

    return ShiftAgent(config)


def test_learns_to_prefer_the_candidate_with_the_higher_reward():
    agent = small_agent()
    rng = np.random.default_rng(0)

    for _ in range(600):
        features = rng.uniform(-1, 1, (3, FEATURE_DIMENSION)).astype(np.float32)
        pick = int(rng.integers(0, 3))

        # One-step episode: the reward is the first feature of the choice.
        agent.remember(features[pick], float(features[pick, 0]), None, True)

    for _ in range(600):
        agent.train_step()

    correct = 0

    for _ in range(200):
        features = rng.uniform(-1, 1, (3, FEATURE_DIMENSION)).astype(np.float32)
        correct += agent.greedy(features) == int(np.argmax(features[:, 0]))

    assert correct >= 180


def test_terminal_target_is_just_the_reward():
    agent = small_agent()
    row = np.linspace(-0.5, 0.5, FEATURE_DIMENSION).astype(np.float32)

    for _ in range(64):
        agent.remember(row, -2.0, None, True)

    for _ in range(800):
        agent.train_step()

    assert agent.scores(row[None, :])[0] == pytest.approx(-2.0, abs=0.15)


def test_padding_never_wins_the_max():
    agent = small_agent()
    rng = np.random.default_rng(1)

    for size in (1, 2, 5, 9):
        for _ in range(20):
            chosen = rng.normal(size=FEATURE_DIMENSION).astype(np.float32)
            nxt = rng.normal(size=(size, FEATURE_DIMENSION)).astype(np.float32)
            agent.remember(chosen, -1.0, nxt, False)

    losses = [agent.train_step() for _ in range(20)]

    assert all(loss is not None and np.isfinite(loss) for loss in losses)


def test_rejects_malformed_transitions():
    agent = small_agent()
    row = np.zeros(FEATURE_DIMENSION, np.float32)

    with pytest.raises(ValueError):
        agent.remember(row, 0.0, np.zeros((1, FEATURE_DIMENSION), np.float32), True)

    with pytest.raises(ValueError):
        agent.remember(row, 0.0, None, False)


def test_no_training_before_enough_replay():
    agent = small_agent()

    assert agent.train_step() is None


def test_residual_stop_row_is_exactly_its_base():
    agent = ShiftAgent(
        ShiftAgentConfig(
            batch_size=32, min_replay_size=32, replay_capacity=500, residual=True
        )
    )

    features = np.random.default_rng(0).normal(size=(3, FEATURE_DIMENSION)).astype(
        np.float32
    )
    features[:, 0] = [0.0, 0.0, 1.0]
    base = np.array([-0.5, -0.2, -1.5], np.float32)

    scores = agent.scores(features, base)

    assert scores[2] == pytest.approx(-1.5)
    assert scores[0] != pytest.approx(-0.5)  # the network adds a future term


def test_residual_agent_learns_the_future_part_only():
    agent = ShiftAgent(
        ShiftAgentConfig(
            batch_size=32,
            min_replay_size=32,
            replay_capacity=2_000,
            hidden_dimension=32,
            learning_rate=3e-3,
            residual=True,
        )
    )
    rng = np.random.default_rng(2)

    for _ in range(600):
        features = rng.uniform(-1, 1, (3, FEATURE_DIMENSION)).astype(np.float32)
        features[:, 0] = 0.0
        base = rng.uniform(-1, 0, 3).astype(np.float32)
        pick = int(rng.integers(0, 3))

        # Total return = base + future, where the future is feature 1.
        agent.remember(
            features[pick],
            float(base[pick] + features[pick, 1]),
            None,
            True,
            chosen_base=float(base[pick]),
        )

    for _ in range(800):
        agent.train_step()

    correct = 0

    for _ in range(200):
        features = rng.uniform(-1, 1, (3, FEATURE_DIMENSION)).astype(np.float32)
        features[:, 0] = 0.0
        base = rng.uniform(-1, 0, 3).astype(np.float32)

        correct += agent.greedy(features, base) == int(
            np.argmax(base + features[:, 1])
        )

    assert correct >= 170
