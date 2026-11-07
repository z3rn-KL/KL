import numpy as np
import pytest
import torch

from rl.proposed.agent import CandidateQNetwork
from rl.shift.encoder import FEATURE_DIMENSION, REWARD_SCALE, ShiftEncoder
from rl.shift.policy import (
    behaviour_clone,
    clone_environment,
    policy_gradient_step,
    run_batch,
)
from rl.shift.environment import ShiftEnvironment, ShiftOrder
from tests.test_shift_environment import random_environment


def profitable_environment(seed, n=4):
    """
    Short hops and valuable orders: serving everything beats deferring.
    """

    rng = np.random.default_rng(seed)

    time = rng.uniform(2.0, 5.0, (n + 1, n + 1))
    np.fill_diagonal(time, 0.0)

    return ShiftEnvironment(
        distance_km=(time / 5.0).tolist(),
        travel_minutes=time.tolist(),
        orders=[ShiftOrder(f"o{k}", 50_000.0, 600.0) for k in range(n)],
    )


def network():
    torch.manual_seed(0)

    return CandidateQNetwork(
        input_dimension=FEATURE_DIMENSION, hidden_dimension=16
    )


def test_greedy_rollout_matches_the_environment_score():
    envs = [random_environment(seed, n=4 + seed) for seed in range(3)]
    expected = []

    log_p, entropy, returns = run_batch(
        envs, ShiftEncoder(), network(), greedy=True
    )

    assert log_p == [None] * 3 and entropy == [None] * 3

    for env, value in zip(envs, returns):
        assert env.done
        assert value == pytest.approx(env.total_reward * REWARD_SCALE)


def test_sampled_rollout_returns_differentiable_log_probs():
    net = network()
    envs = [random_environment(1, n=5), random_environment(2, n=7)]

    log_p, entropy, _ = run_batch(
        envs,
        ShiftEncoder(),
        net,
        greedy=False,
        generator=torch.Generator().manual_seed(0),
    )

    assert all(value.item() < 0.0 for value in log_p)
    assert all(value.item() > 0.0 for value in entropy)

    log_p[0].backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_clone_environment_is_independent():
    env = random_environment(3, n=5)
    copy = clone_environment(env)

    env.step(0)

    assert copy.served_count == 0
    assert copy.orders == env.orders


def test_policy_gradient_improves_the_greedy_policy():
    # Serving every order is clearly better than stopping.
    torch.manual_seed(0)
    net = network()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    encoder = ShiftEncoder()
    instances = [profitable_environment(seed) for seed in range(8)]
    generator = torch.Generator().manual_seed(1)

    def mean_greedy():
        _, _, returns = run_batch(
            [clone_environment(e) for e in instances],
            encoder,
            net,
            greedy=True,
        )

        return float(np.mean(returns))

    before = mean_greedy()

    for _ in range(150):
        policy_gradient_step(
            net,
            optimizer,
            encoder,
            instances,
            samples=6,
            entropy_weight=0.0,
            clip=5.0,
            generator=generator,
        )

    assert mean_greedy() > before


def test_behaviour_cloning_learns_the_demonstrated_choice():
    rng = np.random.default_rng(0)
    demos = []

    for _ in range(300):
        features = rng.uniform(-1, 1, (4, FEATURE_DIMENSION)).astype(np.float32)
        demos.append((features, int(np.argmax(features[:, 1]))))

    net = network()
    loss = behaviour_clone(net, demos, epochs=60, learning_rate=1e-2)

    assert loss < 0.5

    hits = 0

    for features, label in demos[:100]:
        with torch.no_grad():
            hits += int(torch.argmax(net(torch.from_numpy(features)))) == label

    assert hits >= 85


def test_set_scorer_attends_to_near_candidates():
    from rl.shift.set_network import SetScorer

    torch.manual_seed(0)
    net = SetScorer(FEATURE_DIMENSION, 16)

    rows = torch.randn(2, 5, FEATURE_DIMENSION)
    valid = torch.tensor(
        [[True] * 5, [True, True, True, False, False]]
    )
    pairs = torch.rand(2, 5, 5)

    scores = net(rows, valid, pairs)

    assert scores.shape == (2, 5)
    assert torch.isfinite(scores[valid]).all()

    # Padding must not change the scores of the real rows.
    rows_alt = rows.clone()
    rows_alt[1, 3:] = 99.0
    pairs_alt = pairs.clone()
    pairs_alt[1, 3:, :] = 7.0
    pairs_alt[1, :, 3:] = 7.0

    again = net(rows_alt, valid, pairs_alt)

    assert torch.allclose(scores[1, :3], again[1, :3], atol=1e-5)


def test_rollout_and_cloning_work_with_the_set_scorer():
    from rl.shift.set_network import SetScorer

    torch.manual_seed(0)
    net = SetScorer(FEATURE_DIMENSION, 16)
    encoder = ShiftEncoder()

    envs = [profitable_environment(seed, n=5) for seed in range(3)]
    log_p, _, returns = run_batch(
        envs,
        encoder,
        net,
        greedy=False,
        generator=torch.Generator().manual_seed(0),
    )

    assert all(env.done for env in envs)
    assert all(np.isfinite(value) for value in returns)

    log_p[0].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())

    # Cloning a rule that needs the pairwise distances.
    rng = np.random.default_rng(0)
    demos = []

    for _ in range(200):
        features = rng.uniform(-1, 1, (4, FEATURE_DIMENSION)).astype(np.float32)
        pairs = rng.uniform(0, 1, (4, 4)).astype(np.float32)

        # Prefer the candidate with the smallest total distance to the rest.
        demos.append((features, int(np.argmin(pairs.sum(axis=1))), pairs))

    net = SetScorer(FEATURE_DIMENSION, 32)
    loss = behaviour_clone(net, demos, epochs=80, learning_rate=5e-3)

    assert loss < 1.0
