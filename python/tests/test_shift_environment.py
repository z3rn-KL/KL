from itertools import permutations
import random

import pytest

from rl.shift.environment import ShiftEnvironment, ShiftOrder, play
from rl.shift.exact_solver import exact_best_shift


def random_environment(seed: int, n: int = 5, deadline_scale: float = 1.0):
    rng = random.Random(seed)

    time = [
        [0.0 if i == j else rng.uniform(3.0, 25.0) for j in range(n + 1)]
        for i in range(n + 1)
    ]
    dist = [[v / 2.0 for v in row] for row in time]

    orders = [
        ShiftOrder(
            order_id=f"o{k}",
            value=rng.uniform(15_000.0, 60_000.0),
            deadline_minutes=(
                None
                if rng.random() < 0.2
                else rng.uniform(20.0, 120.0) * deadline_scale
            ),
        )
        for k in range(n)
    ]

    return ShiftEnvironment(
        distance_km=dist, travel_minutes=time, orders=orders
    )


def all_plans(n):
    for size in range(n + 1):
        for subset in permutations(range(n), size):
            yield subset


def test_step_rewards_sum_to_the_score():
    env = random_environment(7)
    env.reset()

    total = 0.0
    for action in (2, 0, 4):
        total += env.step(action).reward
    total += env.step(env.stop_action).reward

    assert env.done
    assert total == pytest.approx(env.total_reward)


def test_serving_everything_ends_the_shift_automatically():
    env = random_environment(3)
    env.reset()

    last = None
    for action in range(env.number_of_orders):
        last = env.step(action)

    assert last.done
    assert env.summary()["deferred"] == 0
    assert env.available_actions() == ()


def test_stopping_defers_the_rest():
    env = random_environment(5)
    env.reset()
    env.step(1)
    env.step(env.stop_action)

    assert env.summary()["served"] == 1
    assert env.summary()["deferred"] == env.number_of_orders - 1


def test_invalid_actions_are_rejected():
    env = random_environment(1)
    env.reset()
    env.step(0)

    with pytest.raises(ValueError):
        env.step(0)

    with pytest.raises(ValueError):
        env.step(99)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("deadline_scale", [0.3, 1.0])
def test_exact_matches_brute_force_over_all_plans(seed, deadline_scale):
    env = random_environment(seed, n=5, deadline_scale=deadline_scale)

    best = max(play(env, plan) for plan in all_plans(5))

    score, plan = exact_best_shift(env)

    assert score == pytest.approx(best, abs=1e-6)
    assert play(env, plan) == pytest.approx(score, abs=1e-6)
    assert len(set(plan)) == len(plan)


def test_exact_rejects_large_shifts():
    with pytest.raises(ValueError):
        exact_best_shift(random_environment(0, n=15))


def test_score_sequence_matches_play():
    from rl.shift.environment import score_sequence

    env = random_environment(11, n=6)
    rng = random.Random(0)

    for _ in range(25):
        size = rng.randrange(0, 7)
        plan = rng.sample(range(6), size)

        assert score_sequence(env, plan) == pytest.approx(
            play(env, plan), abs=1e-9
        )
