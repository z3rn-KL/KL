from itertools import permutations
import random

import pytest

from rl.shift.environment import ShiftEnvironment, ShiftOrder, play
from rl.shift.exact_solver import exact_best_shift
from rl.shift.objective import ShiftObjectiveConfig


def random_environment(
    seed: int, n: int = 5, deadline_scale: float = 1.0, config=None
):
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
        distance_km=dist, travel_minutes=time, orders=orders, config=config
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


WAITING = ShiftObjectiveConfig(wait_cost_per_minute=50.0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_matches_brute_force_with_waiting_cost(seed):
    env = random_environment(seed, n=5, config=WAITING)

    best = max(play(env, plan) for plan in all_plans(5))

    score, plan = exact_best_shift(env)

    assert score == pytest.approx(best, abs=1e-6)
    assert play(env, plan) == pytest.approx(score, abs=1e-6)


def test_score_sequence_matches_play_with_waiting_cost():
    from rl.shift.environment import score_sequence

    env = random_environment(12, n=6, config=WAITING)
    rng = random.Random(1)

    for _ in range(20):
        plan = rng.sample(range(6), rng.randrange(0, 7))

        assert score_sequence(env, plan) == pytest.approx(
            play(env, plan), abs=1e-9
        )


def test_waiting_cost_lowers_the_score_by_arrival_minutes():
    plain = random_environment(5, n=4)
    waiting = random_environment(5, n=4, config=WAITING)

    plan = [2, 0, 3, 1]

    play(plain, plan)
    play(waiting, plan)

    summary = waiting.summary()

    assert summary["mean_minutes_from_start"] == pytest.approx(
        sum(waiting.arrival_minutes.values()) / 4
    )
    assert summary["wait_cost"] == pytest.approx(
        50.0 * sum(waiting.arrival_minutes.values())
    )
    assert waiting.total_reward == pytest.approx(
        plain.total_reward - summary["wait_cost"]
    )


def test_waiting_cost_prefers_the_short_order_first():
    # Order 0 takes 10 minutes to reach, order 1 takes 40; a pure distance
    # objective does not care about the order, waiting does.
    time = [[0, 10, 40], [10, 0, 30], [40, 30, 0]]
    orders = [ShiftOrder("a", 50_000.0, None), ShiftOrder("b", 50_000.0, None)]

    def run(config, plan):
        env = ShiftEnvironment(
            distance_km=time, travel_minutes=time, orders=orders, config=config
        )

        return play(env, plan)

    short_first = run(WAITING, [0, 1]) - run(WAITING, [1, 0])
    base = run(None, [0, 1]) - run(None, [1, 0])

    assert short_first > base


def test_order_statuses_are_generated_from_the_run():
    time = [[0, 10, 30, 50], [10, 0, 5, 5], [30, 5, 0, 5], [50, 5, 5, 0]]
    orders = [
        ShiftOrder("early", 50_000.0, 100.0),
        ShiftOrder("late", 50_000.0, 15.0),
        ShiftOrder("left", 50_000.0, None),
    ]

    env = ShiftEnvironment(distance_km=time, travel_minutes=time, orders=orders)
    env.reset()
    env.step(0)  # arrives at 10, due 100: on time
    env.step(1)  # arrives at 25, due 15: late
    env.step(env.stop_action)  # order 2 is deferred

    rows = {r["order_id"]: r for r in env.order_statuses()}

    assert rows["early"]["status"] == "on_time"
    assert rows["early"]["minutes_from_start"] == pytest.approx(10.0)
    assert rows["late"]["status"] == "late"
    assert rows["left"]["status"] == "deferred"
    assert rows["left"]["minutes_from_start"] is None
