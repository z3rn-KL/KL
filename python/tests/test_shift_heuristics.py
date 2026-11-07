import random

import pytest

from rl.shift.environment import score_sequence
from rl.shift.heuristics import (
    local_search,
    multistart_search,
    nearest_plan,
)
from tests.test_shift_environment import random_environment


def test_local_search_never_worsens_and_returns_a_valid_plan():
    for seed in range(5):
        env = random_environment(seed, n=7, deadline_scale=0.5)
        start = nearest_plan(env, stop_when_not_worth=False)

        improved = local_search(env, start)

        assert score_sequence(env, improved) >= score_sequence(env, start) - 1e-9
        assert len(set(improved)) == len(improved)
        assert set(improved) <= set(range(env.number_of_orders))


def test_stop_rule_can_defer_hopeless_orders():
    env = random_environment(2, n=6, deadline_scale=0.05)

    served_all = nearest_plan(env, stop_when_not_worth=False)
    plan = nearest_plan(env, stop_when_not_worth=True)

    assert len(served_all) == 6
    assert len(plan) < 6


def test_multistart_is_at_least_as_good_as_nearest_neighbour():
    env = random_environment(4, n=7, deadline_scale=0.5)
    best = multistart_search(env, random.Random(0), starts=3)

    nn = nearest_plan(env, stop_when_not_worth=True)

    assert score_sequence(env, best) >= score_sequence(env, nn) - 1e-9
