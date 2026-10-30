import random

import pytest

from routing.open_path import (
    brute_force_path,
    improve_path,
    nearest_path,
    optimal_path,
    path_cost,
)


def random_cost(seed, n):
    rng = random.Random(seed)

    return [
        [0.0 if i == j else rng.uniform(1.0, 30.0) for j in range(n + 1)]
        for i in range(n + 1)
    ]


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_optimal_path_matches_brute_force(seed, closed):
    cost = random_cost(seed, 6)

    total, order = optimal_path(cost, closed)

    assert total == pytest.approx(brute_force_path(cost, closed))
    assert path_cost(cost, order, closed) == pytest.approx(total)
    assert sorted(order) == list(range(6))


@pytest.mark.parametrize("closed", [False, True])
def test_improvement_never_worsens_and_stays_a_permutation(closed):
    for seed in range(5):
        cost = random_cost(seed, 8)
        start = nearest_path(cost)

        better = improve_path(cost, start, closed)

        assert path_cost(cost, better, closed) <= (
            path_cost(cost, start, closed) + 1e-9
        )
        assert sorted(better) == list(range(8))


def test_improvement_follows_a_custom_objective():
    cost = random_cost(3, 6)

    # Prefer visiting stop 5 first, whatever the distance.
    def objective(order):
        return path_cost(cost, order) + (1_000.0 if order[0] != 5 else 0.0)

    better = improve_path(cost, nearest_path(cost), objective=objective)

    assert better[0] == 5
