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


@pytest.mark.parametrize("seed", range(5))
def test_optimal_path_matches_brute_force(seed):
    cost = random_cost(seed, 6)

    total, order = optimal_path(cost)

    assert total == pytest.approx(brute_force_path(cost))
    assert path_cost(cost, order) == pytest.approx(total)
    assert sorted(order) == list(range(6))


def test_improvement_never_worsens_and_stays_a_permutation():
    for seed in range(5):
        cost = random_cost(seed, 8)
        start = nearest_path(cost)

        better = improve_path(cost, start)

        assert path_cost(cost, better) <= path_cost(cost, start) + 1e-9
        assert sorted(better) == list(range(8))
