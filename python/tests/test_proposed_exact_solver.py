from itertools import permutations
import random

import pytest

from rl.proposed.environment import ProposedRoutingEnvironment
from rl.proposed.exact_solver import (
    exact_optimal_route,
    route_cost,
)


def random_environment(seed: int, n: int = 6):
    rng = random.Random(seed)

    nodes = n + 1

    time = [
        [0.0 if i == j else rng.uniform(2.0, 20.0) for j in range(nodes)]
        for i in range(nodes)
    ]
    dist = [[value / 2.0 for value in row] for row in time]

    # Tight deadlines so lateness really changes the best order.
    deadlines = [
        None if rng.random() < 0.25 else rng.uniform(8.0, 50.0)
        for _ in range(n)
    ]

    return ProposedRoutingEnvironment(
        distance_matrix_km=dist,
        travel_time_matrix_minutes=time,
        deadlines_minutes=deadlines,
        cluster_ids=[rng.randrange(3) for _ in range(n)],
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_matches_brute_force(seed):
    env = random_environment(seed)

    best = min(
        route_cost(env, order)
        for order in permutations(range(env.number_of_deliveries))
    )

    cost, order = exact_optimal_route(env)

    assert cost == pytest.approx(best, abs=1e-9)
    assert route_cost(env, order) == pytest.approx(cost, abs=1e-9)
    assert sorted(order) == list(range(env.number_of_deliveries))


def test_exact_rejects_large_instances():
    with pytest.raises(ValueError):
        exact_optimal_route(random_environment(0, n=17))
