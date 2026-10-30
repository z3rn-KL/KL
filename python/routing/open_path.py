from __future__ import annotations

import itertools
from typing import Callable


def path_cost(
    cost: list[list[float]],
    order: list[int] | tuple[int, ...],
    closed: bool = False,
) -> float:
    """
    Cost of visiting `order` (indices 0..n-1, node k + 1) starting from
    node 0. With closed=True the route also returns to node 0.
    """

    total = 0.0
    previous = 0

    for k in order:
        total += cost[previous][k + 1]
        previous = k + 1

    if closed:
        total += cost[previous][0]

    return total


def nearest_path(cost: list[list[float]]) -> list[int]:
    n = len(cost) - 1
    remaining = set(range(n))
    order: list[int] = []
    previous = 0

    while remaining:
        k = min(remaining, key=lambda a: cost[previous][a + 1])
        order.append(k)
        remaining.remove(k)
        previous = k + 1

    return order


def improve_path(
    cost: list[list[float]],
    order: list[int],
    closed: bool = False,
    objective: Callable[[list[int]], float] | None = None,
) -> list[int]:
    """
    2-opt (reversal) and relocate moves until no move lowers the objective,
    which is the path cost unless `objective` is given.
    """

    def evaluate(candidate: list[int]) -> float:
        if objective is not None:
            return objective(candidate)

        return path_cost(cost, candidate, closed)

    best = list(order)
    best_cost = evaluate(best)
    n = len(best)

    improved = True

    while improved:
        improved = False

        moves = []

        for i in range(n - 1):
            for j in range(i + 1, n):
                moves.append(best[:i] + best[i : j + 1][::-1] + best[j + 1 :])

        for i in range(n):
            rest = best[:i] + best[i + 1 :]

            for j in range(n):
                if j != i:
                    moves.append(rest[:j] + [best[i]] + rest[j:])

        for move in moves:
            value = evaluate(move)

            if value < best_cost - 1e-12:
                best, best_cost = move, value
                improved = True
                break

    return best


def optimal_path(
    cost: list[list[float]],
    closed: bool = False,
) -> tuple[float, list[int]]:
    """
    Cheapest path from node 0 through every other node (Held-Karp), open or
    (closed=True) returning to node 0.
    """

    n = len(cost) - 1

    if n > 16:
        raise ValueError(f"Exact search supports at most 16 stops, got {n}.")

    best = {(1 << k, k): (cost[0][k + 1], None) for k in range(n)}

    for mask in range(1, 1 << n):
        for last in range(n):
            if (mask, last) not in best:
                continue

            base = best[(mask, last)][0]

            for nxt in range(n):
                bit = 1 << nxt

                if mask & bit:
                    continue

                value = base + cost[last + 1][nxt + 1]
                key = (mask | bit, nxt)

                if key not in best or value < best[key][0]:
                    best[key] = (value, last)

    full = (1 << n) - 1
    def finish(k: int) -> float:
        return best[(full, k)][0] + (cost[k + 1][0] if closed else 0.0)

    last = min(range(n), key=finish)
    total = finish(last)

    order: list[int] = []
    mask = full

    while last is not None:
        order.append(last)
        previous = best[(mask, last)][1]
        mask ^= 1 << last
        last = previous

    order.reverse()

    return total, order


def brute_force_path(cost: list[list[float]], closed: bool = False) -> float:
    n = len(cost) - 1

    return min(
        path_cost(cost, order, closed)
        for order in itertools.permutations(range(n))
    )
