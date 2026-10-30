from __future__ import annotations

import itertools


def path_cost(
    cost: list[list[float]],
    order: list[int] | tuple[int, ...],
) -> float:
    """
    Cost of visiting `order` (indices 1..n) starting from node 0, with no
    return to node 0.
    """

    total = 0.0
    previous = 0

    for k in order:
        total += cost[previous][k + 1]
        previous = k + 1

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
) -> list[int]:
    """
    2-opt (reversal) and relocate moves until no move shortens the path.
    """

    best = list(order)
    best_cost = path_cost(cost, best)
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
            value = path_cost(cost, move)

            if value < best_cost - 1e-12:
                best, best_cost = move, value
                improved = True
                break

    return best


def optimal_path(
    cost: list[list[float]],
) -> tuple[float, list[int]]:
    """
    Cheapest open path from node 0 through every other node (Held-Karp).
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
    last = min(range(n), key=lambda k: best[(full, k)][0])
    total = best[(full, last)][0]

    order: list[int] = []
    mask = full

    while last is not None:
        order.append(last)
        previous = best[(mask, last)][1]
        mask ^= 1 << last
        last = previous

    order.reverse()

    return total, order


def brute_force_path(cost: list[list[float]]) -> float:
    n = len(cost) - 1

    return min(
        path_cost(cost, order) for order in itertools.permutations(range(n))
    )
