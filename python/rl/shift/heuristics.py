from __future__ import annotations

import random

from .environment import ShiftEnvironment, score_sequence
from .objective import order_terms


def nearest_plan(
    env: ShiftEnvironment,
    *,
    stop_when_not_worth: bool,
) -> list[int]:
    """
    Nearest neighbour by travel time, under the shift score.

    With stop_when_not_worth, the shift ends as soon as serving the nearest
    order is worth less than deferring it (immediate gain below zero).
    """

    cfg = env.config
    plan: list[int] = []
    node = 0
    clock = 0.0

    remaining = set(range(env.number_of_orders))

    while remaining:
        k = min(remaining, key=lambda a: env.travel_minutes[node][a + 1])
        travel = env.travel_minutes[node][k + 1]

        if stop_when_not_worth:
            commission, move, late, margin = order_terms(
                cfg,
                value=env.orders[k].value,
                deadline=env.orders[k].deadline_minutes,
                arrival=clock + travel,
                travel_minutes=travel,
                distance_km=env.distance_km[node][k + 1],
            )

            if commission - move - late - margin + cfg.defer_penalty < 0.0:
                break

        plan.append(k)
        remaining.remove(k)
        clock += travel + cfg.service_minutes
        node = k + 1

    return plan


def local_search(
    env: ShiftEnvironment,
    plan: list[int],
) -> list[int]:
    """
    First-improvement search with reversal, relocate, drop and insert moves.
    """

    best = list(plan)
    best_score = score_sequence(env, best)
    all_orders = set(range(env.number_of_orders))

    improved = True

    while improved:
        improved = False
        n = len(best)
        missing = sorted(all_orders - set(best))

        def candidates():
            for i in range(n - 1):
                for j in range(i + 1, n):
                    yield best[:i] + best[i : j + 1][::-1] + best[j + 1 :]

            for i in range(n):
                rest = best[:i] + best[i + 1 :]
                yield rest

                for j in range(n):
                    if j != i:
                        yield rest[:j] + [best[i]] + rest[j:]

            for k in missing:
                for j in range(n + 1):
                    yield best[:j] + [k] + best[j:]

        for candidate in candidates():
            score = score_sequence(env, candidate)

            if score > best_score + 1e-9:
                best, best_score = candidate, score
                improved = True
                break

    return best


def multistart_search(
    env: ShiftEnvironment,
    rng: random.Random,
    starts: int,
) -> list[int]:
    """
    Best plan found by local search from the nearest-neighbour plans and
    from `starts` random plans. A strong reference where exact search is
    out of reach, not a proof of optimality.
    """

    n = env.number_of_orders

    seeds = [
        nearest_plan(env, stop_when_not_worth=True),
        nearest_plan(env, stop_when_not_worth=False),
        *(rng.sample(range(n), n) for _ in range(starts)),
    ]

    plans = [local_search(env, seed) for seed in seeds]

    return max(plans, key=lambda plan: score_sequence(env, plan))
