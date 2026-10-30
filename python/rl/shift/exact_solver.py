from __future__ import annotations

from .environment import ShiftEnvironment
from .objective import order_terms, terminal_terms


MAX_EXACT_ORDERS = 14


def exact_best_shift(
    env: ShiftEnvironment,
) -> tuple[float, tuple[int, ...]]:
    """
    Highest shift score over every choice of which orders to serve and in
    what order (the rest are deferred), by dynamic programming over
    (set of served orders, last order).

    Lateness and the shift-end terms depend on the clock, so each state keeps
    every (time, score) label that no other label beats on both. This is
    exact because the score still to come never rises when a route starts
    later.
    """

    n = env.number_of_orders

    if n > MAX_EXACT_ORDERS:
        raise ValueError(
            f"Exact search supports at most {MAX_EXACT_ORDERS} orders, got {n}."
        )

    cfg = env.config
    dist = env.distance_km
    time = env.travel_minutes
    orders = env.orders

    def serve(prev_node: int, k: int, clock: float, score: float):
        node = k + 1
        travel = time[prev_node][node]
        arrival = clock + travel

        commission, move, late, margin = order_terms(
            cfg,
            value=orders[k].value,
            deadline=orders[k].deadline_minutes,
            arrival=arrival,
            travel_minutes=travel,
            distance_km=dist[prev_node][node],
        )

        return (
            arrival + cfg.service_minutes,
            score + commission - move - late - margin,
        )

    def finish(mask: int, node: int, clock: float, score: float) -> float:
        early, back, overtime, deferred = terminal_terms(
            cfg,
            finish_minutes=clock,
            return_minutes=time[node][0],
            return_km=dist[node][0],
            deferred=n - mask.bit_count(),
        )

        return score + early - back - overtime - deferred

    # Serving nothing is allowed: every order is deferred.
    best_total = finish(0, 0, 0.0, 0.0)
    best_end = None

    # labels[(mask, last)] -> [(clock, score, previous key, previous index)]
    labels: dict[tuple[int, int], list] = {}

    for k in range(n):
        clock, score = serve(0, k, 0.0, 0.0)
        labels[(1 << k, k)] = [(clock, score, None, None)]

    full = (1 << n) - 1

    for mask in range(1, full + 1):
        for last in range(n):
            key = (mask, last)

            if key not in labels:
                continue

            kept = []
            best_score = float("-inf")

            for label in sorted(labels[key], key=lambda x: (x[0], -x[1])):
                if label[1] > best_score + 1e-9:
                    kept.append(label)
                    best_score = label[1]

            labels[key] = kept

            for index, (clock, score, _, _) in enumerate(kept):
                total = finish(mask, last + 1, clock, score)

                if total > best_total:
                    best_total = total
                    best_end = (key, index)

            if mask == full:
                continue

            for nxt in range(n):
                bit = 1 << nxt

                if mask & bit:
                    continue

                target = labels.setdefault((mask | bit, nxt), [])

                for index, (clock, score, _, _) in enumerate(kept):
                    next_clock, next_score = serve(last + 1, nxt, clock, score)
                    target.append((next_clock, next_score, key, index))

    sequence: list[int] = []

    if best_end is not None:
        key, index = best_end

        while key is not None:
            sequence.append(key[1])
            _, _, key, index = labels[key][index]

        sequence.reverse()

    return best_total, tuple(sequence)
