from __future__ import annotations

from .environment import ProposedRoutingEnvironment


MAX_EXACT_DELIVERIES = 16


def route_cost(
    env: ProposedRoutingEnvironment,
    order: tuple[int, ...] | list[int],
) -> float:
    """
    J of a complete route, measured by the environment itself.
    """

    env.reset()

    for action in order:
        env.step(int(action))

    return env.final_objective().total_cost


def exact_optimal_route(
    env: ProposedRoutingEnvironment,
) -> tuple[float, tuple[int, ...]]:
    """
    Minimum J over every delivery order, by dynamic programming over
    (set of delivered, last delivery).

    Lateness depends on the arrival time, so each state keeps every
    (time, cost) label that no other label beats on both. This is exact
    because future cost never decreases when a route starts later.
    """

    n = env.number_of_deliveries

    if n > MAX_EXACT_DELIVERIES:
        raise ValueError(
            f"Exact search supports at most "
            f"{MAX_EXACT_DELIVERIES} deliveries, got {n}."
        )

    cfg = env.objective_config
    ctx = env.objective_context

    legs = ctx.route_leg_count
    valid = ctx.valid_deadline_denominator
    switches = ctx.possible_cluster_switches

    dist = env.distance_matrix_km
    time = env.travel_time_matrix_minutes
    deadlines = env.deadlines_minutes
    clusters = env.cluster_ids

    def leg_cost(i: int, j: int) -> float:
        cost = (
            cfg.distance_weight * dist[i][j] / (legs * cfg.distance_ref_km)
            + cfg.travel_time_weight
            * time[i][j]
            / (legs * cfg.travel_time_ref_minutes)
        )

        if i != 0 and j != 0 and clusters[i - 1] != clusters[j - 1]:
            cost += cfg.cluster_switch_weight / switches

        return cost

    def lateness_cost(node: int, arrival: float) -> float:
        deadline = deadlines[node - 1]

        if deadline is None or arrival <= deadline:
            return 0.0

        return (
            cfg.late_delivery_weight / valid
            + cfg.lateness_weight
            * (arrival - deadline)
            / (valid * cfg.lateness_ref_minutes)
        )

    # labels[(mask, last)] -> list of (time, cost, previous key, previous index)
    labels: dict[tuple[int, int], list] = {}

    for first in range(1, n + 1):
        arrival = time[0][first]

        labels[(1 << (first - 1), first)] = [
            (
                arrival,
                leg_cost(0, first) + lateness_cost(first, arrival),
                None,
                None,
            )
        ]

    full = (1 << n) - 1

    for mask in range(1, full + 1):
        for last in range(1, n + 1):
            key = (mask, last)

            if key not in labels:
                continue

            # Keep only labels that no other label beats on time and cost.
            kept: list = []
            best_cost = float("inf")

            for label in sorted(labels[key], key=lambda x: (x[0], x[1])):
                if label[1] < best_cost - 1e-12:
                    kept.append(label)
                    best_cost = label[1]

            labels[key] = kept

            if mask == full:
                continue

            for nxt in range(1, n + 1):
                bit = 1 << (nxt - 1)

                if mask & bit:
                    continue

                step = leg_cost(last, nxt)
                target = labels.setdefault((mask | bit, nxt), [])

                for index, (t, cost, _, _) in enumerate(kept):
                    arrival = t + time[last][nxt]

                    target.append(
                        (
                            arrival,
                            cost + step + lateness_cost(nxt, arrival),
                            key,
                            index,
                        )
                    )

    best_total = float("inf")
    best_end: tuple[tuple[int, int], int] | None = None

    for last in range(1, n + 1):
        key = (full, last)

        for index, (_, cost, _, _) in enumerate(labels[key]):
            total = cost + leg_cost(last, 0)

            if total < best_total:
                best_total = total
                best_end = (key, index)

    order: list[int] = []
    key, index = best_end

    while key is not None:
        order.append(key[1] - 1)
        _, _, key, index = labels[key][index]

    order.reverse()

    return best_total, tuple(order)
