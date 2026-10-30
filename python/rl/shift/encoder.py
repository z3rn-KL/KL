from __future__ import annotations

import numpy as np

from .environment import ShiftEnvironment
from .objective import order_terms, terminal_terms


FEATURE_NAMES = (
    "is_stop",
    "clock_ratio",
    "remaining_ratio",
    "size_ratio",
    "travel_ratio",
    "km_ratio",
    "value_ratio",
    "deadline_valid",
    "slack_ratio",
    "late_flag",
    "immediate_gain",
    "return_ratio",
    "nearest_remaining_ratio",
    "mean_remaining_ratio",
    "earlier_deadline_share",
    "projected_end_ratio",
)

FEATURE_DIMENSION = len(FEATURE_NAMES)

TIME_REF_MINUTES = 60.0
KM_REF = 10.0
VALUE_REF = 40_000.0
SIZE_REF = 50.0
CLIP = 5.0

# Rewards are in VND-equivalent points; scale them near 1 for the network.
REWARD_SCALE = 1.0 / 50_000.0


class ShiftEncoder:
    """
    One feature row per available action: every unserved order, plus a final
    row for stopping the shift (the remaining orders are deferred).

    The scoring network sees only these rows, so it works for shifts of any
    size. immediate_gain is the step reward of serving the order compared
    with deferring it (stop row: the reward of ending the shift now), both
    divided by the defer penalty.
    """

    feature_names = FEATURE_NAMES
    feature_dimension = FEATURE_DIMENSION

    def encode(
        self,
        env: ShiftEnvironment,
    ) -> tuple[np.ndarray, np.ndarray]:
        actions, rows, _ = self.encode_with_base(env)

        return actions, rows

    def pair_times(
        self,
        env: ShiftEnvironment,
        actions: np.ndarray,
    ) -> np.ndarray:
        """
        Travel time between every two candidate rows, in units of
        TIME_REF_MINUTES; the stop row has zeros. Lets a network see how the
        remaining orders lie relative to one another.
        """

        orders = actions[:-1]
        size = len(actions)

        pairs = np.zeros((size, size), np.float32)
        pairs[:-1, :-1] = (
            env.travel_np[np.ix_(orders + 1, orders + 1)] / TIME_REF_MINUTES
        )

        return pairs

    def encode_with_base(
        self,
        env: ShiftEnvironment,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Like encode, plus the exact immediate reward (scaled) of each action:
        the step reward of serving an order, or the reward of ending the
        shift for the stop row.
        """

        cfg = env.config
        n = env.number_of_orders

        served = np.fromiter(
            ((env.served_mask >> k) & 1 for k in range(n)),
            dtype=bool,
            count=n,
        )
        avail = np.flatnonzero(~served)
        m = len(avail)

        node = env.current_node
        clock = env.elapsed_minutes
        travel_all = env.travel_np
        distance_all = env.distance_np

        travel = travel_all[node, avail + 1]
        distance = distance_all[node, avail + 1]
        arrival = clock + travel

        deadline = env.deadline_np[avail]
        has_deadline = np.isfinite(deadline)

        slack = np.where(
            has_deadline,
            (deadline - arrival) / cfg.early_slack_minutes,
            0.0,
        )
        late = has_deadline & (arrival > deadline)

        gain = np.empty(m)
        immediate = np.empty(m + 1)

        for i, k in enumerate(avail):
            commission, move, late_penalty, margin = order_terms(
                cfg,
                value=env.orders[k].value,
                deadline=env.orders[k].deadline_minutes,
                arrival=arrival[i],
                travel_minutes=travel[i],
                distance_km=distance[i],
            )

            reward = commission - move - late_penalty - margin

            immediate[i] = reward
            gain[i] = (reward + cfg.defer_penalty) / cfg.defer_penalty

        back = travel_all[avail + 1, 0]

        if m > 1:
            block = travel_all[np.ix_(avail + 1, avail + 1)]
            masked = block + np.diag(np.full(m, np.inf))

            nearest = masked.min(axis=1)
            mean_rest = block.sum(axis=1) / (m - 1)

            earlier = (
                (deadline[None, :] < deadline[:, None]) & has_deadline[None, :]
            ).sum(axis=1) / (m - 1)
            earlier = np.where(has_deadline, earlier, 0.0)
        else:
            nearest = np.zeros(m)
            mean_rest = np.zeros(m)
            earlier = np.zeros(m)

        projected = (
            arrival
            + cfg.service_minutes
            + (m - 1) * (mean_rest + cfg.service_minutes)
            + back
        )

        rows = np.zeros((m + 1, FEATURE_DIMENSION))

        rows[:m, 1] = clock / cfg.shift_minutes
        rows[:m, 2] = m / n
        rows[:m, 3] = n / SIZE_REF
        rows[:m, 4] = travel / TIME_REF_MINUTES
        rows[:m, 5] = distance / KM_REF
        rows[:m, 6] = env.value_np[avail] / VALUE_REF
        rows[:m, 7] = has_deadline
        rows[:m, 8] = slack
        rows[:m, 9] = late
        rows[:m, 10] = gain
        rows[:m, 11] = back / TIME_REF_MINUTES
        rows[:m, 12] = nearest / TIME_REF_MINUTES
        rows[:m, 13] = mean_rest / TIME_REF_MINUTES
        rows[:m, 14] = earlier
        rows[:m, 15] = projected / cfg.shift_minutes

        # Stop row.
        early, back_cost, overtime, deferred = terminal_terms(
            cfg,
            finish_minutes=clock,
            return_minutes=travel_all[node, 0],
            return_km=distance_all[node, 0],
            deferred=m,
        )

        from_here = travel_all[node, avail + 1]
        stop = m
        rows[stop, 0] = 1.0
        rows[stop, 1] = clock / cfg.shift_minutes
        rows[stop, 2] = m / n
        rows[stop, 3] = n / SIZE_REF
        stop_reward = early - back_cost - overtime - deferred
        immediate[stop] = stop_reward

        rows[stop, 10] = stop_reward / cfg.defer_penalty
        rows[stop, 11] = travel_all[node, 0] / TIME_REF_MINUTES
        rows[stop, 12] = from_here.min() / TIME_REF_MINUTES
        rows[stop, 13] = from_here.mean() / TIME_REF_MINUTES
        rows[stop, 15] = (
            clock
            + m * (from_here.mean() + cfg.service_minutes)
            + travel_all[node, 0]
        ) / cfg.shift_minutes

        np.clip(rows, -CLIP, CLIP, out=rows)

        actions = np.append(avail, env.stop_action)

        return (
            actions,
            rows.astype(np.float32),
            (immediate * REWARD_SCALE).astype(np.float32),
        )
