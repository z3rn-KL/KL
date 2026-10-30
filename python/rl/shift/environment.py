from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .objective import (
    ShiftObjectiveConfig,
    order_terms,
    terminal_terms,
)


@dataclass(frozen=True)
class ShiftOrder:
    order_id: str
    value: float
    deadline_minutes: float | None


@dataclass(frozen=True)
class ShiftStep:
    reward: float
    done: bool


class ShiftEnvironment:
    """
    One shipper shift. All orders are known and released at t = 0.

    Node 0 is the depot and node k + 1 is order k. Actions:

        0 .. n-1   serve that order next (service time included)
        n          stop the shift; every order not yet served is deferred

    The shift also ends by itself once every order is served, and the
    shipper then drives back to the depot. Vehicle capacity is not modelled.

    The episode return equals the shift score of ShiftObjectiveConfig.
    """

    def __init__(
        self,
        *,
        distance_km: Sequence[Sequence[float]],
        travel_minutes: Sequence[Sequence[float]],
        orders: Sequence[ShiftOrder],
        config: ShiftObjectiveConfig | None = None,
    ) -> None:
        self.distance_km = [list(row) for row in distance_km]
        self.travel_minutes = [list(row) for row in travel_minutes]
        self.orders = tuple(orders)
        self.config = config if config is not None else ShiftObjectiveConfig()

        self.number_of_orders = len(self.orders)

        self._validate()

        self.stop_action = self.number_of_orders

        self.reset()

    def _validate(self) -> None:
        size = self.number_of_orders + 1

        if self.number_of_orders < 1:
            raise ValueError("A shift needs at least one order.")

        for name, matrix in (
            ("distance_km", self.distance_km),
            ("travel_minutes", self.travel_minutes),
        ):
            if len(matrix) != size or any(len(row) != size for row in matrix):
                raise ValueError(f"{name} must be {size} x {size}.")

            if any(
                not math.isfinite(v) or v < 0.0
                for row in matrix
                for v in row
            ):
                raise ValueError(f"{name} must be finite and non-negative.")

        for order in self.orders:
            if order.value <= 0.0:
                raise ValueError("Order value must be > 0.")

            deadline = order.deadline_minutes

            if deadline is not None and (
                not math.isfinite(deadline) or deadline < 0.0
            ):
                raise ValueError("Deadline must be non-negative or None.")

    def reset(self) -> None:
        self.current_node = 0
        self.elapsed_minutes = 0.0
        self.served_mask = 0
        self.served_sequence: list[int] = []
        self.done = False

        self.total_reward = 0.0
        self.distance_total_km = 0.0
        self.on_time = 0
        self.late = 0
        self.revenue = 0.0
        self.cost = 0.0
        self.end_minutes = 0.0
        self.overtime_minutes = 0.0

    @property
    def served_count(self) -> int:
        return self.served_mask.bit_count()

    @property
    def deferred_count(self) -> int:
        return self.number_of_orders - self.served_count

    def available_actions(self) -> tuple[int, ...]:
        if self.done:
            return ()

        return (
            *(
                k
                for k in range(self.number_of_orders)
                if not self.served_mask >> k & 1
            ),
            self.stop_action,
        )

    def step(self, action: int) -> ShiftStep:
        if self.done:
            raise RuntimeError("Shift is already finished.")

        if action == self.stop_action:
            return self._finish(0.0)

        if not 0 <= action < self.number_of_orders:
            raise ValueError(f"Invalid action {action}.")

        if self.served_mask >> action & 1:
            raise ValueError(f"Order {action} was already served.")

        node = action + 1
        order = self.orders[action]
        cfg = self.config

        travel = self.travel_minutes[self.current_node][node]
        distance = self.distance_km[self.current_node][node]
        arrival = self.elapsed_minutes + travel

        commission, move, late_penalty, margin_penalty = order_terms(
            cfg,
            value=order.value,
            deadline=order.deadline_minutes,
            arrival=arrival,
            travel_minutes=travel,
            distance_km=distance,
        )

        reward = commission - move - late_penalty - margin_penalty

        if late_penalty > 0.0:
            self.late += 1
        else:
            self.on_time += 1
            self.revenue += order.value

        self.cost += move
        self.distance_total_km += distance
        self.elapsed_minutes = arrival + cfg.service_minutes
        self.current_node = node
        self.served_mask |= 1 << action
        self.served_sequence.append(action)
        self.total_reward += reward

        if self.deferred_count == 0:
            return self._finish(reward)

        return ShiftStep(reward=reward, done=False)

    def _finish(self, step_reward: float) -> ShiftStep:
        cfg = self.config

        return_minutes = self.travel_minutes[self.current_node][0]
        return_km = self.distance_km[self.current_node][0]

        early, return_cost, overtime, deferred = terminal_terms(
            cfg,
            finish_minutes=self.elapsed_minutes,
            return_minutes=return_minutes,
            return_km=return_km,
            deferred=self.deferred_count,
        )

        reward = early - return_cost - overtime - deferred

        self.end_minutes = self.elapsed_minutes + return_minutes
        self.overtime_minutes = max(0.0, self.end_minutes - cfg.shift_minutes)
        self.distance_total_km += return_km
        self.cost += return_cost
        self.total_reward += reward
        self.done = True

        return ShiftStep(reward=step_reward + reward, done=True)

    def summary(self) -> dict:
        margin = (
            (self.revenue - self.cost) / self.revenue
            if self.revenue > 0.0
            else None
        )

        return {
            "score": self.total_reward,
            "served": self.served_count,
            "on_time": self.on_time,
            "late": self.late,
            "deferred": self.deferred_count,
            "distance_km": self.distance_total_km,
            "end_minutes": self.end_minutes,
            "overtime_minutes": self.overtime_minutes,
            "shift_margin": margin,
        }


def play(
    env: ShiftEnvironment,
    sequence: Sequence[int],
) -> float:
    """
    Serve the orders in `sequence`, then stop the shift. Returns the score.
    """

    env.reset()

    for action in sequence:
        env.step(int(action))

    if not env.done:
        env.step(env.stop_action)

    return env.total_reward


def score_sequence(
    env: ShiftEnvironment,
    sequence: Sequence[int],
) -> float:
    """
    Same score as play(), computed without touching the environment state,
    so local search can evaluate many plans quickly.
    """

    cfg = env.config
    dist = env.distance_km
    time = env.travel_minutes

    node = 0
    clock = 0.0
    score = 0.0

    for k in sequence:
        target = k + 1
        travel = time[node][target]
        arrival = clock + travel
        order = env.orders[k]

        commission, move, late, margin = order_terms(
            cfg,
            value=order.value,
            deadline=order.deadline_minutes,
            arrival=arrival,
            travel_minutes=travel,
            distance_km=dist[node][target],
        )

        score += commission - move - late - margin
        clock = arrival + cfg.service_minutes
        node = target

    early, back, overtime, deferred = terminal_terms(
        cfg,
        finish_minutes=clock,
        return_minutes=time[node][0],
        return_km=dist[node][0],
        deferred=env.number_of_orders - len(sequence),
    )

    return score + early - back - overtime - deferred
