from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .objective import (
    ObjectiveBreakdown,
    ObjectiveConfig,
    ObjectiveContext,
    compute_route_objective,
    compute_step_cost,
)


@dataclass(frozen=True)
class ProposedRoutingState:
    """
    Trạng thái cốt lõi của Proposed RL.

    current_node:
        0 = depot
        1..N = delivery

    visited_mask:
        Bit-mask biểu diễn các delivery đã giao.

    elapsed_time_minutes:
        Số phút đã trôi qua kể từ đầu workload.

    remaining_deliveries:
        Số delivery chưa giao.
    """

    current_node: int
    visited_mask: int
    elapsed_time_minutes: float
    remaining_deliveries: int


@dataclass(frozen=True)
class CandidateInfo:
    """
    Thông tin của một delivery ứng viên trước khi agent chọn.
    """

    action: int
    node_index: int

    distance_km: float
    travel_time_minutes: float

    estimated_arrival_minutes: float

    deadline_valid: bool
    deadline_minutes: float | None
    slack_minutes: float | None
    lateness_minutes: float

    cluster_switched: bool


@dataclass(frozen=True)
class EpisodeTotals:
    """
    Tổng các đại lượng đã phát sinh trong episode.
    """

    total_distance_km: float
    total_travel_time_minutes: float
    total_lateness_minutes: float
    cluster_switches: int


@dataclass(frozen=True)
class StepResult:
    """
    Kết quả trả về sau một action.
    """

    state: ProposedRoutingState

    action: int
    reward: float
    done: bool

    candidate: CandidateInfo

    delivery_step_cost: ObjectiveBreakdown
    return_to_depot_cost: ObjectiveBreakdown | None

    transition_cost: float

    totals: EpisodeTotals


class ProposedRoutingEnvironment:
    """
    Environment riêng của Proposed RL.

    Agent chỉ có một nhiệm vụ:

        chọn delivery tiếp theo.

    Environment chịu trách nhiệm:

        action
            ↓
        road distance / travel time
            ↓
        arrival time
            ↓
        deadline / lateness
            ↓
        cluster switch
            ↓
        step objective cost
            ↓
        reward = - step cost
            ↓
        state mới

    Quy ước road matrix:

        node 0     = depot
        node 1..N  = delivery

    Quy ước action:

        action 0 -> delivery node 1
        action 1 -> delivery node 2
        ...
    """

    def __init__(
        self,
        *,
        distance_matrix_km: Sequence[Sequence[float]],
        travel_time_matrix_minutes: Sequence[Sequence[float]],
        deadlines_minutes: Sequence[float | None],
        cluster_ids: Sequence[int],
        objective_config: ObjectiveConfig | None = None,
    ) -> None:
        self.distance_matrix_km = [
            list(row)
            for row in distance_matrix_km
        ]

        self.travel_time_matrix_minutes = [
            list(row)
            for row in travel_time_matrix_minutes
        ]

        self.deadlines_minutes = list(deadlines_minutes)
        self.cluster_ids = list(cluster_ids)

        self.objective_config = (
            objective_config
            if objective_config is not None
            else ObjectiveConfig()
        )

        self._validate_inputs()

        self.number_of_deliveries = len(
            self.deadlines_minutes
        )

        self.objective_context = ObjectiveContext(
            number_of_deliveries=(
                self.number_of_deliveries
            ),
            number_of_valid_deadlines=sum(
                deadline is not None
                for deadline in self.deadlines_minutes
            ),
        )

        self._all_visited_mask = (
            (1 << self.number_of_deliveries) - 1
        )

        self.reset()

    def _validate_inputs(self) -> None:
        node_count = len(
            self.distance_matrix_km
        )

        if node_count < 2:
            raise ValueError(
                "Road matrix must contain depot "
                "and at least one delivery."
            )

        if (
            len(self.travel_time_matrix_minutes)
            != node_count
        ):
            raise ValueError(
                "Distance matrix and travel-time matrix "
                "must have the same size."
            )

        for row in self.distance_matrix_km:
            if len(row) != node_count:
                raise ValueError(
                    "Distance matrix must be square."
                )

        for row in self.travel_time_matrix_minutes:
            if len(row) != node_count:
                raise ValueError(
                    "Travel-time matrix must be square."
                )

        delivery_count = node_count - 1

        if len(self.deadlines_minutes) != delivery_count:
            raise ValueError(
                "deadlines_minutes must contain exactly "
                "one value per delivery."
            )

        if len(self.cluster_ids) != delivery_count:
            raise ValueError(
                "cluster_ids must contain exactly "
                "one value per delivery."
            )

        for row in self.distance_matrix_km:
            for value in row:
                if (
                    not math.isfinite(value)
                    or value < 0.0
                ):
                    raise ValueError(
                        "Distance matrix values must be "
                        "finite and non-negative."
                    )

        for row in self.travel_time_matrix_minutes:
            for value in row:
                if (
                    not math.isfinite(value)
                    or value < 0.0
                ):
                    raise ValueError(
                        "Travel-time matrix values must be "
                        "finite and non-negative."
                    )

        for deadline in self.deadlines_minutes:
            if deadline is None:
                continue

            if (
                not math.isfinite(deadline)
                or deadline < 0.0
            ):
                raise ValueError(
                    "Deadline must be non-negative "
                    "or None."
                )

    def reset(self) -> ProposedRoutingState:
        self.current_node = 0
        self.visited_mask = 0
        self.elapsed_time_minutes = 0.0
        self.done = False

        self.route_actions: list[int] = []

        self.total_distance_km = 0.0
        self.total_travel_time_minutes = 0.0
        self.total_lateness_minutes = 0.0
        self.cluster_switches = 0

        return self.state

    @property
    def state(self) -> ProposedRoutingState:
        return ProposedRoutingState(
            current_node=self.current_node,
            visited_mask=self.visited_mask,
            elapsed_time_minutes=(
                self.elapsed_time_minutes
            ),
            remaining_deliveries=(
                self.number_of_deliveries
                - self.visited_mask.bit_count()
            ),
        )

    @property
    def totals(self) -> EpisodeTotals:
        return EpisodeTotals(
            total_distance_km=(
                self.total_distance_km
            ),
            total_travel_time_minutes=(
                self.total_travel_time_minutes
            ),
            total_lateness_minutes=(
                self.total_lateness_minutes
            ),
            cluster_switches=(
                self.cluster_switches
            ),
        )

    def is_action_available(
        self,
        action: int,
    ) -> bool:
        if self.done:
            return False

        if not (
            0 <= action < self.number_of_deliveries
        ):
            return False

        bit = 1 << action

        return (self.visited_mask & bit) == 0

    def available_actions(self) -> tuple[int, ...]:
        if self.done:
            return ()

        return tuple(
            action
            for action in range(
                self.number_of_deliveries
            )
            if self.is_action_available(action)
        )

    def _validate_action(
        self,
        action: int,
    ) -> None:
        if self.done:
            raise RuntimeError(
                "Episode is already finished."
            )

        if not isinstance(action, int):
            raise TypeError(
                "Action must be an integer."
            )

        if not (
            0 <= action < self.number_of_deliveries
        ):
            raise ValueError(
                f"Action must be in range "
                f"0..{self.number_of_deliveries - 1}."
            )

        if not self.is_action_available(action):
            raise ValueError(
                f"Delivery action {action} "
                "has already been visited."
            )

    def candidate_info(
        self,
        action: int,
    ) -> CandidateInfo:
        """
        Tính thông tin mà agent có thể biết
        TRƯỚC KHI chọn candidate.
        """

        self._validate_action(action)

        node_index = action + 1

        distance_km = (
            self.distance_matrix_km[
                self.current_node
            ][node_index]
        )

        travel_time_minutes = (
            self.travel_time_matrix_minutes[
                self.current_node
            ][node_index]
        )

        estimated_arrival_minutes = (
            self.elapsed_time_minutes
            + travel_time_minutes
        )

        deadline = self.deadlines_minutes[action]

        deadline_valid = deadline is not None

        if deadline is None:
            slack_minutes = None
            lateness_minutes = 0.0

        else:
            slack_minutes = (
                deadline
                - estimated_arrival_minutes
            )

            lateness_minutes = max(
                0.0,
                estimated_arrival_minutes
                - deadline,
            )

        # Depot không thuộc cluster nào,
        # nên depot -> delivery đầu tiên
        # không tính là cluster switch.
        if self.current_node == 0:
            cluster_switched = False

        else:
            current_delivery_action = (
                self.current_node - 1
            )

            current_cluster = self.cluster_ids[
                current_delivery_action
            ]

            candidate_cluster = self.cluster_ids[
                action
            ]

            cluster_switched = (
                current_cluster != candidate_cluster
            )

        return CandidateInfo(
            action=action,
            node_index=node_index,
            distance_km=distance_km,
            travel_time_minutes=(
                travel_time_minutes
            ),
            estimated_arrival_minutes=(
                estimated_arrival_minutes
            ),
            deadline_valid=deadline_valid,
            deadline_minutes=deadline,
            slack_minutes=slack_minutes,
            lateness_minutes=lateness_minutes,
            cluster_switched=cluster_switched,
        )

    def step(
        self,
        action: int,
    ) -> StepResult:
        """
        Thực hiện một action.

        Reward:

            r_t = - Delta J_t

        Nếu đây là delivery cuối cùng,
        environment tự động thêm chặng quay về depot.
        """

        candidate = self.candidate_info(action)

        delivery_step_cost = compute_step_cost(
            distance_km=candidate.distance_km,
            travel_time_minutes=(
                candidate.travel_time_minutes
            ),
            lateness_minutes=(
                candidate.lateness_minutes
            ),
            cluster_switched=(
                candidate.cluster_switched
            ),
            context=self.objective_context,
            config=self.objective_config,
        )

        self.total_distance_km += (
            candidate.distance_km
        )

        self.total_travel_time_minutes += (
            candidate.travel_time_minutes
        )

        self.total_lateness_minutes += (
            candidate.lateness_minutes
        )

        if candidate.cluster_switched:
            self.cluster_switches += 1

        self.elapsed_time_minutes = (
            candidate.estimated_arrival_minutes
        )

        self.current_node = candidate.node_index

        self.visited_mask |= (
            1 << action
        )

        self.route_actions.append(action)

        transition_cost = (
            delivery_step_cost.total_cost
        )

        return_to_depot_cost = None

        all_visited = (
            self.visited_mask
            == self._all_visited_mask
        )

        if all_visited:
            return_distance_km = (
                self.distance_matrix_km[
                    self.current_node
                ][0]
            )

            return_time_minutes = (
                self.travel_time_matrix_minutes[
                    self.current_node
                ][0]
            )

            return_to_depot_cost = compute_step_cost(
                distance_km=return_distance_km,
                travel_time_minutes=(
                    return_time_minutes
                ),
                lateness_minutes=0.0,
                cluster_switched=False,
                context=self.objective_context,
                config=self.objective_config,
            )

            transition_cost += (
                return_to_depot_cost.total_cost
            )

            self.total_distance_km += (
                return_distance_km
            )

            self.total_travel_time_minutes += (
                return_time_minutes
            )

            self.elapsed_time_minutes += (
                return_time_minutes
            )

            self.current_node = 0
            self.done = True

        reward = -transition_cost

        return StepResult(
            state=self.state,
            action=action,
            reward=reward,
            done=self.done,
            candidate=candidate,
            delivery_step_cost=(
                delivery_step_cost
            ),
            return_to_depot_cost=(
                return_to_depot_cost
            ),
            transition_cost=transition_cost,
            totals=self.totals,
        )

    def final_objective(
        self,
    ) -> ObjectiveBreakdown:
        """
        Trả về J(pi) của route hoàn chỉnh.
        """

        if not self.done:
            raise RuntimeError(
                "final_objective() is only available "
                "after episode completion."
            )

        return compute_route_objective(
            total_distance_km=(
                self.total_distance_km
            ),
            total_travel_time_minutes=(
                self.total_travel_time_minutes
            ),
            total_lateness_minutes=(
                self.total_lateness_minutes
            ),
            cluster_switches=(
                self.cluster_switches
            ),
            context=self.objective_context,
            config=self.objective_config,
        )