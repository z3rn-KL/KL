from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numbers import Integral

from domain import Delivery
from prioritization import DeliveryPriorityModel
from routing import RoadMatrixResult, TravelEfficiencyScorer


@dataclass(frozen=True)
class RoutingState:
    """
    Discrete state used by tabular RL.

    current_index:
        Matrix index of current location.
        0 = depot
        1..N = deliveries

    visited_mask:
        Bit mask describing which deliveries have already been visited.

    elapsed_time_bucket:
        Discretized elapsed route time.
    """

    current_index: int
    visited_mask: int
    elapsed_time_bucket: int


@dataclass(frozen=True)
class RewardConfig:
    """
    RL reward configuration.

    Reward:
        + priority bonus
        + travel-efficiency bonus
        - distance penalty
        - travel-time penalty
        - late-delivery penalty
        - lateness penalty
        - optional K-Means cluster-switch penalty

    cluster_switch_penalty defaults to 0.0 so all existing experiments
    remain backward compatible.
    """

    priority_bonus: float = 1.00
    travel_efficiency_bonus: float = 1.00

    distance_penalty: float = 0.50
    travel_time_penalty: float = 0.50

    late_delivery_penalty: float = 4.00
    lateness_penalty: float = 1.00

    # Soft spatial guidance from K-Means.
    cluster_switch_penalty: float = 0.0

    distance_scale_km: float = 10.0
    travel_time_scale_minutes: float = 15.0
    lateness_scale_minutes: float = 30.0

    time_bucket_minutes: float = 10.0

    def validate(self) -> None:
        weights = [
            self.priority_bonus,
            self.travel_efficiency_bonus,
            self.distance_penalty,
            self.travel_time_penalty,
            self.late_delivery_penalty,
            self.lateness_penalty,
            self.cluster_switch_penalty,
        ]

        if any(not np.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError(
                "Reward weights cannot be negative."
            )

        scales = [
            self.distance_scale_km,
            self.travel_time_scale_minutes,
            self.lateness_scale_minutes,
            self.time_bucket_minutes,
        ]

        if any(not np.isfinite(scale) or scale <= 0 for scale in scales):
            raise ValueError(
                "Reward scales must be greater than 0."
            )


@dataclass(frozen=True)
class RoutingStepInfo:
    """
    Detailed information for one RL action.

    The cluster fields are optional so old callers remain compatible.
    """

    delivery_id: str
    action: int

    from_matrix_index: int
    to_matrix_index: int

    departure_time: datetime
    arrival_time: datetime

    leg_distance_km: float
    leg_travel_time_minutes: float

    priority_score: float
    travel_efficiency_score: float

    on_time: bool | None
    lateness_minutes: float | None

    return_distance_km: float
    return_travel_time_minutes: float

    from_cluster_id: int | None = None
    to_cluster_id: int | None = None
    cluster_switched: bool = False
    cluster_switch_penalty: float = 0.0


class RoutingEnvironment:
    """
    Delivery-routing environment used by Q-Learning, SARSA and DQN.

    Action
    ------
    An action is the zero-based delivery index:
        0 -> first delivery
        1 -> second delivery
        ...

    State
    -----
    RoutingState(
        current_index,
        visited_mask,
        elapsed_time_bucket,
    )

    Episode
    -------
    Starts at the depot. The agent repeatedly selects one unvisited
    delivery. When the last delivery is visited, the environment
    automatically adds the return trip to the depot and terminates.

    K-Means integration
    -------------------
    delivery_cluster_ids may optionally contain one K-Means cluster ID
    for each delivery.

    A soft penalty is applied only when moving between two deliveries
    belonging to different clusters:

        reward_new =
            reward_original - cluster_switch_penalty

    The depot is not assigned to a cluster. Therefore depot -> first
    delivery and last delivery -> depot are not cluster switches.

    With cluster_switch_penalty=0.0, the original reward is unchanged.

    Notes
    -----
    Customer service duration is excluded because XeDu does not provide
    an authoritative service-time field.
    """

    def __init__(
        self,
        deliveries: list[Delivery] | tuple[Delivery, ...],
        road_matrix: RoadMatrixResult,
        start_time: datetime,
        priority_model: DeliveryPriorityModel | None = None,
        reward_config: RewardConfig | None = None,
        delivery_cluster_ids: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.deliveries = tuple(deliveries)
        self.road_matrix = road_matrix
        self.start_time = start_time

        self.priority_model = (
            priority_model
            or DeliveryPriorityModel()
        )

        self.reward_config = (
            reward_config
            or RewardConfig()
        )
        self.reward_config.validate()

        if delivery_cluster_ids is None:
            self.delivery_cluster_ids = None
        else:
            if len(delivery_cluster_ids) != len(self.deliveries):
                raise ValueError(
                    "delivery_cluster_ids must have "
                    "the same length as deliveries."
                )

            if any(isinstance(value, bool) or not isinstance(value, Integral)
                   for value in delivery_cluster_ids):
                raise ValueError("delivery_cluster_ids must contain integer labels")
            self.delivery_cluster_ids = tuple(int(v) for v in delivery_cluster_ids)

        self.travel_scorer = TravelEfficiencyScorer()

        self._validate_inputs()

        self.delivery_count = len(self.deliveries)
        self.full_visited_mask = (
            (1 << self.delivery_count) - 1
        )

        self.current_matrix_index = 0
        self.visited_mask = 0
        self.current_time = self.start_time

        self.total_distance_km = 0.0
        self.total_travel_time_minutes = 0.0

        self.cluster_switches = 0
        self.done = False

    def reset(self) -> RoutingState:
        """
        Reset the environment to the depot.
        """

        self.current_matrix_index = 0
        self.visited_mask = 0
        self.current_time = self.start_time

        self.total_distance_km = 0.0
        self.total_travel_time_minutes = 0.0

        self.cluster_switches = 0
        self.done = False

        return self.state()

    def state(self) -> RoutingState:
        """
        Return the current discrete RL state.
        """

        elapsed_minutes = max(
            0.0,
            (
                self.current_time
                - self.start_time
            ).total_seconds()
            / 60.0,
        )

        bucket = int(
            elapsed_minutes
            // self.reward_config.time_bucket_minutes
        )

        return RoutingState(
            current_index=self.current_matrix_index,
            visited_mask=self.visited_mask,
            elapsed_time_bucket=bucket,
        )

    def available_actions(self) -> tuple[int, ...]:
        """
        Return all currently unvisited delivery indices.
        """

        if self.done:
            return tuple()

        actions = []

        for action in range(self.delivery_count):
            if not self._is_visited(action):
                actions.append(action)

        return tuple(actions)

    @property
    def cluster_switch_rate(self) -> float:
        """
        Fraction of delivery-to-delivery transitions that switch cluster.

        Depot legs are excluded. Before at least two deliveries have been
        visited, the rate is defined as 0.0.
        """

        visited_count = self.visited_mask.bit_count()
        transitions = max(0, visited_count - 1)

        if transitions == 0:
            return 0.0

        return (
            self.cluster_switches
            / transitions
        )

    def step(
        self,
        action: int,
    ) -> tuple[
        RoutingState,
        float,
        bool,
        RoutingStepInfo,
    ]:
        """
        Execute one delivery action.

        Returns:
            next_state
            reward
            done
            info
        """

        self._validate_action(action)

        available_actions = (
            self.available_actions()
        )

        travel_scores = (
            self._calculate_travel_efficiency(
                actions=available_actions
            )
        )

        action_position = (
            available_actions.index(action)
        )

        travel_efficiency_score = float(
            travel_scores[action_position]
        )

        to_matrix_index = action + 1
        from_matrix_index = (
            self.current_matrix_index
        )

        from_cluster_id = (
            self._cluster_for_matrix_index(
                from_matrix_index
            )
        )

        to_cluster_id = (
            self._cluster_for_action(action)
        )

        cluster_switched = (
            from_cluster_id is not None
            and to_cluster_id is not None
            and from_cluster_id != to_cluster_id
        )

        applied_cluster_penalty = 0.0

        if cluster_switched:
            applied_cluster_penalty = float(
                self.reward_config
                .cluster_switch_penalty
            )

        leg_distance = float(
            self.road_matrix.distance_km[
                from_matrix_index,
                to_matrix_index,
            ]
        )

        leg_travel_time = float(
            self.road_matrix.travel_time_minutes[
                from_matrix_index,
                to_matrix_index,
            ]
        )

        if (
            not np.isfinite(leg_distance)
            or not np.isfinite(
                leg_travel_time
            )
        ):
            raise ValueError(
                "Selected action is unreachable."
            )

        departure_time = self.current_time

        priority_result = (
            self.priority_model.calculate(
                delivery=self.deliveries[action],
                current_time=departure_time,
                travel_efficiency_score=(
                    travel_efficiency_score
                ),
            )
        )

        arrival_time = (
            departure_time
            + timedelta(
                minutes=leg_travel_time
            )
        )

        delivery = self.deliveries[action]

        (
            on_time,
            lateness_minutes,
        ) = self._calculate_lateness(
            delivery=delivery,
            arrival_time=arrival_time,
        )

        reward = (
            self._calculate_action_reward(
                priority_score=(
                    priority_result.priority_score
                ),
                travel_efficiency_score=(
                    travel_efficiency_score
                ),
                leg_distance_km=(
                    leg_distance
                ),
                leg_travel_time_minutes=(
                    leg_travel_time
                ),
                on_time=on_time,
                lateness_minutes=(
                    lateness_minutes
                ),
            )
        )

        if cluster_switched:
            reward -= (
                applied_cluster_penalty
            )
            self.cluster_switches += 1

        self.total_distance_km += leg_distance
        self.total_travel_time_minutes += (
            leg_travel_time
        )

        self.current_time = arrival_time
        self.current_matrix_index = (
            to_matrix_index
        )

        self.visited_mask |= (
            1 << action
        )

        return_distance = 0.0
        return_travel_time = 0.0

        if (
            self.visited_mask
            == self.full_visited_mask
        ):
            (
                return_distance,
                return_travel_time,
            ) = self._return_to_depot()

            reward += (
                self._calculate_return_reward(
                    distance_km=(
                        return_distance
                    ),
                    travel_time_minutes=(
                        return_travel_time
                    ),
                )
            )

            self.done = True
            self.current_matrix_index = 0

        info = RoutingStepInfo(
            delivery_id=delivery.delivery_id,
            action=action,
            from_matrix_index=(
                from_matrix_index
            ),
            to_matrix_index=(
                to_matrix_index
            ),
            departure_time=departure_time,
            arrival_time=arrival_time,
            leg_distance_km=leg_distance,
            leg_travel_time_minutes=(
                leg_travel_time
            ),
            priority_score=(
                priority_result.priority_score
            ),
            travel_efficiency_score=(
                travel_efficiency_score
            ),
            on_time=on_time,
            lateness_minutes=(
                lateness_minutes
            ),
            return_distance_km=(
                return_distance
            ),
            return_travel_time_minutes=(
                return_travel_time
            ),
            from_cluster_id=(
                from_cluster_id
            ),
            to_cluster_id=(
                to_cluster_id
            ),
            cluster_switched=(
                cluster_switched
            ),
            cluster_switch_penalty=(
                applied_cluster_penalty
            ),
        )

        return (
            self.state(),
            float(reward),
            self.done,
            info,
        )

    def _cluster_for_matrix_index(
        self,
        matrix_index: int,
    ) -> int | None:
        """
        Return the K-Means cluster for a matrix node.

        Matrix index 0 is the depot and has no delivery cluster.
        """

        if self.delivery_cluster_ids is None:
            return None

        if matrix_index == 0:
            return None

        delivery_index = (
            matrix_index - 1
        )

        if (
            delivery_index < 0
            or delivery_index
            >= self.delivery_count
        ):
            raise ValueError(
                "Matrix index is outside the "
                "delivery range."
            )

        return (
            self.delivery_cluster_ids[
                delivery_index
            ]
        )

    def _cluster_for_action(
        self,
        action: int,
    ) -> int | None:
        """
        Return the K-Means cluster for an action.
        """

        if self.delivery_cluster_ids is None:
            return None

        return self.delivery_cluster_ids[
            action
        ]

    def _calculate_travel_efficiency(
        self,
        actions: tuple[int, ...],
    ) -> np.ndarray:
        distances = []
        travel_times = []

        for action in actions:
            matrix_index = action + 1

            distances.append(
                float(
                    self.road_matrix.distance_km[
                        self.current_matrix_index,
                        matrix_index,
                    ]
                )
            )

            travel_times.append(
                float(
                    self.road_matrix
                    .travel_time_minutes[
                        self.current_matrix_index,
                        matrix_index,
                    ]
                )
            )

        result = (
            self.travel_scorer
            .score_candidates(
                distance_km=distances,
                travel_time_minutes=(
                    travel_times
                ),
            )
        )

        return result.travel_efficiency

    @staticmethod
    def _calculate_lateness(
        delivery: Delivery,
        arrival_time: datetime,
    ) -> tuple[
        bool | None,
        float | None,
    ]:
        deadline = (
            delivery.expected_delivery_time
        )

        if deadline is None:
            return (
                None,
                None,
            )

        on_time = (
            arrival_time <= deadline
        )

        lateness_minutes = max(
            0.0,
            (
                arrival_time
                - deadline
            ).total_seconds()
            / 60.0,
        )

        return (
            on_time,
            lateness_minutes,
        )

    def _calculate_action_reward(
        self,
        priority_score: float,
        travel_efficiency_score: float,
        leg_distance_km: float,
        leg_travel_time_minutes: float,
        on_time: bool | None,
        lateness_minutes: float | None,
    ) -> float:
        config = self.reward_config

        reward = (
            config.priority_bonus
            * priority_score
        )

        reward += (
            config.travel_efficiency_bonus
            * travel_efficiency_score
        )

        reward -= (
            config.distance_penalty
            * (
                leg_distance_km
                / config.distance_scale_km
            )
        )

        reward -= (
            config.travel_time_penalty
            * (
                leg_travel_time_minutes
                / config.travel_time_scale_minutes
            )
        )

        if on_time is False:
            reward -= (
                config.late_delivery_penalty
            )

        if (
            lateness_minutes is not None
            and lateness_minutes > 0
        ):
            reward -= (
                config.lateness_penalty
                * (
                    lateness_minutes
                    / config.lateness_scale_minutes
                )
            )

        return float(reward)

    def _calculate_return_reward(
        self,
        distance_km: float,
        travel_time_minutes: float,
    ) -> float:
        """
        Terminal return-to-depot penalty.

        No priority, lateness or cluster-switch penalty is applied because
        no delivery occurs during the return leg.
        """

        config = self.reward_config
        reward = 0.0

        reward -= (
            config.distance_penalty
            * (
                distance_km
                / config.distance_scale_km
            )
        )

        reward -= (
            config.travel_time_penalty
            * (
                travel_time_minutes
                / config.travel_time_scale_minutes
            )
        )

        return float(reward)

    def _return_to_depot(
        self,
    ) -> tuple[
        float,
        float,
    ]:
        distance = float(
            self.road_matrix.distance_km[
                self.current_matrix_index,
                0,
            ]
        )

        travel_time = float(
            self.road_matrix
            .travel_time_minutes[
                self.current_matrix_index,
                0,
            ]
        )

        if (
            not np.isfinite(distance)
            or not np.isfinite(
                travel_time
            )
        ):
            raise ValueError(
                "Cannot return to depot."
            )

        self.total_distance_km += distance
        self.total_travel_time_minutes += (
            travel_time
        )

        self.current_time = (
            self.current_time
            + timedelta(
                minutes=travel_time
            )
        )

        return (
            distance,
            travel_time,
        )

    def _is_visited(
        self,
        action: int,
    ) -> bool:
        return bool(
            self.visited_mask
            & (
                1 << action
            )
        )

    def _validate_action(
        self,
        action: int,
    ) -> None:
        if self.done:
            raise ValueError(
                "Episode has already finished."
            )

        if not isinstance(
            action,
            int,
        ):
            raise ValueError(
                "Action must be an integer."
            )

        if (
            action < 0
            or action >= self.delivery_count
        ):
            raise ValueError(
                "Action is outside the "
                "delivery index range."
            )

        if self._is_visited(action):
            raise ValueError(
                "Delivery has already been visited."
            )

    def _validate_inputs(
        self,
    ) -> None:
        if not self.deliveries:
            raise ValueError(
                "deliveries cannot be empty."
            )

        delivery_ids = [
            str(delivery.delivery_id)
            for delivery
            in self.deliveries
        ]

        if (
            len(delivery_ids)
            != len(set(delivery_ids))
        ):
            raise ValueError(
                "Delivery IDs must be unique."
            )

        if (
            self.delivery_cluster_ids
            is not None
            and len(
                self.delivery_cluster_ids
            )
            != len(self.deliveries)
        ):
            raise ValueError(
                "delivery_cluster_ids must have "
                "the same length as deliveries."
            )

        expected_size = (
            len(self.deliveries) + 1
        )

        distance_shape = (
            self.road_matrix
            .distance_km.shape
        )

        travel_time_shape = (
            self.road_matrix
            .travel_time_minutes.shape
        )

        expected_shape = (
            expected_size,
            expected_size,
        )

        if distance_shape != expected_shape:
            raise ValueError(
                "Distance matrix size does not "
                "match deliveries."
            )

        if (
            travel_time_shape
            != expected_shape
        ):
            raise ValueError(
                "Travel-time matrix size does not "
                "match deliveries."
            )