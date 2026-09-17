from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from domain import Delivery

from prioritization import (
    DeliveryPriorityModel,
)

from routing.road_matrix import (
    RoadMatrixBuilder,
)

from routing.road_network import (
    RoadNetworkService,
)

from routing.travel_efficiency import (
    TravelEfficiencyScorer,
)


@dataclass(frozen=True)
class PriorityNearestNeighborStep:
    """
    One decision made by the
    Priority-aware Nearest Neighbor router.
    """

    sequence: int

    delivery_id: str

    from_node: int
    to_node: int

    departure_time: datetime
    arrival_time: datetime

    leg_distance_km: float
    leg_travel_time_minutes: float

    priority_score: float

    deadline_urgency: float | None
    service_priority: float | None
    waiting_time_score: float | None
    travel_efficiency_score: float | None
    information_coverage: float


@dataclass(frozen=True)
class PriorityNearestNeighborRoute:
    """
    Complete Priority-aware NN route.
    """

    delivery_order: tuple[str, ...]

    node_order: tuple[int, ...]

    steps: tuple[
        PriorityNearestNeighborStep,
        ...
    ]

    start_time: datetime
    end_time: datetime

    total_distance_km: float
    total_travel_time_minutes: float

    returned_to_depot: bool

    def number_of_deliveries(
        self,
    ) -> int:
        return len(
            self.delivery_order
        )


class PriorityNearestNeighborRouter:
    """
    Priority-aware Nearest Neighbor.

    At every routing step:

        1. Determine the current road node.

        2. Calculate road distance and
           estimated travel time from the
           current node to all unvisited
           deliveries.

        3. Convert those road metrics into
           travel-efficiency scores.

        4. Recalculate Priority Model V3
           for every candidate using the
           current simulated route time.

        5. Select the delivery with the
           highest priority score.

        6. Move to that delivery and advance
           current time by road travel time.

    Priority Model V3 currently uses:

        Deadline urgency   55%
        Service priority   20%
        Waiting time       15%
        Travel efficiency  10%

    Service duration at the customer is not
    included here because no authoritative
    service-time field is currently available
    from the XeDu dataset.
    """

    def __init__(
        self,
        road_network: RoadNetworkService,
        priority_model: (
            DeliveryPriorityModel
            | None
        ) = None,
    ) -> None:
        self.road_network = (
            road_network
        )

        self.priority_model = (
            priority_model
            or DeliveryPriorityModel()
        )

        self.matrix_builder = (
            RoadMatrixBuilder(
                road_network
            )
        )

        self.travel_scorer = (
            TravelEfficiencyScorer()
        )

    def build_route(
        self,
        depot_node: int,
        deliveries: (
            list[Delivery]
            | tuple[Delivery, ...]
        ),
        delivery_nodes: (
            list[int]
            | tuple[int, ...]
        ),
        start_time: datetime,
        return_to_depot: bool = True,
    ) -> PriorityNearestNeighborRoute:
        """
        Build a dynamic priority-aware route.
        """

        deliveries = tuple(
            deliveries
        )

        delivery_nodes = tuple(
            int(
                node
            )
            for node
            in delivery_nodes
        )

        depot_node = int(
            depot_node
        )

        self._validate_inputs(
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
        )

        all_nodes = (
            depot_node,
            *delivery_nodes,
        )

        matrix_result = (
            self.matrix_builder.build(
                all_nodes
            )
        )

        unvisited = set(
            range(
                len(
                    deliveries
                )
            )
        )

        current_matrix_index = 0
        current_time = start_time

        delivery_order = []

        node_order = [
            depot_node
        ]

        steps = []

        total_distance_km = 0.0
        total_travel_time_minutes = 0.0

        sequence = 1

        while unvisited:
            candidate_indices = sorted(
                unvisited
            )

            distances = []
            travel_times = []

            for delivery_index in (
                candidate_indices
            ):
                matrix_index = (
                    delivery_index
                    + 1
                )

                distance = (
                    matrix_result
                    .distance_km[
                        current_matrix_index,
                        matrix_index,
                    ]
                )

                travel_time = (
                    matrix_result
                    .travel_time_minutes[
                        current_matrix_index,
                        matrix_index,
                    ]
                )

                distances.append(
                    float(
                        distance
                    )
                )

                travel_times.append(
                    float(
                        travel_time
                    )
                )

            travel_result = (
                self.travel_scorer
                .score_candidates(
                    distance_km=distances,
                    travel_time_minutes=(
                        travel_times
                    ),
                )
            )

            best = self._select_candidate(
                deliveries=deliveries,
                candidate_indices=(
                    candidate_indices
                ),
                travel_efficiency=(
                    travel_result
                    .travel_efficiency
                ),
                current_time=current_time,
            )

            if best is None:
                raise ValueError(
                    "No reachable delivery "
                    "candidate is available."
                )

            (
                selected_delivery_index,
                selected_priority_result,
            ) = best

            next_matrix_index = (
                selected_delivery_index
                + 1
            )

            leg_distance = float(
                matrix_result
                .distance_km[
                    current_matrix_index,
                    next_matrix_index,
                ]
            )

            leg_travel_time = float(
                matrix_result
                .travel_time_minutes[
                    current_matrix_index,
                    next_matrix_index,
                ]
            )

            if (
                not np.isfinite(
                    leg_distance
                )
                or not np.isfinite(
                    leg_travel_time
                )
            ):
                raise ValueError(
                    "Selected delivery is "
                    "unreachable through the "
                    "road network."
                )

            departure_time = (
                current_time
            )

            arrival_time = (
                current_time
                + timedelta(
                    minutes=(
                        leg_travel_time
                    )
                )
            )

            delivery = deliveries[
                selected_delivery_index
            ]

            from_node = int(
                all_nodes[
                    current_matrix_index
                ]
            )

            to_node = int(
                delivery_nodes[
                    selected_delivery_index
                ]
            )

            steps.append(
                PriorityNearestNeighborStep(
                    sequence=sequence,
                    delivery_id=(
                        delivery.delivery_id
                    ),
                    from_node=from_node,
                    to_node=to_node,
                    departure_time=(
                        departure_time
                    ),
                    arrival_time=(
                        arrival_time
                    ),
                    leg_distance_km=(
                        leg_distance
                    ),
                    leg_travel_time_minutes=(
                        leg_travel_time
                    ),
                    priority_score=(
                        selected_priority_result
                        .priority_score
                    ),
                    deadline_urgency=(
                        selected_priority_result
                        .deadline_urgency
                    ),
                    service_priority=(
                        selected_priority_result
                        .service_priority
                    ),
                    waiting_time_score=(
                        selected_priority_result
                        .waiting_time_score
                    ),
                    travel_efficiency_score=(
                        selected_priority_result
                        .travel_efficiency_score
                    ),
                    information_coverage=(
                        selected_priority_result
                        .information_coverage
                    ),
                )
            )

            total_distance_km += (
                leg_distance
            )

            total_travel_time_minutes += (
                leg_travel_time
            )

            delivery_order.append(
                delivery.delivery_id
            )

            node_order.append(
                to_node
            )

            unvisited.remove(
                selected_delivery_index
            )

            current_matrix_index = (
                next_matrix_index
            )

            current_time = (
                arrival_time
            )

            sequence += 1

        if return_to_depot:
            return_distance = float(
                matrix_result
                .distance_km[
                    current_matrix_index,
                    0,
                ]
            )

            return_travel_time = float(
                matrix_result
                .travel_time_minutes[
                    current_matrix_index,
                    0,
                ]
            )

            if (
                not np.isfinite(
                    return_distance
                )
                or not np.isfinite(
                    return_travel_time
                )
            ):
                raise ValueError(
                    "Cannot return to depot "
                    "through the road network."
                )

            total_distance_km += (
                return_distance
            )

            total_travel_time_minutes += (
                return_travel_time
            )

            current_time = (
                current_time
                + timedelta(
                    minutes=(
                        return_travel_time
                    )
                )
            )

            node_order.append(
                depot_node
            )

        return (
            PriorityNearestNeighborRoute(
                delivery_order=tuple(
                    delivery_order
                ),
                node_order=tuple(
                    node_order
                ),
                steps=tuple(
                    steps
                ),
                start_time=start_time,
                end_time=current_time,
                total_distance_km=(
                    total_distance_km
                ),
                total_travel_time_minutes=(
                    total_travel_time_minutes
                ),
                returned_to_depot=(
                    return_to_depot
                ),
            )
        )

    def _select_candidate(
        self,
        deliveries: tuple[
            Delivery,
            ...
        ],
        candidate_indices: list[int],
        travel_efficiency: np.ndarray,
        current_time: datetime,
    ):
        """
        Calculate dynamic priority for every
        currently reachable candidate.

        Highest priority wins.

        Tie-breaking:
            1. Higher priority score
            2. Higher deadline urgency
            3. Higher travel efficiency
            4. Lower original delivery index
        """

        candidates = []

        for local_index, (
            delivery_index
        ) in enumerate(
            candidate_indices
        ):
            travel_score = float(
                travel_efficiency[
                    local_index
                ]
            )

            if not np.isfinite(
                travel_score
            ):
                continue

            result = (
                self.priority_model
                .calculate(
                    delivery=(
                        deliveries[
                            delivery_index
                        ]
                    ),
                    current_time=(
                        current_time
                    ),
                    travel_efficiency_score=(
                        travel_score
                    ),
                )
            )

            deadline_score = (
                result.deadline_urgency
            )

            if deadline_score is None:
                deadline_score = -1.0

            candidates.append(
                (
                    result.priority_score,
                    deadline_score,
                    travel_score,
                    -delivery_index,
                    delivery_index,
                    result,
                )
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
            ),
        )

        return (
            best[4],
            best[5],
        )

    @staticmethod
    def _validate_inputs(
        deliveries: tuple[
            Delivery,
            ...
        ],
        delivery_nodes: tuple[
            int,
            ...
        ],
    ) -> None:
        if not deliveries:
            raise ValueError(
                "deliveries cannot be empty."
            )

        if (
            len(
                deliveries
            )
            != len(
                delivery_nodes
            )
        ):
            raise ValueError(
                "deliveries and delivery_nodes "
                "must have the same length."
            )

        delivery_ids = [
            delivery.delivery_id
            for delivery in deliveries
        ]

        if (
            len(
                set(
                    delivery_ids
                )
            )
            != len(
                delivery_ids
            )
        ):
            raise ValueError(
                "Delivery IDs must be unique."
            )