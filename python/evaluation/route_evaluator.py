from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from domain import Delivery

from routing import (
    RoadMatrixBuilder,
    RoadNetworkService,
)


@dataclass(frozen=True)
class DeliveryEvaluation:
    """
    Evaluation result for one delivery.
    """

    sequence: int
    delivery_id: str

    arrival_time: datetime

    expected_delivery_time: datetime | None

    has_deadline: bool
    on_time: bool | None

    lateness_minutes: float | None

    leg_distance_km: float
    leg_travel_time_minutes: float


@dataclass(frozen=True)
class RouteEvaluationResult:
    """
    Common evaluation metrics for one route.
    """

    algorithm: str

    number_of_deliveries: int

    deliveries_with_deadline: int
    deliveries_without_deadline: int

    on_time_deliveries: int
    late_deliveries: int

    on_time_rate: float | None

    total_lateness_minutes: float
    mean_lateness_minutes: float
    max_lateness_minutes: float

    total_distance_km: float
    total_travel_time_minutes: float

    start_time: datetime
    end_time: datetime

    returned_to_depot: bool

    delivery_results: tuple[
        DeliveryEvaluation,
        ...
    ]


class RouteEvaluator:
    """
    Evaluate any fixed delivery order using
    the same road network and service metrics.

    This makes comparisons between:

        Nearest Neighbor
        Priority-aware NN
        Clarke-Wright
        RL

    consistent.

    Notes
    -----
    - Missing deadlines are not imputed.
    - Orders without deadlines are excluded
      from on-time-rate calculation.
    - Customer service duration is currently
      not included because the XeDu dataset
      does not provide an authoritative
      service-time field.
    """

    def __init__(
        self,
        road_network: RoadNetworkService,
    ) -> None:
        self.road_network = road_network

        self.matrix_builder = (
            RoadMatrixBuilder(
                road_network
            )
        )

    def evaluate(
        self,
        algorithm: str,
        depot_node: int,
        deliveries: (
            list[Delivery]
            | tuple[Delivery, ...]
        ),
        delivery_nodes: (
            list[int]
            | tuple[int, ...]
        ),
        delivery_order: (
            list[str]
            | tuple[str, ...]
        ),
        start_time: datetime,
        return_to_depot: bool = True,
    ) -> RouteEvaluationResult:
        deliveries = tuple(
            deliveries
        )

        delivery_nodes = tuple(
            int(node)
            for node in delivery_nodes
        )

        delivery_order = tuple(
            str(delivery_id)
            for delivery_id in delivery_order
        )

        depot_node = int(
            depot_node
        )

        self._validate_inputs(
            deliveries=deliveries,
            delivery_nodes=delivery_nodes,
            delivery_order=delivery_order,
        )

        delivery_by_id = {
            str(delivery.delivery_id): delivery
            for delivery in deliveries
        }

        node_by_id = {
            str(delivery.delivery_id): node
            for delivery, node
            in zip(
                deliveries,
                delivery_nodes,
                strict=True,
            )
        }

        ordered_nodes = [
            depot_node,
        ]

        for delivery_id in delivery_order:
            ordered_nodes.append(
                node_by_id[
                    delivery_id
                ]
            )

        matrix_result = (
            self.matrix_builder.build(
                ordered_nodes
            )
        )

        current_time = start_time

        total_distance_km = 0.0
        total_travel_time_minutes = 0.0

        delivery_results = []

        for sequence, delivery_id in enumerate(
            delivery_order,
            start=1,
        ):
            from_index = (
                sequence - 1
            )

            to_index = sequence

            leg_distance = float(
                matrix_result.distance_km[
                    from_index,
                    to_index,
                ]
            )

            leg_travel_time = float(
                matrix_result
                .travel_time_minutes[
                    from_index,
                    to_index,
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
                    f"Unreachable route leg "
                    f"before delivery "
                    f"{delivery_id}."
                )

            current_time = (
                current_time
                + timedelta(
                    minutes=leg_travel_time
                )
            )

            total_distance_km += (
                leg_distance
            )

            total_travel_time_minutes += (
                leg_travel_time
            )

            delivery = delivery_by_id[
                delivery_id
            ]

            deadline = (
                delivery.expected_delivery_time
            )

            if deadline is None:
                has_deadline = False
                on_time = None
                lateness_minutes = None

            else:
                has_deadline = True

                on_time = (
                    current_time
                    <= deadline
                )

                lateness_minutes = max(
                    0.0,
                    (
                        current_time
                        - deadline
                    ).total_seconds()
                    / 60.0,
                )

            delivery_results.append(
                DeliveryEvaluation(
                    sequence=sequence,
                    delivery_id=delivery_id,
                    arrival_time=current_time,
                    expected_delivery_time=(
                        deadline
                    ),
                    has_deadline=(
                        has_deadline
                    ),
                    on_time=on_time,
                    lateness_minutes=(
                        lateness_minutes
                    ),
                    leg_distance_km=(
                        leg_distance
                    ),
                    leg_travel_time_minutes=(
                        leg_travel_time
                    ),
                )
            )

        if return_to_depot:
            last_index = len(
                ordered_nodes
            ) - 1

            return_distance = float(
                matrix_result.distance_km[
                    last_index,
                    0,
                ]
            )

            return_time = float(
                matrix_result
                .travel_time_minutes[
                    last_index,
                    0,
                ]
            )

            if (
                not np.isfinite(
                    return_distance
                )
                or not np.isfinite(
                    return_time
                )
            ):
                raise ValueError(
                    "Cannot return to depot."
                )

            total_distance_km += (
                return_distance
            )

            total_travel_time_minutes += (
                return_time
            )

            current_time = (
                current_time
                + timedelta(
                    minutes=return_time
                )
            )

        valid_deadline_results = [
            result
            for result in delivery_results
            if result.has_deadline
        ]

        deliveries_with_deadline = len(
            valid_deadline_results
        )

        deliveries_without_deadline = (
            len(
                delivery_results
            )
            - deliveries_with_deadline
        )

        on_time_deliveries = sum(
            1
            for result
            in valid_deadline_results
            if result.on_time is True
        )

        late_deliveries = sum(
            1
            for result
            in valid_deadline_results
            if result.on_time is False
        )

        if deliveries_with_deadline > 0:
            on_time_rate = (
                on_time_deliveries
                / deliveries_with_deadline
            )
        else:
            on_time_rate = None

        lateness_values = [
            float(
                result.lateness_minutes
            )
            for result
            in valid_deadline_results
            if result.lateness_minutes
            is not None
        ]

        if lateness_values:
            total_lateness = sum(
                lateness_values
            )

            mean_lateness = (
                total_lateness
                / len(
                    lateness_values
                )
            )

            max_lateness = max(
                lateness_values
            )
        else:
            total_lateness = 0.0
            mean_lateness = 0.0
            max_lateness = 0.0

        return RouteEvaluationResult(
            algorithm=algorithm,
            number_of_deliveries=len(
                delivery_order
            ),
            deliveries_with_deadline=(
                deliveries_with_deadline
            ),
            deliveries_without_deadline=(
                deliveries_without_deadline
            ),
            on_time_deliveries=(
                on_time_deliveries
            ),
            late_deliveries=(
                late_deliveries
            ),
            on_time_rate=(
                on_time_rate
            ),
            total_lateness_minutes=(
                total_lateness
            ),
            mean_lateness_minutes=(
                mean_lateness
            ),
            max_lateness_minutes=(
                max_lateness
            ),
            total_distance_km=(
                total_distance_km
            ),
            total_travel_time_minutes=(
                total_travel_time_minutes
            ),
            start_time=start_time,
            end_time=current_time,
            returned_to_depot=(
                return_to_depot
            ),
            delivery_results=tuple(
                delivery_results
            ),
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
        delivery_order: tuple[
            str,
            ...
        ],
    ) -> None:
        if not deliveries:
            raise ValueError(
                "deliveries cannot be empty."
            )

        if (
            len(deliveries)
            != len(delivery_nodes)
        ):
            raise ValueError(
                "deliveries and delivery_nodes "
                "must have the same length."
            )

        delivery_ids = [
            str(delivery.delivery_id)
            for delivery in deliveries
        ]

        if (
            len(set(delivery_ids))
            != len(delivery_ids)
        ):
            raise ValueError(
                "Delivery IDs must be unique."
            )

        if (
            len(delivery_order)
            != len(delivery_ids)
        ):
            raise ValueError(
                "delivery_order must contain "
                "every delivery exactly once."
            )

        if (
            set(delivery_order)
            != set(delivery_ids)
        ):
            raise ValueError(
                "delivery_order must contain "
                "the same delivery IDs as "
                "deliveries."
            )