from dataclasses import dataclass

import numpy as np

from routing.road_matrix import (
    RoadMatrixBuilder,
)
from routing.road_network import (
    RoadNetworkService,
)


@dataclass(frozen=True)
class NearestNeighborRoute:
    """
    Result of a Nearest Neighbor route.

    delivery_order:
        Delivery IDs in visiting order.

    node_order:
        Road nodes in visiting order,
        including depot at the beginning
        and optionally at the end.

    total_distance_km:
        Total shortest-road distance.

    total_travel_time_minutes:
        Total estimated road travel time.
    """

    delivery_order: tuple[str, ...]

    node_order: tuple[int, ...]

    total_distance_km: float

    total_travel_time_minutes: float

    returned_to_depot: bool

    def number_of_deliveries(
        self,
    ) -> int:
        return len(
            self.delivery_order
        )


class NearestNeighborRouter:
    """
    Road-aware Nearest Neighbor baseline.

    At every step:

        current node
            ->
        choose the closest unvisited delivery
        according to the selected road metric.

    Supported metrics:

        distance
        travel_time

    The algorithm uses the road-distance /
    travel-time matrix rather than geometric
    latitude-longitude distance.
    """

    VALID_METRICS = {
        "distance",
        "travel_time",
    }

    def __init__(
        self,
        road_network: RoadNetworkService,
        metric: str = "distance",
    ) -> None:
        if metric not in self.VALID_METRICS:
            raise ValueError(
                "metric must be either "
                "'distance' or 'travel_time'."
            )

        self.road_network = (
            road_network
        )

        self.metric = metric

        self.matrix_builder = (
            RoadMatrixBuilder(
                road_network
            )
        )

    def build_route(
        self,
        depot_node: int,
        delivery_ids: list[str]
        | tuple[str, ...],
        delivery_nodes: list[int]
        | tuple[int, ...],
        return_to_depot: bool = True,
    ) -> NearestNeighborRoute:
        """
        Build one greedy Nearest Neighbor route.

        Parameters
        ----------
        depot_node:
            Road-network node of the depot.

        delivery_ids:
            Delivery identifiers.

        delivery_nodes:
            Corresponding road-network nodes.

        return_to_depot:
            Whether the route returns to the depot.

        Returns
        -------
        NearestNeighborRoute
        """

        delivery_ids = tuple(
            str(
                delivery_id
            )
            for delivery_id
            in delivery_ids
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
            delivery_ids=delivery_ids,
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

        if self.metric == "distance":
            selection_matrix = (
                matrix_result.distance_km
            )
        else:
            selection_matrix = (
                matrix_result
                .travel_time_minutes
            )

        unvisited = set(
            range(
                len(
                    delivery_ids
                )
            )
        )

        current_matrix_index = 0

        delivery_order = []

        node_order = [
            depot_node
        ]

        total_distance_km = 0.0

        total_travel_time_minutes = (
            0.0
        )

        while unvisited:
            next_delivery_index = (
                self._choose_next_delivery(
                    current_matrix_index=(
                        current_matrix_index
                    ),
                    unvisited=unvisited,
                    selection_matrix=(
                        selection_matrix
                    ),
                )
            )

            if next_delivery_index is None:
                raise ValueError(
                    "At least one remaining "
                    "delivery is unreachable."
                )

            next_matrix_index = (
                next_delivery_index
                + 1
            )

            leg_distance = (
                matrix_result.distance_km[
                    current_matrix_index,
                    next_matrix_index,
                ]
            )

            leg_travel_time = (
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
                    "Nearest Neighbor selected "
                    "an unreachable road leg."
                )

            total_distance_km += float(
                leg_distance
            )

            total_travel_time_minutes += (
                float(
                    leg_travel_time
                )
            )

            delivery_order.append(
                delivery_ids[
                    next_delivery_index
                ]
            )

            node_order.append(
                delivery_nodes[
                    next_delivery_index
                ]
            )

            unvisited.remove(
                next_delivery_index
            )

            current_matrix_index = (
                next_matrix_index
            )

        if return_to_depot:
            return_distance = (
                matrix_result.distance_km[
                    current_matrix_index,
                    0,
                ]
            )

            return_travel_time = (
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
                    "through road network."
                )

            total_distance_km += float(
                return_distance
            )

            total_travel_time_minutes += (
                float(
                    return_travel_time
                )
            )

            node_order.append(
                depot_node
            )

        return NearestNeighborRoute(
            delivery_order=tuple(
                delivery_order
            ),
            node_order=tuple(
                node_order
            ),
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

    @staticmethod
    def _choose_next_delivery(
        current_matrix_index: int,
        unvisited: set[int],
        selection_matrix: np.ndarray,
    ) -> int | None:
        """
        Find nearest reachable unvisited
        delivery.

        When two candidates have exactly the
        same value, the lower delivery index
        is selected for deterministic output.
        """

        best_delivery_index = None

        best_value = float(
            "inf"
        )

        for delivery_index in sorted(
            unvisited
        ):
            matrix_index = (
                delivery_index
                + 1
            )

            value = float(
                selection_matrix[
                    current_matrix_index,
                    matrix_index,
                ]
            )

            if not np.isfinite(
                value
            ):
                continue

            if value < best_value:
                best_value = value

                best_delivery_index = (
                    delivery_index
                )

        return best_delivery_index

    @staticmethod
    def _validate_inputs(
        delivery_ids: tuple[str, ...],
        delivery_nodes: tuple[int, ...],
    ) -> None:
        if not delivery_ids:
            raise ValueError(
                "delivery_ids cannot be empty."
            )

        if (
            len(
                delivery_ids
            )
            != len(
                delivery_nodes
            )
        ):
            raise ValueError(
                "delivery_ids and delivery_nodes "
                "must have the same length."
            )

        if len(
            set(
                delivery_ids
            )
        ) != len(
            delivery_ids
        ):
            raise ValueError(
                "delivery_ids must be unique."
            )