from dataclasses import dataclass

import numpy as np

from routing.road_matrix import (
    RoadMatrixBuilder,
)

from routing.road_network import (
    RoadNetworkService,
)


@dataclass(frozen=True)
class ClarkeWrightRoute:
    """
    Result of the road-aware Clarke-Wright
    Savings heuristic.
    """

    delivery_order: tuple[str, ...]

    node_order: tuple[int, ...]

    total_distance_km: float

    total_travel_time_minutes: float

    initial_separate_distance_km: float

    total_savings_km: float

    merge_count: int

    returned_to_depot: bool

    def number_of_deliveries(
        self,
    ) -> int:
        return len(
            self.delivery_order
        )


class ClarkeWrightRouter:
    """
    Road-aware Parallel Clarke-Wright
    Savings heuristic.

    Initial solution:

        Depot -> A -> Depot
        Depot -> B -> Depot
        Depot -> C -> Depot
        ...

    For directed road networks, merging a
    route ending at i with a route beginning
    at j gives:

        ... -> i -> j -> ...

    Therefore the directional saving is:

        S(i, j)
            =
        d(i, depot)
        + d(depot, j)
        - d(i, j)

    This differs slightly from the common
    symmetric Clarke-Wright formula and is
    appropriate for directed OSM road graphs.

    No capacity constraint is applied in this
    baseline version. Capacity-aware VRP can
    be added later.
    """

    def __init__(
        self,
        road_network: RoadNetworkService,
    ) -> None:
        self.road_network = (
            road_network
        )

        self.matrix_builder = (
            RoadMatrixBuilder(
                road_network
            )
        )

    def build_route(
        self,
        depot_node: int,
        delivery_ids: (
            list[str]
            | tuple[str, ...]
        ),
        delivery_nodes: (
            list[int]
            | tuple[int, ...]
        ),
        return_to_depot: bool = True,
    ) -> ClarkeWrightRoute:
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

        self._validate_reachability(
            matrix_result=matrix_result,
            delivery_count=len(
                delivery_ids
            ),
        )

        initial_distance = (
            self._calculate_initial_distance(
                distance_matrix=(
                    matrix_result
                    .distance_km
                ),
                delivery_count=len(
                    delivery_ids
                ),
            )
        )

        savings = (
            self._calculate_savings(
                distance_matrix=(
                    matrix_result
                    .distance_km
                ),
                delivery_count=len(
                    delivery_ids
                ),
            )
        )

        final_indices, merge_count = (
            self._merge_routes(
                delivery_count=len(
                    delivery_ids
                ),
                savings=savings,
            )
        )

        delivery_order = tuple(
            delivery_ids[
                index
            ]
            for index
            in final_indices
        )

        ordered_delivery_nodes = tuple(
            delivery_nodes[
                index
            ]
            for index
            in final_indices
        )

        (
            total_distance,
            total_travel_time,
        ) = self._calculate_route_metrics(
            route_indices=final_indices,
            distance_matrix=(
                matrix_result
                .distance_km
            ),
            travel_time_matrix=(
                matrix_result
                .travel_time_minutes
            ),
            return_to_depot=(
                return_to_depot
            ),
        )

        if return_to_depot:
            node_order = (
                depot_node,
                *ordered_delivery_nodes,
                depot_node,
            )

            total_savings = (
                initial_distance
                - total_distance
            )
        else:
            node_order = (
                depot_node,
                *ordered_delivery_nodes,
            )

            total_savings = float(
                "nan"
            )

        return ClarkeWrightRoute(
            delivery_order=(
                delivery_order
            ),
            node_order=(
                node_order
            ),
            total_distance_km=(
                total_distance
            ),
            total_travel_time_minutes=(
                total_travel_time
            ),
            initial_separate_distance_km=(
                initial_distance
            ),
            total_savings_km=(
                total_savings
            ),
            merge_count=(
                merge_count
            ),
            returned_to_depot=(
                return_to_depot
            ),
        )

    @staticmethod
    def _calculate_initial_distance(
        distance_matrix: np.ndarray,
        delivery_count: int,
    ) -> float:
        """
        Total distance if every delivery gets
        its own route:

            D -> i -> D
        """

        total = 0.0

        for delivery_index in range(
            delivery_count
        ):
            matrix_index = (
                delivery_index
                + 1
            )

            outbound = float(
                distance_matrix[
                    0,
                    matrix_index,
                ]
            )

            inbound = float(
                distance_matrix[
                    matrix_index,
                    0,
                ]
            )

            total += (
                outbound
                + inbound
            )

        return total

    @staticmethod
    def _calculate_savings(
        distance_matrix: np.ndarray,
        delivery_count: int,
    ) -> list[
        tuple[
            float,
            int,
            int,
        ]
    ]:
        """
        Calculate all directional savings.

        Saving for joining:

            route_i ... -> i
            j -> ... route_j

        is:

            d(i, depot)
            + d(depot, j)
            - d(i, j)
        """

        savings = []

        for i in range(
            delivery_count
        ):
            i_matrix = (
                i
                + 1
            )

            for j in range(
                delivery_count
            ):
                if i == j:
                    continue

                j_matrix = (
                    j
                    + 1
                )

                i_to_depot = float(
                    distance_matrix[
                        i_matrix,
                        0,
                    ]
                )

                depot_to_j = float(
                    distance_matrix[
                        0,
                        j_matrix,
                    ]
                )

                i_to_j = float(
                    distance_matrix[
                        i_matrix,
                        j_matrix,
                    ]
                )

                if (
                    not np.isfinite(
                        i_to_depot
                    )
                    or not np.isfinite(
                        depot_to_j
                    )
                    or not np.isfinite(
                        i_to_j
                    )
                ):
                    continue

                saving = (
                    i_to_depot
                    + depot_to_j
                    - i_to_j
                )

                savings.append(
                    (
                        float(
                            saving
                        ),
                        i,
                        j,
                    )
                )

        # Highest saving first.
        #
        # i and j are included for deterministic
        # tie-breaking.
        savings.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
            )
        )

        return savings

    @staticmethod
    def _merge_routes(
        delivery_count: int,
        savings: list[
            tuple[
                float,
                int,
                int,
            ]
        ],
    ) -> tuple[
        list[int],
        int,
    ]:
        """
        Parallel Clarke-Wright route merging.

        A merge i -> j is allowed only when:

            i is at the end of its route
            j is at the beginning of its route
            i and j belong to different routes
        """

        routes = {
            index: [
                index
            ]
            for index in range(
                delivery_count
            )
        }

        route_of = {
            index: index
            for index in range(
                delivery_count
            )
        }

        merge_count = 0

        for (
            _saving,
            i,
            j,
        ) in savings:
            route_i_id = (
                route_of[
                    i
                ]
            )

            route_j_id = (
                route_of[
                    j
                ]
            )

            if (
                route_i_id
                == route_j_id
            ):
                continue

            route_i = routes[
                route_i_id
            ]

            route_j = routes[
                route_j_id
            ]

            if (
                route_i[
                    -1
                ]
                != i
            ):
                continue

            if (
                route_j[
                    0
                ]
                != j
            ):
                continue

            merged_route = (
                route_i
                + route_j
            )

            routes[
                route_i_id
            ] = merged_route

            del routes[
                route_j_id
            ]

            for delivery_index in (
                merged_route
            ):
                route_of[
                    delivery_index
                ] = route_i_id

            merge_count += 1

            if len(
                routes
            ) == 1:
                break

        if len(
            routes
        ) != 1:
            raise ValueError(
                "Clarke-Wright could not "
                "merge all deliveries into "
                "one route."
            )

        final_route = next(
            iter(
                routes.values()
            )
        )

        return (
            final_route,
            merge_count,
        )

    @staticmethod
    def _calculate_route_metrics(
        route_indices: list[int],
        distance_matrix: np.ndarray,
        travel_time_matrix: np.ndarray,
        return_to_depot: bool,
    ) -> tuple[
        float,
        float,
    ]:
        matrix_order = [
            0,
        ]

        matrix_order.extend(
            index + 1
            for index
            in route_indices
        )

        if return_to_depot:
            matrix_order.append(
                0
            )

        total_distance = 0.0

        total_travel_time = 0.0

        for position in range(
            len(
                matrix_order
            )
            - 1
        ):
            origin = (
                matrix_order[
                    position
                ]
            )

            destination = (
                matrix_order[
                    position
                    + 1
                ]
            )

            distance = float(
                distance_matrix[
                    origin,
                    destination,
                ]
            )

            travel_time = float(
                travel_time_matrix[
                    origin,
                    destination,
                ]
            )

            if (
                not np.isfinite(
                    distance
                )
                or not np.isfinite(
                    travel_time
                )
            ):
                raise ValueError(
                    "Clarke-Wright route "
                    "contains an unreachable "
                    "road leg."
                )

            total_distance += (
                distance
            )

            total_travel_time += (
                travel_time
            )

        return (
            total_distance,
            total_travel_time,
        )

    @staticmethod
    def _validate_reachability(
        matrix_result,
        delivery_count: int,
    ) -> None:
        """
        Every delivery must be reachable from
        the depot and capable of returning to
        it.

        With depot-SCC snapping this should
        always hold for real XeDu workloads.
        """

        for delivery_index in range(
            delivery_count
        ):
            matrix_index = (
                delivery_index
                + 1
            )

            outbound_distance = float(
                matrix_result.distance_km[
                    0,
                    matrix_index,
                ]
            )

            inbound_distance = float(
                matrix_result.distance_km[
                    matrix_index,
                    0,
                ]
            )

            outbound_time = float(
                matrix_result
                .travel_time_minutes[
                    0,
                    matrix_index,
                ]
            )

            inbound_time = float(
                matrix_result
                .travel_time_minutes[
                    matrix_index,
                    0,
                ]
            )

            if (
                not np.isfinite(
                    outbound_distance
                )
                or not np.isfinite(
                    inbound_distance
                )
                or not np.isfinite(
                    outbound_time
                )
                or not np.isfinite(
                    inbound_time
                )
            ):
                raise ValueError(
                    "At least one delivery "
                    "is not mutually reachable "
                    "with the depot."
                )

    @staticmethod
    def _validate_inputs(
        delivery_ids: tuple[
            str,
            ...
        ],
        delivery_nodes: tuple[
            int,
            ...
        ],
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
                "delivery_ids and "
                "delivery_nodes must have "
                "the same length."
            )

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
                "delivery_ids must be unique."
            )