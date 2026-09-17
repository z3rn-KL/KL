from dataclasses import dataclass

import networkx as nx
import numpy as np

from routing.road_network import RoadNetworkService


@dataclass(frozen=True)
class RoadMatrixResult:
    """
    Road-network matrices for a list of road nodes.

    Matrices are directional because the road graph
    may contain one-way streets.

    distance_km[i, j]:
        shortest road distance from node i to node j.

    travel_time_minutes[i, j]:
        shortest estimated travel time from node i to node j.

    np.inf means the destination cannot be reached
    from that origin in the directed graph.
    """

    node_ids: tuple[int, ...]

    distance_km: np.ndarray

    travel_time_minutes: np.ndarray

    def size(self) -> int:
        return len(
            self.node_ids
        )


class RoadMatrixBuilder:
    """
    Efficiently build road distance and travel-time matrices.

    Instead of running shortest-path independently for every
    origin-destination pair, Dijkstra is executed once per
    unique origin node.

    Results are cached inside the builder so repeated requests
    for the same origin node do not need to recompute paths.
    """

    def __init__(
        self,
        road_network: RoadNetworkService,
    ) -> None:
        self.road_network = road_network

        self.graph = road_network.graph

        self._distance_cache: dict[
            int,
            dict,
        ] = {}

        self._travel_time_cache: dict[
            int,
            dict,
        ] = {}

    def build(
        self,
        node_ids: list[int] | tuple[int, ...],
    ) -> RoadMatrixResult:
        """
        Build directional road matrices for the supplied nodes.
        """

        normalized_nodes = tuple(
            int(node_id)
            for node_id in node_ids
        )

        if not normalized_nodes:
            raise ValueError(
                "node_ids cannot be empty."
            )

        self._validate_nodes(
            normalized_nodes
        )

        self._validate_travel_time_data()

        size = len(
            normalized_nodes
        )

        distance_matrix = np.full(
            (
                size,
                size,
            ),
            np.inf,
            dtype=float,
        )

        travel_time_matrix = np.full(
            (
                size,
                size,
            ),
            np.inf,
            dtype=float,
        )

        for origin_index, origin_node in enumerate(
            normalized_nodes
        ):
            distance_lengths = (
                self._get_distance_lengths(
                    origin_node
                )
            )

            travel_time_lengths = (
                self._get_travel_time_lengths(
                    origin_node
                )
            )

            for (
                destination_index,
                destination_node,
            ) in enumerate(
                normalized_nodes
            ):
                if (
                    origin_node
                    == destination_node
                ):
                    distance_matrix[
                        origin_index,
                        destination_index,
                    ] = 0.0

                    travel_time_matrix[
                        origin_index,
                        destination_index,
                    ] = 0.0

                    continue

                distance_m = (
                    distance_lengths.get(
                        destination_node
                    )
                )

                if distance_m is not None:
                    distance_matrix[
                        origin_index,
                        destination_index,
                    ] = (
                        float(
                            distance_m
                        )
                        / 1000.0
                    )

                travel_seconds = (
                    travel_time_lengths.get(
                        destination_node
                    )
                )

                if travel_seconds is not None:
                    travel_time_matrix[
                        origin_index,
                        destination_index,
                    ] = (
                        float(
                            travel_seconds
                        )
                        / 60.0
                    )

        return RoadMatrixResult(
            node_ids=normalized_nodes,
            distance_km=distance_matrix,
            travel_time_minutes=(
                travel_time_matrix
            ),
        )

    def _get_distance_lengths(
        self,
        origin_node: int,
    ) -> dict:
        """
        Get all shortest road distances from one origin.

        Cached after first calculation.
        """

        if (
            origin_node
            not in self._distance_cache
        ):
            self._distance_cache[
                origin_node
            ] = (
                nx.single_source_dijkstra_path_length(
                    self.graph,
                    source=origin_node,
                    weight="length",
                )
            )

        return self._distance_cache[
            origin_node
        ]

    def _get_travel_time_lengths(
        self,
        origin_node: int,
    ) -> dict:
        """
        Get all shortest estimated travel times
        from one origin.

        Cached after first calculation.
        """

        if (
            origin_node
            not in self._travel_time_cache
        ):
            self._travel_time_cache[
                origin_node
            ] = (
                nx.single_source_dijkstra_path_length(
                    self.graph,
                    source=origin_node,
                    weight="travel_time",
                )
            )

        return self._travel_time_cache[
            origin_node
        ]

    def _validate_nodes(
        self,
        node_ids: tuple[int, ...],
    ) -> None:
        missing_nodes = [
            node_id
            for node_id in node_ids
            if node_id
            not in self.graph
        ]

        if missing_nodes:
            preview = (
                missing_nodes[
                    :5
                ]
            )

            raise ValueError(
                "Road graph does not contain "
                f"node(s): {preview}"
            )

    def _validate_travel_time_data(
        self,
    ) -> None:
        """
        Ensure road edges contain estimated travel time.

        The GraphML created in Phase 5B should already
        contain this attribute.
        """

        for (
            _,
            _,
            edge_data,
        ) in self.graph.edges(
            data=True
        ):
            if (
                "travel_time"
                not in edge_data
            ):
                raise ValueError(
                    "Road graph does not contain "
                    "travel_time on every edge. "
                    "Run add_estimated_travel_times() "
                    "before building the matrix."
                )

    def clear_cache(
        self,
    ) -> None:
        """
        Clear cached Dijkstra results.
        """

        self._distance_cache.clear()

        self._travel_time_cache.clear()