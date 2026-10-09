from dataclasses import (
    asdict,
    dataclass,
)

from datetime import (
    datetime,
    timedelta,
)

from typing import (
    Any,
)

from evaluation.operating_cost import (
    OperatingCostBreakdown,
    OperatingCostConfig,
    OperatingCostEvaluator,
)

from routing import (
    RoadNetworkService,
)


@dataclass(frozen=True)
class RouteEvaluationBundle:
    """
    Combined route evaluation for the API layer.

    This wrapper intentionally remains compatible with both:
    - the real RouteEvaluator;
    - older/fake evaluators used by automated tests.
    """

    algorithm: str

    number_of_deliveries: int

    deliveries_with_deadline: int

    deliveries_without_deadline: int

    delivery_order: tuple[
        str,
        ...
    ]

    total_distance_km: float

    total_travel_time_minutes: float

    on_time_deliveries: int

    late_deliveries: int

    on_time_rate: (
        float
        | None
    )

    total_lateness_minutes: float

    mean_lateness_minutes: float

    max_lateness_minutes: float

    start_time: datetime

    end_time: datetime

    returned_to_depot: bool

    estimated_operating_cost: float

    operating_cost_breakdown: (
        OperatingCostBreakdown
    )

    route_geometry: tuple[
        tuple[
            float,
            float,
        ],
        ...
    ]

    def to_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return asdict(
            self
        )


class RouteEvaluationService:
    """
    High-level evaluation wrapper for the API.

    Responsibilities
    ----------------
    1. Reuse the existing RouteEvaluator.
    2. Add estimated operating cost.
    3. Add road-network geometry.
    4. Preserve compatibility with existing tests and benchmarks.

    The original RouteEvaluator is NOT modified.
    """

    def __init__(
        self,
        route_evaluator,
        road_network: (
            RoadNetworkService
        ),
        cost_config: (
            OperatingCostConfig
        ),
    ) -> None:
        self.route_evaluator = (
            route_evaluator
        )

        self.road_network = (
            road_network
        )

        self.cost_evaluator = (
            OperatingCostEvaluator(
                cost_config
            )
        )

    def evaluate(
        self,
        algorithm: str,
        depot_node: int,
        deliveries,
        delivery_nodes,
        delivery_order,
        start_time: datetime,
        return_to_depot: bool = True,
        geometry_weight: str = "length",
    ) -> RouteEvaluationBundle:
        """
        Evaluate one complete route.
        """

        self._validate_route_inputs(
            deliveries=deliveries,
            delivery_nodes=(
                delivery_nodes
            ),
            delivery_order=(
                delivery_order
            ),
        )

        evaluation = (
            self.route_evaluator
            .evaluate(
                algorithm=algorithm,
                depot_node=depot_node,
                deliveries=deliveries,
                delivery_nodes=(
                    delivery_nodes
                ),
                delivery_order=(
                    delivery_order
                ),
                start_time=start_time,
                return_to_depot=(
                    return_to_depot
                ),
            )
        )

        # ====================================================
        # CORE VALUES
        # ====================================================

        number_of_deliveries = int(
            getattr(
                evaluation,
                "number_of_deliveries",
                len(
                    deliveries
                ),
            )
        )

        on_time_deliveries = int(
            getattr(
                evaluation,
                "on_time_deliveries",
                0,
            )
        )

        late_deliveries = int(
            getattr(
                evaluation,
                "late_deliveries",
                0,
            )
        )

        total_lateness_minutes = float(
            getattr(
                evaluation,
                "total_lateness_minutes",
                0.0,
            )
        )

        max_lateness_minutes = float(
            getattr(
                evaluation,
                "max_lateness_minutes",
                0.0,
            )
        )

        # ====================================================
        # BACKWARD-COMPATIBLE DEADLINE METRICS
        # ====================================================

        deliveries_with_deadline = int(
            getattr(
                evaluation,
                "deliveries_with_deadline",
                (
                    on_time_deliveries
                    + late_deliveries
                ),
            )
        )

        deliveries_without_deadline = int(
            getattr(
                evaluation,
                "deliveries_without_deadline",
                max(
                    0,
                    number_of_deliveries
                    - deliveries_with_deadline,
                ),
            )
        )

        raw_on_time_rate = getattr(
            evaluation,
            "on_time_rate",
            None,
        )

        if raw_on_time_rate is None:
            if (
                deliveries_with_deadline
                > 0
            ):
                on_time_rate = (
                    on_time_deliveries
                    / deliveries_with_deadline
                )

            else:
                on_time_rate = None

        else:
            on_time_rate = float(
                raw_on_time_rate
            )

        raw_mean_lateness = getattr(
            evaluation,
            "mean_lateness_minutes",
            None,
        )

        if raw_mean_lateness is None:
            if late_deliveries > 0:
                mean_lateness_minutes = (
                    total_lateness_minutes
                    / late_deliveries
                )

            else:
                mean_lateness_minutes = 0.0

        else:
            mean_lateness_minutes = float(
                raw_mean_lateness
            )

        # ====================================================
        # BACKWARD-COMPATIBLE TIME VALUES
        # ====================================================

        evaluation_start_time = getattr(
            evaluation,
            "start_time",
            start_time,
        )

        evaluation_end_time = getattr(
            evaluation,
            "end_time",
            None,
        )

        if evaluation_end_time is None:
            evaluation_end_time = (
                evaluation_start_time
                + timedelta(
                    minutes=float(
                        evaluation
                        .total_travel_time_minutes
                    )
                )
            )

        returned_to_depot = bool(
            getattr(
                evaluation,
                "returned_to_depot",
                return_to_depot,
            )
        )

        # ====================================================
        # OPERATING COST
        # ====================================================

        operating_cost = (
            self.cost_evaluator
            .evaluate(
                total_distance_km=float(
                    evaluation
                    .total_distance_km
                ),
                total_travel_time_minutes=float(
                    evaluation
                    .total_travel_time_minutes
                ),
                late_deliveries=(
                    late_deliveries
                ),
                total_lateness_minutes=(
                    total_lateness_minutes
                ),
            )
        )

        # ====================================================
        # ROUTE GEOMETRY
        # ====================================================

        route_geometry = (
            self._build_route_geometry(
                depot_node=depot_node,
                deliveries=deliveries,
                delivery_nodes=(
                    delivery_nodes
                ),
                delivery_order=(
                    delivery_order
                ),
                return_to_depot=(
                    return_to_depot
                ),
                weight=(
                    geometry_weight
                ),
            )
        )

        normalized_order = tuple(
            str(
                delivery_id
            )
            for delivery_id
            in delivery_order
        )

        # ====================================================
        # FINAL BUNDLE
        # ====================================================

        return RouteEvaluationBundle(
            algorithm=str(
                getattr(
                    evaluation,
                    "algorithm",
                    algorithm,
                )
            ),

            number_of_deliveries=(
                number_of_deliveries
            ),

            deliveries_with_deadline=(
                deliveries_with_deadline
            ),

            deliveries_without_deadline=(
                deliveries_without_deadline
            ),

            delivery_order=(
                normalized_order
            ),

            total_distance_km=float(
                evaluation
                .total_distance_km
            ),

            total_travel_time_minutes=float(
                evaluation
                .total_travel_time_minutes
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
                total_lateness_minutes
            ),

            mean_lateness_minutes=(
                mean_lateness_minutes
            ),

            max_lateness_minutes=(
                max_lateness_minutes
            ),

            start_time=(
                evaluation_start_time
            ),

            end_time=(
                evaluation_end_time
            ),

            returned_to_depot=(
                returned_to_depot
            ),

            estimated_operating_cost=float(
                operating_cost
                .total_estimated_cost
            ),

            operating_cost_breakdown=(
                operating_cost
            ),

            route_geometry=(
                route_geometry
            ),
        )

    def _build_route_geometry(
        self,
        depot_node: int,
        deliveries,
        delivery_nodes,
        delivery_order,
        return_to_depot: bool,
        weight: str,
    ) -> tuple[
        tuple[
            float,
            float,
        ],
        ...
    ]:
        """
        Build one continuous road-network polyline.

        Route:
            depot
            -> delivery 1
            -> delivery 2
            -> ...
            -> depot

        Coordinates:
            (latitude, longitude)
        """

        node_lookup = {}

        for (
            delivery,
            road_node,
        ) in zip(
            deliveries,
            delivery_nodes,
            strict=True,
        ):
            delivery_id = str(
                delivery.delivery_id
            )

            if (
                delivery_id
                in node_lookup
            ):
                raise ValueError(
                    "Delivery IDs must be "
                    "unique when building "
                    "route geometry."
                )

            node_lookup[
                delivery_id
            ] = int(
                road_node
            )

        normalized_order = [
            str(
                delivery_id
            )
            for delivery_id
            in delivery_order
        ]

        route_nodes = [
            int(
                depot_node
            )
        ]

        for delivery_id in (
            normalized_order
        ):
            if (
                delivery_id
                not in node_lookup
            ):
                raise ValueError(
                    "delivery_order contains "
                    "an unknown delivery ID: "
                    f"{delivery_id}"
                )

            route_nodes.append(
                node_lookup[
                    delivery_id
                ]
            )

        if return_to_depot:
            route_nodes.append(
                int(
                    depot_node
                )
            )

        geometry: list[
            tuple[
                float,
                float,
            ]
        ] = []

        for index in range(
            len(
                route_nodes
            )
            - 1
        ):
            origin_node = (
                route_nodes[
                    index
                ]
            )

            destination_node = (
                route_nodes[
                    index + 1
                ]
            )

            segment = (
                self.road_network
                .shortest_path_coordinates(
                    origin_node=(
                        origin_node
                    ),
                    destination_node=(
                        destination_node
                    ),
                    weight=weight,
                )
            )

            if not segment:
                raise ValueError(
                    "Road-network segment "
                    "returned empty geometry."
                )

            normalized_segment = [
                (
                    float(
                        latitude
                    ),
                    float(
                        longitude
                    ),
                )
                for (
                    latitude,
                    longitude,
                )
                in segment
            ]

            if not geometry:
                geometry.extend(
                    normalized_segment
                )

                continue

            if (
                geometry[
                    -1
                ]
                == normalized_segment[
                    0
                ]
            ):
                geometry.extend(
                    normalized_segment[
                        1:
                    ]
                )

            else:
                geometry.extend(
                    normalized_segment
                )

        return tuple(
            geometry
        )

    @staticmethod
    def _validate_route_inputs(
        deliveries,
        delivery_nodes,
        delivery_order,
    ) -> None:
        delivery_count = len(
            deliveries
        )

        if delivery_count == 0:
            raise ValueError(
                "deliveries cannot be empty."
            )

        if (
            len(
                delivery_nodes
            )
            != delivery_count
        ):
            raise ValueError(
                "delivery_nodes must have "
                "the same length as deliveries."
            )

        delivery_ids = [
            str(
                delivery.delivery_id
            )
            for delivery
            in deliveries
        ]

        if (
            len(
                delivery_ids
            )
            != len(
                set(
                    delivery_ids
                )
            )
        ):
            raise ValueError(
                "Delivery IDs must be unique."
            )

        normalized_order = [
            str(
                delivery_id
            )
            for delivery_id
            in delivery_order
        ]

        if (
            len(
                normalized_order
            )
            != delivery_count
        ):
            raise ValueError(
                "delivery_order must contain "
                "every delivery exactly once."
            )

        if (
            len(
                normalized_order
            )
            != len(
                set(
                    normalized_order
                )
            )
        ):
            raise ValueError(
                "delivery_order contains "
                "duplicate delivery IDs."
            )

        if (
            set(
                normalized_order
            )
            != set(
                delivery_ids
            )
        ):
            raise ValueError(
                "delivery_order does not match "
                "the delivery set."
            )