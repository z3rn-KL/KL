import math

from datetime import (
    datetime,
)

from typing import (
    Literal,
)

import networkx as nx

import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from pydantic import (
    BaseModel,
    Field,
)

import api.dependencies as deps

from api.config import (
    DATASET_BENCHMARK_PATH,
    DQN_ROUTE_PATH,
    FINAL_VALIDATION_RESULTS_PATH,
    MAX_LIVE_ROUTE_DELIVERIES,
    Q_LEARNING_ROUTE_PATH,
    SARSA_ROUTE_PATH,
)

from api.models import (
    OperatingCostConfigRequest,
)

from evaluation.operating_cost import (
    OperatingCostEvaluator,
)


router = APIRouter(
    prefix="/api/routing",
    tags=["Routing"],
)


class LiveRouteRequest(
    BaseModel
):
    algorithm: Literal[
        "nearest_neighbor",
        "clarke_wright",
    ]

    delivery_ids: list[str] = Field(
        min_length=1,
        max_length=(
            MAX_LIVE_ROUTE_DELIVERIES
        ),
    )

    metric: Literal[
        "distance",
        "travel_time",
    ] = "distance"

    start_time: (
        datetime
        | None
    ) = None

    return_to_depot: bool = True

    cost_config: (
        OperatingCostConfigRequest
        | None
    ) = None


def resolve_cost_config(
    request_config: (
        OperatingCostConfigRequest
        | None
    ),
):
    if request_config is None:
        return (
            deps
            .get_default_cost_config()
        )

    return (
        request_config
        .to_domain()
    )


def json_safe(
    value,
):
    if value is None:
        return None

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
        ),
    ):
        return (
            pd.Timestamp(
                value
            )
            .isoformat()
        )

    try:
        if pd.isna(
            value
        ):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except Exception:
            pass

    return value


def row_to_dict(
    row: pd.Series,
) -> dict:
    return {
        str(
            column
        ):
            json_safe(
                value
            )

        for (
            column,
            value,
        )
        in row.items()
    }


def numeric_or_zero(
    value,
) -> float:
    if value is None:
        return 0.0

    try:
        if pd.isna(
            value
        ):
            return 0.0
    except (
        TypeError,
        ValueError,
    ):
        pass

    return float(
        value
    )


def reference_cost(
    reference: dict,
    config,
) -> dict:
    evaluator = (
        OperatingCostEvaluator(
            config
        )
    )

    result = evaluator.evaluate(
        total_distance_km=float(
            reference[
                "total_distance_km"
            ]
        ),
        total_travel_time_minutes=float(
            reference[
                "total_travel_time_minutes"
            ]
        ),
        late_deliveries=int(
            numeric_or_zero(
                reference.get(
                    "late_deliveries"
                )
            )
        ),
        total_lateness_minutes=(
            numeric_or_zero(
                reference.get(
                    "total_lateness_minutes"
                )
            )
        ),
    )

    return {
        "total_estimated_cost":
            result.total_estimated_cost,

        "cost_unit":
            result.cost_unit,

        "distance_cost":
            result.distance_cost,

        "travel_time_cost":
            result.travel_time_cost,

        "late_delivery_cost":
            result.late_delivery_cost,

        "lateness_cost":
            result.lateness_cost,
    }


# ============================================================
# LIVE HEURISTIC ROUTING
# ============================================================

@router.post(
    "/optimize"
)
def optimize_route(
    request: LiveRouteRequest,
) -> dict:
    """
    Run road-aware live routing.

    Supported live algorithms:
    - Nearest Neighbor
    - Clarke-Wright

    RL algorithms are intentionally not trained
    inside HTTP requests.
    """

    if (
        request.algorithm
        == "clarke_wright"
        and request.metric
        != "distance"
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Clarke-Wright in this thesis "
                "is distance-based; metric must "
                "be 'distance'."
            ),
        )

    try:
        (
            deliveries,
            delivery_nodes,
        ) = (
            deps.resolve_deliveries(
                request.delivery_ids
            )
        )

        depot = (
            deps.get_depot_context()
        )

        (
            start_time,
            start_time_source,
        ) = (
            deps
            .resolve_route_start_time(
                deliveries=deliveries,
                delivery_ids=(
                    request.delivery_ids
                ),
                explicit_start_time=(
                    request.start_time
                ),
            )
        )

        if (
            request.algorithm
            == "nearest_neighbor"
        ):
            algorithm_result = (
                deps
                .get_nearest_neighbor_router(
                    request.metric
                )
                .build_route(
                    depot_node=(
                        depot.road_node
                    ),
                    delivery_ids=(
                        request.delivery_ids
                    ),
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    return_to_depot=(
                        request
                        .return_to_depot
                    ),
                )
            )

            algorithm_details = {
                "metric":
                    request.metric,

                "raw_total_distance_km":
                    float(
                        algorithm_result
                        .total_distance_km
                    ),

                "raw_total_travel_time_minutes":
                    float(
                        algorithm_result
                        .total_travel_time_minutes
                    ),
            }

        else:
            algorithm_result = (
                deps
                .get_clarke_wright_router()
                .build_route(
                    depot_node=(
                        depot.road_node
                    ),
                    delivery_ids=(
                        request.delivery_ids
                    ),
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    return_to_depot=(
                        request
                        .return_to_depot
                    ),
                )
            )

            total_savings = float(
                algorithm_result
                .total_savings_km
            )

            algorithm_details = {
                "merge_count":
                    int(
                        algorithm_result
                        .merge_count
                    ),

                "initial_separate_distance_km":
                    float(
                        algorithm_result
                        .initial_separate_distance_km
                    ),

                "total_savings_km":
                    (
                        total_savings
                        if math.isfinite(
                            total_savings
                        )
                        else None
                    ),

                "raw_total_distance_km":
                    float(
                        algorithm_result
                        .total_distance_km
                    ),

                "raw_total_travel_time_minutes":
                    float(
                        algorithm_result
                        .total_travel_time_minutes
                    ),
            }

        cost_config = (
            resolve_cost_config(
                request.cost_config
            )
        )

        evaluation = (
            deps
            .get_route_evaluation_service(
                cost_config
            )
            .evaluate(
                algorithm=(
                    request.algorithm
                ),
                depot_node=(
                    depot.road_node
                ),
                deliveries=deliveries,
                delivery_nodes=(
                    delivery_nodes
                ),
                delivery_order=(
                    algorithm_result
                    .delivery_order
                ),
                start_time=(
                    start_time
                ),
                return_to_depot=(
                    request
                    .return_to_depot
                ),
            )
        )

    except (
        ValueError,
        nx.NetworkXException,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(
                exc
            ),
        ) from exc

    return {
        "source":
            "live_heuristic",

        "online_training":
            False,

        "algorithm":
            request.algorithm,

        "start_time_source":
            start_time_source,

        "depot": {
            "depot_id":
                depot.depot_id,

            "name":
                depot.name,

            "latitude":
                depot.latitude,

            "longitude":
                depot.longitude,

            "road_node":
                depot.road_node,
        },

        "algorithm_details":
            algorithm_details,

        "evaluation":
            evaluation.to_dict(),

        "cost_note":
            (
                "Default cost coefficients come "
                "from the thesis scenario and are "
                "an operational proxy, not verified "
                "XeDu accounting data."
            ),
    }


# ============================================================
# PRECOMPUTED RL ROUTES
# ============================================================

@router.get(
    "/precomputed"
)
def precomputed_route_catalog(
) -> dict:
    guided = pd.read_csv(
        FINAL_VALIDATION_RESULTS_PATH
    )

    guided_ids = sorted(
        pd.to_numeric(
            guided[
                "workload_id"
            ],
            errors="raise",
        )
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    return {
        "q_learning": {
            "available_workload_ids": [
                1
            ]
        },

        "sarsa": {
            "available_workload_ids": [
                1
            ]
        },

        "dqn": {
            "available_workload_ids": [
                1
            ]
        },

        "kmeans_guided_dqn": {
            "available_workload_ids":
                guided_ids
        },

        "note":
            (
                "RL routes are precomputed thesis "
                "artifacts. The API does not train "
                "5000 episodes inside an HTTP request."
            ),
    }


@router.get(
    "/precomputed/{algorithm}"
)
def get_precomputed_route(
    algorithm: Literal[
        "q_learning",
        "sarsa",
        "dqn",
        "kmeans_guided_dqn",
    ],
    workload_id: (
        int
        | None
    ) = Query(
        default=None,
        ge=1,
    ),
    include_geometry: bool = True,
) -> dict:
    config = (
        deps
        .get_default_cost_config()
    )

    if algorithm in {
        "q_learning",
        "sarsa",
        "dqn",
    }:
        resolved_workload_id = (
            1
            if workload_id is None
            else int(
                workload_id
            )
        )

        if resolved_workload_id != 1:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Only the saved "
                    "single-workload route "
                    "(workload_id=1) is "
                    f"available for {algorithm}."
                ),
            )

        route_path = {
            "q_learning":
                Q_LEARNING_ROUTE_PATH,

            "sarsa":
                SARSA_ROUTE_PATH,

            "dqn":
                DQN_ROUTE_PATH,
        }[
            algorithm
        ]

        route_df = (
            pd.read_csv(
                route_path
            )
            .sort_values(
                "sequence"
            )
        )

        delivery_order = (
            route_df[
                "delivery_id"
            ]
            .astype(
                str
            )
            .tolist()
        )

        benchmark = pd.read_csv(
            DATASET_BENCHMARK_PATH
        )

        matched = benchmark[
            (
                benchmark[
                    "workload_id"
                ]
                == resolved_workload_id
            )
            &
            (
                benchmark[
                    "algorithm"
                ]
                == algorithm
            )
        ]

        if (
            len(
                matched
            )
            != 1
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Reference benchmark "
                    "row is missing."
                ),
            )

        reference_row = (
            matched.iloc[
                0
            ]
        )

        workload_row = (
            deps.get_workload_row(
                resolved_workload_id
            )
        )

        start_time = (
            pd.Timestamp(
                workload_row[
                    "wave_end"
                ]
            )
            .to_pydatetime()
        )

    else:
        if workload_id is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "workload_id is required "
                    "for kmeans_guided_dqn."
                ),
            )

        guided = pd.read_csv(
            FINAL_VALIDATION_RESULTS_PATH
        )

        matched = guided[
            guided[
                "workload_id"
            ]
            == int(
                workload_id
            )
        ]

        if (
            len(
                matched
            )
            != 1
        ):
            raise HTTPException(
                status_code=404,
                detail=(
                    "No saved K-Means-guided "
                    "DQN route for "
                    f"workload_id={workload_id}."
                ),
            )

        reference_row = (
            matched.iloc[
                0
            ]
        )

        delivery_order = [
            item.strip()

            for item
            in str(
                reference_row[
                    "delivery_order"
                ]
            ).split(
                "->"
            )

            if item.strip()
        ]

        start_time = (
            pd.Timestamp(
                reference_row[
                    "wave_end"
                ]
            )
            .to_pydatetime()
        )

        resolved_workload_id = int(
            workload_id
        )

    reference = (
        row_to_dict(
            reference_row
        )
    )

    reference[
        "total_distance_km"
    ] = float(
        reference_row[
            "total_distance_km"
        ]
    )

    reference[
        "total_travel_time_minutes"
    ] = float(
        reference_row[
            "total_travel_time_minutes"
        ]
    )

    response = {
        "source":
            "precomputed_thesis_route",

        "online_training":
            False,

        "algorithm":
            algorithm,

        "workload_id":
            resolved_workload_id,

        "delivery_order":
            delivery_order,

        "reference_metrics":
            reference,

        "estimated_operating_cost_from_reference":
            reference_cost(
                reference,
                config,
            ),

        "evaluation":
            None,

        "cost_note":
            (
                "Cost is a scenario proxy. "
                "RL training is not executed "
                "during the request."
            ),
    }

    if include_geometry:
        try:
            (
                deliveries,
                nodes,
            ) = (
                deps.resolve_deliveries(
                    delivery_order
                )
            )

            depot = (
                deps
                .get_depot_context()
            )

            evaluation = (
                deps
                .get_route_evaluation_service(
                    config
                )
                .evaluate(
                    algorithm=algorithm,
                    depot_node=(
                        depot.road_node
                    ),
                    deliveries=(
                        deliveries
                    ),
                    delivery_nodes=(
                        nodes
                    ),
                    delivery_order=(
                        delivery_order
                    ),
                    start_time=(
                        start_time
                    ),
                    return_to_depot=True,
                )
            )

            response[
                "evaluation"
            ] = (
                evaluation.to_dict()
            )

        except (
            ValueError,
            nx.NetworkXException,
        ) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(
                    exc
                ),
            ) from exc

    return response