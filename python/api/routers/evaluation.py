from datetime import (
    datetime,
)

import networkx as nx

from fastapi import (
    APIRouter,
    HTTPException,
)

from pydantic import (
    BaseModel,
    Field,
)

import api.dependencies as deps

from api.config import (
    MAX_LIVE_ROUTE_DELIVERIES,
)

from api.models import (
    OperatingCostConfigRequest,
)

from evaluation.operating_cost import (
    OperatingCostEvaluator,
)


router = APIRouter(
    prefix="/api/evaluation",
    tags=["Evaluation"],
)


class RouteEvaluationRequest(
    BaseModel
):
    delivery_ids: list[str] = Field(
        min_length=1,
        max_length=(
            MAX_LIVE_ROUTE_DELIVERIES
        ),
    )

    delivery_order: list[str] = Field(
        min_length=1,
        max_length=(
            MAX_LIVE_ROUTE_DELIVERIES
        ),
    )

    algorithm_label: str = Field(
        default="custom_route",
        min_length=1,
    )

    start_time: (
        datetime
        | None
    ) = None

    return_to_depot: bool = True

    cost_config: (
        OperatingCostConfigRequest
        | None
    ) = None


class CostEvaluationRequest(
    BaseModel
):
    total_distance_km: float = Field(
        ge=0.0
    )

    total_travel_time_minutes: float = Field(
        ge=0.0
    )

    late_deliveries: int = Field(
        default=0,
        ge=0,
    )

    total_lateness_minutes: float = Field(
        default=0.0,
        ge=0.0,
    )

    cost_config: (
        OperatingCostConfigRequest
        | None
    ) = None


def resolve_cost_config(
    value: (
        OperatingCostConfigRequest
        | None
    ),
):
    if value is None:
        return (
            deps
            .get_default_cost_config()
        )

    return value.to_domain()


@router.post(
    "/route"
)
def evaluate_route(
    request: RouteEvaluationRequest,
) -> dict:
    try:
        (
            deliveries,
            nodes,
        ) = (
            deps.resolve_deliveries(
                request.delivery_ids
            )
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

        service = (
            deps
            .get_route_evaluation_service(
                resolve_cost_config(
                    request.cost_config
                )
            )
        )

        depot = (
            deps
            .get_depot_context()
        )

        result = (
            service.evaluate(
                algorithm=(
                    request
                    .algorithm_label
                    .strip()
                ),
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
                    request
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
        "start_time_source":
            start_time_source,

        "evaluation":
            result.to_dict(),

        "cost_note":
            (
                "Default coefficients are "
                "scenario proxy values unless "
                "custom values are supplied."
            ),
    }


@router.post(
    "/cost"
)
def evaluate_cost(
    request: CostEvaluationRequest,
) -> dict:
    config = (
        resolve_cost_config(
            request.cost_config
        )
    )

    result = (
        OperatingCostEvaluator(
            config
        )
        .evaluate(
            total_distance_km=(
                request
                .total_distance_km
            ),
            total_travel_time_minutes=(
                request
                .total_travel_time_minutes
            ),
            late_deliveries=(
                request
                .late_deliveries
            ),
            total_lateness_minutes=(
                request
                .total_lateness_minutes
            ),
        )
    )

    return {
        "estimated":
            True,

        "actual_xedu_accounting_cost":
            False,

        "result": {
            "total_estimated_cost":
                result
                .total_estimated_cost,

            "cost_unit":
                result.cost_unit,

            "distance_cost":
                result.distance_cost,

            "travel_time_cost":
                result.travel_time_cost,

            "late_delivery_cost":
                result
                .late_delivery_cost,

            "lateness_cost":
                result.lateness_cost,
        },
    }