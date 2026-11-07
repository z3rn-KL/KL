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

import ast
from pathlib import Path

from api.config import (
    DATASET_BENCHMARK_PATH,
    DQN_ROUTE_PATH,
    FINAL_VALIDATION_RESULTS_PATH,
    MAX_LIVE_ROUTE_DELIVERIES,
    PROPOSED_RL_BASELINE_ROUTE_RECOVERY_PATH,
    Q_LEARNING_ROUTE_PATH,
    REPRODUCIBILITY_VALIDATION_RESULTS_PATH,
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

RL_VERIFIED_WORKLOAD_IDS = [1, 29, 42, 62, 71, 90]


def resolve_rl_empirical_route(
    algorithm: str,
    workload_id: int,
) -> dict:
    """
    Resolve empirical route sequence and provenance for Q-Learning and SARSA.
    Supports Workload #1 (single route CSV), #29, #42, #62 (Route Recovery),
    and #71, #90 (Reproducibility Validation).
    """
    if workload_id == 1:
        route_path = {
            "q_learning": Q_LEARNING_ROUTE_PATH,
            "sarsa": SARSA_ROUTE_PATH,
            "dqn": DQN_ROUTE_PATH,
        }.get(algorithm)

        if not route_path or not Path(route_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Single-workload route file for {algorithm} not found.",
            )

        route_df = pd.read_csv(route_path).sort_values("sequence")
        delivery_order = route_df["delivery_id"].astype(str).tolist()

        return {
            "delivery_order": delivery_order,
            "source": "precomputed_thesis_route",
            "route_type": "original_precomputed",
            "provenance": "Tuyến đơn Workload #1 gốc (lưu trực tiếp)",
            "data_status": "exact_verified",
        }

    if algorithm not in {"q_learning", "sarsa"}:
        raise HTTPException(
            status_code=404,
            detail=f"Only single-workload route (workload_id=1) is available for {algorithm}.",
        )

    # Source A: proposed_rl_baseline_route_recovery.csv (WL #29, #42, #62)
    if workload_id in {29, 42, 62}:
        if not Path(PROPOSED_RL_BASELINE_ROUTE_RECOVERY_PATH).exists():
            raise HTTPException(
                status_code=500,
                detail="Route recovery baseline results file is missing.",
            )

        recovery_df = pd.read_csv(PROPOSED_RL_BASELINE_ROUTE_RECOVERY_PATH)
        matched = recovery_df[
            (recovery_df["algorithm"] == algorithm)
            & (recovery_df["workload_id"] == workload_id)
        ]

        if matched.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No route recovery record for {algorithm} on WL #{workload_id}.",
            )

        row = matched.iloc[0]
        raw_order = row["rerun_delivery_order"]
        try:
            delivery_order = [str(x) for x in ast.literal_eval(str(raw_order))]
        except Exception:
            delivery_order = [
                x.strip()
                for x in str(raw_order)
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .split(",")
                if x.strip()
            ]

        provenance = "Tuyến khôi phục đã kiểm chứng từ Route Recovery"
        if workload_id == 29:
            provenance += " (đối chiếu trùng khớp 100% với Reproducibility Validation)"

        return {
            "delivery_order": delivery_order,
            "source": "proposed_rl_baseline_route_recovery",
            "route_type": "validated_recovered",
            "provenance": provenance,
            "data_status": "verified_exact",
        }

    # Source B: reproducibility_validation_results.csv (WL #71, #90)
    if workload_id in {71, 90}:
        if not Path(REPRODUCIBILITY_VALIDATION_RESULTS_PATH).exists():
            raise HTTPException(
                status_code=500,
                detail="Reproducibility validation results file is missing.",
            )

        repro_df = pd.read_csv(REPRODUCIBILITY_VALIDATION_RESULTS_PATH)
        matched = repro_df[
            (repro_df["algorithm"] == algorithm)
            & (repro_df["workload_id"] == workload_id)
        ]

        if matched.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No reproducibility record for {algorithm} on WL #{workload_id}.",
            )

        row = matched.iloc[0]
        raw_order = str(row["rerun_delivery_order"])
        delivery_order = [x.strip() for x in raw_order.split("->") if x.strip()]

        return {
            "delivery_order": delivery_order,
            "source": "reproducibility_validation_results",
            "route_type": "validated_recovered",
            "provenance": "Tuyến khôi phục đã kiểm chứng từ Reproducibility Validation",
            "data_status": "exact_core_match",
        }

    raise HTTPException(
        status_code=404,
        detail=(
            f"No verified route data available for {algorithm} on workload_id={workload_id}. "
            f"Supported workloads: [1, 29, 42, 62, 71, 90]."
        ),
    )


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
            "available_workload_ids": RL_VERIFIED_WORKLOAD_IDS,
            "workload_provenance": {
                "1": "original_precomputed (q_learning_single_workload_route)",
                "29": "validated_recovered (proposed_rl_baseline_route_recovery & reproducibility_validation)",
                "42": "validated_recovered (proposed_rl_baseline_route_recovery)",
                "62": "validated_recovered (proposed_rl_baseline_route_recovery)",
                "71": "validated_recovered (reproducibility_validation_results)",
                "90": "validated_recovered (reproducibility_validation_results)",
            },
        },

        "sarsa": {
            "available_workload_ids": RL_VERIFIED_WORKLOAD_IDS,
            "workload_provenance": {
                "1": "original_precomputed (sarsa_single_workload_route)",
                "29": "validated_recovered (proposed_rl_baseline_route_recovery & reproducibility_validation)",
                "42": "validated_recovered (proposed_rl_baseline_route_recovery)",
                "62": "validated_recovered (proposed_rl_baseline_route_recovery)",
                "71": "validated_recovered (reproducibility_validation_results)",
                "90": "validated_recovered (reproducibility_validation_results)",
            },
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
                "RL routes are precomputed and verified thesis "
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

        if algorithm == "dqn" and resolved_workload_id != 1:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Only the saved "
                    "single-workload route "
                    "(workload_id=1) is "
                    "available for dqn."
                ),
            )

        rl_meta = resolve_rl_empirical_route(
            algorithm,
            resolved_workload_id,
        )
        delivery_order = rl_meta["delivery_order"]

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

        source_name = rl_meta["source"]
        route_type = rl_meta["route_type"]
        provenance = rl_meta["provenance"]
        data_status = rl_meta["data_status"]

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

        source_name = "final_kmeans_guided_dqn_validation"
        route_type = "original_precomputed"
        provenance = "Tuyến kiểm chứng K-Means-guided DQN chính thức"
        data_status = "verified_exact"

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
            source_name,

        "route_type":
            route_type,

        "provenance":
            provenance,

        "data_status":
            data_status,

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