from typing import Any

import numpy as np

from fastapi import (
    APIRouter,
    HTTPException,
)

from pydantic import (
    BaseModel,
    Field,
)

from clustering.adverse_deliveries import (
    AdverseDelivery,
    AdverseDeliveryReport,
    screen_adverse_deliveries,
)

from domain import (
    Delivery,
)


router = APIRouter(
    prefix="/api/adverse-deliveries",
    tags=["Adverse Deliveries"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class AdverseDeliveryPointRequest(
    BaseModel
):
    delivery_id: str = Field(
        min_length=1
    )

    latitude: float = Field(
        ge=-90.0,
        le=90.0,
    )

    longitude: float = Field(
        ge=-180.0,
        le=180.0,
    )

    weight: float | None = Field(
        default=None,
        ge=0.0,
    )


class AdverseDeliveryScreenRequest(
    BaseModel
):
    """
    Screen deliveries using a precomputed directed
    road-distance matrix.

    Matrix convention
    -----------------
    depot_included=True:

        index 0     = depot
        index 1..n  = deliveries in request order

        matrix shape = (n+1, n+1)

    depot_included=False:

        index 0..n-1 = deliveries in request order

        matrix shape = (n, n)

    DBSCAN labels
    -------------
    If supplied, labels must follow the same delivery order.

    label == -1 means DBSCAN spatial noise.
    """

    deliveries: list[
        AdverseDeliveryPointRequest
    ] = Field(
        min_length=1
    )

    distance_km: list[
        list[
            float
        ]
    ]

    dbscan_labels: list[int] | None = None

    isolation_multiplier: float = Field(
        default=3.0,
        gt=1.0,
    )

    detour_threshold_km: float | None = Field(
        default=None,
        ge=0.0,
    )

    depot_included: bool = True


# ============================================================
# CONVERSION
# ============================================================

def request_to_deliveries(
    points: list[
        AdverseDeliveryPointRequest
    ],
) -> list[Delivery]:
    identifiers = [
        point.delivery_id.strip()
        for point in points
    ]

    if any(
        not identifier
        for identifier
        in identifiers
    ):
        raise ValueError(
            "delivery_id cannot be empty."
        )

    if (
        len(
            identifiers
        )
        != len(
            set(
                identifiers
            )
        )
    ):
        raise ValueError(
            "Delivery IDs must be unique."
        )

    return [
        Delivery(
            delivery_id=(
                point.delivery_id.strip()
            ),
            latitude=float(
                point.latitude
            ),
            longitude=float(
                point.longitude
            ),
            weight=(
                None
                if point.weight is None
                else float(
                    point.weight
                )
            ),
        )
        for point
        in points
    ]


# ============================================================
# SERIALIZATION
# ============================================================

def serialize_entry(
    entry: AdverseDelivery,
) -> dict[str, Any]:
    return {
        "delivery_id":
            str(
                entry.delivery_id
            ),

        "flagged":
            bool(
                entry.reasons
            ),

        "dbscan_noise":
            bool(
                entry.dbscan_noise
            ),

        "nearest_peer_distance_km":
            (
                None
                if entry
                .nearest_peer_distance_km
                is None
                else float(
                    entry
                    .nearest_peer_distance_km
                )
            ),

        "isolation_ratio":
            (
                None
                if entry
                .isolation_ratio
                is None
                else float(
                    entry
                    .isolation_ratio
                )
            ),

        "insertion_detour_km":
            (
                None
                if entry
                .insertion_detour_km
                is None
                else float(
                    entry
                    .insertion_detour_km
                )
            ),

        "reasons": [
            str(
                reason
            )
            for reason
            in entry.reasons
        ],
    }


def serialize_report(
    report: AdverseDeliveryReport,
) -> dict:
    entries = [
        serialize_entry(
            entry
        )
        for entry
        in report.entries
    ]

    flagged = [
        serialize_entry(
            entry
        )
        for entry
        in report.flagged
    ]

    reason_counts: dict[
        str,
        int,
    ] = {}

    for entry in (
        report.flagged
    ):
        for reason in (
            entry.reasons
        ):
            reason_counts[
                reason
            ] = (
                reason_counts.get(
                    reason,
                    0,
                )
                + 1
            )

    return {
        "total_deliveries":
            len(
                report.entries
            ),

        "flagged_deliveries":
            len(
                report.flagged
            ),

        "flagged_ratio":
            (
                len(
                    report.flagged
                )
                / len(
                    report.entries
                )
                if report.entries
                else 0.0
            ),

        "parameters": {
            "isolation_multiplier":
                float(
                    report
                    .isolation_multiplier
                ),

            "detour_threshold_km":
                (
                    None
                    if report
                    .detour_threshold_km
                    is None
                    else float(
                        report
                        .detour_threshold_km
                    )
                ),
        },

        "reason_counts":
            reason_counts,

        "flagged":
            flagged,

        "entries":
            entries,

        "interpretation": {
            "dbscan_spatial_noise":
                (
                    "Delivery was labelled as "
                    "DBSCAN spatial noise."
                ),

            "road_isolation_candidate":
                (
                    "Nearest road-network peer "
                    "distance is unusually high "
                    "relative to the workload median."
                ),

            "unreachable_from_all_peers":
                (
                    "No finite directed road distance "
                    "to/from any delivery peer was found."
                ),

            "high_insertion_detour_candidate":
                (
                    "Best insertion detour proxy exceeds "
                    "the caller-supplied threshold."
                ),
        },

        "warning":
            (
                "Adverse-delivery flags are investigation "
                "candidates. insertion_detour_km is a route "
                "difficulty proxy, not an operating-cost estimate."
            ),
    }


# ============================================================
# SCREEN ENDPOINT
# ============================================================

@router.post(
    "/screen"
)
def screen_deliveries(
    request: AdverseDeliveryScreenRequest,
) -> dict:
    """
    Detect deliveries that may be spatially isolated
    or difficult to insert into a route.

    This endpoint delegates all screening logic to the
    existing thesis implementation:

        clustering.adverse_deliveries
        .screen_adverse_deliveries

    No new screening rule is introduced by the API.
    """

    try:
        deliveries = (
            request_to_deliveries(
                request.deliveries
            )
        )

        matrix = np.asarray(
            request.distance_km,
            dtype=float,
        )

        report = (
            screen_adverse_deliveries(
                deliveries=deliveries,
                distance_km=matrix,
                dbscan_labels=(
                    request.dbscan_labels
                ),
                isolation_multiplier=float(
                    request
                    .isolation_multiplier
                ),
                detour_threshold_km=(
                    request
                    .detour_threshold_km
                ),
                depot_included=(
                    request
                    .depot_included
                ),
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(
                exc
            ),
        ) from exc

    return {
        "algorithm":
            "adverse_delivery_screening",

        "distance_metric":
            "directed_road_distance_km",

        "depot_included":
            bool(
                request.depot_included
            ),

        **serialize_report(
            report
        ),
    }