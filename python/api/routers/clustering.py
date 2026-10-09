from functools import lru_cache

import numpy as np
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

from api.config import (
    KMEANS_RESULT_PATH,
)

from clustering import (
    KMeansDeliveryClusterer,
    evaluate_clustering,
)

from clustering.dbscan_clusterer import (
    DBSCANDeliveryClusterer,
)

from domain import (
    Delivery,
)


router = APIRouter(
    prefix="/api/clustering",
    tags=["Clustering"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class DeliveryPointRequest(
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


class KMeansRunRequest(
    BaseModel
):
    deliveries: list[
        DeliveryPointRequest
    ] = Field(
        min_length=1
    )

    n_clusters: int = Field(
        default=10,
        ge=1,
    )

    random_state: int = 42

    n_init: int = Field(
        default=20,
        ge=1,
    )


class DBSCANRunRequest(
    BaseModel
):
    deliveries: list[
        DeliveryPointRequest
    ] = Field(
        min_length=1
    )

    eps_km: float = Field(
        default=0.5,
        gt=0.0,
    )

    min_samples: int = Field(
        default=5,
        ge=1,
    )


# ============================================================
# PRECOMPUTED K-MEANS RESULT
# ============================================================

@lru_cache(maxsize=1)
def load_precomputed_kmeans(
) -> pd.DataFrame:
    if not KMEANS_RESULT_PATH.exists():
        raise FileNotFoundError(
            "K-Means result file not found: "
            f"{KMEANS_RESULT_PATH}"
        )

    dataframe = pd.read_csv(
        KMEANS_RESULT_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "K-Means result file is empty."
        )

    if (
        "cluster_id"
        not in dataframe.columns
    ):
        raise ValueError(
            "K-Means result file does not "
            "contain cluster_id."
        )

    return dataframe


def find_first_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    return None


def json_safe(
    value,
):
    if value is None:
        return None

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        if np.isnan(
            value
        ):
            return None

        return float(
            value
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

    return value


def serialize_precomputed_points(
    dataframe: pd.DataFrame,
) -> list[dict]:
    id_column = find_first_column(
        dataframe,
        [
            "id",
            "delivery_id",
            "deliveryId",
        ],
    )

    latitude_column = find_first_column(
        dataframe,
        [
            "receiverLat",
            "latitude",
            "lat",
        ],
    )

    longitude_column = find_first_column(
        dataframe,
        [
            "receiverLng",
            "longitude",
            "lng",
            "lon",
        ],
    )

    items = []

    for _, row in dataframe.iterrows():
        item = {
            "cluster_id": int(
                row[
                    "cluster_id"
                ]
            ),
        }

        if id_column is not None:
            item[
                "delivery_id"
            ] = str(
                row[
                    id_column
                ]
            )

        if latitude_column is not None:
            item[
                "latitude"
            ] = json_safe(
                row[
                    latitude_column
                ]
            )

        if longitude_column is not None:
            item[
                "longitude"
            ] = json_safe(
                row[
                    longitude_column
                ]
            )

        items.append(
            item
        )

    return items


# ============================================================
# DOMAIN CONVERSION
# ============================================================

def request_to_deliveries(
    points: list[
        DeliveryPointRequest
    ],
) -> list[Delivery]:
    identifiers = [
        point.delivery_id.strip()
        for point in points
    ]

    if any(
        not identifier
        for identifier in identifiers
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

def serialize_clusters(
    clusters,
) -> list[dict]:
    result = []

    for cluster in clusters:
        result.append(
            {
                "cluster_id":
                    int(
                        cluster.cluster_id
                    ),

                "size":
                    int(
                        cluster.size()
                    ),

                "total_weight":
                    float(
                        cluster.total_weight()
                    ),

                "centroid": {
                    "latitude":
                        cluster
                        .centroid_latitude,

                    "longitude":
                        cluster
                        .centroid_longitude,
                },

                "delivery_ids": [
                    str(
                        delivery.delivery_id
                    )
                    for delivery
                    in cluster.deliveries
                ],
            }
        )

    return result


def serialize_assignments(
    deliveries: list[Delivery],
    labels,
) -> list[dict]:
    return [
        {
            "delivery_id":
                str(
                    delivery.delivery_id
                ),

            "latitude":
                float(
                    delivery.latitude
                ),

            "longitude":
                float(
                    delivery.longitude
                ),

            "cluster_id":
                int(
                    label
                ),
        }
        for (
            delivery,
            label,
        )
        in zip(
            deliveries,
            labels,
            strict=True,
        )
    ]


# ============================================================
# GET PRECOMPUTED K-MEANS
# ============================================================

@router.get(
    "/kmeans"
)
def get_precomputed_kmeans(
    offset: int = Query(
        default=0,
        ge=0,
    ),
    limit: int = Query(
        default=200,
        ge=1,
        le=500,
    ),
) -> dict:
    """
    Return the authoritative precomputed K-Means result.

    The thesis configuration uses K=10.
    """

    dataframe = (
        load_precomputed_kmeans()
    )

    cluster_sizes = (
        dataframe[
            "cluster_id"
        ]
        .value_counts()
        .sort_index()
    )

    total = int(
        len(
            dataframe
        )
    )

    page = dataframe.iloc[
        offset:
        offset + limit
    ]

    return {
        "algorithm":
            "kmeans",

        "source":
            "precomputed_thesis_result",

        "n_clusters":
            int(
                dataframe[
                    "cluster_id"
                ].nunique()
            ),

        "total_deliveries":
            total,

        "cluster_sizes": {
            str(
                int(
                    cluster_id
                )
            ): int(
                count
            )
            for (
                cluster_id,
                count,
            )
            in cluster_sizes.items()
        },

        "offset":
            int(
                offset
            ),

        "limit":
            int(
                limit
            ),

        "returned":
            int(
                len(
                    page
                )
            ),

        "items":
            serialize_precomputed_points(
                page
            ),
    }


# ============================================================
# RUN K-MEANS
# ============================================================

@router.post(
    "/kmeans/run"
)
def run_kmeans(
    request: KMeansRunRequest,
) -> dict:
    """
    Run K-Means on delivery coordinates supplied by the caller.

    Coordinates are internally projected to local x/y kilometres
    by the existing thesis clustering implementation.
    """

    try:
        deliveries = (
            request_to_deliveries(
                request.deliveries
            )
        )

        if (
            request.n_clusters
            > len(
                deliveries
            )
        ):
            raise ValueError(
                "n_clusters cannot be greater "
                "than the number of deliveries."
            )

        clusterer = (
            KMeansDeliveryClusterer(
                n_clusters=(
                    request.n_clusters
                ),
                random_state=(
                    request.random_state
                ),
                n_init=(
                    request.n_init
                ),
            )
        )

        result = clusterer.fit(
            deliveries
        )

        metrics = (
            evaluate_clustering(
                result
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
            "kmeans",

        "parameters": {
            "n_clusters":
                int(
                    request.n_clusters
                ),

            "random_state":
                int(
                    request.random_state
                ),

            "n_init":
                int(
                    request.n_init
                ),
        },

        "metrics": {
            "n_clusters":
                int(
                    metrics.n_clusters
                ),

            "inertia":
                float(
                    metrics.inertia
                ),

            "silhouette":
                (
                    None
                    if metrics.silhouette
                    is None
                    else float(
                        metrics.silhouette
                    )
                ),

            "min_cluster_size":
                int(
                    metrics.min_cluster_size
                ),

            "max_cluster_size":
                int(
                    metrics.max_cluster_size
                ),

            "mean_cluster_size":
                float(
                    metrics.mean_cluster_size
                ),

            "cluster_size_cv":
                float(
                    metrics.cluster_size_cv
                ),

            "total_weight":
                float(
                    metrics.total_weight_kg
                ),
        },

        "clusters":
            serialize_clusters(
                result.clusters
            ),

        "assignments":
            serialize_assignments(
                deliveries,
                result.labels,
            ),
    }


# ============================================================
# RUN DBSCAN
# ============================================================

@router.post(
    "/dbscan/run"
)
def run_dbscan(
    request: DBSCANRunRequest,
) -> dict:
    """
    Run DBSCAN on caller-supplied delivery coordinates.

    eps_km is expressed in kilometres.

    DBSCAN noise points receive cluster_id = -1.
    """

    try:
        deliveries = (
            request_to_deliveries(
                request.deliveries
            )
        )

        clusterer = (
            DBSCANDeliveryClusterer(
                eps_km=(
                    request.eps_km
                ),
                min_samples=(
                    request.min_samples
                ),
            )
        )

        result = clusterer.fit(
            deliveries
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
            "dbscan",

        "parameters": {
            "eps_km":
                float(
                    result.eps_km
                ),

            "min_samples":
                int(
                    result.min_samples
                ),
        },

        "metrics": {
            "n_clusters":
                int(
                    result.n_clusters
                ),

            "n_noise":
                int(
                    result.n_noise
                ),

            "noise_ratio":
                float(
                    result.noise_ratio
                ),
        },

        "clusters":
            serialize_clusters(
                result.clusters
            ),

        "noise_delivery_ids": [
            str(
                delivery.delivery_id
            )
            for delivery
            in result.noise_deliveries
        ],

        "assignments":
            serialize_assignments(
                deliveries,
                result.labels,
            ),
    }