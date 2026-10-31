from functools import lru_cache

import numpy as np
import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from api.config import (
    ANONYMIZE_DELIVERIES,
    KMEANS_RESULT_PATH,
    XEDU_DATASET_PATH,
)


router = APIRouter(
    prefix="/api/deliveries",
    tags=["Deliveries"],
)


# ============================================================
# DATA LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_deliveries(
) -> pd.DataFrame:
    """
    Load the authoritative cleaned XeDu dataset.

    The dataframe is cached because this dataset is read-only
    for the current API stage.
    """

    if not XEDU_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{XEDU_DATASET_PATH}"
        )

    dataframe = pd.read_csv(
        XEDU_DATASET_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "Delivery dataset is empty."
        )

    return dataframe


# ============================================================
# JSON CONVERSION
# ============================================================

def json_safe_value(
    value,
):
    """
    Convert NumPy / Pandas values to JSON-safe Python values.
    """

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

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

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


def dataframe_records(
    dataframe: pd.DataFrame,
) -> list[dict]:
    records = []

    for _, row in (
        dataframe.iterrows()
    ):
        record = {}

        for (
            column,
            value,
        ) in row.items():
            record[
                str(
                    column
                )
            ] = json_safe_value(
                value
            )

        records.append(
            record
        )

    return records


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_delivery_id_column(
    dataframe: pd.DataFrame,
) -> str:
    candidates = [
        "id",
        "delivery_id",
        "deliveryId",
    ]

    for candidate in (
        candidates
    ):
        if (
            candidate
            in dataframe.columns
        ):
            return candidate

    raise ValueError(
        "Cannot find delivery ID column. "
        f"Available columns: "
        f"{dataframe.columns.tolist()}"
    )


def find_first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in (
            dataframe.columns
        ):
            return candidate

    return None


# ============================================================
# ENDPOINTS
# ============================================================

PERSONAL_COLUMNS = ["shipper", "shipper_id", "image"]
ADDRESS_COLUMNS = ["senderAddress", "receiverAddress"]
COORDINATE_COLUMNS = ["senderLat", "senderLng", "receiverLat", "receiverLng"]


def anonymize(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drop shipper ids and photo links, cut addresses down to district level
    and round coordinates to 3 decimals (about 110 m).

    Applied before search and filters so that ``q`` cannot be used to probe
    the hidden street addresses.
    """

    dataframe = dataframe.drop(columns=PERSONAL_COLUMNS, errors="ignore")

    for column in ADDRESS_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = (
                dataframe[column]
                .astype(str)
                .str.split(",")
                .str[-3:]
                .str.join(",")
                .str.strip()
            )

    for column in COORDINATE_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].round(3)

    return dataframe


@lru_cache(maxsize=1)
def load_deliveries_with_clusters() -> pd.DataFrame:
    """
    Cleaned dataset joined with the thesis K-Means label (cluster_id),
    anonymized unless XEDU_ANONYMIZE=0.
    """

    dataframe = load_deliveries().copy()

    if (
        "cluster_id" not in dataframe.columns
        and KMEANS_RESULT_PATH.exists()
    ):
        labels = pd.read_csv(
            KMEANS_RESULT_PATH,
            usecols=["id", "cluster_id"],
        )
        labels["id"] = labels["id"].astype(str)
        labels = labels.drop_duplicates("id")
        dataframe["id"] = dataframe["id"].astype(str)
        dataframe = dataframe.merge(
            labels,
            on="id",
            how="left",
        )

    return anonymize(dataframe) if ANONYMIZE_DELIVERIES else dataframe


SORTABLE_COLUMNS = {
    "createdAt",
    "expectedDeliveryTime",
    "weight",
    "haversineDistanceKm",
    "cluster_id",
    "mdh",
}


@router.get("")
def list_deliveries(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    q: str | None = Query(
        default=None,
        description="Search in order code, product, receiver address, id.",
    ),
    service_type: str | None = None,
    is_late: bool | None = None,
    cluster_id: int | None = None,
    sort_by: str | None = None,
    sort_dir: str = Query(default="asc", pattern="^(asc|desc)$"),
) -> dict:
    """
    Paginated list with optional search / filter / sort.

    Without any filter the behaviour is identical to the original
    endpoint (original columns plus the K-Means ``cluster_id``).
    """

    dataframe = load_deliveries_with_clusters()

    if q:
        needle = q.strip().lower()
        haystack = (
            dataframe["id"].astype(str)
            + " "
            + dataframe.get("mdh", "").astype(str)
            + " "
            + dataframe.get("package_name", "").astype(str)
            + " "
            + dataframe.get("receiverAddress", "").astype(str)
        ).str.lower()
        dataframe = dataframe[
            haystack.str.contains(needle, regex=False)
        ]

    if service_type:
        dataframe = dataframe[
            dataframe["serviceType"].astype(str) == service_type
        ]

    if is_late is not None and "isLate" in dataframe.columns:
        late = dataframe["isLate"].astype(str).str.lower() == "true"
        dataframe = dataframe[late if is_late else ~late]

    if cluster_id is not None and "cluster_id" in dataframe.columns:
        dataframe = dataframe[dataframe["cluster_id"] == cluster_id]

    if sort_by:
        if sort_by not in SORTABLE_COLUMNS or sort_by not in dataframe.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot sort by '{sort_by}'.",
            )
        dataframe = dataframe.sort_values(
            sort_by,
            ascending=(sort_dir == "asc"),
            kind="stable",
            na_position="last",
        )

    total = len(dataframe)
    page = dataframe.iloc[offset: offset + limit]

    return {
        "total": int(total),
        "offset": int(offset),
        "limit": int(limit),
        "returned": int(len(page)),
        "items": dataframe_records(page),
    }


@router.get(
    "/stats"
)
def delivery_stats() -> dict:
    """
    Return lightweight statistics for the cleaned delivery dataset.
    """

    dataframe = (
        load_deliveries()
    )

    id_column = (
        find_delivery_id_column(
            dataframe
        )
    )

    expected_time_column = (
        find_first_existing_column(
            dataframe,
            [
                "expectedDeliveryTime",
                "expected_delivery_time",
            ],
        )
    )

    shipper_column = (
        find_first_existing_column(
            dataframe,
            [
                "shipperId",
                "shipper_id",
                "shipper",
            ],
        )
    )

    service_column = (
        find_first_existing_column(
            dataframe,
            [
                "serviceType",
                "service_type",
                "service",
            ],
        )
    )

    stats = {
        "deliveries": int(
            len(
                dataframe
            )
        ),
        "columns": int(
            len(
                dataframe.columns
            )
        ),
        "delivery_id_column":
            id_column,
        "column_names": [
            str(
                column
            )
            for column
            in dataframe.columns
        ],
    }

    if (
        expected_time_column
        is not None
    ):
        stats[
            "missing_expected_delivery_time"
        ] = int(
            dataframe[
                expected_time_column
            ]
            .isna()
            .sum()
        )

    if (
        shipper_column
        is not None
    ):
        stats[
            "unique_shippers"
        ] = int(
            dataframe[
                shipper_column
            ]
            .nunique(
                dropna=True
            )
        )

    if (
        service_column
        is not None
    ):
        service_counts = (
            dataframe[
                service_column
            ]
            .fillna(
                "UNKNOWN"
            )
            .astype(
                str
            )
            .value_counts()
        )

        stats[
            "service_type_counts"
        ] = {
            str(
                key
            ): int(
                value
            )
            for (
                key,
                value,
            )
            in service_counts.items()
        }

    return stats


@router.get(
    "/{delivery_id}"
)
def get_delivery(
    delivery_id: str,
) -> dict:
    """
    Return one delivery by its authoritative ID.
    """

    dataframe = (
        load_deliveries_with_clusters()
    )

    id_column = (
        find_delivery_id_column(
            dataframe
        )
    )

    ids = (
        dataframe[
            id_column
        ]
        .astype(
            str
        )
        .str
        .strip()
    )

    normalized_id = (
        str(
            delivery_id
        )
        .strip()
    )

    matched = dataframe[
        ids
        == normalized_id
    ]

    if matched.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "Delivery not found: "
                f"{normalized_id}"
            ),
        )

    if (
        len(
            matched
        )
        > 1
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "Duplicate delivery ID "
                "detected in dataset."
            ),
        )

    return (
        dataframe_records(
            matched
        )[0]
    )