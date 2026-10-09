from functools import lru_cache

import numpy as np
import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from api.config import (
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

@router.get("")
def list_deliveries(
    offset: int = Query(
        default=0,
        ge=0,
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> dict:
    """
    Return a paginated list of deliveries.

    The endpoint intentionally returns the original cleaned
    dataset columns so the frontend can inspect all available
    delivery information.
    """

    dataframe = (
        load_deliveries()
    )

    total = len(
        dataframe
    )

    page = dataframe.iloc[
        offset:
        offset + limit
    ]

    return {
        "total": int(
            total
        ),
        "offset": int(
            offset
        ),
        "limit": int(
            limit
        ),
        "returned": int(
            len(
                page
            )
        ),
        "items": (
            dataframe_records(
                page
            )
        ),
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
        load_deliveries()
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