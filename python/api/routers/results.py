from functools import (
    lru_cache,
)

import numpy as np

import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from api.config import (
    DATASET_BENCHMARK_PATH,
    DATASET_BENCHMARK_SUMMARY_PATH,
    DATASET_WORKLOADS_PATH,
    FINAL_VALIDATION_PAIRED_PATH,
    FINAL_VALIDATION_SUMMARY_PATH,
    SENSITIVITY_PAIRWISE_PATH,
    SENSITIVITY_SUMMARY_PATH,
)


router = APIRouter(
    prefix="/api/results",
    tags=["Results"],
)


@lru_cache(maxsize=16)
def load_csv(
    path_string: str,
) -> pd.DataFrame:
    return pd.read_csv(
        path_string
    )


def safe_value(
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


def records(
    dataframe: pd.DataFrame,
) -> list[dict]:
    return [
        {
            str(
                column
            ):
                safe_value(
                    value
                )

            for (
                column,
                value,
            )
            in row.items()
        }

        for _, row
        in dataframe.iterrows()
    ]


@router.get(
    "/overview"
)
def results_overview(
) -> dict:
    benchmark_summary = (
        load_csv(
            str(
                DATASET_BENCHMARK_SUMMARY_PATH
            )
        )
    )

    benchmark_runs = (
        load_csv(
            str(
                DATASET_BENCHMARK_PATH
            )
        )
    )

    final = (
        load_csv(
            str(
                FINAL_VALIDATION_SUMMARY_PATH
            )
        )
        .iloc[
            0
        ]
    )

    sensitivity = (
        load_csv(
            str(
                SENSITIVITY_SUMMARY_PATH
            )
        )
    )

    selected = sensitivity[
        np.isclose(
            sensitivity[
                "cluster_switch_penalty"
            ],
            0.50,
        )
    ]

    lambda_row = (
        None
        if selected.empty
        else selected.iloc[
            0
        ]
    )

    return {
        "benchmark": {
            "workloads":
                int(
                    benchmark_summary[
                        "workloads"
                    ].max()
                ),

            "deliveries":
                int(
                    benchmark_summary[
                        "deliveries"
                    ].max()
                ),

            "algorithm_runs":
                int(
                    len(
                        benchmark_runs
                    )
                ),

            "algorithms":
                records(
                    benchmark_summary
                ),
        },

        "kmeans_guided_dqn": {
            "selected_lambda":
                0.50,

            "sensitivity_at_selected_lambda":
                (
                    None
                    if lambda_row is None

                    else records(
                        pd.DataFrame(
                            [
                                lambda_row
                            ]
                        )
                    )[
                        0
                    ]
                ),

            "final_validation":
                records(
                    pd.DataFrame(
                        [
                            final
                        ]
                    )
                )[
                    0
                ],
        },
    }


@router.get(
    "/benchmark/summary"
)
def benchmark_summary(
) -> dict:
    dataframe = (
        load_csv(
            str(
                DATASET_BENCHMARK_SUMMARY_PATH
            )
        )
    )

    return {
        "rows":
            len(
                dataframe
            ),

        "items":
            records(
                dataframe
            ),
    }


@router.get(
    "/benchmark/runs"
)
def benchmark_runs(
    algorithm: (
        str
        | None
    ) = None,
    workload_id: (
        int
        | None
    ) = Query(
        default=None,
        ge=1,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
) -> dict:
    dataframe = (
        load_csv(
            str(
                DATASET_BENCHMARK_PATH
            )
        )
        .copy()
    )

    if algorithm is not None:
        allowed = {
            "nearest_neighbor",
            "clarke_wright",
            "q_learning",
            "sarsa",
            "dqn",
        }

        if algorithm not in allowed:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown algorithm: "
                    f"{algorithm}"
                ),
            )

        dataframe = dataframe[
            dataframe[
                "algorithm"
            ]
            == algorithm
        ]

    if workload_id is not None:
        dataframe = dataframe[
            dataframe[
                "workload_id"
            ]
            == int(
                workload_id
            )
        ]

    total = len(
        dataframe
    )

    page = dataframe.iloc[
        offset:
        offset + limit
    ]

    return {
        "total":
            int(
                total
            ),

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
            records(
                page
            ),
    }


@router.get(
    "/workloads"
)
def workloads(
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
    dataframe = (
        load_csv(
            str(
                DATASET_WORKLOADS_PATH
            )
        )
        .copy()
    )

    page = dataframe.iloc[
        offset:
        offset + limit
    ].copy()

    items = []

    for _, row in (
        page.iterrows()
    ):
        item = {
            str(
                column
            ):
                safe_value(
                    value
                )

            for (
                column,
                value,
            )
            in row.items()
        }

        item[
            "delivery_ids"
        ] = [
            part.strip()

            for part
            in str(
                row[
                    "delivery_ids"
                ]
            ).split(
                "->"
            )

            if part.strip()
        ]

        items.append(
            item
        )

    return {
        "total":
            int(
                len(
                    dataframe
                )
            ),

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
                    items
                )
            ),

        "items":
            items,
    }


@router.get(
    "/sensitivity"
)
def sensitivity_results(
) -> dict:
    summary = (
        load_csv(
            str(
                SENSITIVITY_SUMMARY_PATH
            )
        )
    )

    pairwise = (
        load_csv(
            str(
                SENSITIVITY_PAIRWISE_PATH
            )
        )
    )

    return {
        "selected_lambda":
            0.50,

        "summary":
            records(
                summary
            ),

        "pairwise_vs_lambda_zero":
            records(
                pairwise
            ),
    }


@router.get(
    "/final-validation"
)
def final_validation_results(
) -> dict:
    summary = (
        load_csv(
            str(
                FINAL_VALIDATION_SUMMARY_PATH
            )
        )
    )

    paired = (
        load_csv(
            str(
                FINAL_VALIDATION_PAIRED_PATH
            )
        )
    )

    return {
        "summary":
            records(
                summary
            )[
                0
            ],

        "paired_workloads":
            records(
                paired
            ),

        "scope_note":
            (
                "Distance/time/SLA use all "
                "12 validation workloads; "
                "cluster-switch paired comparison "
                "is available for 6 workloads "
                "with vanilla route information."
            ),
    }