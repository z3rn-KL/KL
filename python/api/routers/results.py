import json
from functools import (
    lru_cache,
)

import numpy as np

import pandas as pd

from typing import (
    Literal,
)

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from api.config import (
    DAILY_PLANNING_DAILY_PATH,
    DAILY_PLANNING_SENSITIVITY_PATH,
    DAILY_PLANNING_SUMMARY_PATH,
    DATASET_BENCHMARK_PATH,
    DATASET_BENCHMARK_SUMMARY_PATH,
    DATASET_WORKLOADS_PATH,
    FINAL_VALIDATION_PAIRED_PATH,
    FINAL_VALIDATION_SUMMARY_PATH,
    PROPOSED_RL_TEST_PAIRWISE_PATH,
    REAL_REPLAY_SUMMARY_PATH,
    PROPOSED_RL_TEST_SUMMARY_PATH,
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


# ============================================================
# DAILY PLANNING
# ============================================================

AllocationPolicy = Literal[
    "dedicated",
    "merge_small",
]

OvertimePolicy = Literal[
    "overtime",
    "defer",
    "hybrid",
]

DAILY_PLANNING_NOTE = (
    "30-day simulation, 4-shipper pool, 10h shift, "
    "10 min service time. Load x1 is real XeDu data; "
    "higher loads are stress scenarios resampled from "
    "the same dataset. Cost coefficients are scenario "
    "parameters, not real XeDu costs. over_cap_shipper_days "
    "counts shipper-days above the 2h overtime cap."
)


def filter_policies(
    dataframe: pd.DataFrame,
    load_multiplier: int | None,
    allocation: str | None,
    overtime_policy: str | None,
) -> pd.DataFrame:
    for column, value in (
        ("load_multiplier", load_multiplier),
        ("allocation", allocation),
        ("overtime_policy", overtime_policy),
    ):
        if value is not None:
            dataframe = dataframe[
                dataframe[column] == value
            ]

    return dataframe


@router.get(
    "/daily-planning/summary"
)
def daily_planning_summary(
    load_multiplier: int | None = Query(
        default=None
    ),
    allocation: AllocationPolicy | None = Query(
        default=None
    ),
    overtime_policy: OvertimePolicy | None = Query(
        default=None
    ),
) -> dict:
    summary = load_csv(
        str(DAILY_PLANNING_SUMMARY_PATH)
    )

    # Cheapest policy per load among those that respect the cap.
    legal = summary[
        summary["over_cap_shipper_days"] == 0
    ]

    best = legal.loc[
        legal.groupby("load_multiplier")[
            "total_cost"
        ].idxmin()
    ]

    rows = filter_policies(
        summary,
        load_multiplier,
        allocation,
        overtime_policy,
    )

    return {
        "note": DAILY_PLANNING_NOTE,
        "rows": records(rows),
        "best_within_cap": records(best),
    }


@router.get(
    "/daily-planning/daily"
)
def daily_planning_daily(
    load_multiplier: int = Query(),
    allocation: AllocationPolicy = Query(),
    overtime_policy: OvertimePolicy = Query(),
) -> dict:
    daily = filter_policies(
        load_csv(str(DAILY_PLANNING_DAILY_PATH)),
        load_multiplier,
        allocation,
        overtime_policy,
    )

    if daily.empty:
        raise HTTPException(
            status_code=404,
            detail="No daily rows for this scenario.",
        )

    return {
        "note": DAILY_PLANNING_NOTE,
        "days": records(
            daily.sort_values("day")
        ),
    }


@router.get(
    "/daily-planning/sensitivity"
)
def daily_planning_sensitivity(
    parameter: str | None = Query(
        default=None
    ),
) -> dict:
    sensitivity = load_csv(
        str(DAILY_PLANNING_SENSITIVITY_PATH)
    )

    if parameter is not None:
        sensitivity = sensitivity[
            sensitivity["parameter"] == parameter
        ]

        if sensitivity.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown parameter: {parameter}",
            )

    return {
        "note": DAILY_PLANNING_NOTE,
        "rows": records(sensitivity),
    }


# ============================================================
# PROPOSED RL
# ============================================================

@router.get(
    "/proposed-rl/final-test"
)
def proposed_rl_final_test(
) -> dict:
    return {
        "scope_note": (
            "Held-out Test: 16 workloads, 131 deliveries. "
            "The model and objective were locked on Train and "
            "Validation before Test; nothing was tuned on Test."
        ),
        "summary": records(
            load_csv(
                str(PROPOSED_RL_TEST_SUMMARY_PATH)
            )
        ),
        "pairwise_vs_baselines": records(
            load_csv(
                str(PROPOSED_RL_TEST_PAIRWISE_PATH)
            )
        ),
    }


# ============================================================
# REAL DATA
# ============================================================

@router.get(
    "/real-data"
)
def real_data_results(
) -> dict:
    if not REAL_REPLAY_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Real-data replay has not been run. "
                "Run python experiments/real_data_replay.py."
            ),
        )

    summary = json.loads(
        REAL_REPLAY_SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    return {
        **summary,
        "scope_note": (
            "Orders, coordinates, deadlines and delivery times are "
            "real XeDu data. Route distances and times are computed on "
            "the OpenStreetMap road graph with free-flow speeds; they "
            "are not measured and no algorithm was tried in the field."
        ),
    }
