"""
Final integration validation for the thesis:
    Vanilla DQN vs K-Means-guided DQN (lambda_c = 0.50)

Design
------
- The authoritative 108-workload manifest is read from
  results/dataset_algorithm_workloads.csv.
- Vanilla DQN metrics are reused from
  results/dataset_algorithm_all_results.csv.
- Twelve validation workloads are selected deterministically:
  two workloads (chronological median + chronological final) for each
  N in {5, 6, 8, 10, 12, 13}.
- If a selected lambda=0.50 result already exists in the sensitivity CSV,
  it is reused. Otherwise only the guided DQN is trained.
- Autosave/resume is enabled for newly trained guided runs.

Important methodological note
-----------------------------
The original dataset benchmark did not save DQN route sequences, so vanilla
cluster-switch metrics are only available for workloads that were also run in
the earlier sensitivity experiment (lambda=0.00). Distance/time/SLA comparisons
are available for all 12 workloads.

Outputs
-------
results/final_kmeans_guided_dqn_validation_results.csv
results/final_kmeans_guided_dqn_validation_paired.csv
results/final_kmeans_guided_dqn_validation_summary.csv
results/final_kmeans_guided_dqn_validation_workloads.csv
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from evaluation import RouteEvaluator
from routing import RoadMatrixBuilder

from experiments.kmeans_guided_dqn_sensitivity import (
    calculate_cluster_statistics,
    calculate_route_cluster_metrics,
    get_cluster_ids_for_deliveries,
    prepare_experiment_dataframe,
    run_dqn,
)

from experiments.q_learning_single_workload_experiment import (
    build_workload,
    get_depot_node,
    load_road_network,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

WORKLOAD_MANIFEST_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_workloads.csv"
)

BASELINE_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_all_results.csv"
)

SENSITIVITY_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_results.csv"
)

GUIDED_OUTPUT_PATH = (
    RESULTS_DIR
    / "final_kmeans_guided_dqn_validation_results.csv"
)

PAIRED_OUTPUT_PATH = (
    RESULTS_DIR
    / "final_kmeans_guided_dqn_validation_paired.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "final_kmeans_guided_dqn_validation_summary.csv"
)

WORKLOAD_OUTPUT_PATH = (
    RESULTS_DIR
    / "final_kmeans_guided_dqn_validation_workloads.csv"
)


FINAL_WORKLOAD_SIZES = [
    5,
    6,
    8,
    10,
    12,
    13,
]

GUIDED_LAMBDA = 0.50

EXPECTED_DATASET_WORKLOADS = 108


RESULT_COLUMNS = [
    "workload_id",
    "wave_start",
    "wave_end",
    "number_of_deliveries",
    "cluster_switch_penalty",
    "number_of_clusters",
    "dominant_cluster_size",
    "dominant_cluster_share",
    "total_distance_km",
    "total_travel_time_minutes",
    "on_time_deliveries",
    "late_deliveries",
    "on_time_rate",
    "total_lateness_minutes",
    "max_lateness_minutes",
    "training_runtime_seconds",
    "route_runtime_seconds",
    "matrix_preparation_seconds",
    "greedy_reward",
    "reward_per_delivery",
    "model_size",
    "final_epsilon",
    "device",
    "cluster_switches",
    "cluster_switch_rate",
    "cluster_sequence",
    "delivery_order",
    "result_source",
]


# ============================================================
# GENERIC HELPERS
# ============================================================

def normalize_timestamp(value) -> str:
    """
    Return a stable UTC ISO timestamp string.
    """

    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is None:
        timestamp = (
            timestamp.tz_localize(
                "UTC"
            )
        )
    else:
        timestamp = (
            timestamp.tz_convert(
                "UTC"
            )
        )

    return timestamp.isoformat()


def parse_delivery_ids(
    value,
) -> list[str]:
    """
    Parse:
        id -> id -> id
    """

    if pd.isna(
        value
    ):
        return []

    return [
        item.strip()
        for item
        in str(value).split(
            "->"
        )
        if item.strip()
    ]


# ============================================================
# AUTHORITATIVE WORKLOAD MANIFEST
# ============================================================

def load_workload_manifest(
) -> pd.DataFrame:

    if not WORKLOAD_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: "
            f"{WORKLOAD_MANIFEST_PATH}"
        )

    df = pd.read_csv(
        WORKLOAD_MANIFEST_PATH
    )

    required = {
        "workload_id",
        "wave_start",
        "wave_end",
        "number_of_deliveries",
        "delivery_ids",
    }

    missing = sorted(
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Workload manifest thiếu cột: "
            + ", ".join(
                missing
            )
        )

    df = df.copy()

    df[
        "workload_id"
    ] = pd.to_numeric(
        df[
            "workload_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    df[
        "number_of_deliveries"
    ] = pd.to_numeric(
        df[
            "number_of_deliveries"
        ],
        errors="raise",
    ).astype(
        int
    )

    df[
        "wave_start_key"
    ] = df[
        "wave_start"
    ].map(
        normalize_timestamp
    )

    df[
        "wave_end_key"
    ] = df[
        "wave_end"
    ].map(
        normalize_timestamp
    )

    if (
        len(df)
        != EXPECTED_DATASET_WORKLOADS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_DATASET_WORKLOADS} "
            f"workload rows, "
            f"found {len(df)}."
        )

    if df[
        "workload_id"
    ].duplicated().any():
        raise ValueError(
            "dataset_algorithm_workloads.csv "
            "có workload_id trùng."
        )

    expected_ids = set(
        range(
            1,
            EXPECTED_DATASET_WORKLOADS + 1,
        )
    )

    actual_ids = set(
        df[
            "workload_id"
        ].tolist()
    )

    if (
        actual_ids
        != expected_ids
    ):
        raise ValueError(
            "workload_id trong manifest "
            "không phải dãy 1..108."
        )

    for _, row in df.iterrows():

        ids = parse_delivery_ids(
            row[
                "delivery_ids"
            ]
        )

        expected_n = int(
            row[
                "number_of_deliveries"
            ]
        )

        if (
            len(ids)
            != expected_n
        ):
            raise ValueError(
                "Manifest delivery count mismatch at "
                f"workload_id="
                f"{int(row['workload_id'])}: "
                f"manifest N="
                f"{expected_n}, "
                f"parsed IDs="
                f"{len(ids)}"
            )

        if (
            len(ids)
            != len(
                set(ids)
            )
        ):
            raise ValueError(
                "Manifest có delivery_id trùng tại "
                f"workload_id="
                f"{int(row['workload_id'])}."
            )

    return (
        df
        .sort_values(
            "workload_id"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# VANILLA DQN BASELINE
# ============================================================

def load_baseline_dqn(
    manifest: pd.DataFrame,
) -> pd.DataFrame:

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: "
            f"{BASELINE_PATH}"
        )

    df = pd.read_csv(
        BASELINE_PATH
    )

    required = {
        "workload_id",
        "wave_start",
        "wave_end",
        "algorithm",
        "number_of_deliveries",
        "on_time_deliveries",
        "late_deliveries",
        "on_time_rate",
        "total_lateness_minutes",
        "max_lateness_minutes",
        "total_distance_km",
        "total_travel_time_minutes",
        "training_runtime_seconds",
        "greedy_reward",
        "model_size",
        "final_epsilon",
        "device",
    }

    missing = sorted(
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Baseline CSV thiếu cột: "
            + ", ".join(
                missing
            )
        )

    dqn = (
        df[
            df[
                "algorithm"
            ].astype(
                str
            )
            == "dqn"
        ]
        .copy()
        .sort_values(
            "workload_id"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(dqn)
        != EXPECTED_DATASET_WORKLOADS
    ):
        raise ValueError(
            "Expected 108 DQN baseline rows, "
            f"found {len(dqn)}."
        )

    if dqn[
        "workload_id"
    ].duplicated().any():
        raise ValueError(
            "Baseline DQN có workload_id trùng."
        )

    dqn[
        "workload_id"
    ] = pd.to_numeric(
        dqn[
            "workload_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    dqn[
        "wave_start_key"
    ] = dqn[
        "wave_start"
    ].map(
        normalize_timestamp
    )

    dqn[
        "wave_end_key"
    ] = dqn[
        "wave_end"
    ].map(
        normalize_timestamp
    )

    check = (
        manifest[
            [
                "workload_id",
                "wave_start_key",
                "wave_end_key",
                "number_of_deliveries",
            ]
        ]
        .merge(
            dqn[
                [
                    "workload_id",
                    "wave_start_key",
                    "wave_end_key",
                    "number_of_deliveries",
                ]
            ],
            on="workload_id",
            how="outer",
            suffixes=(
                "_manifest",
                "_baseline",
            ),
            validate="one_to_one",
        )
    )

    mismatch = check[
        (
            check[
                "wave_start_key_manifest"
            ]
            != check[
                "wave_start_key_baseline"
            ]
        )
        |
        (
            check[
                "wave_end_key_manifest"
            ]
            != check[
                "wave_end_key_baseline"
            ]
        )
        |
        (
            check[
                "number_of_deliveries_manifest"
            ]
            != check[
                "number_of_deliveries_baseline"
            ]
        )
    ]

    if not mismatch.empty:

        ids = ", ".join(
            str(
                value
            )
            for value
            in mismatch[
                "workload_id"
            ].tolist()
        )

        raise ValueError(
            "Manifest và baseline DQN "
            "không khớp tại workload_id: "
            + ids
        )

    return dqn


# ============================================================
# SENSITIVITY REUSE
# ============================================================

def load_sensitivity(
) -> pd.DataFrame:

    if not SENSITIVITY_PATH.exists():

        print(
            "Không thấy sensitivity CSV -> "
            "sẽ train đủ guided workloads."
        )

        return pd.DataFrame()

    df = pd.read_csv(
        SENSITIVITY_PATH
    )

    required = {
        "workload_size",
        "wave_start",
        "cluster_switch_penalty",
        "cluster_switches",
        "cluster_switch_rate",
    }

    if not required.issubset(
        df.columns
    ):
        print(
            "Sensitivity CSV schema "
            "không phù hợp -> không reuse."
        )

        return pd.DataFrame()

    df = df.copy()

    df[
        "workload_size"
    ] = pd.to_numeric(
        df[
            "workload_size"
        ],
        errors="raise",
    ).astype(
        int
    )

    df[
        "wave_start_key"
    ] = df[
        "wave_start"
    ].map(
        normalize_timestamp
    )

    return df


def find_sensitivity_row(
    sensitivity_df: pd.DataFrame,
    size: int,
    wave_start,
    penalty: float,
):

    if sensitivity_df.empty:
        return None

    key = normalize_timestamp(
        wave_start
    )

    matched = sensitivity_df[
        (
            sensitivity_df[
                "workload_size"
            ]
            == int(
                size
            )
        )
        &
        (
            sensitivity_df[
                "wave_start_key"
            ]
            == key
        )
        &
        np.isclose(
            pd.to_numeric(
                sensitivity_df[
                    "cluster_switch_penalty"
                ],
                errors="coerce",
            ),
            penalty,
        )
    ]

    if (
        len(matched)
        == 0
    ):
        return None

    if (
        len(matched)
        > 1
    ):
        raise ValueError(
            "Sensitivity duplicate: "
            f"N={size}, "
            f"wave={key}, "
            f"lambda={penalty}"
        )

    return matched.iloc[
        0
    ]


def sensitivity_guided_to_result(
    sensitivity_row: pd.Series,
    baseline_row: pd.Series,
) -> dict | None:

    required_from_sensitivity = {
        "wave_start",
        "wave_end",
        "number_of_clusters",
        "dominant_cluster_size",
        "dominant_cluster_share",
        "number_of_deliveries",
        "total_distance_km",
        "total_travel_time_minutes",
        "on_time_deliveries",
        "late_deliveries",
        "on_time_rate",
        "total_lateness_minutes",
        "max_lateness_minutes",
        "training_runtime_seconds",
        "route_runtime_seconds",
        "matrix_preparation_seconds",
        "greedy_reward",
        "reward_per_delivery",
        "model_size",
        "final_epsilon",
        "device",
        "cluster_switches",
        "cluster_switch_rate",
        "cluster_sequence",
        "delivery_order",
    }

    if not (
        required_from_sensitivity
        .issubset(
            sensitivity_row.index
        )
    ):
        return None

    row = {
        column:
            sensitivity_row[
                column
            ]
        for column
        in RESULT_COLUMNS
        if column
        in sensitivity_row.index
    }

    row[
        "workload_id"
    ] = int(
        baseline_row[
            "workload_id"
        ]
    )

    row[
        "wave_start"
    ] = baseline_row[
        "wave_start"
    ]

    row[
        "wave_end"
    ] = baseline_row[
        "wave_end"
    ]

    row[
        "number_of_deliveries"
    ] = int(
        baseline_row[
            "number_of_deliveries"
        ]
    )

    row[
        "cluster_switch_penalty"
    ] = GUIDED_LAMBDA

    row[
        "result_source"
    ] = (
        "reused_sensitivity"
    )

    return row


# ============================================================
# SELECT FINAL VALIDATION WORKLOADS
# ============================================================

def select_final_workloads(
    manifest: pd.DataFrame,
    baseline_dqn: pd.DataFrame,
) -> pd.DataFrame:

    selections = []

    for size in (
        FINAL_WORKLOAD_SIZES
    ):

        candidates = (
            manifest[
                manifest[
                    "number_of_deliveries"
                ]
                == size
            ]
            .sort_values(
                "wave_start_key"
            )
            .reset_index(
                drop=True
            )
        )

        if (
            len(candidates)
            < 2
        ):
            raise ValueError(
                f"Không đủ 2 workload "
                f"cho N={size}."
            )

        indices = [
            (
                "median",
                len(candidates)
                // 2,
            ),
            (
                "final",
                len(candidates)
                - 1,
            ),
        ]

        for (
            role,
            index,
        ) in indices:

            manifest_row = (
                candidates.iloc[
                    index
                ]
            )

            workload_id = int(
                manifest_row[
                    "workload_id"
                ]
            )

            baseline_match = (
                baseline_dqn[
                    baseline_dqn[
                        "workload_id"
                    ]
                    == workload_id
                ]
            )

            if (
                len(
                    baseline_match
                )
                != 1
            ):
                raise ValueError(
                    "Không tìm thấy đúng 1 "
                    "baseline DQN cho "
                    f"workload_id="
                    f"{workload_id}."
                )

            combined = (
                baseline_match
                .iloc[
                    0
                ]
                .copy()
            )

            combined[
                "selection_role"
            ] = role

            combined[
                "manifest_delivery_ids"
            ] = (
                manifest_row[
                    "delivery_ids"
                ]
            )

            selections.append(
                combined
            )

    selected = pd.DataFrame(
        selections
    )

    if selected[
        "workload_id"
    ].duplicated().any():
        raise ValueError(
            "Final selection "
            "có workload_id trùng."
        )

    return (
        selected
        .sort_values(
            [
                "number_of_deliveries",
                "selection_role",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# REBUILD ONE WORKLOAD FROM MANIFEST
# ============================================================

def build_workload_dataframe_from_manifest(
    prepared_dataframe: pd.DataFrame,
    selected_row: pd.Series,
) -> pd.DataFrame:

    expected_ids = parse_delivery_ids(
        selected_row[
            "manifest_delivery_ids"
        ]
    )

    if not expected_ids:
        raise ValueError(
            "Manifest workload "
            "không có delivery_ids."
        )

    dataframe = (
        prepared_dataframe
        .copy()
    )

    dataframe[
        "delivery_id"
    ] = (
        dataframe[
            "delivery_id"
        ]
        .astype(
            str
        )
        .str
        .strip()
    )

    if dataframe[
        "delivery_id"
    ].duplicated().any():
        raise ValueError(
            "Prepared dataframe "
            "có delivery_id trùng; "
            "không thể map manifest an toàn."
        )

    indexed = (
        dataframe
        .set_index(
            "delivery_id",
            drop=False,
        )
    )

    missing = [
        delivery_id
        for delivery_id
        in expected_ids
        if delivery_id
        not in indexed.index
    ]

    if missing:
        raise ValueError(
            "Prepared dataframe thiếu "
            "delivery IDs của "
            f"workload_id="
            f"{int(selected_row['workload_id'])}: "
            + ", ".join(
                missing
            )
        )

    workload_df = (
        indexed
        .loc[
            expected_ids
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    expected_n = int(
        selected_row[
            "number_of_deliveries"
        ]
    )

    if (
        len(
            workload_df
        )
        != expected_n
    ):
        raise ValueError(
            "Rebuilt workload size mismatch tại "
            f"workload_id="
            f"{int(selected_row['workload_id'])}."
        )

    if (
        "wave_start"
        in workload_df.columns
    ):

        wave_keys = {
            normalize_timestamp(
                value
            )
            for value
            in workload_df[
                "wave_start"
            ]
        }

        expected_wave = (
            normalize_timestamp(
                selected_row[
                    "wave_start"
                ]
            )
        )

        if (
            wave_keys
            != {
                expected_wave
            }
        ):
            raise ValueError(
                "wave_start mismatch tại "
                f"workload_id="
                f"{int(selected_row['workload_id'])}."
            )

    if (
        "wave_end"
        in workload_df.columns
    ):

        wave_end_keys = {
            normalize_timestamp(
                value
            )
            for value
            in workload_df[
                "wave_end"
            ]
        }

        expected_end = (
            normalize_timestamp(
                selected_row[
                    "wave_end"
                ]
            )
        )

        if (
            wave_end_keys
            != {
                expected_end
            }
        ):
            raise ValueError(
                "wave_end mismatch tại "
                f"workload_id="
                f"{int(selected_row['workload_id'])}."
            )

    if (
        "cluster_id"
        not in workload_df.columns
    ):
        raise ValueError(
            "Prepared dataframe "
            "không có cluster_id."
        )

    if workload_df[
        "cluster_id"
    ].isna().any():
        raise ValueError(
            "Có delivery thiếu cluster_id tại "
            f"workload_id="
            f"{int(selected_row['workload_id'])}."
        )

    return workload_df


# ============================================================
# AUTOSAVE / RESUME
# ============================================================

def load_existing_results(
) -> pd.DataFrame:

    if not GUIDED_OUTPUT_PATH.exists():
        return pd.DataFrame(
            columns=RESULT_COLUMNS
        )

    df = pd.read_csv(
        GUIDED_OUTPUT_PATH
    )

    if (
        "workload_id"
        not in df.columns
    ):
        raise ValueError(
            "Existing validation CSV "
            "thiếu workload_id."
        )

    df[
        "workload_id"
    ] = pd.to_numeric(
        df[
            "workload_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    if df[
        "workload_id"
    ].duplicated().any():
        raise ValueError(
            "Existing validation CSV "
            "có workload_id trùng."
        )

    return df


def append_result(
    row: dict,
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_df = (
        pd.DataFrame(
            [
                row
            ]
        )
        .reindex(
            columns=RESULT_COLUMNS
        )
    )

    if GUIDED_OUTPUT_PATH.exists():

        row_df.to_csv(
            GUIDED_OUTPUT_PATH,
            mode="a",
            header=False,
            index=False,
        )

    else:

        row_df.to_csv(
            GUIDED_OUTPUT_PATH,
            index=False,
        )


# ============================================================
# SAVE FINAL WORKLOAD MANIFEST
# ============================================================

def save_final_manifest(
    selected: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:

    rows = []

    for _, selected_row in (
        selected.iterrows()
    ):

        workload_df = (
            build_workload_dataframe_from_manifest(
                prepared_dataframe,
                selected_row,
            )
        )

        counts = (
            workload_df[
                "cluster_id"
            ]
            .astype(
                int
            )
            .value_counts()
            .sort_index()
        )

        reusable = (
            find_sensitivity_row(
                sensitivity_df,
                int(
                    selected_row[
                        "number_of_deliveries"
                    ]
                ),
                selected_row[
                    "wave_start"
                ],
                GUIDED_LAMBDA,
            )
            is not None
        )

        rows.append(
            {
                "workload_id":
                    int(
                        selected_row[
                            "workload_id"
                        ]
                    ),

                "selection_role":
                    selected_row[
                        "selection_role"
                    ],

                "wave_start":
                    selected_row[
                        "wave_start"
                    ],

                "wave_end":
                    selected_row[
                        "wave_end"
                    ],

                "number_of_deliveries":
                    len(
                        workload_df
                    ),

                "number_of_clusters":
                    int(
                        workload_df[
                            "cluster_id"
                        ].nunique()
                    ),

                "dominant_cluster_share":
                    float(
                        counts.max()
                        / len(
                            workload_df
                        )
                    ),

                "cluster_distribution":
                    "; ".join(
                        f"{int(cluster_id)}:"
                        f"{int(count)}"
                        for (
                            cluster_id,
                            count,
                        )
                        in counts.items()
                    ),

                "delivery_ids":
                    " -> ".join(
                        workload_df[
                            "delivery_id"
                        ].astype(
                            str
                        )
                    ),

                "lambda_050_available_from_sensitivity":
                    reusable,
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        WORKLOAD_OUTPUT_PATH,
        index=False,
    )


# ============================================================
# FINAL PAIRED COMPARISON
# ============================================================

def build_paired(
    selected: pd.DataFrame,
    guided_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:

    baseline = (
        selected[
            [
                "workload_id",
                "wave_start",
                "wave_end",
                "number_of_deliveries",
                "selection_role",
                "total_distance_km",
                "total_travel_time_minutes",
                "on_time_deliveries",
                "late_deliveries",
                "on_time_rate",
                "total_lateness_minutes",
                "max_lateness_minutes",
                "training_runtime_seconds",
            ]
        ]
        .rename(
            columns={
                "total_distance_km":
                    "baseline_distance_km",

                "total_travel_time_minutes":
                    "baseline_travel_time_minutes",

                "on_time_deliveries":
                    "baseline_on_time_deliveries",

                "late_deliveries":
                    "baseline_late_deliveries",

                "on_time_rate":
                    "baseline_on_time_rate",

                "total_lateness_minutes":
                    "baseline_total_lateness_minutes",

                "max_lateness_minutes":
                    "baseline_max_lateness_minutes",

                "training_runtime_seconds":
                    "baseline_training_runtime_seconds",
            }
        )
    )

    guided = (
        guided_df[
            [
                "workload_id",
                "total_distance_km",
                "total_travel_time_minutes",
                "on_time_deliveries",
                "late_deliveries",
                "on_time_rate",
                "total_lateness_minutes",
                "max_lateness_minutes",
                "training_runtime_seconds",
                "route_runtime_seconds",
                "number_of_clusters",
                "dominant_cluster_share",
                "cluster_switches",
                "cluster_switch_rate",
                "cluster_sequence",
                "delivery_order",
                "result_source",
            ]
        ]
        .rename(
            columns={
                "total_distance_km":
                    "guided_distance_km",

                "total_travel_time_minutes":
                    "guided_travel_time_minutes",

                "on_time_deliveries":
                    "guided_on_time_deliveries",

                "late_deliveries":
                    "guided_late_deliveries",

                "on_time_rate":
                    "guided_on_time_rate",

                "total_lateness_minutes":
                    "guided_total_lateness_minutes",

                "max_lateness_minutes":
                    "guided_max_lateness_minutes",

                "training_runtime_seconds":
                    "guided_training_runtime_seconds",

                "route_runtime_seconds":
                    "guided_route_runtime_seconds",

                "cluster_switches":
                    "guided_cluster_switches",

                "cluster_switch_rate":
                    "guided_cluster_switch_rate",
            }
        )
    )

    paired = baseline.merge(
        guided,
        on="workload_id",
        how="inner",
        validate="one_to_one",
    )

    paired[
        "distance_improvement_pct"
    ] = (
        (
            paired[
                "baseline_distance_km"
            ]
            - paired[
                "guided_distance_km"
            ]
        )
        / paired[
            "baseline_distance_km"
        ]
        * 100.0
    )

    paired[
        "time_improvement_pct"
    ] = (
        (
            paired[
                "baseline_travel_time_minutes"
            ]
            - paired[
                "guided_travel_time_minutes"
            ]
        )
        / paired[
            "baseline_travel_time_minutes"
        ]
        * 100.0
    )

    paired[
        "on_time_rate_change"
    ] = (
        paired[
            "guided_on_time_rate"
        ]
        - paired[
            "baseline_on_time_rate"
        ]
    )

    paired[
        "baseline_cluster_switches"
    ] = np.nan

    paired[
        "baseline_cluster_switch_rate"
    ] = np.nan

    for (
        index,
        row,
    ) in paired.iterrows():

        sensitivity_baseline = (
            find_sensitivity_row(
                sensitivity_df,
                int(
                    row[
                        "number_of_deliveries"
                    ]
                ),
                row[
                    "wave_start"
                ],
                0.0,
            )
        )

        if (
            sensitivity_baseline
            is None
        ):
            continue

        paired.loc[
            index,
            "baseline_cluster_switches",
        ] = float(
            sensitivity_baseline[
                "cluster_switches"
            ]
        )

        paired.loc[
            index,
            "baseline_cluster_switch_rate",
        ] = float(
            sensitivity_baseline[
                "cluster_switch_rate"
            ]
        )

    paired[
        "cluster_switch_reduction"
    ] = (
        paired[
            "baseline_cluster_switches"
        ]
        - paired[
            "guided_cluster_switches"
        ]
    )

    paired[
        "cluster_switch_rate_reduction"
    ] = (
        paired[
            "baseline_cluster_switch_rate"
        ]
        - paired[
            "guided_cluster_switch_rate"
        ]
    )

    return (
        paired
        .sort_values(
            [
                "number_of_deliveries",
                "selection_role",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:

    switch_subset = paired[
        paired[
            "baseline_cluster_switch_rate"
        ].notna()
    ]

    deliveries = int(
        paired[
            "number_of_deliveries"
        ].sum()
    )

    baseline_on_time = int(
        paired[
            "baseline_on_time_deliveries"
        ].sum()
    )

    guided_on_time = int(
        paired[
            "guided_on_time_deliveries"
        ].sum()
    )

    baseline_distance_mean = float(
        paired[
            "baseline_distance_km"
        ].mean()
    )

    guided_distance_mean = float(
        paired[
            "guided_distance_km"
        ].mean()
    )

    baseline_time_mean = float(
        paired[
            "baseline_travel_time_minutes"
        ].mean()
    )

    guided_time_mean = float(
        paired[
            "guided_travel_time_minutes"
        ].mean()
    )

    summary_row = {
        "guided_lambda":
            GUIDED_LAMBDA,

        "workloads":
            len(
                paired
            ),

        "deliveries":
            deliveries,

        "mean_baseline_distance_km":
            baseline_distance_mean,

        "mean_guided_distance_km":
            guided_distance_mean,

        "ratio_of_means_distance_improvement_pct":
            (
                (
                    baseline_distance_mean
                    - guided_distance_mean
                )
                / baseline_distance_mean
                * 100.0
            ),

        "mean_pairwise_distance_improvement_pct":
            paired[
                "distance_improvement_pct"
            ].mean(),

        "distance_win_rate":
            (
                paired[
                    "distance_improvement_pct"
                ]
                > 0
            ).mean(),

        "distance_tie_rate":
            np.isclose(
                paired[
                    "distance_improvement_pct"
                ],
                0.0,
                atol=1e-9,
            ).mean(),

        "mean_baseline_travel_time_minutes":
            baseline_time_mean,

        "mean_guided_travel_time_minutes":
            guided_time_mean,

        "ratio_of_means_time_improvement_pct":
            (
                (
                    baseline_time_mean
                    - guided_time_mean
                )
                / baseline_time_mean
                * 100.0
            ),

        "mean_pairwise_time_improvement_pct":
            paired[
                "time_improvement_pct"
            ].mean(),

        "time_win_rate":
            (
                paired[
                    "time_improvement_pct"
                ]
                > 0
            ).mean(),

        "time_tie_rate":
            np.isclose(
                paired[
                    "time_improvement_pct"
                ],
                0.0,
                atol=1e-9,
            ).mean(),

        "baseline_weighted_on_time_rate":
            (
                baseline_on_time
                / deliveries
            ),

        "guided_weighted_on_time_rate":
            (
                guided_on_time
                / deliveries
            ),

        "baseline_total_late_deliveries":
            int(
                paired[
                    "baseline_late_deliveries"
                ].sum()
            ),

        "guided_total_late_deliveries":
            int(
                paired[
                    "guided_late_deliveries"
                ].sum()
            ),

        "mean_guided_cluster_switches_all_12":
            paired[
                "guided_cluster_switches"
            ].mean(),

        "mean_guided_cluster_switch_rate_all_12":
            paired[
                "guided_cluster_switch_rate"
            ].mean(),

        "switch_comparison_workloads":
            len(
                switch_subset
            ),

        "mean_baseline_switch_rate_where_available":
            (
                switch_subset[
                    "baseline_cluster_switch_rate"
                ].mean()
                if not switch_subset.empty
                else np.nan
            ),

        "mean_guided_switch_rate_where_available":
            (
                switch_subset[
                    "guided_cluster_switch_rate"
                ].mean()
                if not switch_subset.empty
                else np.nan
            ),

        "mean_switch_rate_reduction_where_available":
            (
                switch_subset[
                    "cluster_switch_rate_reduction"
                ].mean()
                if not switch_subset.empty
                else np.nan
            ),

        "relative_switch_rate_reduction_pct_where_available":
            (
                (
                    switch_subset[
                        "cluster_switch_rate_reduction"
                    ].mean()
                    / switch_subset[
                        "baseline_cluster_switch_rate"
                    ].mean()
                    * 100.0
                )
                if (
                    not switch_subset.empty
                    and switch_subset[
                        "baseline_cluster_switch_rate"
                    ].mean()
                    > 0
                )
                else np.nan
            ),

        "mean_guided_training_runtime_seconds":
            paired[
                "guided_training_runtime_seconds"
            ].mean(),

        "newly_trained_workloads":
            int(
                (
                    paired[
                        "result_source"
                    ]
                    == "new_training"
                ).sum()
            ),

        "reused_sensitivity_workloads":
            int(
                (
                    paired[
                        "result_source"
                    ]
                    == "reused_sensitivity"
                ).sum()
            ),
    }

    return pd.DataFrame(
        [
            summary_row
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "=" * 76
    )
    print(
        "FINAL K-MEANS-GUIDED DQN VALIDATION"
    )
    print(
        "=" * 76
    )

    print(
        f"lambda_c             : "
        f"{GUIDED_LAMBDA:.2f}"
    )

    print(
        f"workload sizes       : "
        f"{FINAL_WORKLOAD_SIZES}"
    )

    print(
        "selection            : "
        "chronological median + final per size"
    )

    print(
        f"target workloads     : "
        f"{len(FINAL_WORKLOAD_SIZES) * 2}"
    )

    print(
        "authoritative source : "
        "dataset_algorithm_workloads.csv"
    )

    print(
        "vanilla baseline     : "
        "reuse dataset_algorithm_all_results.csv"
    )

    print(
        "sensitivity reuse    : enabled"
    )

    print(
        "autosave / resume    : enabled"
    )

    print(
        "=" * 76
    )

    manifest = (
        load_workload_manifest()
    )

    baseline_dqn = (
        load_baseline_dqn(
            manifest
        )
    )

    sensitivity_df = (
        load_sensitivity()
    )

    selected = (
        select_final_workloads(
            manifest,
            baseline_dqn,
        )
    )

    (
        prepared_dataframe,
        delivery_lookup,
    ) = (
        prepare_experiment_dataframe()
    )

    save_final_manifest(
        selected,
        prepared_dataframe,
        sensitivity_df,
    )

    print()
    print(
        "Selected workloads:"
    )

    print(
        selected[
            [
                "workload_id",
                "number_of_deliveries",
                "selection_role",
                "wave_start",
            ]
        ].to_string(
            index=False
        )
    )

    existing = (
        load_existing_results()
    )

    selected_ids = set(
        selected[
            "workload_id"
        ].astype(
            int
        )
    )

    completed_ids = set(
        existing[
            "workload_id"
        ].astype(
            int
        )
    )

    unexpected = (
        completed_ids
        - selected_ids
    )

    if unexpected:
        raise ValueError(
            "Output final cũ chứa workload "
            "không thuộc selection hiện tại: "
            + ", ".join(
                str(
                    value
                )
                for value
                in sorted(
                    unexpected
                )
            )
            + ". Hãy đổi tên hoặc xóa output final cũ."
        )

    # ========================================================
    # REUSE LAMBDA=0.50 FROM SENSITIVITY
    # ========================================================

    reused_now = 0

    for _, baseline_row in (
        selected.iterrows()
    ):

        workload_id = int(
            baseline_row[
                "workload_id"
            ]
        )

        if (
            workload_id
            in completed_ids
        ):
            continue

        source = (
            find_sensitivity_row(
                sensitivity_df,
                int(
                    baseline_row[
                        "number_of_deliveries"
                    ]
                ),
                baseline_row[
                    "wave_start"
                ],
                GUIDED_LAMBDA,
            )
        )

        if (
            source
            is None
        ):
            continue

        result = (
            sensitivity_guided_to_result(
                source,
                baseline_row,
            )
        )

        if (
            result
            is None
        ):
            continue

        append_result(
            result
        )

        completed_ids.add(
            workload_id
        )

        reused_now += 1

        print(
            "Reuse sensitivity: "
            f"workload_id="
            f"{workload_id}, "
            f"N="
            f"{int(baseline_row['number_of_deliveries'])}"
        )

    remaining = [
        int(
            workload_id
        )
        for workload_id
        in selected[
            "workload_id"
        ]
        if int(
            workload_id
        )
        not in completed_ids
    ]

    print()

    print(
        f"Sensitivity reused now : "
        f"{reused_now}"
    )

    print(
        f"Completed total        : "
        f"{len(completed_ids)}/"
        f"{len(selected)}"
    )

    print(
        f"New guided DQN runs    : "
        f"{len(remaining)}"
    )

    if remaining:

        print(
            "Remaining IDs          : "
            + ", ".join(
                map(
                    str,
                    remaining,
                )
            )
        )

    # ========================================================
    # TRAIN ONLY MISSING GUIDED DQN
    # ========================================================

    if remaining:

        service = (
            load_road_network()
        )

        (
            _,
            depot_node,
        ) = (
            get_depot_node(
                service
            )
        )

        matrix_builder = (
            RoadMatrixBuilder(
                service
            )
        )

        evaluator = (
            RouteEvaluator(
                road_network=service
            )
        )

        run_index = 0

        for _, baseline_row in (
            selected.iterrows()
        ):

            workload_id = int(
                baseline_row[
                    "workload_id"
                ]
            )

            if (
                workload_id
                in completed_ids
            ):
                continue

            run_index += 1

            workload_df = (
                build_workload_dataframe_from_manifest(
                    prepared_dataframe,
                    baseline_row,
                )
            )

            wave_start = (
                pd.Timestamp(
                    baseline_row[
                        "wave_start"
                    ]
                ).to_pydatetime()
            )

            wave_end = (
                pd.Timestamp(
                    baseline_row[
                        "wave_end"
                    ]
                ).to_pydatetime()
            )

            (
                deliveries,
                delivery_nodes,
            ) = (
                build_workload(
                    workload_df=(
                        workload_df
                    ),
                    delivery_lookup=(
                        delivery_lookup
                    ),
                )
            )

            delivery_ids = [
                str(
                    delivery.delivery_id
                )
                for delivery
                in deliveries
            ]

            expected_ids = (
                parse_delivery_ids(
                    baseline_row[
                        "manifest_delivery_ids"
                    ]
                )
            )

            if (
                delivery_ids
                != expected_ids
            ):
                raise ValueError(
                    "build_workload changed "
                    "delivery order at "
                    f"workload_id="
                    f"{workload_id}."
                )

            cluster_ids = (
                get_cluster_ids_for_deliveries(
                    deliveries=(
                        deliveries
                    ),
                    workload_df=(
                        workload_df
                    ),
                )
            )

            cluster_stats = (
                calculate_cluster_statistics(
                    cluster_ids
                )
            )

            cluster_lookup = dict(
                zip(
                    delivery_ids,
                    cluster_ids,
                )
            )

            print()
            print(
                "-" * 76
            )

            print(
                f"New run "
                f"{run_index}/"
                f"{len(remaining)} | "
                f"ID={workload_id} | "
                f"N={len(deliveries)} | "
                f"lambda=0.50"
            )

            print(
                f"Wave: "
                f"{wave_start}"
            )

            print(
                "Clusters="
                f"{cluster_stats['number_of_clusters']} | "
                "dominant_share="
                f"{cluster_stats['dominant_cluster_share']:.4f}"
            )

            timer = (
                perf_counter()
            )

            road_matrix = (
                matrix_builder.build(
                    (
                        depot_node,
                        *delivery_nodes,
                    )
                )
            )

            matrix_runtime = (
                perf_counter()
                - timer
            )

            print(
                f"Road matrix: "
                f"{matrix_runtime:.3f}s"
            )

            (
                rollout,
                training,
                training_runtime,
                route_runtime,
                environment,
            ) = (
                run_dqn(
                    deliveries=(
                        deliveries
                    ),
                    road_matrix=(
                        road_matrix
                    ),
                    start_time=(
                        wave_end
                    ),
                    cluster_ids=(
                        cluster_ids
                    ),
                    cluster_switch_penalty=(
                        GUIDED_LAMBDA
                    ),
                )
            )

            evaluation = (
                evaluator.evaluate(
                    algorithm=(
                        "kmeans_guided_dqn"
                    ),
                    depot_node=(
                        depot_node
                    ),
                    deliveries=(
                        deliveries
                    ),
                    delivery_nodes=(
                        delivery_nodes
                    ),
                    delivery_order=(
                        rollout.delivery_order
                    ),
                    start_time=(
                        wave_end
                    ),
                    return_to_depot=True,
                )
            )

            (
                switches,
                switch_rate,
                cluster_sequence,
            ) = (
                calculate_route_cluster_metrics(
                    delivery_order=(
                        rollout.delivery_order
                    ),
                    delivery_cluster_lookup=(
                        cluster_lookup
                    ),
                )
            )

            if (
                switches
                != environment.cluster_switches
            ):
                raise ValueError(
                    "Cluster switch mismatch: "
                    f"route="
                    f"{switches}, "
                    f"environment="
                    f"{environment.cluster_switches}"
                )

            row = {
                "workload_id":
                    workload_id,

                "wave_start":
                    wave_start,

                "wave_end":
                    wave_end,

                "number_of_deliveries":
                    evaluation
                    .number_of_deliveries,

                "cluster_switch_penalty":
                    GUIDED_LAMBDA,

                "number_of_clusters":
                    cluster_stats[
                        "number_of_clusters"
                    ],

                "dominant_cluster_size":
                    cluster_stats[
                        "dominant_cluster_size"
                    ],

                "dominant_cluster_share":
                    cluster_stats[
                        "dominant_cluster_share"
                    ],

                "total_distance_km":
                    evaluation
                    .total_distance_km,

                "total_travel_time_minutes":
                    evaluation
                    .total_travel_time_minutes,

                "on_time_deliveries":
                    evaluation
                    .on_time_deliveries,

                "late_deliveries":
                    evaluation
                    .late_deliveries,

                "on_time_rate":
                    evaluation
                    .on_time_rate,

                "total_lateness_minutes":
                    evaluation
                    .total_lateness_minutes,

                "max_lateness_minutes":
                    evaluation
                    .max_lateness_minutes,

                "training_runtime_seconds":
                    training_runtime,

                "route_runtime_seconds":
                    route_runtime,

                "matrix_preparation_seconds":
                    matrix_runtime,

                "greedy_reward":
                    rollout
                    .total_reward,

                "reward_per_delivery":
                    (
                        rollout
                        .total_reward
                        / len(
                            deliveries
                        )
                    ),

                "model_size":
                    training
                    .parameter_count,

                "final_epsilon":
                    training
                    .final_epsilon,

                "device":
                    training
                    .device,

                "cluster_switches":
                    switches,

                "cluster_switch_rate":
                    switch_rate,

                "cluster_sequence":
                    cluster_sequence,

                "delivery_order":
                    " -> ".join(
                        map(
                            str,
                            rollout
                            .delivery_order,
                        )
                    ),

                "result_source":
                    "new_training",
            }

            append_result(
                row
            )

            completed_ids.add(
                workload_id
            )

            baseline_distance = float(
                baseline_row[
                    "total_distance_km"
                ]
            )

            baseline_time = float(
                baseline_row[
                    "total_travel_time_minutes"
                ]
            )

            distance_change = (
                (
                    evaluation
                    .total_distance_km
                    - baseline_distance
                )
                / baseline_distance
                * 100.0
            )

            time_change = (
                (
                    evaluation
                    .total_travel_time_minutes
                    - baseline_time
                )
                / baseline_time
                * 100.0
            )

            print(
                f"Vanilla distance : "
                f"{baseline_distance:.4f} km"
            )

            print(
                f"Guided distance  : "
                f"{evaluation.total_distance_km:.4f} km"
            )

            print(
                f"Distance change  : "
                f"{distance_change:+.2f}%"
            )

            print(
                f"Vanilla time     : "
                f"{baseline_time:.4f} min"
            )

            print(
                f"Guided time      : "
                f"{evaluation.total_travel_time_minutes:.4f} min"
            )

            print(
                f"Time change      : "
                f"{time_change:+.2f}%"
            )

            print(
                f"On-time          : "
                f"{evaluation.on_time_rate:.4f}"
            )

            print(
                f"Cluster switches : "
                f"{switches}"
            )

            print(
                f"Switch rate      : "
                f"{switch_rate:.4f}"
            )

            print(
                f"Training         : "
                f"{training_runtime:.2f}s "
                f"({training.device})"
            )

            del environment
            del training
            del rollout

    # ========================================================
    # FINAL REPORTS
    # ========================================================

    if not GUIDED_OUTPUT_PATH.exists():
        raise RuntimeError(
            "Không có final guided result CSV."
        )

    guided_df = pd.read_csv(
        GUIDED_OUTPUT_PATH
    )

    guided_df[
        "workload_id"
    ] = pd.to_numeric(
        guided_df[
            "workload_id"
        ],
        errors="raise",
    ).astype(
        int
    )

    guided_df = (
        guided_df[
            guided_df[
                "workload_id"
            ].isin(
                selected_ids
            )
        ]
        .copy()
    )

    if guided_df[
        "workload_id"
    ].duplicated().any():
        raise RuntimeError(
            "Final guided output "
            "có workload_id trùng."
        )

    if (
        len(
            guided_df
        )
        != len(
            selected
        )
    ):

        found = set(
            guided_df[
                "workload_id"
            ].tolist()
        )

        missing = sorted(
            selected_ids
            - found
        )

        raise RuntimeError(
            "Final validation chưa đủ workload. "
            "Missing IDs: "
            + ", ".join(
                map(
                    str,
                    missing,
                )
            )
        )

    paired = (
        build_paired(
            selected,
            guided_df,
            sensitivity_df,
        )
    )

    summary = (
        build_summary(
            paired
        )
    )

    paired.to_csv(
        PAIRED_OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "FINAL PAIRED VALIDATION"
    )

    print(
        "=" * 76
    )

    print(
        paired[
            [
                "workload_id",
                "number_of_deliveries",
                "selection_role",
                "baseline_distance_km",
                "guided_distance_km",
                "distance_improvement_pct",
                "baseline_travel_time_minutes",
                "guided_travel_time_minutes",
                "time_improvement_pct",
                "baseline_on_time_rate",
                "guided_on_time_rate",
                "baseline_cluster_switch_rate",
                "guided_cluster_switch_rate",
                "result_source",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "=" * 76
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 76
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()

    print(
        "Lưu ý: cluster-switch comparison "
        "chỉ dùng các workload có vanilla "
        "route metrics từ sensitivity; "
        "distance/time/SLA dùng đủ 12 workload."
    )

    print()
    print(
        "Created:"
    )

    for path in [
        GUIDED_OUTPUT_PATH,
        PAIRED_OUTPUT_PATH,
        SUMMARY_OUTPUT_PATH,
        WORKLOAD_OUTPUT_PATH,
    ]:
        print(
            "-",
            path,
        )

    print()

    print(
        "Final K-Means-guided DQN "
        "validation complete."
    )


if __name__ == "__main__":
    main()