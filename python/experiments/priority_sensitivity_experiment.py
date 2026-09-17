from pathlib import Path
from itertools import combinations

import pandas as pd

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)

from domain import Delivery

from prioritization import (
    DeliveryPriorityModel,
    PriorityWeights,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

DETAIL_RESULTS_PATH = (
    RESULTS_DIR
    / "priority_sensitivity_all_waves_details.csv"
)

WAVE_SUMMARY_PATH = (
    RESULTS_DIR
    / "priority_sensitivity_wave_summary.csv"
)

GLOBAL_SUMMARY_PATH = (
    RESULTS_DIR
    / "priority_sensitivity_global_summary.csv"
)

MOST_SENSITIVE_WAVE_PATH = (
    RESULTS_DIR
    / "priority_most_sensitive_wave.csv"
)


WAVE_MINUTES = 60


WEIGHT_CONFIGURATIONS = {
    "deadline_focus": PriorityWeights(
        deadline_urgency=0.70,
        service_priority=0.15,
        waiting_time=0.15,
        travel_efficiency=0.00,
    ),

    "balanced": PriorityWeights(
        deadline_urgency=0.60,
        service_priority=0.20,
        waiting_time=0.20,
        travel_efficiency=0.00,
    ),

    "service_focus": PriorityWeights(
        deadline_urgency=0.45,
        service_priority=0.35,
        waiting_time=0.20,
        travel_efficiency=0.00,
    ),

    "waiting_focus": PriorityWeights(
        deadline_urgency=0.45,
        service_priority=0.20,
        waiting_time=0.35,
        travel_efficiency=0.00,
    ),
}


def load_delivery_dataframe() -> pd.DataFrame:
    dataframe = DataLoader.load_csv(
        DATASET_PATH
    )

    dataframe = (
        DataPreprocessor.prepare_deliveries(
            dataframe
        )
    )

    deliveries = (
        DeliveryMapper.from_dataframe(
            dataframe
        )
    )

    rows = []

    for delivery in deliveries:
        rows.append(
            {
                "delivery_id":
                    delivery.delivery_id,

                "latitude":
                    delivery.latitude,

                "longitude":
                    delivery.longitude,

                "weight":
                    delivery.weight,

                "created_at":
                    delivery.created_at,

                "expected_delivery_time":
                    delivery.expected_delivery_time,

                "delivered_at":
                    delivery.delivered_at,

                "service_type":
                    delivery.service_type,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise ValueError(
            "No deliveries found."
        )

    result = result.dropna(
        subset=[
            "created_at",
        ]
    ).reset_index(
        drop=True
    )

    return result


def assign_waves(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    frequency = (
        f"{WAVE_MINUTES}min"
    )

    dataframe[
        "wave_start"
    ] = dataframe[
        "created_at"
    ].dt.floor(
        frequency
    )

    dataframe[
        "wave_end"
    ] = (
        dataframe[
            "wave_start"
        ]
        + pd.Timedelta(
            minutes=WAVE_MINUTES
        )
    )

    return dataframe


def create_delivery_from_row(
    row: pd.Series,
) -> Delivery:
    return Delivery(
        delivery_id=str(
            row[
                "delivery_id"
            ]
        ),

        latitude=float(
            row[
                "latitude"
            ]
        ),

        longitude=float(
            row[
                "longitude"
            ]
        ),

        weight=(
            None
            if pd.isna(
                row[
                    "weight"
                ]
            )
            else float(
                row[
                    "weight"
                ]
            )
        ),

        created_at=row[
            "created_at"
        ],

        expected_delivery_time=row[
            "expected_delivery_time"
        ],

        delivered_at=row[
            "delivered_at"
        ],

        service_type=row[
            "service_type"
        ],
    )


def evaluate_configuration_for_wave(
    wave_df: pd.DataFrame,
    configuration_name: str,
    weights: PriorityWeights,
) -> pd.DataFrame:
    model = DeliveryPriorityModel(
        weights=weights
    )

    wave_start = (
        wave_df[
            "wave_start"
        ].iloc[0]
    )

    wave_end = (
        wave_df[
            "wave_end"
        ].iloc[0]
    )

    rows = []

    for _, row in wave_df.iterrows():
        delivery = (
            create_delivery_from_row(
                row
            )
        )

        result = model.calculate(
            delivery=delivery,
            current_time=wave_end,
        )

        rows.append(
            {
                "configuration":
                    configuration_name,

                "delivery_id":
                    result.delivery_id,

                "wave_start":
                    wave_start,

                "wave_end":
                    wave_end,

                "wave_size":
                    len(wave_df),

                "service_type":
                    delivery.service_type,

                "sla_progress":
                    result.sla_progress,

                "deadline_urgency":
                    result.deadline_urgency,

                "service_priority":
                    result.service_priority,

                "waiting_time_score":
                    result.waiting_time_score,

                "information_coverage":
                    result.information_coverage,

                "priority_score":
                    result.priority_score,

                "weight_deadline":
                    weights.deadline_urgency,

                "weight_service":
                    weights.service_priority,

                "weight_waiting":
                    weights.waiting_time,
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df = (
        result_df
        .sort_values(
            by=[
                "priority_score",
                "deadline_urgency",
                "waiting_time_score",
                "delivery_id",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    result_df[
        "priority_rank"
    ] = (
        result_df.index
        + 1
    )

    return result_df


def evaluate_all_waves(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    all_results = []

    grouped = dataframe.groupby(
        "wave_start",
        sort=True,
    )

    total_waves = (
        dataframe[
            "wave_start"
        ].nunique()
    )

    print()
    print(
        "========================================"
    )
    print(
        "Evaluating all delivery waves"
    )
    print(
        "========================================"
    )

    print(
        f"Total waves: {total_waves}"
    )

    for wave_number, (
        wave_start,
        wave_df,
    ) in enumerate(
        grouped,
        start=1,
    ):
        if (
            wave_number == 1
            or wave_number % 100 == 0
            or wave_number == total_waves
        ):
            print(
                f"Processing wave "
                f"{wave_number}/{total_waves}: "
                f"{wave_start}"
            )

        for (
            configuration_name,
            weights,
        ) in WEIGHT_CONFIGURATIONS.items():

            result_df = (
                evaluate_configuration_for_wave(
                    wave_df=wave_df,
                    configuration_name=(
                        configuration_name
                    ),
                    weights=weights,
                )
            )

            all_results.append(
                result_df
            )

    if not all_results:
        raise ValueError(
            "No sensitivity results generated."
        )

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def build_wave_rank_comparison(
    wave_details: pd.DataFrame,
) -> pd.DataFrame:
    rank_matrix = (
        wave_details.pivot(
            index="delivery_id",
            columns="configuration",
            values="priority_rank",
        )
    )

    rank_columns = list(
        WEIGHT_CONFIGURATIONS.keys()
    )

    rank_matrix = rank_matrix[
        rank_columns
    ]

    rank_matrix[
        "rank_min"
    ] = rank_matrix.min(
        axis=1
    )

    rank_matrix[
        "rank_max"
    ] = rank_matrix[
        rank_columns
    ].max(
        axis=1
    )

    rank_matrix[
        "rank_range"
    ] = (
        rank_matrix[
            "rank_max"
        ]
        - rank_matrix[
            "rank_min"
        ]
    )

    return rank_matrix


def calculate_wave_correlations(
    wave_details: pd.DataFrame,
) -> dict:
    rank_matrix = (
        wave_details.pivot(
            index="delivery_id",
            columns="configuration",
            values="priority_rank",
        )
    )

    correlation_values = []

    for config_a, config_b in combinations(
        WEIGHT_CONFIGURATIONS.keys(),
        2,
    ):
        if (
            config_a not in rank_matrix.columns
            or config_b not in rank_matrix.columns
        ):
            continue

        pair_df = rank_matrix[
            [
                config_a,
                config_b,
            ]
        ].dropna()

        if len(pair_df) < 2:
            continue

        correlation = (
            pair_df[
                config_a
            ].corr(
                pair_df[
                    config_b
                ],
                method="spearman",
            )
        )

        if pd.notna(
            correlation
        ):
            correlation_values.append(
                float(
                    correlation
                )
            )

    if not correlation_values:
        return {
            "mean_spearman": 1.0,
            "min_spearman": 1.0,
            "max_spearman": 1.0,
        }

    return {
        "mean_spearman":
            float(
                sum(
                    correlation_values
                )
                / len(
                    correlation_values
                )
            ),

        "min_spearman":
            float(
                min(
                    correlation_values
                )
            ),

        "max_spearman":
            float(
                max(
                    correlation_values
                )
            ),
    }


def build_wave_summary(
    details_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    grouped = details_df.groupby(
        "wave_start",
        sort=True,
    )

    for wave_start, wave_details in grouped:
        wave_size = int(
            wave_details[
                "wave_size"
            ].iloc[0]
        )

        comparison = (
            build_wave_rank_comparison(
                wave_details
            )
        )

        correlations = (
            calculate_wave_correlations(
                wave_details
            )
        )

        changed_orders = int(
            (
                comparison[
                    "rank_range"
                ]
                > 0
            ).sum()
        )

        unchanged_orders = int(
            (
                comparison[
                    "rank_range"
                ]
                == 0
            ).sum()
        )

        max_rank_change = float(
            comparison[
                "rank_range"
            ].max()
        )

        mean_rank_change = float(
            comparison[
                "rank_range"
            ].mean()
        )

        if wave_size > 0:
            changed_order_ratio = (
                changed_orders
                / wave_size
            )
        else:
            changed_order_ratio = 0.0

        rows.append(
            {
                "wave_start":
                    wave_start,

                "wave_end":
                    wave_details[
                        "wave_end"
                    ].iloc[0],

                "wave_size":
                    wave_size,

                "changed_orders":
                    changed_orders,

                "unchanged_orders":
                    unchanged_orders,

                "changed_order_ratio":
                    changed_order_ratio,

                "mean_rank_change":
                    mean_rank_change,

                "max_rank_change":
                    max_rank_change,

                "mean_spearman":
                    correlations[
                        "mean_spearman"
                    ],

                "min_spearman":
                    correlations[
                        "min_spearman"
                    ],

                "max_spearman":
                    correlations[
                        "max_spearman"
                    ],
            }
        )

    summary_df = pd.DataFrame(
        rows
    )

    summary_df = (
        summary_df
        .sort_values(
            by=[
                "max_rank_change",
                "changed_order_ratio",
                "min_spearman",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return summary_df


def build_global_summary(
    wave_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    total_waves = len(
        wave_summary_df
    )

    waves_with_changes = int(
        (
            wave_summary_df[
                "changed_orders"
            ]
            > 0
        ).sum()
    )

    waves_without_changes = (
        total_waves
        - waves_with_changes
    )

    change_ratio = (
        waves_with_changes
        / total_waves
        if total_waves > 0
        else 0.0
    )

    multi_order_waves = (
        wave_summary_df[
            wave_summary_df[
                "wave_size"
            ]
            >= 2
        ]
    )

    if not multi_order_waves.empty:
        mean_spearman = float(
            multi_order_waves[
                "mean_spearman"
            ].mean()
        )

        minimum_spearman = float(
            multi_order_waves[
                "min_spearman"
            ].min()
        )
    else:
        mean_spearman = 1.0
        minimum_spearman = 1.0

    global_max_rank_change = float(
        wave_summary_df[
            "max_rank_change"
        ].max()
    )

    mean_changed_order_ratio = float(
        wave_summary_df[
            "changed_order_ratio"
        ].mean()
    )

    summary = pd.DataFrame(
        [
            {
                "total_waves":
                    total_waves,

                "waves_with_rank_changes":
                    waves_with_changes,

                "waves_without_rank_changes":
                    waves_without_changes,

                "wave_change_ratio":
                    change_ratio,

                "mean_changed_order_ratio":
                    mean_changed_order_ratio,

                "global_max_rank_change":
                    global_max_rank_change,

                "mean_spearman":
                    mean_spearman,

                "minimum_spearman":
                    minimum_spearman,
            }
        ]
    )

    return summary


def find_most_sensitive_wave(
    wave_summary_df: pd.DataFrame,
) -> pd.Series:
    if wave_summary_df.empty:
        raise ValueError(
            "Wave summary is empty."
        )

    return wave_summary_df.iloc[
        0
    ]


def extract_most_sensitive_wave_details(
    details_df: pd.DataFrame,
    wave_start,
) -> pd.DataFrame:
    sensitive_df = details_df[
        details_df[
            "wave_start"
        ]
        == wave_start
    ].copy()

    comparison = (
        sensitive_df.pivot(
            index="delivery_id",
            columns="configuration",
            values="priority_rank",
        )
    )

    comparison = comparison.rename(
        columns={
            name:
                f"rank_{name}"
            for name
            in WEIGHT_CONFIGURATIONS
        }
    )

    comparison = comparison.reset_index()

    score_matrix = (
        sensitive_df.pivot(
            index="delivery_id",
            columns="configuration",
            values="priority_score",
        )
    )

    score_matrix = score_matrix.rename(
        columns={
            name:
                f"score_{name}"
            for name
            in WEIGHT_CONFIGURATIONS
        }
    )

    score_matrix = score_matrix.reset_index()

    result = comparison.merge(
        score_matrix,
        on="delivery_id",
        how="left",
    )

    rank_columns = [
        f"rank_{name}"
        for name
        in WEIGHT_CONFIGURATIONS
    ]

    result[
        "rank_min"
    ] = result[
        rank_columns
    ].min(
        axis=1
    )

    result[
        "rank_max"
    ] = result[
        rank_columns
    ].max(
        axis=1
    )

    result[
        "rank_range"
    ] = (
        result[
            "rank_max"
        ]
        - result[
            "rank_min"
        ]
    )

    metadata = (
        sensitive_df[
            [
                "delivery_id",
                "service_type",
                "sla_progress",
                "deadline_urgency",
                "service_priority",
                "waiting_time_score",
                "information_coverage",
            ]
        ]
        .drop_duplicates(
            subset=[
                "delivery_id",
            ]
        )
    )

    result = result.merge(
        metadata,
        on="delivery_id",
        how="left",
    )

    result = (
        result
        .sort_values(
            by=[
                "rank_range",
                "rank_min",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return result


def print_global_summary(
    global_summary_df: pd.DataFrame,
) -> None:
    row = global_summary_df.iloc[
        0
    ]

    print()
    print(
        "========================================"
    )
    print(
        "Global Sensitivity Summary"
    )
    print(
        "========================================"
    )

    print(
        f"Total waves: "
        f"{int(row['total_waves'])}"
    )

    print(
        f"Waves with ranking changes: "
        f"{int(row['waves_with_rank_changes'])}"
    )

    print(
        f"Waves without ranking changes: "
        f"{int(row['waves_without_rank_changes'])}"
    )

    print(
        f"Wave change ratio: "
        f"{row['wave_change_ratio']:.4f}"
    )

    print(
        f"Mean changed-order ratio: "
        f"{row['mean_changed_order_ratio']:.4f}"
    )

    print(
        f"Global max rank change: "
        f"{row['global_max_rank_change']:.0f}"
    )

    print(
        f"Mean Spearman: "
        f"{row['mean_spearman']:.4f}"
    )

    print(
        f"Minimum Spearman: "
        f"{row['minimum_spearman']:.4f}"
    )


def print_most_sensitive_wave(
    wave_summary_df: pd.DataFrame,
    sensitive_details_df: pd.DataFrame,
) -> None:
    wave = (
        find_most_sensitive_wave(
            wave_summary_df
        )
    )

    print()
    print(
        "========================================"
    )
    print(
        "Most Sensitive Delivery Wave"
    )
    print(
        "========================================"
    )

    print(
        f"Wave start: "
        f"{wave['wave_start']}"
    )

    print(
        f"Wave end: "
        f"{wave['wave_end']}"
    )

    print(
        f"Orders: "
        f"{int(wave['wave_size'])}"
    )

    print(
        f"Changed orders: "
        f"{int(wave['changed_orders'])}"
    )

    print(
        f"Changed-order ratio: "
        f"{wave['changed_order_ratio']:.4f}"
    )

    print(
        f"Mean rank change: "
        f"{wave['mean_rank_change']:.4f}"
    )

    print(
        f"Max rank change: "
        f"{wave['max_rank_change']:.0f}"
    )

    print(
        f"Mean Spearman: "
        f"{wave['mean_spearman']:.4f}"
    )

    print(
        f"Minimum Spearman: "
        f"{wave['min_spearman']:.4f}"
    )

    print()
    print(
        "Largest ranking changes:"
    )

    columns = [
        "delivery_id",
        "service_type",
        "sla_progress",
        "deadline_urgency",
        "waiting_time_score",
        "rank_deadline_focus",
        "rank_balanced",
        "rank_service_focus",
        "rank_waiting_focus",
        "rank_range",
    ]

    print(
        sensitive_details_df[
            columns
        ]
        .head(
            20
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


def print_top_sensitive_waves(
    wave_summary_df: pd.DataFrame,
) -> None:
    print()
    print(
        "========================================"
    )
    print(
        "Top 10 Sensitive Waves"
    )
    print(
        "========================================"
    )

    columns = [
        "wave_start",
        "wave_size",
        "changed_orders",
        "changed_order_ratio",
        "mean_rank_change",
        "max_rank_change",
        "mean_spearman",
        "min_spearman",
    ]

    print(
        wave_summary_df[
            columns
        ]
        .head(
            10
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


def save_results(
    details_df: pd.DataFrame,
    wave_summary_df: pd.DataFrame,
    global_summary_df: pd.DataFrame,
    sensitive_details_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    details_df.to_csv(
        DETAIL_RESULTS_PATH,
        index=False,
    )

    wave_summary_df.to_csv(
        WAVE_SUMMARY_PATH,
        index=False,
    )

    global_summary_df.to_csv(
        GLOBAL_SUMMARY_PATH,
        index=False,
    )

    sensitive_details_df.to_csv(
        MOST_SENSITIVE_WAVE_PATH,
        index=False,
    )

    print()
    print(
        "========================================"
    )
    print(
        "Results Saved"
    )
    print(
        "========================================"
    )

    print(
        DETAIL_RESULTS_PATH
    )

    print(
        WAVE_SUMMARY_PATH
    )

    print(
        GLOBAL_SUMMARY_PATH
    )

    print(
        MOST_SENSITIVE_WAVE_PATH
    )


def main() -> None:
    print()
    print(
        "========================================"
    )
    print(
        "Priority Sensitivity Experiment V2"
    )
    print(
        "========================================"
    )

    dataframe = (
        load_delivery_dataframe()
    )

    dataframe = (
        assign_waves(
            dataframe
        )
    )

    print(
        f"Deliveries: "
        f"{len(dataframe)}"
    )

    print(
        f"Wave duration: "
        f"{WAVE_MINUTES} minutes"
    )

    print(
        f"Unique waves: "
        f"{dataframe['wave_start'].nunique()}"
    )

    details_df = (
        evaluate_all_waves(
            dataframe
        )
    )

    wave_summary_df = (
        build_wave_summary(
            details_df
        )
    )

    global_summary_df = (
        build_global_summary(
            wave_summary_df
        )
    )

    most_sensitive_wave = (
        find_most_sensitive_wave(
            wave_summary_df
        )
    )

    sensitive_details_df = (
        extract_most_sensitive_wave_details(
            details_df=details_df,
            wave_start=(
                most_sensitive_wave[
                    "wave_start"
                ]
            ),
        )
    )

    print_global_summary(
        global_summary_df
    )

    print_top_sensitive_waves(
        wave_summary_df
    )

    print_most_sensitive_wave(
        wave_summary_df,
        sensitive_details_df,
    )

    save_results(
        details_df=details_df,
        wave_summary_df=wave_summary_df,
        global_summary_df=global_summary_df,
        sensitive_details_df=(
            sensitive_details_df
        ),
    )


if __name__ == "__main__":
    main()