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

SUMMARY_PATH = (
    RESULTS_DIR
    / "wave_duration_experiment_summary.csv"
)

DETAIL_PATH = (
    RESULTS_DIR
    / "wave_duration_experiment_details.csv"
)


WAVE_DURATIONS = [
    30,
    60,
    120,
    180,
]


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


def assign_waves(
    dataframe: pd.DataFrame,
    wave_minutes: int,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    frequency = (
        f"{wave_minutes}min"
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
            minutes=wave_minutes
        )
    )

    return dataframe


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

                "priority_score":
                    result.priority_score,

                "deadline_urgency":
                    result.deadline_urgency,

                "service_priority":
                    result.service_priority,

                "waiting_time_score":
                    result.waiting_time_score,

                "information_coverage":
                    result.information_coverage,
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


def evaluate_wave(
    wave_df: pd.DataFrame,
) -> pd.DataFrame:
    all_results = []

    for (
        configuration_name,
        weights,
    ) in WEIGHT_CONFIGURATIONS.items():

        result_df = (
            evaluate_configuration_for_wave(
                wave_df=wave_df,
                configuration_name=configuration_name,
                weights=weights,
            )
        )

        all_results.append(
            result_df
        )

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def build_rank_matrix(
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

    return rank_matrix[
        rank_columns
    ]


def calculate_wave_sensitivity(
    wave_details: pd.DataFrame,
) -> dict:
    rank_matrix = (
        build_rank_matrix(
            wave_details
        )
    )

    rank_min = rank_matrix.min(
        axis=1
    )

    rank_max = rank_matrix.max(
        axis=1
    )

    rank_range = (
        rank_max
        - rank_min
    )

    changed_orders = int(
        (
            rank_range > 0
        ).sum()
    )

    wave_size = len(
        rank_matrix
    )

    changed_order_ratio = (
        changed_orders
        / wave_size
        if wave_size > 0
        else 0.0
    )

    max_rank_change = float(
        rank_range.max()
        if not rank_range.empty
        else 0.0
    )

    mean_rank_change = float(
        rank_range.mean()
        if not rank_range.empty
        else 0.0
    )

    correlations = []

    for config_a, config_b in combinations(
        WEIGHT_CONFIGURATIONS.keys(),
        2,
    ):
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
            correlations.append(
                float(
                    correlation
                )
            )

    if correlations:
        mean_spearman = float(
            sum(
                correlations
            )
            / len(
                correlations
            )
        )

        minimum_spearman = float(
            min(
                correlations
            )
        )
    else:
        mean_spearman = 1.0
        minimum_spearman = 1.0

    return {
        "changed_orders":
            changed_orders,

        "changed_order_ratio":
            changed_order_ratio,

        "mean_rank_change":
            mean_rank_change,

        "max_rank_change":
            max_rank_change,

        "mean_spearman":
            mean_spearman,

        "minimum_spearman":
            minimum_spearman,
    }


def evaluate_wave_duration(
    dataframe: pd.DataFrame,
    wave_minutes: int,
) -> tuple[dict, pd.DataFrame]:
    wave_df = (
        assign_waves(
            dataframe,
            wave_minutes=wave_minutes,
        )
    )

    grouped = wave_df.groupby(
        "wave_start",
        sort=True,
    )

    total_waves = (
        wave_df[
            "wave_start"
        ].nunique()
    )

    wave_sizes = (
        grouped
        .size()
    )

    detail_rows = []

    total_changed_waves = 0

    changed_order_ratios = []
    mean_rank_changes = []
    max_rank_changes = []
    mean_spearmans = []
    minimum_spearmans = []

    print()
    print(
        "----------------------------------------"
    )

    print(
        f"Evaluating {wave_minutes}-minute waves"
    )

    print(
        f"Total waves: {total_waves}"
    )

    for wave_number, (
        wave_start,
        current_wave_df,
    ) in enumerate(
        grouped,
        start=1,
    ):
        if (
            wave_number == 1
            or wave_number % 200 == 0
            or wave_number == total_waves
        ):
            print(
                f"Processing "
                f"{wave_number}/{total_waves}"
            )

        wave_details = (
            evaluate_wave(
                current_wave_df
            )
        )

        sensitivity = (
            calculate_wave_sensitivity(
                wave_details
            )
        )

        if (
            sensitivity[
                "changed_orders"
            ]
            > 0
        ):
            total_changed_waves += 1

        changed_order_ratios.append(
            sensitivity[
                "changed_order_ratio"
            ]
        )

        mean_rank_changes.append(
            sensitivity[
                "mean_rank_change"
            ]
        )

        max_rank_changes.append(
            sensitivity[
                "max_rank_change"
            ]
        )

        if len(
            current_wave_df
        ) >= 2:
            mean_spearmans.append(
                sensitivity[
                    "mean_spearman"
                ]
            )

            minimum_spearmans.append(
                sensitivity[
                    "minimum_spearman"
                ]
            )

        detail_rows.append(
            {
                "wave_minutes":
                    wave_minutes,

                "wave_start":
                    wave_start,

                "wave_size":
                    len(
                        current_wave_df
                    ),

                **sensitivity,
            }
        )

    waves_ge_2 = int(
        (
            wave_sizes >= 2
        ).sum()
    )

    waves_ge_5 = int(
        (
            wave_sizes >= 5
        ).sum()
    )

    waves_ge_10 = int(
        (
            wave_sizes >= 10
        ).sum()
    )

    summary = {
        "wave_minutes":
            wave_minutes,

        "total_waves":
            int(
                total_waves
            ),

        "mean_orders_per_wave":
            float(
                wave_sizes.mean()
            ),

        "median_orders_per_wave":
            float(
                wave_sizes.median()
            ),

        "min_orders_per_wave":
            int(
                wave_sizes.min()
            ),

        "max_orders_per_wave":
            int(
                wave_sizes.max()
            ),

        "waves_ge_2":
            waves_ge_2,

        "waves_ge_2_ratio":
            (
                waves_ge_2
                / total_waves
            ),

        "waves_ge_5":
            waves_ge_5,

        "waves_ge_5_ratio":
            (
                waves_ge_5
                / total_waves
            ),

        "waves_ge_10":
            waves_ge_10,

        "waves_ge_10_ratio":
            (
                waves_ge_10
                / total_waves
            ),

        "waves_with_rank_changes":
            total_changed_waves,

        "wave_change_ratio":
            (
                total_changed_waves
                / total_waves
            ),

        "mean_changed_order_ratio":
            float(
                sum(
                    changed_order_ratios
                )
                / len(
                    changed_order_ratios
                )
            ),

        "mean_rank_change":
            float(
                sum(
                    mean_rank_changes
                )
                / len(
                    mean_rank_changes
                )
            ),

        "global_max_rank_change":
            float(
                max(
                    max_rank_changes
                )
            ),

        "mean_spearman":
            (
                float(
                    sum(
                        mean_spearmans
                    )
                    / len(
                        mean_spearmans
                    )
                )
                if mean_spearmans
                else 1.0
            ),

        "minimum_spearman":
            (
                float(
                    min(
                        minimum_spearmans
                    )
                )
                if minimum_spearmans
                else 1.0
            ),
    }

    details_df = pd.DataFrame(
        detail_rows
    )

    return (
        summary,
        details_df,
    )


def run_experiment():
    print()
    print(
        "========================================"
    )
    print(
        "Wave Duration Experiment"
    )
    print(
        "========================================"
    )

    dataframe = (
        load_delivery_dataframe()
    )

    print(
        f"Deliveries: "
        f"{len(dataframe)}"
    )

    print(
        f"Durations: "
        f"{WAVE_DURATIONS}"
    )

    summaries = []
    detail_frames = []

    for wave_minutes in WAVE_DURATIONS:
        (
            summary,
            details_df,
        ) = evaluate_wave_duration(
            dataframe=dataframe,
            wave_minutes=wave_minutes,
        )

        summaries.append(
            summary
        )

        detail_frames.append(
            details_df
        )

    summary_df = pd.DataFrame(
        summaries
    )

    details_df = pd.concat(
        detail_frames,
        ignore_index=True,
    )

    return (
        summary_df,
        details_df,
    )


def print_summary(
    summary_df: pd.DataFrame,
) -> None:
    print()
    print(
        "========================================"
    )
    print(
        "Wave Duration Summary"
    )
    print(
        "========================================"
    )

    columns = [
        "wave_minutes",
        "total_waves",
        "mean_orders_per_wave",
        "median_orders_per_wave",
        "max_orders_per_wave",
        "waves_ge_2_ratio",
        "waves_ge_5_ratio",
        "waves_ge_10_ratio",
        "waves_with_rank_changes",
        "wave_change_ratio",
        "mean_changed_order_ratio",
        "global_max_rank_change",
        "mean_spearman",
        "minimum_spearman",
    ]

    print(
        summary_df[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


def save_results(
    summary_df: pd.DataFrame,
    details_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    details_df.to_csv(
        DETAIL_PATH,
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
        SUMMARY_PATH
    )

    print(
        DETAIL_PATH
    )


def main() -> None:
    (
        summary_df,
        details_df,
    ) = run_experiment()

    print_summary(
        summary_df
    )

    save_results(
        summary_df,
        details_df,
    )


if __name__ == "__main__":
    main()