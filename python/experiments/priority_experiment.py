from pathlib import Path

import pandas as pd

from data import DataLoader, DataPreprocessor, DeliveryMapper
from prioritization import DeliveryPriorityModel


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

RESULTS_PATH = (
    RESULTS_DIR
    / "xedu_priority_wave_results.csv"
)

WAVE_MINUTES = 60


def load_deliveries():
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

    return dataframe, deliveries


def build_delivery_dataframe(
    deliveries,
) -> pd.DataFrame:
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

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:
        raise ValueError(
            "No deliveries available."
        )

    if (
        dataframe["created_at"]
        .isna()
        .all()
    ):
        raise ValueError(
            "No valid created_at timestamps."
        )

    return dataframe


def assign_delivery_waves(
    dataframe: pd.DataFrame,
    wave_minutes: int = WAVE_MINUTES,
) -> pd.DataFrame:
    """
    Group orders into planning waves.

    Example for 60-minute waves:

        created_at 08:10
            ->
        wave_start 08:00
        wave_end   09:00

    Priority is evaluated at wave_end.
    """

    if wave_minutes <= 0:
        raise ValueError(
            "wave_minutes must be greater than 0."
        )

    dataframe = dataframe.copy()

    dataframe = dataframe.dropna(
        subset=[
            "created_at",
        ]
    ).reset_index(
        drop=True
    )

    wave_frequency = (
        f"{wave_minutes}min"
    )

    dataframe[
        "wave_start"
    ] = dataframe[
        "created_at"
    ].dt.floor(
        wave_frequency
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


def calculate_wave_priorities(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate priority independently inside each wave.
    """

    model = DeliveryPriorityModel()

    result_rows = []

    grouped = dataframe.groupby(
        "wave_start",
        sort=True,
    )

    for wave_start, wave_df in grouped:
        wave_end = (
            wave_df[
                "wave_end"
            ].iloc[0]
        )

        wave_rows = []

        for _, row in wave_df.iterrows():
            from domain import Delivery

            delivery = Delivery(
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

            result = model.calculate(
                delivery=delivery,
                current_time=wave_end,
            )

            wave_rows.append(
                {
                    "delivery_id":
                        result.delivery_id,

                    "wave_start":
                        wave_start,

                    "wave_end":
                        wave_end,

                    "service_type":
                        delivery.service_type,

                    "created_at":
                        delivery.created_at,

                    "expected_delivery_time":
                        delivery.expected_delivery_time,

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

                    "base_priority_score":
                        result.base_priority_score,

                    "priority_score":
                        result.priority_score,
                }
            )

        wave_result_df = pd.DataFrame(
            wave_rows
        )

        wave_result_df = (
            wave_result_df
            .sort_values(
                by=[
                    "priority_score",
                    "deadline_urgency",
                    "waiting_time_score",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        wave_result_df[
            "priority_rank_in_wave"
        ] = (
            wave_result_df.index
            + 1
        )

        wave_result_df[
            "wave_size"
        ] = len(
            wave_result_df
        )

        result_rows.extend(
            wave_result_df
            .to_dict(
                orient="records"
            )
        )

    results_df = pd.DataFrame(
        result_rows
    )

    return results_df


def print_global_summary(
    results_df: pd.DataFrame,
) -> None:
    print()
    print(
        "========================================"
    )
    print(
        "XeDu Wave Priority Experiment"
    )
    print(
        "========================================"
    )

    print(
        f"Total evaluated deliveries: "
        f"{len(results_df)}"
    )

    print(
        f"Total waves: "
        f"{results_df['wave_start'].nunique()}"
    )

    print(
        f"Wave duration: "
        f"{WAVE_MINUTES} minutes"
    )

    print()
    print(
        "Wave size statistics:"
    )

    wave_sizes = (
        results_df[
            [
                "wave_start",
                "wave_size",
            ]
        ]
        .drop_duplicates()
        [
            "wave_size"
        ]
    )

    print(
        wave_sizes.describe()
    )

    print()
    print(
        "Priority score statistics:"
    )

    print(
        results_df[
            "priority_score"
        ].describe()
    )

    print()
    print(
        "SLA progress statistics:"
    )

    print(
        results_df[
            "sla_progress"
        ].describe()
    )

    print()
    print(
        "Missing deadline count:"
    )

    print(
        results_df[
            "deadline_urgency"
        ].isna().sum()
    )


def print_largest_wave(
    results_df: pd.DataFrame,
    top_n: int = 20,
) -> None:
    wave_sizes = (
        results_df
        .groupby(
            "wave_start"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    largest_wave_start = (
        wave_sizes.index[0]
    )

    largest_wave = (
        results_df[
            results_df[
                "wave_start"
            ]
            == largest_wave_start
        ]
        .sort_values(
            by="priority_rank_in_wave"
        )
    )

    print()
    print(
        "========================================"
    )
    print(
        "Largest Delivery Wave"
    )
    print(
        "========================================"
    )

    print(
        f"Wave start: "
        f"{largest_wave_start}"
    )

    print(
        f"Wave end: "
        f"{largest_wave['wave_end'].iloc[0]}"
    )

    print(
        f"Orders: "
        f"{len(largest_wave)}"
    )

    print()
    print(
        f"Top {top_n} priority deliveries:"
    )

    columns = [
        "priority_rank_in_wave",
        "delivery_id",
        "service_type",
        "sla_progress",
        "deadline_urgency",
        "service_priority",
        "waiting_time_score",
        "information_coverage",
        "priority_score",
    ]

    print(
        largest_wave[
            columns
        ]
        .head(
            top_n
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "Top service-type distribution:"
    )

    print(
        largest_wave
        .head(
            top_n
        )[
            "service_type"
        ]
        .value_counts(
            dropna=False
        )
    )


def save_results(
    results_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_PATH
    )


def main() -> None:
    _, deliveries = load_deliveries()

    delivery_df = (
        build_delivery_dataframe(
            deliveries
        )
    )

    wave_df = (
        assign_delivery_waves(
            delivery_df,
            wave_minutes=WAVE_MINUTES,
        )
    )

    results_df = (
        calculate_wave_priorities(
            wave_df
        )
    )

    print_global_summary(
        results_df
    )

    print_largest_wave(
        results_df,
        top_n=20,
    )

    save_results(
        results_df
    )


if __name__ == "__main__":
    main()