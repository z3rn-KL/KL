from pathlib import Path

import numpy as np
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
    / "priority_weight_grid_summary.csv"
)

DETAIL_PATH = (
    RESULTS_DIR
    / "priority_weight_grid_details.csv"
)


WAVE_MINUTES = 120

WEIGHT_STEP = 0.05

DEADLINE_MIN = 0.40
DEADLINE_MAX = 0.70

SERVICE_MIN = 0.10
SERVICE_MAX = 0.40

WAITING_MIN = 0.10
WAITING_MAX = 0.40


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


def generate_weight_grid():
    """
    Generate candidate weight combinations.

    Requirements:
        deadline + service + waiting = 1.0

    travel weight remains 0 because real road
    information has not been integrated yet.
    """

    configurations = []

    deadline_values = np.arange(
        DEADLINE_MIN,
        DEADLINE_MAX + 0.001,
        WEIGHT_STEP,
    )

    service_values = np.arange(
        SERVICE_MIN,
        SERVICE_MAX + 0.001,
        WEIGHT_STEP,
    )

    for deadline_weight in deadline_values:
        for service_weight in service_values:

            waiting_weight = (
                1.0
                - deadline_weight
                - service_weight
            )

            if (
                waiting_weight
                < WAITING_MIN - 1e-9
                or waiting_weight
                > WAITING_MAX + 1e-9
            ):
                continue

            deadline_weight = round(
                float(deadline_weight),
                2,
            )

            service_weight = round(
                float(service_weight),
                2,
            )

            waiting_weight = round(
                float(waiting_weight),
                2,
            )

            configuration_name = (
                f"D{int(deadline_weight * 100):02d}_"
                f"S{int(service_weight * 100):02d}_"
                f"W{int(waiting_weight * 100):02d}"
            )

            configurations.append(
                (
                    configuration_name,
                    PriorityWeights(
                        deadline_urgency=deadline_weight,
                        service_priority=service_weight,
                        waiting_time=waiting_weight,
                        travel_efficiency=0.0,
                    ),
                )
            )

    return configurations


def calculate_priority_rows(
    dataframe: pd.DataFrame,
    configuration_name: str,
    weights: PriorityWeights,
) -> pd.DataFrame:

    model = DeliveryPriorityModel(
        weights=weights
    )

    rows = []

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

            delivery = (
                create_delivery_from_row(
                    row
                )
            )

            result = model.calculate(
                delivery=delivery,
                current_time=wave_end,
            )

            wave_rows.append(
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
                        len(
                            wave_df
                        ),

                    "service_type":
                        delivery.service_type,

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
                }
            )

        wave_result = pd.DataFrame(
            wave_rows
        )

        wave_result = (
            wave_result
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

        wave_result[
            "priority_rank"
        ] = (
            wave_result.index
            + 1
        )

        rows.extend(
            wave_result.to_dict(
                orient="records"
            )
        )

    return pd.DataFrame(
        rows
    )


def safe_spearman(
    dataframe: pd.DataFrame,
    left_column: str,
    right_column: str,
) -> float:

    valid = dataframe[
        [
            left_column,
            right_column,
        ]
    ].dropna()

    if len(valid) < 2:
        return 0.0

    if (
        valid[left_column].nunique() < 2
        or valid[right_column].nunique() < 2
    ):
        return 0.0

    correlation = (
        valid[
            left_column
        ].corr(
            valid[
                right_column
            ],
            method="spearman",
        )
    )

    if pd.isna(
        correlation
    ):
        return 0.0

    return float(
        correlation
    )


def calculate_component_alignment(
    configuration_df: pd.DataFrame,
) -> dict:
    """
    Measure how strongly final priority follows
    each individual component.

    High positive correlation means the final
    priority ranking strongly follows that factor.
    """

    deadline_alignment = (
        safe_spearman(
            configuration_df,
            "priority_score",
            "deadline_urgency",
        )
    )

    service_alignment = (
        safe_spearman(
            configuration_df,
            "priority_score",
            "service_priority",
        )
    )

    waiting_alignment = (
        safe_spearman(
            configuration_df,
            "priority_score",
            "waiting_time_score",
        )
    )

    alignments = np.array(
        [
            deadline_alignment,
            service_alignment,
            waiting_alignment,
        ],
        dtype=float,
    )

    return {
        "deadline_alignment":
            deadline_alignment,

        "service_alignment":
            service_alignment,

        "waiting_alignment":
            waiting_alignment,

        "mean_alignment":
            float(
                alignments.mean()
            ),

        "alignment_std":
            float(
                alignments.std()
            ),

        "alignment_range":
            float(
                alignments.max()
                - alignments.min()
            ),
    }


def calculate_top_quartile_metrics(
    configuration_df: pd.DataFrame,
) -> dict:
    """
    Analyze deliveries placed in the top 25%
    of each delivery wave.
    """

    selected_frames = []

    grouped = configuration_df.groupby(
        "wave_start",
        sort=False,
    )

    for _, wave_df in grouped:

        wave_size = len(
            wave_df
        )

        top_count = max(
            1,
            int(
                np.ceil(
                    wave_size
                    * 0.25
                )
            ),
        )

        top_df = (
            wave_df
            .sort_values(
                "priority_rank"
            )
            .head(
                top_count
            )
        )

        selected_frames.append(
            top_df
        )

    if not selected_frames:
        return {
            "top25_mean_deadline_urgency":
                0.0,

            "top25_mean_service_priority":
                0.0,

            "top25_mean_waiting_score":
                0.0,
        }

    top_df = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    return {
        "top25_mean_deadline_urgency":
            float(
                top_df[
                    "deadline_urgency"
                ].mean()
            ),

        "top25_mean_service_priority":
            float(
                top_df[
                    "service_priority"
                ].mean()
            ),

        "top25_mean_waiting_score":
            float(
                top_df[
                    "waiting_time_score"
                ].mean()
            ),
    }


def calculate_score_spread(
    configuration_df: pd.DataFrame,
) -> dict:
    priority_scores = (
        configuration_df[
            "priority_score"
        ]
    )

    return {
        "mean_priority_score":
            float(
                priority_scores.mean()
            ),

        "std_priority_score":
            float(
                priority_scores.std()
            ),

        "min_priority_score":
            float(
                priority_scores.min()
            ),

        "max_priority_score":
            float(
                priority_scores.max()
            ),
    }


def normalize_column(
    series: pd.Series,
) -> pd.Series:
    minimum = series.min()
    maximum = series.max()

    if np.isclose(
        minimum,
        maximum,
    ):
        return pd.Series(
            np.ones(
                len(
                    series
                )
            ),
            index=series.index,
        )

    return (
        series
        - minimum
    ) / (
        maximum
        - minimum
    )


def build_balance_score(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Experimental balance score.

    Desired properties:

    1. Good mean alignment with all three factors.
    2. Low difference between factor alignments.
    3. Good priority-score spread.

    This score is ONLY an experimental selection aid.
    It is not treated as ground truth.
    """

    summary_df = summary_df.copy()

    summary_df[
        "normalized_mean_alignment"
    ] = normalize_column(
        summary_df[
            "mean_alignment"
        ]
    )

    summary_df[
        "normalized_balance"
    ] = (
        1.0
        - normalize_column(
            summary_df[
                "alignment_range"
            ]
        )
    )

    summary_df[
        "normalized_score_spread"
    ] = normalize_column(
        summary_df[
            "std_priority_score"
        ]
    )

    summary_df[
        "balance_score"
    ] = (
        0.40
        * summary_df[
            "normalized_mean_alignment"
        ]
        + 0.40
        * summary_df[
            "normalized_balance"
        ]
        + 0.20
        * summary_df[
            "normalized_score_spread"
        ]
    )

    return summary_df


def evaluate_configuration(
    dataframe: pd.DataFrame,
    configuration_name: str,
    weights: PriorityWeights,
):
    details_df = (
        calculate_priority_rows(
            dataframe=dataframe,
            configuration_name=configuration_name,
            weights=weights,
        )
    )

    alignment = (
        calculate_component_alignment(
            details_df
        )
    )

    top_metrics = (
        calculate_top_quartile_metrics(
            details_df
        )
    )

    spread = (
        calculate_score_spread(
            details_df
        )
    )

    summary = {
        "configuration":
            configuration_name,

        "deadline_weight":
            weights.deadline_urgency,

        "service_weight":
            weights.service_priority,

        "waiting_weight":
            weights.waiting_time,

        **alignment,

        **top_metrics,

        **spread,
    }

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
        "Priority Weight Grid Experiment"
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

    configurations = (
        generate_weight_grid()
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

    print(
        f"Weight configurations: "
        f"{len(configurations)}"
    )

    summaries = []
    detail_frames = []

    for index, (
        configuration_name,
        weights,
    ) in enumerate(
        configurations,
        start=1,
    ):
        print(
            f"Processing "
            f"{index}/{len(configurations)}: "
            f"{configuration_name}"
        )

        (
            summary,
            details_df,
        ) = evaluate_configuration(
            dataframe=dataframe,
            configuration_name=configuration_name,
            weights=weights,
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

    summary_df = (
        build_balance_score(
            summary_df
        )
    )

    summary_df = (
        summary_df
        .sort_values(
            by=[
                "balance_score",
                "alignment_range",
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

    summary_df[
        "candidate_rank"
    ] = (
        summary_df.index
        + 1
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
        "Top 15 Balanced Weight Candidates"
    )
    print(
        "========================================"
    )

    columns = [
        "candidate_rank",
        "configuration",
        "deadline_weight",
        "service_weight",
        "waiting_weight",
        "deadline_alignment",
        "service_alignment",
        "waiting_alignment",
        "mean_alignment",
        "alignment_range",
        "std_priority_score",
        "balance_score",
    ]

    print(
        summary_df[
            columns
        ]
        .head(
            15
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
        "========================================"
    )
    print(
        "Top Candidate Details"
    )
    print(
        "========================================"
    )

    best = summary_df.iloc[
        0
    ]

    print(
        f"Configuration: "
        f"{best['configuration']}"
    )

    print(
        f"Deadline weight: "
        f"{best['deadline_weight']:.2f}"
    )

    print(
        f"Service weight: "
        f"{best['service_weight']:.2f}"
    )

    print(
        f"Waiting weight: "
        f"{best['waiting_weight']:.2f}"
    )

    print(
        f"Deadline alignment: "
        f"{best['deadline_alignment']:.4f}"
    )

    print(
        f"Service alignment: "
        f"{best['service_alignment']:.4f}"
    )

    print(
        f"Waiting alignment: "
        f"{best['waiting_alignment']:.4f}"
    )

    print(
        f"Alignment range: "
        f"{best['alignment_range']:.4f}"
    )

    print(
        f"Priority score std: "
        f"{best['std_priority_score']:.4f}"
    )

    print(
        f"Experimental balance score: "
        f"{best['balance_score']:.4f}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The balance score is an experimental "
        "selection criterion, not ground truth."
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