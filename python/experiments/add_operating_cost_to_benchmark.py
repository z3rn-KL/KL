"""
Add estimated operating cost to the canonical dataset-scale benchmark.

This script DOES NOT rerun routing algorithms.

It post-processes the existing canonical benchmark:

    results/dataset_algorithm_all_results.csv

using the existing:

    evaluation.operating_cost.OperatingCostEvaluator

and the vehicle cost coefficients defined by the thesis scenario.

Important
---------
The resulting cost is an estimated operational-cost proxy.

It MUST NOT be presented as verified XeDu accounting cost.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.operating_cost import (
    OperatingCostConfig,
    OperatingCostEvaluator,
)

from simulation.scenario import (
    ScenarioFactory,
)


# ============================================================
# PATHS
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

INPUT_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_all_results.csv"
)

OUTPUT_DETAIL_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_cost_results.csv"
)

OUTPUT_SUMMARY_PATH = (
    RESULTS_DIR
    / "thesis_dataset_algorithm_cost_summary.csv"
)

OUTPUT_FINDINGS_PATH = (
    RESULTS_DIR
    / "thesis_dataset_cost_findings.txt"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "chapter4"
    / "figures"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "figure_estimated_operating_cost.png"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "workload_id",
    "algorithm",
    "number_of_deliveries",
    "on_time_deliveries",
    "late_deliveries",
    "total_lateness_minutes",
    "total_distance_km",
    "total_travel_time_minutes",
}


ALGORITHM_ORDER = [
    "nearest_neighbor",
    "clarke_wright",
    "q_learning",
    "sarsa",
    "dqn",
]


ALGORITHM_LABELS = {
    "nearest_neighbor":
        "Nearest Neighbor",

    "clarke_wright":
        "Clarke-Wright",

    "q_learning":
        "Q-Learning",

    "sarsa":
        "SARSA",

    "dqn":
        "DQN",
}


# ============================================================
# CONFIGURATION
# ============================================================

def build_cost_config(
) -> OperatingCostConfig:
    """
    Reuse the cost coefficients from the thesis scenario.

    These values are treated as scenario assumptions,
    not verified XeDu accounting data.
    """

    scenario = (
        ScenarioFactory
        .create_single_depot_scenario()
    )

    if not scenario.vehicles:
        raise ValueError(
            "Scenario contains no vehicles."
        )

    vehicle = (
        scenario.vehicles[
            0
        ]
    )

    distance_cost_per_km = float(
        vehicle.cost_per_km
    )

    travel_time_cost_per_hour = float(
        vehicle.cost_per_hour
    )

    travel_time_cost_per_minute = (
        travel_time_cost_per_hour
        / 60.0
    )

    return OperatingCostConfig(
        distance_cost_per_km=(
            distance_cost_per_km
        ),
        travel_time_cost_per_minute=(
            travel_time_cost_per_minute
        ),

        # No authoritative late-delivery financial
        # penalty is available from XeDu.
        late_delivery_penalty=0.0,
        lateness_cost_per_minute=0.0,

        cost_unit=(
            "VND_proxy_scenario"
        ),
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def load_benchmark(
) -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: "
            f"{INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "Benchmark file is empty."
        )

    missing = (
        REQUIRED_COLUMNS
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "Benchmark is missing required "
            f"columns: {sorted(missing)}"
        )

    numeric_columns = [
        "workload_id",
        "number_of_deliveries",
        "on_time_deliveries",
        "late_deliveries",
        "total_lateness_minutes",
        "total_distance_km",
        "total_travel_time_minutes",
    ]

    for column in numeric_columns:
        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="raise",
        )

    return dataframe


# ============================================================
# COST CALCULATION
# ============================================================

def add_operating_cost(
    dataframe: pd.DataFrame,
    config: OperatingCostConfig,
) -> pd.DataFrame:
    evaluator = (
        OperatingCostEvaluator(
            config
        )
    )

    result = (
        dataframe
        .copy()
    )

    distance_costs = []
    travel_time_costs = []
    late_delivery_costs = []
    lateness_costs = []
    total_costs = []

    for row in (
        result.itertuples(
            index=False
        )
    ):
        cost = (
            evaluator.evaluate(
                total_distance_km=float(
                    row.total_distance_km
                ),
                total_travel_time_minutes=float(
                    row
                    .total_travel_time_minutes
                ),
                late_deliveries=int(
                    row.late_deliveries
                ),
                total_lateness_minutes=float(
                    row
                    .total_lateness_minutes
                ),
            )
        )

        distance_costs.append(
            cost.distance_cost
        )

        travel_time_costs.append(
            cost.travel_time_cost
        )

        late_delivery_costs.append(
            cost.late_delivery_cost
        )

        lateness_costs.append(
            cost.lateness_cost
        )

        total_costs.append(
            cost.total_estimated_cost
        )

    result[
        "estimated_distance_cost"
    ] = distance_costs

    result[
        "estimated_travel_time_cost"
    ] = travel_time_costs

    result[
        "estimated_late_delivery_cost"
    ] = late_delivery_costs

    result[
        "estimated_lateness_cost"
    ] = lateness_costs

    result[
        "estimated_operating_cost"
    ] = total_costs

    result[
        "distance_cost_per_km"
    ] = (
        config.distance_cost_per_km
    )

    result[
        "travel_time_cost_per_minute"
    ] = (
        config
        .travel_time_cost_per_minute
    )

    result[
        "late_delivery_penalty"
    ] = (
        config.late_delivery_penalty
    )

    result[
        "lateness_cost_per_minute"
    ] = (
        config
        .lateness_cost_per_minute
    )

    result[
        "cost_unit"
    ] = (
        config.cost_unit
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for algorithm in (
        ALGORITHM_ORDER
    ):
        subset = dataframe[
            dataframe[
                "algorithm"
            ]
            == algorithm
        ]

        if subset.empty:
            continue

        runs = int(
            len(
                subset
            )
        )

        workloads = int(
            subset[
                "workload_id"
            ]
            .nunique()
        )

        deliveries = int(
            subset[
                "number_of_deliveries"
            ]
            .sum()
        )

        on_time = int(
            subset[
                "on_time_deliveries"
            ]
            .sum()
        )

        late = int(
            subset[
                "late_deliveries"
            ]
            .sum()
        )

        valid_deadline_deliveries = (
            on_time
            + late
        )

        weighted_on_time_rate = (
            on_time
            / valid_deadline_deliveries

            if valid_deadline_deliveries
            > 0

            else float(
                "nan"
            )
        )

        total_cost = float(
            subset[
                "estimated_operating_cost"
            ]
            .sum()
        )

        mean_cost = float(
            subset[
                "estimated_operating_cost"
            ]
            .mean()
        )

        cost_per_delivery = (
            total_cost
            / deliveries
            if deliveries > 0
            else float(
                "nan"
            )
        )

        rows.append(
            {
                "algorithm":
                    algorithm,

                "algorithm_label":
                    ALGORITHM_LABELS[
                        algorithm
                    ],

                "runs":
                    runs,

                "workloads":
                    workloads,

                "deliveries":
                    deliveries,

                "valid_deadline_deliveries":
                    valid_deadline_deliveries,

                "late_deliveries":
                    late,

                "weighted_on_time_rate":
                    weighted_on_time_rate,

                "mean_distance_km":
                    float(
                        subset[
                            "total_distance_km"
                        ]
                        .mean()
                    ),

                "mean_travel_time_minutes":
                    float(
                        subset[
                            "total_travel_time_minutes"
                        ]
                        .mean()
                    ),

                "mean_estimated_cost_per_workload":
                    mean_cost,

                "total_estimated_operating_cost":
                    total_cost,

                "estimated_cost_per_delivery":
                    cost_per_delivery,

                "mean_estimated_distance_cost":
                    float(
                        subset[
                            "estimated_distance_cost"
                        ]
                        .mean()
                    ),

                "mean_estimated_travel_time_cost":
                    float(
                        subset[
                            "estimated_travel_time_cost"
                        ]
                        .mean()
                    ),

                "cost_unit":
                    str(
                        subset[
                            "cost_unit"
                        ]
                        .iloc[
                            0
                        ]
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    if summary.empty:
        return summary

    nn = summary[
        summary[
            "algorithm"
        ]
        == "nearest_neighbor"
    ]

    if len(
        nn
    ) != 1:
        raise ValueError(
            "Nearest Neighbor baseline "
            "was not found exactly once."
        )

    baseline_cost = float(
        nn.iloc[
            0
        ][
            "mean_estimated_cost_per_workload"
        ]
    )

    summary[
        "estimated_cost_improvement_vs_nn_pct"
    ] = (
        (
            baseline_cost
            - summary[
                "mean_estimated_cost_per_workload"
            ]
        )
        / baseline_cost
        * 100.0
    )

    return summary


# ============================================================
# FIGURE
# ============================================================

def create_figure(
    summary: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = (
        summary[
            "algorithm_label"
        ]
        .tolist()
    )

    values = (
        summary[
            "mean_estimated_cost_per_workload"
        ]
        .tolist()
    )

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        labels,
        values,
    )

    plt.xlabel(
        "Algorithm"
    )

    plt.ylabel(
        "Mean estimated operating cost "
        "(VND proxy / workload)"
    )

    plt.title(
        "Estimated Operating Cost "
        "by Routing Algorithm"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FINDINGS
# ============================================================

def write_findings(
    summary: pd.DataFrame,
    config: OperatingCostConfig,
) -> None:
    ranked = (
        summary
        .sort_values(
            "mean_estimated_cost_per_workload"
        )
        .reset_index(
            drop=True
        )
    )

    best = ranked.iloc[
        0
    ]

    lines = [
        "ESTIMATED OPERATING COST BENCHMARK",
        "=" * 72,
        "",
        "Important:",
        (
            "- These values are scenario-based "
            "operating-cost proxies."
        ),
        (
            "- They are NOT verified XeDu "
            "accounting costs."
        ),
        "",
        "Cost configuration:",
        (
            "- distance_cost_per_km = "
            f"{config.distance_cost_per_km:.4f}"
        ),
        (
            "- travel_time_cost_per_minute = "
            f"{config.travel_time_cost_per_minute:.4f}"
        ),
        (
            "- late_delivery_penalty = "
            f"{config.late_delivery_penalty:.4f}"
        ),
        (
            "- lateness_cost_per_minute = "
            f"{config.lateness_cost_per_minute:.4f}"
        ),
        (
            "- cost_unit = "
            f"{config.cost_unit}"
        ),
        "",
        "Ranking by mean estimated operating cost:",
    ]

    for index, row in (
        ranked.iterrows()
    ):
        lines.append(
            (
                f"{index + 1}. "
                f"{row['algorithm_label']}: "
                f"{row['mean_estimated_cost_per_workload']:.2f} "
                f"{row['cost_unit']} / workload "
                f"({row['estimated_cost_improvement_vs_nn_pct']:+.2f}% "
                "vs Nearest Neighbor)"
            )
        )

    lines.extend(
        [
            "",
            "Best estimated-cost algorithm:",
            (
                f"- {best['algorithm_label']} "
                f"with mean cost "
                f"{best['mean_estimated_cost_per_workload']:.2f} "
                f"{best['cost_unit']} per workload."
            ),
            "",
            "Interpretation:",
            (
                "- Because the current proxy uses only "
                "distance and travel-time financial terms, "
                "algorithms with shorter routes and lower "
                "travel time naturally obtain lower "
                "estimated operating cost."
            ),
            (
                "- No monetary penalty for late deliveries "
                "is added because the dataset does not provide "
                "an authoritative business penalty value."
            ),
            "",
        ]
    )

    OUTPUT_FINDINGS_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# AUDIT
# ============================================================

def audit_outputs(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    input_rows = len(
        detail
    )

    expected_summary_rows = len(
        ALGORITHM_ORDER
    )

    if input_rows != 540:
        print(
            "WARNING:"
            f" benchmark contains "
            f"{input_rows} rows, "
            "not the historical expected 540."
        )

    if (
        len(
            summary
        )
        != expected_summary_rows
    ):
        raise ValueError(
            "Summary does not contain "
            "all five algorithms."
        )

    if (
        detail[
            "estimated_operating_cost"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Estimated cost contains NaN."
        )

    if (
        detail[
            "estimated_operating_cost"
        ]
        .lt(
            0
        )
        .any()
    ):
        raise ValueError(
            "Estimated cost cannot be negative."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("=" * 72)
    print(
        "ESTIMATED OPERATING COST BENCHMARK"
    )
    print("=" * 72)

    benchmark = (
        load_benchmark()
    )

    print(
        f"Input rows: "
        f"{len(benchmark)}"
    )

    print(
        f"Workloads: "
        f"{benchmark['workload_id'].nunique()}"
    )

    print(
        f"Algorithms: "
        f"{benchmark['algorithm'].nunique()}"
    )

    config = (
        build_cost_config()
    )

    print()
    print(
        "Cost configuration:"
    )

    print(
        "distance_cost_per_km = "
        f"{config.distance_cost_per_km:.4f}"
    )

    print(
        "travel_time_cost_per_minute = "
        f"{config.travel_time_cost_per_minute:.4f}"
    )

    print(
        "late_delivery_penalty = "
        f"{config.late_delivery_penalty:.4f}"
    )

    print(
        "lateness_cost_per_minute = "
        f"{config.lateness_cost_per_minute:.4f}"
    )

    print(
        "cost_unit = "
        f"{config.cost_unit}"
    )

    detail = (
        add_operating_cost(
            dataframe=benchmark,
            config=config,
        )
    )

    summary = (
        build_summary(
            detail
        )
    )

    audit_outputs(
        detail=detail,
        summary=summary,
    )

    detail.to_csv(
        OUTPUT_DETAIL_PATH,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY_PATH,
        index=False,
    )

    write_findings(
        summary=summary,
        config=config,
    )

    create_figure(
        summary
    )

    print()
    print("=" * 72)
    print(
        "COST SUMMARY"
    )
    print("=" * 72)

    display_columns = [
        "algorithm_label",
        "mean_distance_km",
        "mean_travel_time_minutes",
        "weighted_on_time_rate",
        "mean_estimated_cost_per_workload",
        "estimated_cost_per_delivery",
        "estimated_cost_improvement_vs_nn_pct",
    ]

    print(
        summary[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Created:"
    )

    print(
        f"- {OUTPUT_DETAIL_PATH}"
    )

    print(
        f"- {OUTPUT_SUMMARY_PATH}"
    )

    print(
        f"- {OUTPUT_FINDINGS_PATH}"
    )

    print(
        f"- {FIGURE_PATH}"
    )

    print()
    print(
        "Operating-cost benchmark complete."
    )
    print()


if __name__ == "__main__":
    main()