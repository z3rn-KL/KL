"""
Dataset-scale benchmark for the thesis.

This experiment reuses the complete benchmark pipeline implemented in
core_algorithm_full_benchmark.py, but expands the workload selection from
exactly 5 deliveries to every 120-minute wave containing at least 5
deliveries.

Expected XeDu benchmark set:
    108 workloads
    813 deliveries
    workload size: 5 -> 16

Algorithms:
    - Nearest Neighbor
    - Clarke-Wright
    - Q-Learning
    - SARSA
    - DQN

Important:
    The original 31 x size-5 benchmark is preserved unchanged.
"""

from pathlib import Path

import pandas as pd

import experiments.core_algorithm_full_benchmark as base


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MIN_WORKLOAD_SIZE = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MASTER_OUTPUT_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_all_results.csv"
)

SUMMARY_OUTPUT_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_final_summary.csv"
)

WORKLOAD_OUTPUT_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_workloads.csv"
)


def select_dataset_workloads(
    dataframe: pd.DataFrame,
) -> list[pd.DataFrame]:
    """
    Select every chronological wave containing at least
    MIN_WORKLOAD_SIZE deliveries.

    Unlike the controlled benchmark, workload size is NOT fixed.

    Expected workload sizes in the current XeDu dataset:
        5, 6, ..., 16

    Waves are sorted chronologically so workload_id remains stable
    across interrupted/resumed benchmark runs.
    """

    workloads: list[pd.DataFrame] = []

    for (
        _,
        group,
    ) in dataframe.groupby(
        "wave_start"
    ):
        if len(group) < MIN_WORKLOAD_SIZE:
            continue

        group = (
            group
            .sort_values(
                "delivery_id"
            )
            .copy()
        )

        workloads.append(
            group
        )

    workloads.sort(
        key=lambda group: (
            group[
                "wave_start"
            ].iloc[0]
        )
    )

    return workloads


def configure_base_benchmark() -> None:
    """
    Redirect the existing full benchmark implementation to the
    dataset-scale experiment.

    This preserves:
        - routing implementations;
        - RL configurations;
        - 5000 training episodes;
        - seed 42;
        - autosave;
        - resume;
        - evaluation logic;
        - final summary logic.

    Only workload selection and output files are changed.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base.MASTER_OUTPUT_PATH = (
        MASTER_OUTPUT_PATH
    )

    base.SUMMARY_OUTPUT_PATH = (
        SUMMARY_OUTPUT_PATH
    )

    base.WORKLOAD_OUTPUT_PATH = (
        WORKLOAD_OUTPUT_PATH
    )

    base.select_all_workloads = (
        select_dataset_workloads
    )


def print_experiment_plan() -> None:
    print()
    print(
        "=" * 64
    )
    print(
        "XeDu Dataset-Scale Core Algorithm Benchmark"
    )
    print(
        "=" * 64
    )

    print(
        f"Minimum workload size : "
        f"{MIN_WORKLOAD_SIZE}"
    )

    print(
        f"Training episodes     : "
        f"{base.TRAINING_EPISODES}"
    )

    print(
        f"Random seed           : "
        f"{base.RANDOM_SEED}"
    )

    print(
        "Algorithms            : "
        + ", ".join(
            base.ALGORITHMS
        )
    )

    print(
        "Resume mode           : ENABLED"
    )

    print()
    print(
        "Output files:"
    )

    print(
        f"  Master   : "
        f"{MASTER_OUTPUT_PATH}"
    )

    print(
        f"  Summary  : "
        f"{SUMMARY_OUTPUT_PATH}"
    )

    print(
        f"  Workloads: "
        f"{WORKLOAD_OUTPUT_PATH}"
    )

    print(
        "=" * 64
    )
    print()


def main() -> None:
    configure_base_benchmark()

    print_experiment_plan()

    base.main()


if __name__ == "__main__":
    main()