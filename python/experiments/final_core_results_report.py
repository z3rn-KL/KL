from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)


MASTER_INPUT_PATH = (
    RESULTS_DIR
    / "core_algorithm_all_results.csv"
)

SUMMARY_INPUT_PATH = (
    RESULTS_DIR
    / "core_algorithm_final_summary.csv"
)


FINAL_SUMMARY_PATH = (
    RESULTS_DIR
    / "thesis_core_algorithm_summary.csv"
)

FINAL_RL_SUMMARY_PATH = (
    RESULTS_DIR
    / "thesis_rl_algorithm_summary.csv"
)

DISTANCE_FIGURE_PATH = (
    FIGURES_DIR
    / "core_mean_distance.png"
)

TRAVEL_TIME_FIGURE_PATH = (
    FIGURES_DIR
    / "core_mean_travel_time.png"
)

TRAINING_RUNTIME_FIGURE_PATH = (
    FIGURES_DIR
    / "rl_training_runtime.png"
)

RL_REWARD_FIGURE_PATH = (
    FIGURES_DIR
    / "rl_mean_reward.png"
)


ALGORITHM_ORDER = [
    "nearest_neighbor",
    "clarke_wright",
    "q_learning",
    "sarsa",
    "dqn",
]


DISPLAY_NAMES = {
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


RL_ALGORITHMS = [
    "q_learning",
    "sarsa",
    "dqn",
]


def load_results():
    if not MASTER_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: "
            f"{MASTER_INPUT_PATH}"
        )

    if not SUMMARY_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: "
            f"{SUMMARY_INPUT_PATH}"
        )

    master_df = pd.read_csv(
        MASTER_INPUT_PATH
    )

    summary_df = pd.read_csv(
        SUMMARY_INPUT_PATH
    )

    return (
        master_df,
        summary_df,
    )


def prepare_summary(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = (
        summary_df.copy()
    )

    dataframe[
        "display_name"
    ] = dataframe[
        "algorithm"
    ].map(
        DISPLAY_NAMES
    )

    order_lookup = {
        algorithm: index
        for index, algorithm
        in enumerate(
            ALGORITHM_ORDER
        )
    }

    dataframe[
        "_order"
    ] = dataframe[
        "algorithm"
    ].map(
        order_lookup
    )

    dataframe = (
        dataframe
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    nn_row = (
        dataframe[
            dataframe[
                "algorithm"
            ]
            == "nearest_neighbor"
        ]
        .iloc[
            0
        ]
    )

    nn_distance = float(
        nn_row[
            "mean_distance_km"
        ]
    )

    nn_time = float(
        nn_row[
            "mean_travel_time_minutes"
        ]
    )

    dataframe[
        "distance_change_vs_nn_pct"
    ] = (
        (
            dataframe[
                "mean_distance_km"
            ]
            - nn_distance
        )
        / nn_distance
        * 100.0
    )

    dataframe[
        "travel_time_change_vs_nn_pct"
    ] = (
        (
            dataframe[
                "mean_travel_time_minutes"
            ]
            - nn_time
        )
        / nn_time
        * 100.0
    )

    dataframe[
        "distance_reduction_vs_nn_pct"
    ] = (
        -dataframe[
            "distance_change_vs_nn_pct"
        ]
    )

    dataframe[
        "travel_time_reduction_vs_nn_pct"
    ] = (
        -dataframe[
            "travel_time_change_vs_nn_pct"
        ]
    )

    return dataframe


def build_rl_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rl_df = (
        dataframe[
            dataframe[
                "algorithm"
            ]
            .isin(
                RL_ALGORITHMS
            )
        ]
        .copy()
    )

    q_row = (
        rl_df[
            rl_df[
                "algorithm"
            ]
            == "q_learning"
        ]
        .iloc[
            0
        ]
    )

    q_training = float(
        q_row[
            "mean_training_runtime_seconds"
        ]
    )

    rl_df[
        "training_runtime_vs_q_learning"
    ] = (
        rl_df[
            "mean_training_runtime_seconds"
        ]
        / q_training
    )

    return rl_df


def add_value_labels(
    axis,
    decimals: int = 2,
) -> None:
    for container in (
        axis.containers
    ):
        axis.bar_label(
            container,
            fmt=f"%.{decimals}f",
            padding=3,
        )


def plot_mean_distance(
    dataframe: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    axis.bar(
        dataframe[
            "display_name"
        ],
        dataframe[
            "mean_distance_km"
        ],
    )

    axis.set_title(
        "Mean Route Distance - "
        "31 XeDu Workloads"
    )

    axis.set_xlabel(
        "Algorithm"
    )

    axis.set_ylabel(
        "Mean distance (km)"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    add_value_labels(
        axis,
        decimals=2,
    )

    figure.tight_layout()

    figure.savefig(
        DISTANCE_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_mean_travel_time(
    dataframe: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    axis.bar(
        dataframe[
            "display_name"
        ],
        dataframe[
            "mean_travel_time_minutes"
        ],
    )

    axis.set_title(
        "Mean Travel Time - "
        "31 XeDu Workloads"
    )

    axis.set_xlabel(
        "Algorithm"
    )

    axis.set_ylabel(
        "Mean travel time (minutes)"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    add_value_labels(
        axis,
        decimals=2,
    )

    figure.tight_layout()

    figure.savefig(
        TRAVEL_TIME_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_training_runtime(
    rl_df: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    axis.bar(
        rl_df[
            "display_name"
        ],
        rl_df[
            "mean_training_runtime_seconds"
        ],
    )

    axis.set_title(
        "Mean RL Training Runtime"
    )

    axis.set_xlabel(
        "RL algorithm"
    )

    axis.set_ylabel(
        "Mean training time (seconds)"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    add_value_labels(
        axis,
        decimals=2,
    )

    figure.tight_layout()

    figure.savefig(
        TRAINING_RUNTIME_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_rl_reward(
    rl_df: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    axis.bar(
        rl_df[
            "display_name"
        ],
        rl_df[
            "mean_greedy_reward"
        ],
    )

    axis.set_title(
        "Mean Greedy Reward - "
        "RL Algorithms"
    )

    axis.set_xlabel(
        "RL algorithm"
    )

    axis.set_ylabel(
        "Mean greedy reward"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    add_value_labels(
        axis,
        decimals=4,
    )

    figure.tight_layout()

    figure.savefig(
        RL_REWARD_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def print_core_summary(
    dataframe: pd.DataFrame,
) -> None:
    columns = [
        "display_name",
        "workloads",
        "deliveries",
        "mean_distance_km",
        "mean_travel_time_minutes",
        "mean_on_time_rate",
        "total_late_deliveries",
        "distance_reduction_vs_nn_pct",
        "travel_time_reduction_vs_nn_pct",
    ]

    print()
    print(
        "========================================"
    )

    print(
        "THESIS CORE ALGORITHM SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        dataframe[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


def print_rl_summary(
    dataframe: pd.DataFrame,
) -> None:
    columns = [
        "display_name",
        "mean_distance_km",
        "mean_travel_time_minutes",
        "mean_greedy_reward",
        "mean_training_runtime_seconds",
        "training_runtime_vs_q_learning",
        "mean_model_size",
    ]

    print()
    print(
        "========================================"
    )

    print(
        "RL ALGORITHM SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        dataframe[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


def main():
    print()
    print(
        "========================================"
    )

    print(
        "Final Thesis Results Report"
    )

    print(
        "========================================"
    )

    (
        master_df,
        summary_df,
    ) = load_results()

    print(
        f"Master benchmark rows: "
        f"{len(master_df)}"
    )

    print(
        f"Algorithms: "
        f"{summary_df['algorithm'].nunique()}"
    )

    final_df = (
        prepare_summary(
            summary_df
        )
    )

    rl_df = (
        build_rl_summary(
            final_df
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        FINAL_SUMMARY_PATH,
        index=False,
    )

    rl_df.to_csv(
        FINAL_RL_SUMMARY_PATH,
        index=False,
    )

    plot_mean_distance(
        final_df
    )

    plot_mean_travel_time(
        final_df
    )

    plot_training_runtime(
        rl_df
    )

    plot_rl_reward(
        rl_df
    )

    print_core_summary(
        final_df
    )

    print_rl_summary(
        rl_df
    )

    print()
    print(
        "========================================"
    )

    print(
        "FILES CREATED"
    )

    print(
        "========================================"
    )

    print(
        FINAL_SUMMARY_PATH
    )

    print(
        FINAL_RL_SUMMARY_PATH
    )

    print(
        DISTANCE_FIGURE_PATH
    )

    print(
        TRAVEL_TIME_FIGURE_PATH
    )

    print(
        TRAINING_RUNTIME_FIGURE_PATH
    )

    print(
        RL_REWARD_FIGURE_PATH
    )


if __name__ == "__main__":
    main()