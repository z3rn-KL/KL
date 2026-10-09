from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
    / "final_dataset_benchmark"
)

MASTER_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_all_results.csv"
)


FINAL_OVERALL_PATH = (
    RESULTS_DIR
    / "thesis_dataset_algorithm_summary.csv"
)

FINAL_BY_SIZE_PATH = (
    RESULTS_DIR
    / "thesis_dataset_algorithm_by_size.csv"
)

FINAL_RL_PATH = (
    RESULTS_DIR
    / "thesis_dataset_rl_summary.csv"
)

FINAL_FINDINGS_PATH = (
    RESULTS_DIR
    / "thesis_dataset_final_findings.txt"
)


ALGORITHM_ORDER = [
    "nearest_neighbor",
    "clarke_wright",
    "q_learning",
    "sarsa",
    "dqn",
]

RL_ALGORITHMS = [
    "q_learning",
    "sarsa",
    "dqn",
]


def load_results() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {MASTER_PATH}"
        )

    dataframe = pd.read_csv(
        MASTER_PATH
    )

    required = [
        "workload_id",
        "algorithm",
        "number_of_deliveries",
        "on_time_deliveries",
        "late_deliveries",
        "total_lateness_minutes",
        "max_lateness_minutes",
        "total_distance_km",
        "total_travel_time_minutes",
        "route_runtime_seconds",
        "matrix_preparation_seconds",
        "training_runtime_seconds",
        "greedy_reward",
        "model_size",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Thiếu cột: "
            + ", ".join(missing)
        )

    if len(dataframe) != 540:
        raise ValueError(
            f"Expected 540 runs, found "
            f"{len(dataframe)}."
        )

    if (
        dataframe["workload_id"].nunique()
        != 108
    ):
        raise ValueError(
            "Expected 108 workloads."
        )

    dataframe[
        "number_of_deliveries"
    ] = pd.to_numeric(
        dataframe[
            "number_of_deliveries"
        ],
        errors="raise",
    ).astype(int)

    dataframe[
        "workload_size"
    ] = dataframe[
        "number_of_deliveries"
    ]

    # Reward giữa workload khác kích thước
    # không nên so trực tiếp.
    dataframe[
        "reward_per_delivery"
    ] = (
        dataframe[
            "greedy_reward"
        ]
        / dataframe[
            "number_of_deliveries"
        ]
    )

    return dataframe


def weighted_on_time_rate(
    group: pd.DataFrame,
) -> float:
    deliveries = (
        group[
            "number_of_deliveries"
        ].sum()
    )

    if deliveries <= 0:
        return np.nan

    return (
        group[
            "on_time_deliveries"
        ].sum()
        / deliveries
    )


def build_overall_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for algorithm in ALGORITHM_ORDER:
        group = dataframe[
            dataframe[
                "algorithm"
            ]
            == algorithm
        ]

        if group.empty:
            continue

        total_deliveries = int(
            group[
                "number_of_deliveries"
            ].sum()
        )

        on_time_deliveries = int(
            group[
                "on_time_deliveries"
            ].sum()
        )

        late_deliveries = int(
            group[
                "late_deliveries"
            ].sum()
        )

        rows.append(
            {
                "algorithm":
                    algorithm,

                "workloads":
                    group[
                        "workload_id"
                    ].nunique(),

                "deliveries":
                    total_deliveries,

                "mean_distance_km":
                    group[
                        "total_distance_km"
                    ].mean(),

                "std_distance_km":
                    group[
                        "total_distance_km"
                    ].std(),

                "median_distance_km":
                    group[
                        "total_distance_km"
                    ].median(),

                "mean_travel_time_minutes":
                    group[
                        "total_travel_time_minutes"
                    ].mean(),

                "std_travel_time_minutes":
                    group[
                        "total_travel_time_minutes"
                    ].std(),

                "on_time_deliveries":
                    on_time_deliveries,

                "late_deliveries":
                    late_deliveries,

                "weighted_on_time_rate":
                    weighted_on_time_rate(
                        group
                    ),

                "total_lateness_minutes":
                    group[
                        "total_lateness_minutes"
                    ].sum(),

                "max_lateness_minutes":
                    group[
                        "max_lateness_minutes"
                    ].max(),

                "mean_route_runtime_seconds":
                    group[
                        "route_runtime_seconds"
                    ].mean(),

                "mean_matrix_preparation_seconds":
                    group[
                        "matrix_preparation_seconds"
                    ].mean(),

                "mean_training_runtime_seconds":
                    group[
                        "training_runtime_seconds"
                    ].mean(),

                "total_training_runtime_seconds":
                    group[
                        "training_runtime_seconds"
                    ].sum(),

                "mean_greedy_reward":
                    group[
                        "greedy_reward"
                    ].mean(),

                "mean_reward_per_delivery":
                    group[
                        "reward_per_delivery"
                    ].mean(),

                "mean_model_size":
                    group[
                        "model_size"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_by_size_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        workload_size,
        size_group,
    ) in dataframe.groupby(
        "workload_size"
    ):
        for algorithm in ALGORITHM_ORDER:
            group = size_group[
                size_group[
                    "algorithm"
                ]
                == algorithm
            ]

            if group.empty:
                continue

            rows.append(
                {
                    "workload_size":
                        int(
                            workload_size
                        ),

                    "algorithm":
                        algorithm,

                    "workloads":
                        group[
                            "workload_id"
                        ].nunique(),

                    "deliveries":
                        int(
                            group[
                                "number_of_deliveries"
                            ].sum()
                        ),

                    "mean_distance_km":
                        group[
                            "total_distance_km"
                        ].mean(),

                    "mean_travel_time_minutes":
                        group[
                            "total_travel_time_minutes"
                        ].mean(),

                    "on_time_deliveries":
                        int(
                            group[
                                "on_time_deliveries"
                            ].sum()
                        ),

                    "late_deliveries":
                        int(
                            group[
                                "late_deliveries"
                            ].sum()
                        ),

                    "weighted_on_time_rate":
                        weighted_on_time_rate(
                            group
                        ),

                    "total_lateness_minutes":
                        group[
                            "total_lateness_minutes"
                        ].sum(),

                    "mean_training_runtime_seconds":
                        group[
                            "training_runtime_seconds"
                        ].mean(),

                    "mean_reward_per_delivery":
                        group[
                            "reward_per_delivery"
                        ].mean(),

                    "mean_model_size":
                        group[
                            "model_size"
                        ].mean(),
                }
            )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "workload_size",
                "algorithm",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_rl_summary(
    overall: pd.DataFrame,
) -> pd.DataFrame:
    rl = overall[
        overall[
            "algorithm"
        ].isin(
            RL_ALGORITHMS
        )
    ].copy()

    q_row = rl[
        rl[
            "algorithm"
        ]
        == "q_learning"
    ]

    if not q_row.empty:
        q_training = float(
            q_row.iloc[0][
                "mean_training_runtime_seconds"
            ]
        )

        rl[
            "training_time_relative_to_q_learning"
        ] = (
            rl[
                "mean_training_runtime_seconds"
            ]
            / q_training
        )

    return rl


def pct_reduction(
    baseline: float,
    candidate: float,
) -> float:
    return (
        (
            baseline
            - candidate
        )
        / baseline
        * 100.0
    )


def build_findings(
    overall: pd.DataFrame,
) -> str:
    lookup = (
        overall
        .set_index(
            "algorithm"
        )
    )

    nn = lookup.loc[
        "nearest_neighbor"
    ]

    cw = lookup.loc[
        "clarke_wright"
    ]

    ql = lookup.loc[
        "q_learning"
    ]

    sarsa = lookup.loc[
        "sarsa"
    ]

    dqn = lookup.loc[
        "dqn"
    ]

    dqn_vs_nn_distance = pct_reduction(
        nn[
            "mean_distance_km"
        ],
        dqn[
            "mean_distance_km"
        ],
    )

    dqn_vs_nn_time = pct_reduction(
        nn[
            "mean_travel_time_minutes"
        ],
        dqn[
            "mean_travel_time_minutes"
        ],
    )

    cw_vs_nn_distance = pct_reduction(
        nn[
            "mean_distance_km"
        ],
        cw[
            "mean_distance_km"
        ],
    )

    cw_vs_nn_time = pct_reduction(
        nn[
            "mean_travel_time_minutes"
        ],
        cw[
            "mean_travel_time_minutes"
        ],
    )

    dqn_vs_ql_training = (
        dqn[
            "mean_training_runtime_seconds"
        ]
        / ql[
            "mean_training_runtime_seconds"
        ]
    )

    lines = [
        "FINAL DATASET-SCALE BENCHMARK FINDINGS",
        "=" * 64,
        "",
        "Benchmark:",
        "- 108 workloads",
        "- 813 deliveries per algorithm",
        "- workload size 5–16",
        "- 540 algorithm-workload runs",
        "",
        "Overall route quality:",
        (
            f"- Nearest Neighbor: "
            f"{nn['mean_distance_km']:.4f} km, "
            f"{nn['mean_travel_time_minutes']:.4f} min."
        ),
        (
            f"- Clarke-Wright: "
            f"{cw['mean_distance_km']:.4f} km, "
            f"{cw['mean_travel_time_minutes']:.4f} min."
        ),
        (
            f"- Q-Learning: "
            f"{ql['mean_distance_km']:.4f} km, "
            f"{ql['mean_travel_time_minutes']:.4f} min."
        ),
        (
            f"- SARSA: "
            f"{sarsa['mean_distance_km']:.4f} km, "
            f"{sarsa['mean_travel_time_minutes']:.4f} min."
        ),
        (
            f"- DQN: "
            f"{dqn['mean_distance_km']:.4f} km, "
            f"{dqn['mean_travel_time_minutes']:.4f} min."
        ),
        "",
        "Relative route improvements:",
        (
            f"- Clarke-Wright vs NN: "
            f"{cw_vs_nn_distance:.2f}% distance reduction, "
            f"{cw_vs_nn_time:.2f}% travel-time reduction."
        ),
        (
            f"- DQN vs NN: "
            f"{dqn_vs_nn_distance:.2f}% distance reduction, "
            f"{dqn_vs_nn_time:.2f}% travel-time reduction."
        ),
        "",
        "Delivery-weighted SLA:",
        (
            f"- NN: "
            f"{nn['weighted_on_time_rate'] * 100:.2f}% "
            f"({int(nn['late_deliveries'])} late)."
        ),
        (
            f"- Clarke-Wright: "
            f"{cw['weighted_on_time_rate'] * 100:.2f}% "
            f"({int(cw['late_deliveries'])} late)."
        ),
        (
            f"- Q-Learning: "
            f"{ql['weighted_on_time_rate'] * 100:.2f}% "
            f"({int(ql['late_deliveries'])} late)."
        ),
        (
            f"- SARSA: "
            f"{sarsa['weighted_on_time_rate'] * 100:.2f}% "
            f"({int(sarsa['late_deliveries'])} late)."
        ),
        (
            f"- DQN: "
            f"{dqn['weighted_on_time_rate'] * 100:.2f}% "
            f"({int(dqn['late_deliveries'])} late)."
        ),
        "",
        "RL computational trade-off:",
        (
            f"- Q-Learning mean training: "
            f"{ql['mean_training_runtime_seconds']:.2f} s."
        ),
        (
            f"- SARSA mean training: "
            f"{sarsa['mean_training_runtime_seconds']:.2f} s."
        ),
        (
            f"- DQN mean training: "
            f"{dqn['mean_training_runtime_seconds']:.2f} s."
        ),
        (
            f"- DQN requires approximately "
            f"{dqn_vs_ql_training:.1f}x the mean training time "
            f"of Q-Learning."
        ),
        "",
        "Interpretation:",
        (
            "- Clarke-Wright remains the strongest method for "
            "pure distance/time minimization."
        ),
        (
            "- DQN is the strongest RL method in route quality "
            "and reaches 100% delivery-weighted on-time rate."
        ),
        (
            "- Q-Learning also reaches 100% on-time rate with "
            "much lower training cost."
        ),
        (
            "- SARSA reduces late deliveries relative to the "
            "heuristic baselines but is weaker than DQN overall."
        ),
        (
            "- Reward per delivery should be used instead of raw "
            "episode reward when comparing different workload sizes."
        ),
        (
            "- Results for N=14, 15 and 16 are case evidence only "
            "because each size contains one workload."
        ),
    ]

    return "\n".join(
        lines
    )


def plot_overall_distance(
    overall: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        overall[
            "algorithm"
        ],
        overall[
            "mean_distance_km"
        ],
    )

    plt.ylabel(
        "Mean distance (km)"
    )

    plt.title(
        "Mean Route Distance – 108 Workloads"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_mean_distance.png",
        dpi=180,
    )

    plt.close()


def plot_overall_time(
    overall: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        overall[
            "algorithm"
        ],
        overall[
            "mean_travel_time_minutes"
        ],
    )

    plt.ylabel(
        "Mean travel time (minutes)"
    )

    plt.title(
        "Mean Travel Time – 108 Workloads"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_mean_travel_time.png",
        dpi=180,
    )

    plt.close()


def plot_weighted_sla(
    overall: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        overall[
            "algorithm"
        ],
        overall[
            "weighted_on_time_rate"
        ]
        * 100.0,
    )

    plt.ylabel(
        "On-time deliveries (%)"
    )

    plt.title(
        "Delivery-Weighted On-Time Rate"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_weighted_on_time_rate.png",
        dpi=180,
    )

    plt.close()


def plot_rl_training(
    overall: pd.DataFrame,
) -> None:
    rl = overall[
        overall[
            "algorithm"
        ].isin(
            RL_ALGORITHMS
        )
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        rl[
            "algorithm"
        ],
        rl[
            "mean_training_runtime_seconds"
        ],
    )

    plt.ylabel(
        "Mean training runtime (seconds)"
    )

    plt.title(
        "RL Training Runtime"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_rl_training_runtime.png",
        dpi=180,
    )

    plt.close()


def plot_reward_per_delivery(
    by_size: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in RL_ALGORITHMS:
        subset = by_size[
            by_size[
                "algorithm"
            ]
            == algorithm
        ]

        plt.plot(
            subset[
                "workload_size"
            ],
            subset[
                "mean_reward_per_delivery"
            ],
            marker="o",
            label=algorithm,
        )

    plt.xlabel(
        "Workload size"
    )

    plt.ylabel(
        "Mean greedy reward per delivery"
    )

    plt.title(
        "Normalized RL Reward by Workload Size"
    )

    plt.xticks(
        sorted(
            by_size[
                "workload_size"
            ].unique()
        )
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_rl_reward_per_delivery.png",
        dpi=180,
    )

    plt.close()


def plot_weighted_sla_by_size(
    by_size: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in ALGORITHM_ORDER:
        subset = by_size[
            by_size[
                "algorithm"
            ]
            == algorithm
        ]

        plt.plot(
            subset[
                "workload_size"
            ],
            subset[
                "weighted_on_time_rate"
            ]
            * 100.0,
            marker="o",
            label=algorithm,
        )

    plt.xlabel(
        "Workload size"
    )

    plt.ylabel(
        "Delivery-weighted on-time rate (%)"
    )

    plt.title(
        "On-Time Performance by Workload Size"
    )

    plt.xticks(
        sorted(
            by_size[
                "workload_size"
            ].unique()
        )
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_weighted_on_time_by_size.png",
        dpi=180,
    )

    plt.close()


def plot_model_size(
    by_size: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in RL_ALGORITHMS:
        subset = by_size[
            by_size[
                "algorithm"
            ]
            == algorithm
        ].dropna(
            subset=[
                "mean_model_size"
            ]
        )

        plt.plot(
            subset[
                "workload_size"
            ],
            subset[
                "mean_model_size"
            ],
            marker="o",
            label=algorithm,
        )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Workload size"
    )

    plt.ylabel(
        "Mean model / Q-table size (log scale)"
    )

    plt.title(
        "RL Representation Size by Workload Size"
    )

    plt.xticks(
        sorted(
            by_size[
                "workload_size"
            ].unique()
        )
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "final_rl_model_size.png",
        dpi=180,
    )

    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_results()

    overall = build_overall_summary(
        dataframe
    )

    by_size = build_by_size_summary(
        dataframe
    )

    rl_summary = build_rl_summary(
        overall
    )

    overall.to_csv(
        FINAL_OVERALL_PATH,
        index=False,
    )

    by_size.to_csv(
        FINAL_BY_SIZE_PATH,
        index=False,
    )

    rl_summary.to_csv(
        FINAL_RL_PATH,
        index=False,
    )

    findings = build_findings(
        overall
    )

    FINAL_FINDINGS_PATH.write_text(
        findings,
        encoding="utf-8",
    )

    plot_overall_distance(
        overall
    )

    plot_overall_time(
        overall
    )

    plot_weighted_sla(
        overall
    )

    plot_rl_training(
        overall
    )

    plot_reward_per_delivery(
        by_size
    )

    plot_weighted_sla_by_size(
        by_size
    )

    plot_model_size(
        by_size
    )

    print()
    print(
        "=" * 72
    )
    print(
        "FINAL DATASET BENCHMARK SUMMARY"
    )
    print(
        "=" * 72
    )

    display_columns = [
        "algorithm",
        "workloads",
        "deliveries",
        "mean_distance_km",
        "mean_travel_time_minutes",
        "weighted_on_time_rate",
        "late_deliveries",
        "mean_training_runtime_seconds",
        "mean_reward_per_delivery",
        "mean_model_size",
    ]

    print(
        overall[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        findings
    )

    print()
    print(
        "Created:"
    )

    for path in [
        FINAL_OVERALL_PATH,
        FINAL_BY_SIZE_PATH,
        FINAL_RL_PATH,
        FINAL_FINDINGS_PATH,
        FIGURES_DIR,
    ]:
        print(
            "-",
            path,
        )


if __name__ == "__main__":
    main()