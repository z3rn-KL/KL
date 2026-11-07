from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

import experiments.final_kmeans_guided_dqn_validation as validation

from experiments.proposed_rl_real_workload_smoke import (
    build_deadlines_minutes,
    load_real_workload,
    resolve_start_time,
)

from rl.proposed.agent import (
    ProposedAgentConfig,
    ProposedRoutingAgent,
)
from rl.proposed.candidate_encoder import (
    CandidateEncoder,
)
from rl.proposed.environment import (
    ProposedRoutingEnvironment,
)
from rl.proposed.objective import (
    ObjectiveConfig,
)
from rl.proposed.trainer import (
    evaluate_agent,
    run_training_episode,
)


# ============================================================
# PATHS / EXPERIMENT CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

RUNS_PATH = (
    RESULTS_DIR
    / "proposed_rl_objective_sensitivity_runs.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "proposed_rl_objective_sensitivity_summary.csv"
)

FINDINGS_PATH = (
    RESULTS_DIR
    / "proposed_rl_objective_sensitivity_findings.txt"
)


WORKLOAD_IDS = (
    101,  # N = 5
    106,  # N = 10
    97,   # N = 13
)

TRAINING_EPISODES = 3000
PRINT_INTERVAL = 500

SEED = 42


# ============================================================
# OBJECTIVE CONFIGURATIONS
# ============================================================

OBJECTIVE_CONFIGS = {
    "A_current": ObjectiveConfig(
        distance_weight=0.40,
        travel_time_weight=0.40,
        late_delivery_weight=0.00,
        lateness_weight=0.15,
        cluster_switch_weight=0.05,
    ),

    "B_balanced_sla": ObjectiveConfig(
        distance_weight=0.35,
        travel_time_weight=0.35,
        late_delivery_weight=0.15,
        lateness_weight=0.10,
        cluster_switch_weight=0.05,
    ),

    "C_strong_sla": ObjectiveConfig(
        distance_weight=0.325,
        travel_time_weight=0.325,
        late_delivery_weight=0.20,
        lateness_weight=0.10,
        cluster_switch_weight=0.05,
    ),

    "D_moderate_sla": ObjectiveConfig(
        distance_weight=0.375,
        travel_time_weight=0.375,
        late_delivery_weight=0.10,
        lateness_weight=0.10,
        cluster_switch_weight=0.05,
    ),
}


# ============================================================
# WORKLOAD PREPARATION
# ============================================================

def prepare_workload(
    *,
    workload_id: int,
    matrix_builder,
    depot_node: int,
) -> dict[str, object]:
    """
    Chuẩn bị workload thật một lần.

    Road matrix được build đúng một lần cho mỗi workload
    rồi tái sử dụng cho cả bốn objective configurations.
    """

    workload = load_real_workload(
        workload_id
    )

    start_time = resolve_start_time(
        workload
    )

    deadlines_minutes = (
        build_deadlines_minutes(
            workload=workload,
            start_time=start_time,
        )
    )

    delivery_nodes = [
        int(value)
        for value in workload[
            "road_node"
        ].tolist()
    ]

    cluster_ids = [
        int(value)
        for value in workload[
            "cluster_id"
        ].tolist()
    ]

    node_ids = [
        int(depot_node),
        *delivery_nodes,
    ]

    print()
    print(
        f"Building road matrix for "
        f"workload {workload_id}..."
    )

    matrix_start = perf_counter()

    road_matrix = matrix_builder.build(
        node_ids
    )

    matrix_runtime = (
        perf_counter()
        - matrix_start
    )

    distance_matrix_km = (
        road_matrix.distance_km.tolist()
    )

    travel_time_matrix_minutes = (
        road_matrix
        .travel_time_minutes
        .tolist()
    )

    # Không cần giữ shortest-path cache sau khi
    # matrix đã được copy sang Python lists.
    matrix_builder.clear_cache()

    return {
        "workload_id": workload_id,
        "number_of_deliveries": len(workload),
        "workload": workload,
        "start_time": start_time,
        "deadlines_minutes": deadlines_minutes,
        "cluster_ids": cluster_ids,
        "distance_matrix_km": distance_matrix_km,
        "travel_time_matrix_minutes": (
            travel_time_matrix_minutes
        ),
        "matrix_runtime_seconds": matrix_runtime,
    }


def build_environment(
    *,
    prepared: dict[str, object],
    objective_config: ObjectiveConfig,
) -> ProposedRoutingEnvironment:
    return ProposedRoutingEnvironment(
        distance_matrix_km=(
            prepared["distance_matrix_km"]
        ),
        travel_time_matrix_minutes=(
            prepared[
                "travel_time_matrix_minutes"
            ]
        ),
        deadlines_minutes=(
            prepared["deadlines_minutes"]
        ),
        cluster_ids=(
            prepared["cluster_ids"]
        ),
        objective_config=(
            objective_config
        ),
    )


# ============================================================
# AGENT
# ============================================================

def build_agent(
) -> ProposedRoutingAgent:
    """
    Giữ toàn bộ hyperparameter RL giống nhau giữa
    các cấu hình objective để comparison công bằng.
    """

    return ProposedRoutingAgent(
        config=ProposedAgentConfig(
            learning_rate=1e-3,

            gamma=1.0,

            epsilon_start=1.0,
            epsilon_min=0.02,
            epsilon_decay=0.998,

            replay_capacity=10_000,

            batch_size=64,
            min_replay_size=128,

            target_update_interval=100,

            hidden_dimension=64,

            gradient_clip_norm=5.0,

            seed=SEED,

            device="auto",
        )
    )


# ============================================================
# ONE TRAINING RUN
# ============================================================

def run_one_configuration(
    *,
    prepared: dict[str, object],
    config_name: str,
    objective_config: ObjectiveConfig,
) -> dict[str, object]:
    workload_id = int(
        prepared["workload_id"]
    )

    number_of_deliveries = int(
        prepared["number_of_deliveries"]
    )

    env = build_environment(
        prepared=prepared,
        objective_config=objective_config,
    )

    encoder = CandidateEncoder()

    agent = build_agent()

    print()
    print("=" * 78)

    print(
        f"WORKLOAD {workload_id} | "
        f"N={number_of_deliveries} | "
        f"{config_name}"
    )

    print("=" * 78)

    print(
        "Weights: "
        f"D={objective_config.distance_weight:.3f}, "
        f"T={objective_config.travel_time_weight:.3f}, "
        f"M={objective_config.late_delivery_weight:.3f}, "
        f"L={objective_config.lateness_weight:.3f}, "
        f"S={objective_config.cluster_switch_weight:.3f}"
    )

    training_start = perf_counter()

    for episode in range(
        1,
        TRAINING_EPISODES + 1,
    ):
        run_training_episode(
            env=env,
            encoder=encoder,
            agent=agent,
            episode_index=episode,
        )

        if (
            episode == 1
            or episode % PRINT_INTERVAL == 0
            or episode == TRAINING_EPISODES
        ):
            evaluation = evaluate_agent(
                env=env,
                encoder=encoder,
                agent=agent,
            )

            print(
                f"Episode {episode:4d} | "
                f"eps={agent.epsilon:.4f} | "
                f"J={evaluation.objective_cost:.6f} | "
                f"D={evaluation.total_distance_km:.3f} | "
                f"T={evaluation.total_travel_time_minutes:.3f} | "
                f"M={env.totals.late_deliveries} | "
                f"L={evaluation.total_lateness_minutes:.3f} | "
                f"S={evaluation.cluster_switches}"
            )

    training_runtime = (
        perf_counter()
        - training_start
    )

    # Final greedy inference.
    final = evaluate_agent(
        env=env,
        encoder=encoder,
        agent=agent,
    )

    late_deliveries = (
        env.totals.late_deliveries
    )

    valid_deadlines = (
        env.objective_context
        .number_of_valid_deadlines
    )

    if valid_deadlines == 0:
        on_time_rate = 1.0
    else:
        on_time_rate = (
            (
                valid_deadlines
                - late_deliveries
            )
            / valid_deadlines
        )

    row = {
        "config_name": config_name,

        "workload_id": workload_id,

        "number_of_deliveries": (
            number_of_deliveries
        ),

        "valid_deadlines": (
            valid_deadlines
        ),

        "distance_weight": (
            objective_config.distance_weight
        ),

        "travel_time_weight": (
            objective_config.travel_time_weight
        ),

        "late_delivery_weight": (
            objective_config.late_delivery_weight
        ),

        "lateness_weight": (
            objective_config.lateness_weight
        ),

        "cluster_switch_weight": (
            objective_config.cluster_switch_weight
        ),

        # Lưu J để debug nội bộ.
        # KHÔNG dùng J để so trực tiếp giữa
        # các cấu hình trọng số khác nhau.
        "objective_cost": (
            final.objective_cost
        ),

        "distance_km": (
            final.total_distance_km
        ),

        "travel_time_minutes": (
            final.total_travel_time_minutes
        ),

        "late_deliveries": (
            late_deliveries
        ),

        "lateness_minutes": (
            final.total_lateness_minutes
        ),

        "on_time_rate": (
            on_time_rate
        ),

        "cluster_switches": (
            final.cluster_switches
        ),

        "training_runtime_seconds": (
            training_runtime
        ),

        "route_actions": (
            str(
                tuple(
                    final.route_actions
                )
            )
        ),

        "seed": SEED,

        "training_episodes": (
            TRAINING_EPISODES
        ),
    }

    print()
    print("FINAL")

    print(
        f"Distance: "
        f"{row['distance_km']:.6f} km"
    )

    print(
        f"Travel time: "
        f"{row['travel_time_minutes']:.6f} min"
    )

    print(
        f"Late deliveries: "
        f"{late_deliveries}"
    )

    print(
        f"Lateness: "
        f"{row['lateness_minutes']:.6f} min"
    )

    print(
        f"SLA: "
        f"{on_time_rate * 100:.2f}%"
    )

    print(
        f"Cluster switches: "
        f"{row['cluster_switches']}"
    )

    print(
        f"Training runtime: "
        f"{training_runtime:.2f} s"
    )

    # Giải phóng model/replay trước run kế tiếp.
    del agent
    del encoder
    del env

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# ============================================================
# AGGREGATION
# ============================================================

def build_summary(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for config_name, group in runs.groupby(
        "config_name",
        sort=False,
    ):
        total_valid = int(
            group[
                "valid_deadlines"
            ].sum()
        )

        total_late = int(
            group[
                "late_deliveries"
            ].sum()
        )

        if total_valid == 0:
            weighted_sla = 1.0
        else:
            weighted_sla = (
                (
                    total_valid
                    - total_late
                )
                / total_valid
            )

        rows.append(
            {
                "config_name": config_name,

                "workloads": len(
                    group
                ),

                "deliveries": int(
                    group[
                        "number_of_deliveries"
                    ].sum()
                ),

                "mean_distance_km": float(
                    group[
                        "distance_km"
                    ].mean()
                ),

                "mean_travel_time_minutes": float(
                    group[
                        "travel_time_minutes"
                    ].mean()
                ),

                "total_late_deliveries": (
                    total_late
                ),

                "total_valid_deadlines": (
                    total_valid
                ),

                "weighted_sla": (
                    weighted_sla
                ),

                "total_lateness_minutes": float(
                    group[
                        "lateness_minutes"
                    ].sum()
                ),

                "mean_lateness_minutes": float(
                    group[
                        "lateness_minutes"
                    ].mean()
                ),

                "mean_cluster_switches": float(
                    group[
                        "cluster_switches"
                    ].mean()
                ),

                "mean_training_runtime_seconds": float(
                    group[
                        "training_runtime_seconds"
                    ].mean()
                ),

                "total_training_runtime_seconds": float(
                    group[
                        "training_runtime_seconds"
                    ].sum()
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # So sánh metric vật lý với Config A.
    #
    # Không so objective_cost giữa configs vì weights khác nhau.
    # --------------------------------------------------------

    baseline = summary[
        summary["config_name"]
        == "A_current"
    ].iloc[0]

    baseline_distance = float(
        baseline[
            "mean_distance_km"
        ]
    )

    baseline_time = float(
        baseline[
            "mean_travel_time_minutes"
        ]
    )

    baseline_lateness = float(
        baseline[
            "total_lateness_minutes"
        ]
    )

    summary[
        "distance_improvement_vs_A_pct"
    ] = (
        (
            baseline_distance
            - summary["mean_distance_km"]
        )
        / baseline_distance
        * 100.0
    )

    summary[
        "travel_time_improvement_vs_A_pct"
    ] = (
        (
            baseline_time
            - summary[
                "mean_travel_time_minutes"
            ]
        )
        / baseline_time
        * 100.0
    )

    summary[
        "sla_change_vs_A_percentage_points"
    ] = (
        (
            summary["weighted_sla"]
            - float(
                baseline[
                    "weighted_sla"
                ]
            )
        )
        * 100.0
    )

    summary[
        "late_delivery_change_vs_A"
    ] = (
        summary[
            "total_late_deliveries"
        ]
        - int(
            baseline[
                "total_late_deliveries"
            ]
        )
    )

    if baseline_lateness > 0.0:
        summary[
            "lateness_reduction_vs_A_pct"
        ] = (
            (
                baseline_lateness
                - summary[
                    "total_lateness_minutes"
                ]
            )
            / baseline_lateness
            * 100.0
        )
    else:
        summary[
            "lateness_reduction_vs_A_pct"
        ] = 0.0

    return summary


# ============================================================
# FINDINGS
# ============================================================

def write_findings(
    *,
    summary: pd.DataFrame,
) -> None:
    lines = [
        (
            "PROPOSED RL OBJECTIVE "
            "SENSITIVITY - PILOT"
        ),
        "",
        (
            "Important: objective_cost values are "
            "not compared across configurations "
            "because each configuration uses "
            "different objective weights."
        ),
        "",
        (
            "Selection must use physical/external "
            "metrics: distance, travel time, SLA, "
            "late deliveries, lateness and "
            "cluster switches."
        ),
        "",
    ]

    for row in summary.itertuples(
        index=False
    ):
        lines.extend(
            [
                f"[{row.config_name}]",
                (
                    f"mean_distance_km = "
                    f"{row.mean_distance_km:.6f}"
                ),
                (
                    f"mean_travel_time_minutes = "
                    f"{row.mean_travel_time_minutes:.6f}"
                ),
                (
                    f"total_late_deliveries = "
                    f"{row.total_late_deliveries}"
                ),
                (
                    f"weighted_sla = "
                    f"{row.weighted_sla:.6f}"
                ),
                (
                    f"total_lateness_minutes = "
                    f"{row.total_lateness_minutes:.6f}"
                ),
                (
                    f"mean_cluster_switches = "
                    f"{row.mean_cluster_switches:.6f}"
                ),
                (
                    f"distance_improvement_vs_A_pct = "
                    f"{row.distance_improvement_vs_A_pct:.6f}"
                ),
                (
                    "travel_time_improvement_vs_A_pct = "
                    f"{row.travel_time_improvement_vs_A_pct:.6f}"
                ),
                (
                    "sla_change_vs_A_percentage_points = "
                    f"{row.sla_change_vs_A_percentage_points:.6f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            (
                "This is a pilot sensitivity study "
                "on workloads 101, 106 and 97."
            ),
            (
                "The final objective configuration "
                "must later be selected using the "
                "validation split of the full "
                "108-workload experiment."
            ),
        ]
    )

    FINDINGS_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)

    print(
        "PROPOSED RL OBJECTIVE "
        "SENSITIVITY - PILOT"
    )

    print("=" * 78)

    print(
        f"Workloads: {WORKLOAD_IDS}"
    )

    print(
        f"Objective configs: "
        f"{tuple(OBJECTIVE_CONFIGS.keys())}"
    )

    print(
        f"Training episodes per run: "
        f"{TRAINING_EPISODES}"
    )

    print(
        f"Total training runs: "
        f"{len(WORKLOAD_IDS) * len(OBJECTIVE_CONFIGS)}"
    )

    # --------------------------------------------------------
    # Load road graph once.
    # --------------------------------------------------------

    road_network = (
        validation.load_road_network()
    )

    depot, depot_node = (
        validation.get_depot_node(
            road_network
        )
    )

    depot_node = int(
        depot_node
    )

    print(
        f"Depot: {depot.name}"
    )

    print(
        f"Depot road node: "
        f"{depot_node}"
    )

    matrix_builder = (
        validation.RoadMatrixBuilder(
            road_network
        )
    )

    all_rows: list[
        dict[str, object]
    ] = []

    experiment_start = (
        perf_counter()
    )

    # Workload outer loop:
    # build road matrix once, then run all four configs.
    for workload_id in WORKLOAD_IDS:
        prepared = prepare_workload(
            workload_id=workload_id,
            matrix_builder=matrix_builder,
            depot_node=depot_node,
        )

        for (
            config_name,
            objective_config,
        ) in OBJECTIVE_CONFIGS.items():
            row = run_one_configuration(
                prepared=prepared,
                config_name=config_name,
                objective_config=objective_config,
            )

            all_rows.append(
                row
            )

            # Autosave after every run.
            pd.DataFrame(
                all_rows
            ).to_csv(
                RUNS_PATH,
                index=False,
            )

    experiment_runtime = (
        perf_counter()
        - experiment_start
    )

    matrix_builder.clear_cache()

    runs = pd.DataFrame(
        all_rows
    )

    summary = build_summary(
        runs
    )

    runs.to_csv(
        RUNS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    write_findings(
        summary=summary
    )

    print()
    print("=" * 78)

    print("SENSITIVITY SUMMARY")

    print("=" * 78)

    display_columns = [
        "config_name",
        "mean_distance_km",
        "mean_travel_time_minutes",
        "total_late_deliveries",
        "weighted_sla",
        "total_lateness_minutes",
        "mean_cluster_switches",
        "distance_improvement_vs_A_pct",
        "travel_time_improvement_vs_A_pct",
        "sla_change_vs_A_percentage_points",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Experiment runtime: "
        f"{experiment_runtime / 60.0:.2f} min"
    )

    print()

    print(
        f"Runs saved to: "
        f"{RUNS_PATH}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Findings saved to: "
        f"{FINDINGS_PATH}"
    )


if __name__ == "__main__":
    main()