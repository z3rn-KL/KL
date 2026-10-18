from __future__ import annotations

import csv
import math
from pathlib import Path
from time import perf_counter

import pandas as pd

import experiments.final_kmeans_guided_dqn_validation as validation
import experiments.kmeans_guided_dqn_sensitivity as guided

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
from rl.proposed.trainer import (
    evaluate_agent,
    run_training_episode,
)


# ============================================================
# PATHS / CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

WORKLOAD_ID = 97

TRAINING_EPISODES = 3000
EVALUATION_INTERVAL = 250

HISTORY_PATH = (
    RESULTS_DIR
    / f"proposed_rl_real_workload_{WORKLOAD_ID}_history.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / f"proposed_rl_real_workload_{WORKLOAD_ID}_summary.txt"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_real_workload(
    workload_id: int,
) -> pd.DataFrame:
    """
    Dựng lại đúng workload đã được sử dụng trong
    final K-Means-guided DQN validation.

    Mục đích:
        - giữ cùng workload với baseline cũ;
        - giữ cùng thứ tự delivery;
        - giữ road_node và cluster_id đã chuẩn hóa.
    """

    manifest = validation.load_workload_manifest()

    baseline_dqn = validation.load_baseline_dqn(
        manifest
    )

    selected = validation.select_final_workloads(
        manifest,
        baseline_dqn,
    )

    matches = selected[
        selected["workload_id"] == workload_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one selected workload "
            f"for workload_id={workload_id}, "
            f"found {len(matches)}."
        )

    prepared_dataframe, _ = (
        guided.prepare_experiment_dataframe()
    )

    selected_row = matches.iloc[0]

    workload = (
        validation.build_workload_dataframe_from_manifest(
            prepared_dataframe,
            selected_row,
        )
    )

    if workload.empty:
        raise ValueError(
            f"Workload {workload_id} is empty."
        )

    required_columns = {
        "delivery_id",
        "expected_delivery_time",
        "wave_end",
        "road_node",
        "cluster_id",
    }

    missing_columns = (
        required_columns
        - set(workload.columns)
    )

    if missing_columns:
        raise ValueError(
            "Workload is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return workload.reset_index(
        drop=True
    )


# ============================================================
# TIME / DEADLINE PREPARATION
# ============================================================

def resolve_start_time(
    workload: pd.DataFrame,
) -> pd.Timestamp:
    """
    Pipeline benchmark cũ sử dụng wave_end làm start_time.

    Proposed RL giữ đúng quy ước này để việc so sánh
    deadline / SLA với baseline được công bằng.
    """

    wave_end = pd.to_datetime(
        workload["wave_end"],
        utc=True,
        errors="raise",
    )

    unique_values = (
        wave_end
        .dropna()
        .unique()
    )

    if len(unique_values) != 1:
        raise ValueError(
            "Expected exactly one wave_end "
            "for the workload."
        )

    return pd.Timestamp(
        unique_values[0]
    )


def build_deadlines_minutes(
    *,
    workload: pd.DataFrame,
    start_time: pd.Timestamp,
) -> list[float | None]:
    """
    Chuyển expected_delivery_time thành số phút
    tính từ thời điểm route bắt đầu.

    Ví dụ:

        route_start = 04:00
        deadline    = 09:00

        -> deadline_minutes = 300

    Missing deadline:
        -> None
    """

    deadlines: list[
        float | None
    ] = []

    parsed_deadlines = pd.to_datetime(
        workload["expected_delivery_time"],
        utc=True,
        errors="coerce",
    )

    for deadline in parsed_deadlines:
        if pd.isna(deadline):
            deadlines.append(
                None
            )
            continue

        relative_minutes = (
            (
                pd.Timestamp(deadline)
                - start_time
            ).total_seconds()
            / 60.0
        )

        # Environment hiện tại yêu cầu deadline
        # tương đối không âm.
        #
        # Không tự clamp về 0 vì như vậy sẽ thay đổi
        # ý nghĩa dữ liệu thật.
        if relative_minutes < 0.0:
            raise ValueError(
                "Found a delivery whose deadline is "
                "earlier than route start time. "
                f"Relative deadline = "
                f"{relative_minutes:.6f} minutes. "
                "Negative deadlines must be handled "
                "explicitly before dataset-scale training."
            )

        deadlines.append(
            float(relative_minutes)
        )

    return deadlines


# ============================================================
# REAL ROAD ENVIRONMENT
# ============================================================

def build_real_environment(
    workload: pd.DataFrame,
) -> tuple[
    ProposedRoutingEnvironment,
    object,
    int,
    pd.Timestamp,
]:
    """
    Xây ProposedRoutingEnvironment từ dữ liệu thật:

        depot thật
        +
        road nodes thật
        +
        directional road matrix
        +
        deadline thật
        +
        K-Means cluster thật
    """

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

    # --------------------------------------------------------
    # Load road graph
    # --------------------------------------------------------

    road_network = (
        validation.load_road_network()
    )

    # IMPORTANT:
    #
    # get_depot_node() trả về:
    #
    #     (Depot object, road_node_id)
    #
    # chứ KHÔNG trả trực tiếp một int.
    #
    # Đây là lỗi vừa phát hiện ở lần chạy trước.
    # --------------------------------------------------------

    depot, depot_node = (
        validation.get_depot_node(
            road_network
        )
    )

    depot_node = int(
        depot_node
    )

    print()
    print(
        f"Depot: {depot.name}"
    )

    print(
        f"Depot road node: "
        f"{depot_node}"
    )

    # --------------------------------------------------------
    # Build directional road matrix
    # --------------------------------------------------------

    matrix_builder = (
        validation.RoadMatrixBuilder(
            road_network
        )
    )

    node_ids = [
        depot_node,
        *delivery_nodes,
    ]

    print()
    print("Building real road matrix...")

    print(
        f"Nodes: {len(node_ids)} "
        f"(1 depot + "
        f"{len(delivery_nodes)} deliveries)"
    )

    matrix_start = perf_counter()

    road_matrix = matrix_builder.build(
        node_ids
    )

    matrix_runtime = (
        perf_counter()
        - matrix_start
    )

    print(
        f"Road matrix runtime: "
        f"{matrix_runtime:.3f} s"
    )

    # RoadMatrixResult đã được kiểm tra:
    #
    #     node_ids
    #     distance_km
    #     travel_time_minutes

    environment = ProposedRoutingEnvironment(
        distance_matrix_km=(
            road_matrix.distance_km.tolist()
        ),
        travel_time_matrix_minutes=(
            road_matrix
            .travel_time_minutes
            .tolist()
        ),
        deadlines_minutes=(
            deadlines_minutes
        ),
        cluster_ids=(
            cluster_ids
        ),
    )

    return (
        environment,
        matrix_builder,
        depot_node,
        start_time,
    )


# ============================================================
# ROUTE METRICS
# ============================================================

def compute_route_sla(
    *,
    env: ProposedRoutingEnvironment,
    route_actions: tuple[int, ...],
) -> dict[str, float | int]:
    """
    Tính SLA dùng đúng arrival/lateness logic
    của ProposedRoutingEnvironment.

    Delivery không có deadline:
        không nằm trong mẫu số SLA.
    """

    env.reset()

    valid_deadlines = 0
    on_time_deliveries = 0
    late_deliveries = 0

    for action in route_actions:
        result = env.step(
            action
        )

        candidate = result.candidate

        if not candidate.deadline_valid:
            continue

        valid_deadlines += 1

        if (
            candidate.lateness_minutes
            <= 1e-9
        ):
            on_time_deliveries += 1

        else:
            late_deliveries += 1

    if valid_deadlines == 0:
        on_time_rate = 1.0

    else:
        on_time_rate = (
            on_time_deliveries
            / valid_deadlines
        )

    return {
        "valid_deadlines": (
            valid_deadlines
        ),
        "on_time_deliveries": (
            on_time_deliveries
        ),
        "late_deliveries": (
            late_deliveries
        ),
        "on_time_rate": (
            on_time_rate
        ),
    }


def route_to_delivery_ids(
    *,
    workload: pd.DataFrame,
    route_actions: tuple[int, ...],
) -> tuple[str, ...]:
    """
    Chuyển action index của RL thành delivery_id thật.
    """

    delivery_ids = (
        workload[
            "delivery_id"
        ]
        .astype(str)
        .tolist()
    )

    return tuple(
        delivery_ids[action]
        for action in route_actions
    )


def route_to_text(
    *,
    route_actions: tuple[int, ...],
) -> str:
    """
    Hiển thị route theo D1, D2, ...
    """

    if not route_actions:
        return (
            "Depot -> Depot"
        )

    middle = " -> ".join(
        f"D{action + 1}"
        for action in route_actions
    )

    return (
        f"Depot -> "
        f"{middle} "
        f"-> Depot"
    )


# ============================================================
# RESULT WRITING
# ============================================================

def save_history(
    rows: list[
        dict[str, object]
    ],
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "episode",
        "epsilon",
        "objective_cost",
        "distance_km",
        "travel_time_minutes",
        "lateness_minutes",
        "cluster_switches",
        "route",
    ]

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main() -> None:
    print(
        "=" * 76
    )

    print(
        "PROPOSED RL - "
        "REAL WORKLOAD SMOKE TEST"
    )

    print(
        "=" * 76
    )

    matrix_builder = None

    try:
        # ====================================================
        # 1. Load real workload
        # ====================================================

        workload = load_real_workload(
            WORKLOAD_ID
        )

        print()
        print("WORKLOAD")
        print("-" * 76)

        print(
            f"Workload ID: "
            f"{WORKLOAD_ID}"
        )

        print(
            f"Deliveries: "
            f"{len(workload)}"
        )

        print(
            "Cluster IDs:",
            workload[
                "cluster_id"
            ]
            .astype(int)
            .tolist(),
        )

        # ====================================================
        # 2. Build real road environment
        # ====================================================

        (
            env,
            matrix_builder,
            depot_node,
            start_time,
        ) = build_real_environment(
            workload
        )

        print()
        print(
            f"Depot node: "
            f"{depot_node}"
        )

        print(
            f"Route start time: "
            f"{start_time}"
        )

        print(
            f"Valid deadlines: "
            f"{env.objective_context.number_of_valid_deadlines}"
        )

        print(
            "Deadline minutes:",
            env.deadlines_minutes,
        )

        # ====================================================
        # 3. Proposed RL components
        # ====================================================

        encoder = CandidateEncoder()

        agent = ProposedRoutingAgent(
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

                seed=42,

                device="auto",
            )
        )

        print()
        print("PROPOSED RL")
        print("-" * 76)

        print(
            f"Device: "
            f"{agent.device}"
        )

        print(
            f"Training episodes: "
            f"{TRAINING_EPISODES}"
        )

        print(
            f"Feature dimension: "
            f"{agent.feature_dimension}"
        )

        # ====================================================
        # 4. Evaluation BEFORE training
        # ====================================================

        initial = evaluate_agent(
            env=env,
            encoder=encoder,
            agent=agent,
        )

        initial_sla = compute_route_sla(
            env=env,
            route_actions=(
                initial.route_actions
            ),
        )

        print()
        print("UNTRAINED AGENT")
        print("-" * 76)

        print(
            "Route:",
            route_to_text(
                route_actions=(
                    initial.route_actions
                )
            ),
        )

        print(
            f"Objective: "
            f"{initial.objective_cost:.9f}"
        )

        print(
            f"Distance: "
            f"{initial.total_distance_km:.6f} km"
        )

        print(
            f"Travel time: "
            f"{initial.total_travel_time_minutes:.6f} min"
        )

        print(
            f"Lateness: "
            f"{initial.total_lateness_minutes:.6f} min"
        )

        print(
            f"Cluster switches: "
            f"{initial.cluster_switches}"
        )

        print(
            f"SLA: "
            f"{initial_sla['on_time_rate'] * 100:.2f}%"
        )

        # ====================================================
        # 5. Training
        # ====================================================

        history_rows: list[
            dict[str, object]
        ] = []

        training_start = (
            perf_counter()
        )

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

            should_evaluate = (
                episode == 1
                or (
                    episode
                    % EVALUATION_INTERVAL
                    == 0
                )
                or (
                    episode
                    == TRAINING_EPISODES
                )
            )

            if not should_evaluate:
                continue

            evaluation = evaluate_agent(
                env=env,
                encoder=encoder,
                agent=agent,
            )

            history_rows.append(
                {
                    "episode": episode,

                    "epsilon": (
                        agent.epsilon
                    ),

                    "objective_cost": (
                        evaluation
                        .objective_cost
                    ),

                    "distance_km": (
                        evaluation
                        .total_distance_km
                    ),

                    "travel_time_minutes": (
                        evaluation
                        .total_travel_time_minutes
                    ),

                    "lateness_minutes": (
                        evaluation
                        .total_lateness_minutes
                    ),

                    "cluster_switches": (
                        evaluation
                        .cluster_switches
                    ),

                    "route": (
                        route_to_text(
                            route_actions=(
                                evaluation
                                .route_actions
                            )
                        )
                    ),
                }
            )

            print(
                f"Episode "
                f"{episode:4d} | "
                f"epsilon="
                f"{agent.epsilon:.4f} | "
                f"J="
                f"{evaluation.objective_cost:.6f} | "
                f"D="
                f"{evaluation.total_distance_km:.3f} km | "
                f"T="
                f"{evaluation.total_travel_time_minutes:.3f} min | "
                f"L="
                f"{evaluation.total_lateness_minutes:.3f} | "
                f"S="
                f"{evaluation.cluster_switches}"
            )

        training_runtime = (
            perf_counter()
            - training_start
        )

        # ====================================================
        # 6. Final inference
        # ====================================================

        final = evaluate_agent(
            env=env,
            encoder=encoder,
            agent=agent,
        )

        final_sla = compute_route_sla(
            env=env,
            route_actions=(
                final.route_actions
            ),
        )

        delivery_route = (
            route_to_delivery_ids(
                workload=workload,
                route_actions=(
                    final.route_actions
                ),
            )
        )

        if (
            initial.objective_cost
            != 0.0
        ):
            objective_improvement_pct = (
                (
                    initial.objective_cost
                    - final.objective_cost
                )
                / initial.objective_cost
                * 100.0
            )
        else:
            objective_improvement_pct = (
                0.0
            )

        print()
        print("FINAL EVALUATION")
        print("-" * 76)

        print(
            "Route:",
            route_to_text(
                route_actions=(
                    final.route_actions
                )
            ),
        )

        print(
            "Delivery IDs:",
            delivery_route,
        )

        print(
            f"Objective: "
            f"{final.objective_cost:.9f}"
        )

        print(
            f"Distance: "
            f"{final.total_distance_km:.6f} km"
        )

        print(
            f"Travel time: "
            f"{final.total_travel_time_minutes:.6f} min"
        )

        print(
            f"Lateness: "
            f"{final.total_lateness_minutes:.6f} min"
        )

        print(
            f"Late deliveries: "
            f"{final_sla['late_deliveries']}"
        )

        print(
            f"SLA: "
            f"{final_sla['on_time_rate'] * 100:.2f}%"
        )

        print(
            f"Cluster switches: "
            f"{final.cluster_switches}"
        )

        print(
            "Objective improvement "
            "vs untrained: "
            f"{objective_improvement_pct:.2f}%"
        )

        print(
            f"Training runtime: "
            f"{training_runtime:.2f} s"
        )

        # ====================================================
        # 7. Integrity checks
        # ====================================================

        expected_actions = set(
            range(
                len(workload)
            )
        )

        actual_actions = set(
            final.route_actions
        )

        valid_route = (
            len(final.route_actions)
            == len(workload)
            and actual_actions
            == expected_actions
        )

        numeric_metrics = [
            final.objective_cost,
            final.total_distance_km,
            final.total_travel_time_minutes,
            final.total_lateness_minutes,
        ]

        finite_metrics = all(
            math.isfinite(value)
            and value >= 0.0
            for value in numeric_metrics
        )

        valid_sla = (
            0.0
            <= final_sla[
                "on_time_rate"
            ]
            <= 1.0
        )

        smoke_pass = (
            valid_route
            and finite_metrics
            and valid_sla
        )

        print()

        if smoke_pass:
            print(
                "REAL WORKLOAD "
                "SMOKE STATUS: PASS"
            )
        else:
            print(
                "REAL WORKLOAD "
                "SMOKE STATUS: FAIL"
            )

        # ====================================================
        # 8. Save artifacts
        # ====================================================

        save_history(
            history_rows
        )

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_lines = [
            (
                "PROPOSED RL - "
                "REAL WORKLOAD SMOKE TEST"
            ),
            "",
            (
                "This is an integration validation "
                "on one real workload. "
                "It is not the final thesis benchmark."
            ),
            "",
            (
                f"workload_id = "
                f"{WORKLOAD_ID}"
            ),
            (
                f"number_of_deliveries = "
                f"{len(workload)}"
            ),
            (
                f"depot_node = "
                f"{depot_node}"
            ),
            (
                f"route_start_time = "
                f"{start_time}"
            ),
            (
                f"training_episodes = "
                f"{TRAINING_EPISODES}"
            ),
            (
                f"training_runtime_seconds = "
                f"{training_runtime:.6f}"
            ),
            "",
            "Untrained agent:",
            (
                f"  route = "
                f"{route_to_text(route_actions=initial.route_actions)}"
            ),
            (
                f"  objective = "
                f"{initial.objective_cost:.9f}"
            ),
            (
                f"  distance_km = "
                f"{initial.total_distance_km:.9f}"
            ),
            (
                f"  travel_time_minutes = "
                f"{initial.total_travel_time_minutes:.9f}"
            ),
            (
                f"  lateness_minutes = "
                f"{initial.total_lateness_minutes:.9f}"
            ),
            (
                f"  cluster_switches = "
                f"{initial.cluster_switches}"
            ),
            (
                f"  on_time_rate = "
                f"{initial_sla['on_time_rate']:.9f}"
            ),
            "",
            "Trained agent:",
            (
                f"  route = "
                f"{route_to_text(route_actions=final.route_actions)}"
            ),
            (
                f"  delivery_ids = "
                f"{delivery_route}"
            ),
            (
                f"  objective = "
                f"{final.objective_cost:.9f}"
            ),
            (
                f"  distance_km = "
                f"{final.total_distance_km:.9f}"
            ),
            (
                f"  travel_time_minutes = "
                f"{final.total_travel_time_minutes:.9f}"
            ),
            (
                f"  lateness_minutes = "
                f"{final.total_lateness_minutes:.9f}"
            ),
            (
                f"  late_deliveries = "
                f"{final_sla['late_deliveries']}"
            ),
            (
                f"  on_time_rate = "
                f"{final_sla['on_time_rate']:.9f}"
            ),
            (
                f"  cluster_switches = "
                f"{final.cluster_switches}"
            ),
            (
                "  objective_improvement_pct = "
                f"{objective_improvement_pct:.6f}"
            ),
            "",
            (
                "status = "
                + (
                    "PASS"
                    if smoke_pass
                    else "FAIL"
                )
            ),
        ]

        SUMMARY_PATH.write_text(
            "\n".join(
                summary_lines
            ),
            encoding="utf-8",
        )

        print()

        print(
            f"History saved to: "
            f"{HISTORY_PATH}"
        )

        print(
            f"Summary saved to: "
            f"{SUMMARY_PATH}"
        )

    finally:
        # Dọn cache dù experiment thành công
        # hay gặp exception giữa chừng.
        if matrix_builder is not None:
            matrix_builder.clear_cache()


if __name__ == "__main__":
    main()