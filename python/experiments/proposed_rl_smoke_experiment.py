from __future__ import annotations

import csv
from itertools import permutations
from pathlib import Path
from time import perf_counter

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

HISTORY_PATH = (
    RESULTS_DIR
    / "proposed_rl_smoke_training_history.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "proposed_rl_smoke_summary.txt"
)


TRAINING_EPISODES = 1500
EVALUATION_INTERVAL = 100


def build_smoke_environment(
) -> ProposedRoutingEnvironment:
    """
    Workload kiểm chứng nhỏ gồm 4 deliveries.

    Tuyến được thiết kế để có nghiệm tối ưu rõ:

        depot -> d1 -> d2 -> d3 -> d4 -> depot

    Road matrix là directional, phù hợp với
    cách project thật đang sử dụng road network.
    """

    distance_matrix_km = [
        # depot, d1, d2, d3, d4
        [0.0, 2.0, 6.0, 8.0, 10.0],
        [8.0, 0.0, 2.0, 6.0, 8.0],
        [7.0, 7.0, 0.0, 2.0, 6.0],
        [6.0, 8.0, 7.0, 0.0, 2.0],
        [2.0, 8.0, 7.0, 7.0, 0.0],
    ]

    travel_time_matrix_minutes = [
        # depot, d1, d2, d3, d4
        [0.0, 4.0, 12.0, 16.0, 20.0],
        [16.0, 0.0, 4.0, 12.0, 16.0],
        [14.0, 14.0, 0.0, 4.0, 12.0],
        [12.0, 16.0, 14.0, 0.0, 4.0],
        [4.0, 16.0, 14.0, 14.0, 0.0],
    ]

    deadlines_minutes = [
        6.0,
        15.0,
        25.0,
        40.0,
    ]

    cluster_ids = [
        0,
        0,
        1,
        1,
    ]

    return ProposedRoutingEnvironment(
        distance_matrix_km=distance_matrix_km,
        travel_time_matrix_minutes=(
            travel_time_matrix_minutes
        ),
        deadlines_minutes=deadlines_minutes,
        cluster_ids=cluster_ids,
    )


def evaluate_fixed_route(
    *,
    env: ProposedRoutingEnvironment,
    route_actions: tuple[int, ...],
) -> dict[str, float | int | tuple[int, ...]]:
    """
    Chạy một route cố định qua environment.

    Dùng cho brute-force ground truth.
    """

    env.reset()

    total_reward = 0.0

    for action in route_actions:
        result = env.step(action)

        total_reward += result.reward

    objective = env.final_objective()
    totals = env.totals

    return {
        "route_actions": route_actions,
        "objective_cost": objective.total_cost,
        "total_reward": total_reward,
        "distance_km": totals.total_distance_km,
        "travel_time_minutes": (
            totals.total_travel_time_minutes
        ),
        "lateness_minutes": (
            totals.total_lateness_minutes
        ),
        "cluster_switches": totals.cluster_switches,
    }


def find_bruteforce_optimum(
    env: ProposedRoutingEnvironment,
) -> dict[str, float | int | tuple[int, ...]]:
    """
    Với N=4 chỉ có:

        4! = 24

    route nên có thể kiểm tra toàn bộ.
    """

    best_result = None

    actions = range(
        env.number_of_deliveries
    )

    for route_actions in permutations(actions):
        result = evaluate_fixed_route(
            env=env,
            route_actions=route_actions,
        )

        if (
            best_result is None
            or result["objective_cost"]
            < best_result["objective_cost"]
        ):
            best_result = result

    if best_result is None:
        raise RuntimeError(
            "Brute-force search returned no route."
        )

    return best_result


def route_to_text(
    route_actions: tuple[int, ...],
) -> str:
    deliveries = [
        f"D{action + 1}"
        for action in route_actions
    ]

    return (
        "Depot -> "
        + " -> ".join(deliveries)
        + " -> Depot"
    )


def improvement_percent(
    *,
    baseline: float,
    new: float,
) -> float:
    if baseline == 0.0:
        return 0.0

    return (
        (baseline - new)
        / baseline
        * 100.0
    )


def gap_percent(
    *,
    value: float,
    optimum: float,
) -> float:
    if optimum == 0.0:
        return 0.0

    return (
        (value - optimum)
        / optimum
        * 100.0
    )


def save_history(
    rows: list[dict[str, object]],
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "episode",
        "epsilon",
        "route",
        "objective_cost",
        "optimality_gap_pct",
        "distance_km",
        "travel_time_minutes",
        "lateness_minutes",
        "cluster_switches",
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

        writer.writerows(rows)


def main() -> None:
    env = build_smoke_environment()

    encoder = CandidateEncoder()

    agent = ProposedRoutingAgent(
        config=ProposedAgentConfig(
            learning_rate=1e-3,
            gamma=1.0,

            epsilon_start=1.0,
            epsilon_min=0.02,
            epsilon_decay=0.995,

            replay_capacity=5000,
            batch_size=32,
            min_replay_size=64,

            target_update_interval=50,

            hidden_dimension=64,

            gradient_clip_norm=5.0,

            seed=42,
            device="auto",
        )
    )

    print("=" * 72)
    print("PROPOSED RL - CONTROLLED SMOKE EXPERIMENT")
    print("=" * 72)

    print()
    print(
        f"Device: {agent.device}"
    )

    print(
        f"Training episodes: "
        f"{TRAINING_EPISODES}"
    )

    print(
        f"Candidate feature dimension: "
        f"{agent.feature_dimension}"
    )

    # ---------------------------------------------------------
    # 1. Ground-truth optimum
    # ---------------------------------------------------------

    optimum = find_bruteforce_optimum(
        env
    )

    optimum_cost = float(
        optimum["objective_cost"]
    )

    print()
    print("BRUTE-FORCE OPTIMUM")
    print("-" * 72)

    print(
        "Route:",
        route_to_text(
            optimum["route_actions"]
        ),
    )

    print(
        f"Objective: "
        f"{optimum_cost:.6f}"
    )

    print(
        f"Distance: "
        f"{optimum['distance_km']:.3f} km"
    )

    print(
        f"Travel time: "
        f"{optimum['travel_time_minutes']:.3f} min"
    )

    print(
        f"Lateness: "
        f"{optimum['lateness_minutes']:.3f} min"
    )

    print(
        f"Cluster switches: "
        f"{optimum['cluster_switches']}"
    )

    # ---------------------------------------------------------
    # 2. Agent before training
    # ---------------------------------------------------------

    initial = evaluate_agent(
        env=env,
        encoder=encoder,
        agent=agent,
    )

    initial_gap = gap_percent(
        value=initial.objective_cost,
        optimum=optimum_cost,
    )

    print()
    print("UNTRAINED AGENT")
    print("-" * 72)

    print(
        "Route:",
        route_to_text(
            initial.route_actions
        ),
    )

    print(
        f"Objective: "
        f"{initial.objective_cost:.6f}"
    )

    print(
        f"Optimality gap: "
        f"{initial_gap:.2f}%"
    )

    # ---------------------------------------------------------
    # 3. Training
    # ---------------------------------------------------------

    history_rows: list[
        dict[str, object]
    ] = []

    start_time = perf_counter()

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
            or episode % EVALUATION_INTERVAL == 0
            or episode == TRAINING_EPISODES
        )

        if not should_evaluate:
            continue

        evaluation = evaluate_agent(
            env=env,
            encoder=encoder,
            agent=agent,
        )

        gap = gap_percent(
            value=evaluation.objective_cost,
            optimum=optimum_cost,
        )

        row = {
            "episode": episode,
            "epsilon": agent.epsilon,
            "route": route_to_text(
                evaluation.route_actions
            ),
            "objective_cost": (
                evaluation.objective_cost
            ),
            "optimality_gap_pct": gap,
            "distance_km": (
                evaluation.total_distance_km
            ),
            "travel_time_minutes": (
                evaluation.total_travel_time_minutes
            ),
            "lateness_minutes": (
                evaluation.total_lateness_minutes
            ),
            "cluster_switches": (
                evaluation.cluster_switches
            ),
        }

        history_rows.append(
            row
        )

        print(
            f"Episode {episode:4d} | "
            f"epsilon={agent.epsilon:.4f} | "
            f"objective="
            f"{evaluation.objective_cost:.6f} | "
            f"gap={gap:7.2f}% | "
            f"route="
            f"{evaluation.route_actions}"
        )

    training_runtime = (
        perf_counter()
        - start_time
    )

    # ---------------------------------------------------------
    # 4. Final evaluation
    # ---------------------------------------------------------

    final = evaluate_agent(
        env=env,
        encoder=encoder,
        agent=agent,
    )

    final_gap = gap_percent(
        value=final.objective_cost,
        optimum=optimum_cost,
    )

    improvement_vs_initial = (
        improvement_percent(
            baseline=initial.objective_cost,
            new=final.objective_cost,
        )
    )

    smoke_pass = (
        final_gap <= 5.0
    )

    print()
    print("FINAL EVALUATION")
    print("-" * 72)

    print(
        "Route:",
        route_to_text(
            final.route_actions
        ),
    )

    print(
        f"Objective: "
        f"{final.objective_cost:.6f}"
    )

    print(
        f"Distance: "
        f"{final.total_distance_km:.3f} km"
    )

    print(
        f"Travel time: "
        f"{final.total_travel_time_minutes:.3f} min"
    )

    print(
        f"Lateness: "
        f"{final.total_lateness_minutes:.3f} min"
    )

    print(
        f"Cluster switches: "
        f"{final.cluster_switches}"
    )

    print(
        f"Optimality gap: "
        f"{final_gap:.2f}%"
    )

    print(
        f"Improvement vs untrained: "
        f"{improvement_vs_initial:.2f}%"
    )

    print(
        f"Training runtime: "
        f"{training_runtime:.2f} s"
    )

    print()

    if smoke_pass:
        print(
            "SMOKE STATUS: PASS"
        )
    else:
        print(
            "SMOKE STATUS: NEEDS TUNING"
        )

    # ---------------------------------------------------------
    # 5. Save artifacts
    # ---------------------------------------------------------

    save_history(
        history_rows
    )

    summary_lines = [
        "PROPOSED RL CONTROLLED SMOKE EXPERIMENT",
        "",
        (
            "This experiment is a controlled "
            "mechanism validation, not a thesis "
            "benchmark result."
        ),
        "",
        (
            f"Training episodes: "
            f"{TRAINING_EPISODES}"
        ),
        (
            f"Training runtime seconds: "
            f"{training_runtime:.6f}"
        ),
        "",
        "Brute-force optimum:",
        (
            f"  route = "
            f"{route_to_text(optimum['route_actions'])}"
        ),
        (
            f"  objective = "
            f"{optimum_cost:.9f}"
        ),
        "",
        "Untrained agent:",
        (
            f"  route = "
            f"{route_to_text(initial.route_actions)}"
        ),
        (
            f"  objective = "
            f"{initial.objective_cost:.9f}"
        ),
        (
            f"  gap_pct = "
            f"{initial_gap:.6f}"
        ),
        "",
        "Trained agent:",
        (
            f"  route = "
            f"{route_to_text(final.route_actions)}"
        ),
        (
            f"  objective = "
            f"{final.objective_cost:.9f}"
        ),
        (
            f"  gap_pct = "
            f"{final_gap:.6f}"
        ),
        (
            f"  improvement_vs_untrained_pct = "
            f"{improvement_vs_initial:.6f}"
        ),
        "",
        (
            f"Smoke status = "
            f"{'PASS' if smoke_pass else 'NEEDS_TUNING'}"
        ),
    ]

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
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


if __name__ == "__main__":
    main()