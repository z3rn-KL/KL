from pathlib import Path
from statistics import mean, pstdev
import sys
from time import perf_counter

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

from experiments.proposed_rl_multi_workload_training import (  # noqa: E402
    EPOCHS,
    EPISODES_PER_WORKLOAD,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    OBJECTIVE_CONFIGS,
    RESULTS_DIR,
    VALIDATION_INTERVAL,
    history_path,
    load_split,
    load_workload_environments,
)
from rl.proposed.agent import (  # noqa: E402
    ProposedAgentConfig,
    ProposedRoutingAgent,
)
from rl.proposed.lookahead_encoder import (  # noqa: E402
    LookaheadCandidateEncoder,
)
from rl.proposed.multi_workload_trainer import (  # noqa: E402
    evaluate_workloads,
    train_multi_workload_epoch,
)


CONFIG_NAME = "D_moderate_sla"
SEEDS = (42, 43, 44)

FINDINGS_PATH = RESULTS_DIR / "proposed_rl_lookahead_validation_findings.txt"
HISTORY_PATH = RESULTS_DIR / "proposed_rl_lookahead_validation_history.csv"


def build_agent(seed: int, feature_dimension: int) -> ProposedRoutingAgent:
    # Same hyperparameters as the locked 8-feature model.
    return ProposedRoutingAgent(
        config=ProposedAgentConfig(
            learning_rate=1e-3,
            gamma=1.0,
            epsilon_start=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            epsilon_decay=EPSILON_DECAY,
            replay_capacity=10_000,
            batch_size=64,
            min_replay_size=128,
            target_update_interval=100,
            hidden_dimension=64,
            gradient_clip_norm=5.0,
            seed=seed,
            device="auto",
        ),
        feature_dimension=feature_dimension,
    )


def greedy_nearest(workloads, key: str) -> dict:
    """
    Nearest-neighbour reference run through the same environment,
    so its objective is directly comparable.
    """

    objectives, distances, times = [], [], []
    late = valid = 0

    for workload in workloads:
        env = workload.environment
        env.reset()

        while not env.done:
            action = min(
                env.available_actions(),
                key=lambda a: getattr(env.candidate_info(a), key),
            )
            env.step(action)

        objectives.append(env.final_objective().total_cost)
        distances.append(env.totals.total_distance_km)
        times.append(env.totals.total_travel_time_minutes)
        late += env.totals.late_deliveries
        valid += env.objective_context.number_of_valid_deadlines

    return {
        "objective": mean(objectives),
        "distance_km": mean(distances),
        "travel_time_minutes": mean(times),
        "late_deliveries": late,
        "weighted_sla": (valid - late) / valid,
    }


def train_seed(seed, train, validation, encoder, history):
    agent = build_agent(seed, encoder.feature_dimension)

    best = None

    for epoch in range(1, EPOCHS + 1):
        train_multi_workload_epoch(
            workloads=train,
            encoder=encoder,
            agent=agent,
            epoch=epoch,
            episodes_per_workload=EPISODES_PER_WORKLOAD,
            shuffle=True,
            seed=seed,
        )

        if epoch % VALIDATION_INTERVAL and epoch != EPOCHS:
            continue

        evaluations, summary = evaluate_workloads(
            workloads=validation,
            encoder=encoder,
            agent=agent,
        )

        objective = mean(item.objective_cost for item in evaluations)

        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "validation_objective": objective,
                "distance_km": summary.mean_distance_km,
                "travel_time_minutes": summary.mean_travel_time_minutes,
                "late_deliveries": summary.late_deliveries,
                "weighted_sla": summary.weighted_sla,
            }
        )

        if best is None or objective < best["objective"]:
            best = {
                "objective": objective,
                "epoch": epoch,
                "distance_km": summary.mean_distance_km,
                "travel_time_minutes": summary.mean_travel_time_minutes,
                "late_deliveries": summary.late_deliveries,
                "weighted_sla": summary.weighted_sla,
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in agent.online_network.state_dict().items()
                },
            }

    return best


def previous_model_validation() -> dict:
    """
    Best validation row of the locked 8-feature D model.
    """

    history = pd.read_csv(history_path(CONFIG_NAME))
    best = history[history["is_best_checkpoint"]].iloc[-1]

    return {
        "objective": best["validation_mean_objective"],
        "distance_km": best["validation_mean_distance_km"],
        "travel_time_minutes": best["validation_mean_travel_time_minutes"],
        "late_deliveries": best["validation_late_deliveries"],
        "weighted_sla": best["validation_weighted_sla"],
    }


def main() -> None:
    started = perf_counter()

    split = load_split()
    objective = OBJECTIVE_CONFIGS[CONFIG_NAME]

    train = load_workload_environments(
        split=split,
        split_name="train",
        objective_config=objective,
    )
    validation = load_workload_environments(
        split=split,
        split_name="validation",
        objective_config=objective,
    )

    print(f"train = {len(train)}, validation = {len(validation)} (test not used)")

    encoder = LookaheadCandidateEncoder()
    history: list[dict] = []
    results = []

    for seed in SEEDS:
        best = train_seed(seed, train, validation, encoder, history)
        results.append((seed, best))

        print(
            f"seed {seed}: best epoch {best['epoch']}, "
            f"validation objective {best['objective']:.4f}, "
            f"late {best['late_deliveries']}"
        )

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    best_seed, best_run = min(results, key=lambda item: item[1]["objective"])

    torch.save(
        {
            "seed": best_seed,
            "epoch": best_run["epoch"],
            "feature_names": encoder.feature_names,
            "online_network_state_dict": best_run["state"],
        },
        RESULTS_DIR / "proposed_rl_lookahead_D_best.pt",
    )

    objectives = [run["objective"] for _, run in results]

    reference = {
        "greedy nearest (travel time)": greedy_nearest(
            validation, "travel_time_minutes"
        ),
        "greedy nearest (distance)": greedy_nearest(validation, "distance_km"),
        "previous Proposed RL (8 features)": previous_model_validation(),
    }

    lines = [
        "PROPOSED RL WITH LOOKAHEAD FEATURES - VALIDATION ONLY",
        "",
        f"objective config = {CONFIG_NAME}",
        f"train workloads = {len(train)}, validation workloads = {len(validation)}",
        "test workloads used = 0",
        f"seeds = {list(SEEDS)}",
        "Objective values are comparable: same objective, same validation workloads.",
        "",
        "LOOKAHEAD MODEL (13 features), best checkpoint per seed",
    ]

    for seed, run in results:
        lines.append(
            f"  seed {seed}: epoch {run['epoch']}, "
            f"objective = {run['objective']:.4f}, "
            f"distance = {run['distance_km']:.2f} km, "
            f"late = {run['late_deliveries']}, "
            f"SLA = {run['weighted_sla']:.4f}"
        )

    lines.append(
        f"  mean objective = {mean(objectives):.4f}, "
        f"std = {pstdev(objectives):.4f}, best seed = {best_seed}"
    )
    lines.append("")
    lines.append("REFERENCES ON THE SAME VALIDATION WORKLOADS")

    for name, row in reference.items():
        lines.append(
            f"  {name}: objective = {row['objective']:.4f}, "
            f"distance = {row['distance_km']:.2f} km, "
            f"late = {int(row['late_deliveries'])}, "
            f"SLA = {row['weighted_sla']:.4f}"
        )

    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print(f"\nRuntime: {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
