from pathlib import Path
import random
from statistics import mean
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
    OBJECTIVE_CONFIGS,
    RESULTS_DIR,
    load_split,
    load_workload_environments,
)
from rl.proposed.agent import (  # noqa: E402
    ProposedAgentConfig,
    ProposedRoutingAgent,
)
from rl.proposed.candidate_encoder import CandidateEncoder  # noqa: E402
from rl.proposed.exact_solver import (  # noqa: E402
    exact_optimal_route,
    route_cost,
)
from rl.proposed.trainer import evaluate_agent  # noqa: E402


CONFIG_NAME = "D_moderate_sla"
CHECKPOINT_PATH = (
    RESULTS_DIR / "proposed_rl_multiworkload_D_moderate_sla_best.pt"
)

DETAILS_PATH = RESULTS_DIR / "proposed_rl_headroom_validation_details.csv"
FINDINGS_PATH = RESULTS_DIR / "proposed_rl_headroom_validation_findings.txt"

RANDOM_ORDERS = 200
SEED = 42

METHODS = (
    "random_mean",
    "nearest_time",
    "nearest_distance",
    "nearest_time_local_search",
    "proposed_rl_locked",
    "exact_optimal",
)


def nearest_neighbour(env, key: str) -> list[int]:
    env.reset()

    while not env.done:
        action = min(
            env.available_actions(),
            key=lambda a: getattr(env.candidate_info(a), key),
        )
        env.step(action)

    return list(env.route_actions)


def local_search(env, order: list[int]) -> list[int]:
    """
    First-improvement 2-opt (reversal) and relocate moves under J.
    """

    best = list(order)
    best_cost = route_cost(env, best)
    n = len(best)

    improved = True

    while improved:
        improved = False

        candidates = []

        for i in range(n - 1):
            for j in range(i + 1, n):
                candidates.append(best[:i] + best[i : j + 1][::-1] + best[j + 1 :])

        for i in range(n):
            for j in range(n):
                if i != j:
                    moved = best[:i] + best[i + 1 :]
                    candidates.append(moved[:j] + [best[i]] + moved[j:])

        for candidate in candidates:
            cost = route_cost(env, candidate)

            if cost < best_cost - 1e-12:
                best, best_cost = candidate, cost
                improved = True
                break

    return best


def random_mean_cost(env, rng: random.Random) -> float:
    n = env.number_of_deliveries

    return mean(
        route_cost(env, rng.sample(range(n), n))
        for _ in range(RANDOM_ORDERS)
    )


def load_locked_agent() -> ProposedRoutingAgent:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    agent = ProposedRoutingAgent(
        config=ProposedAgentConfig(**checkpoint["agent_config"]),
        feature_dimension=int(checkpoint["feature_dimension"]),
    )
    agent.online_network.load_state_dict(
        checkpoint["online_network_state_dict"]
    )

    return agent


def physical_summary(env, order) -> dict:
    route_cost(env, order)

    totals = env.totals

    return {
        "distance_km": totals.total_distance_km,
        "late_deliveries": totals.late_deliveries,
    }


def main() -> None:
    started = perf_counter()

    validation = load_workload_environments(
        split=load_split(),
        split_name="validation",
        objective_config=OBJECTIVE_CONFIGS[CONFIG_NAME],
    )

    print(f"validation workloads = {len(validation)} (test not used)")

    encoder = CandidateEncoder()
    agent = load_locked_agent()
    rng = random.Random(SEED)

    rows = []

    for workload in validation:
        env = workload.environment
        n = env.number_of_deliveries

        tic = perf_counter()
        optimal_cost, optimal_order = exact_optimal_route(env)
        exact_seconds = perf_counter() - tic

        # The solver's own formula must agree with the environment.
        assert abs(route_cost(env, optimal_order) - optimal_cost) < 1e-9

        nearest_time = nearest_neighbour(env, "travel_time_minutes")
        nearest_distance = nearest_neighbour(env, "distance_km")
        searched = local_search(env, nearest_time)

        row = {
            "workload_id": workload.workload_id,
            "deliveries": n,
            "random_mean": random_mean_cost(env, rng),
            "nearest_time": route_cost(env, nearest_time),
            "nearest_distance": route_cost(env, nearest_distance),
            "nearest_time_local_search": route_cost(env, searched),
            "proposed_rl_locked": evaluate_agent(
                env=env,
                encoder=encoder,
                agent=agent,
            ).objective_cost,
            "exact_optimal": optimal_cost,
            "exact_seconds": exact_seconds,
        }

        optimal = physical_summary(env, optimal_order)
        row["optimal_distance_km"] = optimal["distance_km"]
        row["optimal_late_deliveries"] = optimal["late_deliveries"]

        rows.append(row)

        print(
            f"WL {workload.workload_id:>3} n={n:>2}  "
            f"optimal={optimal_cost:.4f}  nn={row['nearest_time']:.4f}  "
            f"nn+ls={row['nearest_time_local_search']:.4f}  "
            f"rl={row['proposed_rl_locked']:.4f}  ({exact_seconds:.1f}s)"
        )

    details = pd.DataFrame(rows)
    details.to_csv(DETAILS_PATH, index=False)

    optimal_mean = details["exact_optimal"].mean()
    random_mean_value = details["random_mean"].mean()

    lines = [
        "PROPOSED RL HEADROOM - VALIDATION ONLY",
        "",
        f"objective = {CONFIG_NAME}, workloads = {len(details)}, "
        f"deliveries = {int(details['deliveries'].sum())}, "
        f"sizes = {sorted(details['deliveries'].tolist())}",
        "test workloads used = 0",
        "exact_optimal is the true minimum of J over all delivery orders",
        "(dynamic programming, checked against brute force in the tests).",
        "The locked checkpoint was chosen on these same validation workloads,",
        "so its validation J is slightly optimistic.",
        "",
        f"{'method':28s} {'mean J':>9s} {'gap to opt':>11s} "
        f"{'mean per-WL gap':>16s} {'worst WL gap':>13s} "
        f"{'WLs at optimum':>15s} {'random->opt closed':>19s}",
    ]

    for method in METHODS:
        gaps = (details[method] / details["exact_optimal"] - 1.0) * 100.0
        closed = (
            (random_mean_value - details[method].mean())
            / (random_mean_value - optimal_mean)
            * 100.0
        )

        lines.append(
            f"{method:28s} {details[method].mean():9.4f} "
            f"{(details[method].mean() / optimal_mean - 1.0) * 100.0:10.2f}% "
            f"{gaps.mean():15.2f}% {gaps.max():12.2f}% "
            f"{int((gaps < 1e-7).sum()):>12d}/{len(details)} "
            f"{closed:18.1f}%"
        )

    lines += [
        "",
        f"optimal route: mean distance = "
        f"{details['optimal_distance_km'].mean():.2f} km, "
        f"late deliveries = {int(details['optimal_late_deliveries'].sum())}",
        f"exact solver runtime = {details['exact_seconds'].sum():.1f}s total",
    ]

    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print(f"\nRuntime: {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
