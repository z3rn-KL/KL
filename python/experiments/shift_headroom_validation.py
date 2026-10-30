from pathlib import Path
import random
from statistics import mean
import sys
from time import perf_counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

from experiments.shift_common import RESULTS_DIR, load_sampler  # noqa: E402
from rl.shift.environment import score_sequence  # noqa: E402
from rl.shift.exact_solver import exact_best_shift  # noqa: E402
from rl.shift.heuristics import (  # noqa: E402
    local_search,
    multistart_search,
    nearest_plan,
)


EXACT_SIZES = (8, 10, 12, 14)
EXACT_INSTANCES = 15

LARGE_SIZES = (30,)
LARGE_INSTANCES = 6
LARGE_RANDOM_STARTS = 4

SEED = 42

DETAILS_PATH = RESULTS_DIR / "shift_headroom_validation_details.csv"
FINDINGS_PATH = RESULTS_DIR / "shift_headroom_validation_findings.txt"


def gap_lines(frame: pd.DataFrame, methods: list[str], reference: str) -> list[str]:
    lines = [
        f"{'method':22s} {'mean score':>12s} {'mean gap VND':>13s} "
        f"{'gap % of |ref|':>15s} {'worst gap VND':>14s} {'at reference':>13s}"
    ]

    for method in methods:
        gap = frame[reference] - frame[method]
        percent = gap.sum() / frame[reference].abs().sum() * 100.0

        lines.append(
            f"{method:22s} {frame[method].mean():12.0f} {gap.mean():13.0f} "
            f"{percent:14.2f}% {gap.max():14.0f} "
            f"{int((gap < 1.0).sum()):>9d}/{len(frame)}"
        )

    return lines


def main() -> None:
    started = perf_counter()

    sampler, splits = load_sampler()
    days = splits["validation"]
    rng = random.Random(SEED)

    print(f"validation days = {len(days)} (test not used)")

    rows = []

    for n in EXACT_SIZES:
        for _ in range(EXACT_INSTANCES):
            env = sampler.sample(rng, days, n)

            optimum, plan = exact_best_shift(env)

            assert abs(score_sequence(env, plan) - optimum) < 1e-6

            nn_all = nearest_plan(env, stop_when_not_worth=False)
            nn_stop = nearest_plan(env, stop_when_not_worth=True)

            rows.append(
                {
                    "kind": "exact",
                    "n": n,
                    "reference": optimum,
                    "nn_serve_all": score_sequence(env, nn_all),
                    "nn_stop_rule": score_sequence(env, nn_stop),
                    "nn_all_local_search": score_sequence(
                        env, local_search(env, nn_all)
                    ),
                    "nn_stop_local_search": score_sequence(
                        env, local_search(env, nn_stop)
                    ),
                    "optimal_deferred": n - len(plan),
                }
            )

        print(f"n={n}: exact done ({perf_counter() - started:.0f}s)", flush=True)

    for n in LARGE_SIZES:
        for _ in range(LARGE_INSTANCES):
            env = sampler.sample(rng, days, n)

            best = multistart_search(env, rng, LARGE_RANDOM_STARTS)

            nn_stop = nearest_plan(env, stop_when_not_worth=True)

            rows.append(
                {
                    "kind": "multistart_reference",
                    "n": n,
                    "reference": score_sequence(env, best),
                    "nn_serve_all": score_sequence(
                        env, nearest_plan(env, stop_when_not_worth=False)
                    ),
                    "nn_stop_rule": score_sequence(env, nn_stop),
                    "nn_all_local_search": float("nan"),
                    "nn_stop_local_search": score_sequence(
                        env, local_search(env, nn_stop)
                    ),
                    "optimal_deferred": n - len(best),
                }
            )

        print(f"n={n}: reference done ({perf_counter() - started:.0f}s)", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(DETAILS_PATH, index=False)

    methods = [
        "nn_serve_all",
        "nn_stop_rule",
        "nn_all_local_search",
        "nn_stop_local_search",
    ]

    lines = [
        "SHIFT OBJECTIVE HEADROOM - VALIDATION DAYS ONLY",
        "",
        "Score is the shift score of ShiftObjectiveConfig in VND-equivalent",
        "points (higher is better); gap = reference score - method score.",
        "Costs, fee tariff and penalties are assumed scenario parameters.",
        f"Exact reference: dynamic programming over every plan (sizes {EXACT_SIZES},",
        f"{EXACT_INSTANCES} shifts each). Large sizes use the best of local search",
        "from several starts, which is a lower bound on the true headroom.",
        "",
    ]

    exact = frame[frame["kind"] == "exact"]

    lines.append("EXACT REFERENCE, ALL SIZES")
    lines += gap_lines(exact, methods, "reference")
    lines.append(
        f"optimal plans defer {exact['optimal_deferred'].mean():.2f} "
        f"orders per shift on average"
    )
    lines.append("")

    for n in EXACT_SIZES:
        part = exact[exact["n"] == n]
        lines.append(f"EXACT REFERENCE, n = {n}")
        lines += gap_lines(part, methods, "reference")
        lines.append("")

    large = frame[frame["kind"] == "multistart_reference"]

    for n in LARGE_SIZES:
        part = large[large["n"] == n]
        lines.append(f"MULTI-START REFERENCE (not exact), n = {n}")
        lines += gap_lines(part, [m for m in methods if m != "nn_all_local_search"], "reference")
        lines.append("")

    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print(f"Runtime: {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
