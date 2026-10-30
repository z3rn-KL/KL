import csv
import random
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

from experiments.shift_rl_common import (  # noqa: E402
    FIXED_SETS,
    MAX_EXACT,
    REFERENCES_PATH,
    fingerprint,
    fixed_instances,
    load_sampler,
)
from rl.shift.environment import score_sequence  # noqa: E402
from rl.shift.exact_solver import exact_best_shift  # noqa: E402
from rl.shift.heuristics import (  # noqa: E402
    local_search,
    multistart_search,
    nearest_plan,
)

SEED = 42
STARTS = 2

FIELDS = [
    "set",
    "index",
    "n",
    "fingerprint",
    "reference",
    "reference_kind",
    "nn_serve_all",
    "nn_stop_rule",
    "nn_all_ls",
    "nn_stop_ls",
    "ls_seconds",
]


def main() -> None:
    started = perf_counter()

    sampler, splits = load_sampler()

    # python shift_rl_references.py [set ...]; default: every set.
    names = sys.argv[1:] or list(FIXED_SETS)

    kept = []

    if REFERENCES_PATH.exists():
        with REFERENCES_PATH.open(encoding="utf-8") as handle:
            kept = [r for r in csv.DictReader(handle) if r["set"] not in names]

    rows = []

    for name in names:
        for index, (n, env) in enumerate(fixed_instances(sampler, splits, name)):
            if n <= MAX_EXACT:
                reference, _ = exact_best_shift(env)
                kind = "exact"
            else:
                plan = multistart_search(env, random.Random(SEED + index), STARTS)
                reference = score_sequence(env, plan)
                kind = "best_known"

            nn_all = nearest_plan(env, stop_when_not_worth=False)
            nn_stop = nearest_plan(env, stop_when_not_worth=True)

            tic = perf_counter()
            stop_ls = local_search(env, nn_stop)
            seconds = perf_counter() - tic

            row = {
                "set": name,
                "index": index,
                "n": n,
                "fingerprint": fingerprint(env),
                "reference": reference,
                "reference_kind": kind,
                "nn_serve_all": score_sequence(env, nn_all),
                "nn_stop_rule": score_sequence(env, nn_stop),
                "nn_all_ls": score_sequence(env, local_search(env, nn_all)),
                "nn_stop_ls": score_sequence(env, stop_ls),
                "ls_seconds": seconds,
            }

            # A best-known reference must never fall below a baseline.
            row["reference"] = max(
                row["reference"], row["nn_all_ls"], row["nn_stop_ls"]
            )

            rows.append(row)

            print(
                f"{name} #{index:02d} n={n:2d} {kind:10s} "
                f"reference={row['reference']:.0f} ({perf_counter() - started:.0f}s)",
                flush=True,
            )

    with REFERENCES_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept + rows)

    print(f"Saved {REFERENCES_PATH} in {perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
