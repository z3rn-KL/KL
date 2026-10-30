import multiprocessing
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

from experiments.shift_common import RESULTS_DIR, load_sampler  # noqa: E402
from rl.shift.encoder import ShiftEncoder  # noqa: E402
from rl.shift.environment import score_sequence  # noqa: E402
from rl.shift.heuristics import local_search, nearest_plan  # noqa: E402

DEMOS_PATH = RESULTS_DIR / "shift_demonstrations.npz"

SIZES = (8, 10, 12, 14, 16, 20, 24, 30)
SHIFTS_PER_SIZE = 300
WORKERS = 6

_SAMPLER = None
_DAYS = None


def expert_plan(env):
    """
    The better of nearest neighbour plus local search with and without the
    option to defer orders.
    """

    serve_all = local_search(env, nearest_plan(env, stop_when_not_worth=False))
    stopping = local_search(env, nearest_plan(env, stop_when_not_worth=True))

    if score_sequence(env, serve_all) >= score_sequence(env, stopping):
        return serve_all

    return stopping


def demonstrate(task):
    seed, n = task

    env = _SAMPLER.sample(random.Random(seed), _DAYS, n)
    plan = expert_plan(env)

    steps = list(plan) + ([env.stop_action] if len(plan) < n else [])

    encoder = ShiftEncoder()
    states = []

    env.reset()

    for action in steps:
        if env.done:
            break

        actions, features = encoder.encode(env)
        states.append(
            (
                features,
                int(np.flatnonzero(actions == action)[0]),
                encoder.pair_times(env, actions),
            )
        )
        env.step(int(action))

    return states


def main() -> None:
    global _SAMPLER, _DAYS

    started = perf_counter()

    _SAMPLER, splits = load_sampler()
    _DAYS = splits["train"]

    tasks = [
        (10_000 + 1_000 * index + k, n)
        for index, n in enumerate(SIZES)
        for k in range(SHIFTS_PER_SIZE)
    ]

    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        results = pool.map(demonstrate, tasks, chunksize=8)

    states = [state for shift in results for state in shift]

    np.savez_compressed(
        DEMOS_PATH,
        features=np.concatenate([features for features, _, _ in states]),
        lengths=np.array([len(features) for features, _, _ in states]),
        labels=np.array([label for _, label, _ in states]),
        pairs=np.concatenate([pairs.reshape(-1) for _, _, pairs in states]),
    )

    print(
        f"{len(tasks)} shifts, {len(states)} demonstrated states, "
        f"saved {DEMOS_PATH} in {perf_counter() - started:.0f}s"
    )


if __name__ == "__main__":
    main()
