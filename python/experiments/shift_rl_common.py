import hashlib
import random

from experiments.shift_common import (  # noqa: F401
    NAME,
    OBJECTIVE,
    RESULTS_DIR,
    WAIT,
    load_sampler,
)

REFERENCES_PATH = RESULTS_DIR / f"shift_rl{NAME}_references.csv"

# (shift size, number of shifts) per fixed evaluation set.
FIXED_SETS = {
    "validation": {
        "seed": 101,
        "plan": [(8, 12), (12, 12), (14, 12), (20, 6), (30, 6)],
    },
    "test": {
        "seed": 202,
        "plan": [(10, 15), (14, 15), (20, 6), (30, 6), (50, 4)],
    },
    # A second, untouched test set drawn from the same test days with a new
    # seed, used to compare every variant once the first one was spent.
    "test2": {
        "seed": 303,
        "split": "test",
        "plan": [(10, 15), (14, 15), (20, 6), (30, 6), (50, 4)],
    },
    # A third set for the waiting-cost objective: the first two were already
    # used to compare variants of the original objective.
    "test3": {
        "seed": 404,
        "split": "test",
        "plan": [(10, 15), (14, 15), (20, 6), (30, 6), (50, 4)],
    },
}

TRAIN_SIZES = (8, 10, 12, 14, 16, 20, 24, 30, 40)

# Largest size solved exactly; above it the reference is the best of local
# search from several starts.
MAX_EXACT = 14


def fingerprint(env) -> str:
    text = ",".join(order.order_id for order in env.orders)

    return hashlib.sha1(text.encode()).hexdigest()[:12]


def fixed_instances(sampler, splits, name):
    """
    The same shifts on every call: (set index, size, environment).
    """

    spec = FIXED_SETS[name]
    rng = random.Random(spec["seed"])

    shifts = []

    for n, count in spec["plan"]:
        for _ in range(count):
            shifts.append(
                (n, sampler.sample(rng, splits[spec.get("split", name)], n))
            )

    return shifts
