import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

from experiments.shift_make_demos import DEMOS_PATH  # noqa: E402
from experiments.shift_rl_common import (  # noqa: E402
    REFERENCES_PATH,
    RESULTS_DIR,
    TRAIN_SIZES,
    fingerprint,
    fixed_instances,
    load_sampler,
)
from experiments.shift_rl_train import checkpoint_path, history_path  # noqa: E402
from rl.proposed.agent import CandidateQNetwork  # noqa: E402
from rl.shift.agent import ShiftAgentConfig  # noqa: E402
from rl.shift.encoder import (  # noqa: E402
    FEATURE_DIMENSION,
    FEATURE_NAMES,
    ShiftEncoder,
)
from rl.shift.set_network import SetScorer  # noqa: E402
from rl.shift.policy import (  # noqa: E402
    behaviour_clone,
    clone_environment,
    policy_gradient_step,
    run_batch,
)

INSTANCES_PER_UPDATE = 32
SAMPLES_PER_INSTANCE = 8
LEARNING_RATE = 3e-4
ENTROPY_WEIGHT = 0.005
GRADIENT_CLIP = 1.0
CLONE_EPOCHS = 80


# tag -> (network, behaviour cloning first)
TAGS = {
    "pg": ("mlp", False),
    "pgbc": ("mlp", True),
    "set": ("set", False),
    "setbc": ("set", True),
}


def load_demos():
    data = np.load(DEMOS_PATH)
    features, lengths, labels = data["features"], data["lengths"], data["labels"]
    pairs = data["pairs"]

    edges = np.concatenate([[0], np.cumsum(lengths)])
    pair_edges = np.concatenate([[0], np.cumsum(lengths.astype(np.int64) ** 2)])

    return [
        (
            features[edges[i] : edges[i + 1]],
            int(labels[i]),
            pairs[pair_edges[i] : pair_edges[i + 1]].reshape(
                lengths[i], lengths[i]
            ),
        )
        for i in range(len(lengths))
    ]


def validate(network, encoder, instances, references) -> dict:
    envs = [clone_environment(env) for _, env in instances]
    run_batch(envs, encoder, network, greedy=True)

    gaps, exact_gaps, large_gaps, deferred, late = [], [], [], [], []

    for env, (_, row) in zip(envs, references.iterrows()):
        gap = (row["reference"] - env.total_reward) / abs(row["reference"]) * 100.0

        gaps.append(gap)
        (exact_gaps if row["reference_kind"] == "exact" else large_gaps).append(gap)
        deferred.append(env.deferred_count)
        late.append(env.late)

    return {
        "validation_gap_pct": float(np.mean(gaps)),
        "validation_gap_exact_pct": float(np.mean(exact_gaps)),
        "validation_gap_large_pct": float(np.mean(large_gaps)),
        "validation_deferred": float(np.mean(deferred)),
        "validation_late": float(np.mean(late)),
    }


def main() -> None:
    seed = int(sys.argv[1])
    updates = int(sys.argv[2])
    tag = sys.argv[3]  # see TAGS
    evaluate_every = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    torch.set_num_threads(1)
    torch.manual_seed(seed)

    sampler, splits = load_sampler()

    references = (
        pd.read_csv(REFERENCES_PATH)
        .query("set == 'validation'")
        .sort_values("index")
        .reset_index(drop=True)
    )
    instances = fixed_instances(sampler, splits, "validation")

    assert all(
        fingerprint(env) == row["fingerprint"]
        for (_, env), (_, row) in zip(instances, references.iterrows())
    ), "Validation shifts changed since the references were computed."

    encoder = ShiftEncoder()
    kind, clone_first = TAGS[tag]

    if kind == "set":
        network = SetScorer(FEATURE_DIMENSION, 128)
    else:
        network = CandidateQNetwork(
            input_dimension=FEATURE_DIMENSION,
            hidden_dimension=128,
        )

    if clone_first:
        loss = behaviour_clone(
            network,
            load_demos(),
            epochs=CLONE_EPOCHS,
            seed=seed,
        )
        print(f"seed {seed} [{tag}] behaviour cloning loss {loss:.3f}", flush=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)
    draw = random.Random(seed)

    history = []
    best_gap = float("inf")
    started = perf_counter()
    recent = []

    config = ShiftAgentConfig(seed=seed).__dict__

    def evaluate(update: int) -> None:
        nonlocal best_gap

        row = {
            "episode": update,
            "mean_training_score": float(np.mean(recent)) if recent else float("nan"),
            "seconds": perf_counter() - started,
            **validate(network, encoder, instances, references),
        }

        improved = row["validation_gap_pct"] < best_gap
        row["is_best"] = improved

        if improved:
            best_gap = row["validation_gap_pct"]

            torch.save(
                {
                    "seed": seed,
                    "episode": update,
                    "feature_names": FEATURE_NAMES,
                    "agent_config": config,
                    "network": kind,
                    "online_state_dict": network.state_dict(),
                    "validation_gap_pct": best_gap,
                },
                checkpoint_path(seed, tag),
            )

        history.append(row)
        pd.DataFrame(history).to_csv(history_path(seed, tag), index=False)
        recent.clear()

        print(
            f"seed {seed} [{tag}] update {update:5d} "
            f"val gap {row['validation_gap_pct']:7.2f}% "
            f"(exact {row['validation_gap_exact_pct']:6.2f}%, "
            f"large {row['validation_gap_large_pct']:7.2f}%) "
            f"{'*' if improved else ' '} {row['seconds']:.0f}s",
            flush=True,
        )

    evaluate(0)

    for update in range(1, updates + 1):
        batch = [
            sampler.sample(draw, splits["train"], draw.choice(TRAIN_SIZES))
            for _ in range(INSTANCES_PER_UPDATE)
        ]

        stats = policy_gradient_step(
            network,
            optimizer,
            encoder,
            batch,
            samples=SAMPLES_PER_INSTANCE,
            entropy_weight=ENTROPY_WEIGHT,
            clip=GRADIENT_CLIP,
            generator=generator,
        )

        recent.append(stats["mean_return"] / 1.0)

        if update % evaluate_every == 0:
            evaluate(update)

    print(f"seed {seed} [{tag}] finished, best validation gap {best_gap:.2f}%", flush=True)


if __name__ == "__main__":
    main()
