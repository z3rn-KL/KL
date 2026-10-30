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

from experiments.shift_rl_common import (  # noqa: E402
    REFERENCES_PATH,
    RESULTS_DIR,
    TRAIN_SIZES,
    fingerprint,
    fixed_instances,
    load_sampler,
)
from rl.shift.agent import ShiftAgent, ShiftAgentConfig  # noqa: E402
from rl.shift.encoder import FEATURE_NAMES, ShiftEncoder  # noqa: E402
from rl.shift.trainer import run_episode  # noqa: E402


EPSILON_START = 1.0
EPSILON_END = 0.05
# Epsilon reaches its floor after this share of the episodes (after warm-up).
EPSILON_SHARE = 0.60

# Fully random episodes first, so the replay buffer fills with complete
# shifts before the untrained network gets to choose.
WARMUP_EPISODES = 1_000


def name_part(tag: str) -> str:
    return f"_{tag}" if tag else ""


def checkpoint_path(seed: int, tag: str = "") -> Path:
    return RESULTS_DIR / f"shift_rl{name_part(tag)}_seed{seed}_best.pt"


def history_path(seed: int, tag: str = "") -> Path:
    return RESULTS_DIR / f"shift_rl{name_part(tag)}_seed{seed}_history.csv"


def validate(agent, encoder, instances, references) -> dict:
    """
    Greedy play on the fixed validation shifts; gap = reference - score as a
    percentage of |reference| (positive: worse than the reference).
    """

    gaps, exact_gaps, large_gaps, deferred, late = [], [], [], [], []

    for (n, env), (_, row) in zip(instances, references.iterrows()):
        summary = run_episode(env, encoder, agent)

        gap = (row["reference"] - summary["score"]) / abs(row["reference"]) * 100.0

        gaps.append(gap)
        (exact_gaps if row["reference_kind"] == "exact" else large_gaps).append(gap)
        deferred.append(summary["deferred"])
        late.append(summary["late"])

    return {
        "validation_gap_pct": float(np.mean(gaps)),
        "validation_gap_exact_pct": float(np.mean(exact_gaps)),
        "validation_gap_large_pct": float(np.mean(large_gaps)),
        "validation_deferred": float(np.mean(deferred)),
        "validation_late": float(np.mean(late)),
    }


def main() -> None:
    seed = int(sys.argv[1])
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000
    evaluate_every = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000
    # "" = plain Q network; "resid" = exact immediate reward + learned future.
    tag = sys.argv[4] if len(sys.argv) > 4 else ""

    torch.set_num_threads(1)

    sampler, splits = load_sampler()

    references = (
        pd.read_csv(REFERENCES_PATH)
        .query("set == 'validation'")
        .sort_values("index")
        .reset_index(drop=True)
    )

    instances = fixed_instances(sampler, splits, "validation")

    assert len(instances) == len(references)
    assert all(
        fingerprint(env) == row["fingerprint"]
        for (_, env), (_, row) in zip(instances, references.iterrows())
    ), "Validation shifts changed since the references were computed."

    encoder = ShiftEncoder()
    config = ShiftAgentConfig(seed=seed, residual=tag == "resid")
    agent = ShiftAgent(config)

    rng = np.random.default_rng(seed)
    draw = random.Random(seed)
    train_days = splits["train"]

    decay_episodes = EPSILON_SHARE * episodes
    started = perf_counter()

    history = []
    best_gap = float("inf")
    recent_scores = []

    print(
        f"seed {seed} [{tag or 'plain'}]: {episodes} episodes, validation every "
        f"{evaluate_every}, {len(instances)} validation shifts",
        flush=True,
    )

    for episode in range(1, episodes + 1):
        progress = max(0, episode - WARMUP_EPISODES) / decay_episodes

        epsilon = max(
            EPSILON_END,
            EPSILON_START * (EPSILON_END / EPSILON_START) ** progress,
        )

        env = sampler.sample(draw, train_days, draw.choice(TRAIN_SIZES))

        summary = run_episode(
            env,
            encoder,
            agent,
            rng=rng,
            epsilon=epsilon,
            learn=True,
        )

        recent_scores.append(summary["score"])

        if episode % evaluate_every:
            continue

        row = {
            "episode": episode,
            "epsilon": epsilon,
            "gradient_steps": agent.gradient_steps,
            "mean_training_score": float(np.mean(recent_scores)),
            "seconds": perf_counter() - started,
            **validate(agent, encoder, instances, references),
        }

        recent_scores.clear()

        improved = row["validation_gap_pct"] < best_gap
        row["is_best"] = improved

        if improved:
            best_gap = row["validation_gap_pct"]

            torch.save(
                {
                    "seed": seed,
                    "episode": episode,
                    "feature_names": FEATURE_NAMES,
                    "agent_config": config.__dict__,
                    "online_state_dict": agent.online.state_dict(),
                    "validation_gap_pct": best_gap,
                },
                checkpoint_path(seed, tag),
            )

        history.append(row)
        pd.DataFrame(history).to_csv(history_path(seed, tag), index=False)

        print(
            f"seed {seed} ep {episode:6d} eps {epsilon:.3f} "
            f"train {row['mean_training_score']:10.0f} "
            f"val gap {row['validation_gap_pct']:7.2f}% "
            f"(exact {row['validation_gap_exact_pct']:6.2f}%, "
            f"large {row['validation_gap_large_pct']:7.2f}%) "
            f"{'*' if improved else ' '} {row['seconds']:.0f}s",
            flush=True,
        )

    print(f"seed {seed} finished, best validation gap {best_gap:.2f}%", flush=True)


if __name__ == "__main__":
    main()
