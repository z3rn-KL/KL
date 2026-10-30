from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .encoder import FEATURE_DIMENSION, REWARD_SCALE, ShiftEncoder
from .environment import ShiftEnvironment


def score_rows(
    network: nn.Module,
    rows: torch.Tensor,
    valid: torch.Tensor,
    pairs: torch.Tensor | None,
) -> torch.Tensor:
    """
    Scores [B, W] for padded candidate rows [B, W, F]; padding is -inf.
    """

    if getattr(network, "uses_sets", False):
        scores = network(rows, valid, pairs)
    else:
        batch, width, dimension = rows.shape
        scores = network(rows.reshape(-1, dimension)).reshape(batch, width)

    return scores.masked_fill(~valid, float("-inf"))


def pad_batch(
    items: list[tuple[np.ndarray, np.ndarray | None]],
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.zeros((len(items), width, FEATURE_DIMENSION), np.float32)
    valid = np.zeros((len(items), width), bool)
    pairs = np.zeros((len(items), width, width), np.float32)

    for r, (features, pair) in enumerate(items):
        size = len(features)

        rows[r, :size] = features
        valid[r, :size] = True

        if pair is not None:
            pairs[r, :size, :size] = pair

    return rows, valid, pairs


def clone_environment(env: ShiftEnvironment) -> ShiftEnvironment:
    return ShiftEnvironment(
        distance_km=env.distance_km,
        travel_minutes=env.travel_minutes,
        orders=env.orders,
        config=env.config,
    )


def run_batch(
    envs: list[ShiftEnvironment],
    encoder: ShiftEncoder,
    network: nn.Module,
    *,
    greedy: bool,
    generator: torch.Generator | None = None,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None], list[float]]:
    """
    Plays every shift to the end, one step at a time across all of them.

    The policy is a softmax over the scores of the candidate rows. Returns,
    per shift, the summed log-probability and summed entropy of the actions
    taken (with gradients; None when greedy) and the scaled shift score.
    """

    for env in envs:
        env.reset()

    log_probs: list[list[torch.Tensor]] = [[] for _ in envs]
    entropies: list[list[torch.Tensor]] = [[] for _ in envs]
    active = list(range(len(envs)))

    while active:
        encoded = [encoder.encode(envs[i]) for i in active]
        width = max(len(actions) for actions, _ in encoded)

        sets = getattr(network, "uses_sets", False)

        rows, valid, pairs = pad_batch(
            [
                (
                    features,
                    encoder.pair_times(envs[i], actions) if sets else None,
                )
                for i, (actions, features) in zip(active, encoded)
            ],
            width,
        )

        valid_t = torch.from_numpy(valid)

        scores = score_rows(
            network,
            torch.from_numpy(rows),
            valid_t,
            torch.from_numpy(pairs),
        )

        if greedy:
            pick = scores.argmax(dim=1)
        else:
            log_p = torch.log_softmax(scores, dim=1)
            pick = torch.multinomial(
                log_p.exp().detach(), 1, generator=generator
            ).squeeze(1)

            chosen = log_p.gather(1, pick[:, None]).squeeze(1)
            entropy = -(log_p.exp() * log_p.masked_fill(~valid_t, 0.0)).sum(dim=1)

        still = []

        for r, i in enumerate(active):
            if not greedy:
                log_probs[i].append(chosen[r])
                entropies[i].append(entropy[r])

            envs[i].step(int(encoded[r][0][int(pick[r])]))

            if not envs[i].done:
                still.append(i)

        active = still

    returns = [env.total_reward * REWARD_SCALE for env in envs]

    if greedy:
        return [None] * len(envs), [None] * len(envs), returns

    return (
        [torch.stack(items).sum() for items in log_probs],
        [torch.stack(items).sum() for items in entropies],
        returns,
    )


def policy_gradient_step(
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    encoder: ShiftEncoder,
    instances: list[ShiftEnvironment],
    *,
    samples: int,
    entropy_weight: float,
    clip: float,
    generator: torch.Generator | None = None,
) -> dict:
    """
    REINFORCE. Each shift is played `samples` times; the baseline of a run is
    the mean of the other runs on the same shift, so shifts of different size
    and difficulty do not mix.
    """

    envs = [clone_environment(env) for env in instances for _ in range(samples)]

    log_probs, entropies, returns = run_batch(
        envs, encoder, network, greedy=False, generator=generator
    )

    returns_t = torch.tensor(returns).reshape(len(instances), samples)
    baseline = (returns_t.sum(dim=1, keepdim=True) - returns_t) / (samples - 1)
    advantage = (returns_t - baseline).reshape(-1)

    advantage = advantage / (advantage.std() + 1e-6)

    log_prob_t = torch.stack(log_probs)
    entropy_t = torch.stack(entropies)

    loss = -(advantage.detach() * log_prob_t).mean() - entropy_weight * (
        entropy_t.mean()
    )

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), clip)
    optimizer.step()

    return {
        "mean_return": float(returns_t.mean()),
        "mean_entropy": float(entropy_t.mean()),
        "loss": float(loss.item()),
    }


def behaviour_clone(
    network: nn.Module,
    demos: list[tuple],
    *,
    epochs: int,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> float:
    """
    Supervised pre-training: softmax cross-entropy of the demonstrated
    action over the candidate rows. Returns the final mean loss.
    """

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    loss_value = float("nan")

    for _ in range(epochs):
        order = rng.permutation(len(demos))
        total = 0.0

        for start in range(0, len(order), batch_size):
            items = [demos[i] for i in order[start : start + batch_size]]
            width = max(len(item[0]) for item in items)

            x, mask, pairs = pad_batch(
                [(item[0], item[2] if len(item) > 2 else None) for item in items],
                width,
            )
            y = np.array([item[1] for item in items], np.int64)

            logits = score_rows(
                network,
                torch.from_numpy(x),
                torch.from_numpy(mask),
                torch.from_numpy(pairs),
            )

            loss = nn.functional.cross_entropy(logits, torch.from_numpy(y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item()) * len(items)

        loss_value = total / len(demos)

    return loss_value
