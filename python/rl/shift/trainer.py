from __future__ import annotations

import numpy as np

from .agent import ShiftAgent
from .encoder import REWARD_SCALE, ShiftEncoder
from .environment import ShiftEnvironment


def run_episode(
    env: ShiftEnvironment,
    encoder: ShiftEncoder,
    agent: ShiftAgent,
    *,
    rng: np.random.Generator | None = None,
    epsilon: float = 0.0,
    stop_probability: float = 0.02,
    learn: bool = False,
) -> dict:
    """
    One shift. With probability epsilon a step is random: an unserved order
    (or, rarely, stopping the shift). Otherwise the highest-scoring action.

    With learn=True every transition goes to the replay buffer and the agent
    takes one gradient step per environment step.
    """

    env.reset()

    while not env.done:
        actions, features, base = encoder.encode_with_base(env)
        stop_position = len(actions) - 1

        if rng is not None and rng.random() < epsilon:
            if rng.random() < stop_probability:
                pick = stop_position
            else:
                pick = int(rng.integers(0, stop_position))
        else:
            pick = agent.greedy(features, base)

        step = env.step(int(actions[pick]))

        if learn:
            next_features = next_base = None

            if not step.done:
                _, next_features, next_base = encoder.encode_with_base(env)

            agent.remember(
                features[pick],
                step.reward * REWARD_SCALE,
                next_features,
                step.done,
                chosen_base=base[pick],
                next_base=next_base,
            )
            agent.train_step()

    return env.summary()
