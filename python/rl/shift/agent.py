from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from ..proposed.agent import CandidateQNetwork
from .encoder import FEATURE_DIMENSION


@dataclass(frozen=True)
class ShiftAgentConfig:
    hidden_dimension: int = 128
    learning_rate: float = 5e-4
    gamma: float = 1.0
    batch_size: int = 128
    replay_capacity: int = 100_000
    min_replay_size: int = 5_000
    tau: float = 0.005
    gradient_clip_norm: float = 5.0
    seed: int = 42
    # Q = exact immediate reward + learned future value (stop row: exact).
    residual: bool = False

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")

        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")

        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1].")

        if self.min_replay_size < self.batch_size:
            raise ValueError("min_replay_size must be >= batch_size.")

        if self.replay_capacity < self.min_replay_size:
            raise ValueError("replay_capacity must be >= min_replay_size.")


class ShiftAgent:
    """
    Double DQN that scores one candidate row at a time, so the same network
    handles a shift of any size. Training batches pad the next-state
    candidate rows and mask the padding.

    Learns Q(s, a) = reward now + best reward still to come; the episode is
    finite, so gamma = 1 makes the return equal to the shift score.
    """

    def __init__(
        self,
        config: ShiftAgentConfig | None = None,
        feature_dimension: int = FEATURE_DIMENSION,
    ) -> None:
        self.config = config if config is not None else ShiftAgentConfig()
        self.feature_dimension = feature_dimension

        torch.manual_seed(self.config.seed)
        self._rng = np.random.default_rng(self.config.seed)

        self.online = CandidateQNetwork(
            input_dimension=feature_dimension,
            hidden_dimension=self.config.hidden_dimension,
        )
        self.target = CandidateQNetwork(
            input_dimension=feature_dimension,
            hidden_dimension=self.config.hidden_dimension,
        )
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = Adam(
            self.online.parameters(),
            lr=self.config.learning_rate,
        )
        self.loss_function = nn.SmoothL1Loss()

        capacity = self.config.replay_capacity

        self._chosen = np.zeros((capacity, feature_dimension), np.float32)
        self._chosen_base = np.zeros(capacity, np.float32)
        self._reward = np.zeros(capacity, np.float32)
        self._done = np.zeros(capacity, bool)
        self._next: list[np.ndarray | None] = [None] * capacity
        self._next_base: list[np.ndarray | None] = [None] * capacity
        self._size = 0
        self._cursor = 0

        self.gradient_steps = 0

    def _q(
        self,
        network: nn.Module,
        features: torch.Tensor,
        base: torch.Tensor | None,
    ) -> torch.Tensor:
        value = network(features)

        if not self.config.residual:
            return value

        # Feature 0 is the stop flag: ending the shift has no future value.
        return base + value * (1.0 - features[..., 0])

    def scores(
        self,
        features: np.ndarray,
        base: np.ndarray | None = None,
    ) -> np.ndarray:
        with torch.no_grad():
            return self._q(
                self.online,
                torch.from_numpy(features),
                None if base is None else torch.from_numpy(base),
            ).numpy()

    def greedy(
        self,
        features: np.ndarray,
        base: np.ndarray | None = None,
    ) -> int:
        return int(np.argmax(self.scores(features, base)))

    def remember(
        self,
        chosen: np.ndarray,
        reward: float,
        next_features: np.ndarray | None,
        done: bool,
        chosen_base: float = 0.0,
        next_base: np.ndarray | None = None,
    ) -> None:
        if done and next_features is not None:
            raise ValueError("A terminal transition has no next candidates.")

        if not done and (next_features is None or len(next_features) == 0):
            raise ValueError("A non-terminal transition needs next candidates.")

        i = self._cursor

        self._chosen[i] = chosen
        self._chosen_base[i] = chosen_base
        self._reward[i] = reward
        self._done[i] = done
        self._next[i] = None if done else next_features
        self._next_base[i] = None if done else next_base

        self._cursor = (i + 1) % self.config.replay_capacity
        self._size = min(self._size + 1, self.config.replay_capacity)

    def can_train(self) -> bool:
        return self._size >= self.config.min_replay_size

    def train_step(self) -> float | None:
        if not self.can_train():
            return None

        cfg = self.config
        batch = self._rng.integers(0, self._size, cfg.batch_size)

        chosen = torch.from_numpy(self._chosen[batch])
        chosen_base = torch.from_numpy(self._chosen_base[batch])
        reward = torch.from_numpy(self._reward[batch])
        done = torch.from_numpy(self._done[batch])

        nexts = [self._next[i] for i in batch]
        width = max((len(x) for x in nexts if x is not None), default=1)

        padded = np.zeros(
            (cfg.batch_size, width, self.feature_dimension),
            np.float32,
        )
        padded_base = np.zeros((cfg.batch_size, width), np.float32)
        valid = np.zeros((cfg.batch_size, width), bool)

        for row, (x, i) in enumerate(zip(nexts, batch)):
            if x is not None:
                padded[row, : len(x)] = x
                valid[row, : len(x)] = True

                if cfg.residual:
                    padded_base[row, : len(x)] = self._next_base[i]

        padded_t = torch.from_numpy(padded)
        padded_base_t = torch.from_numpy(padded_base)
        valid_t = torch.from_numpy(valid)

        with torch.no_grad():
            online_next = self._q(self.online, padded_t, padded_base_t)
            online_next = online_next.masked_fill(~valid_t, float("-inf"))
            best = online_next.argmax(dim=1, keepdim=True)

            target_next = self._q(self.target, padded_t, padded_base_t)
            next_value = target_next.gather(1, best).squeeze(1)
            next_value = torch.where(
                done,
                torch.zeros_like(next_value),
                next_value,
            )

            target = reward + cfg.gamma * next_value

        loss = self.loss_function(
            self._q(self.online, chosen, chosen_base),
            target,
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online.parameters(),
            cfg.gradient_clip_norm,
        )
        self.optimizer.step()

        with torch.no_grad():
            for target_p, online_p in zip(
                self.target.parameters(),
                self.online.parameters(),
            ):
                target_p.lerp_(online_p, cfg.tau)

        self.gradient_steps += 1

        return float(loss.item())
