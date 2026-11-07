from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Sequence

import torch
from torch import nn
from torch.optim import Adam

from .candidate_encoder import (
    EncodedCandidateSet,
    FEATURE_NAMES,
)


@dataclass(frozen=True)
class ProposedAgentConfig:
    """
    Cấu hình Candidate-scoring Q-agent.

    gamma = 1.0:
        phù hợp với objective finite-horizon:

            total_reward = -J(pi)

        khi reward_t = -Delta J_t.

    epsilon:
        dùng epsilon-greedy để exploration.

    target_update_interval:
        số gradient step giữa hai lần đồng bộ
        target network.
    """

    learning_rate: float = 1e-3

    gamma: float = 1.0

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.998

    replay_capacity: int = 10_000
    batch_size: int = 64
    min_replay_size: int = 128

    target_update_interval: int = 100

    hidden_dimension: int = 64

    gradient_clip_norm: float = 5.0

    seed: int = 42

    device: str = "auto"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be > 0."
            )

        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(
                "gamma must be in [0, 1]."
            )

        if not 0.0 <= self.epsilon_min <= 1.0:
            raise ValueError(
                "epsilon_min must be in [0, 1]."
            )

        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon_start must be in [0, 1]."
            )

        if self.epsilon_min > self.epsilon_start:
            raise ValueError(
                "epsilon_min cannot exceed "
                "epsilon_start."
            )

        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError(
                "epsilon_decay must be in (0, 1]."
            )

        if self.replay_capacity <= 0:
            raise ValueError(
                "replay_capacity must be > 0."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0."
            )

        if self.min_replay_size <= 0:
            raise ValueError(
                "min_replay_size must be > 0."
            )

        if self.min_replay_size > self.replay_capacity:
            raise ValueError(
                "min_replay_size cannot exceed "
                "replay_capacity."
            )

        if self.target_update_interval <= 0:
            raise ValueError(
                "target_update_interval must be > 0."
            )

        if self.hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be > 0."
            )

        if self.gradient_clip_norm <= 0.0:
            raise ValueError(
                "gradient_clip_norm must be > 0."
            )


@dataclass(frozen=True)
class ReplayTransition:
    """
    Một transition dùng cho TD learning.

    chosen_features:
        feature vector tương ứng với action đã chọn.

    next_candidate_features:
        toàn bộ feature vector của các action hợp lệ
        tại state kế tiếp.

        Nếu done=True thì tuple này có thể rỗng.
    """

    chosen_features: tuple[float, ...]
    reward: float

    next_candidate_features: tuple[
        tuple[float, ...],
        ...
    ]

    done: bool


class ReplayBuffer:
    """
    Replay buffer đơn giản, có RNG riêng
    để reproducibility tốt hơn.
    """

    def __init__(
        self,
        *,
        capacity: int,
        seed: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError(
                "capacity must be > 0."
            )

        self._buffer: deque[ReplayTransition] = deque(
            maxlen=capacity
        )

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def append(
        self,
        transition: ReplayTransition,
    ) -> None:
        self._buffer.append(transition)

    def sample(
        self,
        batch_size: int,
    ) -> list[ReplayTransition]:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0."
            )

        if batch_size > len(self._buffer):
            raise ValueError(
                "Cannot sample more transitions "
                "than currently stored."
            )

        return self._rng.sample(
            list(self._buffer),
            batch_size,
        )


class CandidateQNetwork(nn.Module):
    """
    Neural network đánh giá MỘT candidate tại một state.

    Input:
        candidate feature vector có dimension = 8.

    Output:
        một scalar Q-value.

    Ta không tạo N output cố định như DQN routing cũ.

    Thay vào đó:

        x_(t,i)
            ↓
        Q-network
            ↓
        Q(s_t, a_i)

    rồi áp dụng cùng network cho tất cả candidate.
    """

    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
    ) -> None:
        super().__init__()

        if input_dimension <= 0:
            raise ValueError(
                "input_dimension must be > 0."
            )

        if hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be > 0."
            )

        self.network = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dimension,
                hidden_dimension,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dimension,
                1,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input shape:

            [batch, feature_dimension]

        Output shape:

            [batch]
        """

        return self.network(
            features
        ).squeeze(-1)


@dataclass(frozen=True)
class TrainingStepResult:
    loss: float
    mean_q_value: float
    mean_target_value: float
    replay_size: int
    gradient_step: int


class ProposedRoutingAgent:
    """
    Candidate-scoring Q-learning agent.

    Agent học:

        Q_theta(s_t, a_i)

    bằng neural network dùng feature vector của
    candidate delivery.

    Learning target:

        y_t = r_t

    nếu terminal, hoặc:

        y_t =
            r_t
            + gamma *
              max_a' Q_target(s_{t+1}, a')

    nếu episode chưa kết thúc.
    """

    def __init__(
        self,
        *,
        config: ProposedAgentConfig | None = None,
        feature_dimension: int | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ProposedAgentConfig()
        )

        self.feature_dimension = (
            feature_dimension
            if feature_dimension is not None
            else len(FEATURE_NAMES)
        )

        self.device = self._resolve_device(
            self.config.device
        )

        self._seed_everything(
            self.config.seed
        )

        self.online_network = CandidateQNetwork(
            input_dimension=self.feature_dimension,
            hidden_dimension=(
                self.config.hidden_dimension
            ),
        ).to(self.device)

        self.target_network = CandidateQNetwork(
            input_dimension=self.feature_dimension,
            hidden_dimension=(
                self.config.hidden_dimension
            ),
        ).to(self.device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_capacity,
            seed=self.config.seed,
        )

        self.epsilon = (
            self.config.epsilon_start
        )

        self.gradient_steps = 0

        self._action_rng = random.Random(
            self.config.seed
        )

    @staticmethod
    def _resolve_device(
        requested_device: str,
    ) -> torch.device:
        if requested_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            return torch.device("cpu")

        return torch.device(
            requested_device
        )

    @staticmethod
    def _seed_everything(
        seed: int,
    ) -> None:
        random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _features_tensor(
        self,
        feature_matrix: Sequence[
            Sequence[float]
        ],
    ) -> torch.Tensor:
        return torch.tensor(
            feature_matrix,
            dtype=torch.float32,
            device=self.device,
        )

    def select_action(
        self,
        encoded_candidates: EncodedCandidateSet,
        *,
        explore: bool = True,
    ) -> int:
        """
        Chọn delivery action từ candidate set.

        Exploration:
            random trong các action HỢP LỆ.

        Exploitation:
            argmax Q_theta(s, a_i).
        """

        if not encoded_candidates.candidates:
            raise ValueError(
                "Cannot select action from "
                "an empty candidate set."
            )

        should_explore = (
            explore
            and self._action_rng.random()
            < self.epsilon
        )

        if should_explore:
            candidate = self._action_rng.choice(
                encoded_candidates.candidates
            )

            return candidate.action

        feature_matrix = (
            encoded_candidates.feature_matrix
        )

        with torch.no_grad():
            q_values = self.online_network(
                self._features_tensor(
                    feature_matrix
                )
            )

        best_index = int(
            torch.argmax(q_values).item()
        )

        return encoded_candidates.candidates[
            best_index
        ].action

    def remember(
        self,
        *,
        chosen_features: Sequence[float],
        reward: float,
        next_candidate_features: Sequence[
            Sequence[float]
        ],
        done: bool,
    ) -> None:
        if len(chosen_features) != self.feature_dimension:
            raise ValueError(
                "chosen_features has incorrect "
                "dimension."
            )

        normalized_next_features = tuple(
            tuple(
                float(value)
                for value in feature_vector
            )
            for feature_vector
            in next_candidate_features
        )

        for feature_vector in normalized_next_features:
            if len(feature_vector) != self.feature_dimension:
                raise ValueError(
                    "next candidate feature vector "
                    "has incorrect dimension."
                )

        if done and normalized_next_features:
            raise ValueError(
                "Terminal transition must not "
                "contain next candidate actions."
            )

        transition = ReplayTransition(
            chosen_features=tuple(
                float(value)
                for value in chosen_features
            ),
            reward=float(reward),
            next_candidate_features=(
                normalized_next_features
            ),
            done=bool(done),
        )

        self.replay_buffer.append(
            transition
        )

    def can_train(self) -> bool:
        required_size = max(
            self.config.batch_size,
            self.config.min_replay_size,
        )

        return (
            len(self.replay_buffer)
            >= required_size
        )

    def _max_target_q(
        self,
        next_candidate_features: tuple[
            tuple[float, ...],
            ...
        ],
    ) -> torch.Tensor:
        if not next_candidate_features:
            return torch.tensor(
                0.0,
                dtype=torch.float32,
                device=self.device,
            )

        feature_tensor = self._features_tensor(
            next_candidate_features
        )

        q_values = self.target_network(
            feature_tensor
        )

        return torch.max(q_values)

    def train_step(
        self,
    ) -> TrainingStepResult | None:
        """
        Thực hiện một gradient update.

        Trả None nếu replay buffer chưa đủ dữ liệu.
        """

        if not self.can_train():
            return None

        transitions = self.replay_buffer.sample(
            self.config.batch_size
        )

        chosen_features = self._features_tensor(
            [
                transition.chosen_features
                for transition in transitions
            ]
        )

        q_values = self.online_network(
            chosen_features
        )

        target_values: list[torch.Tensor] = []

        with torch.no_grad():
            for transition in transitions:
                reward_tensor = torch.tensor(
                    transition.reward,
                    dtype=torch.float32,
                    device=self.device,
                )

                if transition.done:
                    target = reward_tensor

                else:
                    max_next_q = self._max_target_q(
                        transition.next_candidate_features
                    )

                    target = (
                        reward_tensor
                        + self.config.gamma
                        * max_next_q
                    )

                target_values.append(
                    target
                )

            target_tensor = torch.stack(
                target_values
            )

        loss = self.loss_function(
            q_values,
            target_tensor,
        )

        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config.gradient_clip_norm,
        )

        self.optimizer.step()

        self.gradient_steps += 1

        if (
            self.gradient_steps
            % self.config.target_update_interval
            == 0
        ):
            self.sync_target_network()

        return TrainingStepResult(
            loss=float(
                loss.detach().cpu().item()
            ),
            mean_q_value=float(
                q_values
                .detach()
                .mean()
                .cpu()
                .item()
            ),
            mean_target_value=float(
                target_tensor
                .detach()
                .mean()
                .cpu()
                .item()
            ),
            replay_size=len(
                self.replay_buffer
            ),
            gradient_step=(
                self.gradient_steps
            ),
        )

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

    def decay_epsilon(self) -> float:
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon
            * self.config.epsilon_decay,
        )

        return self.epsilon

    def q_values(
        self,
        encoded_candidates: EncodedCandidateSet,
    ) -> tuple[float, ...]:
        """
        Helper phục vụ inference / debugging.
        """

        if not encoded_candidates.candidates:
            return ()

        with torch.no_grad():
            values = self.online_network(
                self._features_tensor(
                    encoded_candidates.feature_matrix
                )
            )

        return tuple(
            float(value)
            for value in values
            .detach()
            .cpu()
            .tolist()
        )