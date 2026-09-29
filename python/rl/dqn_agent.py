from collections import deque
from dataclasses import dataclass
from random import Random

import torch
from torch import nn
from torch.nn import functional as F

from rl.routing_environment import (
    RoutingEnvironment,
    RoutingState,
)


@dataclass(frozen=True)
class DQNConfig:
    """
    Hyperparameters for Deep Q-Network.

    DQN uses the same RoutingEnvironment,
    state definition, action space, and
    reward function as Q-Learning and SARSA.
    """

    learning_rate: float = 0.001

    discount_factor: float = 0.95

    epsilon_start: float = 1.00

    epsilon_min: float = 0.05

    epsilon_decay: float = 0.998

    episodes: int = 2000

    batch_size: int = 64

    replay_capacity: int = 10000

    min_replay_size: int = 128

    target_update_interval: int = 100

    hidden_dim: int = 128

    gradient_clip_norm: float = 5.0

    elapsed_time_bucket_scale: float = 36.0

    random_seed: int = 42

    device: str = "auto"

    def validate(self) -> None:
        if not (
            self.learning_rate
            > 0.0
        ):
            raise ValueError(
                "learning_rate must be greater "
                "than 0."
            )

        if not (
            0.0
            <= self.discount_factor
            <= 1.0
        ):
            raise ValueError(
                "discount_factor must be in "
                "[0, 1]."
            )

        if not (
            0.0
            <= self.epsilon_start
            <= 1.0
        ):
            raise ValueError(
                "epsilon_start must be in "
                "[0, 1]."
            )

        if not (
            0.0
            <= self.epsilon_min
            <= 1.0
        ):
            raise ValueError(
                "epsilon_min must be in "
                "[0, 1]."
            )

        if (
            self.epsilon_min
            > self.epsilon_start
        ):
            raise ValueError(
                "epsilon_min cannot be greater "
                "than epsilon_start."
            )

        if not (
            0.0
            < self.epsilon_decay
            <= 1.0
        ):
            raise ValueError(
                "epsilon_decay must be in "
                "(0, 1]."
            )

        if self.episodes <= 0:
            raise ValueError(
                "episodes must be greater "
                "than 0."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater "
                "than 0."
            )

        if self.replay_capacity <= 0:
            raise ValueError(
                "replay_capacity must be greater "
                "than 0."
            )

        if self.min_replay_size <= 0:
            raise ValueError(
                "min_replay_size must be greater "
                "than 0."
            )

        if (
            self.min_replay_size
            > self.replay_capacity
        ):
            raise ValueError(
                "min_replay_size cannot exceed "
                "replay_capacity."
            )

        if self.target_update_interval <= 0:
            raise ValueError(
                "target_update_interval must be "
                "greater than 0."
            )

        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater "
                "than 0."
            )

        if self.gradient_clip_norm <= 0.0:
            raise ValueError(
                "gradient_clip_norm must be "
                "greater than 0."
            )

        if self.elapsed_time_bucket_scale <= 0.0:
            raise ValueError(
                "elapsed_time_bucket_scale must "
                "be greater than 0."
            )

        if self.device not in (
            "auto",
            "cpu",
            "cuda",
        ):
            raise ValueError(
                "device must be one of: "
                "'auto', 'cpu', 'cuda'."
            )


@dataclass(frozen=True)
class DQNTransition:
    """
    One replay-memory transition.
    """

    state: RoutingState

    action: int

    reward: float

    next_state: RoutingState

    next_actions: tuple[
        int,
        ...
    ]

    done: bool


@dataclass(frozen=True)
class DQNEpisodeResult:
    """
    Result of one DQN episode.
    """

    episode: int

    epsilon: float

    total_reward: float

    delivery_order: tuple[
        str,
        ...
    ]

    action_order: tuple[
        int,
        ...
    ]

    total_distance_km: float

    total_travel_time_minutes: float

    steps: int

    mean_loss: float | None


@dataclass(frozen=True)
class DQNTrainingResult:
    """
    Summary of DQN training.
    """

    episodes: tuple[
        DQNEpisodeResult,
        ...
    ]

    final_epsilon: float

    replay_size: int

    optimization_steps: int

    parameter_count: int

    device: str


class ReplayBuffer:
    """
    Fixed-size replay memory for DQN.
    """

    def __init__(
        self,
        capacity: int,
        random_seed: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError(
                "capacity must be greater "
                "than 0."
            )

        self.buffer = deque(
            maxlen=capacity
        )

        self.random = Random(
            random_seed
        )

    def append(
        self,
        transition: DQNTransition,
    ) -> None:
        self.buffer.append(
            transition
        )

    def sample(
        self,
        batch_size: int,
    ) -> list[DQNTransition]:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater "
                "than 0."
            )

        if (
            batch_size
            > len(self.buffer)
        ):
            raise ValueError(
                "batch_size cannot exceed "
                "current replay size."
            )

        return self.random.sample(
            list(
                self.buffer
            ),
            batch_size,
        )

    def clear(
        self,
    ) -> None:
        self.buffer.clear()

    def __len__(
        self,
    ) -> int:
        return len(
            self.buffer
        )


class QNetwork(nn.Module):
    """
    Feed-forward neural network that
    approximates Q(s, a).
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                action_dim,
            ),
        )

    def forward(
        self,
        state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            state_tensor
        )


class DQNAgent:
    """
    Deep Q-Network agent for the existing
    RoutingEnvironment.

    State encoding:

        current node one-hot
        +
        visited-delivery mask
        +
        normalized elapsed-time bucket

    Invalid delivery actions are masked.
    """

    def __init__(
        self,
        delivery_count: int,
        config: DQNConfig
        | None = None,
    ) -> None:
        if delivery_count <= 0:
            raise ValueError(
                "delivery_count must be greater "
                "than 0."
            )

        self.delivery_count = (
            delivery_count
        )

        self.config = (
            config
            or DQNConfig()
        )

        self.config.validate()

        self.random = Random(
            self.config.random_seed
        )

        torch.manual_seed(
            self.config.random_seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                self.config.random_seed
            )

        self.device = (
            self._resolve_device()
        )

        self.input_dim = (
            self.delivery_count
            + 1
            + self.delivery_count
            + 1
        )

        self.action_dim = (
            self.delivery_count
        )

        self.online_network = QNetwork(
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            hidden_dim=(
                self.config.hidden_dim
            ),
        ).to(
            self.device
        )

        self.target_network = QNetwork(
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            hidden_dim=(
                self.config.hidden_dim
            ),
        ).to(
            self.device
        )

        self.sync_target_network()

        self.target_network.eval()

        self.optimizer = (
            torch.optim.Adam(
                self.online_network.parameters(),
                lr=self.config.learning_rate,
            )
        )

        self.replay_buffer = ReplayBuffer(
            capacity=(
                self.config.replay_capacity
            ),
            random_seed=(
                self.config.random_seed
            ),
        )

        self.optimization_steps = 0

    def _resolve_device(
        self,
    ) -> torch.device:
        if self.config.device == "cpu":
            return torch.device(
                "cpu"
            )

        if self.config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is "
                    "not available."
                )

            return torch.device(
                "cuda"
            )

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    @property
    def parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter
            in self.online_network.parameters()
        )

    def encode_state(
        self,
        state: RoutingState,
    ) -> torch.Tensor:
        """
        Encode RoutingState as one flat
        floating-point tensor.
        """

        if not (
            0
            <= state.current_index
            <= self.delivery_count
        ):
            raise ValueError(
                "state.current_index is outside "
                "the valid range."
            )

        current_position = [
            0.0
            for _ in range(
                self.delivery_count
                + 1
            )
        ]

        current_position[
            state.current_index
        ] = 1.0

        visited = []

        for action in range(
            self.delivery_count
        ):
            is_visited = (
                state.visited_mask
                & (
                    1 << action
                )
            )

            visited.append(
                1.0
                if is_visited
                else 0.0
            )

        normalized_time = (
            float(
                state.elapsed_time_bucket
            )
            / self.config
            .elapsed_time_bucket_scale
        )

        vector = (
            current_position
            + visited
            + [
                normalized_time
            ]
        )

        return torch.tensor(
            vector,
            dtype=torch.float32,
            device=self.device,
        )

    def best_action(
        self,
        state: RoutingState,
        available_actions: (
            list[int]
            | tuple[int, ...]
        ),
    ) -> int:
        """
        Select the valid action with the
        highest neural-network Q-value.

        Tie-breaking is deterministic.
        """

        actions = tuple(
            sorted(
                available_actions
            )
        )

        if not actions:
            raise ValueError(
                "available_actions cannot "
                "be empty."
            )

        self.online_network.eval()

        with torch.no_grad():
            state_tensor = (
                self.encode_state(
                    state
                )
                .unsqueeze(
                    0
                )
            )

            q_values = (
                self.online_network(
                    state_tensor
                )[
                    0
                ]
            )

        best = actions[
            0
        ]

        best_value = float(
            q_values[
                best
            ].item()
        )

        for action in actions[
            1:
        ]:
            value = float(
                q_values[
                    action
                ].item()
            )

            if value > best_value:
                best = action
                best_value = value

        return best

    def choose_action(
        self,
        state: RoutingState,
        available_actions: (
            list[int]
            | tuple[int, ...]
        ),
        epsilon: float,
    ) -> int:
        """
        Epsilon-greedy action selection
        with invalid-action masking.
        """

        actions = tuple(
            available_actions
        )

        if not actions:
            raise ValueError(
                "available_actions cannot "
                "be empty."
            )

        if not (
            0.0
            <= epsilon
            <= 1.0
        ):
            raise ValueError(
                "epsilon must be in [0, 1]."
            )

        explore = (
            self.random.random()
            < epsilon
        )

        if explore:
            return self.random.choice(
                actions
            )

        return self.best_action(
            state=state,
            available_actions=actions,
        )

    def remember(
        self,
        state: RoutingState,
        action: int,
        reward: float,
        next_state: RoutingState,
        next_actions: (
            list[int]
            | tuple[int, ...]
        ),
        done: bool,
    ) -> None:
        transition = DQNTransition(
            state=state,
            action=int(
                action
            ),
            reward=float(
                reward
            ),
            next_state=next_state,
            next_actions=tuple(
                next_actions
            ),
            done=bool(
                done
            ),
        )

        self.replay_buffer.append(
            transition
        )

    def optimize(
        self,
    ) -> float | None:
        """
        Sample one replay minibatch and
        perform one DQN gradient update.
        """

        required_size = max(
            self.config.batch_size,
            self.config.min_replay_size,
        )

        if (
            len(
                self.replay_buffer
            )
            < required_size
        ):
            return None

        batch = (
            self.replay_buffer.sample(
                self.config.batch_size
            )
        )

        states = torch.stack(
            [
                self.encode_state(
                    transition.state
                )
                for transition
                in batch
            ]
        )

        next_states = torch.stack(
            [
                self.encode_state(
                    transition.next_state
                )
                for transition
                in batch
            ]
        )

        actions = torch.tensor(
            [
                transition.action
                for transition
                in batch
            ],
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            [
                transition.reward
                for transition
                in batch
            ],
            dtype=torch.float32,
            device=self.device,
        )

        done_flags = torch.tensor(
            [
                transition.done
                for transition
                in batch
            ],
            dtype=torch.bool,
            device=self.device,
        )

        self.online_network.train()

        current_q = (
            self.online_network(
                states
            )
            .gather(
                1,
                actions.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )

        self.target_network.eval()

        with torch.no_grad():
            next_q_values = (
                self.target_network(
                    next_states
                )
            )

            max_next_q = torch.zeros(
                len(
                    batch
                ),
                dtype=torch.float32,
                device=self.device,
            )

            for index, transition in enumerate(
                batch
            ):
                if transition.done:
                    continue

                if not transition.next_actions:
                    raise ValueError(
                        "Non-terminal transition "
                        "must have next_actions."
                    )

                action_indices = torch.tensor(
                    transition.next_actions,
                    dtype=torch.long,
                    device=self.device,
                )

                valid_values = (
                    next_q_values[
                        index
                    ]
                    .index_select(
                        0,
                        action_indices,
                    )
                )

                max_next_q[
                    index
                ] = valid_values.max()

            targets = (
                rewards
                + (
                    ~done_flags
                ).float()
                * self.config
                .discount_factor
                * max_next_q
            )

        loss = F.smooth_l1_loss(
            current_q,
            targets,
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config.gradient_clip_norm,
        )

        self.optimizer.step()

        self.optimization_steps += 1

        if (
            self.optimization_steps
            % self.config
            .target_update_interval
            == 0
        ):
            self.sync_target_network()

        return float(
            loss.item()
        )

    def sync_target_network(
        self,
    ) -> None:
        """
        Copy online-network weights to the
        target network.
        """

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def train_episode(
        self,
        environment: RoutingEnvironment,
        episode: int,
        epsilon: float,
    ) -> DQNEpisodeResult:
        """
        Train for one complete routing
        episode.
        """

        state = (
            environment.reset()
        )

        total_reward = 0.0

        delivery_order = []

        action_order = []

        losses = []

        steps = 0

        while True:
            available_actions = (
                environment
                .available_actions()
            )

            action = (
                self.choose_action(
                    state=state,
                    available_actions=(
                        available_actions
                    ),
                    epsilon=epsilon,
                )
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = environment.step(
                action
            )

            next_actions = (
                environment
                .available_actions()
            )

            self.remember(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_actions=next_actions,
                done=done,
            )

            loss = self.optimize()

            if loss is not None:
                losses.append(
                    loss
                )

            total_reward += (
                reward
            )

            delivery_order.append(
                info.delivery_id
            )

            action_order.append(
                action
            )

            steps += 1

            state = next_state

            if done:
                break

        mean_loss = None

        if losses:
            mean_loss = (
                sum(
                    losses
                )
                / len(
                    losses
                )
            )

        return DQNEpisodeResult(
            episode=episode,
            epsilon=float(
                epsilon
            ),
            total_reward=float(
                total_reward
            ),
            delivery_order=tuple(
                delivery_order
            ),
            action_order=tuple(
                action_order
            ),
            total_distance_km=float(
                environment
                .total_distance_km
            ),
            total_travel_time_minutes=float(
                environment
                .total_travel_time_minutes
            ),
            steps=steps,
            mean_loss=(
                None
                if mean_loss is None
                else float(
                    mean_loss
                )
            ),
        )

    def train(
        self,
        environment: RoutingEnvironment,
    ) -> DQNTrainingResult:
        """
        Train DQN for the configured number
        of episodes.
        """

        epsilon = (
            self.config
            .epsilon_start
        )

        episode_results = []

        for episode in range(
            1,
            self.config.episodes
            + 1,
        ):
            result = (
                self.train_episode(
                    environment=environment,
                    episode=episode,
                    epsilon=epsilon,
                )
            )

            episode_results.append(
                result
            )

            epsilon = max(
                self.config.epsilon_min,
                epsilon
                * self.config
                .epsilon_decay,
            )

        return DQNTrainingResult(
            episodes=tuple(
                episode_results
            ),
            final_epsilon=float(
                epsilon
            ),
            replay_size=len(
                self.replay_buffer
            ),
            optimization_steps=(
                self.optimization_steps
            ),
            parameter_count=(
                self.parameter_count
            ),
            device=str(
                self.device
            ),
        )

    def greedy_rollout(
        self,
        environment: RoutingEnvironment,
    ) -> DQNEpisodeResult:
        """
        Evaluate the learned DQN policy
        without exploration or training.
        """

        state = (
            environment.reset()
        )

        total_reward = 0.0

        delivery_order = []

        action_order = []

        steps = 0

        while True:
            actions = (
                environment
                .available_actions()
            )

            action = (
                self.best_action(
                    state=state,
                    available_actions=actions,
                )
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = environment.step(
                action
            )

            total_reward += (
                reward
            )

            delivery_order.append(
                info.delivery_id
            )

            action_order.append(
                action
            )

            steps += 1

            state = next_state

            if done:
                break

        return DQNEpisodeResult(
            episode=0,
            epsilon=0.0,
            total_reward=float(
                total_reward
            ),
            delivery_order=tuple(
                delivery_order
            ),
            action_order=tuple(
                action_order
            ),
            total_distance_km=float(
                environment
                .total_distance_km
            ),
            total_travel_time_minutes=float(
                environment
                .total_travel_time_minutes
            ),
            steps=steps,
            mean_loss=None,
        )

    def clear_replay_buffer(
        self,
    ) -> None:
        self.replay_buffer.clear()