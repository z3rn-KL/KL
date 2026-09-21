from dataclasses import dataclass
from random import Random

from rl.routing_environment import (
    RoutingEnvironment,
    RoutingState,
)


@dataclass(frozen=True)
class QLearningConfig:
    """
    Hyperparameters for tabular Q-Learning.

    These are initial experimental values.
    They are not final thesis parameters.
    """

    learning_rate: float = 0.20

    discount_factor: float = 0.95

    epsilon_start: float = 1.00

    epsilon_min: float = 0.05

    epsilon_decay: float = 0.995

    episodes: int = 2000

    random_seed: int = 42

    def validate(self) -> None:
        if not (
            0.0
            < self.learning_rate
            <= 1.0
        ):
            raise ValueError(
                "learning_rate must be in "
                "(0, 1]."
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


@dataclass(frozen=True)
class EpisodeResult:
    """
    Result of one Q-Learning episode.
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


@dataclass(frozen=True)
class TrainingResult:
    """
    Summary of Q-Learning training.
    """

    episodes: tuple[
        EpisodeResult,
        ...
    ]

    final_epsilon: float

    q_table_size: int


class QLearningAgent:
    """
    Tabular Q-Learning agent for the
    RoutingEnvironment.

    Q-table key:

        (state, action)

    where state is a hashable RoutingState.
    """

    def __init__(
        self,
        config: QLearningConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            or QLearningConfig()
        )

        self.config.validate()

        self.q_table: dict[
            tuple[
                RoutingState,
                int,
            ],
            float,
        ] = {}

        self.random = Random(
            self.config.random_seed
        )

    def get_q_value(
        self,
        state: RoutingState,
        action: int,
    ) -> float:
        """
        Return Q(s, a).

        Unseen state-action pairs start at 0.
        """

        return float(
            self.q_table.get(
                (
                    state,
                    action,
                ),
                0.0,
            )
        )

    def set_q_value(
        self,
        state: RoutingState,
        action: int,
        value: float,
    ) -> None:
        self.q_table[
            (
                state,
                action,
            )
        ] = float(
            value
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
        Return greedy action.

        Tie-breaking is deterministic:
        the smaller action index wins.
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

        best = actions[
            0
        ]

        best_value = (
            self.get_q_value(
                state,
                best,
            )
        )

        for action in actions[
            1:
        ]:
            value = (
                self.get_q_value(
                    state,
                    action,
                )
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
        Epsilon-greedy action selection.
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

    def update(
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
    ) -> float:
        """
        Apply one Q-Learning update.

        Q(s,a) =
            Q(s,a)
            + alpha * (
                target - Q(s,a)
            )

        target =
            reward
            if terminal

        otherwise:

            reward
            + gamma
            * max_a' Q(s',a')
        """

        current_q = (
            self.get_q_value(
                state,
                action,
            )
        )

        if done:
            target = float(
                reward
            )

        else:
            if not next_actions:
                raise ValueError(
                    "Non-terminal state must "
                    "have available actions."
                )

            max_next_q = max(
                self.get_q_value(
                    next_state,
                    next_action,
                )
                for next_action
                in next_actions
            )

            target = (
                float(
                    reward
                )
                + self.config
                .discount_factor
                * max_next_q
            )

        new_q = (
            current_q
            + self.config.learning_rate
            * (
                target
                - current_q
            )
        )

        self.set_q_value(
            state=state,
            action=action,
            value=new_q,
        )

        return float(
            new_q
        )

    def train_episode(
        self,
        environment: RoutingEnvironment,
        episode: int,
        epsilon: float,
    ) -> EpisodeResult:
        """
        Train for exactly one episode.
        """

        state = (
            environment.reset()
        )

        total_reward = 0.0

        delivery_order = []

        action_order = []

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

            self.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_actions=next_actions,
                done=done,
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

        return EpisodeResult(
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
        )

    def train(
        self,
        environment: RoutingEnvironment,
    ) -> TrainingResult:
        """
        Train the Q-table for the configured
        number of episodes.
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

        return TrainingResult(
            episodes=tuple(
                episode_results
            ),
            final_epsilon=float(
                epsilon
            ),
            q_table_size=len(
                self.q_table
            ),
        )

    def greedy_rollout(
        self,
        environment: RoutingEnvironment,
    ) -> EpisodeResult:
        """
        Evaluate the learned policy without
        exploration and without updating the
        Q-table.
        """

        state = (
            environment.reset()
        )

        delivery_order = []

        action_order = []

        total_reward = 0.0

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

        return EpisodeResult(
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
        )

    def clear_q_table(
        self,
    ) -> None:
        self.q_table.clear()