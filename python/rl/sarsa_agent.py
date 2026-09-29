from dataclasses import dataclass
from random import Random

from rl.routing_environment import (
    RoutingEnvironment,
    RoutingState,
)


@dataclass(frozen=True)
class SarsaConfig:
    """
    Hyperparameters for tabular SARSA.

    The default values intentionally match
    the Q-Learning configuration so that
    both algorithms can be compared under
    the same experimental conditions.
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
    Result of one SARSA episode.
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
    Summary of SARSA training.
    """

    episodes: tuple[
        EpisodeResult,
        ...
    ]

    final_epsilon: float

    q_table_size: int


class SarsaAgent:
    """
    Tabular SARSA agent for the
    RoutingEnvironment.

    SARSA is an on-policy TD control
    algorithm.

    Q-table key:

        (state, action)

    where state is a hashable
    RoutingState.
    """

    def __init__(
        self,
        config: SarsaConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            or SarsaConfig()
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

        Unseen state-action pairs start
        at 0.
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
        """
        Store Q(s, a).
        """

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
        Return the greedy action.

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
        Select an action using an
        epsilon-greedy policy.
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
        next_action: int | None,
        done: bool,
    ) -> float:
        """
        Apply one SARSA update.

        Q(s,a) =
            Q(s,a)
            + alpha * (
                target - Q(s,a)
            )

        Terminal target:

            target = reward

        Non-terminal SARSA target:

            target =
                reward
                + gamma
                * Q(s', a')

        The important difference from
        Q-Learning is that a' is the
        action actually selected by the
        current epsilon-greedy policy.
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
            if next_action is None:
                raise ValueError(
                    "Non-terminal SARSA update "
                    "requires next_action."
                )

            next_q = (
                self.get_q_value(
                    next_state,
                    next_action,
                )
            )

            target = (
                float(
                    reward
                )
                + self.config
                .discount_factor
                * next_q
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
        Train for exactly one SARSA
        episode.

        SARSA follows:

            S -> A -> R -> S' -> A'

        The next action is selected before
        updating Q(s, a), which makes the
        algorithm on-policy.
        """

        state = (
            environment.reset()
        )

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

        total_reward = 0.0

        delivery_order = []

        action_order = []

        steps = 0

        while True:
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

            if done:
                self.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_action=None,
                    done=True,
                )

                break

            next_actions = (
                environment
                .available_actions()
            )

            next_action = (
                self.choose_action(
                    state=next_state,
                    available_actions=(
                        next_actions
                    ),
                    epsilon=epsilon,
                )
            )

            self.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                done=False,
            )

            state = next_state
            action = next_action

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
        Train the SARSA Q-table for the
        configured number of episodes.
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
        Evaluate the learned policy with
        epsilon = 0 and without updating
        the Q-table.
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
        """
        Remove every learned state-action
        value.
        """

        self.q_table.clear()