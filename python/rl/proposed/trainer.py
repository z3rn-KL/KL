from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from .agent import (
    ProposedRoutingAgent,
    TrainingStepResult,
)
from .candidate_encoder import (
    CandidateEncoder,
    EncodedCandidate,
)
from .environment import (
    EpisodeTotals,
    ProposedRoutingEnvironment,
)


@dataclass(frozen=True)
class EpisodeResult:
    """
    Kết quả của một episode huấn luyện.
    """

    episode_index: int

    route_actions: tuple[int, ...]

    total_reward: float
    objective_cost: float

    total_distance_km: float
    total_travel_time_minutes: float
    total_lateness_minutes: float
    cluster_switches: int

    gradient_updates: int
    mean_loss: float | None

    epsilon_before: float
    epsilon_after: float


@dataclass(frozen=True)
class EvaluationResult:
    """
    Kết quả inference với exploration=False.
    """

    route_actions: tuple[int, ...]

    total_reward: float
    objective_cost: float

    total_distance_km: float
    total_travel_time_minutes: float
    total_lateness_minutes: float
    cluster_switches: int


@dataclass(frozen=True)
class TrainingRunResult:
    """
    Tổng hợp toàn bộ quá trình train.
    """

    episodes: tuple[EpisodeResult, ...]

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def final_epsilon(self) -> float:
        if not self.episodes:
            raise RuntimeError(
                "Training run contains no episodes."
            )

        return self.episodes[-1].epsilon_after

    @property
    def mean_reward(self) -> float:
        if not self.episodes:
            raise RuntimeError(
                "Training run contains no episodes."
            )

        return mean(
            episode.total_reward
            for episode in self.episodes
        )

    @property
    def mean_objective_cost(self) -> float:
        if not self.episodes:
            raise RuntimeError(
                "Training run contains no episodes."
            )

        return mean(
            episode.objective_cost
            for episode in self.episodes
        )


def _find_encoded_candidate(
    *,
    action: int,
    candidates: Sequence[EncodedCandidate],
) -> EncodedCandidate:
    for candidate in candidates:
        if candidate.action == action:
            return candidate

    raise RuntimeError(
        f"Action {action} was not found "
        "in encoded candidate set."
    )


def run_training_episode(
    *,
    env: ProposedRoutingEnvironment,
    encoder: CandidateEncoder,
    agent: ProposedRoutingAgent,
    episode_index: int,
) -> EpisodeResult:
    """
    Chạy một episode huấn luyện hoàn chỉnh.

    Quy trình:

        reset environment
            ↓
        encode candidates
            ↓
        epsilon-greedy action
            ↓
        environment.step(action)
            ↓
        reward = -Delta J
            ↓
        encode next candidates
            ↓
        lưu replay transition
            ↓
        train_step()
            ↓
        lặp tới terminal
    """

    env.reset()

    total_reward = 0.0

    training_results: list[
        TrainingStepResult
    ] = []

    epsilon_before = agent.epsilon

    while not env.done:
        encoded_candidates = (
            encoder.encode_available_actions(
                env
            )
        )

        action = agent.select_action(
            encoded_candidates,
            explore=True,
        )

        chosen_candidate = (
            _find_encoded_candidate(
                action=action,
                candidates=(
                    encoded_candidates.candidates
                ),
            )
        )

        step_result = env.step(
            action
        )

        total_reward += (
            step_result.reward
        )

        if step_result.done:
            next_candidate_features = ()
        else:
            next_encoded = (
                encoder.encode_available_actions(
                    env
                )
            )

            next_candidate_features = (
                next_encoded.feature_matrix
            )

        agent.remember(
            chosen_features=(
                chosen_candidate.features
            ),
            reward=step_result.reward,
            next_candidate_features=(
                next_candidate_features
            ),
            done=step_result.done,
        )

        training_result = (
            agent.train_step()
        )

        if training_result is not None:
            training_results.append(
                training_result
            )

    final_objective = (
        env.final_objective()
    )

    epsilon_after = (
        agent.decay_epsilon()
    )

    if training_results:
        mean_loss = mean(
            result.loss
            for result in training_results
        )
    else:
        mean_loss = None

    return EpisodeResult(
        episode_index=episode_index,
        route_actions=tuple(
            env.route_actions
        ),
        total_reward=total_reward,
        objective_cost=(
            final_objective.total_cost
        ),
        total_distance_km=(
            env.totals.total_distance_km
        ),
        total_travel_time_minutes=(
            env.totals.total_travel_time_minutes
        ),
        total_lateness_minutes=(
            env.totals.total_lateness_minutes
        ),
        cluster_switches=(
            env.totals.cluster_switches
        ),
        gradient_updates=len(
            training_results
        ),
        mean_loss=mean_loss,
        epsilon_before=epsilon_before,
        epsilon_after=epsilon_after,
    )


def train_agent(
    *,
    env: ProposedRoutingEnvironment,
    encoder: CandidateEncoder,
    agent: ProposedRoutingAgent,
    episodes: int,
) -> TrainingRunResult:
    """
    Train agent trên một environment/workload.

    Đây là phiên bản controlled trainer đầu tiên.

    Bước dataset-scale sau này sẽ mở rộng thành:
        nhiều workload
        train / validation / test
    """

    if episodes <= 0:
        raise ValueError(
            "episodes must be > 0."
        )

    results: list[EpisodeResult] = []

    for episode_index in range(
        1,
        episodes + 1,
    ):
        result = run_training_episode(
            env=env,
            encoder=encoder,
            agent=agent,
            episode_index=episode_index,
        )

        results.append(
            result
        )

    return TrainingRunResult(
        episodes=tuple(results)
    )


def evaluate_agent(
    *,
    env: ProposedRoutingEnvironment,
    encoder: CandidateEncoder,
    agent: ProposedRoutingAgent,
) -> EvaluationResult:
    """
    Chạy inference không exploration và không training.

    Agent luôn chọn:

        argmax Q(s, a)

    Không:
        - thêm transition vào replay
        - cập nhật network
        - decay epsilon
    """

    env.reset()

    total_reward = 0.0

    while not env.done:
        encoded_candidates = (
            encoder.encode_available_actions(
                env
            )
        )

        action = agent.select_action(
            encoded_candidates,
            explore=False,
        )

        step_result = env.step(
            action
        )

        total_reward += (
            step_result.reward
        )

    objective = env.final_objective()

    return EvaluationResult(
        route_actions=tuple(
            env.route_actions
        ),
        total_reward=total_reward,
        objective_cost=(
            objective.total_cost
        ),
        total_distance_km=(
            env.totals.total_distance_km
        ),
        total_travel_time_minutes=(
            env.totals.total_travel_time_minutes
        ),
        total_lateness_minutes=(
            env.totals.total_lateness_minutes
        ),
        cluster_switches=(
            env.totals.cluster_switches
        ),
    )