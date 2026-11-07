from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import mean
from typing import Callable, Sequence

from .agent import ProposedRoutingAgent
from .candidate_encoder import CandidateEncoder
from .environment import ProposedRoutingEnvironment
from .trainer import (
    EvaluationResult,
    evaluate_agent,
    run_training_episode,
)


@dataclass(frozen=True)
class WorkloadEnvironment:
    """
    Một workload có ID và environment tương ứng.
    """

    workload_id: int
    environment: ProposedRoutingEnvironment


@dataclass(frozen=True)
class EpochResult:
    """
    Tổng hợp một epoch đi qua toàn bộ train workloads.
    """

    epoch: int
    workloads: int
    episodes: int

    mean_training_reward: float
    mean_training_objective: float

    epsilon_start: float
    epsilon_end: float


@dataclass(frozen=True)
class WorkloadEvaluation:
    workload_id: int

    number_of_deliveries: int

    route_actions: tuple[int, ...]

    objective_cost: float

    distance_km: float
    travel_time_minutes: float

    late_deliveries: int
    lateness_minutes: float

    valid_deadlines: int
    on_time_deliveries: int
    on_time_rate: float

    cluster_switches: int


@dataclass(frozen=True)
class ValidationSummary:
    workloads: int
    deliveries: int

    mean_distance_km: float
    mean_travel_time_minutes: float

    late_deliveries: int
    valid_deadlines: int
    weighted_sla: float

    total_lateness_minutes: float
    mean_lateness_minutes: float

    mean_cluster_switches: float


def _validate_workloads(
    workloads: Sequence[WorkloadEnvironment],
) -> None:
    if not workloads:
        raise ValueError(
            "workloads cannot be empty."
        )

    workload_ids = [
        workload.workload_id
        for workload in workloads
    ]

    if (
        len(set(workload_ids))
        != len(workload_ids)
    ):
        raise ValueError(
            "workload_id values must be unique."
        )


def train_multi_workload_epoch(
    *,
    workloads: Sequence[WorkloadEnvironment],
    encoder: CandidateEncoder,
    agent: ProposedRoutingAgent,
    epoch: int,
    episodes_per_workload: int = 1,
    shuffle: bool = True,
    seed: int = 42,
) -> EpochResult:
    """
    Một epoch:

        shuffle train workloads
            ↓
        cùng một agent đi qua từng workload
            ↓
        replay buffer được giữ lại
            ↓
        network tiếp tục học giữa các workloads

    Quan trọng:
        KHÔNG tạo agent mới cho mỗi workload.
    """

    _validate_workloads(
        workloads
    )

    if epoch <= 0:
        raise ValueError(
            "epoch must be > 0."
        )

    if episodes_per_workload <= 0:
        raise ValueError(
            "episodes_per_workload must be > 0."
        )

    ordered_workloads = list(
        workloads
    )

    if shuffle:
        rng = random.Random(
            seed + epoch
        )

        rng.shuffle(
            ordered_workloads
        )

    epsilon_start = agent.epsilon

    rewards: list[float] = []
    objectives: list[float] = []

    episode_counter = 0

    for workload in ordered_workloads:
        env = workload.environment

        for _ in range(
            episodes_per_workload
        ):
            episode_counter += 1

            result = run_training_episode(
                env=env,
                encoder=encoder,
                agent=agent,
                episode_index=(
                    episode_counter
                ),
            )

            rewards.append(
                result.total_reward
            )

            objectives.append(
                result.objective_cost
            )

    return EpochResult(
        epoch=epoch,

        workloads=len(
            ordered_workloads
        ),

        episodes=(
            episode_counter
        ),

        mean_training_reward=mean(
            rewards
        ),

        mean_training_objective=mean(
            objectives
        ),

        epsilon_start=epsilon_start,

        epsilon_end=(
            agent.epsilon
        ),
    )


def evaluate_workload(
    *,
    workload: WorkloadEnvironment,
    encoder: CandidateEncoder,
    agent: ProposedRoutingAgent,
) -> WorkloadEvaluation:
    """
    Greedy evaluation.

    Không thêm replay.
    Không gradient update.
    Không decay epsilon.
    """

    env = workload.environment

    result: EvaluationResult = (
        evaluate_agent(
            env=env,
            encoder=encoder,
            agent=agent,
        )
    )

    totals = env.totals

    valid_deadlines = (
        env.objective_context
        .number_of_valid_deadlines
    )

    late_deliveries = (
        totals.late_deliveries
    )

    on_time_deliveries = max(
        0,
        valid_deadlines
        - late_deliveries,
    )

    if valid_deadlines == 0:
        on_time_rate = 1.0
    else:
        on_time_rate = (
            on_time_deliveries
            / valid_deadlines
        )

    return WorkloadEvaluation(
        workload_id=(
            workload.workload_id
        ),

        number_of_deliveries=(
            env.number_of_deliveries
        ),

        route_actions=(
            result.route_actions
        ),

        objective_cost=(
            result.objective_cost
        ),

        distance_km=(
            result.total_distance_km
        ),

        travel_time_minutes=(
            result.total_travel_time_minutes
        ),

        late_deliveries=(
            late_deliveries
        ),

        lateness_minutes=(
            result.total_lateness_minutes
        ),

        valid_deadlines=(
            valid_deadlines
        ),

        on_time_deliveries=(
            on_time_deliveries
        ),

        on_time_rate=(
            on_time_rate
        ),

        cluster_switches=(
            result.cluster_switches
        ),
    )


def evaluate_workloads(
    *,
    workloads: Sequence[WorkloadEnvironment],
    encoder: CandidateEncoder,
    agent: ProposedRoutingAgent,
) -> tuple[
    tuple[WorkloadEvaluation, ...],
    ValidationSummary,
]:
    """
    Đánh giá một agent trên nhiều workloads.
    """

    _validate_workloads(
        workloads
    )

    evaluations = tuple(
        evaluate_workload(
            workload=workload,
            encoder=encoder,
            agent=agent,
        )
        for workload in workloads
    )

    total_valid_deadlines = sum(
        result.valid_deadlines
        for result in evaluations
    )

    total_late_deliveries = sum(
        result.late_deliveries
        for result in evaluations
    )

    if total_valid_deadlines == 0:
        weighted_sla = 1.0
    else:
        weighted_sla = (
            (
                total_valid_deadlines
                - total_late_deliveries
            )
            / total_valid_deadlines
        )

    summary = ValidationSummary(
        workloads=len(
            evaluations
        ),

        deliveries=sum(
            result.number_of_deliveries
            for result in evaluations
        ),

        mean_distance_km=mean(
            result.distance_km
            for result in evaluations
        ),

        mean_travel_time_minutes=mean(
            result.travel_time_minutes
            for result in evaluations
        ),

        late_deliveries=(
            total_late_deliveries
        ),

        valid_deadlines=(
            total_valid_deadlines
        ),

        weighted_sla=(
            weighted_sla
        ),

        total_lateness_minutes=sum(
            result.lateness_minutes
            for result in evaluations
        ),

        mean_lateness_minutes=mean(
            result.lateness_minutes
            for result in evaluations
        ),

        mean_cluster_switches=mean(
            result.cluster_switches
            for result in evaluations
        ),
    )

    return (
        evaluations,
        summary,
    )