import pytest

from rl.proposed.agent import (
    ProposedAgentConfig,
    ProposedRoutingAgent,
)
from rl.proposed.candidate_encoder import (
    CandidateEncoder,
)
from rl.proposed.environment import (
    ProposedRoutingEnvironment,
)
from rl.proposed.multi_workload_trainer import (
    WorkloadEnvironment,
    evaluate_workloads,
    train_multi_workload_epoch,
)


def build_environment(
    offset: float = 0.0,
):
    return ProposedRoutingEnvironment(
        distance_matrix_km=[
            [0.0, 2.0 + offset, 4.0],
            [2.0 + offset, 0.0, 2.0],
            [4.0, 2.0, 0.0],
        ],

        travel_time_matrix_minutes=[
            [0.0, 4.0 + offset, 8.0],
            [4.0 + offset, 0.0, 4.0],
            [8.0, 4.0, 0.0],
        ],

        deadlines_minutes=[
            20.0,
            30.0,
        ],

        cluster_ids=[
            0,
            1,
        ],
    )


def build_agent():
    return ProposedRoutingAgent(
        config=ProposedAgentConfig(
            learning_rate=1e-3,

            gamma=1.0,

            epsilon_start=1.0,
            epsilon_min=0.0,
            epsilon_decay=0.9,

            replay_capacity=100,

            batch_size=2,
            min_replay_size=2,

            target_update_interval=2,

            hidden_dimension=16,

            seed=42,

            device="cpu",
        )
    )


def build_workloads():
    return (
        WorkloadEnvironment(
            workload_id=1,
            environment=(
                build_environment(
                    offset=0.0
                )
            ),
        ),

        WorkloadEnvironment(
            workload_id=2,
            environment=(
                build_environment(
                    offset=1.0
                )
            ),
        ),
    )


def test_one_agent_trains_across_two_workloads():
    workloads = build_workloads()

    encoder = CandidateEncoder()

    agent = build_agent()

    result = train_multi_workload_epoch(
        workloads=workloads,
        encoder=encoder,
        agent=agent,
        epoch=1,
        episodes_per_workload=1,
        shuffle=False,
    )

    assert result.workloads == 2
    assert result.episodes == 2

    # Mỗi workload có 2 deliveries.
    assert len(
        agent.replay_buffer
    ) == 4


def test_replay_is_shared_across_workloads():
    workloads = build_workloads()

    encoder = CandidateEncoder()

    agent = build_agent()

    train_multi_workload_epoch(
        workloads=workloads,
        encoder=encoder,
        agent=agent,
        epoch=1,
        episodes_per_workload=2,
        shuffle=False,
    )

    # 2 workloads
    # x 2 episodes
    # x 2 deliveries.
    assert len(
        agent.replay_buffer
    ) == 8


def test_epsilon_decays_across_all_episodes():
    workloads = build_workloads()

    encoder = CandidateEncoder()

    agent = build_agent()

    train_multi_workload_epoch(
        workloads=workloads,
        encoder=encoder,
        agent=agent,
        epoch=1,
        episodes_per_workload=1,
        shuffle=False,
    )

    assert agent.epsilon == pytest.approx(
        0.9 ** 2
    )


def test_multi_workload_evaluation_is_valid():
    workloads = build_workloads()

    encoder = CandidateEncoder()

    agent = build_agent()

    evaluations, summary = (
        evaluate_workloads(
            workloads=workloads,
            encoder=encoder,
            agent=agent,
        )
    )

    assert len(
        evaluations
    ) == 2

    assert (
        summary.workloads
        == 2
    )

    assert (
        summary.deliveries
        == 4
    )

    assert (
        0.0
        <= summary.weighted_sla
        <= 1.0
    )


def test_evaluation_does_not_modify_replay():
    workloads = build_workloads()

    encoder = CandidateEncoder()

    agent = build_agent()

    train_multi_workload_epoch(
        workloads=workloads,
        encoder=encoder,
        agent=agent,
        epoch=1,
        episodes_per_workload=1,
        shuffle=False,
    )

    replay_before = len(
        agent.replay_buffer
    )

    evaluate_workloads(
        workloads=workloads,
        encoder=encoder,
        agent=agent,
    )

    replay_after = len(
        agent.replay_buffer
    )

    assert (
        replay_before
        == replay_after
    )


def test_duplicate_workload_ids_are_rejected():
    workloads = (
        WorkloadEnvironment(
            workload_id=1,
            environment=build_environment(),
        ),

        WorkloadEnvironment(
            workload_id=1,
            environment=build_environment(),
        ),
    )

    encoder = CandidateEncoder()
    agent = build_agent()

    with pytest.raises(ValueError):
        train_multi_workload_epoch(
            workloads=workloads,
            encoder=encoder,
            agent=agent,
            epoch=1,
        )