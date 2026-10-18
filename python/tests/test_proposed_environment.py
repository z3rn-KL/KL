import pytest

from rl.proposed.environment import (
    ProposedRoutingEnvironment,
)


def build_environment(
    deadlines=None,
):
    distance_matrix = [
        [0.0, 2.0, 4.0, 6.0],
        [2.0, 0.0, 2.0, 4.0],
        [4.0, 2.0, 0.0, 2.0],
        [6.0, 4.0, 2.0, 0.0],
    ]

    travel_time_matrix = [
        [0.0, 4.0, 8.0, 12.0],
        [4.0, 0.0, 4.0, 8.0],
        [8.0, 4.0, 0.0, 4.0],
        [12.0, 8.0, 4.0, 0.0],
    ]

    if deadlines is None:
        deadlines = [
            10.0,
            20.0,
            30.0,
        ]

    cluster_ids = [
        0,
        0,
        1,
    ]

    return ProposedRoutingEnvironment(
        distance_matrix_km=distance_matrix,
        travel_time_matrix_minutes=(
            travel_time_matrix
        ),
        deadlines_minutes=deadlines,
        cluster_ids=cluster_ids,
    )


def test_reset_starts_at_depot():
    env = build_environment()

    state = env.reset()

    assert state.current_node == 0
    assert state.visited_mask == 0
    assert state.elapsed_time_minutes == 0.0
    assert state.remaining_deliveries == 3


def test_all_actions_available_initially():
    env = build_environment()

    assert env.available_actions() == (
        0,
        1,
        2,
    )


def test_first_delivery_is_not_cluster_switch():
    env = build_environment()

    info = env.candidate_info(0)

    assert info.cluster_switched is False


def test_same_cluster_is_not_switch():
    env = build_environment()

    env.step(0)

    info = env.candidate_info(1)

    assert info.cluster_switched is False


def test_different_cluster_is_switch():
    env = build_environment()

    env.step(1)

    info = env.candidate_info(2)

    assert info.cluster_switched is True


def test_lateness_is_computed_from_arrival():
    env = build_environment(
        deadlines=[
            3.0,
            20.0,
            30.0,
        ]
    )

    info = env.candidate_info(0)

    assert info.estimated_arrival_minutes == 4.0
    assert info.lateness_minutes == 1.0
    assert info.slack_minutes == -1.0


def test_visited_action_is_removed():
    env = build_environment()

    env.step(0)

    assert env.available_actions() == (
        1,
        2,
    )

    with pytest.raises(ValueError):
        env.step(0)


def test_missing_deadline_has_no_lateness():
    env = build_environment(
        deadlines=[
            None,
            20.0,
            30.0,
        ]
    )

    info = env.candidate_info(0)

    assert info.deadline_valid is False
    assert info.slack_minutes is None
    assert info.lateness_minutes == 0.0


def test_final_step_returns_to_depot():
    env = build_environment()

    env.step(0)
    env.step(1)
    result = env.step(2)

    assert result.done is True
    assert result.state.current_node == 0

    assert (
        result.return_to_depot_cost
        is not None
    )

    assert env.available_actions() == ()


def test_route_totals_include_return_to_depot():
    env = build_environment()

    env.step(0)
    env.step(1)
    env.step(2)

    totals = env.totals

    # Route:
    #
    # depot -> d1 = 2 km
    # d1 -> d2    = 2 km
    # d2 -> d3    = 2 km
    # d3 -> depot = 6 km
    #
    # total = 12 km

    assert totals.total_distance_km == pytest.approx(
        12.0
    )

    # Travel time:
    #
    # 4 + 4 + 4 + 12 = 24 minutes

    assert (
        totals.total_travel_time_minutes
        == pytest.approx(24.0)
    )

    # Cluster sequence:
    #
    # 0 -> 0 -> 1
    #
    # chỉ có 1 switch

    assert totals.cluster_switches == 1


def test_total_reward_matches_negative_objective():
    env = build_environment()

    total_reward = 0.0

    total_reward += env.step(0).reward
    total_reward += env.step(1).reward
    total_reward += env.step(2).reward

    objective = env.final_objective()

    assert total_reward == pytest.approx(
        -objective.total_cost
    )


def test_action_after_episode_finished_is_rejected():
    env = build_environment()

    env.step(0)
    env.step(1)
    env.step(2)

    with pytest.raises(RuntimeError):
        env.step(0)