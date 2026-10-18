import pytest

from rl.proposed.candidate_encoder import (
    CandidateEncoder,
    FEATURE_NAMES,
)
from rl.proposed.environment import (
    ProposedRoutingEnvironment,
)
from rl.proposed.objective import ObjectiveConfig
def test_late_delivery_count_is_recorded():
    env = build_environment(
        deadlines=[
            3.0,
            20.0,
            30.0,
        ]
    )

    env.step(0)

    assert env.totals.late_deliveries == 1


def test_on_time_delivery_does_not_increment_late_count():
    env = build_environment(
        deadlines=[
            10.0,
            20.0,
            30.0,
        ]
    )

    env.step(0)

    assert env.totals.late_deliveries == 0


def test_reward_matches_objective_with_late_delivery_penalty():
    config = ObjectiveConfig(
        distance_weight=0.35,
        travel_time_weight=0.35,
        late_delivery_weight=0.15,
        lateness_weight=0.10,
        cluster_switch_weight=0.05,
    )

    env = ProposedRoutingEnvironment(
        distance_matrix_km=[
            [0.0, 2.0, 4.0, 6.0],
            [2.0, 0.0, 2.0, 4.0],
            [4.0, 2.0, 0.0, 2.0],
            [6.0, 4.0, 2.0, 0.0],
        ],

        travel_time_matrix_minutes=[
            [0.0, 4.0, 8.0, 12.0],
            [4.0, 0.0, 4.0, 8.0],
            [8.0, 4.0, 0.0, 4.0],
            [12.0, 8.0, 4.0, 0.0],
        ],

        # d1 sẽ bị trễ:
        # depot -> d1 = 4 phút
        # deadline = 3 phút
        deadlines_minutes=[
            3.0,
            20.0,
            30.0,
        ],

        cluster_ids=[
            0,
            0,
            1,
        ],

        objective_config=config,
    )

    total_reward = 0.0

    total_reward += env.step(0).reward
    total_reward += env.step(1).reward
    total_reward += env.step(2).reward

    objective = env.final_objective()

    assert env.totals.late_deliveries == 1

    assert total_reward == pytest.approx(
        -objective.total_cost
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

    return ProposedRoutingEnvironment(
        distance_matrix_km=distance_matrix,
        travel_time_matrix_minutes=(
            travel_time_matrix
        ),
        deadlines_minutes=deadlines,
        cluster_ids=[
            0,
            0,
            1,
        ],
    )


def test_feature_dimension_is_eight():
    encoder = CandidateEncoder()

    assert encoder.feature_dimension == 8
    assert len(FEATURE_NAMES) == 8


def test_initial_actions_are_encoded():
    env = build_environment()
    encoder = CandidateEncoder()

    encoded = encoder.encode_available_actions(
        env
    )

    assert encoded.actions == (
        0,
        1,
        2,
    )

    assert len(encoded.feature_matrix) == 3


def test_feature_vector_has_expected_dimension():
    env = build_environment()
    encoder = CandidateEncoder()

    encoded = encoder.encode_action(
        env=env,
        action=0,
    )

    assert len(encoded.features) == 8


def test_initial_remaining_ratio_is_one():
    env = build_environment()
    encoder = CandidateEncoder()

    encoded = encoder.encode_action(
        env=env,
        action=0,
    )

    remaining_ratio = encoded.features[1]

    assert remaining_ratio == pytest.approx(
        1.0
    )


def test_visited_action_disappears():
    env = build_environment()
    encoder = CandidateEncoder()

    env.step(0)

    encoded = encoder.encode_available_actions(
        env
    )

    assert encoded.actions == (
        1,
        2,
    )


def test_remaining_ratio_updates_after_step():
    env = build_environment()
    encoder = CandidateEncoder()

    env.step(0)

    encoded = encoder.encode_action(
        env=env,
        action=1,
    )

    remaining_ratio = encoded.features[1]

    assert remaining_ratio == pytest.approx(
        2.0 / 3.0
    )


def test_missing_deadline_uses_validity_feature():
    env = build_environment(
        deadlines=[
            None,
            20.0,
            30.0,
        ]
    )

    encoder = CandidateEncoder()

    encoded = encoder.encode_action(
        env=env,
        action=0,
    )

    slack_ratio = encoded.features[4]
    deadline_valid = encoded.features[5]
    lateness_ratio = encoded.features[6]

    assert slack_ratio == 0.0
    assert deadline_valid == 0.0
    assert lateness_ratio == 0.0


def test_cluster_switch_feature():
    env = build_environment()
    encoder = CandidateEncoder()

    # Delivery action 1 thuộc cluster 0.
    env.step(1)

    # Delivery action 2 thuộc cluster 1.
    encoded = encoder.encode_action(
        env=env,
        action=2,
    )

    cluster_switch = encoded.features[7]

    assert cluster_switch == 1.0


def test_same_cluster_feature_is_zero():
    env = build_environment()
    encoder = CandidateEncoder()

    env.step(0)

    encoded = encoder.encode_action(
        env=env,
        action=1,
    )

    cluster_switch = encoded.features[7]

    assert cluster_switch == 0.0


def test_negative_slack_is_preserved():
    env = build_environment(
        deadlines=[
            2.0,
            20.0,
            30.0,
        ]
    )

    encoder = CandidateEncoder()

    encoded = encoder.encode_action(
        env=env,
        action=0,
    )

    slack_ratio = encoded.features[4]
    lateness_ratio = encoded.features[6]

    assert slack_ratio < 0.0
    assert lateness_ratio > 0.0