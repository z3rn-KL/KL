import pytest

from rl.proposed.environment import ProposedRoutingEnvironment
from rl.proposed.lookahead_encoder import (
    LOOKAHEAD_FEATURE_NAMES,
    LookaheadCandidateEncoder,
)


def line_environment(cluster_ids=(0, 0, 1)):
    """
    Depot and three deliveries on a line, 4 minutes apart:

        D --4-- 1 --4-- 2 --4-- 3
    """

    times = [
        [abs(i - j) * 4.0 for j in range(4)]
        for i in range(4)
    ]

    return ProposedRoutingEnvironment(
        distance_matrix_km=[[v / 2 for v in row] for row in times],
        travel_time_matrix_minutes=times,
        deadlines_minutes=[None, None, None],
        cluster_ids=list(cluster_ids),
    )


def features_by_name(encoded):
    return dict(zip(LOOKAHEAD_FEATURE_NAMES, encoded.features))


def test_feature_dimension_matches_names():
    encoder = LookaheadCandidateEncoder()

    encoded = encoder.encode_available_actions(line_environment())

    assert encoder.feature_dimension == len(LOOKAHEAD_FEATURE_NAMES)
    assert all(
        len(candidate.features) == len(LOOKAHEAD_FEATURE_NAMES)
        for candidate in encoded.candidates
    )


def test_lookahead_values_for_first_delivery():
    env = line_environment()

    values = features_by_name(
        LookaheadCandidateEncoder().encode_action(env=env, action=0)
    )

    ref = env.objective_config.travel_time_ref_minutes

    # Starting at the depot, the detour is the whole round trip (4 + 4).
    assert values["return_ratio"] == pytest.approx(4.0 / ref)
    assert values["detour_ratio"] == pytest.approx(8.0 / ref)
    assert values["nearest_remaining_ratio"] == pytest.approx(4.0 / ref)
    assert values["mean_remaining_ratio"] == pytest.approx(6.0 / ref)
    # Delivery 2 shares the cluster, delivery 3 does not.
    assert values["same_cluster_remaining_ratio"] == pytest.approx(0.5)


def test_last_delivery_has_no_remaining_features():
    env = line_environment()
    env.step(0)
    env.step(1)

    values = features_by_name(
        LookaheadCandidateEncoder().encode_action(env=env, action=2)
    )

    assert values["nearest_remaining_ratio"] == 0.0
    assert values["mean_remaining_ratio"] == 0.0
    assert values["same_cluster_remaining_ratio"] == 0.0
    assert values["return_ratio"] > 0.0
