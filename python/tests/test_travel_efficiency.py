import numpy as np
import pytest

from routing.travel_efficiency import (
    TravelEfficiencyScorer,
    TravelEfficiencyWeights,
)


def test_closer_and_faster_candidate_scores_higher():
    scorer = TravelEfficiencyScorer()

    result = scorer.score_candidates(
        distance_km=[
            1.0,
            3.0,
            5.0,
        ],
        travel_time_minutes=[
            5.0,
            10.0,
            20.0,
        ],
    )

    scores = (
        result.travel_efficiency
    )

    assert (
        scores[0]
        > scores[1]
        > scores[2]
    )


def test_best_candidate_scores_one():
    scorer = TravelEfficiencyScorer()

    result = scorer.score_candidates(
        distance_km=[
            1.0,
            3.0,
            5.0,
        ],
        travel_time_minutes=[
            5.0,
            10.0,
            20.0,
        ],
    )

    assert result.travel_efficiency[
        0
    ] == pytest.approx(
        1.0
    )


def test_worst_candidate_scores_zero():
    scorer = TravelEfficiencyScorer()

    result = scorer.score_candidates(
        distance_km=[
            1.0,
            3.0,
            5.0,
        ],
        travel_time_minutes=[
            5.0,
            10.0,
            20.0,
        ],
    )

    assert result.travel_efficiency[
        2
    ] == pytest.approx(
        0.0
    )


def test_travel_time_has_more_influence():
    weights = TravelEfficiencyWeights(
        distance=0.40,
        travel_time=0.60,
    )

    scorer = TravelEfficiencyScorer(
        weights=weights
    )

    result = scorer.score_candidates(
        distance_km=[
            1.0,
            2.0,
        ],
        travel_time_minutes=[
            20.0,
            5.0,
        ],
    )

    # Candidate 0 is closer,
    # but candidate 1 is much faster.
    #
    # Since time weight is 0.60,
    # candidate 1 should win.

    assert (
        result.travel_efficiency[1]
        >
        result.travel_efficiency[0]
    )


def test_equal_candidates_receive_equal_scores():
    scorer = TravelEfficiencyScorer()

    result = scorer.score_candidates(
        distance_km=[
            2.0,
            2.0,
            2.0,
        ],
        travel_time_minutes=[
            8.0,
            8.0,
            8.0,
        ],
    )

    assert np.allclose(
        result.travel_efficiency,
        [
            1.0,
            1.0,
            1.0,
        ],
    )


def test_unreachable_candidate_scores_zero():
    scorer = TravelEfficiencyScorer()

    result = scorer.score_candidates(
        distance_km=[
            1.0,
            np.inf,
            3.0,
        ],
        travel_time_minutes=[
            5.0,
            np.inf,
            10.0,
        ],
    )

    assert result.travel_efficiency[
        1
    ] == pytest.approx(
        0.0
    )


def test_single_candidate_is_fully_efficient():
    scorer = TravelEfficiencyScorer()

    result = scorer.score_candidates(
        distance_km=[
            2.5,
        ],
        travel_time_minutes=[
            7.0,
        ],
    )

    assert result.travel_efficiency[
        0
    ] == pytest.approx(
        1.0
    )


def test_negative_distance_is_rejected():
    scorer = TravelEfficiencyScorer()

    with pytest.raises(
        ValueError
    ):
        scorer.score_candidates(
            distance_km=[
                -1.0,
                2.0,
            ],
            travel_time_minutes=[
                5.0,
                10.0,
            ],
        )


def test_negative_travel_time_is_rejected():
    scorer = TravelEfficiencyScorer()

    with pytest.raises(
        ValueError
    ):
        scorer.score_candidates(
            distance_km=[
                1.0,
                2.0,
            ],
            travel_time_minutes=[
                -5.0,
                10.0,
            ],
        )


def test_mismatched_arrays_are_rejected():
    scorer = TravelEfficiencyScorer()

    with pytest.raises(
        ValueError
    ):
        scorer.score_candidates(
            distance_km=[
                1.0,
                2.0,
            ],
            travel_time_minutes=[
                5.0,
            ],
        )


def test_weights_must_sum_to_one():
    weights = TravelEfficiencyWeights(
        distance=0.50,
        travel_time=0.60,
    )

    with pytest.raises(
        ValueError
    ):
        TravelEfficiencyScorer(
            weights=weights
        )