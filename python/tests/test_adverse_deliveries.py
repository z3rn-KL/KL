import numpy as np
import pytest
from domain import Delivery
from clustering import align_cluster_ids, screen_adverse_deliveries


def test_noise_and_road_isolation_are_separate():
    deliveries = [Delivery(str(i), 10, 106) for i in range(4)]
    matrix = np.array([
        [0, 1, 2, 2, 20],
        [1, 0, 1, 1, 20],
        [2, 1, 0, 1, 20],
        [2, 1, 1, 0, 20],
        [20, 20, 20, 20, 0],
    ], dtype=float)
    report = screen_adverse_deliveries(deliveries, matrix, dbscan_labels=[0, -1, 0, 0])
    assert "dbscan_spatial_noise" in report.entries[1].reasons
    assert "road_isolation_candidate" not in report.entries[1].reasons
    assert "road_isolation_candidate" in report.entries[3].reasons
    assert "dbscan_spatial_noise" not in report.entries[3].reasons
    assert all("high_insertion_detour_candidate" not in e.reasons for e in report.entries)


def test_empty_and_invalid_matrix():
    with pytest.raises(ValueError):
        screen_adverse_deliveries([], np.zeros((1, 1)))
    with pytest.raises(ValueError):
        screen_adverse_deliveries([Delivery('a', 10, 106)], np.zeros((1, 1)))


def test_no_peer_does_not_invent_isolation():
    report = screen_adverse_deliveries([Delivery('only', 10, 106)], np.zeros((2, 2)))
    assert report.flagged == ()
    assert report.entries[0].nearest_peer_distance_km is None


def test_alignment_is_by_identity_not_csv_order():
    ds = [Delivery('b', 10, 106), Delivery('a', 10, 106)]
    assert align_cluster_ids(ds, {'a': 10, 'b': 20}) == [20, 10]
    with pytest.raises(ValueError, match='Missing'):
        align_cluster_ids(ds, {'a': 10})


def test_alignment_rejects_fractional_cluster_and_duplicates():
    ds = [Delivery('a', 10, 106), Delivery('b', 10, 106)]
    with pytest.raises(ValueError, match='integer'):
        align_cluster_ids(ds, {'a': 1.7, 'b': 2})
    with pytest.raises(ValueError, match='Duplicate'):
        align_cluster_ids([ds[0], ds[0]], {'a': 0})


def test_unreachable_delivery_is_flagged_without_inventing_cost():
    ds = [Delivery('a', 10, 106), Delivery('b', 10, 106)]
    matrix = np.array([[0, 1, np.inf], [1, 0, np.inf], [np.inf, np.inf, 0]])
    report = screen_adverse_deliveries(ds, matrix)
    assert 'unreachable_from_all_peers' in report.entries[1].reasons
    assert report.entries[1].insertion_detour_km is None
