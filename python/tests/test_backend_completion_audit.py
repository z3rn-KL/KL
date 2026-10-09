import numpy as np
import pytest
from domain import Delivery
from clustering.adverse_deliveries import screen_adverse_deliveries
from experiments.backend_completion_audit import adverse_sample_audit, benchmark_consistency_audit


def test_peer_only_matrix_does_not_invent_depot():
    ds = [Delivery(str(i), 10, 106) for i in range(3)]
    matrix = np.array([[0, 1, 20], [1, 0, 20], [20, 20, 0]], dtype=float)
    report = screen_adverse_deliveries(ds, matrix, depot_included=False)
    assert len(report.entries) == 3
    assert all(e.insertion_detour_km is None for e in report.entries)


def test_peer_only_rejects_detour_threshold():
    ds = [Delivery(str(i), 10, 106) for i in range(2)]
    with pytest.raises(ValueError, match='depot'):
        screen_adverse_deliveries(ds, np.zeros((2, 2)), depot_included=False, detour_threshold_km=2)


def test_saved_experiments_auditable(tmp_path):
    assert adverse_sample_audit(tmp_path)['sample_size'] == 15
    audit = benchmark_consistency_audit(tmp_path)
    assert audit['paired_workloads'] > 0
    assert audit['issues'] == []
