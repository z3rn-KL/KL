"""Reproducible, non-destructive offline checks for backend research outputs.

Run: PYTHONPATH=python python -m experiments.backend_completion_audit
This intentionally does not claim retraining equivalence.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from clustering.adverse_deliveries import screen_adverse_deliveries
from domain import Delivery

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
OUT = ROOT / 'audit_outputs'


def adverse_sample_audit(output_dir=OUT):
    """Use only the saved 15-stop road-distance sample; no invented depot."""
    matrix = pd.read_csv(RESULTS / 'xedu_sample_distance_matrix_km.csv', index_col=0)
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError('Duplicate delivery identifiers in sample road matrix')
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    if set(matrix.index) != set(matrix.columns):
        raise ValueError('Sample road matrix row/column IDs differ')
    matrix = matrix.loc[matrix.index, matrix.index]
    labels = pd.read_csv(RESULTS / 'dbscan_best_labels.csv', dtype={'delivery_id': str})
    if labels.delivery_id.duplicated().any():
        raise ValueError('Duplicate identifiers in DBSCAN labels')
    lookup = labels.set_index('delivery_id')
    if not set(matrix.index).issubset(lookup.index):
        raise ValueError('Missing DBSCAN labels for road sample')
    ds = [Delivery(str(i), 0, 0) for i in matrix.index]
    report = screen_adverse_deliveries(ds, matrix.to_numpy(dtype=float),
        dbscan_labels=lookup.loc[matrix.index, 'dbscan_label'].tolist(),
        depot_included=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(delivery_id=x.delivery_id, dbscan_noise=x.dbscan_noise,
        nearest_peer_distance_km=x.nearest_peer_distance_km,
        isolation_ratio=x.isolation_ratio, insertion_detour_km=x.insertion_detour_km,
        reasons=';'.join(x.reasons)) for x in report.entries]
    pd.DataFrame(rows).to_csv(output_dir / 'adverse_road_sample.csv', index=False)
    return {'sample_size': len(rows), 'flagged': len(report.flagged),
        'note': 'peer-to-peer directed road distances; no depot, so detour estimates are unavailable'}


def benchmark_consistency_audit(output_dir=OUT):
    """Validate prior CSV arithmetic and joins, NOT scientific reproduction."""
    paired = pd.read_csv(RESULTS / 'final_kmeans_guided_dqn_validation_paired.csv')
    manifest = pd.read_csv(RESULTS / 'dataset_algorithm_workloads.csv')
    problems = []
    for name, df in [('paired', paired), ('manifest', manifest)]:
        if df.workload_id.isna().any() or df.workload_id.duplicated().any():
            problems.append(f'{name}: missing/duplicate workload_id')
    if not set(paired.workload_id).issubset(set(manifest.workload_id)):
        problems.append('Paired workload IDs missing from manifest')
    for prefix in ('baseline', 'guided'):
        distance = pd.to_numeric(paired[f'{prefix}_distance_km'], errors='coerce')
        time = pd.to_numeric(paired[f'{prefix}_travel_time_minutes'], errors='coerce')
        if (distance < 0).any() or (time < 0).any() or distance.isna().any() or time.isna().any():
            problems.append(f'{prefix}: invalid distance/travel time')
        count = paired[f'{prefix}_on_time_deliveries'] + paired[f'{prefix}_late_deliveries']
        if not np.allclose(count, paired.number_of_deliveries, atol=0, rtol=0):
            problems.append(f'{prefix}: inconsistent on-time + late count')
        rate = paired[f'{prefix}_on_time_deliveries'] / paired.number_of_deliveries
        if not np.allclose(rate, paired[f'{prefix}_on_time_rate'], atol=1e-8, rtol=0):
            problems.append(f'{prefix}: inconsistent SLA rate')
    for key, base, guided in [('distance_improvement_pct', 'baseline_distance_km', 'guided_distance_km'),
                               ('time_improvement_pct', 'baseline_travel_time_minutes', 'guided_travel_time_minutes')]:
        b = paired[base].to_numpy(dtype=float)
        g = paired[guided].to_numpy(dtype=float)
        expected = np.where(b != 0, (b-g)/np.where(b==0,1,b)*100, np.nan)
        actual = paired[key].to_numpy(dtype=float)
        mismatch = np.isfinite(expected) & ~np.isclose(expected, actual, atol=1e-5, rtol=0)
        if mismatch.any():
            problems.append(f'{key}: {int(mismatch.sum())} arithmetic mismatches')
    baseline_switch_missing = int(paired.baseline_cluster_switches.isna().sum())
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {'manifest_workloads': len(manifest), 'paired_workloads': len(paired),
              'missing_vanilla_route_switch_metrics': baseline_switch_missing,
              'issues': problems,
              'limitations': ['CSV arithmetic only; no DQN retraining',
                             'baseline cluster switching unavailable without original routes',
                             'cannot infer model superiority from these checks']}
    (output_dir / 'benchmark_consistency.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    print(json.dumps({'adverse': adverse_sample_audit(),
                      'benchmark': benchmark_consistency_audit()}, indent=2))

if __name__ == '__main__':
    main()
