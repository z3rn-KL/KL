"""Read-only endpoints for RL stability and clustering-quality artifacts.

These expose CSV files that already exist in ``results/`` so the web UI can
show (a) Q-Learning stability across seeds / hyper-parameters and
(b) how K (K-Means) and eps/min_samples (DBSCAN) were chosen.
Nothing is recomputed here.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.config import RESULTS_DIR

router = APIRouter(
    prefix="/api/results",
    tags=["Diagnostics"],
)

STABILITY_OVERVIEW = RESULTS_DIR / "q_learning_stability_overview.csv"
STABILITY_SEEDS = RESULTS_DIR / "q_learning_stability_seed_summary.csv"
HYPERPARAMETERS = RESULTS_DIR / "q_learning_hyperparameter_summary.csv"
KMEANS_ELBOW = RESULTS_DIR / "thesis_kmeans_elbow_summary.csv"
DBSCAN_GRID = RESULTS_DIR / "dbscan_grid_results.csv"


@lru_cache(maxsize=8)
def _read(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Result file not found: {path.name}",
        )
    return pd.read_csv(path)


def _clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _records(df: pd.DataFrame, drop: tuple[str, ...] = ()) -> list[dict]:
    df = df.drop(columns=[c for c in drop if c in df.columns])
    return [
        {str(k): _clean(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


@router.get("/stability")
def q_learning_stability() -> dict:
    """Q-Learning across random seeds and hyper-parameter configurations.

    Scope note: these runs use ONE small workload (5 deliveries) where the
    optimum can be found by exhaustive search, so they show stability of the
    learner, not performance at dataset scale.
    """

    overview = _read(str(STABILITY_OVERVIEW))
    seeds = _read(str(STABILITY_SEEDS))
    hyper = _read(str(HYPERPARAMETERS))

    # The route strings are long; the UI only needs the counts.
    return {
        "scope_note": (
            "Dữ liệu ổn định được chạy trên một workload nhỏ (5 đơn) để so "
            "với nghiệm tối ưu vét cạn; không đại diện cho quy mô toàn bộ "
            "bộ dữ liệu."
        ),
        "overview": _records(
            overview,
            drop=("exact_best_route", "most_common_greedy_route"),
        )[0],
        "seeds": _records(seeds, drop=("greedy_route",)),
        "hyperparameters": _records(hyper),
    }


@router.get("/clustering-quality")
def clustering_quality() -> dict:
    """K-Means elbow/silhouette table and the DBSCAN eps/min_samples grid."""

    kmeans = _read(str(KMEANS_ELBOW))
    dbscan = _read(str(DBSCAN_GRID))

    return {
        "kmeans": _records(kmeans),
        "dbscan": _records(dbscan),
        "selected": {
            "kmeans_k": 10,
            "dbscan_eps_km": 2.0,
            "dbscan_min_samples": 8,
        },
    }
