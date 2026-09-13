from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score

from .kmeans_clusterer import (
    KMeansClusteringResult,
)


@dataclass(frozen=True)
class ClusteringMetrics:
    n_clusters: int

    inertia: float
    silhouette: float | None

    min_cluster_size: int
    max_cluster_size: int
    mean_cluster_size: float

    cluster_size_cv: float

    total_weight_kg: float
    min_cluster_weight_kg: float
    max_cluster_weight_kg: float
    mean_cluster_weight_kg: float

    cluster_weight_cv: float


def evaluate_clustering(
    result: KMeansClusteringResult,
) -> ClusteringMetrics:

    clusters = result.clusters

    sizes = np.array(
        [
            cluster.size()
            for cluster in clusters
        ],
        dtype=np.float64,
    )

    weights = np.array(
        [
            cluster.total_weight()
            for cluster in clusters
        ],
        dtype=np.float64,
    )

    silhouette = None

    if (
        result.n_clusters > 1
        and result.n_clusters
        < len(result.projected_points)
    ):
        silhouette = float(
            silhouette_score(
                result.projected_points,
                result.labels,
                metric="euclidean",
            )
        )

    size_mean = float(
        np.mean(sizes)
    )

    weight_mean = float(
        np.mean(weights)
    )

    size_cv = (
        float(
            np.std(sizes)
            / size_mean
        )
        if size_mean > 0
        else 0.0
    )

    weight_cv = (
        float(
            np.std(weights)
            / weight_mean
        )
        if weight_mean > 0
        else 0.0
    )

    return ClusteringMetrics(
        n_clusters=result.n_clusters,

        inertia=result.inertia,
        silhouette=silhouette,

        min_cluster_size=int(
            np.min(sizes)
        ),
        max_cluster_size=int(
            np.max(sizes)
        ),
        mean_cluster_size=size_mean,

        cluster_size_cv=size_cv,

        total_weight_kg=float(
            np.sum(weights)
        ),

        min_cluster_weight_kg=float(
            np.min(weights)
        ),
        max_cluster_weight_kg=float(
            np.max(weights)
        ),
        mean_cluster_weight_kg=weight_mean,

        cluster_weight_cv=weight_cv,
    )