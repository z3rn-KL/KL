from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from domain import Cluster, Delivery

from .geo_projection import (
    LocalGeoProjection,
)


@dataclass
class DBSCANClusteringResult:
    """
    Result returned by DBSCAN clustering.

    Important:
    projected coordinates and eps are
    expressed in kilometres.
    """

    clusters: list[Cluster]

    labels: np.ndarray

    projected_points: np.ndarray

    n_clusters: int

    n_noise: int

    noise_ratio: float

    noise_deliveries: list[Delivery]

    eps_km: float

    min_samples: int

    projection: LocalGeoProjection


class DBSCANDeliveryClusterer:
    """
    Density-based clustering for delivery
    locations.

    DBSCAN differs from K-Means because it:

    - does not require the number of
      clusters in advance;
    - can discover irregular cluster shapes;
    - can mark isolated points as noise.

    sklearn DBSCAN uses label -1 for noise.

    Since LocalGeoProjection produces
    coordinates in kilometres, eps must
    also be given in kilometres.
    """

    def __init__(
        self,
        eps_km: float,
        min_samples: int = 5,
    ) -> None:
        if eps_km <= 0.0:
            raise ValueError(
                "eps_km phải > 0."
            )

        if min_samples < 1:
            raise ValueError(
                "min_samples phải >= 1."
            )

        self.eps_km = float(
            eps_km
        )

        self.min_samples = int(
            min_samples
        )

    def fit(
        self,
        deliveries: list[Delivery],
    ) -> DBSCANClusteringResult:
        """
        Cluster deliveries using DBSCAN.

        Noise points receive label -1.
        """

        self._validate_deliveries(
            deliveries
        )

        coordinates = np.array(
            [
                [
                    delivery.latitude,
                    delivery.longitude,
                ]
                for delivery
                in deliveries
            ],
            dtype=np.float64,
        )

        projection = (
            LocalGeoProjection
            .from_coordinates(
                coordinates
            )
        )

        projected_points = (
            projection.transform(
                coordinates
            )
        )

        model = DBSCAN(
            eps=self.eps_km,
            min_samples=self.min_samples,
            metric="euclidean",
        )

        labels = model.fit_predict(
            projected_points
        )

        cluster_labels = sorted(
            int(label)
            for label
            in np.unique(
                labels
            )
            if int(label) != -1
        )

        clusters = (
            self._build_clusters(
                deliveries=deliveries,
                labels=labels,
                projected_points=(
                    projected_points
                ),
                cluster_labels=(
                    cluster_labels
                ),
                projection=projection,
            )
        )

        noise_deliveries = [
            delivery
            for delivery, label
            in zip(
                deliveries,
                labels,
                strict=True,
            )
            if int(label) == -1
        ]

        n_noise = len(
            noise_deliveries
        )

        n_clusters = len(
            cluster_labels
        )

        noise_ratio = (
            n_noise
            / len(
                deliveries
            )
        )

        return DBSCANClusteringResult(
            clusters=clusters,
            labels=labels,
            projected_points=(
                projected_points
            ),
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_ratio=float(
                noise_ratio
            ),
            noise_deliveries=(
                noise_deliveries
            ),
            eps_km=self.eps_km,
            min_samples=(
                self.min_samples
            ),
            projection=projection,
        )

    def _build_clusters(
        self,
        deliveries: list[Delivery],
        labels: np.ndarray,
        projected_points: np.ndarray,
        cluster_labels: list[int],
        projection: LocalGeoProjection,
    ) -> list[Cluster]:
        """
        Convert DBSCAN labels into domain
        Cluster objects.

        Noise label -1 is intentionally not
        converted into a normal Cluster.
        """

        clusters: list[Cluster] = []

        for cluster_id in (
            cluster_labels
        ):
            cluster_indices = np.where(
                labels
                == cluster_id
            )[
                0
            ]

            cluster_deliveries = [
                deliveries[
                    int(index)
                ]
                for index
                in cluster_indices
            ]

            cluster_points = (
                projected_points[
                    cluster_indices
                ]
            )

            projected_centroid = (
                np.mean(
                    cluster_points,
                    axis=0,
                )
                .reshape(
                    1,
                    2,
                )
            )

            centroid_coordinates = (
                projection
                .inverse_transform(
                    projected_centroid
                )
            )

            centroid_latitude = float(
                centroid_coordinates[
                    0,
                    0,
                ]
            )

            centroid_longitude = float(
                centroid_coordinates[
                    0,
                    1,
                ]
            )

            cluster = Cluster(
                cluster_id=cluster_id,
                deliveries=(
                    cluster_deliveries
                ),
                centroid_latitude=(
                    centroid_latitude
                ),
                centroid_longitude=(
                    centroid_longitude
                ),
            )

            clusters.append(
                cluster
            )

        return clusters

    def _validate_deliveries(
        self,
        deliveries: list[Delivery],
    ) -> None:
        if not deliveries:
            raise ValueError(
                "Danh sách Delivery rỗng."
            )

        for delivery in deliveries:
            if not (
                -90
                <= delivery.latitude
                <= 90
            ):
                raise ValueError(
                    f"Delivery "
                    f"{delivery.delivery_id} "
                    "có latitude không hợp lệ."
                )

            if not (
                -180
                <= delivery.longitude
                <= 180
            ):
                raise ValueError(
                    f"Delivery "
                    f"{delivery.delivery_id} "
                    "có longitude không hợp lệ."
                )