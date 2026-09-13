from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from domain import Cluster, Delivery

from .geo_projection import LocalGeoProjection


@dataclass
class KMeansClusteringResult:
    clusters: list[Cluster]

    labels: np.ndarray

    projected_points: np.ndarray
    projected_centroids: np.ndarray

    inertia: float

    n_clusters: int

    projection: LocalGeoProjection


class KMeansDeliveryClusterer:
    def __init__(
        self,
        n_clusters: int,
        random_state: int = 42,
        n_init: int = 20,
    ):
        if n_clusters < 1:
            raise ValueError(
                "n_clusters phải >= 1."
            )

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(
        self,
        deliveries: list[Delivery],
    ) -> KMeansClusteringResult:

        self._validate_deliveries(
            deliveries
        )

        coordinates = np.array(
            [
                [
                    delivery.latitude,
                    delivery.longitude,
                ]
                for delivery in deliveries
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

        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        labels = model.fit_predict(
            projected_points
        )

        centroid_coordinates = (
            projection.inverse_transform(
                model.cluster_centers_
            )
        )

        clusters = self._build_clusters(
            deliveries=deliveries,
            labels=labels,
            centroid_coordinates=centroid_coordinates,
        )

        return KMeansClusteringResult(
            clusters=clusters,
            labels=labels,
            projected_points=projected_points,
            projected_centroids=model.cluster_centers_,
            inertia=float(model.inertia_),
            n_clusters=self.n_clusters,
            projection=projection,
        )

    def _build_clusters(
        self,
        deliveries: list[Delivery],
        labels: np.ndarray,
        centroid_coordinates: np.ndarray,
    ) -> list[Cluster]:

        clusters: list[Cluster] = []

        for cluster_id in range(
            self.n_clusters
        ):
            cluster_deliveries = [
                delivery
                for delivery, label
                in zip(
                    deliveries,
                    labels,
                )
                if int(label) == cluster_id
            ]

            centroid_latitude = float(
                centroid_coordinates[
                    cluster_id,
                    0,
                ]
            )

            centroid_longitude = float(
                centroid_coordinates[
                    cluster_id,
                    1,
                ]
            )

            cluster = Cluster(
                cluster_id=cluster_id,
                deliveries=cluster_deliveries,
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

        if self.n_clusters > len(
            deliveries
        ):
            raise ValueError(
                "n_clusters không được lớn hơn "
                "số Delivery."
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