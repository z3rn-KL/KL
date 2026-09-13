from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class LocalGeoProjection:
    """
    Chuyển latitude/longitude sang hệ tọa độ cục bộ x/y theo km.

    Phép chiếu này dùng cho clustering trong một khu vực tương đối nhỏ
    như TP.HCM. Không dùng thay cho road distance.
    """

    reference_latitude: float
    reference_longitude: float

    EARTH_RADIUS_KM: ClassVar[float] = 6371.0088

    @classmethod
    def from_coordinates(
        cls,
        coordinates: np.ndarray,
    ) -> "LocalGeoProjection":

        cls._validate_coordinates(coordinates)

        return cls(
            reference_latitude=float(
                np.mean(coordinates[:, 0])
            ),
            reference_longitude=float(
                np.mean(coordinates[:, 1])
            ),
        )

    def transform(
        self,
        coordinates: np.ndarray,
    ) -> np.ndarray:

        self._validate_coordinates(coordinates)

        latitudes = coordinates[:, 0]
        longitudes = coordinates[:, 1]

        reference_lat_rad = np.deg2rad(
            self.reference_latitude
        )

        x = (
            self.EARTH_RADIUS_KM
            * np.cos(reference_lat_rad)
            * np.deg2rad(
                longitudes
                - self.reference_longitude
            )
        )

        y = (
            self.EARTH_RADIUS_KM
            * np.deg2rad(
                latitudes
                - self.reference_latitude
            )
        )

        return np.column_stack(
            (
                x,
                y,
            )
        )

    def inverse_transform(
        self,
        projected_coordinates: np.ndarray,
    ) -> np.ndarray:

        projected_coordinates = np.asarray(
            projected_coordinates,
            dtype=np.float64,
        )

        if (
            projected_coordinates.ndim != 2
            or projected_coordinates.shape[1] != 2
        ):
            raise ValueError(
                "Projected coordinates phải có shape (n, 2)."
            )

        x = projected_coordinates[:, 0]
        y = projected_coordinates[:, 1]

        reference_lat_rad = np.deg2rad(
            self.reference_latitude
        )

        latitudes = (
            self.reference_latitude
            + np.rad2deg(
                y / self.EARTH_RADIUS_KM
            )
        )

        longitudes = (
            self.reference_longitude
            + np.rad2deg(
                x
                / (
                    self.EARTH_RADIUS_KM
                    * np.cos(reference_lat_rad)
                )
            )
        )

        return np.column_stack(
            (
                latitudes,
                longitudes,
            )
        )

    @staticmethod
    def _validate_coordinates(
        coordinates: np.ndarray,
    ) -> None:

        coordinates = np.asarray(
            coordinates,
            dtype=np.float64,
        )

        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 2
        ):
            raise ValueError(
                "Coordinates phải có shape (n, 2)."
            )

        if len(coordinates) == 0:
            raise ValueError(
                "Không có tọa độ để xử lý."
            )

        if not np.isfinite(coordinates).all():
            raise ValueError(
                "Coordinates chứa NaN hoặc infinity."
            )

        latitudes = coordinates[:, 0]
        longitudes = coordinates[:, 1]

        if not (
            np.all(
                (-90 <= latitudes)
                & (latitudes <= 90)
            )
        ):
            raise ValueError(
                "Latitude không hợp lệ."
            )

        if not (
            np.all(
                (-180 <= longitudes)
                & (longitudes <= 180)
            )
        ):
            raise ValueError(
                "Longitude không hợp lệ."
            )