from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TravelEfficiencyWeights:
    """
    Experimental weights used to combine
    road distance and estimated travel time.

    Travel time receives slightly more weight
    because the thesis also considers
    on-time delivery performance.

    These are experimental parameters,
    not values supplied by XeDu or OSM.
    """

    distance: float = 0.40
    travel_time: float = 0.60

    def validate(self) -> None:
        weights = [
            self.distance,
            self.travel_time,
        ]

        if any(
            weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Travel-efficiency weights "
                "cannot be negative."
            )

        total = sum(
            weights
        )

        if total <= 0:
            raise ValueError(
                "At least one travel-efficiency "
                "weight must be greater than 0."
            )

        if abs(
            total - 1.0
        ) > 1e-9:
            raise ValueError(
                "Travel-efficiency weights "
                "must sum to 1.0."
            )


@dataclass(frozen=True)
class TravelEfficiencyResult:
    """
    Detailed travel-efficiency scores
    for a set of candidate deliveries.
    """

    distance_efficiency: np.ndarray

    time_efficiency: np.ndarray

    travel_efficiency: np.ndarray


class TravelEfficiencyScorer:
    """
    Convert road distance and travel time
    into normalized efficiency scores.

    Higher score:
        more efficient candidate.

    Lower score:
        less efficient candidate.

    The scorer is relative to the current
    candidate set.

    Example:

        candidate A: 1 km
        candidate B: 3 km
        candidate C: 5 km

    A receives the highest distance efficiency,
    while C receives the lowest.
    """

    def __init__(
        self,
        weights: TravelEfficiencyWeights | None = None,
    ) -> None:
        self.weights = (
            weights
            or TravelEfficiencyWeights()
        )

        self.weights.validate()

    def score_candidates(
        self,
        distance_km,
        travel_time_minutes,
    ) -> TravelEfficiencyResult:
        """
        Score multiple candidate destinations.

        Parameters
        ----------
        distance_km:
            Road distance from the current
            location to every candidate.

        travel_time_minutes:
            Estimated road travel time from
            the current location to every
            candidate.

        Returns
        -------
        TravelEfficiencyResult

        Notes
        -----
        NaN or infinite values are considered
        unreachable/invalid and receive 0.
        """

        distances = np.asarray(
            distance_km,
            dtype=float,
        )

        travel_times = np.asarray(
            travel_time_minutes,
            dtype=float,
        )

        self._validate_arrays(
            distances,
            travel_times,
        )

        distance_efficiency = (
            self._inverse_minmax(
                distances
            )
        )

        time_efficiency = (
            self._inverse_minmax(
                travel_times
            )
        )

        combined = (
            self.weights.distance
            * distance_efficiency
            + self.weights.travel_time
            * time_efficiency
        )

        valid_mask = (
            np.isfinite(
                distances
            )
            & np.isfinite(
                travel_times
            )
        )

        combined = np.where(
            valid_mask,
            combined,
            0.0,
        )

        combined = np.clip(
            combined,
            0.0,
            1.0,
        )

        return TravelEfficiencyResult(
            distance_efficiency=(
                distance_efficiency
            ),
            time_efficiency=(
                time_efficiency
            ),
            travel_efficiency=(
                combined
            ),
        )

    @staticmethod
    def _inverse_minmax(
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Convert smaller-is-better values
        into larger-is-better scores.

        Formula:

            score =
                1 -
                (x - min)
                / (max - min)

        Minimum value -> 1
        Maximum value -> 0

        Invalid values -> 0

        If all valid values are equal,
        they are equally efficient and
        therefore receive 1.
        """

        result = np.zeros(
            values.shape,
            dtype=float,
        )

        valid_mask = np.isfinite(
            values
        )

        valid_values = values[
            valid_mask
        ]

        if valid_values.size == 0:
            return result

        minimum = float(
            valid_values.min()
        )

        maximum = float(
            valid_values.max()
        )

        if np.isclose(
            minimum,
            maximum,
        ):
            result[
                valid_mask
            ] = 1.0

            return result

        normalized = (
            valid_values
            - minimum
        ) / (
            maximum
            - minimum
        )

        efficiency = (
            1.0
            - normalized
        )

        result[
            valid_mask
        ] = efficiency

        return np.clip(
            result,
            0.0,
            1.0,
        )

    @staticmethod
    def _validate_arrays(
        distances: np.ndarray,
        travel_times: np.ndarray,
    ) -> None:
        if distances.ndim != 1:
            raise ValueError(
                "distance_km must be "
                "a one-dimensional array."
            )

        if travel_times.ndim != 1:
            raise ValueError(
                "travel_time_minutes must be "
                "a one-dimensional array."
            )

        if len(
            distances
        ) == 0:
            raise ValueError(
                "Candidate arrays cannot be empty."
            )

        if (
            distances.shape
            != travel_times.shape
        ):
            raise ValueError(
                "Distance and travel-time arrays "
                "must have the same shape."
            )

        finite_distances = distances[
            np.isfinite(
                distances
            )
        ]

        finite_times = travel_times[
            np.isfinite(
                travel_times
            )
        ]

        if np.any(
            finite_distances < 0
        ):
            raise ValueError(
                "Road distance cannot be negative."
            )

        if np.any(
            finite_times < 0
        ):
            raise ValueError(
                "Travel time cannot be negative."
            )