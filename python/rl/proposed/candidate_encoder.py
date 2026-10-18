from __future__ import annotations

from dataclasses import dataclass
import math

from .environment import (
    CandidateInfo,
    ProposedRoutingEnvironment,
)


FEATURE_NAMES = (
    "elapsed_ratio",
    "remaining_ratio",
    "distance_ratio",
    "travel_time_ratio",
    "slack_ratio",
    "deadline_valid",
    "lateness_ratio",
    "cluster_switch",
)


@dataclass(frozen=True)
class CandidateEncoderConfig:
    """
    Cấu hình biểu diễn state + candidate.

    slack_clip:
        Giới hạn slack sau khi normalize.

    elapsed_clip:
        Giới hạn elapsed ratio để tránh giá trị quá lớn.
    """

    slack_clip: float = 2.0
    elapsed_clip: float = 2.0

    def __post_init__(self) -> None:
        if self.slack_clip <= 0.0:
            raise ValueError(
                "slack_clip must be > 0."
            )

        if self.elapsed_clip <= 0.0:
            raise ValueError(
                "elapsed_clip must be > 0."
            )


@dataclass(frozen=True)
class EncodedCandidate:
    """
    Một action và feature vector tương ứng.
    """

    action: int
    features: tuple[float, ...]


@dataclass(frozen=True)
class EncodedCandidateSet:
    """
    Toàn bộ candidate hợp lệ tại state hiện tại.
    """

    candidates: tuple[EncodedCandidate, ...]

    @property
    def actions(self) -> tuple[int, ...]:
        return tuple(
            candidate.action
            for candidate in self.candidates
        )

    @property
    def feature_matrix(
        self,
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            candidate.features
            for candidate in self.candidates
        )

    @property
    def feature_dimension(self) -> int:
        return len(FEATURE_NAMES)


def _clip(
    value: float,
    lower: float,
    upper: float,
) -> float:
    return max(
        lower,
        min(value, upper),
    )


def _validate_finite(
    features: tuple[float, ...],
) -> None:
    if not all(
        math.isfinite(value)
        for value in features
    ):
        raise ValueError(
            "Encoded candidate contains "
            "non-finite feature values."
        )


class CandidateEncoder:
    """
    Chuyển state + candidate thành vector số
    cho Proposed RL agent.

    Feature vector:

        0. elapsed_ratio
        1. remaining_ratio
        2. distance_ratio
        3. travel_time_ratio
        4. slack_ratio
        5. deadline_valid
        6. lateness_ratio
        7. cluster_switch

    Không đưa raw cluster_id vào neural network
    vì nhãn K-Means chỉ mang tính định danh.
    """

    def __init__(
        self,
        config: CandidateEncoderConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else CandidateEncoderConfig()
        )

    @property
    def feature_names(
        self,
    ) -> tuple[str, ...]:
        return FEATURE_NAMES

    @property
    def feature_dimension(self) -> int:
        return len(FEATURE_NAMES)

    def encode_candidate(
        self,
        *,
        env: ProposedRoutingEnvironment,
        candidate: CandidateInfo,
    ) -> EncodedCandidate:
        objective_config = env.objective_config
        context = env.objective_context

        # ---------------------------------
        # 1. Global routing state
        # ---------------------------------

        elapsed_denominator = (
            context.route_leg_count
            * objective_config.travel_time_ref_minutes
        )

        elapsed_ratio = (
            env.elapsed_time_minutes
            / elapsed_denominator
        )

        elapsed_ratio = _clip(
            elapsed_ratio,
            0.0,
            self.config.elapsed_clip,
        )

        remaining_ratio = (
            env.state.remaining_deliveries
            / env.number_of_deliveries
        )

        # ---------------------------------
        # 2. Candidate travel information
        # ---------------------------------

        distance_ratio = (
            candidate.distance_km
            / objective_config.distance_ref_km
        )

        travel_time_ratio = (
            candidate.travel_time_minutes
            / objective_config.travel_time_ref_minutes
        )

        # ---------------------------------
        # 3. Deadline information
        # ---------------------------------

        deadline_valid = float(
            candidate.deadline_valid
        )

        if candidate.slack_minutes is None:
            slack_ratio = 0.0
        else:
            slack_ratio = (
                candidate.slack_minutes
                / objective_config.lateness_ref_minutes
            )

            slack_ratio = _clip(
                slack_ratio,
                -self.config.slack_clip,
                self.config.slack_clip,
            )

        lateness_ratio = (
            candidate.lateness_minutes
            / objective_config.lateness_ref_minutes
        )

        # ---------------------------------
        # 4. Spatial cluster relation
        # ---------------------------------

        cluster_switch = float(
            candidate.cluster_switched
        )

        features = (
            float(elapsed_ratio),
            float(remaining_ratio),
            float(distance_ratio),
            float(travel_time_ratio),
            float(slack_ratio),
            float(deadline_valid),
            float(lateness_ratio),
            float(cluster_switch),
        )

        _validate_finite(features)

        return EncodedCandidate(
            action=candidate.action,
            features=features,
        )

    def encode_action(
        self,
        *,
        env: ProposedRoutingEnvironment,
        action: int,
    ) -> EncodedCandidate:
        candidate = env.candidate_info(
            action
        )

        return self.encode_candidate(
            env=env,
            candidate=candidate,
        )

    def encode_available_actions(
        self,
        env: ProposedRoutingEnvironment,
    ) -> EncodedCandidateSet:
        candidates = tuple(
            self.encode_action(
                env=env,
                action=action,
            )
            for action in env.available_actions()
        )

        return EncodedCandidateSet(
            candidates=candidates
        )