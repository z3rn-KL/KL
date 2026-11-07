from __future__ import annotations

from .candidate_encoder import (
    CandidateEncoder,
    EncodedCandidate,
    FEATURE_NAMES,
)
from .environment import (
    CandidateInfo,
    ProposedRoutingEnvironment,
)


LOOKAHEAD_FEATURE_NAMES = (
    *FEATURE_NAMES,
    "return_ratio",
    "detour_ratio",
    "nearest_remaining_ratio",
    "mean_remaining_ratio",
    "same_cluster_remaining_ratio",
)


class LookaheadCandidateEncoder(CandidateEncoder):
    """
    Base features plus what the candidate leaves behind:
    the way back to the depot and the deliveries still to do.

    The base encoder only describes the next hop, so the Q-network
    cannot tell a candidate that strands the route from one that
    keeps it compact.
    """

    @property
    def feature_names(self) -> tuple[str, ...]:
        return LOOKAHEAD_FEATURE_NAMES

    @property
    def feature_dimension(self) -> int:
        return len(LOOKAHEAD_FEATURE_NAMES)

    def encode_candidate(
        self,
        *,
        env: ProposedRoutingEnvironment,
        candidate: CandidateInfo,
    ) -> EncodedCandidate:
        base = super().encode_candidate(
            env=env,
            candidate=candidate,
        )

        time_ref = env.objective_config.travel_time_ref_minutes
        times = env.travel_time_matrix_minutes
        node = candidate.node_index

        return_time = times[node][0]

        # Extra time versus going home directly from here.
        detour = (
            candidate.travel_time_minutes
            + return_time
            - times[env.current_node][0]
        )

        others = [
            action
            for action in env.available_actions()
            if action != candidate.action
        ]

        if others:
            gaps = [times[node][other + 1] for other in others]

            nearest = min(gaps)
            mean_gap = sum(gaps) / len(gaps)

            same_cluster = sum(
                env.cluster_ids[other]
                == env.cluster_ids[candidate.action]
                for other in others
            ) / len(others)
        else:
            nearest = mean_gap = same_cluster = 0.0

        features = (
            *base.features,
            return_time / time_ref,
            detour / time_ref,
            nearest / time_ref,
            mean_gap / time_ref,
            same_cluster,
        )

        return EncodedCandidate(
            action=base.action,
            features=features,
        )
