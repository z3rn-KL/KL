"""Explainable delivery outlier screening for the routing research project.

Spatial DBSCAN noise is evidence of isolation, NOT evidence of extra cost.
Road detour is assessed separately against the other stops.  All road
matrices use the existing convention: index 0 depot, index i+1 delivery i.
"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from domain import Delivery


@dataclass(frozen=True)
class AdverseDelivery:
    delivery_id: str
    dbscan_noise: bool
    nearest_peer_distance_km: float | None
    isolation_ratio: float | None
    insertion_detour_km: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AdverseDeliveryReport:
    entries: tuple[AdverseDelivery, ...]
    # Flags indicate investigation candidates, not measured operating cost.
    isolation_multiplier: float
    detour_threshold_km: float | None

    @property
    def flagged(self) -> tuple[AdverseDelivery, ...]:
        return tuple(e for e in self.entries if e.reasons)


def screen_adverse_deliveries(
    deliveries: Sequence[Delivery],
    distance_km: np.ndarray,
    *,
    dbscan_labels: Sequence[int] | None = None,
    isolation_multiplier: float = 3.0,
    detour_threshold_km: float | None = None,
    depot_included: bool = True,
) -> AdverseDeliveryReport:
    """Screen points using *directed road distances* rather than map distance.

    Nearest peer distance uses min(outbound, inbound) for each pair when
    reachable, reflecting proximity (not round-trip feasibility).  The
    isolation score is nearest-peer distance divided by the median nearest-
    peer distance of the workload. A zero median never produces a fabricated
    infinite score: positive-distance points are isolated candidates.

    The optional detour metric computes the cheapest insertion delta over all
    distinct other nodes i,j (including the depot), i -> point -> j minus
    i -> j. This is a *proxy*, not the actual marginal route cost; it must
    not be used as an operational cost estimate. No detour reason is assigned
    unless the caller supplies a meaningful threshold in km.
    """
    n = len(deliveries)
    if not n:
        raise ValueError("deliveries cannot be empty")
    ids = [str(d.delivery_id) for d in deliveries]
    if len(set(ids)) != n:
        raise ValueError("delivery IDs must be unique")
    if not np.isfinite(isolation_multiplier) or isolation_multiplier <= 1:
        raise ValueError("isolation_multiplier must be finite and > 1")
    if detour_threshold_km is not None and (
        not np.isfinite(detour_threshold_km) or detour_threshold_km < 0
    ):
        raise ValueError("detour_threshold_km must be finite and >= 0")
    matrix = np.asarray(distance_km, dtype=float)
    expected = n + 1 if depot_included else n
    if matrix.shape != (expected, expected):
        raise ValueError("distance matrix shape does not match depot_included setting")
    if not depot_included and detour_threshold_km is not None:
        raise ValueError("detour screening needs an explicit depot and full matrix")
    if np.isnan(matrix).any() or (matrix < 0).any():
        raise ValueError("distance matrix cannot contain NaN/negative distances")
    if dbscan_labels is not None and len(dbscan_labels) != n:
        raise ValueError("DBSCAN labels must match delivery order")

    peers: list[float] = []
    detours: list[float | None] = []
    for node in range(1, n + 1) if depot_included else range(n):
        distances = [
            min(matrix[node, other], matrix[other, node])
            for other in (range(1, n + 1) if depot_included else range(n)) if other != node
        ]
        peers.append(float(min(distances)) if distances else float("nan"))

        candidate_costs = []
        remaining = [i for i in range(n + 1) if i != node] if depot_included else []
        for before in remaining:
            for after in remaining:
                if before == after:
                    continue
                direct = matrix[before, after]
                to_point = matrix[before, node]
                from_point = matrix[node, after]
                if all(np.isfinite(v) for v in (direct, to_point, from_point)):
                    candidate_costs.append(max(0.0, float(to_point + from_point - direct)))
        detours.append(min(candidate_costs) if candidate_costs else None)

    finite_peers = [p for p in peers if np.isfinite(p)]
    baseline = float(np.median(finite_peers)) if finite_peers else None
    entries = []
    for i, d in enumerate(deliveries):
        p = peers[i]
        if baseline is None or not np.isfinite(p):
            ratio = None
        elif baseline == 0:
            ratio = 1.0 if p == 0 else None
        else:
            ratio = p / baseline
        reasons = []
        is_noise = dbscan_labels is not None and int(dbscan_labels[i]) == -1
        if is_noise:
            reasons.append("dbscan_spatial_noise")
        if np.isinf(p):
            reasons.append("unreachable_from_all_peers")
        elif baseline is not None and (
            (baseline == 0 and p > 0)
            or (ratio is not None and ratio >= isolation_multiplier)
        ):
            reasons.append("road_isolation_candidate")
        if (detour_threshold_km is not None and detours[i] is not None
                and detours[i] >= detour_threshold_km):
            reasons.append("high_insertion_detour_candidate")
        entries.append(AdverseDelivery(
            delivery_id=str(d.delivery_id), dbscan_noise=bool(is_noise),
            nearest_peer_distance_km=float(p) if np.isfinite(p) else None,
            isolation_ratio=ratio, insertion_detour_km=detours[i],
            reasons=tuple(reasons),
        ))
    return AdverseDeliveryReport(tuple(entries), isolation_multiplier, detour_threshold_km)
