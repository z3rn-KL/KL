"""Align cluster assignments to the exact delivery order used by RL."""
from typing import Mapping, Sequence
from numbers import Integral
from domain import Delivery


def align_cluster_ids(deliveries: Sequence[Delivery], assignments: Mapping[str, int]) -> list[int]:
    """Never depend on CSV row order; reject duplicate/missing assignments upstream."""
    ids = [str(d.delivery_id) for d in deliveries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate delivery IDs")
    missing = [key for key in ids if key not in assignments]
    if missing:
        raise ValueError(f"Missing cluster assignments: {missing}")
    result = []
    for key in ids:
        value = assignments[key]
        if isinstance(value, bool) or not isinstance(value, Integral):
            # CSV readers should validate integer values before invoking this helper.
            raise ValueError(f"Cluster ID for {key} must be an integer: {value!r}")
        result.append(int(value))
    return result
