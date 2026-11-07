from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import pandas as pd

from .environment import ShiftEnvironment, ShiftOrder
from .objective import FeeTariff, ShiftObjectiveConfig


def split_days(
    days: Sequence,
    seed: int = 42,
    fractions: tuple[float, float] = (0.6, 0.2),
) -> dict[str, set]:
    """
    Train / validation / test split by calendar day, so no day (and no
    order) appears in two splits.
    """

    ordered = sorted(set(days))
    random.Random(seed).shuffle(ordered)

    train_end = int(len(ordered) * fractions[0])
    validation_end = train_end + int(len(ordered) * fractions[1])

    return {
        "train": set(ordered[:train_end]),
        "validation": set(ordered[train_end:validation_end]),
        "test": set(ordered[validation_end:]),
    }


class ShiftSampler:
    """
    Draws shifts of real orders: n orders from one K-Means cluster (one
    cluster is handed to one shipper), with deadlines taken from the
    original lead time and order values from an assumed fee tariff.

    pool needs: id, node (row in the global road matrix), lead_minutes,
    shippingDistance (metres), serviceType, cluster_id, local_day.
    """

    def __init__(
        self,
        pool: pd.DataFrame,
        distance_km: np.ndarray,
        travel_minutes: np.ndarray,
        *,
        tariff: FeeTariff | None = None,
        config: ShiftObjectiveConfig | None = None,
    ) -> None:
        tariff = tariff if tariff is not None else FeeTariff()

        self.pool = pool.assign(
            value=[
                tariff.fee(meters / 1000.0, service)
                for meters, service in zip(
                    pool["shippingDistance"], pool["serviceType"]
                )
            ]
        ).reset_index(drop=True)

        self.distance_km = distance_km
        self.travel_minutes = travel_minutes
        self.config = config if config is not None else ShiftObjectiveConfig()
        self._eligible_cache: dict = {}

    def eligible(self, days: set, n: int) -> list[pd.DataFrame]:
        key = (id(days), n)

        if key not in self._eligible_cache:
            subset = self.pool[self.pool["local_day"].isin(days)]

            self._eligible_cache[key] = [
                group
                for _, group in subset.groupby("cluster_id")
                if len(group) >= n
            ]

        return self._eligible_cache[key]

    def sample(
        self,
        rng: random.Random,
        days: set,
        n: int,
    ) -> ShiftEnvironment:
        groups = self.eligible(days, n)

        if not groups:
            raise ValueError(f"No cluster has {n} orders in these days.")

        weights = [len(group) for group in groups]
        group = rng.choices(groups, weights=weights)[0]

        picked = group.iloc[rng.sample(range(len(group)), n)]

        nodes = [0, *picked["node"].astype(int).tolist()]
        index = np.ix_(nodes, nodes)

        orders = [
            ShiftOrder(
                order_id=str(row.id),
                value=float(row.value),
                deadline_minutes=(
                    None
                    if pd.isna(row.lead_minutes)
                    else float(row.lead_minutes)
                ),
            )
            for row in picked.itertuples()
        ]

        return ShiftEnvironment(
            distance_km=self.distance_km[index],
            travel_minutes=self.travel_minutes[index],
            orders=orders,
            config=self.config,
        )
