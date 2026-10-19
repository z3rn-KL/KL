from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl.proposed.objective import (
    ObjectiveConfig,
    ObjectiveContext,
    compute_route_objective,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

SPLIT_PATH = (
    RESULTS_DIR
    / "proposed_rl_workload_split.csv"
)

BASELINE_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_all_results.csv"
)

PROPOSED_DETAILS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_test_details.csv"
)

TEST_CACHE_DIR = (
    RESULTS_DIR
    / "proposed_rl_test_road_cache"
)

DETAILS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_locked_objective_details.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_locked_objective_summary.csv"
)

PAIRWISE_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_locked_objective_pairwise.csv"
)

FINDINGS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_locked_objective_findings.txt"
)


# ============================================================
# FINAL LOCKED DESIGN
# ============================================================

EXPECTED_TEST_WORKLOADS = 16
EXPECTED_TEST_DELIVERIES = 131

PROPOSED_NAME = "Proposed RL"

ALGORITHM_ORDER = (
    "NN",
    "CW",
    "Q-Learning",
    "SARSA",
    "DQN",
    PROPOSED_NAME,
)


LOCKED_OBJECTIVE = ObjectiveConfig(
    distance_weight=0.375,
    travel_time_weight=0.375,
    late_delivery_weight=0.10,
    lateness_weight=0.10,
    cluster_switch_weight=0.05,
)


# ============================================================
# COLUMN HELPERS
# ============================================================

def resolve_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    if required:
        raise ValueError(
            "Could not find required column. "
            f"Tried {candidates}. "
            f"Available columns: "
            f"{list(dataframe.columns)}"
        )

    return None


def normalize_algorithm_name(
    value: object,
) -> str:
    raw = str(
        value
    ).strip()

    key = (
        raw.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    aliases = {
        "nn": "NN",
        "nearestneighbor": "NN",
        "nearestneighbour": "NN",

        "cw": "CW",
        "clarkewright": "CW",
        "clarkewrightsavings": "CW",

        "ql": "Q-Learning",
        "qlearning": "Q-Learning",
        "qlearn": "Q-Learning",

        "sarsa": "SARSA",

        "dqn": "DQN",
        "deepqnetwork": "DQN",
    }

    if key not in aliases:
        raise ValueError(
            f"Unknown algorithm name: {raw!r}"
        )

    return aliases[
        key
    ]


# ============================================================
# FROZEN TEST SPLIT
# ============================================================

def load_test_split() -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing split file: "
            f"{SPLIT_PATH}"
        )

    split = pd.read_csv(
        SPLIT_PATH
    )

    required = {
        "workload_id",
        "number_of_deliveries",
        "split",
    }

    missing = (
        required
        - set(split.columns)
    )

    if missing:
        raise ValueError(
            "Split missing columns: "
            f"{sorted(missing)}"
        )

    split[
        "workload_id"
    ] = split[
        "workload_id"
    ].astype(int)

    split[
        "number_of_deliveries"
    ] = split[
        "number_of_deliveries"
    ].astype(int)

    test = (
        split[
            split[
                "split"
            ].astype(str)
            == "test"
        ]
        .copy()
        .sort_values(
            [
                "number_of_deliveries",
                "workload_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(test)
        != EXPECTED_TEST_WORKLOADS
    ):
        raise ValueError(
            "Expected exactly "
            f"{EXPECTED_TEST_WORKLOADS} "
            "Test workloads."
        )

    total_deliveries = int(
        test[
            "number_of_deliveries"
        ].sum()
    )

    if (
        total_deliveries
        != EXPECTED_TEST_DELIVERIES
    ):
        raise ValueError(
            "Expected exactly "
            f"{EXPECTED_TEST_DELIVERIES} "
            "Test deliveries, "
            f"found {total_deliveries}."
        )

    return test


# ============================================================
# TEST WORKLOAD METADATA
# ============================================================

def load_workload_metadata(
    workload_id: int,
) -> dict[str, Any]:
    """
    Load canonical delivery ordering and K-Means
    cluster assignments from the frozen Test cache.
    """

    path = (
        TEST_CACHE_DIR
        / f"workload_{workload_id}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Test cache: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        n = int(
            data[
                "number_of_deliveries"
            ][0]
        )

        delivery_ids = tuple(
            str(value)
            for value in data[
                "delivery_ids"
            ].tolist()
        )

        cluster_ids = tuple(
            int(value)
            for value in data[
                "cluster_ids"
            ].tolist()
        )

        depot_node = int(
            data[
                "depot_node"
            ][0]
        )

        deadline_valid = tuple(
            bool(value)
            for value in data[
                "deadline_valid"
            ].tolist()
        )

    if len(
        delivery_ids
    ) != n:
        raise ValueError(
            f"Delivery ID count mismatch "
            f"for workload {workload_id}."
        )

    if len(
        cluster_ids
    ) != n:
        raise ValueError(
            f"Cluster ID count mismatch "
            f"for workload {workload_id}."
        )

    if len(
        deadline_valid
    ) != n:
        raise ValueError(
            f"Deadline count mismatch "
            f"for workload {workload_id}."
        )

    return {
        "workload_id": (
            workload_id
        ),

        "number_of_deliveries": (
            n
        ),

        "delivery_ids": (
            delivery_ids
        ),

        "cluster_ids": (
            cluster_ids
        ),

        "depot_node": (
            depot_node
        ),

        "valid_deadlines": int(
            sum(
                deadline_valid
            )
        ),
    }


# ============================================================
# ROUTE PARSING
# ============================================================

def parse_route_value(
    raw: object,
) -> list[Any]:
    """
    Parse route representations commonly used by
    previous benchmark scripts.

    Supported examples:

        [0, 2, 1]
        (0, 2, 1)
        "0 -> 2 -> 1"
        "Depot -> D1 -> D3 -> Depot"
        ["delivery_id_1", "delivery_id_2"]
    """

    if isinstance(
        raw,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):
        return list(
            raw
        )

    if pd.isna(
        raw
    ):
        raise ValueError(
            "Route value is missing."
        )

    text = str(
        raw
    ).strip()

    if not text:
        raise ValueError(
            "Route value is empty."
        )

    # First try Python literal.
    try:
        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            (
                list,
                tuple,
            ),
        ):
            return list(
                parsed
            )

    except (
        ValueError,
        SyntaxError,
    ):
        pass

    # Arrow-separated route.
    if "->" in text:
        return [
            token.strip()
            for token in text.split(
                "->"
            )
            if token.strip()
        ]

    # Comma-separated fallback.
    cleaned = (
        text
        .strip("[]()")
    )

    if "," in cleaned:
        return [
            token.strip().strip(
                "'\""
            )
            for token in cleaned.split(
                ","
            )
            if token.strip()
        ]

    raise ValueError(
        "Unable to parse route: "
        f"{text!r}"
    )


def is_depot_token(
    token: object,
    *,
    depot_node: int,
) -> bool:
    text = str(
        token
    ).strip().lower()

    if text in {
        "depot",
        "depot01",
        "depot_01",
        "depot 01",
    }:
        return True

    try:
        value = int(
            token
        )
    except (
        TypeError,
        ValueError,
    ):
        return False

    return (
        value == depot_node
    )


def normalize_route_to_actions(
    *,
    raw_route: object,
    metadata: dict[str, Any],
) -> tuple[int, ...]:
    """
    Convert stored route representation into the
    Proposed RL action convention:

        0..N-1

    This function fails loudly if route representation
    cannot be interpreted as one complete permutation.
    """

    n = int(
        metadata[
            "number_of_deliveries"
        ]
    )

    depot_node = int(
        metadata[
            "depot_node"
        ]
    )

    delivery_ids = tuple(
        metadata[
            "delivery_ids"
        ]
    )

    sequence = parse_route_value(
        raw_route
    )

    # --------------------------------------------------------
    # Remove explicit depot at both ends when route is:
    #
    # depot -> N deliveries -> depot
    # --------------------------------------------------------

    if len(
        sequence
    ) == n + 2:
        first = sequence[0]
        last = sequence[-1]

        first_text = str(
            first
        ).strip().lower()

        last_text = str(
            last
        ).strip().lower()

        explicit_depot_pair = (
            is_depot_token(
                first,
                depot_node=depot_node,
            )
            and is_depot_token(
                last,
                depot_node=depot_node,
            )
        )

        # Common matrix-node representation:
        # 0 = depot, 1..N = deliveries.
        zero_depot_pair = (
            first_text == "0"
            and last_text == "0"
        )

        if (
            explicit_depot_pair
            or zero_depot_pair
        ):
            sequence = (
                sequence[
                    1:-1
                ]
            )

    if len(
        sequence
    ) != n:
        raise ValueError(
            "Route length mismatch. "
            f"Expected {n} deliveries, "
            f"found {len(sequence)}. "
            f"Raw route={raw_route!r}"
        )

    # --------------------------------------------------------
    # Case 1:
    # Exact canonical delivery IDs.
    # --------------------------------------------------------

    delivery_id_to_action = {
        str(
            delivery_id
        ): index

        for index, delivery_id
        in enumerate(
            delivery_ids
        )
    }

    string_sequence = [
        str(
            value
        ).strip().strip(
            "'\""
        )
        for value in sequence
    ]

    if all(
        value
        in delivery_id_to_action
        for value
        in string_sequence
    ):
        actions = tuple(
            delivery_id_to_action[
                value
            ]
            for value
            in string_sequence
        )

        _validate_action_permutation(
            actions,
            n,
        )

        return actions

    # --------------------------------------------------------
    # Case 2:
    # D1, D2, ... DN labels.
    # --------------------------------------------------------

    d_matches = [
        re.fullmatch(
            r"[dD](\d+)",
            value,
        )
        for value in string_sequence
    ]

    if all(
        match is not None
        for match in d_matches
    ):
        actions = tuple(
            int(
                match.group(1)
            )
            - 1

            for match
            in d_matches

            if match is not None
        )

        _validate_action_permutation(
            actions,
            n,
        )

        return actions

    # --------------------------------------------------------
    # Case 3:
    # Integer route.
    #
    # Support both:
    #
    # 0..N-1  -> action indices
    #
    # 1..N    -> road-matrix delivery indices
    # --------------------------------------------------------

    try:
        integer_sequence = [
            int(
                value
            )
            for value
            in sequence
        ]

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Unable to map route to delivery actions. "
            f"Route={raw_route!r}"
        ) from exc

    zero_based_expected = set(
        range(
            n
        )
    )

    one_based_expected = set(
        range(
            1,
            n + 1,
        )
    )

    values = set(
        integer_sequence
    )

    if (
        values
        == zero_based_expected
        and len(
            integer_sequence
        ) == n
    ):
        actions = tuple(
            integer_sequence
        )

        _validate_action_permutation(
            actions,
            n,
        )

        return actions

    if (
        values
        == one_based_expected
        and len(
            integer_sequence
        ) == n
    ):
        actions = tuple(
            value - 1
            for value
            in integer_sequence
        )

        _validate_action_permutation(
            actions,
            n,
        )

        return actions

    raise ValueError(
        "Integer route is not a complete delivery "
        f"permutation for N={n}: "
        f"{integer_sequence}"
    )


def _validate_action_permutation(
    actions: tuple[int, ...],
    n: int,
) -> None:
    if len(
        actions
    ) != n:
        raise ValueError(
            "Action route length mismatch."
        )

    if set(
        actions
    ) != set(
        range(
            n
        )
    ):
        raise ValueError(
            "Route is not a complete permutation "
            f"of 0..{n - 1}: {actions}"
        )


# ============================================================
# CLUSTER SWITCHES
# ============================================================

def count_cluster_switches(
    *,
    actions: tuple[int, ...],
    cluster_ids: tuple[int, ...],
) -> int:
    """
    Count delivery-to-delivery cluster changes.

    Depot legs are intentionally excluded,
    matching the Proposed RL objective definition.
    """

    if len(
        actions
    ) <= 1:
        return 0

    switches = 0

    for previous, current in zip(
        actions[:-1],
        actions[1:],
    ):
        previous_cluster = (
            cluster_ids[
                previous
            ]
        )

        current_cluster = (
            cluster_ids[
                current
            ]
        )

        if (
            previous_cluster
            != current_cluster
        ):
            switches += 1

    return switches


# ============================================================
# LOCKED OBJECTIVE
# ============================================================

def compute_locked_objective(
    *,
    n: int,
    valid_deadlines: int,
    distance_km: float,
    travel_time_minutes: float,
    late_deliveries: int,
    lateness_minutes: float,
    cluster_switches: int,
) -> float:
    context = ObjectiveContext(
        number_of_deliveries=(
            n
        ),

        number_of_valid_deadlines=(
            valid_deadlines
        ),
    )

    breakdown = compute_route_objective(
        total_distance_km=(
            distance_km
        ),

        total_travel_time_minutes=(
            travel_time_minutes
        ),

        late_deliveries=(
            late_deliveries
        ),

        total_lateness_minutes=(
            lateness_minutes
        ),

        cluster_switches=(
            cluster_switches
        ),

        context=(
            context
        ),

        config=(
            LOCKED_OBJECTIVE
        ),
    )

    return float(
        breakdown.total_cost
    )


# ============================================================
# BASELINE RESULTS
# ============================================================

def load_baseline_rows(
    test_split: pd.DataFrame,
) -> pd.DataFrame:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline results: "
            f"{BASELINE_PATH}"
        )

    raw = pd.read_csv(
        BASELINE_PATH
    )

    workload_col = resolve_column(
        raw,
        (
            "workload_id",
            "workload",
        ),
    )

    algorithm_col = resolve_column(
        raw,
        (
            "algorithm",
            "algorithm_name",
            "method",
        ),
    )

    size_col = resolve_column(
        raw,
        (
            "number_of_deliveries",
            "delivery_count",
            "n_deliveries",
            "size",
        ),
    )

    distance_col = resolve_column(
        raw,
        (
            "total_distance_km",
            "distance_km",
        ),
    )

    time_col = resolve_column(
        raw,
        (
            "total_travel_time_minutes",
            "travel_time_minutes",
        ),
    )

    late_col = resolve_column(
        raw,
        (
            "late_deliveries",
            "number_of_late_deliveries",
        ),
    )

    lateness_col = resolve_column(
        raw,
        (
            "total_lateness_minutes",
            "lateness_minutes",
        ),
    )

    route_col = resolve_column(
        raw,
        (
            "delivery_order",
            "route_actions",
            "route_order",
            "delivery_sequence",
            "optimized_route",
            "best_route",
            "route",
            "tour",
        ),
    )

    test_ids = set(
        test_split[
            "workload_id"
        ].astype(int)
    )

    raw = (
        raw[
            raw[
                workload_col
            ]
            .astype(int)
            .isin(
                test_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    rows: list[
        dict[str, object]
    ] = []

    for source_row in raw.itertuples(
        index=False
    ):
        source = source_row._asdict()

        workload_id = int(
            source[
                workload_col
            ]
        )

        algorithm = (
            normalize_algorithm_name(
                source[
                    algorithm_col
                ]
            )
        )

        n = int(
            source[
                size_col
            ]
        )

        metadata = (
            load_workload_metadata(
                workload_id
            )
        )

        if (
            int(
                metadata[
                    "number_of_deliveries"
                ]
            )
            != n
        ):
            raise ValueError(
                f"N mismatch for workload "
                f"{workload_id}."
            )

        actions = (
            normalize_route_to_actions(
                raw_route=(
                    source[
                        route_col
                    ]
                ),

                metadata=(
                    metadata
                ),
            )
        )

        cluster_switches = (
            count_cluster_switches(
                actions=actions,

                cluster_ids=(
                    metadata[
                        "cluster_ids"
                    ]
                ),
            )
        )

        valid_deadlines = int(
            metadata[
                "valid_deadlines"
            ]
        )

        distance_km = float(
            source[
                distance_col
            ]
        )

        travel_time_minutes = float(
            source[
                time_col
            ]
        )

        late_deliveries = int(
            source[
                late_col
            ]
        )

        lateness_minutes = float(
            source[
                lateness_col
            ]
        )

        locked_objective = (
            compute_locked_objective(
                n=n,

                valid_deadlines=(
                    valid_deadlines
                ),

                distance_km=(
                    distance_km
                ),

                travel_time_minutes=(
                    travel_time_minutes
                ),

                late_deliveries=(
                    late_deliveries
                ),

                lateness_minutes=(
                    lateness_minutes
                ),

                cluster_switches=(
                    cluster_switches
                ),
            )
        )

        rows.append(
            {
                "workload_id": (
                    workload_id
                ),

                "algorithm": (
                    algorithm
                ),

                "number_of_deliveries": (
                    n
                ),

                "distance_km": (
                    distance_km
                ),

                "travel_time_minutes": (
                    travel_time_minutes
                ),

                "late_deliveries": (
                    late_deliveries
                ),

                "valid_deadlines": (
                    valid_deadlines
                ),

                "lateness_minutes": (
                    lateness_minutes
                ),

                "cluster_switches": (
                    cluster_switches
                ),

                "cluster_switch_rate": (
                    cluster_switches
                    / max(
                        1,
                        n - 1,
                    )
                ),

                "locked_objective": (
                    locked_objective
                ),

                "route_actions": str(
                    actions
                ),
            }
        )

    baseline = pd.DataFrame(
        rows
    )

    expected_rows = (
        EXPECTED_TEST_WORKLOADS
        * 5
    )

    if len(
        baseline
    ) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} "
            "baseline rows, "
            f"found {len(baseline)}."
        )

    return baseline


# ============================================================
# PROPOSED RL
# ============================================================

def load_proposed_rows(
    test_split: pd.DataFrame,
) -> pd.DataFrame:
    if not PROPOSED_DETAILS_PATH.exists():
        raise FileNotFoundError(
            "Missing Proposed RL Test details: "
            f"{PROPOSED_DETAILS_PATH}"
        )

    raw = pd.read_csv(
        PROPOSED_DETAILS_PATH
    )

    required = {
        "workload_id",
        "number_of_deliveries",
        "distance_km",
        "travel_time_minutes",
        "late_deliveries",
        "valid_deadlines",
        "lateness_minutes",
        "route_actions",
    }

    missing = (
        required
        - set(raw.columns)
    )

    if missing:
        raise ValueError(
            "Proposed RL details missing: "
            f"{sorted(missing)}"
        )

    expected_ids = set(
        test_split[
            "workload_id"
        ].astype(int)
    )

    actual_ids = set(
        raw[
            "workload_id"
        ].astype(int)
    )

    if (
        actual_ids
        != expected_ids
    ):
        raise ValueError(
            "Proposed RL workloads do not "
            "match frozen Test split."
        )

    rows: list[
        dict[str, object]
    ] = []

    for source_row in raw.itertuples(
        index=False
    ):
        source = source_row._asdict()

        workload_id = int(
            source[
                "workload_id"
            ]
        )

        n = int(
            source[
                "number_of_deliveries"
            ]
        )

        metadata = (
            load_workload_metadata(
                workload_id
            )
        )

        actions = (
            normalize_route_to_actions(
                raw_route=(
                    source[
                        "route_actions"
                    ]
                ),

                metadata=(
                    metadata
                ),
            )
        )

        cluster_switches = (
            count_cluster_switches(
                actions=(
                    actions
                ),

                cluster_ids=(
                    metadata[
                        "cluster_ids"
                    ]
                ),
            )
        )

        # Verify against the stored Proposed metric
        # when available.
        if (
            "cluster_switches"
            in raw.columns
        ):
            stored_switches = int(
                source[
                    "cluster_switches"
                ]
            )

            if (
                stored_switches
                != cluster_switches
            ):
                raise ValueError(
                    "Proposed RL cluster-switch "
                    f"mismatch for workload "
                    f"{workload_id}: "
                    f"stored={stored_switches}, "
                    f"recomputed={cluster_switches}"
                )

        distance_km = float(
            source[
                "distance_km"
            ]
        )

        travel_time_minutes = float(
            source[
                "travel_time_minutes"
            ]
        )

        late_deliveries = int(
            source[
                "late_deliveries"
            ]
        )

        valid_deadlines = int(
            source[
                "valid_deadlines"
            ]
        )

        lateness_minutes = float(
            source[
                "lateness_minutes"
            ]
        )

        locked_objective = (
            compute_locked_objective(
                n=n,

                valid_deadlines=(
                    valid_deadlines
                ),

                distance_km=(
                    distance_km
                ),

                travel_time_minutes=(
                    travel_time_minutes
                ),

                late_deliveries=(
                    late_deliveries
                ),

                lateness_minutes=(
                    lateness_minutes
                ),

                cluster_switches=(
                    cluster_switches
                ),
            )
        )

        rows.append(
            {
                "workload_id": (
                    workload_id
                ),

                "algorithm": (
                    PROPOSED_NAME
                ),

                "number_of_deliveries": (
                    n
                ),

                "distance_km": (
                    distance_km
                ),

                "travel_time_minutes": (
                    travel_time_minutes
                ),

                "late_deliveries": (
                    late_deliveries
                ),

                "valid_deadlines": (
                    valid_deadlines
                ),

                "lateness_minutes": (
                    lateness_minutes
                ),

                "cluster_switches": (
                    cluster_switches
                ),

                "cluster_switch_rate": (
                    cluster_switches
                    / max(
                        1,
                        n - 1,
                    )
                ),

                "locked_objective": (
                    locked_objective
                ),

                "route_actions": str(
                    actions
                ),
            }
        )

    proposed = pd.DataFrame(
        rows
    )

    if (
        len(proposed)
        != EXPECTED_TEST_WORKLOADS
    ):
        raise ValueError(
            "Expected exactly 16 "
            "Proposed RL rows."
        )

    return proposed


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    details: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for algorithm in ALGORITHM_ORDER:
        group = details[
            details[
                "algorithm"
            ]
            == algorithm
        ]

        if (
            len(group)
            != EXPECTED_TEST_WORKLOADS
        ):
            raise ValueError(
                f"{algorithm}: expected 16 rows, "
                f"found {len(group)}."
            )

        total_valid = int(
            group[
                "valid_deadlines"
            ].sum()
        )

        total_late = int(
            group[
                "late_deliveries"
            ].sum()
        )

        if total_valid == 0:
            weighted_sla = 1.0
        else:
            weighted_sla = (
                (
                    total_valid
                    - total_late
                )
                / total_valid
            )

        rows.append(
            {
                "algorithm": (
                    algorithm
                ),

                "workloads": (
                    len(group)
                ),

                "deliveries": int(
                    group[
                        "number_of_deliveries"
                    ].sum()
                ),

                "mean_distance_km": float(
                    group[
                        "distance_km"
                    ].mean()
                ),

                "mean_travel_time_minutes": float(
                    group[
                        "travel_time_minutes"
                    ].mean()
                ),

                "late_deliveries": (
                    total_late
                ),

                "weighted_sla": (
                    weighted_sla
                ),

                "total_lateness_minutes": float(
                    group[
                        "lateness_minutes"
                    ].sum()
                ),

                "mean_cluster_switches": float(
                    group[
                        "cluster_switches"
                    ].mean()
                ),

                "mean_cluster_switch_rate": float(
                    group[
                        "cluster_switch_rate"
                    ].mean()
                ),

                "mean_locked_objective": float(
                    group[
                        "locked_objective"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROPOSED RL VS EACH BASELINE
# ============================================================

def build_pairwise(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    proposed = (
        summary[
            summary[
                "algorithm"
            ]
            == PROPOSED_NAME
        ]
        .iloc[0]
    )

    rows: list[
        dict[str, object]
    ] = []

    for baseline_name in ALGORITHM_ORDER:
        if (
            baseline_name
            == PROPOSED_NAME
        ):
            continue

        baseline = (
            summary[
                summary[
                    "algorithm"
                ]
                == baseline_name
            ]
            .iloc[0]
        )

        baseline_objective = float(
            baseline[
                "mean_locked_objective"
            ]
        )

        proposed_objective = float(
            proposed[
                "mean_locked_objective"
            ]
        )

        if baseline_objective > 0.0:
            objective_improvement = (
                (
                    baseline_objective
                    - proposed_objective
                )
                / baseline_objective
                * 100.0
            )
        else:
            objective_improvement = (
                math.nan
            )

        baseline_switches = float(
            baseline[
                "mean_cluster_switches"
            ]
        )

        proposed_switches = float(
            proposed[
                "mean_cluster_switches"
            ]
        )

        if baseline_switches > 0.0:
            switch_reduction = (
                (
                    baseline_switches
                    - proposed_switches
                )
                / baseline_switches
                * 100.0
            )
        else:
            switch_reduction = (
                math.nan
            )

        rows.append(
            {
                "baseline_algorithm": (
                    baseline_name
                ),

                "baseline_mean_locked_objective": (
                    baseline_objective
                ),

                "proposed_mean_locked_objective": (
                    proposed_objective
                ),

                # Positive means Proposed RL has
                # lower/better locked objective.
                "locked_objective_improvement_pct": (
                    objective_improvement
                ),

                "baseline_mean_cluster_switches": (
                    baseline_switches
                ),

                "proposed_mean_cluster_switches": (
                    proposed_switches
                ),

                # Positive means Proposed RL switches
                # clusters less often.
                "cluster_switch_reduction_pct": (
                    switch_reduction
                ),

                "sla_change_percentage_points": (
                    (
                        float(
                            proposed[
                                "weighted_sla"
                            ]
                        )
                        - float(
                            baseline[
                                "weighted_sla"
                            ]
                        )
                    )
                    * 100.0
                ),

                "late_delivery_change": (
                    int(
                        proposed[
                            "late_deliveries"
                        ]
                    )
                    - int(
                        baseline[
                            "late_deliveries"
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FINDINGS
# ============================================================

def write_findings(
    *,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    lines = [
        (
            "PROPOSED RL - FINAL LOCKED "
            "OBJECTIVE COMPARISON"
        ),

        "",

        (
            "Held-out Test set:"
        ),

        (
            "  workloads = 16"
        ),

        (
            "  deliveries = 131"
        ),

        "",

        (
            "Locked objective:"
        ),

        (
            "J = 0.375*D_bar + "
            "0.375*T_bar + "
            "0.10*M_bar + "
            "0.10*L_bar + "
            "0.05*S_bar"
        ),

        "",

        (
            "All six algorithms are evaluated "
            "with the same objective after routes "
            "were already produced."
        ),

        (
            "No training or Test tuning occurs "
            "in this analysis."
        ),

        "",

        "SUMMARY",
        "",
    ]

    for row in summary.itertuples(
        index=False
    ):
        lines.extend(
            [
                (
                    f"[{row.algorithm}]"
                ),

                (
                    "mean_distance_km = "
                    f"{row.mean_distance_km:.9f}"
                ),

                (
                    "mean_travel_time_minutes = "
                    f"{row.mean_travel_time_minutes:.9f}"
                ),

                (
                    f"late_deliveries = "
                    f"{row.late_deliveries}"
                ),

                (
                    f"weighted_sla = "
                    f"{row.weighted_sla:.9f}"
                ),

                (
                    "total_lateness_minutes = "
                    f"{row.total_lateness_minutes:.9f}"
                ),

                (
                    "mean_cluster_switches = "
                    f"{row.mean_cluster_switches:.9f}"
                ),

                (
                    "mean_cluster_switch_rate = "
                    f"{row.mean_cluster_switch_rate:.9f}"
                ),

                (
                    "mean_locked_objective = "
                    f"{row.mean_locked_objective:.9f}"
                ),

                "",
            ]
        )

    lines.extend(
        [
            (
                "PROPOSED RL VS BASELINES"
            ),
            "",
        ]
    )

    for row in pairwise.itertuples(
        index=False
    ):
        lines.extend(
            [
                (
                    f"[vs {row.baseline_algorithm}]"
                ),

                (
                    "locked objective improvement = "
                    f"{row.locked_objective_improvement_pct:.6f}%"
                ),

                (
                    "cluster switch reduction = "
                    f"{row.cluster_switch_reduction_pct:.6f}%"
                ),

                (
                    "SLA change = "
                    f"{row.sla_change_percentage_points:.6f} "
                    "percentage points"
                ),

                (
                    "late delivery change = "
                    f"{row.late_delivery_change}"
                ),

                "",
            ]
        )

    lines.extend(
        [
            (
                "Interpretation:"
            ),

            (
                "Positive locked-objective improvement "
                "means Proposed RL has lower J."
            ),

            (
                "Positive cluster-switch reduction "
                "means Proposed RL performs fewer "
                "delivery-to-delivery cluster changes."
            ),

            "",
            (
                "This analysis does not replace the "
                "previous distance/time/SLA comparison; "
                "it complements it using the exact "
                "objective proposed by the thesis."
            ),
        ]
    )

    FINDINGS_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 94)

    print(
        "PROPOSED RL - FINAL LOCKED "
        "OBJECTIVE COMPARISON"
    )

    print("=" * 94)

    print()
    print(
        "NO TRAINING."
    )

    print(
        "NO OBJECTIVE TUNING."
    )

    print(
        "NO MODEL SELECTION."
    )

    print()

    print(
        "Locked objective:"
    )

    print(
        "J = 0.375 D_bar + "
        "0.375 T_bar + "
        "0.10 M_bar + "
        "0.10 L_bar + "
        "0.05 S_bar"
    )

    # --------------------------------------------------------
    # Frozen Test
    # --------------------------------------------------------

    test_split = (
        load_test_split()
    )

    print()
    print(
        "Frozen Test split:"
    )

    print(
        f"  workloads = "
        f"{len(test_split)}"
    )

    print(
        "  deliveries = "
        f"{test_split['number_of_deliveries'].sum()}"
    )

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    print()
    print(
        "Loading five baseline algorithms..."
    )

    baseline = (
        load_baseline_rows(
            test_split
        )
    )

    print(
        f"Baseline rows: "
        f"{len(baseline)}"
    )

    # --------------------------------------------------------
    # Proposed RL
    # --------------------------------------------------------

    print()
    print(
        "Loading locked Proposed RL Test routes..."
    )

    proposed = (
        load_proposed_rows(
            test_split
        )
    )

    print(
        f"Proposed RL rows: "
        f"{len(proposed)}"
    )

    # --------------------------------------------------------
    # Combined
    # --------------------------------------------------------

    details = pd.concat(
        [
            baseline,
            proposed,
        ],
        ignore_index=True,
    )

    expected_rows = (
        EXPECTED_TEST_WORKLOADS
        * len(
            ALGORITHM_ORDER
        )
    )

    if (
        len(details)
        != expected_rows
    ):
        raise ValueError(
            f"Expected {expected_rows} "
            "combined rows, "
            f"found {len(details)}."
        )

    details.to_csv(
        DETAILS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        build_summary(
            details
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Pairwise
    # --------------------------------------------------------

    pairwise = (
        build_pairwise(
            summary
        )
    )

    pairwise.to_csv(
        PAIRWISE_PATH,
        index=False,
    )

    write_findings(
        summary=(
            summary
        ),

        pairwise=(
            pairwise
        ),
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print("=" * 94)

    print(
        "FINAL LOCKED OBJECTIVE SUMMARY"
    )

    print("=" * 94)

    display = summary.copy()

    display[
        "weighted_sla"
    ] = (
        display[
            "weighted_sla"
        ]
        * 100.0
    )

    print(
        display[
            [
                "algorithm",
                "mean_distance_km",
                "mean_travel_time_minutes",
                "late_deliveries",
                "weighted_sla",
                "total_lateness_minutes",
                "mean_cluster_switches",
                "mean_cluster_switch_rate",
                "mean_locked_objective",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 94)

    print(
        "PROPOSED RL - LOCKED OBJECTIVE "
        "VS BASELINES"
    )

    print("=" * 94)

    print(
        pairwise.to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  objective improvement > 0 "
        "=> Proposed RL has lower/better J."
    )

    print(
        "  cluster switch reduction > 0 "
        "=> Proposed RL changes cluster less often."
    )

    print()
    print(
        f"Details: "
        f"{DETAILS_PATH}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Pairwise: "
        f"{PAIRWISE_PATH}"
    )

    print(
        f"Findings: "
        f"{FINDINGS_PATH}"
    )


if __name__ == "__main__":
    main()