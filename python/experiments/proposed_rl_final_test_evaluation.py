from __future__ import annotations

import gc
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

import experiments.final_kmeans_guided_dqn_validation as validation
import experiments.kmeans_guided_dqn_sensitivity as guided

from experiments.proposed_rl_dataset_preflight import (
    load_manifest,
    load_split,
)

from rl.proposed.agent import (
    ProposedAgentConfig,
    ProposedRoutingAgent,
)

from rl.proposed.candidate_encoder import (
    CandidateEncoder,
)

from rl.proposed.environment import (
    ProposedRoutingEnvironment,
)

from rl.proposed.multi_workload_trainer import (
    WorkloadEnvironment,
    evaluate_workloads,
)

from rl.proposed.objective import (
    ObjectiveConfig,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_CACHE_DIR = (
    RESULTS_DIR
    / "proposed_rl_test_road_cache"
)

CHECKPOINT_PATH = (
    RESULTS_DIR
    / "proposed_rl_multiworkload_D_moderate_sla_best.pt"
)

DETAILS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_test_details.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_test_summary.csv"
)

FINDINGS_PATH = (
    RESULTS_DIR
    / "proposed_rl_final_test_findings.txt"
)


# ============================================================
# LOCKED EXPERIMENT CONFIGURATION
# ============================================================

EXPECTED_TEST_WORKLOADS = 16

EXPECTED_CONFIG_NAME = (
    "D_moderate_sla"
)

LOCKED_OBJECTIVE = ObjectiveConfig(
    distance_weight=0.375,
    travel_time_weight=0.375,
    late_delivery_weight=0.10,
    lateness_weight=0.10,
    cluster_switch_weight=0.05,
)


# ============================================================
# WORKLOAD BUILDING
# ============================================================

def build_workload_dataframe(
    *,
    workload_id: int,
    manifest: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    matches = manifest[
        manifest["workload_id"]
        == workload_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one manifest row for "
            f"workload {workload_id}, "
            f"found {len(matches)}."
        )

    workload = (
        validation
        .build_workload_dataframe_from_manifest(
            prepared_dataframe,
            matches.iloc[0],
        )
        .reset_index(
            drop=True
        )
    )

    if workload.empty:
        raise ValueError(
            f"Workload {workload_id} is empty."
        )

    return workload


def prepare_deadlines(
    workload: pd.DataFrame,
) -> list[float | None]:
    """
    Convert absolute deadlines into minutes
    relative to wave_end.

    This uses exactly the convention validated
    during dataset preflight.
    """

    wave_end = pd.to_datetime(
        workload["wave_end"],
        utc=True,
        errors="raise",
    )

    unique_wave_end = (
        wave_end
        .dropna()
        .unique()
    )

    if len(unique_wave_end) != 1:
        raise ValueError(
            "Expected exactly one wave_end."
        )

    start_time = pd.Timestamp(
        unique_wave_end[0]
    )

    expected = pd.to_datetime(
        workload[
            "expected_delivery_time"
        ],
        utc=True,
        errors="coerce",
    )

    deadlines: list[
        float | None
    ] = []

    for deadline in expected:
        if pd.isna(deadline):
            deadlines.append(
                None
            )
            continue

        relative_minutes = (
            (
                pd.Timestamp(deadline)
                - start_time
            ).total_seconds()
            / 60.0
        )

        if relative_minutes < 0.0:
            raise ValueError(
                "Negative relative deadline "
                "encountered in final Test."
            )

        deadlines.append(
            float(
                relative_minutes
            )
        )

    return deadlines


# ============================================================
# TEST ROAD CACHE
# ============================================================

def cache_test_workload(
    *,
    workload_id: int,
    expected_size: int,
    manifest: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
    depot_node: int,
    matrix_builder,
) -> Path:
    workload = build_workload_dataframe(
        workload_id=workload_id,
        manifest=manifest,
        prepared_dataframe=prepared_dataframe,
    )

    if len(workload) != expected_size:
        raise ValueError(
            f"Workload {workload_id}: "
            f"expected N={expected_size}, "
            f"rebuilt N={len(workload)}."
        )

    deadlines = prepare_deadlines(
        workload
    )

    delivery_nodes = [
        int(value)
        for value in workload[
            "road_node"
        ].tolist()
    ]

    cluster_ids = [
        int(value)
        for value in workload[
            "cluster_id"
        ].tolist()
    ]

    delivery_ids = [
        str(value)
        for value in workload[
            "delivery_id"
        ].tolist()
    ]

    node_ids = [
        int(depot_node),
        *delivery_nodes,
    ]

    road_matrix = (
        matrix_builder.build(
            node_ids
        )
    )

    distance_km = np.asarray(
        road_matrix.distance_km,
        dtype=np.float64,
    )

    travel_time_minutes = np.asarray(
        road_matrix.travel_time_minutes,
        dtype=np.float64,
    )

    expected_shape = (
        expected_size + 1,
        expected_size + 1,
    )

    if distance_km.shape != expected_shape:
        raise ValueError(
            f"Invalid distance matrix shape "
            f"for workload {workload_id}."
        )

    if (
        travel_time_minutes.shape
        != expected_shape
    ):
        raise ValueError(
            f"Invalid travel-time matrix shape "
            f"for workload {workload_id}."
        )

    if not np.isfinite(
        distance_km
    ).all():
        raise ValueError(
            f"Non-finite distance in "
            f"workload {workload_id}."
        )

    if not np.isfinite(
        travel_time_minutes
    ).all():
        raise ValueError(
            f"Non-finite travel time in "
            f"workload {workload_id}."
        )

    deadline_values = np.asarray(
        [
            np.nan
            if value is None
            else value
            for value in deadlines
        ],
        dtype=np.float64,
    )

    deadline_valid = np.asarray(
        [
            value is not None
            for value in deadlines
        ],
        dtype=bool,
    )

    cache_path = (
        TEST_CACHE_DIR
        / f"workload_{workload_id}.npz"
    )

    np.savez_compressed(
        cache_path,

        workload_id=np.asarray(
            [workload_id],
            dtype=np.int64,
        ),

        number_of_deliveries=np.asarray(
            [expected_size],
            dtype=np.int64,
        ),

        depot_node=np.asarray(
            [depot_node],
            dtype=np.int64,
        ),

        delivery_nodes=np.asarray(
            delivery_nodes,
            dtype=np.int64,
        ),

        delivery_ids=np.asarray(
            delivery_ids,
            dtype=str,
        ),

        cluster_ids=np.asarray(
            cluster_ids,
            dtype=np.int64,
        ),

        deadlines_minutes=(
            deadline_values
        ),

        deadline_valid=(
            deadline_valid
        ),

        distance_km=(
            distance_km
        ),

        travel_time_minutes=(
            travel_time_minutes
        ),
    )

    # Avoid accumulation of road shortest-path cache.
    matrix_builder.clear_cache()

    return cache_path


def prepare_test_cache(
    *,
    split: pd.DataFrame,
) -> None:
    TEST_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_rows = (
        split[
            split["split"]
            == "test"
        ]
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

    if len(test_rows) != EXPECTED_TEST_WORKLOADS:
        raise ValueError(
            f"Expected "
            f"{EXPECTED_TEST_WORKLOADS} "
            f"Test workloads, "
            f"found {len(test_rows)}."
        )

    manifest = load_manifest()

    prepared_dataframe, _ = (
        guided.prepare_experiment_dataframe()
    )

    print()
    print(
        "Loading road graph for final Test..."
    )

    road_network = (
        validation.load_road_network()
    )

    depot, depot_node = (
        validation.get_depot_node(
            road_network
        )
    )

    depot_node = int(
        depot_node
    )

    print(
        f"Depot: {depot.name}"
    )

    print(
        f"Depot road node: "
        f"{depot_node}"
    )

    matrix_builder = (
        validation.RoadMatrixBuilder(
            road_network
        )
    )

    print()
    print(
        "Preparing 16 held-out Test road matrices..."
    )

    total_start = perf_counter()

    for index, row in test_rows.iterrows():
        workload_id = int(
            row["workload_id"]
        )

        n = int(
            row[
                "number_of_deliveries"
            ]
        )

        cache_path = (
            TEST_CACHE_DIR
            / f"workload_{workload_id}.npz"
        )

        start = perf_counter()

        # Build only if missing.
        if not cache_path.exists():
            cache_test_workload(
                workload_id=(
                    workload_id
                ),

                expected_size=(
                    n
                ),

                manifest=(
                    manifest
                ),

                prepared_dataframe=(
                    prepared_dataframe
                ),

                depot_node=(
                    depot_node
                ),

                matrix_builder=(
                    matrix_builder
                ),
            )

            source = (
                "BUILT"
            )

        else:
            source = (
                "EXISTING"
            )

        runtime = (
            perf_counter()
            - start
        )

        print(
            f"[{index + 1:2d}/16] "
            f"workload={workload_id:3d} | "
            f"N={n:2d} | "
            f"{source:8s} | "
            f"{runtime:.3f} s"
        )

    matrix_builder.clear_cache()

    print()
    print(
        "Test cache preparation runtime: "
        f"{(perf_counter() - total_start) / 60.0:.2f} min"
    )


# ============================================================
# CACHE → ENVIRONMENT
# ============================================================

def load_test_environment(
    *,
    workload_id: int,
) -> ProposedRoutingEnvironment:
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

        distance_matrix = (
            data[
                "distance_km"
            ]
            .astype(float)
            .tolist()
        )

        travel_time_matrix = (
            data[
                "travel_time_minutes"
            ]
            .astype(float)
            .tolist()
        )

        raw_deadlines = (
            data[
                "deadlines_minutes"
            ]
            .astype(float)
        )

        valid_mask = (
            data[
                "deadline_valid"
            ]
            .astype(bool)
        )

        cluster_ids = (
            data[
                "cluster_ids"
            ]
            .astype(int)
            .tolist()
        )

        deadlines: list[
            float | None
        ] = []

        for value, valid in zip(
            raw_deadlines,
            valid_mask,
        ):
            if valid:
                deadlines.append(
                    float(value)
                )
            else:
                deadlines.append(
                    None
                )

    if len(deadlines) != n:
        raise ValueError(
            f"Deadline count mismatch "
            f"for workload {workload_id}."
        )

    if len(cluster_ids) != n:
        raise ValueError(
            f"Cluster count mismatch "
            f"for workload {workload_id}."
        )

    return ProposedRoutingEnvironment(
        distance_matrix_km=(
            distance_matrix
        ),

        travel_time_matrix_minutes=(
            travel_time_matrix
        ),

        deadlines_minutes=(
            deadlines
        ),

        cluster_ids=(
            cluster_ids
        ),

        objective_config=(
            LOCKED_OBJECTIVE
        ),
    )


def build_test_workloads(
    split: pd.DataFrame,
) -> tuple[
    WorkloadEnvironment,
    ...,
]:
    test_rows = (
        split[
            split["split"]
            == "test"
        ]
        .sort_values(
            [
                "number_of_deliveries",
                "workload_id",
            ]
        )
    )

    workloads: list[
        WorkloadEnvironment
    ] = []

    for row in test_rows.itertuples(
        index=False
    ):
        workload_id = int(
            row.workload_id
        )

        workloads.append(
            WorkloadEnvironment(
                workload_id=(
                    workload_id
                ),

                environment=(
                    load_test_environment(
                        workload_id=(
                            workload_id
                        )
                    )
                ),
            )
        )

    return tuple(
        workloads
    )


# ============================================================
# LOCKED CHECKPOINT
# ============================================================

def load_locked_agent(
) -> tuple[
    ProposedRoutingAgent,
    dict,
]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing locked checkpoint: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    config_name = str(
        checkpoint[
            "config_name"
        ]
    )

    if config_name != EXPECTED_CONFIG_NAME:
        raise ValueError(
            "Unexpected checkpoint config: "
            f"{config_name}"
        )

    objective_dict = checkpoint[
        "objective_config"
    ]

    checkpoint_objective = (
        ObjectiveConfig(
            **objective_dict
        )
    )

    if (
        asdict(
            checkpoint_objective
        )
        != asdict(
            LOCKED_OBJECTIVE
        )
    ):
        raise ValueError(
            "Checkpoint objective does not "
            "match locked final objective."
        )

    agent_config = (
        ProposedAgentConfig(
            **checkpoint[
                "agent_config"
            ]
        )
    )

    agent = ProposedRoutingAgent(
        config=(
            agent_config
        ),

        feature_dimension=int(
            checkpoint[
                "feature_dimension"
            ]
        ),
    )

    agent.online_network.load_state_dict(
        checkpoint[
            "online_network_state_dict"
        ]
    )

    agent.target_network.load_state_dict(
        checkpoint[
            "target_network_state_dict"
        ]
    )

    # Final Test is inference only.
    # epsilon value does not matter because
    # evaluate_agent() uses greedy selection.
    agent.epsilon = float(
        checkpoint[
            "epsilon"
        ]
    )

    return (
        agent,
        checkpoint,
    )


# ============================================================
# RESULTS
# ============================================================

def evaluations_to_dataframe(
    evaluations,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for result in evaluations:
        rows.append(
            {
                "workload_id": (
                    result.workload_id
                ),

                "number_of_deliveries": (
                    result.number_of_deliveries
                ),

                "distance_km": (
                    result.distance_km
                ),

                "travel_time_minutes": (
                    result.travel_time_minutes
                ),

                "late_deliveries": (
                    result.late_deliveries
                ),

                "valid_deadlines": (
                    result.valid_deadlines
                ),

                "on_time_deliveries": (
                    result.on_time_deliveries
                ),

                "on_time_rate": (
                    result.on_time_rate
                ),

                "lateness_minutes": (
                    result.lateness_minutes
                ),

                "cluster_switches": (
                    result.cluster_switches
                ),

                "objective_cost": (
                    result.objective_cost
                ),

                "route_actions": str(
                    result.route_actions
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 82)

    print(
        "PROPOSED RL - "
        "FINAL HELD-OUT TEST EVALUATION"
    )

    print("=" * 82)

    print()
    print(
        "LOCKED OBJECTIVE:"
    )

    print(
        "J = "
        "0.375 D_bar + "
        "0.375 T_bar + "
        "0.10 M_bar + "
        "0.10 L_bar + "
        "0.05 S_bar"
    )

    print()
    print(
        "NO TRAINING WILL OCCUR "
        "IN THIS SCRIPT."
    )

    # --------------------------------------------------------
    # 1. Frozen split
    # --------------------------------------------------------

    split = load_split()

    test_rows = (
        split[
            split["split"]
            == "test"
        ]
    )

    if len(test_rows) != 16:
        raise ValueError(
            "Expected exactly 16 Test workloads."
        )

    print()
    print(
        "Held-out Test workloads: 16"
    )

    # --------------------------------------------------------
    # 2. Build Test road matrices now that model is locked
    # --------------------------------------------------------

    prepare_test_cache(
        split=split
    )

    # --------------------------------------------------------
    # 3. Load locked Validation-selected model
    # --------------------------------------------------------

    print()
    print(
        "Loading locked D checkpoint..."
    )

    (
        agent,
        checkpoint,
    ) = load_locked_agent()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint config: "
        f"{checkpoint['config_name']}"
    )

    print(
        f"Device: "
        f"{agent.device}"
    )

    # --------------------------------------------------------
    # 4. Build held-out environments
    # --------------------------------------------------------

    test_workloads = (
        build_test_workloads(
            split
        )
    )

    if len(test_workloads) != 16:
        raise ValueError(
            "Expected 16 Test environments."
        )

    encoder = CandidateEncoder()

    # --------------------------------------------------------
    # 5. FINAL TEST INFERENCE
    # --------------------------------------------------------

    print()
    print(
        "Running greedy inference on "
        "held-out Test workloads..."
    )

    inference_start = (
        perf_counter()
    )

    (
        evaluations,
        summary,
    ) = evaluate_workloads(
        workloads=(
            test_workloads
        ),

        encoder=(
            encoder
        ),

        agent=(
            agent
        ),
    )

    inference_runtime = (
        perf_counter()
        - inference_start
    )

    details = (
        evaluations_to_dataframe(
            evaluations
        )
    )

    details.to_csv(
        DETAILS_PATH,
        index=False,
    )

    summary_df = pd.DataFrame(
        [
            {
                "config_name": (
                    EXPECTED_CONFIG_NAME
                ),

                "checkpoint_epoch": int(
                    checkpoint[
                        "epoch"
                    ]
                ),

                "test_workloads": (
                    summary.workloads
                ),

                "test_deliveries": (
                    summary.deliveries
                ),

                "mean_distance_km": (
                    summary.mean_distance_km
                ),

                "mean_travel_time_minutes": (
                    summary
                    .mean_travel_time_minutes
                ),

                "late_deliveries": (
                    summary.late_deliveries
                ),

                "valid_deadlines": (
                    summary.valid_deadlines
                ),

                "weighted_sla": (
                    summary.weighted_sla
                ),

                "total_lateness_minutes": (
                    summary
                    .total_lateness_minutes
                ),

                "mean_lateness_minutes": (
                    summary
                    .mean_lateness_minutes
                ),

                "mean_cluster_switches": (
                    summary
                    .mean_cluster_switches
                ),

                "inference_runtime_seconds": (
                    inference_runtime
                ),
            }
        ]
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Print FINAL result
    # --------------------------------------------------------

    print()
    print("=" * 82)

    print(
        "FINAL TEST RESULT"
    )

    print("=" * 82)

    print(
        f"Workloads: "
        f"{summary.workloads}"
    )

    print(
        f"Deliveries: "
        f"{summary.deliveries}"
    )

    print(
        f"Mean distance: "
        f"{summary.mean_distance_km:.6f} km"
    )

    print(
        f"Mean travel time: "
        f"{summary.mean_travel_time_minutes:.6f} min"
    )

    print(
        f"Late deliveries: "
        f"{summary.late_deliveries}"
    )

    print(
        f"Weighted SLA: "
        f"{summary.weighted_sla * 100:.2f}%"
    )

    print(
        f"Total lateness: "
        f"{summary.total_lateness_minutes:.6f} min"
    )

    print(
        f"Mean lateness/workload: "
        f"{summary.mean_lateness_minutes:.6f} min"
    )

    print(
        f"Mean cluster switches: "
        f"{summary.mean_cluster_switches:.6f}"
    )

    print(
        f"Inference runtime: "
        f"{inference_runtime:.6f} s"
    )

    # --------------------------------------------------------
    # 7. Findings
    # --------------------------------------------------------

    findings = [
        (
            "PROPOSED RL FINAL HELD-OUT TEST"
        ),

        "",

        (
            "Model selection was completed using "
            "Train and Validation only."
        ),

        (
            "No additional training or objective "
            "tuning was performed on Test."
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
            f"checkpoint_epoch = "
            f"{checkpoint['epoch']}"
        ),

        (
            f"test_workloads = "
            f"{summary.workloads}"
        ),

        (
            f"test_deliveries = "
            f"{summary.deliveries}"
        ),

        (
            f"mean_distance_km = "
            f"{summary.mean_distance_km:.9f}"
        ),

        (
            "mean_travel_time_minutes = "
            f"{summary.mean_travel_time_minutes:.9f}"
        ),

        (
            f"late_deliveries = "
            f"{summary.late_deliveries}"
        ),

        (
            f"weighted_sla = "
            f"{summary.weighted_sla:.9f}"
        ),

        (
            "total_lateness_minutes = "
            f"{summary.total_lateness_minutes:.9f}"
        ),

        (
            "mean_cluster_switches = "
            f"{summary.mean_cluster_switches:.9f}"
        ),

        (
            "inference_runtime_seconds = "
            f"{inference_runtime:.9f}"
        ),
    ]

    FINDINGS_PATH.write_text(
        "\n".join(
            findings
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Details saved to: "
        f"{DETAILS_PATH}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Findings saved to: "
        f"{FINDINGS_PATH}"
    )

    # --------------------------------------------------------
    # 8. Cleanup
    # --------------------------------------------------------

    del evaluations
    del test_workloads
    del encoder
    del agent

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()