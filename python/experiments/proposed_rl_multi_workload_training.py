from __future__ import annotations

import copy
import gc
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import pandas as pd
import torch

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
    train_multi_workload_epoch,
)
from rl.proposed.objective import (
    ObjectiveConfig,
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

CACHE_DIR = (
    RESULTS_DIR
    / "proposed_rl_road_cache"
)

CACHE_MANIFEST_PATH = (
    RESULTS_DIR
    / "proposed_rl_road_cache_manifest.csv"
)

COMPARISON_PATH = (
    RESULTS_DIR
    / "proposed_rl_multiworkload_validation_comparison.csv"
)

FINDINGS_PATH = (
    RESULTS_DIR
    / "proposed_rl_multiworkload_validation_findings.txt"
)


# ============================================================
# TRAINING DESIGN
# ============================================================

SEED = 42

TRAIN_WORKLOADS = 76
VALIDATION_WORKLOADS = 16

EPOCHS = 40
EPISODES_PER_WORKLOAD = 1

VALIDATION_INTERVAL = 5

TOTAL_TRAINING_EPISODES = (
    EPOCHS
    * TRAIN_WORKLOADS
    * EPISODES_PER_WORKLOAD
)

EPSILON_START = 1.0
EPSILON_MIN = 0.02

# Epsilon giảm đến gần minimum ở khoảng 90%
# tổng training episodes.
EPSILON_TARGET_FRACTION = 0.90

EPSILON_DECAY = (
    EPSILON_MIN
    / EPSILON_START
) ** (
    1.0
    / (
        TOTAL_TRAINING_EPISODES
        * EPSILON_TARGET_FRACTION
    )
)


# ============================================================
# OBJECTIVE CANDIDATES
# ============================================================

OBJECTIVE_CONFIGS = {
    "A_current": ObjectiveConfig(
        distance_weight=0.40,
        travel_time_weight=0.40,
        late_delivery_weight=0.00,
        lateness_weight=0.15,
        cluster_switch_weight=0.05,
    ),

    "D_moderate_sla": ObjectiveConfig(
        distance_weight=0.375,
        travel_time_weight=0.375,
        late_delivery_weight=0.10,
        lateness_weight=0.10,
        cluster_switch_weight=0.05,
    ),
}


# ============================================================
# RESULT PATH HELPERS
# ============================================================

def history_path(
    config_name: str,
) -> Path:
    return (
        RESULTS_DIR
        / (
            "proposed_rl_multiworkload_"
            f"{config_name}_history.csv"
        )
    )


def checkpoint_path(
    config_name: str,
) -> Path:
    return (
        RESULTS_DIR
        / (
            "proposed_rl_multiworkload_"
            f"{config_name}_best.pt"
        )
    )


def validation_details_path(
    config_name: str,
) -> Path:
    return (
        RESULTS_DIR
        / (
            "proposed_rl_multiworkload_"
            f"{config_name}_validation_details.csv"
        )
    )


# ============================================================
# SPLIT / CACHE VALIDATION
# ============================================================

def load_split() -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_PATH}"
        )

    split = pd.read_csv(
        SPLIT_PATH
    )

    split["workload_id"] = (
        split["workload_id"]
        .astype(int)
    )

    split["number_of_deliveries"] = (
        split["number_of_deliveries"]
        .astype(int)
    )

    split["split"] = (
        split["split"]
        .astype(str)
    )

    if len(split) != 108:
        raise ValueError(
            "Expected 108 workloads in split."
        )

    train_count = int(
        (
            split["split"]
            == "train"
        ).sum()
    )

    validation_count = int(
        (
            split["split"]
            == "validation"
        ).sum()
    )

    test_count = int(
        (
            split["split"]
            == "test"
        ).sum()
    )

    if train_count != 76:
        raise ValueError(
            f"Expected 76 train workloads, "
            f"found {train_count}."
        )

    if validation_count != 16:
        raise ValueError(
            f"Expected 16 validation workloads, "
            f"found {validation_count}."
        )

    if test_count != 16:
        raise ValueError(
            f"Expected 16 test workloads, "
            f"found {test_count}."
        )

    return split


def validate_cache(
    split: pd.DataFrame,
) -> None:
    if not CACHE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Missing road-cache manifest: "
            f"{CACHE_MANIFEST_PATH}"
        )

    cache_manifest = pd.read_csv(
        CACHE_MANIFEST_PATH
    )

    cache_manifest[
        "workload_id"
    ] = cache_manifest[
        "workload_id"
    ].astype(int)

    if len(cache_manifest) != 92:
        raise ValueError(
            "Expected exactly 92 cached "
            "Train+Validation workloads, "
            f"found {len(cache_manifest)}."
        )

    development_ids = set(
        split.loc[
            split["split"].isin(
                [
                    "train",
                    "validation",
                ]
            ),
            "workload_id",
        ].astype(int)
    )

    cached_ids = set(
        cache_manifest[
            "workload_id"
        ].astype(int)
    )

    if development_ids != cached_ids:
        raise ValueError(
            "Road cache does not match frozen "
            "Train+Validation split."
        )

    test_ids = set(
        split.loc[
            split["split"]
            == "test",
            "workload_id",
        ].astype(int)
    )

    leaked_test = (
        cached_ids
        & test_ids
    )

    if leaked_test:
        raise ValueError(
            "Test workload found in development "
            f"road cache: {sorted(leaked_test)}"
        )


# ============================================================
# CACHE → ENVIRONMENT
# ============================================================

def load_environment_from_cache(
    *,
    workload_id: int,
    objective_config: ObjectiveConfig,
) -> ProposedRoutingEnvironment:
    path = (
        CACHE_DIR
        / f"workload_{workload_id}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing cache for workload "
            f"{workload_id}: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        cached_id = int(
            data[
                "workload_id"
            ][0]
        )

        if cached_id != workload_id:
            raise ValueError(
                "Cache workload ID mismatch."
            )

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

        cluster_ids = (
            data[
                "cluster_ids"
            ]
            .astype(int)
            .tolist()
        )

    if len(deadlines) != n:
        raise ValueError(
            f"Deadline count mismatch for "
            f"workload {workload_id}."
        )

    if len(cluster_ids) != n:
        raise ValueError(
            f"Cluster count mismatch for "
            f"workload {workload_id}."
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
            objective_config
        ),
    )


def load_workload_environments(
    *,
    split: pd.DataFrame,
    split_name: str,
    objective_config: ObjectiveConfig,
) -> tuple[
    WorkloadEnvironment,
    ...,
]:
    rows = (
        split[
            split["split"]
            == split_name
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

    for row in rows.itertuples(
        index=False
    ):
        workload_id = int(
            row.workload_id
        )

        env = load_environment_from_cache(
            workload_id=(
                workload_id
            ),

            objective_config=(
                objective_config
            ),
        )

        workloads.append(
            WorkloadEnvironment(
                workload_id=(
                    workload_id
                ),

                environment=(
                    env
                ),
            )
        )

    return tuple(
        workloads
    )


# ============================================================
# AGENT
# ============================================================

def build_agent() -> ProposedRoutingAgent:
    config = ProposedAgentConfig(
        learning_rate=1e-3,

        gamma=1.0,

        epsilon_start=(
            EPSILON_START
        ),

        epsilon_min=(
            EPSILON_MIN
        ),

        epsilon_decay=(
            EPSILON_DECAY
        ),

        replay_capacity=10_000,

        batch_size=64,

        min_replay_size=128,

        target_update_interval=100,

        hidden_dimension=64,

        gradient_clip_norm=5.0,

        seed=SEED,

        device="auto",
    )

    return ProposedRoutingAgent(
        config=config
    )


# ============================================================
# CHECKPOINT
# ============================================================

def state_dict_to_cpu(
    state_dict: dict,
) -> dict:
    return {
        key: value.detach().cpu().clone()
        if torch.is_tensor(value)
        else copy.deepcopy(value)

        for key, value
        in state_dict.items()
    }


def save_checkpoint(
    *,
    path: Path,
    agent: ProposedRoutingAgent,
    epoch: int,
    config_name: str,
    objective_config: ObjectiveConfig,
    mean_validation_objective: float,
    validation_summary,
) -> None:
    checkpoint = {
        "config_name": (
            config_name
        ),

        "epoch": (
            epoch
        ),

        "seed": (
            SEED
        ),

        "epochs": (
            EPOCHS
        ),

        "episodes_per_workload": (
            EPISODES_PER_WORKLOAD
        ),

        "total_training_episodes": (
            TOTAL_TRAINING_EPISODES
        ),

        "epsilon": float(
            agent.epsilon
        ),

        "gradient_steps": int(
            agent.gradient_steps
        ),

        "feature_dimension": int(
            agent.feature_dimension
        ),

        "objective_config": (
            asdict(
                objective_config
            )
        ),

        "agent_config": (
            asdict(
                agent.config
            )
        ),

        "mean_validation_objective": (
            float(
                mean_validation_objective
            )
        ),

        "validation_summary": {
            "workloads": int(
                validation_summary.workloads
            ),

            "deliveries": int(
                validation_summary.deliveries
            ),

            "mean_distance_km": float(
                validation_summary
                .mean_distance_km
            ),

            "mean_travel_time_minutes": float(
                validation_summary
                .mean_travel_time_minutes
            ),

            "late_deliveries": int(
                validation_summary
                .late_deliveries
            ),

            "valid_deadlines": int(
                validation_summary
                .valid_deadlines
            ),

            "weighted_sla": float(
                validation_summary
                .weighted_sla
            ),

            "total_lateness_minutes": float(
                validation_summary
                .total_lateness_minutes
            ),

            "mean_lateness_minutes": float(
                validation_summary
                .mean_lateness_minutes
            ),

            "mean_cluster_switches": float(
                validation_summary
                .mean_cluster_switches
            ),
        },

        "online_network_state_dict": (
            state_dict_to_cpu(
                agent
                .online_network
                .state_dict()
            )
        ),

        "target_network_state_dict": (
            state_dict_to_cpu(
                agent
                .target_network
                .state_dict()
            )
        ),

        "optimizer_state_dict": (
            copy.deepcopy(
                agent.optimizer.state_dict()
            )
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    *,
    path: Path,
    agent: ProposedRoutingAgent,
) -> dict:
    checkpoint = torch.load(
        path,
        map_location=(
            agent.device
        ),
        weights_only=False,
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

    agent.optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    agent.epsilon = float(
        checkpoint[
            "epsilon"
        ]
    )

    agent.gradient_steps = int(
        checkpoint[
            "gradient_steps"
        ]
    )

    return checkpoint


# ============================================================
# VALIDATION
# ============================================================

def evaluate_validation(
    *,
    workloads: tuple[
        WorkloadEnvironment,
        ...,
    ],
    encoder: CandidateEncoder,
    agent: ProposedRoutingAgent,
) -> tuple[
    tuple,
    object,
    float,
]:
    (
        evaluations,
        summary,
    ) = evaluate_workloads(
        workloads=(
            workloads
        ),

        encoder=(
            encoder
        ),

        agent=(
            agent
        ),
    )

    mean_validation_objective = mean(
        result.objective_cost
        for result in evaluations
    )

    return (
        evaluations,
        summary,
        float(
            mean_validation_objective
        ),
    )


def evaluations_to_dataframe(
    *,
    config_name: str,
    evaluations: tuple,
) -> pd.DataFrame:
    rows = []

    for result in evaluations:
        rows.append(
            {
                "config_name": (
                    config_name
                ),

                "workload_id": (
                    result.workload_id
                ),

                "number_of_deliveries": (
                    result.number_of_deliveries
                ),

                "objective_cost": (
                    result.objective_cost
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

                "lateness_minutes": (
                    result.lateness_minutes
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

                "cluster_switches": (
                    result.cluster_switches
                ),

                "route_actions": str(
                    result.route_actions
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TRAIN ONE OBJECTIVE
# ============================================================

def train_one_model(
    *,
    config_name: str,
    objective_config: ObjectiveConfig,
    split: pd.DataFrame,
) -> dict[str, object]:
    print()
    print("=" * 82)

    print(
        f"TRAINING MODEL: "
        f"{config_name}"
    )

    print("=" * 82)

    print(
        "Objective weights:"
    )

    print(
        f"  distance      = "
        f"{objective_config.distance_weight:.3f}"
    )

    print(
        f"  travel time   = "
        f"{objective_config.travel_time_weight:.3f}"
    )

    print(
        f"  late delivery = "
        f"{objective_config.late_delivery_weight:.3f}"
    )

    print(
        f"  lateness      = "
        f"{objective_config.lateness_weight:.3f}"
    )

    print(
        f"  cluster       = "
        f"{objective_config.cluster_switch_weight:.3f}"
    )

    print()

    print(
        f"Epochs: "
        f"{EPOCHS}"
    )

    print(
        f"Train workloads: "
        f"{TRAIN_WORKLOADS}"
    )

    print(
        f"Episodes/workload/epoch: "
        f"{EPISODES_PER_WORKLOAD}"
    )

    print(
        f"Total training episodes: "
        f"{TOTAL_TRAINING_EPISODES}"
    )

    print(
        f"Epsilon decay: "
        f"{EPSILON_DECAY:.8f}"
    )

    # --------------------------------------------------------
    # Build environments
    # --------------------------------------------------------

    train_workloads = (
        load_workload_environments(
            split=split,

            split_name="train",

            objective_config=(
                objective_config
            ),
        )
    )

    validation_workloads = (
        load_workload_environments(
            split=split,

            split_name="validation",

            objective_config=(
                objective_config
            ),
        )
    )

    if (
        len(train_workloads)
        != TRAIN_WORKLOADS
    ):
        raise ValueError(
            "Unexpected train workload count."
        )

    if (
        len(validation_workloads)
        != VALIDATION_WORKLOADS
    ):
        raise ValueError(
            "Unexpected validation workload count."
        )

    encoder = CandidateEncoder()

    agent = build_agent()

    print(
        f"Device: "
        f"{agent.device}"
    )

    print(
        f"Feature dimension: "
        f"{agent.feature_dimension}"
    )

    # --------------------------------------------------------
    # Initial validation
    # --------------------------------------------------------

    (
        initial_evaluations,
        initial_summary,
        initial_mean_objective,
    ) = evaluate_validation(
        workloads=(
            validation_workloads
        ),

        encoder=(
            encoder
        ),

        agent=(
            agent
        ),
    )

    print()
    print(
        "INITIAL VALIDATION"
    )

    print(
        f"  mean objective = "
        f"{initial_mean_objective:.6f}"
    )

    print(
        f"  distance       = "
        f"{initial_summary.mean_distance_km:.3f} km"
    )

    print(
        f"  travel time    = "
        f"{initial_summary.mean_travel_time_minutes:.3f} min"
    )

    print(
        f"  late deliveries= "
        f"{initial_summary.late_deliveries}"
    )

    print(
        f"  SLA            = "
        f"{initial_summary.weighted_sla * 100:.2f}%"
    )

    print(
        f"  lateness       = "
        f"{initial_summary.total_lateness_minutes:.3f} min"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history_rows: list[
        dict[str, object]
    ] = []

    best_mean_objective = math.inf

    best_epoch = None

    best_path = checkpoint_path(
        config_name
    )

    model_start = perf_counter()

    cumulative_episodes = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_start = perf_counter()

        epoch_result = (
            train_multi_workload_epoch(
                workloads=(
                    train_workloads
                ),

                encoder=(
                    encoder
                ),

                agent=(
                    agent
                ),

                epoch=(
                    epoch
                ),

                episodes_per_workload=(
                    EPISODES_PER_WORKLOAD
                ),

                shuffle=True,

                seed=SEED,
            )
        )

        cumulative_episodes += (
            epoch_result.episodes
        )

        epoch_runtime = (
            perf_counter()
            - epoch_start
        )

        row = {
            "config_name": (
                config_name
            ),

            "epoch": (
                epoch
            ),

            "cumulative_episodes": (
                cumulative_episodes
            ),

            "mean_training_reward": (
                epoch_result
                .mean_training_reward
            ),

            "mean_training_objective": (
                epoch_result
                .mean_training_objective
            ),

            "epsilon_start": (
                epoch_result
                .epsilon_start
            ),

            "epsilon_end": (
                epoch_result
                .epsilon_end
            ),

            "gradient_steps": int(
                agent.gradient_steps
            ),

            "epoch_runtime_seconds": (
                epoch_runtime
            ),

            "validation_ran": (
                False
            ),

            "validation_mean_objective": (
                np.nan
            ),

            "validation_mean_distance_km": (
                np.nan
            ),

            "validation_mean_travel_time_minutes": (
                np.nan
            ),

            "validation_late_deliveries": (
                np.nan
            ),

            "validation_weighted_sla": (
                np.nan
            ),

            "validation_total_lateness_minutes": (
                np.nan
            ),

            "validation_mean_cluster_switches": (
                np.nan
            ),

            "is_best_checkpoint": (
                False
            ),
        }

        should_validate = (
            epoch % VALIDATION_INTERVAL == 0
            or epoch == EPOCHS
        )

        if should_validate:
            (
                evaluations,
                validation_summary,
                mean_validation_objective,
            ) = evaluate_validation(
                workloads=(
                    validation_workloads
                ),

                encoder=(
                    encoder
                ),

                agent=(
                    agent
                ),
            )

            row[
                "validation_ran"
            ] = True

            row[
                "validation_mean_objective"
            ] = (
                mean_validation_objective
            )

            row[
                "validation_mean_distance_km"
            ] = (
                validation_summary
                .mean_distance_km
            )

            row[
                "validation_mean_travel_time_minutes"
            ] = (
                validation_summary
                .mean_travel_time_minutes
            )

            row[
                "validation_late_deliveries"
            ] = (
                validation_summary
                .late_deliveries
            )

            row[
                "validation_weighted_sla"
            ] = (
                validation_summary
                .weighted_sla
            )

            row[
                "validation_total_lateness_minutes"
            ] = (
                validation_summary
                .total_lateness_minutes
            )

            row[
                "validation_mean_cluster_switches"
            ] = (
                validation_summary
                .mean_cluster_switches
            )

            improved = (
                mean_validation_objective
                < best_mean_objective
                - 1e-12
            )

            if improved:
                best_mean_objective = (
                    mean_validation_objective
                )

                best_epoch = (
                    epoch
                )

                row[
                    "is_best_checkpoint"
                ] = True

                save_checkpoint(
                    path=(
                        best_path
                    ),

                    agent=(
                        agent
                    ),

                    epoch=(
                        epoch
                    ),

                    config_name=(
                        config_name
                    ),

                    objective_config=(
                        objective_config
                    ),

                    mean_validation_objective=(
                        mean_validation_objective
                    ),

                    validation_summary=(
                        validation_summary
                    ),
                )

            print(
                f"Epoch {epoch:2d}/{EPOCHS} | "
                f"episodes={cumulative_episodes:4d} | "
                f"eps={agent.epsilon:.4f} | "
                f"trainJ="
                f"{epoch_result.mean_training_objective:.6f} | "
                f"valJ="
                f"{mean_validation_objective:.6f} | "
                f"D="
                f"{validation_summary.mean_distance_km:.3f} | "
                f"T="
                f"{validation_summary.mean_travel_time_minutes:.3f} | "
                f"late="
                f"{validation_summary.late_deliveries} | "
                f"SLA="
                f"{validation_summary.weighted_sla * 100:.2f}% | "
                f"L="
                f"{validation_summary.total_lateness_minutes:.3f}"
                + (
                    " | BEST"
                    if improved
                    else ""
                )
            )

        else:
            print(
                f"Epoch {epoch:2d}/{EPOCHS} | "
                f"episodes={cumulative_episodes:4d} | "
                f"eps={agent.epsilon:.4f} | "
                f"trainJ="
                f"{epoch_result.mean_training_objective:.6f}"
            )

        history_rows.append(
            row
        )

        # Autosave training history after every epoch.
        pd.DataFrame(
            history_rows
        ).to_csv(
            history_path(
                config_name
            ),
            index=False,
        )

    model_runtime = (
        perf_counter()
        - model_start
    )

    if best_epoch is None:
        raise RuntimeError(
            "No validation checkpoint was saved."
        )

    # --------------------------------------------------------
    # Reload best checkpoint
    # --------------------------------------------------------

    checkpoint = load_checkpoint(
        path=(
            best_path
        ),

        agent=(
            agent
        ),
    )

    (
        best_evaluations,
        best_summary,
        best_validation_objective,
    ) = evaluate_validation(
        workloads=(
            validation_workloads
        ),

        encoder=(
            encoder
        ),

        agent=(
            agent
        ),
    )

    validation_df = (
        evaluations_to_dataframe(
            config_name=(
                config_name
            ),

            evaluations=(
                best_evaluations
            ),
        )
    )

    validation_df.to_csv(
        validation_details_path(
            config_name
        ),
        index=False,
    )

    print()
    print(
        f"BEST {config_name}"
    )

    print(
        "-" * 82
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Mean validation objective: "
        f"{best_validation_objective:.6f}"
    )

    print(
        f"Mean distance: "
        f"{best_summary.mean_distance_km:.6f} km"
    )

    print(
        f"Mean travel time: "
        f"{best_summary.mean_travel_time_minutes:.6f} min"
    )

    print(
        f"Late deliveries: "
        f"{best_summary.late_deliveries}"
    )

    print(
        f"Weighted SLA: "
        f"{best_summary.weighted_sla * 100:.2f}%"
    )

    print(
        f"Total lateness: "
        f"{best_summary.total_lateness_minutes:.6f} min"
    )

    print(
        f"Mean cluster switches: "
        f"{best_summary.mean_cluster_switches:.6f}"
    )

    print(
        f"Training runtime: "
        f"{model_runtime / 60.0:.2f} min"
    )

    result = {
        "config_name": (
            config_name
        ),

        "best_epoch": (
            best_epoch
        ),

        "training_episodes": (
            TOTAL_TRAINING_EPISODES
        ),

        "training_runtime_seconds": (
            model_runtime
        ),

        # Stored for within-model diagnostics only.
        # Do not compare this field across A and D.
        "mean_validation_objective": (
            best_validation_objective
        ),

        "validation_workloads": (
            best_summary.workloads
        ),

        "validation_deliveries": (
            best_summary.deliveries
        ),

        "mean_distance_km": (
            best_summary
            .mean_distance_km
        ),

        "mean_travel_time_minutes": (
            best_summary
            .mean_travel_time_minutes
        ),

        "late_deliveries": (
            best_summary
            .late_deliveries
        ),

        "valid_deadlines": (
            best_summary
            .valid_deadlines
        ),

        "weighted_sla": (
            best_summary
            .weighted_sla
        ),

        "total_lateness_minutes": (
            best_summary
            .total_lateness_minutes
        ),

        "mean_lateness_minutes": (
            best_summary
            .mean_lateness_minutes
        ),

        "mean_cluster_switches": (
            best_summary
            .mean_cluster_switches
        ),

        "checkpoint_path": str(
            best_path
        ),

        "epsilon_decay": (
            EPSILON_DECAY
        ),

        "final_checkpoint_epsilon": float(
            checkpoint[
                "epsilon"
            ]
        ),
    }

    # --------------------------------------------------------
    # Cleanup before the other model
    # --------------------------------------------------------

    del best_evaluations
    del validation_df
    del train_workloads
    del validation_workloads
    del encoder
    del agent

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# A vs D VALIDATION COMPARISON
# ============================================================

def build_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:
    if set(
        results[
            "config_name"
        ]
    ) != {
        "A_current",
        "D_moderate_sla",
    }:
        raise ValueError(
            "Expected A_current and D_moderate_sla."
        )

    comparison = (
        results.copy()
    )

    a = comparison[
        comparison[
            "config_name"
        ]
        == "A_current"
    ].iloc[0]

    # Physical metric comparison.
    # Objective values are deliberately not compared.
    comparison[
        "distance_change_vs_A_pct"
    ] = (
        (
            comparison[
                "mean_distance_km"
            ]
            - float(
                a[
                    "mean_distance_km"
                ]
            )
        )
        / float(
            a[
                "mean_distance_km"
            ]
        )
        * 100.0
    )

    comparison[
        "travel_time_change_vs_A_pct"
    ] = (
        (
            comparison[
                "mean_travel_time_minutes"
            ]
            - float(
                a[
                    "mean_travel_time_minutes"
                ]
            )
        )
        / float(
            a[
                "mean_travel_time_minutes"
            ]
        )
        * 100.0
    )

    comparison[
        "sla_change_vs_A_percentage_points"
    ] = (
        (
            comparison[
                "weighted_sla"
            ]
            - float(
                a[
                    "weighted_sla"
                ]
            )
        )
        * 100.0
    )

    comparison[
        "late_delivery_change_vs_A"
    ] = (
        comparison[
            "late_deliveries"
        ]
        - int(
            a[
                "late_deliveries"
            ]
        )
    )

    baseline_lateness = float(
        a[
            "total_lateness_minutes"
        ]
    )

    if baseline_lateness > 0.0:
        comparison[
            "lateness_change_vs_A_pct"
        ] = (
            (
                comparison[
                    "total_lateness_minutes"
                ]
                - baseline_lateness
            )
            / baseline_lateness
            * 100.0
        )
    else:
        comparison[
            "lateness_change_vs_A_pct"
        ] = 0.0

    return comparison


def determine_pareto_status(
    comparison: pd.DataFrame,
) -> str:
    """
    No arbitrary final winner is selected here.

    We only determine whether one model dominates
    the other on the main external validation metrics.

    Lower is better:
        distance
        time
        late deliveries
        total lateness

    If neither dominates, the result is a trade-off
    and will be interpreted before Test is opened.
    """

    a = comparison[
        comparison[
            "config_name"
        ]
        == "A_current"
    ].iloc[0]

    d = comparison[
        comparison[
            "config_name"
        ]
        == "D_moderate_sla"
    ].iloc[0]

    metric_names = [
        "mean_distance_km",
        "mean_travel_time_minutes",
        "late_deliveries",
        "total_lateness_minutes",
    ]

    a_values = [
        float(
            a[name]
        )
        for name in metric_names
    ]

    d_values = [
        float(
            d[name]
        )
        for name in metric_names
    ]

    a_no_worse = all(
        a_value <= d_value + 1e-12

        for a_value, d_value
        in zip(
            a_values,
            d_values,
        )
    )

    a_strictly_better = any(
        a_value < d_value - 1e-12

        for a_value, d_value
        in zip(
            a_values,
            d_values,
        )
    )

    d_no_worse = all(
        d_value <= a_value + 1e-12

        for a_value, d_value
        in zip(
            a_values,
            d_values,
        )
    )

    d_strictly_better = any(
        d_value < a_value - 1e-12

        for a_value, d_value
        in zip(
            a_values,
            d_values,
        )
    )

    if (
        a_no_worse
        and a_strictly_better
    ):
        return (
            "A_DOMINATES_D"
        )

    if (
        d_no_worse
        and d_strictly_better
    ):
        return (
            "D_DOMINATES_A"
        )

    return (
        "TRADE_OFF"
    )


# ============================================================
# FINDINGS
# ============================================================

def write_findings(
    *,
    comparison: pd.DataFrame,
    pareto_status: str,
    total_runtime_seconds: float,
) -> None:
    lines = [
        (
            "PROPOSED RL MULTI-WORKLOAD "
            "VALIDATION"
        ),

        "",

        (
            f"seed = {SEED}"
        ),

        (
            f"train_workloads = "
            f"{TRAIN_WORKLOADS}"
        ),

        (
            f"validation_workloads = "
            f"{VALIDATION_WORKLOADS}"
        ),

        (
            "test_workloads_used = 0"
        ),

        (
            f"epochs = "
            f"{EPOCHS}"
        ),

        (
            "episodes_per_workload_per_epoch = "
            f"{EPISODES_PER_WORKLOAD}"
        ),

        (
            f"training_episodes_per_model = "
            f"{TOTAL_TRAINING_EPISODES}"
        ),

        (
            f"epsilon_decay = "
            f"{EPSILON_DECAY:.12f}"
        ),

        "",

        (
            "Important:"
        ),

        (
            "Objective values of A and D are not "
            "compared directly because they use "
            "different objective weights."
        ),

        "",

        (
            "External validation metrics are used "
            "for A-vs-D comparison:"
        ),

        (
            "distance, travel time, late deliveries, "
            "weighted SLA, lateness and cluster switches."
        ),

        "",

        (
            f"pareto_status = "
            f"{pareto_status}"
        ),

        "",

        (
            f"total_experiment_runtime_seconds = "
            f"{total_runtime_seconds:.6f}"
        ),
    ]

    for row in comparison.itertuples(
        index=False
    ):
        lines.extend(
            [
                "",
                (
                    f"[{row.config_name}]"
                ),

                (
                    f"best_epoch = "
                    f"{row.best_epoch}"
                ),

                (
                    f"mean_distance_km = "
                    f"{row.mean_distance_km:.6f}"
                ),

                (
                    "mean_travel_time_minutes = "
                    f"{row.mean_travel_time_minutes:.6f}"
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
                    f"{row.total_lateness_minutes:.6f}"
                ),

                (
                    "mean_cluster_switches = "
                    f"{row.mean_cluster_switches:.6f}"
                ),
            ]
        )

    lines.extend(
        [
            "",

            (
                "No final Test-set conclusion is "
                "made in this experiment."
            ),

            (
                "The objective/model configuration "
                "must be locked before Test road "
                "matrices are prepared."
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

    print("=" * 82)

    print(
        "PROPOSED RL - "
        "MULTI-WORKLOAD TRAINING / VALIDATION"
    )

    print("=" * 82)

    print()
    print(
        f"Train workloads: "
        f"{TRAIN_WORKLOADS}"
    )

    print(
        f"Validation workloads: "
        f"{VALIDATION_WORKLOADS}"
    )

    print(
        "Test workloads accessed: 0"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        "Training episodes/model: "
        f"{TOTAL_TRAINING_EPISODES}"
    )

    print(
        f"Epsilon decay: "
        f"{EPSILON_DECAY:.8f}"
    )

    split = load_split()

    validate_cache(
        split
    )

    print()
    print(
        "Frozen split / road cache check: PASS"
    )

    experiment_start = (
        perf_counter()
    )

    result_rows: list[
        dict[str, object]
    ] = []

    for (
        config_name,
        objective_config,
    ) in OBJECTIVE_CONFIGS.items():
        result = train_one_model(
            config_name=(
                config_name
            ),

            objective_config=(
                objective_config
            ),

            split=(
                split
            ),
        )

        result_rows.append(
            result
        )

        # Autosave after each complete model.
        pd.DataFrame(
            result_rows
        ).to_csv(
            COMPARISON_PATH,
            index=False,
        )

    total_runtime = (
        perf_counter()
        - experiment_start
    )

    results = pd.DataFrame(
        result_rows
    )

    comparison = build_comparison(
        results
    )

    pareto_status = (
        determine_pareto_status(
            comparison
        )
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    write_findings(
        comparison=(
            comparison
        ),

        pareto_status=(
            pareto_status
        ),

        total_runtime_seconds=(
            total_runtime
        ),
    )

    print()
    print("=" * 82)

    print(
        "FINAL VALIDATION COMPARISON"
    )

    print("=" * 82)

    display_columns = [
        "config_name",
        "best_epoch",
        "mean_distance_km",
        "mean_travel_time_minutes",
        "late_deliveries",
        "weighted_sla",
        "total_lateness_minutes",
        "mean_cluster_switches",
        "distance_change_vs_A_pct",
        "travel_time_change_vs_A_pct",
        "sla_change_vs_A_percentage_points",
        "late_delivery_change_vs_A",
        "lateness_change_vs_A_pct",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Pareto status: "
        f"{pareto_status}"
    )

    print(
        f"Total experiment runtime: "
        f"{total_runtime / 60.0:.2f} min"
    )

    print()

    print(
        "IMPORTANT: Test workloads were "
        "not accessed."
    )

    print()

    print(
        f"Comparison saved to: "
        f"{COMPARISON_PATH}"
    )

    print(
        f"Findings saved to: "
        f"{FINDINGS_PATH}"
    )


if __name__ == "__main__":
    main()