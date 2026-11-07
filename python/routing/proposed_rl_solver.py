"""
Proposed RL Route Solver.

Uses the locked PyTorch model checkpoint (D_moderate_sla_best.pt)
trained with 8 invariant candidate features to sequence delivery stops
for a single shipper route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import torch

from rl.proposed.agent import (
    ProposedAgentConfig,
    ProposedRoutingAgent,
)
from rl.proposed.candidate_encoder import CandidateEncoder
from rl.proposed.environment import ProposedRoutingEnvironment
from rl.proposed.objective import ObjectiveConfig
from rl.proposed.trainer import evaluate_agent


_CACHED_AGENT: ProposedRoutingAgent | None = None
_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "proposed_rl_multiworkload_D_moderate_sla_best.pt"
)

LOCKED_OBJECTIVE = ObjectiveConfig(
    distance_weight=0.375,
    travel_time_weight=0.375,
    late_delivery_weight=0.10,
    lateness_weight=0.10,
    cluster_switch_weight=0.05,
)


def get_proposed_agent() -> ProposedRoutingAgent:
    global _CACHED_AGENT
    if _CACHED_AGENT is not None:
        return _CACHED_AGENT

    if not _CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Proposed RL checkpoint not found at: {_CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(_CHECKPOINT_PATH, map_location="cpu")
    saved_agent_config = checkpoint.get("agent_config", {})
    if isinstance(saved_agent_config, dict):
        agent_config = ProposedAgentConfig(
            hidden_dimension=saved_agent_config.get("hidden_dimension", 64),
            learning_rate=saved_agent_config.get("learning_rate", 1e-3),
            device="cpu",
        )
    else:
        agent_config = ProposedAgentConfig(device="cpu")

    agent = ProposedRoutingAgent(
        config=agent_config,
        feature_dimension=checkpoint.get("feature_dimension", 8),
    )
    agent.online_network.load_state_dict(checkpoint["online_network_state_dict"])
    agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
    agent.epsilon = 0.0  # Pure greedy inference
    _CACHED_AGENT = agent
    return _CACHED_AGENT


def solve_proposed_rl_route_with_trace(
    *,
    delivery_ids: Sequence[str],
    distance_matrix_km: np.ndarray,
    travel_time_matrix_minutes: np.ndarray,
    deadlines_minutes: Sequence[float | None],
    cluster_ids: Sequence[int],
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Sequence delivery_ids using the Proposed RL agent and record an explainable
    step-by-step decision trace with candidate Q-scores and 8 state features.
    """
    n = len(delivery_ids)
    if n == 0:
        return [], []
    if n == 1:
        return list(delivery_ids), [{
            "step_index": 1,
            "current_node": "DEPOT",
            "chosen_delivery_id": delivery_ids[0],
            "chosen_action_index": 0,
            "chosen_q_score": 1.0,
            "candidates_evaluated": [{
                "delivery_id": delivery_ids[0],
                "action_index": 0,
                "q_score": 1.0,
                "features": [0.0] * 8,
                "feature_dict": {},
            }],
            "step_reward": 0.0,
            "reward_breakdown": {
                "distance_cost": 0.0,
                "travel_time_cost": 0.0,
                "late_delivery_cost": 0.0,
                "lateness_cost": 0.0,
                "cluster_switch_cost": 0.0,
                "total_step_cost": 0.0,
            },
            "cumulative_distance_km": float(distance_matrix_km[0, 1]),
            "cumulative_time_minutes": float(travel_time_matrix_minutes[0, 1]),
            "lateness_minutes": 0.0,
            "cluster_switches": 0,
            "partial_route": [delivery_ids[0]],
        }]

    agent = get_proposed_agent()
    encoder = CandidateEncoder()

    env = ProposedRoutingEnvironment(
        distance_matrix_km=distance_matrix_km.tolist(),
        travel_time_matrix_minutes=travel_time_matrix_minutes.tolist(),
        deadlines_minutes=deadlines_minutes,
        cluster_ids=cluster_ids,
        objective_config=LOCKED_OBJECTIVE,
    )

    env.reset()
    decision_trace = []

    feature_names = [
        "norm_distance",
        "norm_travel_time",
        "deadline_tightness",
        "slack_ratio",
        "detour_penalty",
        "cluster_alignment",
        "return_depot_impact",
        "candidate_density",
    ]

    while not env.done:
        encoded_candidates = encoder.encode_available_actions(env)
        if not encoded_candidates.candidates:
            break

        feature_matrix = encoded_candidates.feature_matrix
        with torch.no_grad():
            q_values = agent.online_network(agent._features_tensor(feature_matrix))
            q_list = q_values.squeeze(-1).tolist()
            if isinstance(q_list, float):
                q_list = [q_list]

        best_index = int(torch.argmax(q_values).item())
        action = encoded_candidates.candidates[best_index].action
        chosen_q = float(q_list[best_index])

        candidates_eval = []
        for c_idx, cand in enumerate(encoded_candidates.candidates):
            cand_feats = list(cand.features)
            candidates_eval.append({
                "action_index": int(cand.action),
                "delivery_id": delivery_ids[cand.action],
                "q_score": round(float(q_list[c_idx]), 4),
                "features": [round(float(v), 4) for v in cand_feats],
                "feature_dict": {
                    feature_names[i]: round(float(cand_feats[i]), 4)
                    for i in range(min(len(feature_names), len(cand_feats)))
                },
            })

        # Sort candidates by Q-score descending
        candidates_eval.sort(key=lambda x: x["q_score"], reverse=True)

        current_node = delivery_ids[env.route_actions[-1]] if env.route_actions else "DEPOT"

        step_res = env.step(action)

        reward_breakdown = None
        if step_res.delivery_step_cost:
            reward_breakdown = {
                "distance_cost": round(float(step_res.delivery_step_cost.distance_cost), 4),
                "travel_time_cost": round(float(step_res.delivery_step_cost.travel_time_cost), 4),
                "late_delivery_cost": round(float(step_res.delivery_step_cost.late_delivery_cost), 4),
                "lateness_cost": round(float(step_res.delivery_step_cost.lateness_cost), 4),
                "cluster_switch_cost": round(float(step_res.delivery_step_cost.cluster_switch_cost), 4),
                "total_step_cost": round(float(step_res.delivery_step_cost.total_cost), 4),
            }

        decision_trace.append({
            "step_index": len(decision_trace) + 1,
            "current_node": current_node,
            "chosen_delivery_id": delivery_ids[action],
            "chosen_action_index": int(action),
            "chosen_q_score": round(chosen_q, 4),
            "candidates_evaluated": candidates_eval,
            "step_reward": round(float(step_res.reward), 4),
            "reward_breakdown": reward_breakdown,
            "cumulative_distance_km": round(float(env.totals.total_distance_km), 2),
            "cumulative_time_minutes": round(float(env.totals.total_travel_time_minutes), 1),
            "lateness_minutes": round(float(env.totals.total_lateness_minutes), 1),
            "cluster_switches": int(env.totals.cluster_switches),
            "partial_route": [delivery_ids[a] for a in env.route_actions],
        })

    ordered_ids = [delivery_ids[a] for a in env.route_actions]
    return ordered_ids, decision_trace


def solve_proposed_rl_route(
    *,
    delivery_ids: Sequence[str],
    distance_matrix_km: np.ndarray,
    travel_time_matrix_minutes: np.ndarray,
    deadlines_minutes: Sequence[float | None],
    cluster_ids: Sequence[int],
) -> list[str]:
    """
    Sequence delivery_ids using the Proposed RL agent.
    
    distance_matrix_km and travel_time_matrix_minutes must be (N+1, N+1)
    where row/col 0 is the depot and rows/cols 1..N correspond to delivery_ids.
    """
    ordered_ids, _ = solve_proposed_rl_route_with_trace(
        delivery_ids=delivery_ids,
        distance_matrix_km=distance_matrix_km,
        travel_time_matrix_minutes=travel_time_matrix_minutes,
        deadlines_minutes=deadlines_minutes,
        cluster_ids=cluster_ids,
    )
    return ordered_ids
