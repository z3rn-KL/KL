from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PYTHON_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PROJECT_ROOT = (
    PYTHON_DIR
    .parent
)

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MAPS_DIR = (
    PROJECT_ROOT
    / "maps"
)


# ============================================================
# AUTHORITATIVE INPUTS
# ============================================================

XEDU_DATASET_PATH = (
    DATASET_DIR
    / "xedu"
    / "xedu_cleaned.csv"
)

ROAD_GRAPH_PATH = (
    MAPS_DIR
    / "xedu_drive.graphml"
)

SNAPPED_NODES_PATH = (
    RESULTS_DIR
    / "xedu_delivery_road_nodes_scc.csv"
)

KMEANS_RESULT_PATH = (
    RESULTS_DIR
    / "xedu_kmeans_clusters.csv"
)


# ============================================================
# DATASET-SCALE BENCHMARK
# ============================================================

DATASET_BENCHMARK_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_all_results.csv"
)

DATASET_BENCHMARK_SUMMARY_PATH = (
    RESULTS_DIR
    / "thesis_dataset_algorithm_summary.csv"
)

DATASET_WORKLOADS_PATH = (
    RESULTS_DIR
    / "dataset_algorithm_workloads.csv"
)


# ============================================================
# K-MEANS-GUIDED DQN SENSITIVITY
# ============================================================

SENSITIVITY_RESULTS_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_results.csv"
)

SENSITIVITY_SUMMARY_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_summary.csv"
)

SENSITIVITY_PAIRWISE_PATH = (
    RESULTS_DIR
    / "kmeans_guided_dqn_sensitivity_pairwise.csv"
)


# ============================================================
# FINAL GUIDED-DQN VALIDATION
# ============================================================

FINAL_VALIDATION_RESULTS_PATH = (
    RESULTS_DIR
    / "final_kmeans_guided_dqn_validation_results.csv"
)

FINAL_VALIDATION_SUMMARY_PATH = (
    RESULTS_DIR
    / "final_kmeans_guided_dqn_validation_summary.csv"
)

FINAL_VALIDATION_PAIRED_PATH = (
    RESULTS_DIR
    / "final_kmeans_guided_dqn_validation_paired.csv"
)


# ============================================================
# PRECOMPUTED RL ROUTES
# ============================================================

Q_LEARNING_ROUTE_PATH = (
    RESULTS_DIR
    / "q_learning_single_workload_route.csv"
)

SARSA_ROUTE_PATH = (
    RESULTS_DIR
    / "sarsa_single_workload_route.csv"
)

DQN_ROUTE_PATH = (
    RESULTS_DIR
    / "dqn_single_workload_route.csv"
)


# ============================================================
# API SETTINGS
# ============================================================

API_TITLE = (
    "Delivery Routing Optimization API"
)

API_DESCRIPTION = (
    "Backend API for the bachelor thesis "
    "combining clustering, routing heuristics, "
    "road-network evaluation and reinforcement "
    "learning for logistics."
)

API_VERSION = "0.2.0"


CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


# Keep live road-network requests small enough
# for an interactive web demo.
MAX_LIVE_ROUTE_DELIVERIES = 30