from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

from experiments.q_learning_single_workload_experiment import (  # noqa: E402
    get_depot_node,
    load_road_network,
)


SNAPPED_PATH = (
    PROJECT_ROOT
    / "results"
    / "xedu_delivery_road_nodes_scc.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "daily_planning_road_matrix.npz"
)


def build_sparse_graph(
    graph,
    weight: str,
) -> tuple[csr_matrix, dict[int, int]]:
    """
    Convert the directed OSM multigraph into a CSR matrix.

    Parallel edges keep the minimum weight, matching
    networkx shortest-path behaviour on MultiDiGraph.
    """

    node_index = {
        int(node): index
        for index, node in enumerate(graph.nodes)
    }

    best: dict[tuple[int, int], float] = {}

    for u, v, data in graph.edges(data=True):
        key = (node_index[int(u)], node_index[int(v)])
        value = float(data[weight])

        if key not in best or value < best[key]:
            best[key] = value

    rows = np.fromiter((key[0] for key in best), dtype=np.int64)
    cols = np.fromiter((key[1] for key in best), dtype=np.int64)
    values = np.fromiter(best.values(), dtype=float)

    size = len(node_index)

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(size, size),
    )

    return matrix, node_index


def all_pairs_between(
    matrix: csr_matrix,
    node_index: dict[int, int],
    road_nodes: list[int],
) -> np.ndarray:
    indices = np.asarray(
        [node_index[node] for node in road_nodes]
    )

    lengths = dijkstra(
        matrix,
        directed=True,
        indices=indices,
    )

    return lengths[:, indices]


def main() -> None:
    started = time.perf_counter()

    service = load_road_network()

    _, depot_node = get_depot_node(service)

    snapped = pd.read_csv(SNAPPED_PATH)

    delivery_nodes = sorted(
        set(snapped["road_node"].astype(int))
        - {depot_node}
    )

    # Node 0 is always the depot.
    road_nodes = [depot_node, *delivery_nodes]

    print(f"Matrix nodes: {len(road_nodes)}")

    distance_graph, node_index = build_sparse_graph(
        service.graph,
        "length",
    )

    time_graph, _ = build_sparse_graph(
        service.graph,
        "travel_time",
    )

    distance_km = (
        all_pairs_between(distance_graph, node_index, road_nodes)
        / 1000.0
    )

    travel_time_minutes = (
        all_pairs_between(time_graph, node_index, road_nodes)
        / 60.0
    )

    unreachable = int(
        (~np.isfinite(distance_km)).sum()
        + (~np.isfinite(travel_time_minutes)).sum()
    )

    if unreachable:
        raise RuntimeError(
            f"{unreachable} unreachable node pairs found."
        )

    np.savez_compressed(
        OUTPUT_PATH,
        road_nodes=np.asarray(road_nodes, dtype=np.int64),
        distance_km=distance_km,
        travel_time_minutes=travel_time_minutes,
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
