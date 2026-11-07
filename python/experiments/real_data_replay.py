import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import osmnx as ox
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

import experiments.daily_planning_experiment as daily  # noqa: E402
from experiments.daily_planning_prepare_matrix import (  # noqa: E402
    all_pairs_between,
    build_sparse_graph,
)
from experiments.q_learning_single_workload_experiment import (  # noqa: E402
    get_depot_node,
    load_road_network,
)
from experiments.resnap_deliveries_to_depot_scc import (  # noqa: E402
    build_depot_scc_graph,
    find_depot_scc,
)
from routing.open_path import (  # noqa: E402
    improve_path,
    nearest_path,
    optimal_path,
    path_cost,
)


RESULTS_DIR = PROJECT_ROOT / "results"
DETAILS_PATH = RESULTS_DIR / "real_replay_clean_shifts.csv"
FINDINGS_PATH = RESULTS_DIR / "real_replay_findings.txt"
SUMMARY_PATH = RESULTS_DIR / "real_replay_summary.json"

LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh"

MIN_ORDERS = 3
MAX_ORDERS = 13
MAX_PAIR_GAP_MINUTES = 120.0
BOOTSTRAP = 5000
SEED = 42


def load_shipper_days() -> pd.DataFrame:
    """
    Real orders with the road node of the receiver, the delivery day and
    the sender location, sorted by real delivery time.
    """

    orders = daily.load_pool()

    for column in ("createdAt", "deliveredAt", "expectedDeliveryTime"):
        orders[column] = pd.to_datetime(
            orders[column],
            utc=True,
            format="ISO8601",
            errors="coerce",
        )

    orders = orders.dropna(subset=["deliveredAt"]).copy()

    orders["delivery_day"] = (
        orders["deliveredAt"]
        .dt.tz_convert(LOCAL_TIMEZONE)
        .dt.date
    )

    orders["sender"] = (
        orders["senderLat"].round(5).astype(str)
        + ","
        + orders["senderLng"].round(5).astype(str)
    )

    return orders.sort_values(["delivery_day", "shipper_id", "deliveredAt"])


def time_model_fidelity(orders: pd.DataFrame) -> tuple[list[str], dict]:
    """
    Compares the road model's travel time between two consecutive
    deliveries of one shipper with the real time between them.
    """

    travel = np.load(daily.MATRIX_PATH)["travel_time_minutes"]

    real, model = [], []

    for _, shift in orders.groupby(["delivery_day", "shipper_id"]):
        if len(shift) < 2:
            continue

        times = shift["deliveredAt"].tolist()
        nodes = shift["node"].tolist()

        for i in range(len(shift) - 1):
            gap = (times[i + 1] - times[i]).total_seconds() / 60.0

            if 0.0 < gap <= MAX_PAIR_GAP_MINUTES:
                real.append(gap)
                model.append(travel[nodes[i], nodes[i + 1]])

    real = np.array(real)
    model = np.array(model)

    rho = spearmanr(model, real).statistic

    stats = {
        "pairs": int(len(real)),
        "max_gap_minutes": MAX_PAIR_GAP_MINUTES,
        "real_median_minutes": float(np.median(real)),
        "real_mean_minutes": float(real.mean()),
        "model_median_minutes": float(np.median(model)),
        "model_mean_minutes": float(model.mean()),
        "spearman": float(rho),
        "real_shorter_than_model_share": float((real < model).mean()),
        "median_overhead_minutes": float(np.median(real - model)),
    }

    return [
        "TIME MODEL VS REALITY",
        f"consecutive deliveries of one shipper, gap <= {MAX_PAIR_GAP_MINUTES:.0f} min: "
        f"{len(real)} pairs",
        f"real minutes between them: median {np.median(real):.1f}, "
        f"mean {real.mean():.1f}",
        f"road-model travel minutes:  median {np.median(model):.1f}, "
        f"mean {model.mean():.1f}",
        f"Spearman correlation model vs real = {rho:.2f}",
        f"share of pairs where the real gap is shorter than the model's "
        f"travel time = {(real < model).mean() * 100:.1f}%",
        f"median (real - model) = {np.median(real - model):.1f} min "
        f"(service, pickups and waiting are not in the model)",
    ], stats


def clean_shifts(orders: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Shipper-days with MIN_ORDERS..MAX_ORDERS orders, one pickup location,
    and every order created before the first delivery: a static problem
    whose depot is the real pickup point.
    """

    shifts = []

    for _, shift in orders.groupby(["delivery_day", "shipper_id"]):
        if not MIN_ORDERS <= len(shift) <= MAX_ORDERS:
            continue

        if shift["sender"].nunique() != 1:
            continue

        if shift["createdAt"].max() > shift["deliveredAt"].min():
            continue

        shifts.append(shift)

    return shifts


def deadline_order_stats(shifts: list[pd.DataFrame]) -> dict:
    """
    How closely the real delivery order follows the deadlines.
    """

    exact, rhos = 0, []

    for shift in shifts:
        if shift["expectedDeliveryTime"].isna().any():
            continue

        rank = shift["expectedDeliveryTime"].rank(method="first").to_numpy()
        exact += int((np.diff(rank) > 0).all())

        if len(set(rank)) > 1:
            rhos.append(spearmanr(range(len(shift)), rank).statistic)

    return {
        "earliest_deadline_first_shifts": exact,
        "shifts": len(shifts),
        "mean_spearman_position_vs_deadline": float(np.nanmean(rhos)),
    }


def bootstrap_mean_interval(values: np.ndarray, rng) -> tuple[float, float]:
    means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(BOOTSTRAP)
    ]

    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    started = perf_counter()

    orders = load_shipper_days()
    shifts = clean_shifts(orders)

    print(
        f"{len(orders)} delivered orders, "
        f"{orders.groupby(['delivery_day', 'shipper_id']).ngroups} shipper-days, "
        f"{len(shifts)} clean shifts"
    )

    fidelity, fidelity_stats = time_model_fidelity(orders)

    service = load_road_network()
    _, anchor = get_depot_node(service)

    scc = find_depot_scc(service.graph, anchor)
    scc_graph = build_depot_scc_graph(service.graph, scc)

    senders = sorted({shift["sender"].iloc[0] for shift in shifts})
    sender_lookup = {}

    for key in senders:
        lat, lng = (float(v) for v in key.split(","))
        sender_lookup[key] = int(ox.distance.nearest_nodes(scc_graph, X=lng, Y=lat))

    receivers = {int(n) for shift in shifts for n in shift["road_node"]}
    nodes = sorted(set(sender_lookup.values()) | receivers)

    print(f"road matrix over {len(nodes)} nodes")

    distance_graph, node_index = build_sparse_graph(service.graph, "length")
    time_graph, _ = build_sparse_graph(service.graph, "travel_time")

    km = all_pairs_between(distance_graph, node_index, nodes) / 1000.0
    minutes = all_pairs_between(time_graph, node_index, nodes) / 60.0
    position = {node: i for i, node in enumerate(nodes)}

    rows = []
    skipped = 0

    for number, shift in enumerate(shifts):
        stops = [
            sender_lookup[shift["sender"].iloc[0]],
            *[int(n) for n in shift["road_node"]],
        ]
        index = [position[node] for node in stops]

        row = {"shift": number, "orders": len(shift)}
        usable = True

        for name, matrix in (("time", minutes), ("km", km)):
            cost = matrix[np.ix_(index, index)].tolist()

            if not np.isfinite(matrix[np.ix_(index, index)]).all():
                usable = False
                break

            n = len(shift)
            real = list(range(n))
            nearest = nearest_path(cost)
            searched = improve_path(cost, nearest)
            optimum, _ = optimal_path(cost)

            row[f"real_{name}"] = path_cost(cost, real)
            row[f"nearest_{name}"] = path_cost(cost, nearest)
            row[f"nearest_search_{name}"] = path_cost(cost, searched)
            row[f"optimal_{name}"] = optimum

        if usable:
            rows.append(row)
        else:
            skipped += 1

    details = pd.DataFrame(rows)
    details.to_csv(DETAILS_PATH, index=False)

    rng = np.random.default_rng(SEED)

    late = orders.dropna(subset=["expectedDeliveryTime"])
    late_rate = (late["deliveredAt"] > late["expectedDeliveryTime"]).mean()

    lines = [
        "REAL-DATA REPLAY (XeDu, no simulated orders)",
        "",
        "Inputs are real: orders, receiver coordinates, deadlines, real delivery",
        "times and the OpenStreetMap road network. Distances and times of routes",
        "are computed on the road graph with OSM free-flow speeds; they are not",
        "measured. No algorithm was tried in the field.",
        "",
        "OBSERVED IN THE DATA",
        f"orders delivered = {len(orders)}; late against the real deadline = "
        f"{late_rate * 100:.1f}% of {len(late)} orders with a deadline",
        "",
    ]

    lines += fidelity

    lines += [
        "",
        f"CLEAN SHIFTS: {len(details)} shipper-days with {MIN_ORDERS}-{MAX_ORDERS} orders,",
        "one pickup point (used as the depot), and every order created before",
        "the first delivery. The route starts at the pickup point and ends at the",
        "last delivery (no return). The real order is the order of delivery",
        f"times. sizes: min {details['orders'].min()}, "
        f"median {details['orders'].median():.0f}, max {details['orders'].max()}; "
        f"skipped for unreachable road nodes: {skipped}",
        "",
    ]

    replay = {}

    for name, unit in (("time", "min"), ("km", "km")):
        replay[name] = []
        lines.append(f"ROAD {name.upper()} ({unit}), mean per shift")
        lines.append(
            f"{'method':24s} {'mean':>8s} {'mean excess over optimum':>26s} "
            f"{'95% CI':>18s} {'at optimum':>11s}"
        )

        optimum = details[f"optimal_{name}"].to_numpy()

        for method in ("real", "nearest", "nearest_search", "optimal"):
            values = details[f"{method}_{name}"].to_numpy()
            excess = (values / optimum - 1.0) * 100.0
            low, high = bootstrap_mean_interval(excess, rng)

            replay[name].append(
                {
                    "method": method,
                    "mean": float(values.mean()),
                    "mean_excess_pct": float(excess.mean()),
                    "ci_low_pct": low,
                    "ci_high_pct": high,
                    "at_optimum": int((excess < 1e-6).sum()),
                }
            )

            lines.append(
                f"{method:24s} {values.mean():8.2f} {excess.mean():25.2f}% "
                f"{low:8.2f}% to {high:6.2f}% {int((excess < 1e-6).sum()):>7d}/{len(details)}"
            )

        lines.append("")

    lines += [
        "Reading this: the optimum is a lower bound for a static route. Real",
        "shippers also handled traffic, service time, customer availability and",
        "orders not visible in this data, so a gap to the optimum is not a",
        "saving that can be promised. The 'clean' filter keeps a small, selected",
        "subset of all shifts.",
    ]

    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sizes = details["orders"].value_counts().sort_index()

    summary = {
        "observed": {
            "orders_delivered": int(len(orders)),
            "orders_with_deadline": int(len(late)),
            "late_rate": float(late_rate),
        },
        "time_model": fidelity_stats,
        "clean_shifts": {
            "count": int(len(details)),
            "min_orders": int(details["orders"].min()),
            "median_orders": float(details["orders"].median()),
            "max_orders": int(details["orders"].max()),
            "size_counts": {str(k): int(v) for k, v in sizes.items()},
            "skipped_unreachable": int(skipped),
        },
        "deadline_order": deadline_order_stats(shifts),
        "replay": replay,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))
    print(f"\nRuntime: {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
