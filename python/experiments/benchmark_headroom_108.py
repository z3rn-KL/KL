from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

import experiments.daily_planning_experiment as daily  # noqa: E402
from routing.open_path import (  # noqa: E402
    improve_path,
    nearest_path,
    optimal_path,
    path_cost,
)


RESULTS_DIR = PROJECT_ROOT / "results"
WORKLOADS_PATH = RESULTS_DIR / "dataset_algorithm_workloads.csv"
BENCHMARK_PATH = RESULTS_DIR / "dataset_algorithm_all_results.csv"
GUIDED_PATH = RESULTS_DIR / "final_kmeans_guided_dqn_validation_results.csv"

DETAILS_PATH = RESULTS_DIR / "benchmark_headroom_108_details.csv"
FINDINGS_PATH = RESULTS_DIR / "benchmark_headroom_108_findings.txt"

BENCHMARK_ALGORITHMS = (
    "nearest_neighbor",
    "clarke_wright",
    "q_learning",
    "sarsa",
    "dqn",
)

BOOTSTRAP = 5000
TOLERANCE = 1e-6
SEED = 42

LABEL = {
    "nearest_neighbor": "Nearest Neighbor",
    "clarke_wright": "Clarke-Wright",
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
    "dqn": "DQN",
    "guided_dqn": "K-Means-guided DQN",
    "nn_search": "NN + local search",
    "optimal": "Tối ưu chính xác",
}


def lateness(order, times, deadlines):
    """
    Late deliveries and total minutes late of a route that leaves the depot
    at the start of the wave (arrival = cumulative road time, as the
    benchmark evaluator does).
    """

    clock, previous, late, minutes = 0.0, 0, 0, 0.0

    for k in order:
        clock += times[previous][k + 1]
        previous = k + 1

        if deadlines[k] is not None and clock > deadlines[k]:
            late += 1
            minutes += clock - deadlines[k]

    return late, minutes


def route_metrics(order, times, distances):
    return (
        path_cost(times, order, closed=True),
        path_cost(distances, order, closed=True),
    )


def main() -> None:
    started = perf_counter()

    pool = daily.load_pool()
    node_of = dict(zip(pool["id"].astype(str), pool["node"].astype(int)))
    expected_of = dict(zip(pool["id"].astype(str), pool["expected"]))

    matrix = np.load(daily.MATRIX_PATH)
    benchmark = pd.read_csv(BENCHMARK_PATH)
    guided = pd.read_csv(GUIDED_PATH).set_index("workload_id")
    workloads = pd.read_csv(WORKLOADS_PATH)

    by_key = benchmark.set_index(["workload_id", "algorithm"])

    rows = []

    for workload in workloads.itertuples():
        ids = workload.delivery_ids.split(" -> ")
        nodes = [0, *(node_of[i] for i in ids)]

        index = np.ix_(nodes, nodes)
        times = matrix["travel_time_minutes"][index].tolist()
        distances = matrix["distance_km"][index].tolist()

        wave_end = pd.to_datetime(workload.wave_end, utc=True)
        deadlines = [
            None
            if pd.isna(expected_of[i])
            else (expected_of[i] - wave_end).total_seconds() / 60.0
            for i in ids
        ]

        row = {
            "workload_id": workload.workload_id,
            "n": len(ids),
            "valid_deadlines": sum(d is not None for d in deadlines),
        }

        for unit, cost in (("min", times), ("km", distances)):
            optimum, optimum_order = optimal_path(cost, closed=True)
            nearest = nearest_path(cost)
            searched = improve_path(cost, nearest, closed=True)

            row[f"optimal_{unit}"] = optimum
            row[f"nn_search_{unit}"] = path_cost(cost, searched, closed=True)

            if unit == "km":
                row["own_nn_late"], _ = lateness(nearest, times, deadlines)
                row["nn_search_late"], row["nn_search_late_min"] = lateness(
                    searched, times, deadlines
                )
                row["optimal_km_late"], row["optimal_km_late_min"] = lateness(
                    optimum_order, times, deadlines
                )

                # Distance plus a heavy charge for lateness, in km-equivalents.
                def aware(order):
                    late, minutes = lateness(order, times, deadlines)

                    return (
                        path_cost(distances, order, closed=True)
                        + 20.0 * late
                        + 0.2 * minutes
                    )

                aware_order = improve_path(cost, searched, objective=aware)

                row["aware_km"] = path_cost(distances, aware_order, closed=True)
                row["aware_late"], row["aware_late_min"] = lateness(
                    aware_order, times, deadlines
                )

            # Our own nearest neighbour, to check the benchmark's.
            nn_time, nn_km = route_metrics(nearest, times, distances)
            row[f"own_nn_by_{unit}_min"] = nn_time
            row[f"own_nn_by_{unit}_km"] = nn_km

        for algorithm in BENCHMARK_ALGORITHMS:
            record = by_key.loc[(workload.workload_id, algorithm)]

            row[f"{algorithm}_min"] = record["total_travel_time_minutes"]
            row[f"{algorithm}_km"] = record["total_distance_km"]
            row[f"{algorithm}_late"] = record["late_deliveries"]
            row[f"{algorithm}_late_min"] = record["total_lateness_minutes"]

        if workload.workload_id in guided.index:
            record = guided.loc[workload.workload_id]

            row["guided_dqn_min"] = record["total_travel_time_minutes"]
            row["guided_dqn_km"] = record["total_distance_km"]

        rows.append(row)

    details = pd.DataFrame(rows)
    details.to_csv(DETAILS_PATH, index=False)

    # ---- consistency checks -------------------------------------------
    checks = []

    for unit in ("min", "km"):
        optimum = details[f"optimal_{unit}"]

        for algorithm in (*BENCHMARK_ALGORITHMS, "guided_dqn", "nn_search"):
            column = f"{algorithm}_{unit}"

            if column not in details:
                continue

            values = details[column].dropna()
            below = int((values < optimum[values.index] * (1 - TOLERANCE)).sum())

            checks.append(
                f"{LABEL[algorithm]:20s} {unit:3s}: {below} workloads below the "
                f"exact optimum (must be 0)"
            )

    nn = details["nearest_neighbor_km"]
    match_by_distance = int(
        np.isclose(nn, details["own_nn_by_km_km"], rtol=1e-6).sum()
    )
    match_by_time = int(
        np.isclose(nn, details["own_nn_by_min_km"], rtol=1e-6).sum()
    )

    rng = np.random.default_rng(SEED)

    def gap(frame, algorithm, unit):
        return (
            frame[f"{algorithm}_{unit}"] / frame[f"optimal_{unit}"] - 1.0
        ) * 100.0

    def interval(values):
        means = [
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(BOOTSTRAP)
        ]

        return np.percentile(means, 2.5), np.percentile(means, 97.5)

    def table(frame, algorithms, title):
        lines = [title]

        for unit, name in (("km", "QUÃNG ĐƯỜNG"), ("min", "THỜI GIAN")):
            lines.append(f"  {name}: dài hơn tối ưu, trung bình mỗi đợt")
            lines.append(
                f"  {'thuật toán':20s} {'TB (%)':>8s} {'CI 95%':>16s} "
                f"{'đúng tối ưu':>12s} {'tệ nhất (%)':>12s}"
            )

            for algorithm in algorithms:
                values = gap(frame, algorithm, unit).dropna().to_numpy()
                low, high = interval(values)

                lines.append(
                    f"  {LABEL[algorithm]:20s} {values.mean():8.2f} "
                    f"{low:7.2f} đến {high:5.2f} "
                    f"{int((values < 1e-4).sum()):>8d}/{len(values):<3d} "
                    f"{values.max():12.2f}"
                )

            lines.append("")

        return lines

    lines = [
        "BENCHMARK 108 ĐỢT SO VỚI NGHIỆM TỐI ƯU CHÍNH XÁC",
        "",
        "Tuyến đóng (quay về kho) trên đồ thị đường OSM. Nghiệm tối ưu tính",
        "riêng cho quãng đường và cho thời gian bằng quy hoạch động (đã kiểm",
        "chứng với vét cạn). Các thuật toán lấy từ kết quả benchmark đã lưu.",
        "Thời gian là tính theo mô hình (tốc độ lý tưởng), không đo thực tế.",
        "",
        "KIỂM TRA TÍNH NHẤT QUÁN",
        *checks,
        f"NN tự tính khớp NN benchmark: theo quãng đường {match_by_distance}/108, "
        f"theo thời gian {match_by_time}/108",
        "",
    ]

    algorithms = [*BENCHMARK_ALGORITHMS, "nn_search"]
    lines += table(details, algorithms, f"TẤT CẢ {len(details)} ĐỢT")

    subset = details.dropna(subset=["guided_dqn_km"])
    lines += table(
        subset,
        [*BENCHMARK_ALGORITHMS, "guided_dqn", "nn_search"],
        f"{len(subset)} ĐỢT KIỂM CHỨNG (có Guided DQN)",
    )

    total_valid = int(details["valid_deadlines"].sum())

    lines.append(f"ĐÚNG HẠN (108 đợt, {total_valid} đơn có hạn giao, tính theo mô hình)")
    lines.append(
        f"  kiểm tra cách tính: NN tự tính trễ {int(details['own_nn_late'].sum())} đơn, "
        f"NN benchmark trễ {int(details['nearest_neighbor_late'].sum())} đơn (phải bằng nhau)"
    )
    lines.append(f"  {'phương pháp':34s} {'đơn trễ':>8s} {'phút trễ':>9s} {'km TB':>8s}")

    for algorithm in BENCHMARK_ALGORITHMS:
        lines.append(
            f"  {LABEL[algorithm]:34s} "
            f"{int(details[f'{algorithm}_late'].sum()):8d} "
            f"{details[f'{algorithm}_late_min'].sum():9.1f} "
            f"{details[f'{algorithm}_km'].mean():8.2f}"
        )

    for label, late, minutes, km in (
        ("NN + local search (chỉ quãng đường)", "nn_search_late", "nn_search_late_min", "nn_search_km"),
        ("Tối ưu quãng đường (chính xác)", "optimal_km_late", "optimal_km_late_min", "optimal_km"),
        ("NN + local search có tính hạn giao", "aware_late", "aware_late_min", "aware_km"),
    ):
        lines.append(
            f"  {label:34s} {int(details[late].sum()):8d} "
            f"{details[minutes].sum():9.1f} {details[km].mean():8.2f}"
        )

    lines.append("")
    lines.append("SO SÁNH CẶP VỚI NN + LOCAL SEARCH (quãng đường, 108 đợt)")

    for algorithm in ("dqn", "clarke_wright", "q_learning", "sarsa"):
        diff = (
            details[f"{algorithm}_km"] - details["nn_search_km"]
        ).to_numpy()
        wins = int((diff < -1e-6).sum())
        losses = int((diff > 1e-6).sum())
        ties = len(diff) - wins - losses

        try:
            p_value = wilcoxon(diff[np.abs(diff) > 1e-9]).pvalue
        except ValueError:
            p_value = float("nan")

        lines.append(
            f"  {LABEL[algorithm]:20s} thắng {wins:3d}, hòa {ties:3d}, thua "
            f"{losses:3d} | chênh TB {diff.mean():+.3f} km | Wilcoxon p = {p_value:.2g}"
        )

    lines.append("")
    lines.append("THEO CỠ ĐỢT (quãng đường, dài hơn tối ưu, trung bình %)")
    lines.append(
        f"  {'cỡ':10s} {'số đợt':>7s} "
        + " ".join(f"{LABEL[a][:12]:>13s}" for a in ("nearest_neighbor", "clarke_wright", "dqn", "nn_search"))
    )

    for name, low, high in (("5-7", 5, 7), ("8-10", 8, 10), ("11-16", 11, 16)):
        part = details[(details["n"] >= low) & (details["n"] <= high)]

        lines.append(
            f"  {name:10s} {len(part):7d} "
            + " ".join(
                f"{gap(part, a, 'km').mean():13.2f}"
                for a in ("nearest_neighbor", "clarke_wright", "dqn", "nn_search")
            )
        )

    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nRuntime: {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
