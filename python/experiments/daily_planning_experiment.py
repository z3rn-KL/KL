from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

from simulation.daily_planning import (  # noqa: E402
    ALLOCATION_POLICIES,
    OVERTIME_POLICIES,
    Order,
    PlanningConfig,
    simulate_horizon,
)


DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "xedu" / "xedu_cleaned.csv"
)

SNAPPED_PATH = (
    PROJECT_ROOT / "results" / "xedu_delivery_road_nodes_scc.csv"
)

MATRIX_PATH = (
    PROJECT_ROOT / "results" / "daily_planning_road_matrix.npz"
)

RESULTS_DIR = PROJECT_ROOT / "results"

SUMMARY_PATH = RESULTS_DIR / "daily_planning_summary.csv"
DAILY_PATH = RESULTS_DIR / "daily_planning_daily.csv"
SENSITIVITY_PATH = RESULTS_DIR / "daily_planning_sensitivity.csv"
FINDINGS_PATH = RESULTS_DIR / "daily_planning_findings.txt"

SEED = 42
HORIZON_DAYS = 30
LOAD_MULTIPLIERS = (1, 2, 4, 8, 12, 16)

# Real data: 2.3 active shippers per day on average (max 6).
# A fixed pool of 4 shippers makes the 10-hour shift binding
# once the load is scaled up.
POOL_SHIPPERS = 4
SENSITIVITY_LOAD = 12
LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh"


def load_pool() -> pd.DataFrame:
    """
    All real orders with their road-matrix node, weight and lead time.
    """

    raw = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")

    snapped = pd.read_csv(SNAPPED_PATH)

    matrix = np.load(MATRIX_PATH)

    node_to_index = {
        int(node): index
        for index, node in enumerate(matrix["road_nodes"])
    }

    raw["id"] = raw["id"].astype(str)
    snapped["delivery_id"] = snapped["delivery_id"].astype(str)

    frame = raw.merge(
        snapped,
        left_on="id",
        right_on="delivery_id",
        how="inner",
        validate="one_to_one",
    )

    frame["created"] = pd.to_datetime(
        frame["createdAt"],
        utc=True,
        format="ISO8601",
    )

    frame["expected"] = pd.to_datetime(
        frame["expectedDeliveryTime"],
        utc=True,
        format="ISO8601",
        errors="coerce",
    )

    frame["local_day"] = (
        frame["created"]
        .dt.tz_convert(LOCAL_TIMEZONE)
        .dt.normalize()
        .dt.tz_localize(None)
    )

    lead = (
        (frame["expected"] - frame["created"])
        .dt.total_seconds()
        / 60.0
    )

    # Orders are treated as released at the start of the working
    # day, so the deadline is the original lead time.
    frame["lead_minutes"] = lead.where(lead >= 0.0)

    frame["node"] = frame["road_node"].astype(int).map(
        lambda node: node_to_index[node]
    )

    frame["weight"] = frame["weight"].fillna(1.0).clip(lower=0.1)

    return frame.reset_index(drop=True)


def busiest_window(
    pool: pd.DataFrame,
    days: int,
) -> pd.DatetimeIndex:
    counts = pool.groupby("local_day").size()

    calendar = pd.date_range(
        counts.index.min(),
        counts.index.max(),
        freq="D",
    )

    counts = counts.reindex(calendar, fill_value=0)

    rolling = counts.rolling(days).sum()

    end = rolling.idxmax()

    return pd.date_range(end=end, periods=days, freq="D")


def to_order(
    row,
    release_day: int,
    suffix: str = "",
) -> Order:
    lead = row.lead_minutes

    return Order(
        order_id=f"{row.id}{suffix}",
        node=int(row.node),
        latitude=float(row.receiverLat),
        longitude=float(row.receiverLng),
        weight=float(row.weight),
        release_day=release_day,
        deadline_minutes=None if pd.isna(lead) else float(lead),
    )


def build_orders_by_day(
    pool: pd.DataFrame,
    window: pd.DatetimeIndex,
    multiplier: int,
) -> dict[int, list[Order]]:
    """
    Real orders for each day of the window. For multiplier m > 1,
    (m - 1) * n extra orders per day are resampled from the whole
    dataset, which keeps the real spatial/weight/deadline mix.
    """

    rng = np.random.default_rng(SEED + multiplier)

    by_day: dict[int, list[Order]] = {}

    for day_index, day in enumerate(window):
        real = pool[pool["local_day"] == day]

        orders = [
            to_order(row, day_index)
            for row in real.itertuples()
        ]

        extra = (multiplier - 1) * len(real)

        if extra > 0:
            picks = rng.integers(0, len(pool), size=extra)

            orders.extend(
                to_order(
                    pool.iloc[int(pick)],
                    day_index,
                    suffix=f"#x{position}",
                )
                for position, pick in enumerate(picks)
            )

        by_day[day_index] = orders

    return by_day


def summarise(
    result,
    label: dict,
) -> dict:
    days = result.days

    work_minutes = sum(day.work_minutes for day in days)
    shipper_days = sum(day.shippers_used for day in days)

    delivered = result.delivered

    return {
        **label,
        "orders": result.total_orders,
        "delivered": delivered,
        "final_backlog": result.final_backlog,
        "on_time_rate": (
            result.delivered_on_time / result.total_orders
            if result.total_orders
            else 1.0
        ),
        "same_day_rate": (
            result.delivered_on_time / delivered
            if delivered
            else 1.0
        ),
        "mean_backlog": float(
            np.mean([day.deferred_out for day in days])
        ),
        "max_backlog": max(day.deferred_out for day in days),
        "order_days_deferred": sum(day.deferred_out for day in days),
        "overtime_hours": (
            sum(day.overtime_minutes for day in days) / 60.0
        ),
        "over_cap_shipper_days": sum(
            day.over_cap_shipper_days for day in days
        ),
        "mean_shippers_used": shipper_days / len(days),
        "distance_km": sum(day.distance_km for day in days),
        "wage_cost": sum(day.wage_cost for day in days),
        "overtime_cost": sum(day.overtime_cost for day in days),
        "distance_cost": sum(day.distance_cost for day in days),
        "defer_cost": sum(day.defer_cost for day in days),
        "late_cost": sum(day.late_cost for day in days),
        "total_cost": result.total_cost,
        "cost_per_order": (
            result.total_cost / result.total_orders
            if result.total_orders
            else 0.0
        ),
        "work_hours": work_minutes / 60.0,
    }


def run_grid(
    pool,
    window,
    distance_km,
    travel_time,
    base: PlanningConfig,
    multipliers,
    daily_rows: list | None = None,
    extra_label: dict | None = None,
) -> list[dict]:
    rows = []

    for multiplier in multipliers:
        orders_by_day = build_orders_by_day(
            pool,
            window,
            multiplier,
        )

        for allocation in ALLOCATION_POLICIES:
            for policy in OVERTIME_POLICIES:
                config = replace(
                    base,
                    allocation=allocation,
                    overtime_policy=policy,
                )

                result = simulate_horizon(
                    orders_by_day,
                    distance_km=distance_km,
                    travel_time=travel_time,
                    config=config,
                    n_days=len(window),
                )

                label = {
                    "load_multiplier": multiplier,
                    "allocation": allocation,
                    "overtime_policy": policy,
                    **(extra_label or {}),
                }

                rows.append(summarise(result, label))

                if daily_rows is not None:
                    for day in result.days:
                        daily_rows.append(
                            {
                                **label,
                                "day": day.day,
                                "incoming": day.incoming_orders,
                                "backlog_in": day.backlog_in,
                                "delivered": day.delivered,
                                "deferred_out": day.deferred_out,
                                "shippers_used": day.shippers_used,
                                "overtime_minutes": day.overtime_minutes,
                                "total_cost": day.total_cost,
                            }
                        )

    return rows


def best_row(group: pd.DataFrame) -> pd.Series:
    """
    Cheapest row among policies that respect the overtime cap.
    """

    legal = group[group["over_cap_shipper_days"] == 0]

    return legal.loc[legal["total_cost"].idxmin()]


def write_findings(summary: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    lines = [
        "DAILY PLANNING: OVERTIME / DEFERRAL AND CLUSTER ALLOCATION",
        "",
        f"horizon_days = {HORIZON_DAYS}",
        f"shipper_pool = {POOL_SHIPPERS}, shift = 10h, service time = 10 min/stop",
        "Load multiplier 1 is the real XeDu data; higher values are",
        "stress scenarios resampled from the same dataset.",
        "Cost coefficients are scenario parameters, not real XeDu costs.",
        f"over_cap = shipper-days with overtime above the legal-style cap of",
        f"{int(PlanningConfig().max_overtime_minutes)} min/day; only the hybrid and defer policies respect it.",
        "",
    ]

    for multiplier, group in summary.groupby("load_multiplier"):
        lines.append(f"[load x{multiplier}]  orders = {int(group['orders'].iloc[0])}")

        best = best_row(group)
        cheapest = group.loc[group["total_cost"].idxmin()]

        for row in group.itertuples():
            lines.append(
                f"  {row.allocation:11s} {row.overtime_policy:8s} "
                f"cost/order = {row.cost_per_order:10.0f}  "
                f"on_time = {row.on_time_rate:.3f}  "
                f"OT_h = {row.overtime_hours:7.1f}  "
                f"over_cap = {int(row.over_cap_shipper_days):4d}  "
                f"final_backlog = {int(row.final_backlog):4d}  "
                f"mean_shippers = {row.mean_shippers_used:5.2f}"
            )

        lines.append(
            f"  best within cap = {best['allocation']} + "
            f"{best['overtime_policy']}"
        )

        if cheapest["over_cap_shipper_days"] > 0:
            lines.append(
                f"  cheapest overall = {cheapest['allocation']} + "
                f"{cheapest['overtime_policy']}, but it breaks the cap "
                f"on {int(cheapest['over_cap_shipper_days'])} shipper-days"
            )

        lines.append("")

    lines.append(f"SENSITIVITY (load x{SENSITIVITY_LOAD}): best policy within the cap")

    for (name, value), group in sensitivity.groupby(
        ["parameter", "value"]
    ):
        best = best_row(group)

        lines.append(
            f"  {name} = {value}: best within cap = {best['allocation']} + "
            f"{best['overtime_policy']} "
            f"(cost/order = {best['cost_per_order']:.0f})"
        )

    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()

    matrix = np.load(MATRIX_PATH)
    distance_km = matrix["distance_km"]
    travel_time = matrix["travel_time_minutes"]

    pool = load_pool()

    window = busiest_window(pool, HORIZON_DAYS)

    in_window = int(pool["local_day"].isin(window).sum())

    print(
        f"Window {window[0].date()} -> {window[-1].date()}, "
        f"{in_window} real orders"
    )

    base = PlanningConfig(seed=SEED, n_shippers=POOL_SHIPPERS)

    daily_rows: list[dict] = []

    summary = pd.DataFrame(
        run_grid(
            pool,
            window,
            distance_km,
            travel_time,
            base,
            LOAD_MULTIPLIERS,
            daily_rows=daily_rows,
        )
    )

    print(
        summary[
            [
                "load_multiplier",
                "allocation",
                "overtime_policy",
                "on_time_rate",
                "overtime_hours",
                "final_backlog",
                "mean_shippers_used",
                "cost_per_order",
            ]
        ].to_string(index=False)
    )

    # ----------------------------------------------------
    # Sensitivity at a heavy load (x5)
    # ----------------------------------------------------

    sensitivity_rows: list[dict] = []

    variations = {
        "defer_penalty_per_order_day": (20_000.0, 60_000.0, 150_000.0),
        "overtime_multiplier": (1.0, 1.5, 2.0),
        "max_overtime_minutes": (60.0, 120.0, 180.0),
        "n_shippers": (3, 4, 6),
        "small_cluster_threshold": (2, 3, 5),
    }

    for name, values in variations.items():
        for value in values:
            sensitivity_rows.extend(
                run_grid(
                    pool,
                    window,
                    distance_km,
                    travel_time,
                    replace(base, **{name: value}),
                    (SENSITIVITY_LOAD,),
                    extra_label={"parameter": name, "value": value},
                )
            )

    sensitivity = pd.DataFrame(sensitivity_rows)

    summary.to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(daily_rows).to_csv(DAILY_PATH, index=False)
    sensitivity.to_csv(SENSITIVITY_PATH, index=False)

    write_findings(summary, sensitivity)

    print()
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {FINDINGS_PATH}")
    print(f"Runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
