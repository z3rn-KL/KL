import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "python"),
)

from experiments.shift_rl_common import (  # noqa: E402
    REFERENCES_PATH,
    RESULTS_DIR,
    fingerprint,
    fixed_instances,
    load_sampler,
)
from experiments.shift_rl_train import checkpoint_path  # noqa: E402
from rl.shift.agent import ShiftAgent, ShiftAgentConfig  # noqa: E402
from rl.shift.encoder import FEATURE_DIMENSION, ShiftEncoder  # noqa: E402
from rl.shift.policy import clone_environment, run_batch  # noqa: E402
from rl.shift.set_network import SetScorer  # noqa: E402
from rl.shift.trainer import run_episode  # noqa: E402


# name -> (checkpoint tag, seeds)
VARIANTS = {
    "dqn": ("", (42, 43, 44)),
    "dqn_residual": ("resid", (42, 43, 44)),
    "policy_gradient": ("pg", (42, 43)),
    "clone_then_policy_gradient": ("pgbc", (42, 43, 44)),
    "attention_scratch": ("set", (42, 43)),
    "attention_clone_then_pg": ("setbc", (42, 43, 44)),
}

LABEL = {
    "dqn": "DQN, Q thường",
    "dqn_residual": "DQN, Q = thưởng tức thời + tương lai",
    "policy_gradient": "Gradient chính sách từ đầu",
    "clone_then_policy_gradient": "Bắt chước, rồi gradient chính sách",
    "attention_scratch": "Attention, gradient chính sách từ đầu",
    "attention_clone_then_pg": "Attention, bắt chước rồi gradient",
    "nn_serve_all": "NN giao hết",
    "nn_stop_rule": "NN dừng khi không đáng",
    "nn_all_ls": "NN + local search",
    "nn_stop_ls": "NN dừng + local search",
}

BASELINES = ("nn_serve_all", "nn_stop_rule", "nn_all_ls", "nn_stop_ls")

# python shift_rl_final_test.py [test | test2]
SET_NAME = sys.argv[1] if len(sys.argv) > 1 else "test"

SUFFIX = "" if SET_NAME == "test" else f"_{SET_NAME}"

DETAILS_PATH = RESULTS_DIR / f"shift_rl_final_test{SUFFIX}_details.csv"
FINDINGS_PATH = RESULTS_DIR / f"shift_rl_final_test{SUFFIX}_findings.txt"
SUMMARY_PATH = RESULTS_DIR / f"shift_rl_summary{SUFFIX}.json"


def load_runner(tag: str, seed: int, encoder):
    """
    A function that plays one shift greedily and returns its summary, plus
    the checkpoint.
    """

    checkpoint = torch.load(
        checkpoint_path(seed, tag),
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("network") == "set":
        network = SetScorer(FEATURE_DIMENSION, 128)
        network.load_state_dict(checkpoint["online_state_dict"])

        def run(env):
            copy = clone_environment(env)
            run_batch([copy], encoder, network, greedy=True)

            return copy.summary()

        return run, checkpoint

    agent = ShiftAgent(ShiftAgentConfig(**checkpoint["agent_config"]))
    agent.online.load_state_dict(checkpoint["online_state_dict"])

    return (lambda env: run_episode(env, encoder, agent)), checkpoint


def gap(reference, score):
    return (reference - score) / abs(reference) * 100.0


def main() -> None:
    torch.set_num_threads(1)

    sampler, splits = load_sampler()

    references = (
        pd.read_csv(REFERENCES_PATH)
        .query("set == @SET_NAME")
        .sort_values("index")
        .reset_index(drop=True)
    )

    instances = fixed_instances(sampler, splits, SET_NAME)

    assert len(instances) == len(references)
    assert all(
        fingerprint(env) == row["fingerprint"]
        for (_, env), (_, row) in zip(instances, references.iterrows())
    ), "Test shifts changed since the references were computed."

    encoder = ShiftEncoder()
    rows = references.copy()

    runs = {}  # (variant, seed) -> validation gap, mean seconds

    for variant, (tag, seeds) in VARIANTS.items():
        for seed in seeds:
            run, checkpoint = load_runner(tag, seed, encoder)

            scores, late, deferred, overtime, seconds = [], [], [], [], []

            for _, env in instances:
                tic = perf_counter()
                summary = run(env)
                seconds.append(perf_counter() - tic)

                scores.append(summary["score"])
                late.append(summary["late"])
                deferred.append(summary["deferred"])
                overtime.append(summary["overtime_minutes"])

            column = f"{variant}_{seed}"
            rows[column] = scores
            rows[f"{column}_late"] = late
            rows[f"{column}_deferred"] = deferred
            rows[f"{column}_overtime"] = overtime

            runs[(variant, seed)] = {
                "validation_gap": checkpoint["validation_gap_pct"],
                "episode": checkpoint["episode"],
                "seconds": float(np.mean(seconds)),
            }

    rows.to_csv(DETAILS_PATH, index=False)

    # One headline model, chosen on validation only.
    headline = min(runs, key=lambda key: runs[key]["validation_gap"])
    headline_column = f"{headline[0]}_{headline[1]}"

    def variant_gaps(variant, part):
        seeds = VARIANTS[variant][1]

        return np.array(
            [gap(part["reference"], part[f"{variant}_{s}"]).mean() for s in seeds]
        )

    def best_seed(variant):
        seeds = VARIANTS[variant][1]

        return min(seeds, key=lambda s: runs[(variant, s)]["validation_gap"])

    def groups():
        yield "tất cả", rows

        for n in sorted(rows["n"].unique()):
            yield f"n = {n}", rows[rows["n"] == n]

    lines = [
        f"SHIFT RL: KIỂM TRA CUỐI TRÊN TẬP {SET_NAME.upper()} (chạm một lần)",
        "",
        "Hàm điểm theo ca làm (hoa hồng, phạt trễ, làm thêm giờ, biên lợi nhuận,",
        "dời đơn); mọi hệ số tiền là giả định, dữ liệu XeDu không có giá đơn.",
        f"Ca dựng từ đơn thật trong các ngày test, {len(rows)} ca. Tham chiếu: nghiệm",
        "tối ưu chính xác khi n <= 14, còn lại là tốt nhất tìm được bằng local",
        "search nhiều điểm xuất phát (cận dưới của tối ưu, có thể bị vượt).",
        "Khoảng cách = (tham chiếu - điểm) / |tham chiếu|, càng thấp càng tốt.",
        "Các phiên bản và seed được chọn bằng validation; test không dùng để chọn.",
        "",
        "KIỂM ĐỊNH (validation) của từng lần chạy",
    ]

    for (variant, seed), info in runs.items():
        lines.append(
            f"  {LABEL[variant]:42s} seed {seed}: khoảng cách validation "
            f"{info['validation_gap']:6.2f}% (checkpoint ở bước {info['episode']})"
        )

    lines += [
        "",
        f"Mô hình chính (validation tốt nhất): {LABEL[headline[0]]}, seed {headline[1]}",
        "",
    ]

    summary_groups = []

    for title, part in groups():
        entry = {"group": title, "shifts": int(len(part)), "methods": []}
        summary_groups.append(entry)

        lines.append(f"[{title}] {len(part)} ca")
        lines.append(
            f"  {'phương pháp':42s} {'khoảng cách TB (%)':>19s} {'đạt tham chiếu':>15s}"
        )

        for baseline in BASELINES:
            g = gap(part["reference"], part[baseline])

            lines.append(
                f"  {LABEL[baseline]:42s} {g.mean():19.2f} "
                f"{int((g < 1e-4).sum()):>11d}/{len(part)}"
            )

            entry["methods"].append(
                {
                    "key": baseline,
                    "label": LABEL[baseline],
                    "kind": "heuristic",
                    "mean_gap_pct": float(g.mean()),
                    "std_gap_pct": None,
                    "at_reference": int((g < 1e-4).sum()),
                }
            )

        for variant in VARIANTS:
            per_seed = variant_gaps(variant, part)
            chosen = gap(
                part["reference"], part[f"{variant}_{best_seed(variant)}"]
            )

            lines.append(
                f"  {LABEL[variant]:42s} {per_seed.mean():12.2f} ± {per_seed.std():4.2f}"
                f"   (seed chọn: {chosen.mean():.2f}%)"
            )

            entry["methods"].append(
                {
                    "key": variant,
                    "label": LABEL[variant],
                    "kind": "rl",
                    "mean_gap_pct": float(per_seed.mean()),
                    "std_gap_pct": float(per_seed.std()),
                    "at_reference": int((chosen < 1e-4).sum()),
                }
            )

        lines.append("")

    lines.append(f"SO SÁNH CẶP MÔ HÌNH CHÍNH VỚI NN + LOCAL SEARCH (điểm số, tất cả ca)")

    for baseline in ("nn_all_ls", "nn_stop_ls"):
        diff = rows[headline_column] - rows[baseline]

        lines.append(
            f"  so với {LABEL[baseline]:24s}: thắng {int((diff > 1).sum())}, "
            f"hòa {int((diff.abs() <= 1).sum())}, thua {int((diff < -1).sum())}, "
            f"chênh trung bình {diff.mean():+.0f} điểm/ca"
        )

    lines += [
        "",
        "MÔ HÌNH CHÍNH, TRUNG BÌNH MỖI CA",
        f"  đơn trễ {rows[f'{headline_column}_late'].mean():.2f}, "
        f"đơn dời {rows[f'{headline_column}_deferred'].mean():.2f}, "
        f"làm thêm giờ {rows[f'{headline_column}_overtime'].mean():.1f} phút",
        "",
        "THỜI GIAN CHẠY MỖI CA",
        f"  mô hình chính suy luận: {runs[headline]['seconds'] * 1000:.1f} ms | "
        f"NN dừng + local search: {rows['ls_seconds'].mean():.3f} s",
    ]

    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pairs = {}

    for baseline in ("nn_all_ls", "nn_stop_ls"):
        diff = rows[headline_column] - rows[baseline]

        pairs[baseline] = {
            "wins": int((diff > 1).sum()),
            "ties": int((diff.abs() <= 1).sum()),
            "losses": int((diff < -1).sum()),
            "mean_difference_points": float(diff.mean()),
        }

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "test_set": SET_NAME,
                "shifts": int(len(rows)),
                "headline": {
                    "variant": headline[0],
                    "label": LABEL[headline[0]],
                    "seed": headline[1],
                    "validation_gap_pct": runs[headline]["validation_gap"],
                    "late_per_shift": float(rows[f"{headline_column}_late"].mean()),
                    "deferred_per_shift": float(rows[f"{headline_column}_deferred"].mean()),
                    "overtime_minutes_per_shift": float(rows[f"{headline_column}_overtime"].mean()),
                    "inference_ms": runs[headline]["seconds"] * 1000.0,
                },
                "local_search_seconds_per_shift": float(rows["ls_seconds"].mean()),
                "paired_vs": pairs,
                "groups": summary_groups,
                "scope_note": (
                    "Shifts are built from real orders (one K-Means cluster per "
                    "shift, deadlines from the original lead time). All money "
                    "coefficients are assumptions: XeDu data has no order price."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
