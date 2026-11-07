from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

import experiments.final_kmeans_guided_dqn_validation as validation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

SPLIT_PATH = (
    RESULTS_DIR
    / "proposed_rl_workload_split.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "proposed_rl_workload_split_summary.csv"
)

FINDINGS_PATH = (
    RESULTS_DIR
    / "proposed_rl_workload_split_findings.txt"
)


SEED = 42

TARGET_TRAIN = 76
TARGET_VALIDATION = 16
TARGET_TEST = 16

# Các workload đã dùng trong pilot sensitivity.
PILOT_WORKLOAD_IDS = {
    101,
    106,
    97,
}

# N=5..13 có đủ mẫu để xuất hiện trong
# cả development và held-out test.
COMMON_SIZES = tuple(
    range(5, 14)
)

# N=14,15,16 mỗi size chỉ có đúng một workload.
# Không thể stratify hợp lý nên giữ trong train.
RARE_TRAIN_ONLY_SIZES = {
    14,
    15,
    16,
}


def load_workloads() -> pd.DataFrame:
    manifest = (
        validation.load_workload_manifest()
        .copy()
    )

    required_columns = {
        "workload_id",
        "number_of_deliveries",
    }

    missing = (
        required_columns
        - set(manifest.columns)
    )

    if missing:
        raise ValueError(
            "Manifest is missing columns: "
            f"{sorted(missing)}"
        )

    manifest = (
        manifest[
            [
                "workload_id",
                "number_of_deliveries",
            ]
        ]
        .drop_duplicates(
            subset=["workload_id"]
        )
        .copy()
    )

    manifest["workload_id"] = (
        manifest["workload_id"]
        .astype(int)
    )

    manifest["number_of_deliveries"] = (
        manifest["number_of_deliveries"]
        .astype(int)
    )

    manifest = (
        manifest
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

    if len(manifest) != 108:
        raise ValueError(
            "Expected exactly 108 workloads, "
            f"found {len(manifest)}."
        )

    return manifest


def take_one_size(
    *,
    pool: list[dict[str, int]],
    size: int,
    rng: random.Random,
) -> dict[str, int]:
    """
    Lấy ngẫu nhiên một workload có đúng size=N
    khỏi pool.
    """

    indices = [
        index
        for index, record in enumerate(pool)
        if int(
            record["number_of_deliveries"]
        ) == size
    ]

    if not indices:
        raise ValueError(
            f"No remaining workload for N={size}."
        )

    selected_index = rng.choice(
        indices
    )

    return pool.pop(
        selected_index
    )


def size_counts(
    pool: list[dict[str, int]],
) -> dict[int, int]:
    counts: dict[int, int] = {}

    for record in pool:
        size = int(
            record["number_of_deliveries"]
        )

        counts[size] = (
            counts.get(size, 0)
            + 1
        )

    return counts


def take_validation_filler(
    *,
    pool: list[dict[str, int]],
    rng: random.Random,
) -> dict[str, int]:
    """
    Khi bổ sung validation, luôn giữ lại ít nhất
    một workload của mỗi N=5..13 cho Test.
    """

    counts = size_counts(
        pool
    )

    eligible_indices = [
        index
        for index, record in enumerate(pool)
        if (
            int(
                record[
                    "number_of_deliveries"
                ]
            )
            in COMMON_SIZES
            and counts[
                int(
                    record[
                        "number_of_deliveries"
                    ]
                )
            ]
            > 1
        )
    ]

    if not eligible_indices:
        raise RuntimeError(
            "Cannot fill validation while "
            "preserving test size coverage."
        )

    selected_index = rng.choice(
        eligible_indices
    )

    return pool.pop(
        selected_index
    )


def build_split(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rng = random.Random(
        SEED
    )

    records = (
        manifest
        .to_dict("records")
    )

    rare_train: list[
        dict[str, int]
    ] = []

    common_pool: list[
        dict[str, int]
    ] = []

    for record in records:
        size = int(
            record["number_of_deliveries"]
        )

        if size in RARE_TRAIN_ONLY_SIZES:
            rare_train.append(
                record
            )
        else:
            common_pool.append(
                record
            )

    # --------------------------------------------------------
    # 1. Remove pilot workloads from common pool.
    # --------------------------------------------------------

    validation_records: list[
        dict[str, int]
    ] = []

    remaining_pool: list[
        dict[str, int]
    ] = []

    for record in common_pool:
        workload_id = int(
            record["workload_id"]
        )

        if workload_id in PILOT_WORKLOAD_IDS:
            validation_records.append(
                record
            )
        else:
            remaining_pool.append(
                record
            )

    # --------------------------------------------------------
    # 2. Validation phải có ít nhất một workload
    #    cho mỗi N=5..13.
    #
    # Pilot đã cung cấp:
    # N=5, N=10, N=13.
    # --------------------------------------------------------

    validation_sizes = {
        int(
            record["number_of_deliveries"]
        )
        for record in validation_records
    }

    for size in COMMON_SIZES:
        if size in validation_sizes:
            continue

        selected = take_one_size(
            pool=remaining_pool,
            size=size,
            rng=rng,
        )

        validation_records.append(
            selected
        )

        validation_sizes.add(
            size
        )

    # --------------------------------------------------------
    # 3. Fill validation đến đúng 16 workloads,
    #    nhưng bảo vệ một mẫu mỗi size cho test.
    # --------------------------------------------------------

    while (
        len(validation_records)
        < TARGET_VALIDATION
    ):
        selected = take_validation_filler(
            pool=remaining_pool,
            rng=rng,
        )

        validation_records.append(
            selected
        )

    if (
        len(validation_records)
        != TARGET_VALIDATION
    ):
        raise RuntimeError(
            "Validation allocation failed."
        )

    # --------------------------------------------------------
    # 4. Test bắt buộc có N=5..13.
    # --------------------------------------------------------

    test_records: list[
        dict[str, int]
    ] = []

    for size in COMMON_SIZES:
        selected = take_one_size(
            pool=remaining_pool,
            size=size,
            rng=rng,
        )

        test_records.append(
            selected
        )

    # --------------------------------------------------------
    # 5. Fill test đến đúng 16.
    # --------------------------------------------------------

    while (
        len(test_records)
        < TARGET_TEST
    ):
        selected_index = rng.randrange(
            len(remaining_pool)
        )

        selected = remaining_pool.pop(
            selected_index
        )

        test_records.append(
            selected
        )

    if (
        len(test_records)
        != TARGET_TEST
    ):
        raise RuntimeError(
            "Test allocation failed."
        )

    # --------------------------------------------------------
    # 6. Tất cả còn lại -> Train.
    #
    # Bao gồm toàn bộ N=14,15,16.
    # --------------------------------------------------------

    train_records = (
        remaining_pool
        + rare_train
    )

    if len(train_records) != TARGET_TRAIN:
        raise RuntimeError(
            "Expected "
            f"{TARGET_TRAIN} train workloads, "
            f"found {len(train_records)}."
        )

    rows: list[
        dict[str, object]
    ] = []

    for record in train_records:
        rows.append(
            {
                **record,
                "split": "train",
                "is_pilot_workload": False,
            }
        )

    for record in validation_records:
        workload_id = int(
            record["workload_id"]
        )

        rows.append(
            {
                **record,
                "split": "validation",
                "is_pilot_workload": (
                    workload_id
                    in PILOT_WORKLOAD_IDS
                ),
            }
        )

    for record in test_records:
        rows.append(
            {
                **record,
                "split": "test",
                "is_pilot_workload": False,
            }
        )

    split = pd.DataFrame(
        rows
    )

    return (
        split
        .sort_values(
            [
                "split",
                "number_of_deliveries",
                "workload_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def validate_split(
    split: pd.DataFrame,
) -> None:
    if len(split) != 108:
        raise ValueError(
            "Split must contain exactly "
            "108 workloads."
        )

    if (
        split["workload_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Workload leakage detected."
        )

    counts = (
        split["split"]
        .value_counts()
        .to_dict()
    )

    expected = {
        "train": TARGET_TRAIN,
        "validation": TARGET_VALIDATION,
        "test": TARGET_TEST,
    }

    if counts != expected:
        raise ValueError(
            "Unexpected split counts: "
            f"{counts}"
        )

    validation_ids = set(
        split.loc[
            split["split"]
            == "validation",
            "workload_id",
        ].astype(int)
    )

    test_ids = set(
        split.loc[
            split["split"]
            == "test",
            "workload_id",
        ].astype(int)
    )

    if not (
        PILOT_WORKLOAD_IDS
        <= validation_ids
    ):
        raise ValueError(
            "Pilot workloads are not all "
            "in validation."
        )

    if (
        PILOT_WORKLOAD_IDS
        & test_ids
    ):
        raise ValueError(
            "Pilot workload leaked into test."
        )

    # Validation và Test phải phủ N=5..13.
    for split_name in (
        "validation",
        "test",
    ):
        sizes = set(
            split.loc[
                split["split"]
                == split_name,
                "number_of_deliveries",
            ].astype(int)
        )

        missing = (
            set(COMMON_SIZES)
            - sizes
        )

        if missing:
            raise ValueError(
                f"{split_name} is missing sizes: "
                f"{sorted(missing)}"
            )

    # Singleton N=14,15,16 chỉ được nằm trong train.
    rare_not_train = split[
        split[
            "number_of_deliveries"
        ].isin(
            RARE_TRAIN_ONLY_SIZES
        )
        & (
            split["split"]
            != "train"
        )
    ]

    if not rare_not_train.empty:
        raise ValueError(
            "Rare N=14..16 workloads must "
            "remain in training."
        )


def build_summary(
    split: pd.DataFrame,
) -> pd.DataFrame:
    return (
        split
        .groupby(
            [
                "split",
                "number_of_deliveries",
            ],
            as_index=False,
        )
        .agg(
            workloads=(
                "workload_id",
                "count",
            )
        )
        .sort_values(
            [
                "split",
                "number_of_deliveries",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_workloads()

    split = build_split(
        manifest
    )

    validate_split(
        split
    )

    summary = build_summary(
        split
    )

    split.to_csv(
        SPLIT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("=" * 76)
    print(
        "PROPOSED RL - "
        "TRAIN / VALIDATION / TEST SPLIT"
    )
    print("=" * 76)

    print()
    print("SPLIT COUNTS")
    print("-" * 76)

    print(
        split[
            "split"
        ]
        .value_counts()
        .reindex(
            [
                "train",
                "validation",
                "test",
            ]
        )
        .to_string()
    )

    print()
    print("SIZE DISTRIBUTION BY SPLIT")
    print("-" * 76)

    table = (
        summary
        .pivot(
            index="number_of_deliveries",
            columns="split",
            values="workloads",
        )
        .fillna(0)
        .astype(int)
        .reindex(
            columns=[
                "train",
                "validation",
                "test",
            ]
        )
    )

    print(
        table.to_string()
    )

    print()
    print("PILOT WORKLOADS")
    print("-" * 76)

    print(
        split[
            split[
                "is_pilot_workload"
            ]
        ][
            [
                "workload_id",
                "number_of_deliveries",
                "split",
            ]
        ].to_string(
            index=False
        )
    )

    findings = [
        "PROPOSED RL DATASET SPLIT",
        "",
        f"seed = {SEED}",
        "",
        f"train = {TARGET_TRAIN}",
        f"validation = {TARGET_VALIDATION}",
        f"test = {TARGET_TEST}",
        "",
        (
            "Validation and test both contain "
            "representatives for N=5 through N=13."
        ),
        "",
        (
            "N=14,15,16 contain one workload each "
            "and are retained in training."
        ),
        "",
        (
            "Pilot workloads 101, 106 and 97 "
            "are fixed in validation."
        ),
        "",
        (
            "The test set remains untouched during "
            "objective and hyperparameter selection."
        ),
    ]

    FINDINGS_PATH.write_text(
        "\n".join(findings),
        encoding="utf-8",
    )

    print()
    print(
        f"Split saved to: {SPLIT_PATH}"
    )

    print(
        f"Summary saved to: {SUMMARY_PATH}"
    )

    print(
        f"Findings saved to: {FINDINGS_PATH}"
    )


if __name__ == "__main__":
    main()