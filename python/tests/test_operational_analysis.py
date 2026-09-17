from pathlib import Path

from analysis import OperationalAnalyzer
from data import DataLoader


PROJECT_ROOT = Path(
    "/home/deez/Khóa Luận"
)

XEDU_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)


def test_operational_analysis():
    dataframe = DataLoader.load_csv(
        XEDU_PATH
    )

    shipper_workload = (
        OperationalAnalyzer.shipper_workload(
            dataframe
        )
    )

    sender_workload = (
        OperationalAnalyzer.sender_workload(
            dataframe
        )
    )

    relationship = (
        OperationalAnalyzer
        .shipper_sender_relationship(
            dataframe
        )
    )

    print(
        "\n===== SHIPPER WORKLOAD ====="
    )

    print(
        shipper_workload.to_string(
            index=False
        )
    )

    print(
        "\n===== TOP 10 SENDER LOCATIONS ====="
    )

    print(
        sender_workload.head(10).to_string(
            index=False
        )
    )

    print(
        "\n===== TOP 20 SHIPPER-SENDER ====="
    )

    print(
        relationship.head(20).to_string(
            index=False
        )
    )

    assert len(shipper_workload) > 0
    assert len(sender_workload) > 0