from pathlib import Path

from analysis import XeDuAnalyzer
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


def test_xedu_analysis():
    dataframe = DataLoader.load_csv(
        XEDU_PATH
    )

    summary = XeDuAnalyzer.summarize(
        dataframe
    )

    XeDuAnalyzer.print_summary(
        summary
    )

    assert summary["total_rows"] > 0

    assert summary["latitude_min"] is not None
    assert summary["latitude_max"] is not None

    assert summary["longitude_min"] is not None
    assert summary["longitude_max"] is not None