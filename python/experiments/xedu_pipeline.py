from pathlib import Path

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)


PROJECT_ROOT = Path("/home/deez/Khóa Luận")

XEDU_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "xedu"
    / "xedu_cleaned.csv"
)


def test_xedu_full_pipeline():
    # ==============================
    # 1. Load CSV
    # ==============================

    dataframe = DataLoader.load_csv(
        XEDU_PATH
    )

    assert not dataframe.empty

    original_rows = len(dataframe)

    print(
        "\nSố dòng ban đầu:",
        original_rows,
    )

    # ==============================
    # 2. Preprocessing
    # ==============================

    cleaned = DataPreprocessor.prepare_deliveries(
        dataframe
    )

    cleaned_rows = len(cleaned)

    print(
        "Số dòng sau preprocessing:",
        cleaned_rows,
    )

    assert cleaned_rows > 0
    assert cleaned_rows <= original_rows

    # ==============================
    # 3. Mapping
    # ==============================

    deliveries = DeliveryMapper.from_dataframe(
        cleaned
    )

    print(
        "Số Delivery tạo được:",
        len(deliveries),
    )

    assert len(deliveries) == cleaned_rows

    # ==============================
    # 4. Kiểm tra Delivery đầu tiên
    # ==============================

    first_delivery = deliveries[0]

    print("\nDelivery đầu tiên:")
    print(first_delivery)

    assert first_delivery.delivery_id is not None
    assert -90 <= first_delivery.latitude <= 90
    assert -180 <= first_delivery.longitude <= 180