import pandas as pd

from data import (
    DataLoader,
    DataPreprocessor,
    DeliveryMapper,
)


def test_preprocessing():
    dataframe = pd.DataFrame(
        {
            "id": [
                "D001",
                "D002",
                "D003",
            ],
            "receiverLat": [
                10.77,
                10.78,
                None,
            ],
            "receiverLng": [
                106.70,
                106.71,
                106.72,
            ],
            "weight": [
                5.0,
                "10",
                None,
            ],
        }
    )

    preprocessor = DataPreprocessor()

    cleaned = preprocessor.prepare_deliveries(
        dataframe
    )

    assert len(cleaned) == 2


def test_mapper():
    dataframe = pd.DataFrame(
        {
            "id": [
                "D001",
            ],
            "receiverLat": [
                10.77,
            ],
            "receiverLng": [
                106.70,
            ],
            "weight": [
                5.0,
            ],
            "serviceType": [
                "STANDARD",
            ],
        }
    )

    cleaned = DataPreprocessor.prepare_deliveries(
        dataframe
    )

    deliveries = DeliveryMapper.from_dataframe(
        cleaned
    )

    assert len(deliveries) == 1

    delivery = deliveries[0]

    assert delivery.delivery_id == "D001"
    assert delivery.latitude == 10.77
    assert delivery.longitude == 106.70
    assert delivery.weight == 5.0


def test_invalid_coordinate_removed():
    dataframe = pd.DataFrame(
        {
            "id": [
                "D001",
            ],
            "receiverLat": [
                999,
            ],
            "receiverLng": [
                106.70,
            ],
        }
    )

    cleaned = DataPreprocessor.prepare_deliveries(
        dataframe
    )

    assert len(cleaned) == 0