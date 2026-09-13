import pandas as pd

from domain import Delivery


class DeliveryMapper:
    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
    ) -> list[Delivery]:

        deliveries: list[Delivery] = []

        for _, row in dataframe.iterrows():
            delivery = Delivery(
                delivery_id=str(
                    row["id"]
                ),
                latitude=float(
                    row["receiverLat"]
                ),
                longitude=float(
                    row["receiverLng"]
                ),
                weight=DeliveryMapper._optional_float(
                    row,
                    "weight",
                ),
                created_at=DeliveryMapper._optional_datetime(
                    row,
                    "createdAt",
                ),
                expected_delivery_time=DeliveryMapper._optional_datetime(
                    row,
                    "expectedDeliveryTime",
                ),
                delivered_at=DeliveryMapper._optional_datetime(
                    row,
                    "deliveredAt",
                ),
                service_type=DeliveryMapper._optional_string(
                    row,
                    "serviceType",
                ),
            )

            deliveries.append(delivery)

        return deliveries

    @staticmethod
    def _optional_float(
        row: pd.Series,
        column: str,
    ) -> float | None:
        if column not in row.index:
            return None

        value = row[column]

        if pd.isna(value):
            return None

        return float(value)

    @staticmethod
    def _optional_datetime(
        row: pd.Series,
        column: str,
    ):
        if column not in row.index:
            return None

        value = row[column]

        if pd.isna(value):
            return None

        return value.to_pydatetime()

    @staticmethod
    def _optional_string(
        row: pd.Series,
        column: str,
    ) -> str | None:
        if column not in row.index:
            return None

        value = row[column]

        if pd.isna(value):
            return None

        return str(value)