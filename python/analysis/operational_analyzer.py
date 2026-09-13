import pandas as pd


class OperationalAnalyzer:

    @staticmethod
    def shipper_workload(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if "shipper" not in dataframe.columns:
            raise ValueError(
                "Dataset không có cột shipper."
            )

        result = (
            dataframe
            .dropna(subset=["shipper"])
            .groupby("shipper")
            .size()
            .reset_index(name="order_count")
            .sort_values(
                "order_count",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        return result

    @staticmethod
    def sender_workload(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        required = [
            "senderLat",
            "senderLng",
        ]

        for column in required:
            if column not in dataframe.columns:
                raise ValueError(
                    f"Dataset thiếu cột {column}"
                )

        result = (
            dataframe
            .dropna(
                subset=[
                    "senderLat",
                    "senderLng",
                ]
            )
            .groupby(
                [
                    "senderLat",
                    "senderLng",
                ]
            )
            .size()
            .reset_index(
                name="order_count"
            )
            .sort_values(
                "order_count",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        return result

    @staticmethod
    def shipper_sender_relationship(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        required = [
            "shipper",
            "senderLat",
            "senderLng",
        ]

        for column in required:
            if column not in dataframe.columns:
                raise ValueError(
                    f"Dataset thiếu cột {column}"
                )

        result = (
            dataframe
            .dropna(
                subset=required
            )
            .groupby(
                [
                    "shipper",
                    "senderLat",
                    "senderLng",
                ]
            )
            .size()
            .reset_index(
                name="order_count"
            )
            .sort_values(
                "order_count",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        return result