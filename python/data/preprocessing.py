import pandas as pd


class DataPreprocessor:
    REQUIRED_COLUMNS = [
        "id",
        "receiverLat",
        "receiverLng",
    ]

    DATETIME_COLUMNS = [
        "createdAt",
        "expectedDeliveryTime",
        "deliveredAt",
    ]

    @classmethod
    def prepare_deliveries(
        cls,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        df = dataframe.copy()

        cls._validate_required_columns(df)

        # ------------------------------
        # Chuẩn hóa tọa độ
        # ------------------------------

        df["receiverLat"] = pd.to_numeric(
            df["receiverLat"],
            errors="coerce",
        )

        df["receiverLng"] = pd.to_numeric(
            df["receiverLng"],
            errors="coerce",
        )

        # Loại bỏ tọa độ bị thiếu
        df = df.dropna(
            subset=[
                "receiverLat",
                "receiverLng",
            ]
        )

        # Latitude hợp lệ: [-90, 90]
        # Longitude hợp lệ: [-180, 180]
        df = df[
            df["receiverLat"].between(
                -90,
                90,
            )
            & df["receiverLng"].between(
                -180,
                180,
            )
        ]

        # ------------------------------
        # Weight
        # ------------------------------

        if "weight" in df.columns:
            df["weight"] = pd.to_numeric(
                df["weight"],
                errors="coerce",
            )

        # ------------------------------
        # Date time
        # ------------------------------

        for column in cls.DATETIME_COLUMNS:
            if column in df.columns:
                df[column] = pd.to_datetime(
                    df[column],
                    errors="coerce",
                )

        # ------------------------------
        # Delivery ID
        # ------------------------------

        df["id"] = df["id"].astype(str)

        # Loại duplicate ID nếu có
        df = df.drop_duplicates(
            subset=["id"],
            keep="first",
        )

        # Reset index sau khi clean
        df = df.reset_index(
            drop=True
        )

        return df

    @classmethod
    def _validate_required_columns(
        cls,
        dataframe: pd.DataFrame,
    ) -> None:

        missing_columns = [
            column
            for column in cls.REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Dataset thiếu các cột bắt buộc: "
                f"{missing_columns}"
            )