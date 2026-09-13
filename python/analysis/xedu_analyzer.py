import pandas as pd


class XeDuAnalyzer:
    @staticmethod
    def summarize(dataframe: pd.DataFrame) -> dict:
        summary = {}

        summary["total_rows"] = len(dataframe)

        # ==============================
        # SHIPPER
        # ==============================

        if "shipper" in dataframe.columns:
            summary["unique_shippers"] = (
                dataframe["shipper"]
                .dropna()
                .nunique()
            )
        else:
            summary["unique_shippers"] = 0

        # ==============================
        # WEIGHT
        # ==============================

        if "weight" in dataframe.columns:
            weight = pd.to_numeric(
                dataframe["weight"],
                errors="coerce",
            )

            summary["missing_weight"] = int(
                weight.isna().sum()
            )

            summary["weight_min"] = XeDuAnalyzer._safe_float(
                weight.min()
            )

            summary["weight_max"] = XeDuAnalyzer._safe_float(
                weight.max()
            )

            summary["weight_mean"] = XeDuAnalyzer._safe_float(
                weight.mean()
            )

        # ==============================
        # DEADLINE
        # ==============================

        if "expectedDeliveryTime" in dataframe.columns:
            deadlines = pd.to_datetime(
                dataframe["expectedDeliveryTime"],
                errors="coerce",
            )

            summary["missing_deadline"] = int(
                deadlines.isna().sum()
            )

        # ==============================
        # RECEIVER COORDINATES
        # ==============================

        if (
            "receiverLat" in dataframe.columns
            and "receiverLng" in dataframe.columns
        ):
            lat = pd.to_numeric(
                dataframe["receiverLat"],
                errors="coerce",
            )

            lng = pd.to_numeric(
                dataframe["receiverLng"],
                errors="coerce",
            )

            summary["latitude_min"] = XeDuAnalyzer._safe_float(
                lat.min()
            )

            summary["latitude_max"] = XeDuAnalyzer._safe_float(
                lat.max()
            )

            summary["longitude_min"] = XeDuAnalyzer._safe_float(
                lng.min()
            )

            summary["longitude_max"] = XeDuAnalyzer._safe_float(
                lng.max()
            )

        # ==============================
        # SERVICE TYPE
        # ==============================

        if "serviceType" in dataframe.columns:
            service_counts = (
                dataframe["serviceType"]
                .fillna("UNKNOWN")
                .astype(str)
                .value_counts()
                .to_dict()
            )

            summary["service_types"] = service_counts

        # ==============================
        # SENDER LOCATIONS
        # ==============================

        if (
            "senderLat" in dataframe.columns
            and "senderLng" in dataframe.columns
        ):
            sender_locations = (
                dataframe[
                    [
                        "senderLat",
                        "senderLng",
                    ]
                ]
                .dropna()
                .drop_duplicates()
            )

            summary["unique_sender_locations"] = len(
                sender_locations
            )

        return summary

    @staticmethod
    def print_summary(
        summary: dict,
    ) -> None:
        print("\n===== XEDU DATA ANALYSIS =====")

        print(
            "Tổng số dòng:",
            summary.get("total_rows"),
        )

        print(
            "Số shipper khác nhau:",
            summary.get("unique_shippers"),
        )

        print(
            "Số weight bị thiếu:",
            summary.get("missing_weight"),
        )

        print(
            "Weight nhỏ nhất:",
            summary.get("weight_min"),
        )

        print(
            "Weight lớn nhất:",
            summary.get("weight_max"),
        )

        print(
            "Weight trung bình:",
            summary.get("weight_mean"),
        )

        print(
            "Số deadline bị thiếu:",
            summary.get("missing_deadline"),
        )

        print(
            "Latitude range:",
            summary.get("latitude_min"),
            "->",
            summary.get("latitude_max"),
        )

        print(
            "Longitude range:",
            summary.get("longitude_min"),
            "->",
            summary.get("longitude_max"),
        )

        print(
            "Service types:",
            summary.get("service_types"),
        )

        print(
            "Số sender location khác nhau:",
            summary.get("unique_sender_locations"),
        )

    @staticmethod
    def _safe_float(
        value,
    ) -> float | None:
        if pd.isna(value):
            return None

        return float(value)