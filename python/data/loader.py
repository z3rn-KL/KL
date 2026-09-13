from pathlib import Path

import pandas as pd


class DataLoader:
    @staticmethod
    def load_csv(file_path: str | Path) -> pd.DataFrame:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {path}"
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"File không phải CSV: {path}"
            )

        dataframe = pd.read_csv(
            path,
            low_memory=False,
        )

        if dataframe.empty:
            raise ValueError(
                f"Dataset rỗng: {path}"
            )

        return dataframe