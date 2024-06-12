import pandas as pd

from analysis.base import DataSaver
from analysis.gcp.files import gcs


class CSVSaver(DataSaver):
    def save(self, df: pd.DataFrame, output_path: str) -> None:
        gcs.write_csv(df, output_path)


class ParquetSaver(DataSaver):
    def save(self, df: pd.DataFrame, output_path: str) -> None:
        gcs.write_parquet(df, output_path)
