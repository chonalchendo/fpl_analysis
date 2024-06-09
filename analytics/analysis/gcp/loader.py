import pandas as pd

from analysis.base.data_loader import DataLoader
from analysis.gcp.files import gcs


class CSVLoader(DataLoader):
    def load(self, input_path: str) -> pd.DataFrame:
        return gcs.read_csv(input_path)
