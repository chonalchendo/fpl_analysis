import pandas as pd

from processing.abcs.loader import DataLoader
from processing.gcp.files import gcs
from processing.gcp.storage import gcp


class GCPLoader(DataLoader):
    def load(self, bucket: str, blob: str) -> pd.DataFrame:
        return gcp.read_df_from_bucket(bucket_name=bucket, blob_name=blob)


class CSVLoader(DataLoader):
    def load(self, bucket: str, blob: str) -> pd.DataFrame:
        path = f"{bucket}/{blob}"
        return gcs.read_csv(path)
