import gcsfs
import joblib
import pandas as pd
from rich import print

from app.core.config import get_settings


class GCS:
    def __init__(self):
        self.fs = gcsfs.GCSFileSystem(
            project=get_settings().GCP_PROJECT,
            token=get_settings().GCP_SERVICE_ACCOUNT_JSON_PATH,
        )
        self.storage_options = {"token": get_settings().GCP_SERVICE_ACCOUNT_JSON_PATH}

    def read_csv(self, path: str) -> pd.DataFrame:
        load = f"gcs://{path}"
        return pd.read_csv(load, storage_options=self.storage_options)

    def read_pickle(self, bucket: str, path: str) -> pd.DataFrame:
        load = f"gcs://{bucket}/{path}"
        return pd.read_pickle(load, storage_options=self.storage_options)

    def read_parquet(self, path: str) -> pd.DataFrame:
        load = f"gcs://{path}"
        return pd.read_parquet(load, filesystem=self.fs)

    def load_model(self, bucket: str, path: str):
        with self.fs.open(f"{bucket}/{path}", "rb") as f:
            return joblib.load(f)

    def ls(self, bucket: str) -> list:
        return self.fs.ls(bucket)


gcs = GCS()
