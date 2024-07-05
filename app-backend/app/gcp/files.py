import gcsfs
import joblib
import pandas as pd

from app.core.config import get_settings


class GCS:
    def __init__(self, local: bool = False):
        if local:
            project = get_settings().LOCAL_GCP_PROJECT
            service_account_json_path = (
                get_settings().LOCAL_GCP_SERVICE_ACCOUNT_JSON_PATH
            )
        else:
            project = get_settings().GCP_PROJECT
            service_account_json_path = get_settings().GCP_SERVICE_ACCOUNT_JSON_PATH

        self.fs = gcsfs.GCSFileSystem(
            project=project,
            token=service_account_json_path,
        )
        self.storage_options = {"token": service_account_json_path}

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
