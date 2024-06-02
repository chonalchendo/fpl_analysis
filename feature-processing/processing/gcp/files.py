import gcsfs
import numpy as np
import pandas as pd

from processing.core.settings import SETTINGS


class GCS:
    def __init__(self) -> None:
        self.fs = gcsfs.GCSFileSystem(
            project=SETTINGS["GOOGLE_CLOUD_PROJECT"],
            token=SETTINGS["GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH"],
        )
        self.storage_options = {
            "token": SETTINGS["GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH"]
        }

    def list_bucket(
        self,
        bucket: str,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[str]:
        if include:
            ls = [
                blob.split("/")[-1]
                for blob in self.fs.ls(bucket)
                if blob.split("/")[-1] in include
            ]
        elif exclude:
            ls = [
                blob.split("/")[-1]
                for blob in self.fs.ls(bucket)
                if blob.split("/")[-1] not in exclude
            ]
        else:
            ls = [blob.split("/")[-1] for blob in self.fs.ls(bucket)]
        return ls

    def read_csv(self, path: str) -> pd.DataFrame:
        load = f"gs://{path}"
        return pd.read_csv(load, storage_options=self.storage_options)


gcs = GCS()

if __name__ == "__main__":
    df = gcs.read_csv("processed_fbref_db/processed_shooting.csv")
    print(df)
    print(df.columns)
