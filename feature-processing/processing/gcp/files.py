import gcsfs
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
        self, bucket: str, include: str | None = None, exclude: str | None = None
    ) -> list[str]:
        if include:
            ls = [blob.split("/")[-1] for blob in self.fs.ls(bucket) if include in blob]
        elif exclude:
            ls = [
                blob.split("/")[-1]
                for blob in self.fs.ls(bucket)
                if exclude not in blob
            ]
        else:
            ls = [blob.split("/")[-1] for blob in self.fs.ls(bucket)]
        return ls

    def read_csv(self, path: str) -> pd.DataFrame:
        load = f"gs://{path}"
        return pd.read_csv(load, storage_options=self.storage_options)


gcs = GCS()


if __name__ == "__main__":
    # df = gcs.read_csv("fbref_db/Premier-League-wages.csv")
    # print(df)
    # print(df.columns)
    wages = "Premier-League-wages.csv"

    sp = wages.split("-")[0]
    print(sp)
