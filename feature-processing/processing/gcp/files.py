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
    bucket = "processed_fbref_db"
    # files = gcs.list_bucket(bucket, include="wage")
    #
    df = gcs.read_csv("processed_fbref_db/processed_La-Liga-wages.csv")
    df2 = gcs.read_csv("processed_fbref_db/processed_shooting.csv")
    df3 = gcs.read_csv(
        "processed_transfermarkt_db/processed_la_liga_player_valuations.csv"
    )
    df4 = gcs.read_csv("joined_wages_values/la_liga_wages_values.csv")
    df5 = gcs.read_csv("transfermarkt_db/la_liga_player_valuations.csv")

    cols = ["player", "season", "squad"]
    print(df[cols].head())
    print(df3[cols].head())
    print(df4)

    joined = pd.merge(df3, df, on=cols, how="inner", suffixes=("", "_wages"))
    print(joined["squad"].unique())

    print(df3.loc[df3["player"].str.contains("Navas")])
