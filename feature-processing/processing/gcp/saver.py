from typing import Literal

import pandas as pd

from processing.abcs.saver import DataSaver
from processing.gcp.storage import gcp


class GCPSaver(DataSaver):
    def save(self, bucket: str, blob: str, data: pd.DataFrame) -> pd.DataFrame:
        gcp.write_df_to_bucket(bucket_name=bucket, blob_name=blob, data=data)


def save(save: Literal["yes", "no"]) -> GCPSaver | None:
    if save == "yes":
        saver = GCPSaver()
    else:
        saver = None
    return saver
