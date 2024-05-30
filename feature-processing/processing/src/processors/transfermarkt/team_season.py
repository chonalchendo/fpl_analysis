import pandas as pd

from processing.abcs.processor import Processor


class Process(Processor):
    def __init__(self) -> None:
        super().__init__(None)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "team_season"] = df["team"] + " - " + df["season"].astype(str)
        return df
