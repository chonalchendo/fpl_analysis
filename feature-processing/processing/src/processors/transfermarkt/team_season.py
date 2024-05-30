import pandas as pd

from processing.abcs.processor import Processor


class TeamSeason(Processor):
    def __init__(self, features=None):
        super().__init__(features)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "team_season"] = df["team"] + " - " + df["season"].astype(str)
        return df
