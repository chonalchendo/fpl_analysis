import pandas as pd

from processing.abcs.processor import Processor


class Process(Processor):
    """Change team names in valuations dataframe to match wages dataframe"""

    def __init__(self, team_map: dict[str, str]) -> None:
        super().__init__(None)
        self.team_map = team_map

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "team"] = df["team"].map(self.team_map)
        return df
