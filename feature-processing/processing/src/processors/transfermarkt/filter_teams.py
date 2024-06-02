import pandas as pd

from processing.abcs.processor import Processor


class Process(Processor):
    def __init__(self) -> None:
        super().__init__(None)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out Hannover 96 from Bundesliga valuations as it does not appear in
        wages data.
        """
        league = df["league"].values[0]

        if league == "bundesliga":
            df = df.loc[df["squad"] != "hannover-96"]

        return df
