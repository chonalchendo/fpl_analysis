import pandas as pd

from processing.abcs.processor import Processor


class Process(Processor):
    def __init__(self) -> None:
        super().__init__(None)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Redefine season column in wages dataframe to match valuations dataframe"""
        df.loc[:, "season"] = df["season"].apply(lambda x: x.split("-")[0]).astype(int)
        return df
