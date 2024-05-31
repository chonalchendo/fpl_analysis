import pandas as pd

from processing.abcs.processor import Processor


class Process(Processor):
    """Redefine season column in wages dataframe to match valuations dataframe"""

    def __init__(self) -> None:
        super().__init__(None)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "season"] = df["season"].apply(lambda x: x.split("-")[0]).astype(int)
        return df
