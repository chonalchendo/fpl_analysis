import pandas as pd

from processing.abcs.processor import Processor


class Process(Processor):
    """Redefine season column in wages dataframe to match valuations dataframe"""

    def __init__(self) -> None:
        super().__init__(None)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "signed_year"] = (
            df["signed_date"].str.split(" ").str[2].astype("Int32")
        )
        return df
