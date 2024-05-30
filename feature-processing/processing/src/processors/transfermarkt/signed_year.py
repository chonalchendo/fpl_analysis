import pandas as pd

from processing.abcs.processor import Processor


class SignedYear(Processor):
    """Redefine season column in wages dataframe to match valuations dataframe"""

    def __init__(self, features=None):
        super().__init__(features)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "signed_year"] = (
            df["signed_date"].str.split(" ").str[2].astype("Int32")
        )
        return df
