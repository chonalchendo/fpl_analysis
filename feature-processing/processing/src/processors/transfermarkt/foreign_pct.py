import pandas as pd

from processing.abcs.processor import Processor


class Process(Processor):
    def __init__(self) -> None:
        super().__init__(None)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "foreigner_pct"] = round(
            (df["squad_foreigners"] / df["squad_size"]) * 100, 2
        )
        return df
