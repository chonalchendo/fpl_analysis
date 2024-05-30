import pandas as pd

from processing.abcs.processor import Processor


class ColumnFilter(Processor):
    def __init__(self, not_like: str, features=None) -> None:
        super().__init__(features)
        self.not_like = not_like

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        not_like_cols = [col for col in df.columns if self.not_like not in col]
        return df[not_like_cols]
