import pandas as pd

from analysis.base.processor import Processor


class DropColumns(Processor):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        return df.drop(columns=self.features)


class DropNa(Processor):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        return df.dropna(subset=self.features)
