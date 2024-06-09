from typing import Callable

import pandas as pd

from analysis.base.processor import Processor


class ApplyDataFrame(Processor):
    def __init__(self, features: str, function: Callable) -> None:
        super().__init__(features)
        self.function = function

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df[self.features] = df.apply(self.function, axis=1)
        return df
