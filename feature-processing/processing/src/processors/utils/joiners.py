from functools import reduce

import pandas as pd

from processing.abcs.processor import Processor


class MultiJoin(Processor):
    def __init__(self, on: list[str], suffixes: tuple[str, str], features=None) -> None:
        super().__init__(features)
        self.on = on
        self.suffixes = suffixes

    def process(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        return reduce(
            lambda left, right: pd.merge(
                left, right, on=self.on, suffixes=self.suffixes
            ),
            dfs,
        )
