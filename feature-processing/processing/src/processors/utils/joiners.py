from functools import reduce
from typing import Literal

import pandas as pd

from processing.abcs.processor import Processor


class MultiJoin(Processor):
    def __init__(
        self,
        on: list[str],
        how: Literal["inner", "outer", "left", "right"],
        suffixes: tuple[str, str],
    ) -> None:
        super().__init__(None)
        self.on = on
        self.suffixes = suffixes
        self.how = how

    def transform(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        return reduce(
            lambda left, right: pd.merge(
                left, right, on=self.on, how=self.how, suffixes=self.suffixes
            ),
            dfs,
        )


class Concat(Processor):
    def __init__(self, axis: int = 0) -> None:
        super().__init__(None)
        self.axis = axis

    def transform(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(dfs, axis=self.axis)
