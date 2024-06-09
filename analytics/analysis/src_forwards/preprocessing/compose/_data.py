from functools import reduce

import pandas as pd

from analysis.base.compose import BaseComposer
from analysis.base.processor import Processor


class Composer(BaseComposer):
    def __init__(self, processors: list[Processor]) -> None:
        self.processors = processors

    def compose(self, data: pd.DataFrame) -> pd.DataFrame:
        return reduce(
            lambda df, processor: processor.process(df), self.processors, data
        )
