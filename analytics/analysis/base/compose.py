from abc import ABC, abstractmethod

import pandas as pd

from analysis.base.processor import Processor


class BaseComposer(ABC):
    def __init__(self, processors: list[Processor]) -> None:
        self.processors = processors

    @abstractmethod
    def compose(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
