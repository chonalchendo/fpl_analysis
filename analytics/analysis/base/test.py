from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline


class Tester(ABC):
    @abstractmethod
    def test(
        self, models: list[RegressorMixin | Pipeline], X: pd.DataFrame, y: pd.DataFrame
    ) -> None:
        pass

    @abstractmethod
    def blend(self) -> None:
        pass
