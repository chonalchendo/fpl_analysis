from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline


class Tester(ABC):
    @abstractmethod
    def test(
        self,
        models: list[RegressorMixin | Pipeline],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        pass

    @abstractmethod
    def blend(self, y_test: pd.Series) -> None:
        pass

    @property
    @abstractmethod
    def pred_results_(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def performance_(self) -> pd.DataFrame:
        pass
