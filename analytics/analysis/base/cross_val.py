from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold


class CrossValidator(ABC):
    def __init__(self, metric: str, method: KFold) -> None:
        self.metric = metric
        self.method = method

    @abstractmethod
    def validate(
        self,
        models: list[RegressorMixin],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        pass

    @property
    @abstractmethod
    def cv_results_(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def train_scores_(self) -> list[dict]:
        pass

    @property
    @abstractmethod
    def test_scores_(self) -> list[dict]:
        pass
