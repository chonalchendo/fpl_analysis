import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Callable


class ApplyFunction(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature: str,
        apply_func: Callable,
        feature_names_out: str | None = None,
        return_type: str = "numpy",
    ):
        self.feature = feature
        self.apply_func = apply_func
        self.feature_names_out = feature_names_out
        self.return_type = return_type

    def fit(self, X: pd.DataFrame, y=None) -> "ApplyFunction":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        assert self.n_features_in_ == X.shape[1]
        X_ = X.copy()
        X_.loc[:, self.feature] = X.apply(self.apply_func, axis=1)
        if self.return_type == "numpy":
            X_ = X_[self.feature].to_numpy().reshape(-1, 1)
        return X_

    def get_feature_names_out(self, names=None) -> list[str]:
        if self.feature_names_out is None:
            return [self.feature]
        return [self.feature_names_out]
