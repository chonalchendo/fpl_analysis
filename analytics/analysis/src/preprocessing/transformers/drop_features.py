import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.feature_names_in_ = features

    def fit(self, X: pd.DataFrame, y=None) -> "DropFeatures":
        X_ = X.copy()
        self.n_features_in_ = X_.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_ = X.copy()
        check_is_fitted(self)
        # assert self.n_features_in_ == X_.shape[1]
        X_ = X_.drop(columns=self.features)
        return X_
