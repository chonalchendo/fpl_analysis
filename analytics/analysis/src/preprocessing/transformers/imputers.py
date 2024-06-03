import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BoolImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature: str,
        value: int,
        condition_col: str,
        condition: bool,
        return_type: str = "numpy",
    ) -> None:
        self.feature = feature
        self.value = value
        self.condition_col = condition_col
        self.condition = condition
        self.return_type = return_type

    def fit(self, X: pd.DataFrame, y=None) -> "BoolImputer":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        assert self.n_features_in_ == X.shape[1]

        if X.columns.duplicated().any():
            X = X.loc[:, ~X.columns.duplicated()]

        X.loc[X[self.condition_col] == self.condition, self.feature] = self.value
        if self.return_type == "numpy":
            X_ = X[self.feature].to_numpy().reshape(-1, 1)
        if self.return_type == "series":
            X_ = X[self.feature]
        if self.return_type == "pandas":
            X_ = X
        return X_

    def get_feature_names_out(self, names=None) -> str:
        return self.feature


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        features: list[str] | str | None = None,
        value: str | None = None,
        strategy: str | None = None,
        groupby: list[str] | None = None,
        impute_by_col: str | None = None,
        return_type: str = "numpy",
    ) -> None:
        self.features = features
        self.groupby = groupby
        self.strategy = strategy
        self.value = value
        self.impute_by_col = impute_by_col
        self.transformed_cols = []
        self.return_type = return_type

    def _groupby_impute(self, X: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        X_ = X.copy()
        for col in X_.columns:
            if X_[col].isnull().sum() > 0 and np.issubdtype(X_[col].dtype, np.number):
                self.transformed_cols.append(col)
                X_.loc[:, col] = X_[col].fillna(
                    X_.groupby(self.groupby)[col].transform(self.strategy)
                )
        if len(self.transformed_cols) > 1 and self.return_type == "numpy":
            values = X_[self.transformed_cols].to_numpy()
        elif len(self.transformed_cols) > 1 and self.return_type == "pandas":
            values = X_
        else:
            values = X_[self.transformed_cols].to_numpy().reshape(-1, 1)

        return values

    def _fillna_impute(self, X: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        X_ = X.copy()
        if not self.impute_by_col:
            X_.loc[:, self.features] = X_[self.features].fillna(self.value)
        else:
            X_.loc[:, self.features] = X_[self.features].fillna(X_[self.impute_by_col])

        self.transformed_cols.append(self.features)

        if self.return_type == "numpy":
            return X_[self.features].to_numpy().reshape(-1, 1)
        if self.return_type == "pandas":
            return X_

    def fit(self, X: pd.DataFrame, y=None) -> "CustomImputer":
        if self.strategy:
            self.impute_ = self._groupby_impute(X)
        else:
            self.impute_ = self._fillna_impute(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        assert self.n_features_in_ == X.shape[1]
        return self.impute_

    def get_feature_names_out(self, names=None) -> list[str]:
        return self.transformed_cols


class GroupbyImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature: str,
        groupby: list[str],
        strategy: str,
        return_type: str = "numpy",
    ) -> None:
        self.feature = feature
        self.groupby = groupby
        self.strategy = strategy
        self.return_type = return_type

    def fit(self, X: pd.DataFrame, y=None) -> "GroupbyImputer":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        check_is_fitted(self)
        assert self.n_features_in_ == X.shape[1]
        X_ = X.copy()
        X_.loc[:, self.feature] = X_[self.feature].fillna(
            X_.groupby(self.groupby)[self.feature].transform(self.strategy)
        )

        if self.return_type == "pandas":
            return X_
        if self.return_type == "numpy":
            return X_[self.feature].to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, names=None) -> list[str]:
        return [self.feature]
