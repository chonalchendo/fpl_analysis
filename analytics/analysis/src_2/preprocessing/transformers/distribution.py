import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_type: str) -> None:
        self.log_type = log_type

    def fit(self, X: np.ndarray, y=None) -> "LogTransformer":
        check_array(X)
        # self.n_features_in_ = X.shape[1]
        # self.features_names_in_ = X.columns
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        check_array(X)
        # assert self.n_features_in_ == X.shape[1]

        if self.log_type == "log":
            X_log = np.log(X)
        elif self.log_type == "log10":
            X_log = np.log10(X)
        elif self.log_type == "log1p":
            X_log = np.log1p(X)
        return X_log.reshape(-1, 1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        assert self.n_features_in_ == X.shape[1]

        if self.log_type == "log":
            X_inv = np.exp(X)
        elif self.log_type == "log10":
            X_inv = 10**X
        elif self.log_type == "log1p":
            X_inv = np.expm1(X)
        return X_inv

    # def get_feature_names_out(self, names=None) -> str:
    #     return self.features_names_in_
