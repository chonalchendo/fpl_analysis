from typing import Literal
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import RandomizedSearchCV

from analysis.src_2.utils.metrics import model_score


def random_search_tuning(
    model,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    cv: int,
    scoring: Literal["mae", "rmse", "r2"],
    n_iter: int = 10,
) -> tuple:
    random_search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    random_search.fit(X, y)

    return random_search.best_estimator_, random_search.cv_results_


def validate_prediction(
    model: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    scoring: Literal["mae", "rmse", "r2"],
) -> dict:
    y_pred = model.predict(X)
    return model_score(y, y_pred, scoring)
