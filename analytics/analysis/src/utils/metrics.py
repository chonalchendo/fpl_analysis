import numpy as np
from typing import Literal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def model_score(
    y: np.ndarray, y_pred: np.ndarray, scoring: Literal["mae", "rmse", "r2"]
) -> float:
    """Scoring metric for regression model

    Args:
        y (np.ndarray): actual y values
        y_pred (np.ndarray): predicted y values
        scoring (Literal[&quot;mae&quot;, &quot;rmse&quot;, &quot;r2&quot;]):
        scoring metric

    Returns:
        float: score
    """
    if scoring == "mae":
        score = mean_absolute_error(y, y_pred)
    elif scoring == "rmse":
        score = mean_squared_error(y, y_pred, squared=False)
    elif scoring == "r2":
        score = r2_score(y, y_pred)
    return score