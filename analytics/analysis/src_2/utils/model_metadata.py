import datetime as dt
import pandas as pd
from typing import Any

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline


def model_metadata(
    model: RegressorMixin,
    model_params: dict,
    preprocess_steps: Pipeline,
    target_steps: Pipeline,
    metric: str,
    scores: list[float],
    mean_score: float,
    std_score: float,
    X_data: pd.DataFrame,
    y_data: pd.DataFrame,
    date: dt.datetime = dt.datetime.now(),
) -> dict[str, Any]:
    """Create a dictionary containing the model metadata.

    Args:
        model (RegressorMixin): scikit-learn model
        model_params (dict): model hyperparameters
        preprocess_steps (Pipeline): preprocessing steps
        target_steps (Pipeline): target transformation steps
        metric (str): evaluation metric
        scores (list[float]): list of scores
        mean_scores (float): mean score
        std_scores (float): standard deviation of the scores
        X_data (pd.DataFrame): X data
        y_data (pd.DataFrame): y data
        date (dt.datetime): date model created. Defaults to dt.datetime.now().strftime("%Y-%m-%d").

    Returns:
        dict[str, Any]: model metadata
    """
    return {
        "model": model,
        "model_params": model_params,
        "preprocess": preprocess_steps,
        "target": target_steps,
        "metric": metric,
        "scores": scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "X_data": X_data,
        "y_data": y_data,
        "date": date,
    }
