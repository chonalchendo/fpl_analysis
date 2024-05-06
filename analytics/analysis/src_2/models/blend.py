from typing import Callable
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y
from sklearn.base import clone


class BlendedRegressor:
    """Class for blending predictions from multiple models with assigned
    weights based on the model's performance on the training and validation
    data.

    Blending the models together can help to reduce the variance of the
    predictions and prevent overfitting.

    Args:
        models (list[tuple[float, RegressorMixin]]): List models to blend
        together.
        weights (list[float]): List of weights to assign to each model.
        inverse_func (Callable | None): Function transformer to apply to the
        predictions. Defaults to None.
    """

    def __init__(
        self,
        models: list[tuple[str, RegressorMixin]],
        weights: list[float],
        inverse_func: Callable | None = None,
    ) -> None:
        self.models = models
        self.weights = weights
        self.inverse_func = inverse_func
        self.fitted_models = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BlendedRegressor":
        X, y = check_X_y(X, y)
        for model in self.models:
            clone_model = clone(model)
            clone_model.fit(X, y)
            self.fitted_models.append(clone_model)
        return self

    def predict(self, X: pd.DataFrame) -> float:
        predictions = sum(
            weight * model.predict(X)
            for weight, model in zip(self.weights, self.fitted_models)
        )
        if self.inverse_func:
            predictions = self.inverse_func(predictions)

        return predictions
