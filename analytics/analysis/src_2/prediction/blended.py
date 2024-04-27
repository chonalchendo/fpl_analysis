import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y


class BlendedPred(BaseEstimator, RegressorMixin):
    """Class for blending predictions from multiple models with assigned 
    weights based on the model's performance on the training and validation 
    data.
    
    Blending the models together can help to reduce the variance of the 
    predictions and prevent overfitting.

    Args:
        models (list[tuple[float, RegressorMixin]]): List of tuples containing 
        the weight and the model to blend together.
    """
    def __init__(self, models: list[tuple[float, RegressorMixin]]) -> None:
        self.models = models

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BlendedPred":
        X, y = check_X_y(X, y)
        for _, model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> float:
        check_is_fitted(self)
        predictions = 0
        for weight, model in self.models:
            predictions += weight * model.predict(X)
        return predictions
