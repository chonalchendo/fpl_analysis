import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y


class BlendedRegressor(BaseEstimator, RegressorMixin):
    """Class for blending predictions from multiple models with assigned
    weights based on the model's performance on the training and validation
    data.

    Blending the models together can help to reduce the variance of the
    predictions and prevent overfitting.

    Args:
        models (list[tuple[float, RegressorMixin]]): List models to blend together.
        weights (list[float]): List of weights to assign to each model.
    """

    def __init__(self, models: list[RegressorMixin], weights: list[float]) -> None:
        self.models = models
        self.weights = weights

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BlendedRegressor":
        X, y = check_X_y(X, y)
        for _, model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> float:
        check_is_fitted(self)

        predictions = 0
        for weight, model in zip(self.weights, self.models):
            predictions += weight * model.predict(X)
        return predictions
    
    def blend_pred(self, X: pd.DataFrame, y: pd.Series) -> float:
        check_is_fitted(self)
        
        # transform X
        # transform y
        # fit X and y to models
        # predict X with weights
        # return prediction
        
        
