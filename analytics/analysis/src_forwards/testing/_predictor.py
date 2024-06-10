import numpy as np
import pandas as pd
from rich import print
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from analysis.base.test import Tester
from analysis.src_forwards.statistics import calculate_weights, mae_confidence_interval
from analysis.utilities.logging import get_logger

logger = get_logger(__name__)


class Predict(Tester):
    def __init__(self) -> None:
        self.predictions = pd.DataFrame()
        self.performance = pd.DataFrame(columns=["mae", "rmse", "r2", "mae_ci"])

    def test(
        self,
        models: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        for model in models:
            model_name = model["model"].__class__.__name__
            logger.info(f"Testing {model_name}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            logger.info(f"{model_name} predictions: {y_pred}")
            self.predictions[model_name] = y_pred

            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            ci = mae_confidence_interval(y_test, y_pred)

            self.performance.loc[model_name] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mae_ci": ci,
            }

    def blend(self, y_test: pd.Series) -> None:
        mae_scores = self.performance["mae"].to_list()
        weights = calculate_weights(mae_scores)
        preds = self.predictions.to_numpy()
        blend_pred = np.sum(preds * weights, axis=1).reshape(-1, 1)

        self.predictions["blend"] = blend_pred

        mae = mean_absolute_error(y_test, blend_pred)
        rmse = mean_squared_error(y_test, blend_pred, squared=False)
        r2 = r2_score(y_test, blend_pred)
        ci = mae_confidence_interval(y_test, blend_pred)

        self.performance.loc["blend"] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mae_ci": ci,
        }

    @property
    def pred_results_(self) -> pd.DataFrame:
        return self.predictions

    @property
    def performance_(self) -> pd.DataFrame:
        return self.performance
