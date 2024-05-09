from typing import Literal
import xdrlib
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from analysis.src_2.prediction.weights import calculate_weights
from analysis.utilities.logging import get_logger
from analysis.src_2.utils.metrics import model_score
from analysis.src_2.preprocessing.pipeline.build import PipelineBuilder
from analysis.gcp.storage import gcp
from analysis.src_2.utils.model_metadata import model_metadata


logger = get_logger(__name__)


class ModelValidator:
    def __init__(
        self,
        model: list[RegressorMixin],
        search_method: RandomizedSearchCV | GridSearchCV,
        params: list[dict],
    ):
        self.model = model
        self.search_method = search_method
        self.params = params
        self._tuned_models = []
        self._train_scores = []
        self._valid_scores = []

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        cv: int,
        scoring: Literal[
            "neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"
        ],
        n_iter: int = 10,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
        store: bool = False,
    ) -> list[str, dict]:
        return [
            {
                list(pipeline.named_steps.keys())[2]: self._validate(
                    pipeline=pipeline,
                    param=param[1],
                    X=X,
                    y=y,
                    cv=cv,
                    scoring=scoring,
                    n_iter=n_iter,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    store=store
                )
            }
            for pipeline, param in self._pipeline(X)
        ]

    def _pipeline(self, X: pd.DataFrame) -> list[tuple]:
        pipe = PipelineBuilder()
        return [
            (pipe.build(X=X, model=model), param)
            for model, param in zip(self.model, self.params)
        ]

    def _preprocess(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> tuple[RegressorMixin, np.ndarray, np.ndarray]:
        clone_model = clone(pipeline[2])
        y_ = pipeline["target"].fit_transform(y).ravel()
        X_ = pipeline["preprocess"].fit_transform(X, y_)
        return clone_model, X_, y_

    def _random_tune(
        self,
        model,
        params: dict,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        cv: int,
        scoring: str,
        n_iter: int = 10,
    ) -> tuple:
        random_search = RandomizedSearchCV(
            model,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42, 
        )
        random_search.fit(X, y)
        return random_search.best_estimator_, random_search.cv_results_

    def _validate(
        self,
        pipeline: Pipeline,
        param: dict,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        cv: int,
        scoring: str,
        n_iter: int = 10,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
        store: bool = False,
    ) -> dict:

        if not isinstance(y, np.ndarray):
            y = y.to_numpy().reshape(-1, 1)

        model, X_, y_ = self._preprocess(pipeline, X, y)
        
        model_name = list(pipeline.named_steps.keys())[2]
        logger.info(f"tuning {model_name}")
        
        best_model, _ = self._random_tune(model, param, X_, y_, cv, scoring, n_iter)
        self.tuned_models.append(best_model)
        y_pred = best_model.predict(X_)
        
        print(best_model) 

        if scoring == "neg_root_mean_squared_error":
            scoring = "rmse"
        if scoring == "neg_mean_absolute_error":
            scoring = "mae"

        if X_valid is not None and y_valid is not None:
            if not isinstance(y_valid, np.ndarray):
                y_valid = y_valid.to_numpy().reshape(-1, 1)

            _, X_valid_, y_valid_ = self._preprocess(pipeline, X_valid, y_valid)

            y_pred_valid = best_model.predict(X_valid_)

            train_score = model_score(y_, y_pred, scoring)
            valid_score = model_score(y_valid_, y_pred_valid, scoring)

            self._train_scores.append(train_score)
            self._valid_scores.append(valid_score)

            scores = {
                "train_score": train_score,
                "valid_score": valid_score,
            }
            logger.info(f"Tuning scores: {scores}")
        else:
            scores = {"train_score": model_score(y_, y_pred, scoring)}
            logger.info(f"Tuning scores: {scores}")
            
        metadata = model_metadata(
            model_name=model_name,
            model=best_model,
            model_params=best_model.get_params(),
            preprocess_steps=pipeline["preprocess"],
            target_steps=pipeline["target"],
            metric=scoring,
            scores=scores,
            X_data=X,
            y_data=y,
            X_test=X_valid,
            y_test=y_valid,
        )
            
        if store:
            gcp.write_model_to_bucket(
                bucket_name="values_tuned_models",
                blob_name=f"{model_name}_model.pkl",
                model=metadata,
            )

        return metadata

    @property
    def tuned_models(self) -> list[RegressorMixin]:
        return self._tuned_models

    @property
    def performance_weights(self) -> list[float]:
        return calculate_weights(self._valid_scores)
