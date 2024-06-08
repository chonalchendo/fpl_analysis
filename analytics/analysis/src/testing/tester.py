from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.pipeline import Pipeline

from analysis.gcp.storage import gcp
from analysis.src.prediction.weights import calculate_weights
from analysis.src.preprocessing.pipeline.build import PipelineBuilder as pb
from analysis.src.utils.metrics import model_score
from analysis.src.utils.model_metadata import model_metadata


class ModelTester:
    def __init__(self, models: list[RegressorMixin]) -> None:
        self.models = models

    def test(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame | np.ndarray,
        scoring: Literal["rmse", "mae", "r2"],
        store: bool = False,
    ) -> float:
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.X_test_ = X_test
        self.y_test_ = y_test
        self.scoring_ = scoring
        self.test_scores = []
        self.store_ = store

        return [
            {list(pipe.named_steps.keys())[2]: self._tester(pipeline=pipe)}
            for pipe in self._pipeline(X=X_train)
        ]

    def _pipeline(self, X: pd.DataFrame) -> list[Pipeline]:
        return [pb().build(X=X, model=model) for model in self.models]

    def _preprocess(
        self, pipeline: Pipeline, X: pd.DataFrame, y: pd.DataFrame | np.ndarray
    ) -> tuple[RegressorMixin, np.ndarray, np.ndarray]:
        clone_model = clone(pipeline[2])
        y_ = pipeline["target"].fit_transform(y).ravel()
        X_ = pipeline["preprocess"].fit_transform(X, y_)
        X_df = pd.DataFrame(
            X_, columns=pipeline["preprocess"]["transformer"].get_feature_names_out()
        )
        return clone_model, X_df, y_

    def _tester(
        self,
        pipeline: Pipeline,
    ) -> float:
        if not isinstance(self.y_train_, np.ndarray):
            self.y_train_ = self.y_train_.to_numpy().reshape(-1, 1)

        if not isinstance(self.y_test_, np.ndarray):
            self.y_test_ = self.y_test_.to_numpy().reshape(-1, 1)

        model, X_train_, y_train_ = self._preprocess(
            pipeline, self.X_train_, self.y_train_
        )
        _, X_test_, y_test_ = self._preprocess(pipeline, self.X_test_, self.y_test_)

        if X_train_.filter(like="Oceania").shape[1] > 0:
            col_to_drop = X_train_.filter(like="Oceania").columns
            X_train_ = X_train_.drop(columns=col_to_drop)

        X_train_ = X_train_.to_numpy()
        X_test_ = X_test_.to_numpy()

        model.fit(X_train_, y_train_)

        y_pred = model.predict(X_test_)

        score = model_score(y_test_, y_pred, self.scoring_)
        self.test_scores.append(score)

        metadata = model_metadata(
            model_name=list(pipeline.named_steps.keys())[2],
            model=model,
            model_params=model.get_params(),
            preprocess_steps=pipeline["preprocess"],
            target_steps=pipeline["target"],
            metric=self.scoring_,
            scores=score,
            X_data=self.X_train_,
            y_data=self.y_train_,
            X_test=self.X_test_,
            y_test=self.y_test_,
        )

        if self.store_:
            gcp.write_model_to_bucket(
                bucket_name="values_tested_models",
                blob_name=f"{list(pipeline.named_steps.keys())[2]}_model.pkl",
                model=metadata,
            )

        return metadata

    @property
    def get_scores(self) -> list[float]:
        return self.test_scores

    @property
    def performance_weights(self) -> list[float]:
        return calculate_weights(self.test_scores)
