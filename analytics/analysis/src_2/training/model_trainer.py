from typing import Literal, Any
import pandas as pd
from sklearn.base import RegressorMixin
from analysis.src_2.preprocessing.pipeline.build import PipelineBuilder
from analysis.src_2.training.cross_validation import cross_validate


class ModelTrainer(PipelineBuilder):
    def __init__(
        self,
        models: list[str, RegressorMixin],
        drop_features: list[str],
        target_encode_features: list[str],
    ) -> None:
        super().__init__(drop_features, target_encode_features)
        self.models = models
        self.num_features = []
        self.cat_features = []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        cv: Any,
        scoring: Literal["rmse", "mae", "r2"],
    ):
        self._cat_features(X)
        self._num_features(X)

        self.scores_ = [
            {
                name: cross_validate(
                    pipeline=self.build(model=model),
                    X=X,
                    y=y,
                    cv=cv,
                    scoring=scoring,
                )
            }
            for name, model in self.models
        ]

    @property
    def get_scores(self) -> list[dict[str, list[float]]]:
        return self.scores_