from typing import Literal, Any
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from analysis.gcp.storage import gcp
from analysis.src_2.training.cross_validation import cross_validate
from analysis.src_2.prediction.weights import calculate_weights


class ModelTrainer:
    """Class that is used to train models scikit-learn regression models.

    Args:
        models (list[str, RegressorMixin]): list of model names and scikit-learn models
    """

    def __init__(
        self,
        pipelines: list[Pipeline],
    ) -> None:
        self.pipelines = pipelines
        self.num_features = []
        self.cat_features = []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        cv: Any,
        scoring: Literal["rmse", "mae", "r2"],
        store: bool = False,
    ) -> None:
        """Train models.

        Args:
            X (pd.DataFrame): X data
            y (pd.DataFrame): y data
            cv (Any): cross-validation strategy
            scoring (Literal[&quot;rmse&quot;, &quot;mae&quot;, &quot;r2&quot;]): evaluation metric
            store (bool, optional): store models in google cloud bucket. Defaults to False.
        """

        self.metadata = [
            self._cross_validate_model(pipeline, X, y, cv, scoring, store)
            for pipeline in self.pipelines
        ]

    def _cross_validate_model(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.DataFrame,
        cv: Any,
        scoring: Literal["rmse", "mae", "r2"],
        store: bool = False,
    ) -> dict[str, Any]:
        """Cross validate a model.

        Args:
            name (str): model name
            model (Any): scikit-learn model
            X (pd.DataFrame): X data
            y (pd.DataFrame): y data
            cv (Any): cross-validation strategy
            scoring (Literal[&quot;rmse&quot;, &quot;mae&quot;, &quot;r2&quot;]): evaluation metric
            store (bool, optional): store model in GCP bucket. Defaults to False.

        Returns:
            dict: model metadata
        """
        metadata = cross_validate(
            pipeline=pipeline,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
        )

        model_name = list(pipeline.named_steps.items())[2][0]

        if store:
            gcp.write_model_to_bucket(
                bucket_name="values_trained_models",
                blob_name=f"{model_name}_model.pkl",
                model=metadata,
            )

        return metadata

    @property
    def get_metadata(self) -> list[dict[str, Any]]:
        return self.metadata

    @property
    def get_model_names(self) -> list[str]:
        return [model["model_name"] for model in self.metadata]

    @property
    def get_models(self) -> list[RegressorMixin]:
        return [model["model"] for model in self.metadata]

    @property
    def get_mean_scores(self) -> list[float]:
        return [model["mean_score"] for model in self.metadata]

    @property
    def performance_weights(self) -> list[float]:
        mean_scores = [model["mean_score"] for model in self.metadata]
        return calculate_weights(mean_scores)
