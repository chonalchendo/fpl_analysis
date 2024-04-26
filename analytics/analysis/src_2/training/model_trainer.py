from typing import Literal, Any
import pandas as pd
from sklearn.base import RegressorMixin

from analysis.gcp.storage import gcp
from analysis.src_2.preprocessing.pipeline.build import PipelineBuilder
from analysis.src_2.training.cross_validation import cross_validate


class ModelTrainer(PipelineBuilder):
    """Class that is used to train models scikit-learn regression models.

    Args:
        PipelineBuilder: Custom class that builds a scikit-learn pipeline for data preprocessing
        models (list[str, RegressorMixin]): list of model names and scikit-learn models
        drop_features (list[str]): list of features to drop
        target_encode_features (list[str]): list of features to target encode
    """
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
        self._cat_features(X)
        self._num_features(X)

        self.metadata = [
            {name: self._cross_validate_model(name, model, X, y, cv, scoring, store)}
            for name, model in self.models
        ]

    def _cross_validate_model(
        self,
        name: str,
        model: Any,
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
            dict: _description_
        """
        metadata = cross_validate(
            pipeline=self.build(model=model),
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
        )

        if store:
            gcp.write_model_to_bucket(
                bucket_name="values_trained_models",
                blob_name=f"{name}_model.pkl",
                model=metadata,
            )

        return metadata

    @property
    def get_metadata(self) -> list[dict[str, Any]]:
        return self.metadata
