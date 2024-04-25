import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer

from analysis.src_2.preprocessing.transformers.drop_features import DropFeatures
from analysis.src_2.preprocessing.pipeline.process import (
    signed_from_impute,
    signed_year_impute,
    signing_fee_impute,
    stats_imputer,
)
from analysis.src_2.preprocessing.pipeline.transform import (
    num_transformer,
    cat_transformer,
    target_encoder,
    dimension_reduction,
    youth_player_feat,
    pen_taker_feat,
    year_since_signed_feat,
)

    
class PipelineBuilder:
    def __init__(
        self,
        drop_features: list[str],
        target_encode_features: list[str],
    ) -> None:
        self.drop_features = drop_features
        self.target_encode_features = target_encode_features
        self.num_features = []
        self.cat_features = []

    def _cat_features(self, X: pd.DataFrame) -> None:
        self.cat_features = [
            col
            for col in X.select_dtypes(include=["object"]).columns.tolist()
            if col not in self.target_encode_features and col not in self.drop_features
        ]

    def _num_features(self, X: pd.DataFrame) -> None:
        self.num_features = [
            col
            for col in X.select_dtypes(include=["int64", "float64"]).columns.tolist()
            if col not in self.drop_features
        ]

    def _preprocessor(self) -> Pipeline:
        return make_pipeline(
            youth_player_feat,
            signed_from_impute,
            signed_year_impute,
            signing_fee_impute,
            stats_imputer,
            DropFeatures(features=self.drop_features),
        )

    def _X_transformer(self) -> ColumnTransformer:

        return ColumnTransformer(
            [
                ("is_pen_taker", pen_taker_feat, ["penalty_kicks_attempted"]),
                (
                    "year_since_signed",
                    year_since_signed_feat,
                    ["signed_year", "season"],
                ),
                ("target", target_encoder, self.target_encode_features),
                ("decomp", dimension_reduction, ["age", "weekly_wages_euros"]),
                ("num", num_transformer, self.num_features),
                ("cat", cat_transformer, self.cat_features),
            ],
            # remainder="drop",
        )

    def _y_transformer(self) -> Pipeline:
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(
                func=np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one"
            ),
            StandardScaler(),
        )

    def build(self, model: RegressorMixin) -> Pipeline:
        preprocessor = Pipeline(
            [
                ("cleaner", self._preprocessor()),
                ("transformer", self._X_transformer()),
            ],
            verbose=True,
        )

        return Pipeline(
            [
                ("preprocess", preprocessor),
                ("target", self._y_transformer()),
                ("model", model),
            ],
            verbose=True,
        )