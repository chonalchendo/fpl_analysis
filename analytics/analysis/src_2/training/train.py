from typing import Any, Literal
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split

from analysis.gcp.storage import gcp
from analysis.src_2.training.cross_validation import cross_validate
from analysis.src_2.preprocessing.transformers.drop_features import DropFeatures
from analysis.src_2.preprocessing.pipeline.processors import (
    signed_from_impute,
    signed_year_impute,
    signing_fee_impute,
    stats_imputer,
    youth_player_feat,
    pen_taker_feat,
    year_since_signed_feat,
)
from analysis.src_2.preprocessing.pipeline.build import (
    num_transformer,
    cat_transformer,
    target_pipeline,
    dimension_reduction,
)


class TrainModel:
    def __init__(
        self,
        models: list[str, RegressorMixin],
        drop_features: list[str],
        target_encode_features: list[str],
    ) -> None:
        self.models = models
        self.drop_features = drop_features
        self.target_encode_features = target_encode_features
        self.num_features = []
        self.cat_features = []

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
                ("target", target_pipeline, self.target_encode_features),
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

    def _pipeline(self, model: RegressorMixin) -> Pipeline:
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

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Any,
        scoring: Literal["rmse", "mae", "r2"],
    ) -> None:

        self.cat_features = [
            col
            for col in X.select_dtypes(include=["object"]).columns.tolist()
            if col not in self.target_encode_features and col not in self.drop_features
        ]
        self.num_features = [
            col
            for col in X.select_dtypes(include=["int64", "float64"]).columns.tolist()
            if col not in self.drop_features
        ]

        # preprocess data
        scores = [
            {
                name: cross_validate(
                    pipeline=self._pipeline(model=model),
                    X=X,
                    y=y,
                    cv=cv,
                    scoring=scoring,
                )
            }
            for name, model in self.models
        ]

        print(scores)


if __name__ == "__main__":

    df = gcp.read_df_from_bucket(
        bucket_name="wage_vals_stats", blob_name="standard.csv"
    )
    # create train, valid, test splits
    train_set = df.loc[df["season"] != 2023]
    valid_test_set = df.loc[df["season"] == 2023]

    valid_set, test_set = train_test_split(
        valid_test_set,
        test_size=0.5,
        stratify=valid_test_set["position"],
        random_state=42,
    )

    X_train, y_train = (
        train_set.drop("market_value_euro_mill", axis=1),
        train_set["market_value_euro_mill"],
    )

    models = [
        ("rf", RandomForestRegressor(random_state=42)),
        ("lr", LinearRegression()),
    ]

    drop_features = [
        "player_id",
        "rk",
        "general_pos",
        "pos",
        "nation",
        "comp",
        "born",
        "annual_wages_euros",
        "player",
        "age_range",
    ]

    target_encode_features = ["squad", "country", "signed_from"]

    train = TrainModel(
        models=models,
        drop_features=drop_features,
        target_encode_features=target_encode_features,
    )
    train.train(
        X=X_train,
        y=y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="rmse",
    )
