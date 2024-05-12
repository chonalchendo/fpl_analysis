from typing import Callable, Literal
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin

from analysis.src_2.models.blend import BlendedRegressor
from analysis.src_2.utils.metrics import model_score
from analysis.src_2.preprocessing.pipeline.build import PipelineBuilder


def blended_prediction(
    models: list[RegressorMixin],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    weights: list[float],
    scoring: Literal["mae", "r2", "rmse"],
    X_test: pd.DataFrame | None = None,
    y_test: pd.DataFrame | None = None,
    inverse_func: Callable | None = None,
) -> float:
    blend_reg = (
        "blend",
        BlendedRegressor(
            models=models,
            weights=weights,
            inverse_func=inverse_func,
        ),
    )
    pipeline = PipelineBuilder()
    train_pipe = pipeline.build(X_train, blend_reg)
    
    y_train_blend = train_pipe["target"].fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    )
    X_train_blend = train_pipe["preprocess"].fit_transform(
        X_train, y_train_blend.ravel()
    )
    model = train_pipe["blend"]

    if X_test is not None and y_test is not None:
        test_pipe = pipeline.build(X_test, blend_reg)
        y_test_blend = test_pipe["target"].fit_transform(
            y_test.to_numpy().reshape(-1, 1)
        )
        X_test_blend = test_pipe["preprocess"].fit_transform(
            X_test, y_test_blend.ravel()
        )
        
        if X_train_blend.shape[1] != X_test_blend.shape[1]:
            # find the missing columns
            X_train_df = pd.DataFrame(
                X_train_blend,
                columns=train_pipe["preprocess"]["transformer"].get_feature_names_out(),
            )
            
            X_test_df = pd.DataFrame(
                X_test_blend,
                columns=test_pipe["preprocess"]["transformer"].get_feature_names_out(),
            )
            
            missing_cols = np.setxor1d(X_train_df.columns, X_test_df.columns)
            X_train_df = X_train_df.drop(columns=missing_cols)
            
            X_train_blend = X_train_df.to_numpy()
        
        model.fit(X_train_blend, y_train_blend.ravel())
        y_pred = model.predict(X_test_blend)
        y = inverse_func(y_test_blend)
    else:
        y_pred = model.predict(X_train_blend)
        y = inverse_func(y_train_blend)

    return model_score(y.ravel(), y_pred, scoring)
