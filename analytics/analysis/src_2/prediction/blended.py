from typing import Callable, Literal
import pandas as pd
from sklearn.base import RegressorMixin, clone

from analysis.src_2.models.blend import BlendedRegressor
from analysis.src_2.utils.metrics import model_score
from analysis.src_2.preprocessing.pipeline.build import PipelineBuilder


def blended_prediction(
    # pipeline: PipelineBuilder,
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
    blend_pipeline = pipeline.build(X_train, blend_reg)
    
    y_train_blend = blend_pipeline["target"].fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    )
    X_train_blend = blend_pipeline["preprocess"].fit_transform(
        X_train, y_train_blend.ravel()
    )
    model = blend_pipeline["blend"]
    model.fit(X_train_blend, y_train_blend.ravel())

    if X_test is not None and y_test is not None:
        y_test_blend = blend_pipeline["target"].fit_transform(
            y_test.to_numpy().reshape(-1, 1)
        )
        X_test_blend = blend_pipeline["preprocess"].fit_transform(
            X_test, y_test_blend.ravel()
        )
        y_pred = model.predict(X_test_blend)
        y = inverse_func(y_test_blend)
    else:
        y_pred = model.predict(X_train_blend)
        y = inverse_func(y_train_blend)

    return model_score(y.ravel(), y_pred, scoring)
