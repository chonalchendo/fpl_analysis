from typing import Any, Literal
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from analysis.utilities.utils import get_logger
from analysis.src_2.utils.metrics import model_score
from analysis.src_2.utils.model_metadata import model_metadata

logger = get_logger(__name__)


def validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    train_index: np.ndarray,
    test_index: np.ndarray,
    scoring: Literal["mae", "rmse", "r2"],
) -> float:
    """validation function for cross validation

    Args:
        pipeline (Pipeline): scikit learn pipeline
        X (pd.DataFrame): independent variables
        y (pd.Series): dependent variable
        train_index (np.ndarray): train split index
        test_index (np.ndarray): test split index
        scoring (Literal[&quot;mae&quot;, &quot;rmse&quot;, &quot;r2&quot;]): scoring metric

    Returns:
        float: scoring metric
    """
    clone_model = clone(pipeline["model"])

    logger.info("splitting data")
    X_train_folds = X.iloc[train_index]
    y_train_folds = y.iloc[train_index]
    X_test_fold = X.iloc[test_index]
    y_test_fold = y.iloc[test_index]

    logger.info("cleaning training data")
    # clean train data
    y_train_folds = pipeline["target"].fit_transform(
        y_train_folds.to_numpy().reshape(-1, 1)
    )
    X_train_folds = pipeline["preprocess"].fit_transform(
        X_train_folds, y_train_folds.ravel()
    )

    logger.info("cleaning test data")
    # clean test data
    y_test_fold = pipeline["target"].fit_transform(
        y_test_fold.to_numpy().reshape(-1, 1)
    )
    X_test_fold = pipeline["preprocess"].fit_transform(X_test_fold, y_test_fold.ravel())

    logger.info("fitting model")
    clone_model.fit(X_train_folds, y_train_folds.ravel())

    logger.info("predicting")
    y_pred = clone_model.predict(X_test_fold)

    y_pred = np.expm1(y_pred)
    y_test_fold = np.expm1(y_test_fold)

    logger.info("scoring")
    return model_score(y_test_fold, y_pred, scoring)


def cross_validate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring: Literal["mae", "rmse", "r2"],
) -> dict[str, Any]:
    """cross validation function

    Args:
        pipeline (Pipeline): scikit learn pipeline
        X (pd.DataFrame): independent variables
        y (pd.Series): dependent variable
        cv (Any): cross validation method
        scoring (Literal[&quot;mae&quot;, &quot;rmse&quot;, &quot;r2&quot;]):
        scoring metric

    Returns:
        dict[str, Any]: model meta data
    """
    logger.info(f"cross validating model: {pipeline['model']}")

    scores = [
        validation(
            pipeline=pipeline,
            X=X,
            y=y,
            train_index=train_index,
            test_index=test_index,
            scoring=scoring,
        )
        for train_index, test_index in cv.split(X, y)
    ]

    metadata = model_metadata(
        model=pipeline["model"],
        model_params=pipeline["model"].get_params(),
        preprocess_steps=pipeline["preprocess"],
        target_steps=pipeline["target"],
        metric=scoring,
        scores=scores,
        mean_scores=np.mean(scores),
        std_scores=np.std(scores),
        X_data=X,
        y_data=y,
    )

    logger.info(metadata)

    return metadata
