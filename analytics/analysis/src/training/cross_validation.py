from typing import Any, Literal
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from analysis.utilities.logging import get_logger
from analysis.src_2.utils.metrics import model_score
from analysis.src_2.utils.model_metadata import model_metadata
from analysis.src_2.prediction.mean_pred import average_arrays

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
    clone_model = clone(pipeline[2])

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
    
    print(y_pred) 

    logger.info("scoring")
    return model_score(y_test_fold, y_pred, scoring), y_pred


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
    logger.info(f"cross validating model: {pipeline[2]}")

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

    model_name = list(pipeline.named_steps.items())[2][0]
    
    mae_scores = [score[0] for score in scores]
    y_preds = [score[1] for score in scores]

    metadata = model_metadata(
        model_name=model_name,
        model=pipeline[2],
        model_params=pipeline[2].get_params(),
        preprocess_steps=pipeline["preprocess"],
        target_steps=pipeline["target"],
        metric=scoring,
        scores=mae_scores,
        mean_score=np.mean(mae_scores),
        std_score=np.std(mae_scores),
        X_data=X,
        y_data=y,
    )

    logger.info(
        f"\ncross validation scores: {scores}, \nmean: {metadata['mean_score']}, \nstd: {metadata['std_score']}"
    )

    return metadata