from typing import Any
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from analysis.utilities.utils import get_logger

logger = get_logger(__name__)


def hyperparameter_tuning_results(
    cv_results: dict[str, np.ndarray]
) -> list[tuple[Any, Any]]:
    """Display the results of the hyperparameter tuning.

    Args:
        cv_results (dict[str, np.ndarray]): hyperparameter tuning results

    Returns:
        list[tuple[Any, Any]]: list of tuples containing the RMSE score and hyperparameters
    """
    return [
        (np.sqrt(-mean_score), params)
        for mean_score, params in zip(
            cv_results["mean_test_score"], cv_results["params"]
        )
    ]


def grid_search_tuning(
    regressor: Pipeline,
    params: list[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> tuple[Pipeline, dict[str, np.ndarray]]:
    """Tune the hyperparameters of the model using grid search.

    Args:
        regressor (Pipeline): regression model
        params (list[dict]): hyperparameters to tune
        X_train (np.ndarray): X train variables
        y_train (np.ndarray): y train variable
        cv (int, optional): number of cross validations. Defaults to 5.

    Returns:
        tuple[Pipeline, dict[str, np.ndarray]]: best estimator and grid search results
    """
    grid_search = GridSearchCV(
        regressor,
        params,
        cv=cv,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.cv_results_


def randomized_search_tuning(
    regressor: Pipeline,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> tuple[Pipeline, dict[str, np.ndarray]]:
    """Tune the hyperparameters of the model using randomised search.

    Args:
        regressor (Pipeline): regression model
        params (dict): hyperparameters to tune
        X_train (np.ndarray): X train variables
        y_train (np.ndarray): y train variable
        cv (int, optional): number of cross validations. Defaults to 5.

    Returns:
        tuple[Pipeline, dict[str, np.ndarray]]: best estimator and randomised search results
    """
    rnd_search = RandomizedSearchCV(
        regressor,
        param_distributions=params,
        n_iter=10,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X_train, y_train)

    return rnd_search.best_estimator_, rnd_search.cv_results_


def hyperparameter_tuning(
    regressor: Pipeline,
    grid_params: list[dict],
    random_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Pipeline, float]:
    """Tune the hyperparameters of the model using grid search and randomised search.

    Args:
        regressor (Pipeline): regression model pipeline
        grid_params (list[dict]): grid search hyperparameters
        random_params (dict): randomised search hyperparameters
        X_train (np.ndarray): X_train variables
        y_train (np.ndarray): y_train variable

    Returns:
        tuple[Pipeline, float]: best estimator pipeline and RMSE score
    """
    logger.info("Tuning the model using Grid Search")
    gs_best_estimator, gs_cv_results = grid_search_tuning(
        regressor, grid_params, X_train, y_train
    )

    gs_rmse_scores = hyperparameter_tuning_results(gs_cv_results)
    gs_scores = [score for score, _ in gs_rmse_scores]

    logger.info(
        f"\nGrid search RMSE scores: {gs_rmse_scores}\n Grid search best estimator: {gs_best_estimator}"
    )

    logger.info("Tuning the model using Randomised Search")
    rs_best_estimator, rs_cv_results = randomized_search_tuning(
        regressor, random_params, X_train, y_train
    )
    rs_rmse_scores = hyperparameter_tuning_results(rs_cv_results)
    rs_scores = [score for score, _ in rs_rmse_scores]

    logger.info(
        f"\nRandomised search RMSE scores: {rs_rmse_scores}\n Ramdomised search best estimator: {rs_best_estimator}"
    )

    if min(gs_scores) < min(rs_scores):
        return gs_best_estimator, min(gs_scores)
    else:
        return rs_best_estimator, min(rs_scores)
