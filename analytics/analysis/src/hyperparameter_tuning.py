from typing import Any
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# import joblib
# from analysis.settings import ROOT


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
    regressor: Any,
    params: list[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> tuple[Any, dict[str, np.ndarray]]:
    """Tune the hyperparameters of the model using grid search.

    Args:
        regressor (Any): regression model
        params (list[dict]): hyperparameters to tune
        X_train (np.ndarray): X train variables
        y_train (np.ndarray): y train variable
        cv (int, optional): number of cross validations. Defaults to 5.

    Returns:
        tuple[Any, dict[str, np.ndarray]]: best estimator and grid search results
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
    regressor: Any, params: dict, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5
) -> tuple[Any, dict[str, np.ndarray]]:
    """Tune the hyperparameters of the model using randomised search.

    Args:
        regressor (Any): regression model
        params (dict): hyperparameters to tune
        X_train (np.ndarray): X train variables
        y_train (np.ndarray): y train variable
        cv (int, optional): number of cross validations. Defaults to 5.

    Returns:
        tuple[Any, dict[str, np.ndarray]]: best estimator and randomised search results
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


# -------------------------------- Save model -------------------------------- #

# PATH = f"{ROOT}/analysis/models/forest_reg_model.pkl"

# # save model
# joblib.dump(final_model, PATH)
