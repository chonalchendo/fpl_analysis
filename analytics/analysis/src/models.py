import numpy as np
from typing import Any
from analysis.utilities.utils import get_logger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from scipy import stats

logger = get_logger(__name__)


def display_scores(rmse: np.ndarray) -> None:
    """Display the Root Mean Square Error (RMSE) scores of the model.

    Args:
        rmse (np.ndarray): RMSE score
    """
    logger.info(
        f"\n RMSE: {rmse}\n Mean: {rmse.mean()}\n Standard deviation: {rmse.std()}"
    )


def calculate_rmse(y_train: np.ndarray, points_predictions: np.ndarray) -> np.ndarray:
    """Calculate the RMSE of the model.

    Args:
        y_train (np.ndarray): y train variable
        points_predictions (np.ndarray): predicted points
    """
    return np.sqrt(mean_squared_error(y_train, points_predictions))


def confidence_interval(
    prediction: np.ndarray,
    y_test: np.ndarray,
    confidence: int = 0.95,
):
    """Compute confidence interval for the test RMSE.

    Args:
        prediction (np.ndarray): prediction array
        y_test (np.ndarray): y test array
        confidence (int, optional): confidence interval. Defaults to 0.95.

    Returns:
        np.ndarray: confidence interval for RMSE
    """
    squared_errors = (prediction - y_test) ** 2
    return np.sqrt(
        stats.t.interval(
            confidence,
            len(squared_errors) - 1,
            loc=squared_errors.mean(),
            scale=stats.sem(squared_errors),
        )
    )


def build_model(
    regressor: LinearRegression | DecisionTreeRegressor | RandomForestRegressor,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Build the model and predict the points.

    Args:
        regressor (LinearRegression | DecisionTreeRegressor | RandomForestRegressor): regression model
        X (np.ndarray): explanatory variables
        y (np.ndarray): target variable

    Returns:
        np.ndarray: _description_
    """
    regressor.fit(X, y)
    points_predictions = regressor.predict(X)
    return points_predictions


def evaluate_model(
    regressor: LinearRegression | DecisionTreeRegressor | RandomForestRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """Evaluate the model using cross validation. Return the mean RMSE. The
    model with the lowest RMSE is the best model.

    Args:
        regressor (LinearRegression | DecisionTreeRegressor | RandomForestRegressor): regression model
        X_train (np.ndarray): X train variables
        y_train (np.ndarray): y train variable

    Returns:
        np.ndarray: mean RMSE
    """
    scores = cross_val_score(
        regressor,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=10,
    )
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
    return rmse_scores.mean()


def model_evaluation_pipeline(
    names: list[str], models: list[Any], X: np.ndarray, y: np.ndarray
) -> tuple[Any, np.ndarray]:
    """Pipeline that builds, evaluates and selects the best model.

    Args:
        names (list[str]): name of models or hyper tuning methods
        models (list[Any]): models or hyper tuning methods
        X (np.ndarray): Explanatory variables
        y (np.ndarray): Target variable

    Returns:
        Any: best model
    """
    model_scores = []
    for name, model in zip(names, models):
        logger.info(f"Evaluating {name}")
        points_prediction = build_model(model, X, y)

        logger.info(f"Calculating RMSE for {name}")
        calculate_rmse(y, points_prediction)

        logger.info(f"Cross-validate {name}")
        rmse_mean = evaluate_model(model, X, y)
        model_scores.append(rmse_mean)

    smallest_rmse = min(model_scores)
    index_pos = model_scores.index(smallest_rmse)
    model = models[index_pos]
    name = names[index_pos]
    logger.info(
        f"The best method is: {name}\n The rmse score is: {model_scores[index_pos]}"
    )

    return model, smallest_rmse
