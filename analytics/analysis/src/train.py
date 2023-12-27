from typing import Any
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from analysis.src.preprocessing import (
    prepare_data,
    split_x_y_vars,
    X_transformation_pipeline,
)
from analysis.src.models import (
    build_model,
    calculate_rmse,
    confidence_interval,
    model_evaluation_pipeline,
)
from analysis.src.data import db
from analysis.utils import get_logger
from analysis.model_manager import save
from analysis.config import grid_params, random_params
from analysis.src.hyperparameter_tuning import (
    grid_search_tuning,
    randomized_search_tuning,
)


logger = get_logger(__name__)


def train(models: list[Any], names: list[str]) -> None:
    logger.info("Querying data from the database")
    data = db.query("SELECT * FROM player_stats")

    logger.info("Preparing the data for training and testing")
    train_set, test_set = prepare_data(data)

    logger.info("Separating the target variable and exploratory variables")
    X_train, y_train = split_x_y_vars(train_set)

    logger.info("Transforming X_train data")
    X_train_prepared = X_transformation_pipeline(X_train)

    logger.info("Data has been loaded and transformed - ready for training")

    logger.info("Building regression models")
    best_model, _ = model_evaluation_pipeline(
        names=names, models=models, X=X_train_prepared, y=y_train
    )
    logger.info(f"The best model is: {best_model}")

    # tune best model
    logger.info(f"Tuning the {best_model}")

    # grid search
    logger.info("Grid search tuning")
    gs_best_estimator, _ = grid_search_tuning(
        best_model, grid_params, X_train_prepared, y_train
    )
    logger.info(f"Grid search best estimator: {gs_best_estimator}")

    # randomised search
    logger.info("Randomised search tuning")
    rs_best_estimator, _ = randomized_search_tuning(
        best_model, random_params, X_train_prepared, y_train
    )
    logger.info(f"Randomised search best estimator: {rs_best_estimator}")

    # build model on test data and evaluate which is the best
    logger.info("Building models on test data")
    X_test, y_test = split_x_y_vars(test_set)
    X_test_prepared = X_transformation_pipeline(X_test)

    logger.info("Evaluating the best hypertuning method on the test data")

    hpt_names = ["Grid Search", "Randomised Search"]
    models = [gs_best_estimator, rs_best_estimator]

    final_model, final_cv_rmse_score = model_evaluation_pipeline(
        names=hpt_names, models=models, X=X_train_prepared, y=y_train
    )
    logger.info(f"The best hypertuned model is: {final_model}")

    logger.info("Building the final model")
    final_pred = build_model(final_model, X_test_prepared, y_test)

    final_rmse = calculate_rmse(y_test, final_pred)
    logger.info(f"Final RMSE: {final_rmse}")

    ci_95 = confidence_interval(final_pred, y_test)
    logger.info(f"95% confidence interval for {final_model} RMSE score: {ci_95}")

    # load model to gcp bucket
    logger.info("Saving the model to the gcp bucket")
    save(
        model=final_model,
        predictions=final_pred,
        mean_cv_score=final_cv_rmse_score,
        X=X_test,
        y=y_test,
        file_name="points_prediction_model.pkl",
    )

    logger.info(f"{final_model} has been saved to the gcp bucket")
    logger.info("Training complete")


if __name__ == "__main__":
    names = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
    models = [
        LinearRegression(),
        DecisionTreeRegressor(random_state=42),
        RandomForestRegressor(random_state=42),
    ]
    train(models=models, names=names)
