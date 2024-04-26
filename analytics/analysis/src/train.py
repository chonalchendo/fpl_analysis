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
    calculate_rmse,
    confidence_interval,
    model_evaluation_pipeline,
    build_pipeline,
)
from analysis.src.data import db
from analysis.utilities.logging import get_logger
from analysis.models.model_manager import save
from analysis.models.hyperparamter_config import grid_params, random_params
from analysis.src.hyperparameter_tuning import (
    hyperparameter_tuning,
)


logger = get_logger(__name__)


def train(models: list[Any], names: list[str]) -> None:
    """Train and output the best model to the gcp bucket.

    Args:
        models (list[Any]): Regression models to consider
        names (list[str]): Names of each regression model
    """
    logger.info("Querying data from the database")
    data = db.query("SELECT * FROM player_stats")

    logger.info("Preparing the data for training and testing")
    train_set, test_set = prepare_data(data)

    logger.info("Separating the target variable and exploratory variables")
    X_train, y_train = split_x_y_vars(train_set)

    logger.info("Transforming X_train data")
    # X_train_prepared, _ = X_transformation_pipeline(X_train)
    X_train_preprocessor = X_transformation_pipeline(X_train)

    logger.info("Data has been loaded and transformed - ready for training")

    logger.info("Create model pipelines")
    model_pipelines = [
        (name, build_pipeline(model, X_train_preprocessor))
        for name, model in zip(names, models)
    ]

    logger.info("Building regression models")
    best_model_pipeline = model_evaluation_pipeline(
        model_pipelines=model_pipelines,
        X=X_train,
        y=y_train,
    )
    logger.info(f"The best model is: {best_model_pipeline}")

    # tune best model
    logger.info(f"Tuning {best_model_pipeline}")

    final_model, final_model_rmse = hyperparameter_tuning(
        regressor=best_model_pipeline,
        grid_params=grid_params,
        random_params=random_params,
        X_train=X_train,
        y_train=y_train,
    )

    logger.info(f"The best hypertuned model is: {final_model}")

    # build model on test data and evaluate which is the best
    logger.info("Building models on test data")
    X_test, y_test = split_x_y_vars(test_set)

    logger.info("Building the final model")
    final_pred = final_model.predict(X_test)

    final_rmse = calculate_rmse(y_test, final_pred)
    logger.info(f"Final RMSE: {final_rmse}")

    ci_95 = confidence_interval(final_pred, y_test)
    logger.info(f"95% confidence interval for {final_model} RMSE score: {ci_95}")

    # load model to gcp bucket
    logger.info("Saving the model to the gcp bucket")
    save(
        model=final_model,
        predictions=final_pred,
        model_rmse=final_model_rmse,
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
