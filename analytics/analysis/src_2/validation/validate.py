import numpy as np
from rich import print
from sklearn.model_selection import RandomizedSearchCV

from analysis.gcp.storage import gcp
from analysis.src_2.models.regressors import Models
from analysis.src_2.validation.parameters import params_dict
from analysis.src_2.validation.tuning import ModelValidator
from analysis.src_2.prediction.blended import blended_prediction
from analysis.utilities.logging import get_logger

logger = get_logger(__name__)


def validate() -> None:

    train_set = gcp.read_df_from_bucket(
        bucket_name="values_training_data", blob_name="train_set_2017_2022.csv"
    )
    X_train, y_train = (
        train_set.drop("market_value_euro_mill", axis=1),
        train_set["market_value_euro_mill"],
    )

    # load in validation data
    valid_set = gcp.read_df_from_bucket(
        bucket_name="values_validation_data", blob_name="valid_set_2023.csv"
    )
    X_valid, y_valid = (
        valid_set.drop("market_value_euro_mill", axis=1),
        valid_set["market_value_euro_mill"],
    )

    # load in models
    models = Models().get_models
    params = params_dict()

    validator = ModelValidator(
        model=models, search_method=RandomizedSearchCV, params=params
    )

    scores = validator.run(
        X=X_train,
        y=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_iter=10,
        store=True,
    )

    logger.info(scores)

    print(validator.tuned_models)
    print(validator.performance_weights)

    blended_pred = blended_prediction(
        models=validator.tuned_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_valid,
        y_test=y_valid,
        weights=validator.performance_weights,
        scoring="mae",
        inverse_func=np.expm1,
    )

    logger.info(blended_pred)

    # load in hyperparameters
    # tune on training data
    # validate on validation data
    #


if __name__ == "__main__":
    validate()
