import pandas as pd
from rich import print
from functools import lru_cache
from sklearn.base import RegressorMixin

from analysis.gcp.storage import gcp
from analysis.src_2.utils.metrics import model_score


# make predictions
# compare scores
# blend predictions
# log and save results


def make_prediction(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model: RegressorMixin,
) -> pd.DataFrame:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


@lru_cache
def test() -> None:
    train_set = gcp.read_df_from_bucket(
        bucket_name="values_training_data",
        blob_name="train_set_2017_2022.csv",
    )

    test_set = gcp.read_df_from_bucket(
        bucket_name="values_test_data",
        blob_name="test_set_2023.csv",
    )

    X_train, y_train = (
        train_set.drop("market_value_euro_mill", axis=1),
        train_set["market_value_euro_mill"],
    )

    X_test, y_test = (
        test_set.drop("market_value_euro_mill", axis=1),
        test_set["market_value_euro_mill"],
    )

    blobs = gcp.get_gcp_bucket(bucket_name="values_tuned_models").list_blobs()

    tuned_models = [
        gcp.read_model_from_bucket(
            bucket_name="values_tuned_models", blob_name=blob.name
        )["model"]
        for blob in blobs
    ]
    
    predictions = [
        make_prediction(X_train, y_train, X_test, model)
        for model in tuned_models
    ]
    
    print(predictions)
    
    scores = [
        model_score(y_test, pred, "mae")
        for pred in predictions
    ]
    
    print(scores) 


if __name__ == "__main__":
    test()
