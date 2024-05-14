import numpy as np

from analysis.gcp.storage import gcp
from analysis.utilities.logging import get_logger
from analysis.src_2.testing.tester import ModelTester
from analysis.src_2.prediction.blended import blended_prediction


logger = get_logger(__name__)


def test() -> None:
    logger.info("Loading data")
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

    logger.info("Loading tuned models")
    tuned_blobs = [
        gcp.read_model_from_bucket(
            bucket_name="values_tuned_models", blob_name=blob.name
        )
        for blob in blobs
    ]

    tuned_models = [
        dict((k, blob[k]) for k in ["model_name", "model"] if k in blob)
        for blob in tuned_blobs
    ]

    final_models = [(model["model_name"], model["model"]) for model in tuned_models]
    final_models_blend = [model["model"] for model in tuned_models]

    logger.info("Testing models")
    tester = ModelTester(models=final_models)

    logger.info("Scoring models")
    scores = tester.test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scoring="mae",
        store=True,
    )

    logger.info(scores)

    logger.info("Blending predictions")
    blended_pred = blended_prediction(
        models=final_models_blend,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        weights=tester.performance_weights,
        scoring="mae",
        inverse_func=np.expm1,
    )

    logger.info(f"Blended result: {blended_pred}")


if __name__ == "__main__":
    test()
