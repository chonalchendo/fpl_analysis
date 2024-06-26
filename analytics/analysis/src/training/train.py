import numpy as np
from sklearn.model_selection import KFold

from analysis.gcp.storage import gcp
from analysis.src.models.regressors import Models
from analysis.src.prediction.blended import blended_prediction
from analysis.src.preprocessing.pipeline.build import PipelineBuilder
from analysis.src.training.model_trainer import ModelTrainer
from analysis.src.utils.data_splitter import train_valid_test_split
from analysis.utilities.logging import get_logger

logger = get_logger(__name__)


def train() -> None:
    df = gcp.read_df_from_bucket(
        bucket_name="wage_vals_stats", blob_name="standard.csv"
    )

    train_set, _, _ = train_valid_test_split(df, season_split=2023)

    X_train, y_train = (
        train_set.drop("market_value_euro_mill", axis=1),
        train_set["market_value_euro_mill"],
    )

    pipe = PipelineBuilder()
    models = [pipe.build(X_train, model) for model in Models().get_models]

    train = ModelTrainer(pipelines=models)
    train.train(
        X=X_train,
        y=y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="mae",
        store=False,
    )

    logger.info(f"Performance weights: {train.performance_weights}")

    logger.info("Blending models together")
    blended_pred = blended_prediction(
        pipeline=pipe,
        X_train=X_train,
        y_train=y_train,
        weights=train.performance_weights,
        scoring="mae",
        inverse_func=np.expm1,
    )

    logger.info(blended_pred)


if __name__ == "__main__":
    train()
