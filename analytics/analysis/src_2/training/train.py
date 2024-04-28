import numpy as np
from sklearn.model_selection import KFold

from analysis.gcp.storage import gcp
from analysis.utilities.logging import get_logger
from analysis.src_2.utils.data_splitter import train_valid_test_split
from analysis.src_2.preprocessing.pipeline.build import PipelineBuilder
from analysis.src_2.training.model_trainer import ModelTrainer
from analysis.src_2.models.regressors import Models
from analysis.src_2.models.blend import BlendedRegressor
from analysis.src_2.utils.metrics import model_score


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

    print(train.get_model_names)
    print(train.get_mean_scores)
    print(train.performance_weights)

    print("blending")
    blend_reg = (
        "blend",
        BlendedRegressor(models=Models().get_models, weights=train.performance_weights),
    )

    blend_pipeline = pipe.build(X_train, blend_reg)
    y_train_blend = blend_pipeline["target"].fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    )
    X_train_blend = blend_pipeline["preprocess"].fit_transform(
        X_train, y_train_blend.ravel()
    )
    
    
    blend = BlendedRegressor(models=Models().get_models, weights=train.performance_weights)
    y_pred = blend.fit_predict(X_train_blend, y_train_blend.ravel())
    
    y_pred = np.expm1(y_pred)
    y = np.expm1(y_train_blend)
    
    print(model_score(y.ravel(), y_pred, "mae"))


if __name__ == "__main__":
    train()
