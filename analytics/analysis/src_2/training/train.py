from sklearn.model_selection import KFold

from analysis.gcp.storage import gcp
from analysis.utilities.utils import get_logger
from analysis.src_2.utils.data_splitter import train_valid_test_split
from analysis.src_2.training.model_trainer import ModelTrainer
from analysis.src_2.training.models import models
from analysis.src_2.training.features import drop_features, target_encode_features


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

    train = ModelTrainer(
        models=models,
        drop_features=drop_features,
        target_encode_features=target_encode_features,
    )

    train.train(
        X=X_train,
        y=y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="mae",
    )

    # print(train.get_scores)


if __name__ == "__main__":
    train()
