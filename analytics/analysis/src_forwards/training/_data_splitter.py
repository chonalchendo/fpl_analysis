import pandas as pd
from sklearn.model_selection import train_test_split

from analysis.gcp.storage import gcp
from analysis.utilities.logging import get_logger

logger = get_logger(__name__)


def train_valid_test_split(
    df: pd.DataFrame,
    season_split: int,
    stratify_by: str = "position",
    save: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame")

    train_set = df.loc[df["season"] != season_split]
    valid_test_set = df.loc[df["season"] == season_split]

    valid_set, test_set = train_test_split(
        valid_test_set,
        test_size=0.5,
        stratify=valid_test_set[stratify_by],
        random_state=42,
    )

    if save:
        logger.info("Writing train, valid and test sets to bucket")
        gcp.write_df_to_bucket(
            data=train_set,
            bucket_name="values_training_data",
            blob_name=f"train_set_2017_{season_split - 1}.csv",
        )

        gcp.write_df_to_bucket(
            data=valid_set,
            bucket_name="values_validation_data",
            blob_name=f"valid_set_{season_split}.csv",
        )

        gcp.write_df_to_bucket(
            data=test_set,
            bucket_name="values_test_data",
            blob_name=f"test_set_{season_split}.csv",
        )

        gcp.write_df_to_bucket(
            data=valid_test_set,
            bucket_name="values_test_data",
            blob_name=f"valid_test_set_{season_split}.csv",
        )

    return train_set, valid_set, test_set, valid_test_set
