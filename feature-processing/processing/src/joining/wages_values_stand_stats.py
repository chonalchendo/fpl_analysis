import pandas as pd
from rich import print

from processing.gcp.storage import gcp
from processing.utilities.logger import get_logger

logger = get_logger(__name__)


def load_wages_values(league: str) -> pd.DataFrame:
    """Function that specifies a league to load the wages and values data for
    that league

    Args:
        league (str): league to load data for

    Returns:
        pd.DataFrame: dataframe containing the wages and values data for the
        specified league
    """
    return gcp.read_df_from_bucket(
        bucket_name="joined_wages_values", blob_name=f"{league}_wages_values.csv"
    )


def clean_season_col(df: pd.DataFrame) -> pd.DataFrame:
    """Function that cleans the season column in the dataframe

    Args:
        df (pd.DataFrame): dataframe to clean the season column for

    Returns:
        pd.DataFrame: dataframe with the season column cleaned
    """
    df.loc[:, "season"] = df["season"].str[:4]
    df.loc[:, "season"] = df["season"].astype(int)
    return df


def main() -> None:
    logger.info("Loading league wage and valuations data")
    leagues = ["premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1"]
    league_df = pd.concat([load_wages_values(league) for league in leagues])

    logger.info("Loading player stats data")
    stats = gcp.read_df_from_bucket(
        bucket_name="processed_fbref_db", blob_name="processed_standard.csv"
    )
    stats_df = clean_season_col(stats)

    logger.info(
        "Joining league wage and valuations data with standard player stats data"
    )
    # join the two datasets
    joined_all = pd.merge(
        league_df,
        stats_df,
        how="inner",
        left_on=["player", "season", "squad"],
        right_on=["player", "season", "squad"],
        suffixes=("", "_stats"),
    )

    logger.info("Removing duplicated columns")
    cols_to_keep = [col for col in joined_all.columns if "_stats" not in col]
    joined_df = joined_all[cols_to_keep]

    print(joined_df)

    logger.info("Writing joined data to bucket")
    gcp.write_df_to_bucket(
        data=joined_df,
        bucket_name="wage_vals_stats",
        blob_name="standard.csv",
    )

    logger.info("Process complete")


if __name__ == "__main__":
    main()
