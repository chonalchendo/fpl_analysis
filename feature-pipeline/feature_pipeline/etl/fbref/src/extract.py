import pandas as pd
from pathlib import Path
from feature_pipeline.apis.fbref import url_handler, get_data, seasons
from feature_pipeline.utilities.utils import get_logger


logger = get_logger(__name__)


def stats_from_website(stat: str, table: str) -> pd.DataFrame:
    """Extract data from fbref.com for a particular set of statistics including
    standard, keeper, advanced keeper, shooting, passing, passing_types, goal
    and shot creation, defense, possession, playing time and miscellanous.

    Data is extracted for the following seasons: 2017-2018, 2018-2019, 2019-2020,
    2020-2021, 2021-2022, 2022-2023, 2023-2024.

    Args:
        stat (str): stat to extract
        table (str): table to extract

    Returns:
        pd.DataFrame: pandas dataframe of extracted data
    """
    data = []
    logger.info(f"Extracting data from page: {stat} and table: stat_{table}")
    for season in seasons:
        logger.info(f"Extracting data for season: {season}")
        url = url_handler.stats(season=season, stat=stat, table=table)
        df = get_data(url)
        df["season"] = season
        data.append(df)
    logger.info(f"Data extracted for {stat}")
    return pd.concat(data)


def wages_from_website(comp_id: int, comp_name: str) -> pd.DataFrame:
    """Extract wage data from fbref.com for a particular competition.

    Args:
        comp_id (int): competition id
        comp_name (str): competition name

    Returns:
        pd.DataFrame: pandas dataframe of extracted wage data
    """
    data = []
    logger.info(f"Extracting wage data for competition: {comp_name}")
    for season in seasons:
        logger.info(f"Extracting wage data for season: {season}")
        url = url_handler.wages(comp_id=comp_id, comp_name=comp_name, season=season)
        df = get_data(url)
        df["season"] = season
        data.append(df)
    logger.info(f"Wage data extracted for {comp_name}")

    return pd.concat(data)


# -- Main extract functions for player stats and wages - #


def player_stats(stat: str, table: str) -> pd.DataFrame:
    """Main extract function for stats which either extracts data from website
    or from file if it already exists.

    Args:
        stat (str): stat to extract
        table (str): table to extract

    Returns:
        pd.DataFrame: pandas dataframe of extracted data
    """
    file_dir = Path.cwd() / "feature_pipeline" / "data" / "fbref" / "stats"
    path = file_dir / f"{stat}_{table}.pkl"

    if not file_dir.exists():
        logger.info(f"Creating new directory: {file_dir}")
        file_dir.mkdir()

    logger.info(f"Checking if {path} exists")

    if not path.exists():
        logger.info(
            f"\n{path} does not exist. \nExtracting data for {stat} and {table}"
        )
        df = stats_from_website(stat=stat, table=table)
        df.to_pickle(path)
    else:
        logger.info(
            f"\n{path} exists. \nData already extracted from fbref for {stat} and {table} \n Reading data from {path}"
        )
        df = pd.read_pickle(path)

    return df


def player_wages(comp_id: int, comp_name: str) -> pd.DataFrame:
    """Main extract function for wages which either extracts data from website
    or from file if it already exists.

    Args:
        comp_id (int): competition id
        comp_name (str): competition name

    Returns:
        pd.DataFrame: pandas dataframe of extracted data
    """
    file_dir = Path.cwd() / "feature_pipeline" / "data" / "fbref" / "wages"
    path = file_dir / f"{comp_name}_player_wages.pkl"

    if not file_dir.exists():
        logger.info(f"Creating new directory: {file_dir}")
        file_dir.mkdir()

    logger.info(f"Checking if {path} exists")

    if not path.exists():
        logger.info(f"\n{path} does not exist. \nExtracting data for {comp_name} wages")
        df = wages_from_website(comp_id=comp_id, comp_name=comp_name)
        df.to_pickle(path)
    else:
        logger.info(
            f"\n{path} exists. \nData already extracted from fbref for {comp_name} wages \n Reading data from {path}"
        )
        df = pd.read_pickle(path)

    return df


# if __name__ == "__main__":
#     wages = 'https://fbref.com/en/comps/20/2022-2023/wages/2022-2023-Bundesliga-Wages#player_wages'
#     df = pd.read_html(wages)[1]
#     # df = scraper(wages)
#     print(df)
