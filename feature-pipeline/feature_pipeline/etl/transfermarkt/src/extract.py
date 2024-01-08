import pandas as pd
from pathlib import Path
from icecream import ic
from feature_pipeline.utilities.utils import get_logger
from feature_pipeline.apis.transfermarkt import (
    transfermarkt_urls,
    seasons,
    get_team_name_ids,
)
from feature_pipeline.etl.transfermarkt.src import parser


logger = get_logger(__name__)


def league_info_from_website(comp_id: str) -> pd.DataFrame:
    """Extract league information from transfermarkt.com for a particular competition.

    Args:
        comp_id (str): competition id

    Returns:
        pd.DataFrame: pandas dataframe of extracted league information
    """
    dfs = []
    for season in seasons:
        logger.info(f"Scraping {comp_id} league information for season: {season}")
        url = transfermarkt_urls.team_info(competition_id=comp_id, season=season)
        df = parser.team_information(url, season=season, comp_id=comp_id)
        ic(df)
        dfs.append(df)
    logger.info(f"Concatenating dataframes for {comp_id} league information")
    return pd.concat(dfs)


def players_vals_from_website(league: str, season: str) -> pd.DataFrame:
    """Extract player market values from transfermarkt.com for a particular league and season.

    Args:
        league (str): league name
        season (str): football season

    Returns:
        pd.DataFrame: pandas dataframe of extracted player market values
    """
    dfs = []
    teams = get_team_name_ids(season=season, league=league)
    logger.info(f"Scraping {league} player market values for season: {season}")
    for team, team_id in teams:
        logger.info(f"Scraping {team} player market values")

        while True:
            url = transfermarkt_urls.player_values(
                team_name=team, team_id=team_id, season=season
            )
            df = parser.player_market_values(
                url, season=season, league=league, team=team
            )
            ic(df)

            if not df.empty:
                break
            else:
                logger.info(f"Retrying {team} player market values")

        logger.info(f"Finished scraping {team} player market values")
        dfs.append(df)

    logger.info(
        f"Concatenating dataframes for {league} ({season}) player market values"
    )
    return pd.concat(dfs)


def player_valuations_total(league: str) -> pd.DataFrame:
    """Function that extracts player valuations for all seasons for a particular league.

    Args:
        league (str): league name

    Returns:
        pd.DataFrame: pandas dataframe of extracted player market values
    """
    dfs = [
        players_vals_from_website(league=league, season=season) for season in seasons
    ]
    logger.info(
        f"Concatenating dataframes for {league} player market values for seasons: {seasons}"
    )
    return pd.concat(dfs)


# -- Main extract functions for league information and player market values -- #


def leagues_data(comp_id: int, comp_name: str) -> pd.DataFrame:
    """Main extract function for league information which either extracts data from website
    or from file if it already exists.

    Args:
        comp_id (int): competition id
        comp_name (str): competition name

    Returns:
        pd.DataFrame: pandas dataframe of extracted data
    """
    file_dir = Path.cwd() / "feature_pipeline" / "data" / "transfermarkt" / "leagues"
    path = file_dir / f"{comp_name}_info.pkl"

    if not file_dir.exists():
        logger.info(f"Creating new directory: {file_dir}")
        file_dir.mkdir(parents=True)

    logger.info(f"Checking if {path} exists")

    if not path.exists():
        logger.info(f"\n{path} does not exist. \nExtracting data for {comp_name}")
        df = league_info_from_website(comp_id=comp_id)
        df.to_pickle(path)
    else:
        logger.info(
            f"\n{path} exists. \nData already extracted from transfermarkt for {comp_name} \n Reading data from {path}"
        )
        df = pd.read_pickle(path)

    return df


def player_valuations(league: str) -> pd.DataFrame:
    """Main extract function for player market values which either extracts data from website
    or from file if it already exists.

    Args:
        league (str): league name

    Returns:
        pd.DataFrame: pandas dataframe of extracted player market values
    """
    file_dir = Path.cwd() / "feature_pipeline" / "data" / "transfermarkt" / "valuations"
    path = file_dir / f"{league}_player_vals.pkl"

    if not file_dir.exists():
        logger.info(f"Creating new directory: {file_dir}")
        file_dir.mkdir(parents=True)

    logger.info(f"Checking if {path} exists")

    if not path.exists():
        logger.info(
            f"\n{path} does not exist. \nExtracting player valuations data for {league}"
        )
        df = player_valuations_total(league=league)
        df.to_pickle(path)
    else:
        logger.info(
            f"\n{path} exists. \nData already extracted from transfermarkt for {league} \n Reading data from {path}"
        )
        df = pd.read_pickle(path)

    return df
