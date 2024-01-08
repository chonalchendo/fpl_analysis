import pandas as pd
from io import StringIO
from feature_pipeline.utilities.utils import get_logger
from feature_pipeline.scraper.response import get_url_data

logger = get_logger(__name__)


stats: list[str] = [
    "stats",
    "keepers",
    "keepersadv",
    "shooting",
    "passing",
    "passing_types",
    "gca",
    "defense",
    "possession",
    "playingtime",
    "misc",
]

tables: list[str] = [
    "standard",
    "keeper",
    "keeper_adv",
    "shooting",
    "passing",
    "passing_types",
    "gca",
    "defense",
    "possession",
    "playing_time",
    "misc",
]

seasons: list[str] = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
]

comp_ids: list[int] = [9, 11, 12, 20, 13]

comp_names: list[str] = [
    "Premier-League",
    "Serie-A",
    "La-Liga",
    "Bundesliga",
    "Ligue-1",
]


class Urls:
    """Class to handle url creation for fbref.com data"""

    @staticmethod
    def stats(season: str, stat: str, table: str) -> str:
        """Function to handle stats url creation for fbref.com data from the
        top 5 leagues.

        Args:
            season (str): football season
            table (str): table to scrape

        Returns:
            str: url for fbref.com
        """
        return f"https://fbref.com/en/comps/Big5/{season}/{stat}/players/{season}-Big-5-European-Leagues-Stats#stats_{table}"

    @staticmethod
    def wages(comp_id: int, comp_name: str, season: str) -> str:
        """Function to handle wages url creation for fbref.com data from any
        league that has wages data.

        Args:
            comp_id (int): competition id
            comp_name (str): competition name
            season (str): football season

        Returns:
            str: url for fbref.com
        """
        return f"https://fbref.com/en/comps/{comp_id}/{season}/wages/{season}-{comp_name}-Wages#player_wages"


def get_data(url: str) -> pd.DataFrame:
    """Scraper for fbref.com data

    Args:
        url (str): fbref url

    Returns:
        pd.DataFrame: pandas dataframe of scraped data
    """
    resp = get_url_data(url=url)

    html = StringIO(resp.text)

    if "wages" in url:
        return pd.read_html(html)[1]
    else:
        return pd.read_html(html, skiprows=1, header=0)[0]


url_handler = Urls()
