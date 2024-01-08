from __future__ import annotations
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from feature_pipeline.utilities.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TransfermarktUrls:
    """Class for creating urls for scraping data from transfermarkt.com."""

    base: str = "https://www.transfermarkt.co.uk"

    def player_values(self, team_name: str, team_id: str, season: str) -> str:
        """Create url for scraping player market values from transfermarkt.com.

        Args:
            team_name (str): team name
            team_id (str): transfermarkt team id
            season (str): football season

        Returns:
            str: url for transfermarkt.com
        """
        return f"{self.base}/{team_name}/kader/verein/{team_id}/plus/1/galerie/0?saison_id={season}"

    def team_info(self, competition_id: str, season: str) -> str:
        """Create url for scraping team information from transfermarkt.com.

        Args:
            competition_id (str): competition id
            season (str): season

        Returns:
            str: url for transfermarkt.com
        """
        return f"{self.base}/-/startseite/wettbewerb/{competition_id}/plus/?saison_id={season}"


def get_team_name_ids(season: str, league: str) -> zip[tuple[str, str]]:
    """Get team names and ids for a particular league and season.

    Args:
        season (str): football season
        league (str): league name

    Returns:
        zip[tuple[str, str]]: zip of team names and ids
    """
    file_path = Path.cwd() / "feature_pipeline" / "data" / "transfermarkt" / "leagues"
    df = pd.read_pickle(file_path / f"{league}_info.pkl")
    dff = df.loc[df["season"] == season]
    return zip(dff['other_names'], dff['team_id'])


transfermarkt_urls = TransfermarktUrls()

# Transfermakt competition and season ids
competition_ids = ["GB1", "IT1", "ES1", "L1", "FR1"]
competition_names = ["premier_league", "serie_a", "la_liga", "bundesliga", "ligue_1"]
seasons = ["2017", "2018", "2019", "2020", "2021", "2022", "2023"]
