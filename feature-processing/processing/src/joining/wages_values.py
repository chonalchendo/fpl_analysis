import pandas as pd
from dataclasses import dataclass
from rich import print

from processing.gcp.storage import gcp
from processing.utilities.utils import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Class to load wages and valuations from GCP bucket"""

    @staticmethod
    def load_wages(league: str) -> pd.DataFrame:
        """Load wages from GCP bucket

        Args:
            league (str): specify the league to load wages for

        Returns:
            pd.DataFrame: wages for the specified league
        """
        return gcp.read_df_from_bucket(
            bucket_name="processed_fbref_db", blob_name=f"processed_{league}-wages.csv"
        )

    @staticmethod
    def load_valuations(league: str) -> pd.DataFrame:
        """Load player valuations from GCP bucket

        Args:
            league (str): specify the league to load valuations for

        Returns:
            pd.DataFrame: player valuations for the specified league
        """
        return gcp.read_df_from_bucket(
            bucket_name="processed_transfermarkt_db",
            blob_name=f"processed_{league}_player_valuations.csv",
        )


@dataclass
class DataJoiner:
    """Class to join wages and valuations dataframes"""

    _wages_df: pd.DataFrame
    _valuations_df: pd.DataFrame
    _joined_df: pd.DataFrame = None
    _team_map: dict[str, str] = None

    def rename_wage_teams(self) -> None:
        """Rename teams in wages dataframe to match valuations dataframe"""
        league = self._valuations_df["league"].values[0]

        if league == "la_liga":
            replacements = {
                "Betis": "Real Betis",
                "Valladolid": "Real Valladolid",
                "Málaga": "Malaga",
                "Cádiz": "Cadiz",
            }
            self._wages_df.loc[:, "squad"] = (
                self._wages_df["squad"]
                .map(replacements)
                .fillna(self._wages_df["squad"])
            )

        if league == "bundesliga":
            self._wages_df.loc[:, "squad"] = self._wages_df["squad"].replace(
                "M'Gladbach", "Monchengladbach"
            )

    def filter_values_bundesliga(self) -> None:
        """Filter out Hannover 96 from Bundesliga valuations as it does not appear in
        wages data.
        """
        league = self._valuations_df["league"].values[0]

        if league == "bundesliga":
            self._valuations_df = self._valuations_df.loc[
                self._valuations_df["team"] != "hannover-96"
            ]

    def clean_values_team_col(self) -> None:
        """Clean team names in valuations dataframe to match wages dataframe"""
        league = self._valuations_df["league"].values[0]

        # map different regex patterns
        if league == "premier_league":
            pattern = "^(a?fc)-"
        elif league == "la_liga":
            pattern = "^(fc|sd|rcd|ca|ud|cd|deportivo)-"
        elif league == "bundesliga":
            pattern = "^(1-fc|fc|1-fsv|sv|vfb|sc|vfl|spvgg|tsg-1899|borussia|bayer-04|fortuna)-"
        elif league == "serie_a":
            pattern = "^(ac|as|fc|ssc|us)-"
        else:
            pattern = "^(fc-stade|stade|as|ogc|es|aj|ac|sm|ea|rc|fc-girondins|fc|sco|olympique|losc)-"

        self._valuations_df.loc[:, "team"] = self._valuations_df["team"].str.replace(
            pattern, "", regex=True
        )

    def create_team_map(self) -> None:
        """Create a mapping between team names in wages and valuations dataframes"""
        wage_teams = self._wages_df["squad"].unique()
        value_teams = self._valuations_df["team"].unique()

        wage_teams.sort()
        value_teams.sort()

        self._team_map = dict(zip(value_teams, wage_teams))

    def change_values_team_names(self) -> None:
        """Change team names in valuations dataframe to match wages dataframe"""
        self._valuations_df.loc[:, "team"] = self._valuations_df["team"].map(
            self._team_map
        )

    def redefine_wage_season(self) -> None:
        """Redefine season column in wages dataframe to match valuations dataframe"""
        self._wages_df.loc[:, "season"] = (
            self._wages_df["season"].apply(lambda x: x.split("-")[0]).astype(int)
        )

    def drop_wage_columns(self) -> None:
        """Drop unnecessary columns from wages dataframe"""
        wage_cols_to_drop = ["nation", "pos", "notes", "rk", "general_pos", "country"]
        self._wages_df = self._wages_df.drop(columns=wage_cols_to_drop)

    def drop_value_columns(self) -> None:
        """Drop unnecessary columns from valuations dataframe"""
        val_cols_to_drop = [
            "tm_id",
            "tm_name",
            "squad_num",
            "contract_expiry",
            "current_club",
            "signed_date",
        ]
        self._valuations_df = self._valuations_df.drop(columns=val_cols_to_drop)

    def join_wages_values(self) -> None:
        """Join wages and valuations dataframes on player, season and squad"""
        self._joined_df = pd.merge(
            left=self._wages_df,
            right=self._valuations_df,
            how="inner",
            left_on=["player", "season", "squad"],
            right_on=["player", "season", "team"],
            suffixes=("_wage_df", "_value_df"),
        )

    def sort_joined_columns(self) -> None:
        """Sort columns in joined dataframe"""
        # remove duplicate cols
        cols = [col for col in self._joined_df.columns if "_value_df" not in col]
        self._joined_df = self._joined_df[cols]

        # rename columns
        self._joined_df.columns = [
            col.replace("_wage_df", "") for col in self._joined_df.columns
        ]

        # drop team column
        self._joined_df = self._joined_df.drop(columns=["team"])

    def run_wages(self) -> None:
        """Run wages data cleaning steps"""
        self.rename_wage_teams()
        self.redefine_wage_season()
        self.drop_wage_columns()

    def run_values(self) -> None:
        """Run valuations data cleaning steps"""
        self.filter_values_bundesliga()
        self.clean_values_team_col()
        self.drop_value_columns()

    def map_teams(self) -> None:
        """map team names between wages and valuations dataframes for joining"""
        self.create_team_map()
        self.change_values_team_names()

    def run(self) -> pd.DataFrame:
        """Run all data cleaning steps and join wages and valuations dataframes

        Returns:
            pd.DataFrame: joined dataframe
        """
        self.run_wages()
        self.run_values()
        self.map_teams()
        self.join_wages_values()
        self.sort_joined_columns()
        return self._joined_df


def main() -> None:
    wage_leagues = ["Premier-League", "Bundesliga", "La-Liga", "Serie-A", "Ligue-1"]
    value_leagues = ["premier_league", "bundesliga", "la_liga", "serie_a", "ligue_1"]

    for wage_league, value_league in zip(wage_leagues, value_leagues):

        logger.info(f"Loading {wage_league} wages and valuations")
        wages = DataLoader.load_wages(wage_league)
        valuations = DataLoader.load_valuations(value_league)

        logger.info(f"Joining {wage_league} wages and valuations")
        joined_df = DataJoiner(wages, valuations).run()

        logger.info(f"Writing joined {wage_league} dataframe to GCP bucket")
        gcp.write_df_to_bucket(
            bucket_name="joined_wages_values",
            blob_name=f"{value_league}_wages_values.csv",
            data=joined_df
        )
        
        print(joined_df)

    logger.info("Joining complete")


if __name__ == "__main__":
    main()
