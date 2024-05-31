import pandas as pd

from processing.abcs.processor import Processor
from processing.gcp.files import gcs


class Process(Processor):
    """Change team names in valuations dataframe to match wages dataframe"""

    def __init__(self, league: str) -> None:
        super().__init__(None)
        self.league = league

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "premier" in self.league.lower():
            wage_df = gcs.read_csv(
                "processed_fbref_db/processed_Premier-League-wages.csv"
            )
        elif "la" in self.league.lower():
            wage_df = gcs.read_csv("processed_fbref_db/processed_La-Liga-wages.csv")
        elif "serie" in self.league.lower():
            wage_df = gcs.read_csv("processed_fbref_db/processed_Serie-A-wages.csv")
        elif "bundesliga" in self.league.lower():
            wage_df = gcs.read_csv("processed_fbref_db/processed_Bundesliga-wages.csv")
        elif "ligue" in self.league.lower():
            wage_df = gcs.read_csv("processed_fbref_db/processed_Ligue-1-wages.csv")

        wage_teams = wage_df["squad"].unique()
        wage_teams.sort()

        val_teams = df["squad"].unique()
        val_teams.sort()

        team_map = dict(zip(val_teams, wage_teams))

        df.loc[:, "squad"] = df["squad"].map(team_map)
        return df
