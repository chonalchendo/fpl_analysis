import pandas as pd
from rich import print

from processing.abcs.processor import Processor
from processing.gcp.files import gcs


class Process(Processor):
    """Change team names in valuations dataframe to match wages dataframe"""

    def __init__(self, league: str) -> None:
        super().__init__(None)
        self.league = league

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "premier" in self.league.lower():
            path = "processed_fbref_db/processed_Premier-League-wages.csv"
        elif "la" in self.league.lower():
            path = "processed_fbref_db/processed_La-Liga-wages.csv"
        elif "serie" in self.league.lower():
            path = "processed_fbref_db/processed_Serie-A-wages.csv"
        elif "bundesliga" in self.league.lower():
            path = "processed_fbref_db/processed_Bundesliga-wages.csv"
        elif "ligue" in self.league.lower():
            path = "processed_fbref_db/processed_Ligue-1-wages.csv"

        wage_df = gcs.read_csv(path)

        wage_teams = wage_df["squad"].unique().tolist()
        wage_teams = sorted(wage_teams, key=str.lower)

        val_teams = df["squad"].unique().tolist()
        val_teams = sorted(val_teams, key=str.lower)

        team_map = dict(zip(val_teams, wage_teams))

        print(team_map)

        df.loc[:, "squad"] = df["squad"].map(team_map)
        return df
