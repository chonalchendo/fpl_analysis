import pandas as pd
from rich import print

from processing.abcs.processor import Processor


class Process(Processor):
    def __init__(self, league: str) -> None:
        super().__init__(None)
        self.league = league

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.league.lower() == "la":
            replacements = {
                "Betis": "Real Betis",
                "Valladolid": "Real Valladolid",
                "Málaga": "Malaga",
                "Cádiz": "Cadiz",
            }
            df.loc[:, "squad"] = df["squad"].map(replacements).fillna(df["squad"])

        if self.league.lower() == "bundesliga":
            df.loc[:, "squad"] = df["squad"].replace("M'Gladbach", "Monchengladbach")

        print(df["squad"].unique())
        return df
