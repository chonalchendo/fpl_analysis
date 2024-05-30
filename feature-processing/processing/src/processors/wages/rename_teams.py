import pandas as pd

from processing.abcs.processor import Processor


class RenameTeams(Processor):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        league = df["league"].values[0]

        if league == "la_liga":
            replacements = {
                "Betis": "Real Betis",
                "Valladolid": "Real Valladolid",
                "Málaga": "Malaga",
                "Cádiz": "Cadiz",
            }
            df.loc[:, "squad"] = df["squad"].map(replacements).fillna(df["squad"])

        if league == "bundesliga":
            df.loc[:, "squad"] = df["squad"].replace("M'Gladbach", "Monchengladbach")

        return df
