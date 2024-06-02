import pandas as pd

from processing.abcs.processor import Processor


class Process(Processor):
    def __init__(self) -> None:
        super().__init__(None)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean team names in valuations dataframe to match wages dataframe"""
        league = df["league"].values[0]

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

        df.loc[:, "squad"] = df["squad"].str.replace(pattern, "", regex=True)
        return df
