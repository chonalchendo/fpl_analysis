import pandas as pd

from processing.abcs.processor import Processor
from processing.src.processors.transfermarkt import helpers


class Process(Processor):
    def __init__(self) -> None:
        super().__init__(None)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        id_map = helpers.generate_unique_id()
        df.loc[:, "player_id"] = df["player"].map(id_map).astype("Int64")
        return df
