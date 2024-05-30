import pandas as pd

from processing.abcs.processor import Processor
from processing.src.processors.fbref import helpers


class Process(Processor):
    def __init__(self):
        super().__init__(None)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["continent"] = df["country"].apply(helpers.get_continent)
        return df
