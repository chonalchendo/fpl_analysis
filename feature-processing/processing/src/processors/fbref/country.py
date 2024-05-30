import pandas as pd

from processing.abcs.processor import Processor
from processing.src.processors.fbref import helpers


class Process(Processor):
    def __init__(self):
        super().__init__(None)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        codes = helpers.get_fifa_codes()
        df["country"] = df["nation"].map(codes)
        return df
