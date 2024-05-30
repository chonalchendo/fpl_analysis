import pandas as pd
from rich import print

from processing.gcp.loader import GCPLoader
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.transfermarkt import (
    foreign_pct,
    signed_year,
    team_season,
)
from processing.src.processors.utils.cleaners import Imputer


def clean_player_df(blob: str) -> pd.DataFrame:
    dp = DataProcessor(
        processors=[signed_year.Process(), Imputer(features="height")],
        loader=GCPLoader(),
    )

    return dp.process(bucket="transfermarkt_db", blob=blob)


def clean_team_df(blob: str) -> pd.DataFrame:
    dp = DataProcessor(
        processors=[team_season.Process(), foreign_pct.Process()],
        loader=GCPLoader(),
    )

    return dp.process(bucket="transfermarkt_db", blob=blob)


def main() -> None:
    df = clean_team_df(blob="premier_league_team_data.csv")
    print(df)


if __name__ == "__main__":
    main()
