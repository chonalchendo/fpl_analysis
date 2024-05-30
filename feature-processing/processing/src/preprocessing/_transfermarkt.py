import pandas as pd
from rich import print

from processing.gcp.loader import GCPLoader
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.transfermarkt import (
    foreign_pct,
    player_id,
    signed_year,
    team_season,
)
from processing.src.processors.utils.cleaners import Imputer


def clean_player_df(blob: str) -> pd.DataFrame:
    dp = DataProcessor(
        processors=[
            signed_year.Process(),
            Imputer(features="height"),
            player_id.Process(),
        ],
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
    df = clean_player_df(blob="premier_league_player_valuations.csv")
    print(df)


if __name__ == "__main__":
    main()
