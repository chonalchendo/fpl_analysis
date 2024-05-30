import pandas as pd
from rich import print

from processing.gcp.loader import GCPLoader
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.transfermarkt.foreign_pct import ForeignPct
from processing.src.processors.transfermarkt.signed_year import SignedYear
from processing.src.processors.transfermarkt.team_season import TeamSeason
from processing.src.processors.utils.cleaners import ColumnImputer


def clean_player_df(blob: str) -> pd.DataFrame:
    dp = DataProcessor(
        processors=[SignedYear(), ColumnImputer(features="height")],
        loader=GCPLoader(),
    )

    return dp.process(bucket="transfermarkt_db", blob=blob)


def clean_team_df(blob: str) -> pd.DataFrame:
    dp = DataProcessor(
        processors=[TeamSeason(), ForeignPct()],
        loader=GCPLoader(),
    )

    return dp.process(bucket="transfermarkt_db", blob=blob)


def main() -> None:
    df = clean_team_df(blob="premier_league_team_data.csv")
    print(df)


if __name__ == "__main__":
    main()
