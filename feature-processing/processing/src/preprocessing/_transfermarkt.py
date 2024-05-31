from typing import Literal

from rich import print

from processing.gcp.loader import CSVLoader
from processing.gcp.saver import GCPSaver
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.transfermarkt import (
    foreign_pct,
    player_id,
    signed_year,
    team_season,
)
from processing.src.processors.utils.cleaners import Imputer


def clean_player_df(
    blob: str, output_blob: str | None = None, save: Literal["yes", "no"] = "no"
) -> None:
    if save == "yes":
        saver = GCPSaver()
    else:
        saver = None

    dp = DataProcessor(
        processors=[
            signed_year.Process(),
            Imputer(features="height"),
            player_id.Process(),
        ],
        loader=CSVLoader(),
        saver=saver,
    )

    df = dp.process(
        bucket="transfermarkt_db",
        blob=blob,
        output_bucket="processed_transfermarkt_db",
        output_blob=output_blob,
    )
    print(df)


def clean_team_df(
    blob: str, output_blob: str | None = None, save: Literal["yes", "no"] = "no"
) -> None:
    if save == "yes":
        saver = GCPSaver()
    else:
        saver = None

    dp = DataProcessor(
        processors=[team_season.Process(), foreign_pct.Process()],
        loader=CSVLoader(),
        saver=saver,
    )

    df = dp.process(
        bucket="transfermarkt_db",
        blob=blob,
        output_bucket="processed_transfermarkt_db",
        output_blob=output_blob,
    )
    print(df)
