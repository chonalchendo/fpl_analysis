from typing import Literal

from rich import print

from processing.gcp.loader import CSVLoader
from processing.gcp.saver import GCPSaver
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.transfermarkt import (
    filter_teams,
    foreign_pct,
    player_id,
    rename_teams,
    signed_year,
    team_season,
)
from processing.src.processors.utils import cleaners


def run_players(
    blob: str, output_blob: str | None = None, save: Literal["yes", "no"] = "no"
) -> None:
    if save == "yes":
        saver = GCPSaver()
    else:
        saver = None

    dp = DataProcessor(
        processors=[
            signed_year.Process(),
            cleaners.Imputer(features="height"),
            player_id.Process(),
            filter_teams.Process(),
            rename_teams.Process(),
            cleaners.Drop(
                features=[
                    "tm_id",
                    "tm_name",
                    "squad_num",
                    "contract_expiry",
                    "current_club",
                    "signed_date",
                ]
            ),
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


def run_teams(
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
