from typing import Literal

from rich import print

from processing.gcp.buckets import Buckets
from processing.gcp.loader import CSVLoader
from processing.gcp.saver import GCPSaver
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.transfermarkt import (
    filter_teams,
    foreign_pct,
    map_team_names,
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

    league = blob.split("_")[0]

    dp = DataProcessor(
        processors=[
            signed_year.Process(),
            cleaners.Imputer(features="height"),
            player_id.Process(),
            cleaners.Rename(features={"team": "squad"}),
            filter_teams.Process(),
            rename_teams.Process(),
            map_team_names.Process(league=league),
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
        bucket=Buckets.TRANSFERMARKT,
        blob=blob,
        output_bucket=Buckets.PROCESSED_TRANSFERMARKT,
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
        processors=[
            team_season.Process(),
            foreign_pct.Process(),
            cleaners.Rename(features={"team": "squad"}),
        ],
        loader=CSVLoader(),
        saver=saver,
    )

    df = dp.process(
        bucket=Buckets.TRANSFERMARKT,
        blob=blob,
        output_bucket=Buckets.PROCESSED_TRANSFERMARKT,
        output_blob=output_blob,
    )
    print(df)
