from typing import Literal

from rich import print

from processing.abcs.processor import Processor
from processing.gcp.buckets import Buckets
from processing.gcp.loader import CSVLoader
from processing.gcp.saver import GCPSaver
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.fbref import (
    age_range,
    continent,
    country,
    general_pos,
    redefine_season,
    rename_teams,
)
from processing.src.processors.utils import cleaners


def run_stats(blob: str, output_blob: str, save: Literal["yes", "no"] = "no") -> None:
    league = blob.split("-")[0]
    dp = _base(add_processors=[rename_teams.Process(league=league)], save=save)
    df = dp.process(
        bucket=Buckets.FBREF,
        blob=blob,
        output_bucket=Buckets.PROCESSED_FBREF,
        output_blob=output_blob,
    )
    print(df)


def run_wages(blob: str, output_blob: str, save: Literal["yes", "no"] = "no") -> None:
    league = blob.split("-")[0]
    dp = _base(
        add_processors=[
            rename_teams.Process(league=league),
            cleaners.Drop(
                features=["nation", "pos", "notes", "rk", "general_pos", "country"]
            ),
        ],
        save=save,
    )
    df = dp.process(
        bucket=Buckets.FBREF,
        blob=blob,
        output_bucket=Buckets.PROCESSED_FBREF,
        output_blob=output_blob,
    )
    print(df)


def _base(
    add_processors: list[Processor] | None = None,
    save: Literal["yes", "no"] = "no",
) -> DataProcessor:
    if save == "yes":
        saver = GCPSaver()
    else:
        saver = None

    processors = [
        cleaners.Rename(features={"position": "pos"}),
        general_pos.Process(),
        age_range.Process(),
        country.Process(),
        continent.Process(),
        redefine_season.Process(),
    ]
    if add_processors:
        processors.extend(add_processors)

    return DataProcessor(
        processors=processors,
        loader=CSVLoader(),
        saver=saver,
    )
