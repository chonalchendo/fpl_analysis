from typing import Literal

from rich import print

from processing.gcp.loader import CSVLoader
from processing.gcp.saver import GCPSaver
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.fbref import age_range, continent, country, general_pos
from processing.src.processors.utils.cleaners import Rename


def run(blob: str, output_blob: str, save: Literal["yes", "no"]) -> None:
    if save == "yes":
        saver = GCPSaver()
    else:
        saver = None

    dp = DataProcessor(
        processors=[
            Rename(features={"position": "pos"}),
            general_pos.Process(),
            age_range.Process(),
            country.Process(),
            continent.Process(),
        ],
        loader=CSVLoader(),
        saver=saver,
    )
    df = dp.process(
        bucket="fbref_db",
        blob=blob,
        output_bucket="processed_fbref_db",
        output_blob=output_blob,
    )

    print(df)
