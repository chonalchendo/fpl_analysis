from processing.gcp.loader import CSVLoader
from processing.gcp.saver import GCPSaver
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.utils import cleaners
from processing.src.processors.valuations import (
    filter_teams,
    map_team_names,
    rename_teams,
)
from processing.src.processors.wages import redefine_season, rename_teams


def main() -> None:
    # Define the processor for the wages data
    wages_processor = DataProcessor(
        processors=[
            rename_teams.Process(),
            redefine_season.Process(),
            cleaners.Drop(
                features=["nation", "pos", "notes", "rk", "general_pos", "country"]
            ),
        ],
        loader=CSVLoader(),
    )

    # Define the processor for the valuations data
    valuations_processor = DataProcessor(
        processors=[
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
    )

    # Process the wages data
    wages_processor.process(
        bucket="wages_db",
        blob="wages.csv",
        output_bucket="processed_wages_db",
        output_blob="processed_wages.csv",
    )

    # Process the valuations data
    valuations_processor.process(
        bucket="valuations_db",
        blob="valuations.csv",
        output_bucket="processed_valuations_db",
        output_blob="processed_valuations.csv",
    )


if __name__ == "__main__":
    main()
