from processing.gcp.loader import CSVLoader
from processing.gcp.saver import GCPSaver
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.fbref import age_range, continent, country, general_pos


def run(blob: str, output_blob: str) -> None:
    dp = DataProcessor(
        processors=[
            general_pos.Process(),
            age_range.Process(),
            country.Process(),
            continent.Process(),
        ],
        loader=CSVLoader(),
        saver=GCPSaver(),
    )
    dp.process(
        bucket="fbref_db",
        blob=blob,
        output_bucket="processed_fbref_db",
        output_blob=output_blob,
    )
