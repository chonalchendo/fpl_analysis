from rich import print

from processing.gcp.loader import GCPLoader
from processing.gcp.saver import GCPSaver
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.fbref import age_range, continent, country, general_pos


def main() -> None:
    dp = DataProcessor(
        processors=[
            general_pos.Process(),
            age_range.Process(),
            country.Process(),
            continent.Process(),
        ],
        loader=GCPLoader(),
        saver=GCPSaver(),
    )
    df = dp.process(bucket="fbref_db", blob="shooting.csv")
    print(df)


if __name__ == "__main__":
    main()
