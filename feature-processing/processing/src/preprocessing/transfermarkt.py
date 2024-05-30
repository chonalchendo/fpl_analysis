from rich import print

from processing.gcp.loader import GCPLoader
from processing.src.pipeline.data import DataProcessor
from processing.src.processors.transfermarkt.signed_year import SignedYear


def main() -> None:
    df = DataProcessor(
        processors=[
            SignedYear(),
        ],
        loader=GCPLoader(),
    ).process(bucket="transfermarkt_db", blob="premier_league_player_valuations.csv")

    print(df.head())


if __name__ == "__main__":
    main()
