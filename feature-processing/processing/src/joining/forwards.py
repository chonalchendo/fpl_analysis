from rich import print

from processing.gcp.blobs import Blobs
from processing.gcp.buckets import Buckets
from processing.gcp.loader import CSVLoader
from processing.gcp.saver import save
from processing.src.pipeline.join import Bucket, Merge
from processing.src.processors.utils import cleaners
from processing.src.processors.utils.joiners import MultiJoin


def main() -> None:
    stats_joiner = Bucket(
        processors=[cleaners.Filter(not_like="_x")],
        join_method=MultiJoin(
            on=["player", "season", "squad"], how="inner", suffixes=("", "_x")
        ),
        loader=CSVLoader(),
        saver=save("no"),
    )

    stats_df = stats_joiner.process(
        bucket=Buckets.PROCESSED_FBREF, include_blobs=Blobs.FORWARDS
    )

    main_joiner = Merge(
        join_method=MultiJoin(
            on=["player", "season", "squad"], how="inner", suffixes=("", "_x")
        ),
        loader=CSVLoader(),
        saver=save("yes"),
    )

    joined_df = main_joiner.process(
        left_path=f"{Buckets.JOINED_WAGES_VALUES}/top_5_league_values_wages.csv",
        right_df=stats_df,
        output_path=f"{Buckets.WAGE_VALS_STATS}/forwards.csv",
    )

    print(joined_df)


if __name__ == "__main__":
    main()
