from rich import print

from processing.gcp.buckets import Buckets
from processing.gcp.loader import CSVLoader
from processing.gcp.saver import save
from processing.src.pipeline.join import Merge
from processing.src.processors.utils.cleaners import Filter
from processing.src.processors.utils.joiners import MultiJoin


def main() -> None:
    w_v_bucket = Buckets.JOINED_WAGES_VALUES
    left_path = f"{w_v_bucket}/top_5_league_values_wages.csv"

    stat_bucket = Buckets.PROCESSED_FBREF
    right_path = f"{stat_bucket}/processed_standard.csv"

    output_bucket = Buckets.WAGE_VALS_STATS
    output_path = f"{output_bucket}/standard.csv"

    joiner = Merge(
        join_method=MultiJoin(
            on=["player", "season", "squad"], how="inner", suffixes=("", "_stats")
        ),
        loader=CSVLoader(),
        saver=save("yes"),
        processors=[Filter(not_like="_stats")],
    )

    df = joiner.process(
        left_path=left_path, right_path=right_path, output_path=output_path
    )
    print(df)


if __name__ == "__main__":
    main()
