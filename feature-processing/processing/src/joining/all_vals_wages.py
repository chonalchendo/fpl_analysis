from rich import print

from processing.gcp.buckets import Buckets
from processing.gcp.loader import CSVLoader
from processing.gcp.saver import save
from processing.src.pipeline.join import Bucket
from processing.src.processors.utils.joiners import Concat


def main() -> None:
    saver = save("yes")
    bucket = Buckets.JOINED_WAGES_VALUES

    joiner = Bucket(join_method=Concat(), loader=CSVLoader(), saver=saver)
    df = joiner.process(bucket=bucket, output_blob="top_5_league_values_wages.csv")

    print(df)


if __name__ == "__main__":
    main()
