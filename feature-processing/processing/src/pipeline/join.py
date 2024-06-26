from functools import reduce

import pandas as pd

from processing.abcs.loader import DataLoader
from processing.abcs.processor import Processor
from processing.abcs.saver import DataSaver
from processing.gcp.buckets import Buckets
from processing.gcp.files import gcs
from processing.utilities.logger import get_logger

logger = get_logger(__name__)


class Merge:
    def __init__(
        self,
        join_method: Processor,
        loader: DataLoader,
        saver: DataSaver | None = None,
        processors: list[Processor] | None = None,
    ) -> None:
        self.join_method = join_method
        self.loader = loader
        self.saver = saver
        self.processors = processors

    def process(
        self,
        left_path: str | None = None,
        right_path: str | None = None,
        left_df: pd.DataFrame | None = None,
        right_df: pd.DataFrame | None = None,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        logger.info(f"Joining {left_path} or {left_df} and {right_path} or {right_df}")
        if left_path is not None:
            left_bucket = left_path.split("/")[0]
            left_blob = left_path.split("/")[-1]
            left_data = self.loader.load(left_bucket, left_blob)

        if right_path is not None:
            right_bucket = right_path.split("/")[0]
            right_blob = right_path.split("/")[-1]
            right_data = self.loader.load(right_bucket, right_blob)

        if left_df is not None:
            left_data = left_df

        if right_df is not None:
            right_data = right_df

        if left_path is not None and left_df is not None:
            raise ValueError("Cannot provide both left_path and left_df")

        if right_path is not None and right_df is not None:
            raise ValueError("Cannot provide both right_path and right_df")

        joined_df = self.join_method.transform([left_data, right_data])

        if self.processors:
            logger.info(f"Applying {len(self.processors)} processors")
            joined_df = reduce(
                lambda df, processor: processor.transform(df),
                self.processors,
                joined_df,
            )

        if self.saver and output_path:
            logger.info(f"Saving joined data to {output_path}")
            output_bucket = output_path.split("/")[0]
            output_blob = output_path.split("/")[-1]
            self.saver.save(output_bucket, output_blob, joined_df)

        return joined_df


class Bucket:
    def __init__(
        self,
        join_method: Processor,
        loader: DataLoader,
        saver: DataSaver | None = None,
        processors: list[Processor] | None = None,
    ) -> None:
        self.processors = processors
        self.loader = loader
        self.saver = saver
        self.join_method = join_method

    def process(
        self,
        bucket: str,
        include_blobs: list[str] | None = None,
        exclude_blobs: list[str] | None = None,
        output_blob: str | None = None,
    ) -> pd.DataFrame:
        logger.info(f"Searching blobs in {bucket}")
        files = gcs.list_bucket(
            bucket=bucket, include=include_blobs, exclude=exclude_blobs
        )
        logger.info(f"Found {len(files)} files in {bucket}")

        if output_blob is not None:
            dfs = [
                self.loader.load(bucket, file)
                for file in files
                if output_blob not in file
            ]

        else:
            dfs = [self.loader.load(bucket, file) for file in files]

        logger.info(f"Joining {len(dfs)} dataframes")
        joined_df = self.join_method.transform(dfs)

        if self.processors is not None:
            logger.info(f"Applying {len(self.processors)} processors")
            joined_df = reduce(
                lambda df, processor: processor.transform(df),
                self.processors,
                joined_df,
            )

        if self.saver and output_blob:
            logger.info(f"Saving joined data to {output_blob}")
            self.saver.save(bucket, output_blob, joined_df)

        return joined_df


class ValueWage:
    def __init__(
        self,
        join_method: Processor,
        loader: DataLoader,
        saver: DataSaver | None = None,
        processors: list[Processor] | None = None,
    ) -> None:
        self.processors = processors
        self.join_method = join_method
        self.loader = loader
        self.saver = saver

    def process(self, val_blob: str, wage_blob: str) -> pd.DataFrame:
        logger.info(f"Joining {wage_blob} wages and valuations")

        val_df = self.loader.load(
            bucket=Buckets.PROCESSED_TRANSFERMARKT,
            blob=f"processed_{val_blob}_player_valuations.csv",
        )
        wage_df = self.loader.load(
            bucket=Buckets.PROCESSED_FBREF, blob=f"processed_{wage_blob}-wages.csv"
        )

        joined_df = self.join_method.transform([val_df, wage_df])

        if self.processors is not None:
            joined_df = reduce(
                lambda df, processor: processor.transform(df),
                self.processors,
                joined_df,
            )

        if self.saver:
            output_blob = f"{val_blob}_wages_values.csv"
            logger.info(f"Saving joined data to {output_blob}")
            self.saver.save(
                bucket=Buckets.JOINED_WAGES_VALUES, blob=output_blob, data=joined_df
            )

        return joined_df
