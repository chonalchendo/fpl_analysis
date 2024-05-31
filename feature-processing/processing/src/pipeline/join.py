from functools import reduce

import pandas as pd

from processing.abcs.loader import DataLoader
from processing.abcs.processor import Processor
from processing.gcp.buckets import Buckets


class Stats:
    def __init__(
        self, processors: list[Processor], join_method: Processor, loader: DataLoader
    ) -> None:
        self.processors = processors
        self.loader = loader
        self.join_method = join_method

    def process(self, bucket: str, tables: list[str]) -> pd.DataFrame:
        dfs = [self.loader.load(bucket, table) for table in tables]

        joined_df = self.join_method.transform(dfs)

        processed_df = reduce(
            lambda df, processor: processor.transform(df), self.processors, joined_df
        )

        # processed_df = pd.concat(
        #     [processor.transform(joined_df) for processor in self.processors], axis=1
        # )

        return processed_df


class ValueWage:
    def __init__(
        self,
        join_method: Processor,
        loader: DataLoader,
        preprocessors: list[Processor] | None = None,
        postprocessors: list[Processor] | None = None,
    ) -> None:
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors
        self.join_method = join_method
        self.loader = loader

    def process(self, val_blob: str, wage_blob: str) -> pd.DataFrame:
        val_df = self.loader.load(bucket=Buckets.PROCESSED_TRANSFERMARKT, blob=val_blob)
        wage_df = self.loader.load(bucket=Buckets.PROCESSED_FBREF, blob=wage_blob)

        joined_df = self.join_method.transform([val_df, wage_df])

        processed_df = reduce(
            lambda df, processor: processor.transform(df), self.processors, joined_df
        )

        return processed_df
