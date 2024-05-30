import pandas as pd

from processing.abcs.loader import DataLoader
from processing.abcs.processor import Processor


class Datajoiner:
    def __init__(
        self, processors: list[Processor], join_method: Processor, loader: DataLoader
    ) -> None:
        self.processors = processors
        self.loader = loader
        self.join_method = join_method

    def process(self, bucket: str, tables: list[str]) -> pd.DataFrame:
        dfs = [self.loader.load(bucket, table) for table in tables]

        joined_df = self.join_method.process(dfs)

        processed_df = pd.concat(
            [processor.process(joined_df) for processor in self.processors], axis=1
        )

        return processed_df
