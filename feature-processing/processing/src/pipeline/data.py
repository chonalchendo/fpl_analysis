from functools import reduce

import pandas as pd

from processing.abcs.loader import DataLoader
from processing.abcs.processor import Processor
from processing.abcs.saver import DataSaver
from processing.utilities.logger import get_logger

logging = get_logger(__name__)


class DataProcessor:
    def __init__(
        self,
        processors: list[Processor],
        loader: DataLoader,
        saver: DataSaver | None = None,
    ) -> None:
        self.processors = processors
        self.loader = loader
        self.saver = saver

    def process(
        self, bucket: str, blob: str, output_path: str | None = None
    ) -> pd.DataFrame:
        df = self.loader.load(bucket=bucket, blob=blob)
        logging.info(f"Raw data: \n{df.head()}")

        processed_df = reduce(
            lambda df, processor: processor.process(df), self.processors, df
        )

        if self.saver and output_path:
            self.saver.save(df=processed_df, path=output_path)
        logging.info(f"Processed df: \n{processed_df.head()}")
        return processed_df
