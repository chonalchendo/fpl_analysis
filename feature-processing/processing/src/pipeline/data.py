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
        self,
        bucket: str,
        blob: str,
        output_bucket: str | None = None,
        output_blob: str | None = None,
    ) -> pd.DataFrame:
        df = self.loader.load(bucket=bucket, blob=blob)
        logging.info(f"Raw data: \n{df.head()}")

        processed_df = reduce(
            lambda df, processor: processor.transform(df), self.processors, df
        )

        if self.saver and output_bucket and output_blob:
            logging.info(f"Saving processed data to {output_bucket}/{output_blob}")
            self.saver.save(bucket=output_bucket, blob=output_blob, data=processed_df)

        logging.info(f"Processed df: \n{processed_df.head()}")
        return processed_df
