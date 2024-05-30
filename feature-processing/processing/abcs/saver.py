from abc import ABC, abstractmethod

import pandas as pd


class DataSaver(ABC):
    @abstractmethod
    def save(self, bucket: str, blob: str, data: pd.DataFrame) -> None:
        pass
