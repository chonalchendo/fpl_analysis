from abc import ABC, abstractmethod

import pandas as pd


class DataSaver(ABC):
    @abstractmethod
    def save(self, df: pd.DataFrame, path: str) -> None:
        pass
