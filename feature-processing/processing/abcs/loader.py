from abc import ABC, abstractmethod

import pandas as pd


class DataLoader(ABC):
    @abstractmethod
    def load(self, bucket: str, blob: str) -> pd.DataFrame:
        pass