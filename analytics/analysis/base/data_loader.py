from abc import ABC, abstractmethod

import pandas as pd


class DataLoader(ABC):
    @abstractmethod
    def load(self, input_path: str) -> pd.DataFrame:
        pass
