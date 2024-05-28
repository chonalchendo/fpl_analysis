from abc import ABC, abstractmethod

import pandas as pd


class GCPLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        pass
