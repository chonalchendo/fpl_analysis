from abc import ABC, abstractmethod

import pandas as pd


class Processor(ABC):
    def __init__(
        self, features: list[str] | str | dict[str, str] | None = None
    ) -> None:
        self.features = features

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
