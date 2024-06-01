import pandas as pd

from processing.abcs.processor import Processor


class Filter(Processor):
    def __init__(self, not_like: str, features=None) -> None:
        super().__init__(features)
        self.not_like = not_like

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        not_like_cols = [col for col in df.columns if self.not_like not in col]
        return df[not_like_cols]


class Imputer(Processor):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing height values with the median height."""
        df.loc[
            (df[self.features] == 0) | (df[self.features].isna()), self.features
        ] = df[self.features].median()
        return df


class Rename(Processor):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.features)


class Drop(Processor):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.features)


class ReplaceName(Processor):
    def __init__(self, pattern: str, replace: str, features=None) -> None:
        super().__init__(features)
        self.pattern = pattern
        self.replace = replace

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data.columns = [col.replace(pattern, replace) for col in data.columns]
        return data
