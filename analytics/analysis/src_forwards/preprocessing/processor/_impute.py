from typing import Literal

import pandas as pd

from analysis.base.processor import Processor


class ConditionImputer(Processor):
    def __init__(
        self, features: str | list[str], condition: str | int, value: str | int
    ) -> None:
        super().__init__(features)
        self.condition = condition
        self.value = value

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if self.features is not None and len(self.features) == 2:
            df.loc[
                df[self.features[0]] == self.condition, self.features[1]
            ] = self.value
        else:
            df.loc[
                df[self.features[0]] == self.condition, self.features[0]
            ] = self.value
        return df


class GroupbyImputer(Processor):
    def __init__(
        self,
        features: str | list[str],
        groupby: str | list[str],
        method: Literal["mean", "median"],
    ) -> None:
        super().__init__(features)
        self.groupby = groupby
        self.method = method

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df.loc[:, self.features] = df[self.features].fillna(
            df.groupby(self.groupby)[self.features].transform(self.method)
        )
        return df


class FillnaImputer(Processor):
    def __init__(
        self,
        features: str | list[str],
        value: str | int | None = None,
        column_fill: str | None = None,
        method: str | None = None,
    ) -> None:
        super().__init__(features)
        self.value = value
        self.column_fill = column_fill
        self.method = method

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if self.column_fill is not None and self.value is None:
            df.loc[:, self.features] = df[self.features].fillna(df[self.column_fill])

        elif self.method is not None:
            df.loc[:, self.features] = df[self.features].fillna(
                df[self.features].mode().iloc[0]
            )
        else:
            df.loc[:, self.features] = df[self.features].fillna(self.value)
        return df
