import pandas as pd
from dataclasses import dataclass


@dataclass
class WagesData:
    _data: pd.DataFrame
    groupby: str

    def seasonal_manipulate(self) -> None:
        self._data = (
            self._data.groupby(["season", self.groupby])
            .agg(
                avg_weekly_wages=("weekly_wages_euros", "mean"),
                total_weekly_wages=("weekly_wages_euros", "sum"),
                avg_annual_wages=("annual_wages_euros", "mean"),
                total_annual_wages=("annual_wages_euros", "sum"),
                total_players=("player", "size"),
                avg_age=("age", "mean"),
            )
            .round(2)
            .reset_index()
        )

    def total_manipulate(self) -> None:
        if self.groupby == "country":
            countries = self._data["country"].value_counts().index[:10]
            self._data = self._data[self._data["country"].isin(countries)]
            
        # Calculate mean and count together
        grouped = self._data.groupby(self.groupby)
        mean_df = grouped[["weekly_wages_euros", "annual_wages_euros"]].mean().round(2)
        count_df = grouped.size().to_frame(name="count")

        # Merge the mean and count DataFrames
        self._data = (
            pd.merge(mean_df, count_df, left_index=True, right_index=True)
            .sort_values(
                by=["weekly_wages_euros", "annual_wages_euros"], ascending=False
            )
            .reset_index()
        )

    def avg_wage_pct_change(self) -> None:
        self._data.loc[:, "avg_wage_pct_change"] = (
            self._data.groupby(self.groupby)["avg_weekly_wages"]
            .pct_change()
            .mul(100)
            .fillna(0)
            .round(2)
        )

    def total_wage_pct_change(self) -> None:
        self._data.loc[:, "total_wage_pct_change"] = (
            self._data.groupby(self.groupby)["total_weekly_wages"]
            .pct_change()
            .mul(100)
            .fillna(0)
            .round(2)
        )

    def total_players_pct_change(self) -> None:
        self._data.loc[:, "total_players_pct_change"] = (
            self._data.groupby(self.groupby)["total_players"]
            .pct_change()
            .mul(100)
            .fillna(0)
            .round(2)
        )

    def avg_age_pct_change(self) -> None:
        self._data.loc[:, "avg_age_pct_change"] = (
            self._data.groupby(self.groupby)["avg_age"]
            .pct_change()
            .mul(100)
            .fillna(0)
            .round(2)
        )

    def run_seasonal_data(self) -> pd.DataFrame:
        self.seasonal_manipulate()
        self.avg_wage_pct_change()
        self.total_wage_pct_change()
        self.total_players_pct_change()
        self.avg_age_pct_change()
        return self._data

    def run_total_data(self) -> pd.DataFrame:
        self.total_manipulate()
        return self._data
