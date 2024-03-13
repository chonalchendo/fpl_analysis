import pandas as pd
from dataclasses import dataclass


@dataclass
class StatsData:
    """Class to create stats dataframes."""

    _data: pd.DataFrame

    def avgs_df(self, groupby: str, columns: list[str]) -> pd.DataFrame:
        """Create dataframe of average stats.

        Args:
            groupby (str): column to groupby e.g. 'comp'.
            columns (list[str]): columns to average.

        Returns:
            pd.DataFrame: dataframe of average stats
        """
        data = self._data.copy()

        avgs_df = data.groupby(["season", groupby])[columns].mean().reset_index()
        avgs_df.columns = ["season", groupby] + [
            "".join(("avg_", col))
            for col in avgs_df.columns
            if col not in ["season", groupby]
        ]
        return avgs_df

    def top_countries_df(self) -> pd.DataFrame:
        """Create dataframe of top countries by minutes played.

        Returns:
            pd.DataFrame: dataframe of top countries by minutes played
        """
        data = self._data.copy()

        # return countries who have collectively the most minutes played
        mins_played = (
            data.groupby("country")["90s"].sum().sort_values(ascending=False).head(10)
        )
        countries = mins_played.index[:10].tolist()
        return data[data["country"].isin(countries)]

    def top_10_per_position(
        self,
        value: str,
        position: str = "Defender",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Create dataframe of top 10 players per position for a given metric.

        Args:
            value (str): metric to rank by e.g. 'npxG', 'npxG_per90'.
            position (str, optional): player position. Defaults to "Defender".
            ascending (bool, optional): ascend or decend. Defaults to False.

        Returns:
            pd.DataFrame: dataframe of top 10 players per position for a given metric
        """
        data = (
            self._data.loc[self._data["general_pos"] == position][["season", "player", value]]
            .sort_values(value, ascending=ascending)
            .reset_index(drop=True)
            .head(10)
        )

        data.loc[:, "player_season"] = data["player"] + " - " + data["season"]
        return data
