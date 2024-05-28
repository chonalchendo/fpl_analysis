import pandas as pd
from dataclasses import dataclass


@dataclass
class CleanPlayer:
    """Class to clean transfermarkt player values data."""

    _data: pd.DataFrame

    def create_signed_year_col(self) -> None:
        """Creates a signed year column from the signed date column."""
        self._data.loc[:, "signed_year"] = (
            self._data["signed_date"].str.split(" ").str[2].astype("Int32")
        )

    def impute_height_col(self) -> None:
        """Imputes missing height values with the median height."""
        self._data.loc[
            (self._data["height"] == 0) | (self._data["height"].isna()), "height"
        ] = self._data["height"].median()
        
    def pipeline(self) -> pd.DataFrame:
        """Runs all cleaning methods.

        Returns:
            pd.DataFrame: cleaned dataframe
        """
        self.create_signed_year_col()
        self.impute_height_col()
        return self._data
    
    
@dataclass
class CleanTeam:
    _data: pd.DataFrame

    def create_team_season_col(self) -> pd.DataFrame:
        self._data.loc[:, "team_season"] = (
            self._data["team"] + " - " + self._data["season"].astype(str)
        )
        return self._data

    def create_foreign_pct_col(self) -> pd.DataFrame:
        self._data.loc[:, "foreigner_pct"] = round(
            (self._data["squad_foreigners"] / self._data["squad_size"]) * 100, 2
        )
        return self._data

    def pipeline(self) -> pd.DataFrame:
        self.create_team_season_col()
        self.create_foreign_pct_col()
        return self._data
