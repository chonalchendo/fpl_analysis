import pandas as pd
import numpy as np
from dataclasses import dataclass
from functools import reduce
from collections import Counter


def outliers(df: pd.DataFrame) -> pd.Series:
    """Custom function to calculate the number of outliers in a dataframe
    using the interquartile range.

    Args:
        df (pd.DataFrame): pandas dataframe

    Returns:
        pd.Series: number of outliers for each column
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """Custom function to describe the data including null values, data types,
    and unique values.

    Args:
        df (pd.DataFrame): pandas dataframe

    Returns:
        pd.DataFrame: descripive stats of the dataframe
    """
    df = df.select_dtypes(include=[np.number]).copy()

    # check null values
    null_counts = df.isnull().sum().sort_values(ascending=False)

    # calculate percentage of null values
    null_percentages = round(
        df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2
    )

    # check data types
    dtypes = df.dtypes

    # check unique values
    unique_vals = df.nunique().sort_values(ascending=False)

    # min and max values
    min_vals = df.min()
    max_vals = df.max()
    median_vals = round(df.median(), 2)
    mean_vals = round(df.mean(), 2)
    std_vals = round(df.std(), 2)

    # outliers - interquartile range
    outlier_vals = outliers(df)
    outlier_pct = round(outlier_vals / len(df) * 100, 2)

    return pd.concat(
        [
            dtypes,
            null_counts,
            null_percentages,
            unique_vals,
            min_vals,
            max_vals,
            median_vals,
            mean_vals,
            std_vals,
            outlier_vals,
            outlier_pct,
        ],
        axis=1,
        keys=[
            "Data Types",
            "Null Counts",
            "Null %",
            "Unique Values",
            "Min",
            "Max",
            "Median",
            "Mean",
            "Std Dev",
            "Outliers",
            "Outliers %",
        ],
    )


@dataclass
class PlayerValues:
    """Class to quickly analyse player valuations by a categorical column.

    Args:
        _data (pd.DataFrame): pandas dataframe
        column (str): column to group by
    """

    _data: pd.DataFrame
    column: str

    def manipulate_data(self) -> pd.DataFrame:
        """Manipulate the data to group by the column and calculate the mean

        Returns:
            pd.DataFrame: manipulated dataframe
        """
        df = self._data.copy()

        return (
            df.groupby(self.column)[["signing_fee_euro_mill", "market_value_euro_mill"]]
            .mean()
            .sort_values(by="signing_fee_euro_mill", ascending=False)
            .reset_index()
        )

    def create_diff_paid_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a column to show the difference between the market value and
        the signing fee.

        Args:
            df (pd.DataFrame): pandas dataframe

        Returns:
            pd.DataFrame: dataframe with new column
        """
        df.loc[:, "diff_value_paid"] = (
            df["market_value_euro_mill"] - df["signing_fee_euro_mill"]
        )
        return df

    def create_color_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a column to show the color of the difference between the
        market value and the signing fee that are positive or negative.

        Args:
            df (pd.DataFrame): pandas dataframe

        Returns:
            pd.DataFrame: dataframe with new column
        """
        df.loc[:, "color"] = df["diff_value_paid"].apply(
            lambda x: "blue" if x > 0 else "red"
        )
        return df

    def create_diff_val_paid_perc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a column to show the percentage difference between the market
        value and the signing fee.

        Args:
            df (pd.DataFrame): pandas dataframe

        Returns:
            pd.DataFrame: dataframe with new column
        """
        df.loc[:, "diff_value_paid_perc"] = round(
            (df["diff_value_paid"] / df["signing_fee_euro_mill"]) * 100, 2
        )
        return df

    def pipeline(self) -> pd.DataFrame:
        """Runs all cleaning methods.

        Returns:
            pd.DataFrame: cleaned dataframe
        """
        df = self.manipulate_data()
        df.pipe(self.create_diff_paid_col).pipe(self.create_color_col).pipe(
            self.create_diff_val_paid_perc
        )
        return df


class PlayerValData:
    """Class to quickly generate datasets that analyse player valuations from
    Transfermarkt.
    """

    @staticmethod
    def diff_values_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Create a dataframe to show the difference between the market value
        and the signing fee by a categorical column.

        Args:
            df (pd.DataFrame): original dataframe from Transfermarkt
            column (str): column to group by

        Returns:
            pd.DataFrame: dataframe with the difference between the market value
            and the signing fee
        """
        return PlayerValues(_data=df, column=column).pipeline()

    @staticmethod
    def season_count_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Create a dataframe to show the number of appearances by a categorical
        column.

        Args:
            df (pd.DataFrame): original dataframe from Transfermarkt
            column (str): column to group by

        Returns:
            pd.DataFrame: dataframe with the number of appearances
        """
        seasons = df["season"].unique()

        def get_country_counts(season: int) -> pd.DataFrame:
            """Get the number of appearances by a categorical column for a
            specific season.

            Args:
                season (int): season to filter by

            Returns:
                pd.DataFrame: dataframe with the number of appearances
            """
            data = df.loc[df["season"] == season][column].value_counts().reset_index()
            data.columns = ["country", f"{season}"]
            return data

        dfs = [get_country_counts(season) for season in seasons]

        return reduce(lambda left, right: pd.merge(left, right, on="country"), dfs)

    @staticmethod
    def count_season_apps_df(df: pd.DataFrame) -> pd.DataFrame:
        """Create a dataframe to show the number of season appearances by team.

        Args:
            df (pd.DataFrame): original dataframe from Transfermarkt

        Returns:
            pd.DataFrame: dataframe with the number of appearances
        """
        seasons = df["season"].unique()

        # count the number of appearances by team
        teams = []
        for season in seasons:
            values = df.loc[df["season"] == season]["team"].unique().tolist()
            teams.extend(values)

        cnt = Counter(teams)

        # create a dataframe from the counter
        return pd.DataFrame(
            {"teams": list(cnt.keys()), "appearances": list(cnt.values())}
        ).sort_values(by="appearances", ascending=False)

    @staticmethod
    def value_signings_df(df: pd.DataFrame) -> pd.DataFrame:
        """Create a dataframe to show the value for money signings based on
        signing fee and end of year market value.

        Args:
            df (pd.DataFrame): original dataframe from Transfermarkt

        Returns:
            pd.DataFrame: dataframe with the value for money signings
        """

        def filter_signing_season(df: pd.DataFrame, year: int) -> pd.DataFrame:
            """Filter the dataframe by the signing year and season.

            Args:
                df (pd.DataFrame): original dataframe from Transfermarkt
                year (int): year to filter by

            Returns:
                pd.DataFrame: filtered dataframe
            """
            return df.loc[
                (df["signed_year"] == year) & (df["season"] == year)
            ].sort_values("signing_fee_euro_mill", ascending=False)

        data = [
            filter_signing_season(df, year)
            for year in df["signed_year"].unique()
            if year is not None
        ]

        dff = pd.concat(data)

        dff.loc[:, "diff_sign_fee_mv"] = (
            dff["market_value_euro_mill"] - dff["signing_fee_euro_mill"]
        )

        return dff
