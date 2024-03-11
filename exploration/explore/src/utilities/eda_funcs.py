import pandas as pd
import numpy as np


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
