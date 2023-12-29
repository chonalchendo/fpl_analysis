import pandas as pd
from functools import reduce
import plotly.express as px
from plotly.graph_objs import Figure


# --------------------------- Exploration functions -------------------------- #


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """Custom dataframe that describes the data in detail.

    Args:
        df (pd.DataFrame): pandas dataframe

    Returns:
        pd.DataFrame: pandas dataframe
    """
    summary = df.describe(include="all").T
    summary["missing"] = df.isnull().sum()
    summary["missing_pct"] = summary["missing"] / len(df)
    summary["dtype"] = df.dtypes
    summary["unique"] = df.nunique()
    summary["unique_pct"] = summary["unique"] / len(df)
    summary["skew"] = df.skew()
    summary["kurtosis"] = df.kurtosis()

    for col in df.select_dtypes("number").columns:
        summary[f"{col}_value_counts"] = df[col].value_counts()
    return summary


def correlations(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Calculate correlations for a specific variable.

    Args:
        df (pd.DataFrame): pandas dataframe

    Returns:
        pd.DataFrame: pandas dataframe
    """
    return df.select_dtypes("number").corr()[variable].sort_values(ascending=False)


def create_line_plot(df: pd.DataFrame, x: str, y: str, color: str) -> Figure:
    """Create a line plot.

    Args:
        df (pd.DataFrame): pandas dataframe
        x (str): x axis variable
        y (str): y axis variable
        color (str): color variable
    """
    return px.line(df, x=x, y=y, color=color)


def join_multiple_dfs(dfs: list[pd.DataFrame], how: str, on: str) -> pd.DataFrame:
    """Join multiple dataframes.

    Args:
        dfs (list[pd.DataFrame]): list of pandas dataframes
        how (str): how to join the dataframes
        on (str): column to join the dataframes on

    Returns:
        pd.DataFrame: pandas dataframe
    """
    return reduce(lambda left, right: pd.merge(left, right, how=how, on=on), dfs)


def calculate_stats(groupby: str, stat: str, df: pd.DataFrame = df) -> pd.DataFrame:
    """Calculate stats for a given groupby and stat.

    Args:
        groupby (str): column to groupby
        stat (str): stat to calculate
        df (pd.DataFrame, optional): pandas dataframe. Defaults to df.

    Returns:
        pd.DataFrame: grouped by pandas dataframe with calculated stats
    """
    return df.groupby(groupby)[stat].sum().sort_values(ascending=False).reset_index()


def calculate_adjusted_value(
    df: pd.DataFrame, latest_fixture: int, minutes_played: int, stats: list[str]
) -> pd.DataFrame:
    """Calculate adjusted value for players with more than 'n' minutes played.

    Args:
        df (pd.DataFrame):
        latest_fixture (int): latest fixture to adjust value metric
        minutes_played (int): minimum minutes played by player

    Returns:
        pd.DataFrame: pandas dataframe with adjusted value metric
    """

    dfs = [calculate_stats("player", stat, df) for stat in stats]

    dff = join_multiple_dfs(dfs)

    fixture = df.loc[df["gameweek"] == latest_fixture]
    latest_value = dict(zip(fixture["player"], fixture["value"]))

    dff["value"] = dff["player"].map(latest_value)
    dff["pp90"] = dff["total_points"] / dff["minutes"] * 90
    dff["adjusted_value"] = dff["pp90"] / dff["value"] * 100

    return dff.loc[dff["minutes"] > minutes_played].sort_values(
        "adjusted_value", ascending=False
    )


def value_per_opp_difficulty(
    df: pd.DataFrame,
    latest_fixture: str,
    minutes_played: str,
    stats: list[str],
    difficulty: int | None = None,
) -> pd.DataFrame:
    """Calculate adjusted value for players with more than 'n' minutes played
    for a given opponent difficulty.

    Args:
        df (pd.DataFrame): player data
        latest_fixture (str): most recent fixture
        minutes_played (str): minimum minutes played by player
        stats (list[str]): list of stats to calculate
        difficulty (int | None, optional): opposition difficulty. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    if difficulty:
        df = df.loc[df["team_difficulty"] == difficulty]
    return calculate_adjusted_value(df, latest_fixture, minutes_played, stats=stats)


# ---------------------------------- Working --------------------------------- #

# stats = [
#     "total_points",
#     "assists",
#     "clean_sheets",
#     "goals_scored",
#     "goals_conceded",
#     "minutes",
#     "expected_assists",
#     "expected_goal_involvements",
#     "expected_goals_conceded",
#     "expected_goals",
#     "bps",
#     "ict_index",
#     "creativity",
#     "threat",
#     "influence",
# ]

# calculate valuation stats for all players
# dff = value_per_opp_difficulty(df, 17, 500, stats=stats)

# # calculate correlation stats for points per 90 minutes
# dff.select_dtypes("number").corr()["pp90"].sort_values(ascending=False)

# # visualise relationship between points per 90 minutes and other stats
# px.scatter(dff, x="threat", y="pp90", hover_name="player")

# # create a scatter matrix for selected variables
# scatter_matrix(
#     dff[["pp90", "expected_goals", "expected_assists", "total_points"]], figsize=(12, 8)
# )


# calculate_adjusted_value(df, 17, 300)["adjusted_value"].hist(bins=50, figsize=(10, 8))

# calculate_stats("player", "total_points").reset_index()
# calculate_stats("player", "assists")
# calculate_stats("player", "clean_sheets")
# calculate_stats("player", "expected_assists")
# calculate_stats("player", "expected_goal_involvements")
# calculate_stats("player", "expected_goals_conceded")

# gs = calculate_stats("player", "goals_scored")
# xg = calculate_stats("player", "expected_goals")

# dff = (
#     pd.concat([gs, xg], axis=1)
#     .sort_values("goals_scored", ascending=False)
#     .reset_index()
#     .head(20)
# )

# px.scatter(
#     dff,
#     x="goals_scored",
#     y="expected_goals",
#     hover_name="player",
# )

# df.hist(bins=50, figsize=(20, 15))


# df.loc[df["is_home"] == True, "total_points"].agg(
#     ["mean", "median", "std", "min", "max"]
# )
# df.loc[df["is_home"] == False, "total_points"].agg(
#     ["mean", "median", "std", "min", "max"]
# )

# df.loc[df["is_home"] == True].groupby("player")["total_points"].sum().sort_values(
#     ascending=False
# ).head(10)
# df.loc[df["is_home"] == False].groupby("player")["total_points"].sum().sort_values(
#     ascending=False
# ).head(10)
