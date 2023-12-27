# from pathlib import Path
# import sys

# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from functools import reduce
from pandas.plotting import scatter_matrix
import plotly.express as px
from data import db

pd.set_option("display.max_columns", None)

data = db.query("SELECT * FROM player_stats")

df = data.drop(columns=["index"])

# describe the data
df.describe()
df.info()

# check for missing values
df.isnull().sum().sort_values(ascending=False)

df["opponent_team_name"].value_counts()
df["player"].value_counts().head(10)

df["team_difficulty"].value_counts()

df.select_dtypes("number").corr()["total_points"].sort_values(ascending=False)


px.line(df, x="kickoff_time_utc", y="total_points", color="player")
px.line(df, x="kickoff_time_utc", y="value", color="player")


def join_multiple_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    return reduce(
        lambda left, right: pd.merge(left, right, how="outer", on="player"), dfs
    )


def calculate_stats(groupby: str, stat: str, df: pd.DataFrame = df):
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


# calculate best performing players given opposition difficulty
def value_per_opp_difficulty(
    df: pd.DataFrame,
    latest_fixture: str,
    minutes_played: str,
    stats: list[str],
    difficulty: int | None = None,
) -> pd.DataFrame:
    if difficulty:
        df = df.loc[df["team_difficulty"] == difficulty]
    return calculate_adjusted_value(df, latest_fixture, minutes_played, stats=stats)


stats = [
    "total_points",
    "assists",
    "clean_sheets",
    "goals_scored",
    "goals_conceded",
    "minutes",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "expected_goals",
    "bps",
    "ict_index",
    "creativity",
    "threat",
    "influence",
]

# calculate valuation stats for all players
dff = value_per_opp_difficulty(df, 17, 500, stats=stats)

# calculate correlation stats for points per 90 minutes
dff.select_dtypes("number").corr()["pp90"].sort_values(ascending=False)

# visualise relationship between points per 90 minutes and other stats
px.scatter(dff, x="threat", y="pp90", hover_name="player")

# create a scatter matrix for selected variables
scatter_matrix(dff[["pp90", "expected_goals", "expected_assists", "total_points"]], figsize=(12, 8))




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
