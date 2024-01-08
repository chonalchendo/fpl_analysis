from datetime import datetime
import pandas as pd
import numpy as np
from functools import reduce
from typing import Callable
from feature_pipeline.apis.fpl import get_player_ids, map_team_stats


def string_to_int(players_df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns to int.

    Args:
        players_df (pd.DataFrame): Dataframe with players data.

    Returns:
        pd.DataFrame: Dataframe with converted columns.
    """
    string_cols = [
        col for col in players_df.columns if players_df[col].dtype == "object"
    ]
    players_df[string_cols] = pd.to_numeric(
        players_df[string_cols].stack(), errors="coerce"
    ).unstack()
    return players_df


def filter_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Filter fixtures data to only include fixtures that have already been played.
    Args:
        fixtures: Dataframe with fixtures data.
    Returns:
        Dataframe with filtered fixtures data.
    """
    last_fixture = max(fixtures["id"].unique().tolist())
    current_fix = fixtures.loc[
        (fixtures["id"] <= last_fixture) & (fixtures["event"] >= 1)
    ].reset_index(drop=True)
    columns = [
        "id",
        "event",
        "finished",
        "minutes",
        "provisional_start_time",
        "team_a",
        "team_h",
        "team_h_difficulty",
        "team_a_difficulty",
    ]
    return current_fix[columns]


def filter_teams(teams: pd.DataFrame) -> pd.DataFrame:
    """Filter teams data to only include relevant columns.

    Args:
        teams (pd.DataFrame): Dataframe with teams data.

    Returns:
        pd.DataFrame: Dataframe with filtered teams data.
    """
    columns = [
        "id",
        "name",
        "short_name",
        "strength",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]
    return teams[columns]


def rename_fixtures_columns(
    teams_df: pd.DataFrame, fixtures_df: pd.DataFrame
) -> pd.DataFrame:
    """Rename fixtures columns to match the teams data.

    Args:
        teams_df (pd.DataFrame): teams data.
        fixtures_df (pd.DataFrame): fixtures data.

    Returns:
        pd.DataFrame: Dataframe with renamed fixtures columns.
    """
    teams = dict(zip(teams_df["id"], teams_df["name"]))

    fixtures_df["team_h"] = fixtures_df["team_h"].map(teams)
    fixtures_df["team_a"] = fixtures_df["team_a"].map(teams)

    rename_cols = {
        "team_h": "home_team",
        "team_a": "away_team",
        "team_h_difficulty": "away_team_difficulty",
        "team_a_difficulty": "home_team_difficulty",
    }

    return fixtures_df.rename(columns=rename_cols)


# def map_team_stats(teams_df: pd.DataFrame, stat: str) -> dict[str, int]:
#     """Map team stat to team name.

#     Args:
#         df (pd.DataFrame): teams data.
#         stat (str): stat to map.

#     Returns:
#         dict[str, int]: Dictionary with team name as key and stat as value.
#     """
#     return dict(zip(teams_df["name"], teams_df[stat]))


# def impute_stats(team_data: pd.DataFrame, fixture_data: pd.DataFrame) -> pd.DataFrame:
#     """Impute team stats to fixtures data.

#     Args:
#         team_data (pd.DataFrame): teams data.
#         fixture_data (pd.DataFrame): fixtures data.

#     Returns:
#         pd.DataFrame: Dataframe with imputed stats.
#     """
#     columns = team_data.filter(regex="strength_").columns.tolist()
#     for col in columns:
#         if col.endswith("home"):
#             stats = map_team_stats(team_data, col)
#             fixture_data[f"home_team_{col.split('_')[1]}"] = fixture_data[
#                 "home_team"
#             ].map(stats)
#         else:
#             stats = map_team_stats(team_data, col)
#             fixture_data[f"away_team_{col.split('_')[1]}"] = fixture_data[
#                 "away_team"
#             ].map(stats)
#     return fixture_data


def map_opponent_team_stats(player_stats: pd.DataFrame, stat: str) -> pd.Series:
    """Map opponent team stats to player stats.

    Args:
        player_stats (pd.DataFrame): player stats.
        team_stats (pd.DataFrame): teams stats.
        stat (str): stat to map.

    Returns:
        pd.Series: Series with mapped stats.
    """
    home = map_team_stats(f"strength_{stat}_home")
    away = map_team_stats(f"strength_{stat}_away")

    player_stats[f"opponent_{stat}_home"] = player_stats["opponent_team"].map(home)
    player_stats[f"opponent_{stat}_away"] = player_stats["opponent_team"].map(away)
    final_col = np.where(
        player_stats["was_home"] == True,
        player_stats[f"opponent_{stat}_away"],
        player_stats[f"opponent_{stat}_home"],
    )

    player_stats.drop(
        columns=[f"opponent_{stat}_home", f"opponent_{stat}_away"], inplace=True
    )

    return pd.Series(final_col, name=f"opponent_strength_{stat}")


def append_opponent_team_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Append opponent team stats to player stats.

    Args:
        player_stats (pd.DataFrame): player stats.
        team_stats (pd.DataFrame): teams stats.

    Returns:
        pd.DataFrame: Dataframe with appended stats.
    """
    new_stats = [
        map_opponent_team_stats(player_stats, stat)
        for stat in ["overall", "attack", "defence"]
    ]
    new_stats = pd.concat(new_stats, axis=1)
    new_df = pd.concat([player_stats, new_stats], axis=1)
    return new_df


def add_opponent_team_name(player_stats: pd.DataFrame) -> pd.DataFrame:
    teams = map_team_stats("name")
    player_stats["opponent_team_name"] = player_stats["opponent_team"].map(teams)
    return player_stats


def add_player_names(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Add player names to player stats.

    Args:
        player_stats (pd.DataFrame): player stats.

    Returns:
        pd.DataFrame: Dataframe with player names.
    """
    players = get_player_ids()
    names = {k["id"]: k["first_name"] + " " + k["second_name"] for k in players}
    player_stats["player"] = player_stats["element"].map(names)
    return player_stats


def add_team_difficulty(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Add team difficulty to player stats.

    Args:
        player_stats (pd.DataFrame): player stats.

    Returns:
        pd.DataFrame: Dataframe with team difficulty.
    """
    teams = map_team_stats("strength")
    player_stats["team_difficulty"] = player_stats["opponent_team_id"].map(teams)
    return player_stats


# def transform_teams_fixtures(
#     teams: pd.DataFrame, fixtures: pd.DataFrame
# ) -> pd.DataFrame:
#     """Transform teams data.

#     Args:
#         teams (pd.DataFrame): teams data.

#     Returns:
#         pd.DataFrame: Dataframe with transformed teams data.
#     """
#     new_teams = filter_teams(teams)
#     new_fixtures = filter_fixtures(fixtures)
#     new_fixtures = rename_fixtures_columns(new_teams, new_fixtures)
#     return new_teams, new_fixtures


Preprocessor = Callable[[pd.DataFrame], pd.DataFrame]


def compose(*functions: Preprocessor) -> Preprocessor:
    """Compose functions.

    Args:
        *functions: functions to compose.

    Returns:
        function: Composed function.
    """
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


# def transform(players_df: pd.DataFrame) -> pd.DataFrame:
#     """Transform players data.

#     Args:
#         players_df (pd.DataFrame): players data.

#     Returns:
#         pd.DataFrame: Dataframe with transformed players data.
#     """
#     preprocessor = compose(
#         string_to_int,
#         add_player_names,
#         add_opponent_team_name,
#         append_opponent_team_stats,
#     )
#     return preprocessor(players_df)


def change_column_names(players_df: pd.DataFrame) -> pd.DataFrame:
    """Change column names to better describe the data.

    Args:
        players_df (pd.DataFrame): players data.

    Returns:
        pd.DataFrame: Dataframe with changed column names.
    """
    rename = {
        "element": "player_id",
        "fixture": "fixture_id",
        "opponent_team": "opponent_team_id",
        "team_a_score": "away_team_score",
        "team_h_score": "home_team_score",
        "was_home": "is_home",
        "kickoff_time": "kickoff_time_utc",
        "round": "gameweek",
    }

    return players_df.rename(columns=rename)


def add_season_column(players_df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Add season column to players data.

    Args:
        players_df (pd.DataFrame): players data.
        season (str): season to add.

    Returns:
        pd.DataFrame: Dataframe with season column.
    """
    players_df["season"] = season
    return players_df


def add_time_column(players_df: pd.DataFrame) -> pd.DataFrame:
    """Add last updated column to players data.

    Args:
        players_df (pd.DataFrame): players data.

    Returns:
        pd.DataFrame: Dataframe with last updated column.
    """
    players_df["last_updated"] = datetime.now()
    return players_df


def transform(players_df: pd.DataFrame) -> pd.DataFrame:
    """Transform players data.

    Args:
        players_df (pd.DataFrame): players data.

    Returns:
        pd.DataFrame: Dataframe with transformed players data.
    """
    players_df = string_to_int(players_df)
    players_df = add_player_names(players_df)
    players_df = add_opponent_team_name(players_df)
    players_df = append_opponent_team_stats(players_df)
    players_df = change_column_names(players_df)
    players_df = add_season_column(players_df, "2023-24")
    players_df = add_time_column(players_df)
    players_df = add_team_difficulty(players_df)
    return players_df
