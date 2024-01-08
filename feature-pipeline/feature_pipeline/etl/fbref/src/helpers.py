import pandas as pd
import numpy as np
import re


# ------------------------ general cleaning functions ------------------------ #


def remove_row_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Removes row headers from dataframe

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe without row headers
    """
    return df[(df.index - 25) % 26 != 0]


def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column names

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with cleaned column names
    """
    df.columns = [
        re.sub(r"[^a-zA-Z0-9]+", "_", col.replace("%", "_pct")).lower()
        for col in df.columns
    ]
    return df


def clean_nations(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans nation column

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with cleaned nation column
    """
    df["nation"] = df["nation"].str.split(" ").str[1]
    return df


def clean_comp(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans comp column

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with cleaned comp column
    """
    df["comp"] = df["comp"].str.split(" ", n=1).str[1]
    return df


def clean_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans age column

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with cleaned age column
    """
    df["age"] = df["age"].str.split("-").str[0]
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops columns from dataframe

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with dropped columns
    """
    df = df.drop(columns=["matches"])
    return df


def float_dtypes(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Converts selected columns to float16 dtype.

    Args:
        df (pd.DataFrame): fbref dataframe
        columns (list[str]): list of columns to convert to float16

    Returns:
        pd.DataFrame: fbref dataframe with converted columns
    """
    df[columns] = df[columns].astype("float16")
    return df


def int_types(df: pd.DataFrame, columns: str) -> pd.DataFrame:
    """Converts selected columns to Int32 dtype.

    Args:
        df (pd.DataFrame): fbref dataframe
        columns (str): list of columns to convert to Int32

    Returns:
        pd.DataFrame: fbref dataframe with converted columns
    """
    for col in columns:
        try:
            df[col] = np.floor(pd.to_numeric(df[col], errors="coerce")).astype("Int32")
        except ValueError as e:
            print(e)
    return df


def categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Converts object columns to category dtype if unique values < 200.

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with converted columns
    """
    for cols in df.select_dtypes("object").columns:
        if df[cols].nunique() < 200:
            df[cols] = df[cols].astype("category")
    return df


def clean_wages(df: pd.DataFrame, comp: str) -> pd.DataFrame:
    """Cleaning function for player wages

    Args:
        df (pd.DataFrame): player wages dataframe
        wages (str): either weekly or annual
        comp (str): competition name

    Returns:
        pd.DataFrame: cleaned player wages dataframe
    """
    for wage in ["annual_wages", "weekly_wages"]:
        if comp != "Premier-League":
            df[wage] = [
                val.split(" (")[0].replace("€ ", "").replace(",", "")
                for val in df[wage]
            ]
        else:
            df[wage] = [
                val.split(" (")[1].split(", ")[0].replace("€ ", "").replace(",", "")
                for val in df[wage]
            ]

        df = df.rename(columns={wage: f"{wage}_euros"})
    return df


# ------------------------ General cleaning functions ------------------------ #


def general_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """General cleaning function for fbref dataframes

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with general cleaning applied
    """
    return (
        df.pipe(remove_row_headers)
        .pipe(clean_col_names)
        .pipe(drop_columns)
        .pipe(clean_ages)
        .pipe(clean_nations)
        .pipe(clean_comp)
    )


def wage_gen_cleaning(df: pd.DataFrame, comp: str) -> pd.DataFrame:
    return (
        df.pipe(clean_col_names)
        .pipe(clean_nations)
        .pipe(clean_wages, comp=comp)
        
    )


# ------------------------ Xg column cleaning function ----------------------- #


def clean_xg_diff_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans xg difference columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with cleaned xg difference columns
    """
    cols = [col for col in df.columns if "minus" in col]
    for col in cols:
        df[col] = df[col].str.replace("+", "")
    return df


# ---------------------------------- Renaming functions --------------------------------- #


def defensive_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames defensive stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "pos": "position",
        "90s": "90s_played",
        "tkl": "tackles",
        "tklw": "tackles_won",
        "def_3rd": "def_third_tackles",
        "mid_3rd": "mid_third_tackles",
        "att_3rd": "att_third_tackles",
        "tkl_1": "dribblers_tackled",
        "att": "dribblers_challenged",
        "tkl_pct": "tackle_pct",
        "lost": "challenges_lost",
        "sh": "shots_blocked",
        "pass": "passes_blocked",
        "int": "interceptions",
        "tkl_int": "tackles_interceptions",
        "clr": "clearances",
        "err": "errors",
    }
    return df.rename(columns=rename_cols)


def gca_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames goal contributing actions stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "sca": "shot_creating_actions",
        "sca90": "sca_per_90",
        "passlive": "pass_live_sca",
        "passdead": "pass_dead_sca",
        "to": "take_on_sca",
        "sh": "shot_sca",
        "fld": "foul_drawn_sca",
        "def": "defensive_action_sca",
        "gca": "goal_creating_actions",
        "gca90": "gca_per_90",
        "passlive_1": "pass_live_gca",
        "passdead_1": "pass_dead_gca",
        "to_1": "take_on_gca",
        "sh_1": "shot_gca",
        "fld_1": "foul_drawn_gca",
        "def_1": "defensive_action_gca",
    }
    df = df.rename(columns=rename_cols)
    return df


def keeper_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames keeper stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "ga": "goals_against",
        "ga90": "ga_per_90",
        "sota": "shots_on_target_against",
        "w": "wins",
        "d": "draws",
        "l": "losses",
        "cs": "clean_sheets",
        "cs_pct": "clean_sheet_pct",
        "pkatt": "penalties_faced",
        "pka": "penalties_allowed",
        "pksv": "penalties_saved",
        "pkmsv": "penalties_missed",
        "save_pct_1": "penalty_save_pct",
    }
    df = df.rename(columns=rename_cols)
    return df


def ad_keeper(df: pd.DataFrame) -> pd.DataFrame:
    """Renames advanced keeper stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "pka": "penalties_allowed",
        "fk": "free_kick_goals_allowed",
        "ck": "corner_goals_allowed",
        "og": "own_goals",
        "psxg": "post_shot_xg",
        "psxg_sot": "post_shot_xg_on_target",
        "psxg_": "post_shot_xg_minus_ga",
        "_90": "post_shot_xg_minus_ga_per_90",
        "cmp": "passes_completed_plus_40_yards",
        "att": "passes_attempted_plus_40_yards",
        "cmp_pct": "passes_pct_plus_40_yards",
        "att_gk_": "total_passes_attempted",
        "thr": "throws_attempted",
        "avglen": "avg_pass_length",
        "att_1": "goal_kicks_attempted",
        "launch_pct_1": "goal_kick_launch_pct",
        "avglen_1": "avg_goal_kick_length",
        "opp": "crosses_faced",
        "stp": "crosses_stopped",
        "stp_pct": "crosses_stopped_pct",
        "_opa": "defensive_actions_outside_penalty_area",
        "_opa_90": "defensive_opa_per_90",
        "avgdist": "avg_defensive_action_distance",
    }
    df = df.rename(columns=rename_cols)
    return df


def misc_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Renames miscellaneous stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "crdy": "yellow_cards",
        "crdr": "red_cards",
        "2crdy": "second_yellow_cards",
        "fls": "fouls",
        "fld": "fouls_drawn",
        "off": "offsides",
        "crs": "crosses",
        "int": "interceptions",
        "tklw": "tackles_won",
        "pkwon": "penalties_won",
        "pkcon": "penalties_conceded",
        "og": "own_goals",
        "recov": "ball_recoveries",
        "won": "aerials_won",
        "lost": "aerials_lost",
        "won_pct": "aerials_won_pct",
    }
    df = df.rename(columns=rename_cols)
    return df


def passing_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames passing stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "cmp": "passes_completed",
        "att": "passes_attempted",
        "cmp_pct": "pass_completion_pct",
        "totdist": "total_pass_distance",
        "prgdist": "progressive_pass_distance",
        "cmp_1": "short_passes_completed",
        "att_1": "short_passes_attempted",
        "cmp_pct_1": "short_pass_completion_pct",
        "cmp_2": "medium_passes_completed",
        "att_2": "medium_passes_attempted",
        "cmp_pct_2": "medium_pass_completion_pct",
        "cmp_3": "long_passes_completed",
        "att_3": "long_passes_attempted",
        "cmp_pct_3": "long_pass_completion_pct",
        "ast": "assists",
        "a_xag": "assists_minus_xag",
        "kp": "key_passes",
        "1_3": "passes_into_final_third",
        "ppa": "passes_into_penalty_area",
        "crspa": "crosses_into_penalty_area",
        "prgp": "progressive_passes",
    }

    df = df.rename(columns=rename_cols)
    return df


def pass_type_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames passing types stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "att": "passes_attempted",
        "live": "live_ball_passes",
        "dead": "dead_ball_passes",
        "fk": "free_kick_passes",
        "tb": "through_balls",
        "sw": "switches",
        "crs": "crosses",
        "ti": "throw_ins",
        "ck": "corner_kicks",
        "in": "inswinging_corners",
        "out": "outswinging_corners",
        "str": "straight_corners",
        "cmp": "passes_completed",
        "off": "passes_offside",
        "blocks": "passes_blocked",
    }
    df = df.rename(columns=rename_cols)
    return df


def playing_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames playing time stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "min": "mins_played",
        "mn_mp": "mins_per_match",
        "min_pct": "mins_played_pct",
        "mn_start": "mins_per_start",
        "compl": "complete_matches_played",
        "mn_sub": "mins_per_sub",
        "unsub": "unused_sub",
        "ppm": "points_per_match",
        "ong": "team_goals_while_on_pitch",
        "onga": "team_goals_conceded_while_on_pitch",
        "_": "goals_minus_concded_while_on_pitch",
        "_90": "goals_minus_conceded_while_on_pitch_per_90",
        "on_off": "net_goals_on_minus_off_pitch_per_90",
        "onxg": "team_xg_while_on_pitch",
        "onxga": "team_xg_conceded_while_on_pitch",
        "xg_": "team_xg_minus_xga_while_on_pitch",
        "xg_90": "team_xg_minus_xga_while_on_pitch_per_90",
        "on_off_1": "net_xg_on_minus_off_pitch_per_90",
    }
    df = df.rename(columns=rename_cols)
    return df


def possession_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames possession stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "def_pen": "def_pen_area_touches",
        "def_3rd": "def_third_touches",
        "mid_3rd": "mid_third_touches",
        "att_3rd": "att_third_touches",
        "att_pen": "att_pen_area_touches",
        "live": "live_ball_touches",
        "att": "take_ons_attempted",
        "succ": "take_ons_successful",
        "succ_pct": "take_on_succ_pct",
        "tkld": "take_ons_tackled",
        "tkld_pct": "take_on_tackled_pct",
        "totdist": "total_distance_carried",
        "prgdist": "progressive_carries_distance",
        "prgc": "progressive_carries",
        "1_3": "carries_into_final_third",
        "cpa": "carries_into_penalty_area",
        "mis": "miscontrols",
        "dis": "dispossessed",
        "rec": "passes_received",
        "prgr": "progressive_passes_received",
    }
    df = df.rename(columns=rename_cols)
    return df


def shooting_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renames shooting stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "gls": "goals",
        "sh": "shots",
        "sot": "shots_on_target",
        "sot_pct": "shots_on_target_pct",
        "sh_90": "shots_per_90",
        "sot_90": "shots_on_target_per_90",
        "g_sh": "goals_per_shot",
        "g_sot": "goals_per_shot_on_target",
        "dist": "avg_shot_distance",
        "fk": "free_kick_shots",
        "pk": "penalty_kicks",
        "pkatt": "penalty_kicks_attempted",
        "npxg": "non_penalty_xg",
        "npxg_sh": "non_penalty_xg_per_shot",
        "g_xg": "goals_minus_xg",
        "np_g_xg": "non_penalty_goals_minus_xg",
    }
    df = df.rename(columns=rename_cols)
    return df


def stand_stats_col(df: pd.DataFrame) -> pd.DataFrame:
    """Renames standard stats columns

    Args:
        df (pd.DataFrame): fbref dataframe

    Returns:
        pd.DataFrame: fbref dataframe with renamed columns
    """
    rename_cols = {
        "gls": "goals",
        "ast": "assists",
        "g_a": "goals_assists",
        "g_pk": "non_penalty_goals",
        "pk": "penalty_kicks",
        "pkatt": "penalty_kicks_attempted",
        "crdy": "yellow_cards",
        "crdr": "red_cards",
        "npxg": "non_penalty_xg",
        "npxg_xag": "non_pen_xg_plus_xag",
        "prgc": "progressive_carries",
        "prgp": "progressive_passes",
        "prgr": "progressive_passes_received",
        "gls_1": "goals_per_90",
        "ast_1": "assists_per_90",
        "g_a_1": "goals_assists_per_90",
        "g_pk_1": "non_penalty_goals_per_90",
        "g_a_pk": "non_penalty_goals_assists_per_90",
        "xg_1": "xg_per_90",
        "xag_1": "xag_per_90",
        "xg_xag": "xg_plus_xag_per_90",
        "npxg_1": "non_penalty_xg_per_90",
        "npxg_xag_1": "non_penalty_xg_plus_xag_per_90",
    }
    df = df.rename(columns=rename_cols)
    return df
