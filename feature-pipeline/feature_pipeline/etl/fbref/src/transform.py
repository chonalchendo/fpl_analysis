import pandas as pd
from feature_pipeline.etl.fbref.src import helpers as rename
from feature_pipeline.etl.fbref.src.helpers import (
    general_cleaning,
    wage_gen_cleaning,
    float_dtypes,
    int_types,
    categorical_dtypes,
    clean_xg_diff_cols,
)


def get_float_cols(df: pd.DataFrame, similar: list[str]) -> list[str]:
    return [col for col in df.columns if any(stat in col for stat in similar)]


def get_int_cols(df: pd.DataFrame, float_cols: list[str]) -> list[str]:
    return [df.columns[0]] + [col for col in df.columns[6:-1] if col not in float_cols]


# ---------------------------------- Defense --------------------------------- #


def defense(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for defensive stats

    Args:
        df (pd.DataFrame): defensive stats dataframe

    Returns:
        pd.DataFrame: cleaned defensive stats dataframe
    """
    df = df.pipe(general_cleaning).pipe(rename.defensive_cols)

    float_cols = get_float_cols(df, ["90", "pct"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# ------------------------------------ gca ----------------------------------- #


def gca(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for goal and shot creating actions

    Args:
        df (pd.DataFrame): goal and shot creating actions dataframe

    Returns:
        pd.DataFrame: cleaned goal and shot creating actions dataframe
    """
    df = df.pipe(general_cleaning).pipe(rename.gca_cols)

    float_cols = get_float_cols(df, ["90"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# ---------------------------------- Keepers --------------------------------- #


def keeper(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for keeper stats

    Args:
        df (pd.DataFrame): keeper stats dataframe

    Returns:
        pd.DataFrame: cleaned keeper stats dataframe
    """
    df = df.pipe(general_cleaning).pipe(rename.keeper_cols)

    float_cols = get_float_cols(df, ["90", "pct"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# ----------------------------- Advanced keepers ----------------------------- #


def ad_keeper(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for advanced keeper stats

    Args:
        df (pd.DataFrame): advanced keeper stats dataframe

    Returns:
        pd.DataFrame: cleaned advanced keeper stats dataframe
    """
    df = df.pipe(general_cleaning).pipe(rename.ad_keeper)

    float_cols = get_float_cols(df, ["90", "pct", "avg", "xg"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(clean_xg_diff_cols)
        .pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# ------------------------------- miscellaneous ------------------------------ #


def misc(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for miscellaneous stats

    Args:
        df (pd.DataFrame): miscellaneous stats dataframe

    Returns:
        pd.DataFrame: cleaned miscellaneous stats dataframe
    """
    df = df.pipe(general_cleaning).pipe(rename.misc_col_names)

    float_cols = get_float_cols(df, ["90", "pct"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# ---------------------------------- passing --------------------------------- #


def passing(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for passing stats

    Args:
        df (pd.DataFrame): passing stats dataframe

    Returns:
        pd.DataFrame: cleaned passing stats dataframe
    """
    df = df.pipe(general_cleaning).pipe(rename.passing_cols).pipe(clean_xg_diff_cols)

    float_cols = get_float_cols(df, ["90", "pct", "xa"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# ---------------------------------- passing types --------------------------- #


def passing_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for passing types stats

    Args:
        df (pd.DataFrame): passing types stats dataframe

    Returns:
        pd.DataFrame: cleaned passing types stats dataframe
    """
    df = df.pipe(general_cleaning).pipe(rename.pass_type_cols)

    float_cols = get_float_cols(df, ["90", "pct"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# ------------------------------- playing time ------------------------------- #


def playing_time(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for playing time stats

    Args:
        df (pd.DataFrame): playing time stats dataframe

    Returns:
        pd.DataFrame: cleaned playing time stats dataframe
    """
    df = (
        df.pipe(general_cleaning)
        .pipe(rename.playing_time_cols)
        .pipe(clean_xg_diff_cols)
    )

    float_cols = get_float_cols(df, ["90", "pct", "points_per_match", "xg"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# -------------------------------- possession -------------------------------- #


def possession(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for possession stats

    Args:
        df (pd.DataFrame): possession stats dataframe

    Returns:
        pd.DataFrame: cleaned possession stats dataframe
    """
    df = df.pipe(general_cleaning).pipe(rename.possession_cols)

    float_cols = get_float_cols(df, ["90", "pct"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# --------------------------------- shooting --------------------------------- #


def shooting(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for shooting stats

    Args:
        df (pd.DataFrame): shooting stats dataframe

    Returns:
        pd.DataFrame: cleaned shooting stats dataframe
    """
    df = df.pipe(general_cleaning).pipe(rename.shooting_cols)

    float_cols = get_float_cols(df, ["90", "pct", "xg", "avg", "per"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# ------------------------------ standard stats ---------------------------- #


def standard_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning function for standard stats

    Args:
        df (pd.DataFrame): standard stats dataframe

    Returns:
        pd.DataFrame: cleaned standard stats dataframe
    """
    df = df.pipe(general_cleaning).pipe(rename.stand_stats_col)

    float_cols = get_float_cols(df, ["90", "xg", "xag"])
    int_cols = get_int_cols(df, float_cols)

    return (
        df.pipe(float_dtypes, columns=float_cols)
        .pipe(int_types, columns=int_cols)
        .pipe(categorical_dtypes)
    )


# ----------------------- Main transformation function ----------------------- #


def player_stats(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Main transformation function for fbref.com player stats data.

    Args:
        df (pd.DataFrame): player stats dataframe
        table (str): stats table to transform

    Raises:
        ValueError: if table is not found

    Returns:
        pd.DataFrame: transformed dataframe
    """
    if table == "defense":
        df = defense(df)
    elif table == "gca":
        df = gca(df)
    elif table == "keeper":
        df = keeper(df)
    elif table == "keeper_adv":
        df = ad_keeper(df)
    elif table == "misc":
        df = misc(df)
    elif table == "passing":
        df = passing(df)
    elif table == "passing_types":
        df = passing_types(df)
    elif table == "playing_time":
        df = playing_time(df)
    elif table == "possession":
        df = possession(df)
    elif table == "shooting":
        df = shooting(df)
    elif table == "standard":
        df = standard_stats(df)
    else:
        raise ValueError(f"Table: {table} not found")
    return df


# ----------------------------------- wages ---------------------------------- #


def wages(df: pd.DataFrame, comp: str) -> pd.DataFrame:
    """Main transformation function for fbref.com player wages data.

    Args:
        df (pd.DataFrame): player wages dataframe

    Returns:
        pd.DataFrame: transformed dataframe
    """
    df = df.pipe(wage_gen_cleaning, comp=comp)

    int_cols = [df.columns[0]] + [col for col in df.columns[5:8]]

    return df.pipe(int_types, columns=int_cols).pipe(categorical_dtypes)
