import pandas as pd
import numpy as np
import re
from functools import reduce
from typing import Callable


Preprocessor = Callable[[pd.DataFrame], pd.DataFrame]


def compose(*functions: Preprocessor) -> Preprocessor:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


# ------------------------- column cleaning functions ------------------------ #


def clean_value(value: str) -> str:
    value = re.sub("€", "", value)
    if value.endswith("k"):
        new_val = "0." + re.sub("k", "", value)
    elif value.endswith("bn"):
        new_val = re.sub("bn", "", value)
        new_val = float(new_val) * 1000
    else:
        new_val = re.sub("m", "", value)
    return str(new_val)


def change_position(position: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1-\2", position)


# ------------------------------ Main functions ------------------------------ #


def clean_market_values(df: pd.DataFrame) -> pd.DataFrame:
    df["market_value"] = df["market_value"].apply(clean_value).replace("-", np.nan)
    df = df.rename(columns={"market_value": "market_value_euro_mill"})
    return df


def clean_squad_numbers(df: pd.DataFrame) -> pd.DataFrame:
    df["squad_num"] = df["squad_num"].replace("-", "0")
    return df


def clean_signing_fee(df: pd.DataFrame) -> pd.DataFrame:
    df["signing_fee"] = (
        df["signing_fee"]
        .replace("Ablöse ", "", regex=True)
        .apply(clean_value)
        .replace("-|\?", np.nan, regex=True)
        .replace("free transfer|draft", "0", regex=True)
        .replace("", np.nan)
    )
    df = df.rename(columns={"signing_fee": "signing_fee_euro_mill"})
    return df


def clean_contract_expiry(df: pd.DataFrame) -> pd.DataFrame:
    df["contract_expiry"] = df["contract_expiry"].replace("-|NA", np.nan, regex=True)
    return df


def clean_height(df: pd.DataFrame) -> pd.DataFrame:
    df["height"] = df["height"].replace("m|,", "", regex=True).replace("-", "0")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    return df


def clean_position(df: pd.DataFrame) -> pd.DataFrame:
    df["position"] = df["position"].apply(change_position)
    df.loc[df["position"] == "Mittelfeld", "position"] = "Central-Midfield"
    return df


def handle_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    dtypes = {
        "market_value_euro_mill": float,
        "signing_fee_euro_mill": float,
        "contract_expiry": "category",
        "height": float,
        "squad_num": int,
        "country": "category",
        "foot": "category",
        "position": "category",
        "tm_id": int,
        "tm_name": "category",
        "team": "category",
        "season": "category",
    }
    df = df.astype(dtypes)
    return df


# ----------------------------- clean league data ---------------------------- #


def clean_squad_values(df: pd.DataFrame) -> pd.DataFrame:
    df["average_value"] = df["average_value"].apply(clean_value)
    df["total_value"] = df["total_value"].apply(clean_value)

    rename_cols = {
        "average_value": "average_value_euro_mill",
        "total_value": "total_value_euro_mill",
    }
    df = df.rename(columns=rename_cols)
    return df


def handle_league_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    dtypes = {
        "average_value_euro_mill": float,
        "total_value_euro_mill": float,
        "squad_size": int,
        "squad_avg_age": float,
        "squad_foreigners": int,
        "team_id": int,
        "team": "category",
        "other_names": "category",
        "season": "category",
    }
    df = df.astype(dtypes)
    return df


# ------------------------------ Main functions ------------------------------ #


def market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms player market data.

    Args:
        df (pd.DataFrame): pandas dataframe containing player market data.

    Returns:
        pd.DataFrame: transformed dataframe.
    """
    preprocessor = compose(
        clean_squad_numbers,
        clean_market_values,
        clean_signing_fee,
        clean_contract_expiry,
        clean_height,
        clean_position,
        handle_dtypes,
    )
    return preprocessor(df)


def league_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms league data.

    Args:
        df (pd.DataFrame): pandas dataframe containing league data.

    Returns:
        pd.DataFrame: transformed dataframe.
    """
    preprocessor = compose(clean_squad_values, handle_league_dtypes)
    return preprocessor(df)
