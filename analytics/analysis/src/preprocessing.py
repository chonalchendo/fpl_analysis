import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def prepare_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the data for training and testing.

    Args:
        data (pd.DataFrame): raw data - loaded from Postresql database

    Returns:
        tuple[np.ndarray, np.ndarray]: training set and testing set
    """
    columns_to_drop = [
        "index",
        "player_id",
        "player",
        "opponent_team_id",
        "kickoff_time_utc",
        "home_team_score",
        "away_team_score",
        "fixture_id",
        "gameweek",
        "season",
        "last_updated",
        "opponent_strength_attack",
        "opponent_strength_defence",
        "opponent_strength_overall",
        "transfers_balance",
        "transfers_in",
        "transfers_out",
        "starts",
    ]
    data = data.drop(columns=columns_to_drop)

    # split the data for training and testing
    train_set, test_set = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["team_difficulty"]
    )
    return train_set, test_set


def split_x_y_vars(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate the target variable and exploratory variables.

    Args:
        train_set (np.ndarray): training set

    Returns:
        tuple[np.ndarray, np.ndarray]: X variables and y variable
    """
    # separate the target variable and exploratory variables
    X = data.drop(columns=["total_points"], axis=1)
    y = data["total_points"].copy()
    return X, y


def X_transformation_pipeline(X: np.ndarray) -> ColumnTransformer:
    """Transformation pipeline for X train variables.

    Args:
        X (np.ndarray): explanatory variables

    Returns:
        ColumnTransformer: transformation pipeline
    """
    # transformation pipeline
    num_pipeline = make_pipeline(StandardScaler())
    cat_pipeline = make_pipeline(OneHotEncoder())
    ord_pipeline = make_pipeline(OrdinalEncoder())

    num_cols = list(
        X.select_dtypes(include=["number"]).drop(columns=["team_difficulty"])
    )
    cat_cols = list(X.select_dtypes(include=["object"]))
    ord_cols = ["team_difficulty"]

    preprocess_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
            ("ord", ord_pipeline, ord_cols),
        ]
    )
    return preprocess_pipeline
