import time

import pandas as pd

from feature_pipeline.core.settings import SOURCE
from feature_pipeline.utilities.utils import get_logger
from .validation import PlayerHistory
from feature_pipeline.apis.fpl import players_api, get_player_ids, BASE

logger = get_logger(__name__)


def unpack_stats(player_id: str) -> list[PlayerHistory]:
    """Unpack player stats from API response.

    Args:
        player_id (str): player id

    Returns:
        list[PlayerHistory]: list of PlayerHistory objects
    """
    resp = players_api(player_id)
    if not resp:
        logger.info("No player stats found in JSON response.")
    return [PlayerHistory(**stat).model_dump() for stat in resp]


def from_api() -> tuple[pd.DataFrame, dict[str, str]]:
    """Load player data from API.

    Returns:
        tuple[pd.DataFrame, dict[str, str]]: player dataframe and metadata
    """
    player_ids = get_player_ids()
    ids = [player["id"] for player in player_ids]

    player_data = []
    for id in sorted(ids):
        data = unpack_stats(str(id))
        logger.info(f"Data for player id {id} pulled from API")
        player_data.extend(data)
        time.sleep(1)

    df = pd.DataFrame(player_data)
    logger.info("Data has been loaded in a dataframe")

    metadata = {
        "description": "Individual player data for each game of the 2023/24 season",
        "url": f"{BASE}/element-summary/<PLAYER_ID>/",
        "datetime_format": "%Y-%m-%dT%H:%M:%SZ",
    }
    return df, metadata


def from_file() -> tuple[pd.DataFrame, dict[str, str]]:
    """Load player data from file.

    Returns:
        pd.DataFrame: player dataframe
    """
    path = f"{SOURCE}/data/fpl/player_data.pkl"

    df = pd.read_pickle(path)

    metadata = {
        "description": "Individual player data for each game of the 2023/24 season",
        "url": f"{BASE}/element-summary/<PLAYER_ID>/",
        "file_path": path,
        "datetime_format": "%Y-%m-%dT%H:%M:%SZ",
    }
    return df, metadata
