from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import time

import pandas as pd
from rich import print

from feature_pipeline.utils import get_logger
from .validation import PlayerHistory
from feature_pipeline.apis import players_api, get_player_ids, BASE
from .transform import transform

logger = get_logger(__name__)


def unpack_stats(player_id: str) -> list[PlayerHistory]:
    resp = players_api(player_id)
    if not resp:
        logger.info("No player stats found in JSON response.")
    return [PlayerHistory(**stat).model_dump() for stat in resp]


def from_api() -> tuple[pd.DataFrame, dict[str, str]]:
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
    root = Path(__file__).parent.parent
    path = f"{root}/data/raw/player_data.pkl"
    
    df = pd.read_pickle(path)
    
    metadata = {
        "description": "Individual player data for each game of the 2023/24 season",
        "url": f"{BASE}/element-summary/<PLAYER_ID>/",
        'file_path': path,
        "datetime_format": "%Y-%m-%dT%H:%M:%SZ",
    }
    return df, metadata


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    path = f"{root}/data/raw/player_data.pkl"
    
    df = pd.read_pickle(path)
    df = transform(df)
    df.columns
    # df, metadata = extract()

    # df.to_pickle(path)
