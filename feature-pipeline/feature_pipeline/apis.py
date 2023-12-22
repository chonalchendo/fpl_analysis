import httpx
import pandas as pd
from httpx import Response
from feature_pipeline.etl.validation import PlayerInfo
from feature_pipeline.utils import get_logger

BASE = "https://fantasy.premierleague.com/api"
TEAMS = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"

logger = get_logger(__name__)


def api_handler(endpoint: str) -> Response:
    """Handles API requests.

    Args:
        endpoint (str): endpoint to fantasy premier league API.

    Returns:
        Response: Response object.
    """
    try:
        r = httpx.get(f"{BASE}/{endpoint}/")
        r.raise_for_status()
        return r.json()
    except httpx.RequestError as req_err:
        logger.error(f"HTTP Request Error: {req_err}")


def get_player_ids() -> list[dict[str, int | str]] | None:
    """Get player ids from the API.

    Returns:
        list[dict[str, int | str]] | None: List of player ids.
    """
    try:
        endpoint = "bootstrap-static"
        data = api_handler(endpoint)["elements"]
        return [PlayerInfo(**stat).model_dump() for stat in data]
    except httpx.RequestError as req_err:
        logger.error(f"HTTP Request Error: {req_err}")


def players_api(player_id: str) -> Response | None:
    """Get player data from the API.

    Args:
        player_id (str): player id.

    Returns:
        Response | None: Response object.
    """
    try:
        endpoint = "element-summary"
        r = httpx.get(f"{BASE}/{endpoint}/{player_id}/")
        r.raise_for_status()
        return r.json()["history"]
    except httpx.RequestError as req_err:
        logger.error(f"HTTP Request Error: {req_err}")


def get_fixtures_data(endpoint: str = "fixtures") -> pd.DataFrame:
    """
    Get fixtures data from the API.
    Returns:
        Dataframe with fixtures data.
    """
    fixtures = api_handler(endpoint)
    return pd.DataFrame(fixtures)


def get_teams_data(url: str = "bootstrap-static") -> pd.DataFrame:
    """Get teams data from the API.

    Args:
        url (str, optional): api url. Defaults to TEAMS.

    Returns:
        pd.DataFrame: Dataframe with teams data.
    """
    teams = api_handler(url)
    return pd.DataFrame(teams["teams"])


def map_team_stats(col: str) -> dict[int, str]:
    teams = get_teams_data()
    # return {k["id"]: k[col] for k in teams}
    return dict(zip(teams["id"], teams[col]))
