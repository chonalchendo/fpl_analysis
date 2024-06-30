import httpx
from core.settings import API_URL
from models import Dropdown


def get_dropdowns() -> Dropdown:
    url = f"{API_URL}/dropdowns/get"

    # get data for dropdowns
    response = httpx.get(url).json()
    countries = response.get("countries", [])
    leagues = response.get("leagues", [])
    positions = response.get("positions", [])

    return Dropdown(countries=countries, leagues=leagues, positions=positions)
