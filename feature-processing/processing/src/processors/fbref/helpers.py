import pandas as pd
import pycountry
import pycountry_convert as pc
import requests
from bs4 import BeautifulSoup


def get_fifa_codes() -> dict[str, str]:
    """Returns a dictionary of FIFA country codes and country names.

    Returns:
        dict[str, str]: dictionary of FIFA country codes and country names
    """
    fifa_country_codes_url = "https://en.wikipedia.org/wiki/List_of_FIFA_country_codes"
    response = requests.get(fifa_country_codes_url)
    soup = BeautifulSoup(response.text, "html.parser")
    codes_table = soup.find_all("table", {"class": "wikitable"})
    df = pd.read_html(str(codes_table))
    fifa_codes = pd.concat(df[:4])
    return dict(zip(fifa_codes["Code"], fifa_codes["Country"]))


def get_position(pos: str) -> str:
    """Returns the general position of a player.

    Args:
        pos (str): position of player

    Returns:
        str: general position of player
    """
    match pos:
        case pos if isinstance(pos, float):
            position = "Unknown"
        case pos if pos.startswith("D"):
            position = "Defender"
        case pos if pos.startswith("M"):
            position = "Midfielder"
        case pos if pos.startswith("F"):
            position = "Forward"
        case pos if pos.startswith("G"):
            position = "Goalkeeper"
        case _:
            position = "Unknown"
    return position


def get_age_range(age: int) -> str:
    """Returns the age range of a player.

    Args:
        age (int): age of player

    Returns:
        str: age range of player
    """
    match age:
        case age if age < 20:
            range = "Under 20"
        case age if age < 25:
            range = "20-24"
        case age if age < 30:
            range = "25-29"
        case age if age < 35:
            range = "30-34"
        case age if age < 40:
            range = "35-39"
        case _:
            range = "Over 40"
    return range


def get_continent(name: str) -> str:
    """Returns the continent of a country.

    Args:
        alpha3 (str): alpha3 code of country

    Returns:
        str: continent of country
    """
    match name:
        case name if isinstance(name, float):
            return "Unknown"
        case name if "congo" in name.lower():
            name = "congo"
        case name if "ivory" in name.lower():
            name = "Côte d'Ivoire"
        case name if "ireland" in name.lower():
            name = "ireland"
        case name if "verde" in name.lower():
            name = "verde"
        case name if "turkey" in name.lower():
            name = "turkiye"

    alpha2 = pycountry.countries.search_fuzzy(name)[0].alpha_2
    continent_code = pc.country_alpha2_to_continent_code(alpha2)
    continent_name = pc.convert_continent_code_to_continent_name(continent_code)
    return continent_name
