import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from feature_pipeline.scraper.response import alt_get_url_data as get_url_data
from feature_pipeline.utilities.utils import get_logger

logger = get_logger(__name__)


def player_name(names: list[str], index: int) -> str:
    """Parse player name from html.

    Args:
        names (list[str]): player names html
        index (int): index position of player name

    Returns:
        str: player name
    """
    return str(names[index]).split('" class', 1)[0].split('<img alt="', 1)[1]


def extract_age(text: str) -> str:
    """Extract player age from html.

    Args:
        text (str): html text

    Returns:
        str: player age
    """
    return text.split("(", 1)[1].split(")", 1)[0]


def extract_team_signed_from(stat: str) -> str:
    """Extract team signed from from html.

    Args:
        stat (str): html text

    Returns:
        str: team signed from
    """
    return stat.a["title"].split(": ")[0] if stat.a else "NA"


def extract_signing_fee(stat: str) -> str:
    """Extract signing fee from html.

    Args:
        stat (str): html text

    Returns:
        str: signing fee
    """
    return stat.a["title"].split(": ")[1] if stat.a else "0"


def transfermarkt_player_info(link: str, index: int) -> str | None:
    """Extract player name and id from transfermarkt link in html text.

    Args:
        link (str): link in html
        index (int): index position of player name and id in split list

    Returns:
        str | None: player name or id or None
    """
    return (
        link.a["href"].split("/")[index] if link.a and "href" in link.a.attrs else None
    )


def html_handler(
    soup: BeautifulSoup,
) -> tuple[ResultSet, ResultSet, ResultSet, ResultSet, ResultSet]:
    """Handle html parsing for player market values.

    Args:
        soup (BeautifulSoup): BeautifulSoup object

    Returns:
        tuple[ResultSet, ResultSet, ResultSet]: player names, stats and values
    """
    names = soup.find_all("img", {"class": "bilderrahmen-fixed lazy lazy"})
    stats = soup.find_all("td", {"class": "zentriert"})
    values = soup.find_all("td", {"class": "rechts hauptlink"})
    positions = soup.find_all("td", {"class": "posrela"})
    links = soup.find_all("td", {"class": "hauptlink"})

    return names, stats, values, positions, links


def parse_stats(stats: BeautifulSoup, season: str, team: str):
    SQUAD_NUM_INDEX = 0
    AGE_INDEX = 1
    COUNTRY_INDEX = 2

    if season == "2023":
        HEIGHT_INDEX = 3
        FOOT_INDEX = 4
        SIGNED_INDEX = 5
        TEAM_SIGNED_FROM_INDEX = 6
        SIGNING_FEE_INDEX = 6
        CONTRACT_EXPIRY_INDEX = 7
    else:
        HEIGHT_INDEX = 4
        FOOT_INDEX = 5
        SIGNED_INDEX = 6
        TEAM_SIGNED_FROM_INDEX = 7
        SIGNING_FEE_INDEX = 7
        CONTRACT_EXPIRY_INDEX = 8

    squad_nums = [stat.text for stat in stats[SQUAD_NUM_INDEX::8]]
    ages = [extract_age(stat.text) for stat in stats[AGE_INDEX::8]]
    countries = [stat.img["title"] for stat in stats[COUNTRY_INDEX::8]]
    heights = [stat.text for stat in stats[HEIGHT_INDEX::8]]
    foot = [stat.text for stat in stats[FOOT_INDEX::8]]
    signed = [stat.text for stat in stats[SIGNED_INDEX::8]]
    team_signed_from = [
        extract_team_signed_from(stat) for stat in stats[TEAM_SIGNED_FROM_INDEX::8]
    ]
    signing_fee = [extract_signing_fee(stat) for stat in stats[SIGNING_FEE_INDEX::8]]

    if season == "2023":
        contract_expiry = [stat.text for stat in stats[CONTRACT_EXPIRY_INDEX::8]]
        current_club = team
    else:
        contract_expiry = "NA"
        current_club = [stat.img["title"] for stat in stats[3::8]]

    return (
        squad_nums,
        ages,
        countries,
        heights,
        foot,
        current_club,
        signed,
        team_signed_from,
        signing_fee,
        contract_expiry,
    )


def create_player_values_df(
    soup: BeautifulSoup, season: str, league: str, team: str
) -> pd.DataFrame:
    """Create pandas dataframe from scraped player valuation data.

    Args:
        soup (BeautifulSoup): BeautifulSoup object
        season (str): football season

    Returns:
        pd.DataFrame: pandas dataframe of scraped player valuation data
    """
    # load in beautifulsoup objects
    names, stats, values, positions, links = html_handler(soup)

    # parse player names
    player_names = [player_name(names, i) for i in range(0, len(names))]

    # parse player positions
    positions = ["".join(pos.text.strip().split(" ")[-2:]) for pos in positions]

    # parse player market values
    market_values = [value.text.split("/")[0].strip() for value in values]

    # parse player transfermarkt names and ids
    tm_names = [transfermarkt_player_info(link, 1) for link in links[::2]]
    tm_ids = [transfermarkt_player_info(link, 4) for link in links[::2]]

    # parse player stats
    (
        squad_num,
        age,
        countries,
        height,
        foot,
        current_clubs,
        signed,
        team_signed_from,
        signing_fee,
        contract_expiry,
    ) = parse_stats(stats=stats, season=season, team=team)

    return pd.DataFrame(
        {
            "tm_id": tm_ids,
            "tm_name": tm_names,
            "player": player_names,
            "squad_num": squad_num,
            "position": positions,
            "age": age,
            "country": countries,
            "current_club": current_clubs,
            "height": height,
            "foot": foot,
            "signed_date": signed,
            "signed_from": team_signed_from,
            "signing_fee": signing_fee,
            "contract_expiry": contract_expiry,
            "market_value": market_values,
            "season": season,
            "league": league,
            "team": team,
        }
    )


def create_team_info_df(soup: BeautifulSoup, season: str, index: int) -> pd.DataFrame:
    """Create pandas dataframe from scraped team information data.

    Args:
        soup (BeautifulSoup): BeautifulSoup object
        season (str): football season
        index (int): limit length of scraped data

    Returns:
        pd.DataFrame: pandas dataframe of scraped team information data
    """
    names = soup.find_all("td", {"class": "hauptlink no-border-links"})
    stats = soup.find_all("td", {"class": "zentriert"})
    values = soup.find_all("td", {"class": "rechts"})

    teams = [stat.a["title"] for stat in names][:index]
    team_ids = [stat.a["href"].split("/")[4] for stat in names][:index]
    other_names = [stat.a["href"].split("/")[1] for stat in names][:index]
    squad_size = [stat.text for stat in stats[4::4]][:index]
    squad_avg_ages = [stat.text for stat in stats[5::4]][:index]
    squad_foreigners = [stat.text for stat in stats[6::4]][:index]
    average_value = [stat.text for stat in values[2::2]][:index]
    total_value = [stat.text for stat in values[3::2]][:index]

    return pd.DataFrame(
        {
            "team_id": team_ids,
            "team": teams,
            "other_names": other_names,
            "squad_size": squad_size,
            "squad_avg_age": squad_avg_ages,
            "squad_foreigners": squad_foreigners,
            "average_value": average_value,
            "total_value": total_value,
            "season": season,
        }
    )


# ------------------------------ Main Parsing functions ------------------------------ #


def player_market_values(url: str, season: str, league: str, team: str) -> pd.DataFrame:
    """Scrape player market values from transfermarkt.com for a specific team
    and season. Return results as a pandas dataframe.

    Args:
        url (str): team url
        season (str): football season
        league (str): league name
        team (str): team name

    Returns:
        pd.DataFrame: pandas dataframe of scraped player market values
    """
    logger.info(f"Scraping season {season} player market values for {url}")
    resp = get_url_data(url)

    logger.info("Parsing html")
    page_soup = BeautifulSoup(resp.content, "html.parser")

    logger.info("Creating market valuations dataframe")
    return create_player_values_df(page_soup, season, league, team)


def team_information(url: str, season: str, comp_id: str) -> pd.DataFrame:
    """Scrape team information for a league for a given season.
    Return results as a pandas dataframe.

    Args:
        url (str): league url
        season (str): football season

    Returns:
        pd.DataFrame: pandas dataframe of scraped team information
    """
    logger.info(f"Scraping season {season} league information for {url}")
    resp = get_url_data(url)

    logger.info("Parsing html")
    soup = BeautifulSoup(resp.content, "html.parser")

    if comp_id == "L1":
        index = 18
    if comp_id == "FR1" and season == "2023":
        index = 18
    else:
        index = 20

    logger.info("Creating team information dataframe")
    return create_team_info_df(soup, season, index)
