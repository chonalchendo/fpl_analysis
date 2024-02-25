import pandas as pd
import pycountry_convert as pc
import pycountry
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup


def get_fifa_country_codes() -> dict[str, str]:
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


@dataclass
class Cleaning:
    _data: pd.DataFrame

    def create_gen_pos_col(self) -> None:
        """Creates a general position column from the position column."""

        def get_general_position(pos: str) -> str:
            """Returns the general position of a player.

            Args:
                pos (str): position of player

            Returns:
                str: general position of player
            """
            if isinstance(pos, float):
                return "Unknown"
            if pos.startswith("D"):
                position = "Defender"
            elif pos.startswith("M"):
                position = "Midfielder"
            elif pos.startswith("F"):
                position = "Forward"
            elif pos.startswith("G"):
                position = "Goalkeeper"
            else:
                position = "Unknown"
            return position

        self._data["general_pos"] = self._data["pos"].apply(get_general_position)

    def create_age_range_col(self) -> None:
        """Creates an age range column from the age column."""

        def get_age_range(age: int) -> str:
            """Returns the age range of a player.

            Args:
                age (int): age of player

            Returns:
                str: age range of player
            """
            if age < 20:
                range = "Under 20"
            elif age < 25:
                range = "20-24"
            elif age < 30:
                range = "25-29"
            elif age < 35:
                range = "30-34"
            elif age < 40:
                range = "35-39"
            else:
                range = "Over 40"
            return range

        self._data["age_range"] = self._data["age"].apply(get_age_range)

    def create_country_column(self):
        """Creates a country column from the nation column."""
        fifa_codes = get_fifa_country_codes()
        self._data["country"] = self._data["nation"].map(fifa_codes)

    def create_continent_col(self) -> None:
        """Creates a continent column from the nation column."""

        def get_continent(name: str) -> str:
            """Returns the continent of a country.

            Args:
                alpha3 (str): alpha3 code of country

            Returns:
                str: continent of country
            """
            if isinstance(name, float):
                return "Unknown"

            if "congo" in name.lower():
                name = "congo"
            if "ivory" in name.lower():
                name = "Côte d'Ivoire"
            if "ireland" in name.lower():
                name = "ireland"
            if "verde" in name.lower():
                name = "verde"
            if "turkey" in name.lower():
                name = "turkiye"

            alpha2 = pycountry.countries.search_fuzzy(name)[0].alpha_2
            continent_code = pc.country_alpha2_to_continent_code(alpha2)
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
            return continent_name

        self._data["continent"] = self._data["country"].apply(get_continent)

    def pipeline(self) -> pd.DataFrame:
        """Runs all cleaning methods.

        Returns:
            pd.DataFrame: cleaned dataframe
        """
        self.create_gen_pos_col()
        self.create_age_range_col()
        self.create_country_column()
        self.create_continent_col()
        return self._data


@dataclass
class CleanPlayerVals:
    """Class to clean transfermarkt player values data."""

    _data: pd.DataFrame

    def create_signed_year_col(self) -> None:
        """Creates a signed year column from the signed date column."""
        self._data.loc[:, "signed_year"] = (
            self._data["signed_date"].str.split(" ").str[2].astype("Int32")
        )

    def impute_height_col(self) -> None:
        """Imputes missing height values with the median height."""
        self._data.loc[
            (self._data["height"] == 0) | (self._data["height"].isna()), "height"
        ] = self._data["height"].median()

    def pipeline(self) -> pd.DataFrame:
        """Runs all cleaning methods.

        Returns:
            pd.DataFrame: cleaned dataframe
        """
        self.create_signed_year_col()
        self.impute_height_col()
        return self._data
