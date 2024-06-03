import pandas as pd


def create_youth_player_col(x: pd.Series) -> bool:
    if pd.isna(x["squad"]) | pd.isna(x["signed_from"]):
        return False

    squad_words = x["squad"].split()  # Split squad name into words

    for word in squad_words:
        if word in x["signed_from"]:
            return True
        else:
            return False


def create_penalty_taker_col(x: pd.Series) -> bool:
    if x["penalty_kicks_attempted"] > 0:
        return True
    else:
        return False


def create_years_since_signed_col(x: pd.Series) -> int:
    return x["season"] - x["signed_year"]


def create_league_signed_from_col(x: pd.Series) -> bool:
    if pd.isna(x["squad"]) or pd.isna(x["signed_from"]):
        return 'False'

    if x["is_youth_player"] == "True":
        return "youth_signing"

    signed_from_words = str(x["signed_from"]).split()  # Split squad name into words

    for word in signed_from_words:
        if word in [
            "Arsenal",
            "Chelsea",
            "Manchester Utd",
            "Manchester City",
            "Southampton",
            "Liverpool",
            "West Brom",
            "Crystal Palace",
            "Everton",
            "West Ham",
            "Tottenham",
            "Leicester City",
            "Swansea City",
            "Watford",
            "Stoke City",
            "Bournemouth",
            "Huddersfield",
            "Burnley",
            "Newcastle Utd",
            "Brighton",
            "Fulham",
            "Wolves",
            "Cardiff City",
            "Aston Villa",
            "Sheffield Utd",
            "Norwich City",
            "Leeds United",
            "Brentford",
            "Nott'ham Forest",
        ]:
            return "premier_league"
        elif word in [
            "Bayern Munich",
            "Dortmund",
            "Wolfsburg",
            "Schalke 04",
            "Leverkusen",
            "Hoffenheim",
            "RB Leipzig",
            "Hamburger SV",
            "Werder Bremen",
            "Köln",
            "Hertha BSC",
            "Eint Frankfurt",
            "Stuttgart",
            "Augsburg",
            "Freiburg",
            "Mainz 05",
            "Düsseldorf",
            "Nürnberg",
            "Union Berlin",
            "Paderborn 07",
            "Arminia",
            "Bochum",
            "Greuther Fürth",
        ]:
            return "bundesliga"
        elif word in [
            "Barcelona",
            "Real Madrid",
            "Atlético Madrid",
            "Levante",
            "Sevilla",
            "Valencia",
            "Villarreal",
            "Athletic Club",
            "Las Palmas",
            "Espanyol",
            "Real Sociedad",
            "Celta Vigo",
            "Girona",
            "La Coruña",
            "Getafe",
            "Eibar",
            "Alavés",
            "Leganés",
            "Rayo Vallecano",
            "Huesca",
            "Granada",
            "Osasuna",
            "Mallorca",
            "Elche",
            "Almería",
        ]:
            return "la_liga"

        elif word in [
            "Milan",
            "Juventus",
            "Inter",
            "Roma",
            "Napoli",
            "Lazio",
            "Genoa",
            "Bologna",
            "Atalanta",
            "Torino",
            "Benevento",
            "Fiorentina",
            "Cagliari",
            "Udinese",
            "Hellas Verona",
            "Chievo",
            "Crotone",
            "Parma",
            "Frosinone",
            "Empoli",
            "Brescia",
            "Lecce",
            "Spezia",
            "Venezia",
            "Sampdoria",
            "Monza",
            "Salernitana",
            "Cremonese",
        ]:
            return "serie_a"

        elif word in [
            "Paris S-G",
            "Monaco",
            "Marseille",
            "Saint-Étienne",
            "Amiens",
            "Lyon",
            "Toulouse",
            "Lille",
            "Rennes",
            "Dijon",
            "Bordeaux",
            "Nantes",
            "Strasbourg",
            "Metz",
            "Guingamp",
            "Montpellier",
            "Caen",
            "Angers",
            "Troyes",
            "Reims",
            "Brest",
            "Lens",
            "Lorient",
            "Nîmes",
            "Clermont Foot",
            "Auxerre",
            "Ajaccio",
        ]:
            return "ligue_1"
        else:
            return "outside_top_5"
