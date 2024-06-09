import pandas as pd


def youth_player(x: pd.Series) -> str:
    if pd.isna(x["squad"]) or pd.isna(x["signed_from"]):
        return "False"

    squad_words = x["squad"].split()  # Split squad name into words

    for word in squad_words:
        if word in x["signed_from"]:
            return "True"
        else:
            return "False"
