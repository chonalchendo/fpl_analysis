import pandas as pd

from app.core.config import get_settings
from app.schemas import Prediction

settings = get_settings()


async def get_predictions(
    league: str | None, position: str | None, country: str | None, limit: int = 10
):
    # data = gcs.read_parquet("values_predictions/attacking_predictions.parquet")
    #
    data = pd.read_parquet(settings.PREDICTIONS)

    query = {}
    if league is not None:
        query["comp"] = league
    if position is not None:
        query["position"] = position
    if country is not None:
        query["country"] = country

    data = (
        data.loc[(data[list(query)] == pd.Series(query)).all(axis=1)]
        .sort_values(by="RandomForestRegressor", ascending=False)
        .head(limit)
    )
    return [
        Prediction(
            player=player,
            position=position,
            league=league,
            team=team,
            country=country,
            market_value=value,
            prediction=round(pred, 2),
        )
        for player, position, league, team, country, value, pred in zip(
            data["player"],
            data["position"],
            data["comp"],
            data["squad"],
            data["country"],
            data["market_value_euro_mill"],
            data["RandomForestRegressor"],
        )
    ]
