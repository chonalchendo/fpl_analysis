import pandas as pd

from app.gcp.files import gcs
from app.schemas import League, Prediction, Team


async def by_player(player: str) -> Prediction:
    data = gcs.read_parquet("values_predictions/attacking_predictions.parquet")
    pred = data.loc[data["player"] == player]["RandomForestRegressor"]
    return Prediction(player=player, prediction=pred.values[0])


async def by_league(league: str, limit: int) -> list[League]:
    data = gcs.read_parquet("values_predictions/attacking_predictions.parquet")
    df = (
        data.loc[data["comp"] == league]
        .sort_values("RandomForestRegressor", ascending=False)
        .head(limit)
    )
    return [
        League(player=player, league=league, prediction=round(pred, 2))
        for player, pred in zip(df["player"], df["RandomForestRegressor"])
    ]


async def by_team(team: str) -> list[Team]:
    data = gcs.read_parquet("values_predictions/attacking_predictions.parquet")
    df = data.loc[data["squad"] == team].sort_values(
        "RandomForestRegressor", ascending=False
    )
    return [
        Team(player=player, team=team, prediction=round(pred, 2))
        for player, pred in zip(df["player"], df["RandomForestRegressor"])
    ]


async def get_predictions(
    league: str | None, position: str | None, country: str | None, limit: int = 10
):
    data = gcs.read_parquet("values_predictions/attacking_predictions.parquet")

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
    # return data.to_dict(orient="records")
    #
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
