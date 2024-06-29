from fastapi import APIRouter

from app.schemas import League, Prediction, Team
from app.services.predictions import by_league, by_player, by_team, get_predictions

router = APIRouter()


@router.get(
    "/player",
    summary="Predict player valuation",
    description="Predict player market value based on player season statistics",
    response_model=Prediction,
)
async def player_prediction(player: str) -> Prediction:
    return await by_player(player)


@router.get(
    "/league",
    summary="Predict league valuation",
    description="Predict league market value based on player season statistics",
    response_model=list[League],
)
async def league_prediction(league: str, limit: int) -> list[League]:
    return await by_league(league, limit)


@router.get(
    "/team",
    summary="Predict team valuation",
    description="Predict team market value based on player season statistics",
    response_model=list[Team],
)
async def team_prediction(team: str) -> list[Team]:
    return await by_team(team)


@router.get(
    "/predict",
    summary="Get predictions",
    description="Get predictions based on league, position, and age",
)
async def predictions(
    league: str | None = None,
    position: str | None = None,
    country: str | None = None,
    limit: int = 10,
):
    return await get_predictions(league, position, country, limit)
