from pydantic import BaseModel


class Prediction(BaseModel):
    player: str
    position: str
    league: str
    team: str
    country: str
    market_value: float
    prediction: float
