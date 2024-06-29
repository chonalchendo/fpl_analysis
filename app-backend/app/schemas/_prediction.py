from pydantic import BaseModel


class Prediction(BaseModel):
    player: str
    position: str
    league: str
    team: str
    country: str
    market_value: float
    prediction: float


class League(Prediction):
    league: str


class Country(Prediction):
    country: str


class Team(Prediction):
    team: str


class Position(Prediction):
    position: str
