from pydantic import BaseModel


class Dropdowns(BaseModel):
    positions: list[str]
    leagues: list[str]
    countries: list[str]
