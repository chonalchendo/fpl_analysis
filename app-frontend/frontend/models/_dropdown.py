from dataclasses import dataclass


@dataclass
class Dropdown:
    countries: list[str]
    leagues: list[str]
    positions: list[str]
