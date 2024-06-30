from dataclasses import asdict, dataclass


@dataclass
class Query:
    country: str | None
    league: str | None
    position: str | None

    def to_dict(self) -> dict[str, str]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def is_empty(self) -> bool:
        return all(value is None for value in self.__dict__.values())
