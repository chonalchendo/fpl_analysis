from typing import Literal

from rich import print

from processing.gcp.loader import CSVLoader
from processing.gcp.saver import save
from processing.src.pipeline.join import ValueWage
from processing.src.processors.utils import cleaners
from processing.src.processors.utils.joiners import MultiJoin


def _join(
    val_blob: str, wage_blob: str, save_file: Literal["yes", "no"] = "no"
) -> None:
    saver = save(save_file)
    join = ValueWage(
        processors=[cleaners.Filter(not_like="_val")],
        loader=CSVLoader(),
        saver=saver,
        join_method=MultiJoin(
            on=["player", "season", "squad"], how="inner", suffixes=("_val", "")
        ),
    )
    df = join.process(val_blob=val_blob, wage_blob=wage_blob)
    print(df)


def main() -> None:
    wage_leagues = ["Premier-League", "Bundesliga", "La-Liga", "Serie-A", "Ligue-1"]
    value_leagues = ["premier_league", "bundesliga", "la_liga", "serie_a", "ligue_1"]

    for wage, value in zip(wage_leagues, value_leagues):
        _join(value, wage, save_file="yes")


if __name__ == "__main__":
    main()
