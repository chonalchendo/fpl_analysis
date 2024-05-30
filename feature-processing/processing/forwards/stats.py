import pandas as pd
from rich import print

from processing.gcp.loader import GCPLoader
from processing.pipeline.join import Datajoiner
from processing.processors.utils.cleaners import ColumnFilter
from processing.processors.utils.joiners import MultiJoin


def striker_stats() -> pd.DataFrame:
    joiner = Datajoiner(
        processors=[
            ColumnFilter(not_like="_x"),
        ],
        join_method=MultiJoin(on=["player_id", "season", "squad"], suffixes=("", "_x")),
        loader=GCPLoader(),
    )

    return joiner.process(
        "processed_fbref_db",
        [
            "processed_shooting.csv",
            "processed_gca.csv",
            "processed_passing.csv",
            "processed_passing_types.csv",
        ],
    )


if __name__ == "__main__":
    striker_stats = striker_stats()
    print(striker_stats.head())
