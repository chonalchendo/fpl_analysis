import pandas as pd

from app.core.config import get_settings
from app.schemas import Dropdowns

setttings = get_settings()


async def get_dropdowns() -> Dropdowns:
    # get data from gcpj
    # data = gcs.read_parquet("values_predictions/attacking_predictions.parquet")

    data = pd.read_parquet(setttings.PREDICTIONS)

    return Dropdowns(
        leagues=data["comp"].unique().tolist(),
        positions=data["position"].unique().tolist(),
        countries=data["country"].unique().tolist(),
    )
