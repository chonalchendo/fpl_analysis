from app.gcp.files import gcs
from app.schemas import Dropdowns


async def get_dropdowns() -> Dropdowns:
    data = gcs.read_parquet("values_predictions/attacking_predictions.parquet")
    return Dropdowns(
        leagues=data["comp"].unique().tolist(),
        positions=data["position"].unique().tolist(),
        countries=data["country"].unique().tolist(),
    )
