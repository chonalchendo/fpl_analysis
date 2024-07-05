import httpx
import pandas as pd
from core.settings import API_URL
from models import Query


def query_api(query: Query) -> pd.DataFrame:
    if query is not None:
        params = query.to_dict()

        response = httpx.get(
            f"{API_URL}/value_prediction/predict",
            params=params,
        ).json()

        if response:
            return pd.DataFrame(response)
