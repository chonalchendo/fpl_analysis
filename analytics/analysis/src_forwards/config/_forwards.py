from typing import Callable

from sklearn.model_selection import KFold

from analysis.gcp.loader import CSVLoader
from analysis.src_forwards.cross_validate import CrossValidate
from analysis.src_forwards.models import forwards_pipeline, regressors
from analysis.src_forwards.preprocessing.pipeline import forwards_preprocessor


class ForwardsTrain:
    INPUT = "wage_vals_stats/forwards.csv"
    LOADER = CSVLoader()
    CV = CrossValidate(
        metric="neg_mean_absolute_error",
        method=KFold(n_splits=5, shuffle=True, random_state=42),
    )
    PREPROCESSOR = forwards_preprocessor
    PIPELINE: Callable = forwards_pipeline
    MODELS = regressors.models
    TARGET = "market_value_euro_mill"
