import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from analysis.base.cross_val import CrossValidator
from analysis.utilities.logging import get_logger

logger = get_logger(__name__)


class CrossValidate(CrossValidator):
    def __init__(self, metric: str, method: KFold) -> None:
        super().__init__(metric=metric, method=method)
        self.train_scores = {}
        self.test_scores = {}

    def validate(
        self,
        models: list[Pipeline | RegressorMixin],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        results_list = [self._cv_results(model, X, y) for model in models]

        logger.info("Concatenating results")
        self.results = pd.concat(results_list).reset_index(drop=True)

    def _cv_results(
        self, model: RegressorMixin | Pipeline, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        logger.info(f"Training model: {model}")
        cv = cross_validate(
            model,
            X,
            y,
            cv=self.method,
            scoring=self.metric,
            n_jobs=-1,
            return_train_score=True,
        )

        model_name = (
            model["model"].__class__.__name__
            if isinstance(model, Pipeline)
            else model.__class__.__name__
        )

        cv_results = dict(
            model=model_name,
            parameters=str(model.get_params()),
            mean_train_score=-cv["train_score"].mean(),
            mean_test_score=-cv["test_score"].mean(),
            test_score_std=cv["test_score"].std(),
            fit_time=cv["fit_time"].mean(),
        )
        self.train_scores[model_name] = -cv["train_score"]
        self.test_scores[model_name] = -cv["test_score"]

        logger.info(f"Model: {model} - \nCV results: {cv_results}")

        return pd.DataFrame(cv_results, index=[0])

    @property
    def cv_results_(self) -> pd.DataFrame:
        return self.results

    @property
    def train_scores_(self) -> dict[str, list[float]]:
        return self.train_scores

    @property
    def test_scores_(self) -> dict[str, list[float]]:
        return self.test_scores
