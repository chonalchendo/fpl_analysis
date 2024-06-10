from typing import Callable

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from analysis.base.compose import BaseComposer
from analysis.base.cross_val import CrossValidator
from analysis.base.data_loader import DataLoader
from analysis.src_forwards.training._data_splitter import train_valid_test_split
from analysis.utilities.logging import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(
        self,
        preprocessor: BaseComposer,
        sklearn_pipeline: Callable,
        models: list[RegressorMixin],
        cv: CrossValidator,
        loader: DataLoader,
    ) -> None:
        self.preprocessor = preprocessor
        self.sklearn_pipeline = sklearn_pipeline
        self.models = models
        self.cv = cv
        self.loader = loader

    def run(self, input_path: str, target_variable: str) -> None:
        logger.info(f"Loading data from {input_path}")
        df = self.loader.load(input_path)

        logger.info("Splitting data into train, validation, and test sets")
        train_set, _, _, _ = train_valid_test_split(df, season_split=2023)

        logger.info("Preprocessing data")
        train_set_ = self.preprocessor.compose(train_set)

        logger.info("Extracting features and target variable")
        X, y = train_set_.drop(columns=[target_variable]), train_set_[target_variable]

        logger.info("Training models")
        self.pipelines_ = [self.sklearn_pipeline(model) for model in self.models]
        self.cv.validate(models=self.pipelines_, X=X, y=y)

    @property
    def cv_results_(self) -> pd.DataFrame:
        return self.cv.cv_results_

    @property
    def train_scores_(self) -> dict[str, list[float]]:
        return self.cv.train_scores_

    @property
    def test_scores_(self) -> dict[str, list[float]]:
        return self.cv.test_scores_
