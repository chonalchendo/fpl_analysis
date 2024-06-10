from typing import Callable

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from analysis.base.compose import BaseComposer
from analysis.base.data_loader import DataLoader
from analysis.base.test import Tester
from analysis.src_forwards.training import train_valid_test_split
from analysis.utilities.logging import get_logger

logger = get_logger(__name__)


class ModelTester:
    def __init__(
        self,
        preprocessor: BaseComposer,
        sklearn_pipeline: Callable,
        models: list[RegressorMixin],
        tester: Tester,
        loader: DataLoader,
    ) -> None:
        self.preprocessor = preprocessor
        self.sklearn_pipeline = sklearn_pipeline
        self.models = models
        self.tester = tester
        self.loader = loader

    def run(self, input_path: str, target_variable: str) -> None:
        logger.info(f"Loading data from {input_path}")
        df = self.loader.load(input_path)

        logger.info("Splitting data into train and test sets")
        train_set, _, _, self.test_set = train_valid_test_split(df, season_split=2023)

        logger.info("Preprocessing training data")
        train_set_ = self.preprocessor.compose(train_set)

        logger.info("Preprocessing test data")
        self.test_set_ = self.preprocessor.compose(self.test_set)

        logger.info("Creating X_train, y_train, X_test, y_test")
        X_train, y_train = (
            train_set_.drop(columns=[target_variable]),
            train_set_[target_variable],
        )
        X_test, y_test = (
            self.test_set_.drop(columns=[target_variable]),
            self.test_set_[target_variable],
        )

        logger.info("Building model pipelines")
        self.pipelines_ = [self.sklearn_pipeline(model) for model in self.models]

        logger.info("Testing models")
        self.tester.test(
            models=self.pipelines_,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        logger.info("Blending models")
        self.tester.blend(y_test=y_test)

    @property
    def pred_results_(self) -> pd.DataFrame:
        players_df = self.test_set[
            ["player", "position", "comp", "squad", "country", "market_value_euro_mill"]
        ]
        self.tester.pred_results_.index = self.test_set_.index
        return players_df.join(self.tester.pred_results_)

    @property
    def performance_(self) -> pd.DataFrame:
        return self.tester.performance_
