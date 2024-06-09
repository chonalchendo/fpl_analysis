from typing import Callable

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from analysis.base.compose import BaseComposer
from analysis.base.data_loader import DataLoader
from analysis.base.test import Tester
from analysis.utilities.logging import get_logger

logger = get_logger(__name__)


class ModelTester:
    def __init__(
        self,
        preprocessor: BaseComposer,
        sklearn_pipeline: Callable | Pipeline,
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

        logger.info("Preprocessing data")
        df_ = self.preprocessor.compose(df)

        logger.info("Extracting features and target variable")
        X, y = df_.drop(columns=[target_variable]), df_[target_variable]

        logger.info("Testing models")
        self.pipelines_ = [self.sklearn_pipeline(model) for model in self.models]

        logger.info("Testing models")
        self.tester.test(models=self.pipelines_, X=X, y=y)

        logger.info("Blending models")
        self.tester.blend()
