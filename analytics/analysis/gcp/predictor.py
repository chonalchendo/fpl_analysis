import json

import joblib
import numpy as np
from google.cloud import storage
from google.cloud.aiplatform.prediction.sklearn.predictor import SklearnPredictor


class ValuePredictor(SklearnPredictor):
    def __init__(self) -> None:
        return

    def load(self, artifacts_url: str) -> None:
        super().load(artifacts_url)

        with open("preprocessor.joblib", "rb") as f:
            self._preprocessor = joblib.load(f)

    def preprocess(self, prediction_input: np.ndarray) -> np.ndarray:
        y_ = self._preprocessor["target"].fit_transform(prediction_input).ravel()
