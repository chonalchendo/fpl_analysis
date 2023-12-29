from typing import Any
import numpy as np
from sklearn.compose import ColumnTransformer
from analysis.gcp.storage import gcp


def save(
    model: Any,
    predictions: np.ndarray,
    mean_cv_score: np.ndarray,
    X: np.ndarray,
    X_transformer: ColumnTransformer,
    y: np.ndarray,
    file_name: str,
) -> None:
    """Save the model to the gcp bucket.

    Args:
        model (Any): final model
        predictions (np.ndarray): final model predictions on test set
        mean_cv_score (np.ndarray): mean cross validation score
        X (np.ndarray): explanatory test variables
        X_transformer (ColumnTransformer): transformer for X variables
        y (np.ndarray): target test variable
    """
    model_info = {
        "model": model,
        "predictions": predictions,
        "mean_cv_score": mean_cv_score,
        "explanatory_variables": X,
        "explanatory_variables_transformer": X_transformer,
        "target_variable": y,
    }
    gcp.write_blob_to_bucket(file_name, model_info)
