from typing import Any
import numpy as np
from analysis.gcp.storage import gcp


def save(
    model: Any,
    predictions: np.ndarray,
    model_rmse: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    file_name: str,
) -> None:
    """Save the model to the gcp bucket.

    Args:
        model (Any): final model
        predictions (np.ndarray): final model predictions on test set
        model_rmse (np.ndarray): final model rmse score
        X (np.ndarray): explanatory test variables
        y (np.ndarray): target test variable
    """
    model_info = {
        "model": model,
        "predictions": predictions,
        "model_rmse": model_rmse,
        "explanatory_variables": X,
        "target_variable": y,
    }
    gcp.write_blob_to_bucket(file_name, model_info)
