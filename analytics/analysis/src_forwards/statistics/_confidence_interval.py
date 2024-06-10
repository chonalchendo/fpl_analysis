import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample


def mae_confidence_interval(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap_samples: int = 1000,
) -> tuple[float, float]:
    """Calculate the MAE confidence interval for the predicted value.

    Args:
        y_true (np.ndarray): true y values
        y_pred (np.ndarray): predicted y values
        confidence (int, optional): confidence interval. Defaults to 0.95.
        n_bootstrap_samples (int, optional): number of bootstrapped samples.
        Defaults to 1000.

    Returns:
        tuple[float, float]: lower and upper bounds of the confidence interval
    """
    mae_values = []

    for _ in range(n_bootstrap_samples):
        # Bootstrap resampling
        y_true_bootstrap, y_pred_bootstrap = resample(y_true, y_pred)

        # Calculate mean absolute error for the resampled data
        mae = mean_absolute_error(y_true_bootstrap, y_pred_bootstrap)
        mae_values.append(mae)

    # Calculate confidence interval for the mean absolute error
    lower, upper = np.percentile(
        mae_values,
        [(1 - confidence) * 100 / 2, confidence * 100 - (1 - confidence) * 100 / 2],
    )

    return round(lower, 2), round(upper, 2)
