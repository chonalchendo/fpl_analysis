import numpy as np


def calculate_weights(mean_scores: list[float]) -> list[float]:
    """Calculate the weights of the models based on the mean scores.

    Args:
        mean_scores (list[float]): list of mean scores

    Returns:
        list[float]: list of weights
    """
    sums = sum(1 / score for score in mean_scores)
    return [round(((1 / score) / sums), 2) for score in mean_scores]


# def calculate_weights(mean_scores: np.ndarray) -> np.ndarray:
#     """Calculate the weights of the models based on the mean scores.
#
#     Args:
#         mean_scores (np.ndarray): 2D array of mean scores, where each row
#         represents a set of mean scores for different models.
#
#     Returns:
#         np.ndarray: 2D array of weights, where each row represents the
#         weights corresponding to the mean scores of that row.
#     """
#     weights = []
#     for scores in mean_scores:
#         sums = sum(1 / score for score in scores)
#         weights.append([round(((1 / score) / sums), 2) for score in scores])
#     return np.array(weights)
