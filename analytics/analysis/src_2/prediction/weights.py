def calculate_weights(mean_scores: list[float]) -> list[float]:
    """Calculate the weights of the models based on the mean scores.

    Args:
        mean_scores (list[float]): list of mean scores

    Returns:
        list[float]: list of weights
    """
    sums = sum(1 / score for score in mean_scores)
    return [round(((1 / score) / sums), 2) for score in mean_scores]
