def model_metadata(
    model,
    model_params,
    preprocess_steps,
    target_steps,
    metric,
    scores,
    mean_scores,
    std_scores,
    X_data,
    y_data,
) -> dict:
    return {
        "model": model,
        "model_params": model_params,
        "preprocess": preprocess_steps,
        "target": target_steps,
        "metric": metric,
        "scores": scores,
        "mean_score": mean_scores,
        "std_score": std_scores,
        "X_data": X_data,
        "y_data": y_data,
    }
