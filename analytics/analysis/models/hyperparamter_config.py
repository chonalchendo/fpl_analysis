from scipy.stats import randint

# -------------------------------- Grid Search ------------------------------- #

# define the parameter grid to search for the best combination of hyperparameters
grid_params = [
    # try 12 (3×4) combinations of hyperparameters
    {
        "regressor__n_estimators": [3, 10, 30, 50, 100, 200],
        "regressor__max_features": [2, 4, 6, 8, 10, 12, 14, 16],
    },
    # then try 6 (2×3) combinations with bootstrap set as False
    {
        "regressor__bootstrap": [False],
        "regressor__n_estimators": [3, 10],
        "regressor__max_features": [2, 3, 4],
    },
]


# -------------------------------- Randomized Search ------------------------------- #

random_params = {
    "regressor__n_estimators": randint(low=1, high=200),
    "regressor__max_features": randint(low=1, high=16),
}
